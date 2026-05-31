package hotstuff

import (
	"encoding/json"
	"testing"
	"time"

	"go.dedis.ch/kyber/v3"
	libp2pnetwork "evolvbft/evolvbft/network/libp2p"

	"evolvbft/evolvbft/consensus/beacon"
	"evolvbft/evolvbft/consensus/mempool"
	"evolvbft/evolvbft/consensus/pacemaker"
	"evolvbft/evolvbft/consensus/viewchange"
	"evolvbft/evolvbft/crypto"
	octcrypto "evolvbft/evolvbft/crypto"
	"evolvbft/evolvbft/hydra"
	"evolvbft/evolvbft/storage"
	"evolvbft/evolvbft/types"
)

func buildEngineVoteHarness(t *testing.T) (*Engine, map[uint64]*types.Keypair) {
	t.Helper()

	keypairs := make(map[uint64]*types.Keypair, 4)
	validators := make(map[uint64]*types.Validator, 4)
	validatorIDs := []uint64{0, 1, 2, 3}

	for _, id := range validatorIDs {
		kp, err := octcrypto.GenerateKeyPair()
		if err != nil {
			t.Fatalf("generate keypair for validator %d failed: %v", id, err)
		}
		keypairs[id] = &types.Keypair{
			PublicKey:  kp.PublicKey,
			PrivateKey: kp.PrivateKey,
		}
		validators[id] = &types.Validator{
			ID:        id,
			PublicKey: kp.PublicKey,
			Power:     1,
			IsActive:  true,
		}
	}

	valSet := types.NewValidatorSet(1, validators)
	store := storage.NewStorageManager(0)
	engine := &Engine{
		nodeID:         0,
		keypair:        keypairs[0],
		currentEpoch:   1,
		valSet:         valSet,
		storage:        store,
		blockTree:      NewBlockTree(store, NewExecutor(valSet)),
		pacemaker:      pacemaker.NewPacemaker(validatorIDs, 20),
		beacon:         beacon.NewRandomBeacon([]byte("genesis-seed")),
		voteCollectors: make(map[string]*voteCollector),
		seenProposals:  make(map[uint64]map[uint64][]byte),
		rankState:      NewRankState(0, 1),
		rejected:       make(map[string]uint64),
	}

	return engine, keypairs
}

func newSignedVote(t *testing.T, signer *types.Keypair, blockHash []byte, view uint64) *types.Vote {
	t.Helper()

	var blockID types.Hash
	copy(blockID[:], blockHash)

	vote, err := types.NewVoteWithIdentity(blockID, view, 1, 1, 0, signer.PublicKey, signer.PrivateKey, nil)
	if err != nil {
		t.Fatalf("create vote failed: %v", err)
	}
	return vote
}

func stampProposalIdentity(block *types.Block, configID uint64, lane uint64) *types.Block {
	block.ConfigID = configID
	block.LaneID = lane
	block.Hash = block.ComputeHash()
	return block
}

func TestEngineVoteCollectorIgnoresDuplicateVotes(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	blockHash := make([]byte, 32)
	copy(blockHash, []byte("dup-block"))

	vote1 := newSignedVote(t, keypairs[1], blockHash, 1)
	vote2 := newSignedVote(t, keypairs[2], blockHash, 1)

	engine.handleVote(vote1)
	engine.handleVote(vote1)
	engine.handleVote(vote2)

	collector := engine.voteCollectors[engine.collectorKey(1, 1, 1, 0, blockHash)]
	if collector == nil {
		t.Fatalf("expected collector to be created")
	}
	if len(collector.signers) != 2 {
		t.Fatalf("expected 2 unique signers, got %d", len(collector.signers))
	}
	if collector.done {
		t.Fatalf("collector should not be done before quorum")
	}
	if engine.blockTree.GetHighQC().View != 0 {
		t.Fatalf("highQC should stay at genesis before quorum, got %d", engine.blockTree.GetHighQC().View)
	}
}

func TestEngineVoteCollectorHandlesOutOfOrderViews(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	blockHash := make([]byte, 32)
	copy(blockHash, []byte("order-block"))
	engine.pacemaker.AdvanceView(2)

	oldVote := newSignedVote(t, keypairs[1], blockHash, 1)
	nearPastVote := newSignedVote(t, keypairs[2], blockHash, 2)
	futureVote := newSignedVote(t, keypairs[3], blockHash, 5)

	engine.handleVote(oldVote)
	if len(engine.voteCollectors) != 0 {
		t.Fatalf("stale vote should be ignored")
	}

	engine.handleVote(nearPastVote)
	if len(engine.voteCollectors) != 1 {
		t.Fatalf("expected one collector from near-past vote, got %d", len(engine.voteCollectors))
	}

	engine.handleVote(futureVote)
	if len(engine.voteCollectors) != 1 {
		t.Fatalf("future vote should be ignored")
	}
	if engine.voteCollectors[engine.collectorKey(5, 1, 1, 0, blockHash)] != nil {
		t.Fatalf("collector should not be created for future vote")
	}
}

func TestEngineVoteAggregationAdvancesQCAndView(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	blockHash := make([]byte, 32)
	copy(blockHash, []byte("qc-block"))

	engine.handleVote(newSignedVote(t, keypairs[1], blockHash, 1))
	engine.handleVote(newSignedVote(t, keypairs[2], blockHash, 1))
	engine.handleVote(newSignedVote(t, keypairs[3], blockHash, 1))

	collector := engine.voteCollectors[engine.collectorKey(1, 1, 1, 0, blockHash)]
	if collector == nil || !collector.done {
		t.Fatalf("collector should be done after reaching quorum")
	}
	if engine.blockTree.GetHighQC().View != 1 {
		t.Fatalf("expected highQC view 1, got %d", engine.blockTree.GetHighQC().View)
	}
	if engine.pacemaker.GetCurrentView() != 2 {
		t.Fatalf("expected current view 2, got %d", engine.pacemaker.GetCurrentView())
	}
}

func TestEngineHandleVoteIgnoresVotesAfterCollectorDone(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	blockHash := make([]byte, 32)
	copy(blockHash, []byte("done-collector-block"))

	engine.handleVote(newSignedVote(t, keypairs[1], blockHash, 1))
	engine.handleVote(newSignedVote(t, keypairs[2], blockHash, 1))
	engine.handleVote(newSignedVote(t, keypairs[3], blockHash, 1))

	collectorKey := engine.collectorKey(1, 1, 1, 0, blockHash)
	collector := engine.voteCollectors[collectorKey]
	if collector == nil || !collector.done {
		t.Fatalf("collector should be done after reaching quorum")
	}
	beforeView := engine.pacemaker.GetCurrentView()
	beforeHighQC := engine.blockTree.GetHighQC()
	beforeSigners := len(collector.signers)

	engine.handleVote(newSignedVote(t, keypairs[0], blockHash, 1))

	collector = engine.voteCollectors[collectorKey]
	if collector == nil || !collector.done {
		t.Fatalf("collector should remain done after extra vote")
	}
	if got := len(collector.signers); got != beforeSigners {
		t.Fatalf("expected signer count to stay %d after extra vote, got %d", beforeSigners, got)
	}
	if view := engine.pacemaker.GetCurrentView(); view != beforeView {
		t.Fatalf("expected extra vote after done collector not to change pacemaker view, got %d want %d", view, beforeView)
	}
	if highQC := engine.blockTree.GetHighQC(); highQC == nil || beforeHighQC == nil || highQC.View != beforeHighQC.View {
		t.Fatalf("expected extra vote after done collector not to change highQC, got %+v want %+v", highQC, beforeHighQC)
	}
}

func signVoteMessageForTest(t *testing.T, engine *Engine, kp *types.Keypair, senderID uint64, vote *types.Vote) *types.Message {
	t.Helper()
	msg := &types.Message{
		Type:          types.MsgVote,
		SenderID:      senderID,
		View:          vote.View,
		Epoch:         vote.Epoch,
		ConfigID:      vote.ConfigID,
		Lane:          vote.Lane,
		LeaderSetHash: engine.leaderSetHashSnapshot(),
		BarrierView:   vote.View,
		Instance:      vote.Lane,
		Vote:          vote,
	}
	if err := msg.Sign(kp.PrivateKey); err != nil {
		t.Fatalf("sign vote message: %v", err)
	}
	return msg
}

func newSignedQCForTest(t *testing.T, keypairs map[uint64]*types.Keypair, blockHash []byte, view uint64, epoch uint64, configID uint64, lane uint64) *types.QuorumCertificate {
	t.Helper()
	qc := types.NewQuorumCertificateWithIdentity(blockHash, view, epoch, configID, lane, types.PhasePrepare)
	msg := types.QCSigningBytes(blockHash, view, epoch, configID, lane)
	for _, validatorID := range []uint64{1, 2, 3} {
		qc.AddSignature(validatorID, crypto.Sign(msg, keypairs[validatorID].PrivateKey))
	}
	return qc
}

func newSignedTCForTest(t *testing.T, keypairs map[uint64]*types.Keypair, view uint64, epoch uint64, configID uint64, lane uint64, highestQC *types.QuorumCertificate) *types.TimeoutCertificate {
	t.Helper()
	tc := &types.TimeoutCertificate{
		View:       view,
		Epoch:      epoch,
		ConfigID:   configID,
		Lane:       lane,
		HighestQC:  highestQC,
		Signatures: make(map[uint64][]byte, 3),
		NumVoters:  3,
	}
	msg := types.TimeoutSigningBytes(view, epoch, configID, lane)
	for _, validatorID := range []uint64{1, 2, 3} {
		tc.Signatures[validatorID] = crypto.Sign(msg, keypairs[validatorID].PrivateKey)
	}
	return tc
}

func signNewViewMessageForTest(t *testing.T, engine *Engine, kp *types.Keypair, msg *types.Message) *types.Message {
	t.Helper()
	if msg.LeaderSetHash == nil {
		msg.LeaderSetHash = engine.leaderSetHashSnapshot()
	}
	if err := msg.Sign(kp.PrivateKey); err != nil {
		t.Fatalf("sign newview message: %v", err)
	}
	return msg
}

func signProposalMessageForTest(t *testing.T, engine *Engine, kp *types.Keypair, senderID uint64, block *types.Block) *types.Message {
	t.Helper()
	msg := &types.Message{
		Type:          types.MsgProposal,
		SenderID:      senderID,
		View:          block.View,
		Epoch:         block.Epoch,
		ConfigID:      block.ConfigID,
		Lane:          block.LaneID,
		LeaderSetHash: engine.leaderSetHashSnapshot(),
		BarrierView:   block.View,
		Instance:      block.LaneID,
		Block:         block,
	}
	if err := msg.Sign(kp.PrivateKey); err != nil {
		t.Fatalf("sign proposal message: %v", err)
	}
	return msg
}

func TestEngineProposeBlockDoesNotDeadlockOnSnapshot(t *testing.T) {
	engine, _ := buildEngineVoteHarness(t)
	engine.consensusTopic = "test-proposal-topic"
	engine.network = &libp2pnetwork.P2PNetwork{}

	done := make(chan struct{})
	go func() {
		defer func() {
			_ = recover()
			close(done)
		}()
		engine.proposeBlock(nil)
	}()

	select {
	case <-done:
	case <-time.After(100 * time.Millisecond):
		t.Fatalf("expected proposeBlock not to deadlock while snapshotting proposal state")
	}
}

func TestEngineHandleMessageRejectsNilProposalPayload(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	msg := &types.Message{
		Type:          types.MsgProposal,
		SenderID:      1,
		View:          2,
		Epoch:         1,
		ConfigID:      1,
		Lane:          0,
		LeaderSetHash: engine.leaderSetHashSnapshot(),
		BarrierView:   2,
		Instance:      0,
	}
	if err := msg.Sign(keypairs[1].PrivateKey); err != nil {
		t.Fatalf("sign proposal message: %v", err)
	}
	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["nil_proposal_payload"]; got != 1 {
		t.Fatalf("expected nil_proposal_payload rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsProposalSenderMismatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	qc := types.NewQuorumCertificate([]byte("proposal-sender-qc"), 1, 1, types.PhasePrepare)
	block := stampProposalIdentity(types.NewBlock(1, qc.BlockHash, []byte("proposal-sender"), 2, 1, 1, int64(engine.rankState.ExpectedRank(1)), qc, nil), 1, 0)
	msg := signProposalMessageForTest(t, engine, keypairs[2], 2, block)
	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["proposal_sender_mismatch"]; got != 1 {
		t.Fatalf("expected proposal_sender_mismatch rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsProposalViewMismatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	qc := types.NewQuorumCertificate([]byte("proposal-view-qc"), 1, 1, types.PhasePrepare)
	block := stampProposalIdentity(types.NewBlock(1, qc.BlockHash, []byte("proposal-view"), 2, 1, 1, int64(engine.rankState.ExpectedRank(1)), qc, nil), 1, 0)
	msg := signProposalMessageForTest(t, engine, keypairs[1], 1, block)
	msg.View = 3
	if err := msg.Sign(keypairs[1].PrivateKey); err != nil {
		t.Fatalf("resign proposal message: %v", err)
	}
	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["proposal_view_wrapper_mismatch"]; got != 1 {
		t.Fatalf("expected proposal_view_wrapper_mismatch rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsProposalEpochMismatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	qc := types.NewQuorumCertificate([]byte("proposal-epoch-qc"), 1, 1, types.PhasePrepare)
	block := stampProposalIdentity(types.NewBlock(1, qc.BlockHash, []byte("proposal-epoch"), 2, 1, 1, int64(engine.rankState.ExpectedRank(1)), qc, nil), 1, 0)
	msg := signProposalMessageForTest(t, engine, keypairs[1], 1, block)
	msg.Epoch = 2
	if err := msg.Sign(keypairs[1].PrivateKey); err != nil {
		t.Fatalf("resign proposal message: %v", err)
	}
	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["proposal_epoch_wrapper_mismatch"]; got != 1 {
		t.Fatalf("expected proposal_epoch_wrapper_mismatch rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsProposalConfigMismatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	qc := types.NewQuorumCertificate([]byte("proposal-config-qc"), 1, 1, types.PhasePrepare)
	block := stampProposalIdentity(types.NewBlock(1, qc.BlockHash, []byte("proposal-config"), 2, 1, 1, int64(engine.rankState.ExpectedRank(1)), qc, nil), 1, 0)
	msg := signProposalMessageForTest(t, engine, keypairs[1], 1, block)
	msg.Block.ConfigID = 2
	if err := msg.Sign(keypairs[1].PrivateKey); err != nil {
		t.Fatalf("resign proposal message: %v", err)
	}
	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["proposal_config_wrapper_mismatch"]; got != 1 {
		t.Fatalf("expected proposal_config_wrapper_mismatch rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsProposalLaneMismatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	qc := types.NewQuorumCertificate([]byte("proposal-lane-qc"), 1, 1, types.PhasePrepare)
	block := stampProposalIdentity(types.NewBlock(1, qc.BlockHash, []byte("proposal-lane"), 2, 1, 1, int64(engine.rankState.ExpectedRank(1)), qc, nil), 1, 0)
	msg := signProposalMessageForTest(t, engine, keypairs[1], 1, block)
	msg.Block.LaneID = 1
	if err := msg.Sign(keypairs[1].PrivateKey); err != nil {
		t.Fatalf("resign proposal message: %v", err)
	}
	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["proposal_lane_wrapper_mismatch"]; got != 1 {
		t.Fatalf("expected proposal_lane_wrapper_mismatch rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageAcceptsWrapperConsistentProposalUntilInnerValidation(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	badQC := types.NewQuorumCertificateWithIdentity([]byte("proposal-accept-qc"), 1, 1, 1, 0, types.PhasePrepare)
	badQC.AddSignature(1, []byte("bad-sig-1"))
	badQC.AddSignature(2, []byte("bad-sig-2"))
	badQC.AddSignature(3, []byte("bad-sig-3"))
	block := stampProposalIdentity(types.NewBlock(1, badQC.BlockHash, []byte("proposal-accept"), 2, 1, 1, int64(engine.rankState.ExpectedRank(1)), badQC, nil), 1, 0)
	msg := signProposalMessageForTest(t, engine, keypairs[1], 1, block)

	engine.handleMessage(msg)
	stats := engine.GetRejectedStats()
	if stats["nil_proposal_payload"] != 0 || stats["proposal_sender_mismatch"] != 0 || stats["proposal_view_wrapper_mismatch"] != 0 || stats["proposal_epoch_wrapper_mismatch"] != 0 || stats["proposal_config_wrapper_mismatch"] != 0 || stats["proposal_lane_wrapper_mismatch"] != 0 {
		t.Fatalf("expected wrapper-consistent proposal to bypass wrapper rejections, got %#v", stats)
	}
	if stats["proposal_invalid_justify_qc"] != 1 {
		t.Fatalf("expected wrapper-consistent proposal to reach inner justify validation, got %#v", stats)
	}
}

func TestEngineHandleMessageRejectsNilVotePayload(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	msg := &types.Message{
		Type:          types.MsgVote,
		SenderID:      1,
		View:          1,
		Epoch:         1,
		ConfigID:      1,
		Lane:          0,
		LeaderSetHash: engine.leaderSetHashSnapshot(),
		BarrierView:   1,
		Instance:      0,
	}
	if err := msg.Sign(keypairs[1].PrivateKey); err != nil {
		t.Fatalf("sign vote message: %v", err)
	}
	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["nil_vote_payload"]; got != 1 {
		t.Fatalf("expected nil_vote_payload rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsVoteSenderMismatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	blockHash := make([]byte, 32)
	copy(blockHash, []byte("vote-sender-mismatch"))
	vote := newSignedVote(t, keypairs[1], blockHash, 1)
	msg := signVoteMessageForTest(t, engine, keypairs[2], 2, vote)
	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["vote_sender_mismatch"]; got != 1 {
		t.Fatalf("expected vote_sender_mismatch rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsVoteViewMismatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	blockHash := make([]byte, 32)
	copy(blockHash, []byte("vote-view-mismatch"))
	vote := newSignedVote(t, keypairs[1], blockHash, 1)
	msg := signVoteMessageForTest(t, engine, keypairs[1], 1, vote)
	msg.View = 2
	if err := msg.Sign(keypairs[1].PrivateKey); err != nil {
		t.Fatalf("resign vote message: %v", err)
	}
	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["vote_view_mismatch"]; got != 1 {
		t.Fatalf("expected vote_view_mismatch rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsVoteEpochMismatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	blockHash := make([]byte, 32)
	copy(blockHash, []byte("vote-epoch-mismatch"))
	vote := newSignedVote(t, keypairs[1], blockHash, 1)
	msg := signVoteMessageForTest(t, engine, keypairs[1], 1, vote)
	msg.Epoch = 2
	if err := msg.Sign(keypairs[1].PrivateKey); err != nil {
		t.Fatalf("resign vote message: %v", err)
	}
	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["vote_epoch_mismatch"]; got != 1 {
		t.Fatalf("expected vote_epoch_mismatch rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsMessageConfigMismatchBeforeVoteDispatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	blockHash := make([]byte, 32)
	copy(blockHash, []byte("vote-config-mismatch"))
	vote := newSignedVote(t, keypairs[1], blockHash, 1)
	msg := signVoteMessageForTest(t, engine, keypairs[1], 1, vote)
	msg.ConfigID = 2
	if err := msg.Sign(keypairs[1].PrivateKey); err != nil {
		t.Fatalf("resign vote message: %v", err)
	}
	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["message_config_mismatch"]; got != 1 {
		t.Fatalf("expected message_config_mismatch rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsInnerVoteConfigMismatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	blockHash := make([]byte, 32)
	copy(blockHash, []byte("inner-vote-config-mismatch"))
	vote := newSignedVote(t, keypairs[1], blockHash, 1)
	vote.ConfigID = 2
	vote.Signature = nil
	resigned, err := types.NewVoteWithIdentity(vote.BlockID, vote.View, vote.Epoch, vote.ConfigID, vote.Lane, keypairs[1].PublicKey, keypairs[1].PrivateKey, vote.VRFProof)
	if err != nil {
		t.Fatalf("resign mismatched vote: %v", err)
	}
	msg := signVoteMessageForTest(t, engine, keypairs[1], 1, resigned)
	msg.ConfigID = 1
	if err := msg.Sign(keypairs[1].PrivateKey); err != nil {
		t.Fatalf("resign vote message: %v", err)
	}
	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["vote_config_mismatch"]; got != 1 {
		t.Fatalf("expected vote_config_mismatch rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsVoteLaneMismatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	blockHash := make([]byte, 32)
	copy(blockHash, []byte("vote-lane-mismatch"))
	vote := newSignedVote(t, keypairs[1], blockHash, 1)
	msg := signVoteMessageForTest(t, engine, keypairs[1], 1, vote)
	msg.Lane = 0
	msg.Vote.Lane = 1
	if err := msg.Sign(keypairs[1].PrivateKey); err != nil {
		t.Fatalf("resign vote message: %v", err)
	}
	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["vote_lane_mismatch"]; got != 1 {
		t.Fatalf("expected vote_lane_mismatch rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageAcceptsBarrierConsistentVoteWrapper(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	blockHash := make([]byte, 32)
	copy(blockHash, []byte("vote-barrier-consistent"))
	vote := newSignedVote(t, keypairs[1], blockHash, 1)
	msg := signVoteMessageForTest(t, engine, keypairs[1], 1, vote)

	engine.handleMessage(msg)

	stats := engine.GetRejectedStats()
	if stats["invalid_message_signature"] != 0 || stats["vote_sender_mismatch"] != 0 || stats["vote_view_mismatch"] != 0 || stats["vote_epoch_mismatch"] != 0 || stats["vote_config_mismatch"] != 0 || stats["vote_lane_mismatch"] != 0 || stats["vote_barrier_view_mismatch"] != 0 {
		t.Fatalf("expected barrier-consistent vote to bypass wrapper rejections, got %#v", stats)
	}
	collector := engine.voteCollectors[engine.collectorKey(1, 1, 1, 0, blockHash)]
	if collector == nil {
		t.Fatalf("expected barrier-consistent vote to reach inner vote handling")
	}
}

func TestEngineHandleMessageAcceptsBarrierConsistentTimeoutVoteWrapper(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	engine.vcm = viewchange.NewViewChangeManager(engine.nodeID, 20*time.Millisecond, func() uint64 {
		return engine.valSet.QuorumSize
	})
	vote := &types.TimeoutVote{View: 2, Epoch: 1, ConfigID: 1, Lane: 0, VoterID: 1}
	vote.Signature = crypto.Sign(types.TimeoutSigningBytes(vote.View, vote.Epoch, vote.ConfigID, vote.Lane), keypairs[1].PrivateKey)
	msg := &types.Message{Type: types.MsgTimeout, SenderID: 1, View: 2, Epoch: 1, ConfigID: 1, Lane: 0, LeaderSetHash: engine.leaderSetHashSnapshot(), BarrierView: 2, Instance: 0, TimeoutVote: vote}
	if err := msg.Sign(keypairs[1].PrivateKey); err != nil {
		t.Fatalf("sign timeout message: %v", err)
	}

	engine.handleMessage(msg)

	stats := engine.GetRejectedStats()
	if stats["invalid_message_signature"] != 0 || stats["missing_leader_set_hash"] != 0 || stats["leader_set_mismatch"] != 0 || stats["timeout_vote_sender_mismatch"] != 0 || stats["timeout_vote_view_wrapper_mismatch"] != 0 || stats["timeout_vote_epoch_wrapper_mismatch"] != 0 || stats["timeout_vote_config_wrapper_mismatch"] != 0 || stats["timeout_vote_lane_wrapper_mismatch"] != 0 || stats["timeout_vote_barrier_view_mismatch"] != 0 {
		t.Fatalf("expected barrier-consistent timeout vote to bypass wrapper rejections, got %#v", stats)
	}
	if stats["future_timeout_vote"] != 1 {
		t.Fatalf("expected barrier-consistent timeout vote to reach inner view validation, got %#v", stats)
	}
}

func TestEngineHandleMessageRejectsMissingLeaderSetHashOnProposal(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	block := stampProposalIdentity(types.NewBlock(1, make([]byte, 32), nil, 2, 1, 1, 1, nil, nil), 1, 0)
	msg := signProposalMessageForTest(t, engine, keypairs[1], 1, block)
	msg.LeaderSetHash = nil
	if err := msg.Sign(keypairs[1].PrivateKey); err != nil {
		t.Fatalf("re-sign proposal message: %v", err)
	}

	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["missing_leader_set_hash"]; got != 1 {
		t.Fatalf("expected missing_leader_set_hash rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsLeaderSetHashMismatchOnVote(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	blockHash := make([]byte, 32)
	copy(blockHash, []byte("vote-leader-set-mismatch"))
	vote := newSignedVote(t, keypairs[1], blockHash, 1)
	msg := signVoteMessageForTest(t, engine, keypairs[1], 1, vote)
	msg.LeaderSetHash = []byte("wrong-leader-set")
	if err := msg.Sign(keypairs[1].PrivateKey); err != nil {
		t.Fatalf("re-sign vote message: %v", err)
	}

	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["leader_set_mismatch"]; got != 1 {
		t.Fatalf("expected leader_set_mismatch rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsMissingLeaderSetHashOnTimeout(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	engine.vcm = viewchange.NewViewChangeManager(engine.nodeID, 20*time.Millisecond, func() uint64 {
		return engine.valSet.QuorumSize
	})
	vote := &types.TimeoutVote{View: 2, Epoch: 1, ConfigID: 1, Lane: 0, VoterID: 1}
	vote.Signature = crypto.Sign(types.TimeoutSigningBytes(vote.View, vote.Epoch, vote.ConfigID, vote.Lane), keypairs[1].PrivateKey)
	msg := &types.Message{Type: types.MsgTimeout, SenderID: 1, View: 2, Epoch: 1, ConfigID: 1, Lane: 0, BarrierView: 2, Instance: 0, TimeoutVote: vote}
	if err := msg.Sign(keypairs[1].PrivateKey); err != nil {
		t.Fatalf("sign timeout message: %v", err)
	}

	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["missing_leader_set_hash"]; got != 1 {
		t.Fatalf("expected missing_leader_set_hash rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsMissingLeaderSetHashOnNewView(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	qc := newSignedQCForTest(t, keypairs, []byte("newview-missing-leader-set-qc"), 2, 1, 1, 0)
	msg := signNewViewMessageForTest(t, engine, keypairs[1], &types.Message{
		Type:        types.MsgNewView,
		SenderID:    1,
		View:        3,
		Epoch:       1,
		ConfigID:    1,
		Lane:        0,
		BarrierView: 3,
		Instance:    0,
		QC:          qc,
	})
	msg.LeaderSetHash = nil
	if err := msg.Sign(keypairs[1].PrivateKey); err != nil {
		t.Fatalf("re-sign newview message: %v", err)
	}

	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["missing_leader_set_hash"]; got != 1 {
		t.Fatalf("expected missing_leader_set_hash rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsLeaderSetHashMismatchOnNewView(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	qc := newSignedQCForTest(t, keypairs, []byte("newview-leader-set-mismatch-qc"), 2, 1, 1, 0)
	msg := signNewViewMessageForTest(t, engine, keypairs[1], &types.Message{
		Type:        types.MsgNewView,
		SenderID:    1,
		View:        3,
		Epoch:       1,
		ConfigID:    1,
		Lane:        0,
		BarrierView: 3,
		Instance:    0,
		QC:          qc,
	})
	msg.LeaderSetHash = []byte("wrong-leader-set")
	if err := msg.Sign(keypairs[1].PrivateKey); err != nil {
		t.Fatalf("re-sign newview message: %v", err)
	}

	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["leader_set_mismatch"]; got != 1 {
		t.Fatalf("expected leader_set_mismatch rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageAcceptsLeaderSetHashConsistentProposalUntilInnerValidation(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	badQC := types.NewQuorumCertificateWithIdentity([]byte("proposal-leader-set-accept-qc"), 1, 1, 1, 0, types.PhasePrepare)
	badQC.AddSignature(1, []byte("bad-sig-1"))
	badQC.AddSignature(2, []byte("bad-sig-2"))
	badQC.AddSignature(3, []byte("bad-sig-3"))
	block := stampProposalIdentity(types.NewBlock(1, badQC.BlockHash, []byte("proposal-leader-set-accept"), 2, 1, 1, int64(engine.rankState.ExpectedRank(1)), badQC, nil), 1, 0)
	msg := signProposalMessageForTest(t, engine, keypairs[1], 1, block)

	engine.handleMessage(msg)
	stats := engine.GetRejectedStats()
	if stats["missing_leader_set_hash"] != 0 || stats["leader_set_mismatch"] != 0 || stats["proposal_barrier_view_mismatch"] != 0 {
		t.Fatalf("expected leader-set-consistent proposal to bypass leader-set wrapper rejection, got %#v", stats)
	}
	if stats["proposal_invalid_justify_qc"] != 1 {
		t.Fatalf("expected leader-set-consistent proposal to reach inner justify validation, got %#v", stats)
	}
}

func TestPacemakerTimeoutHandlingStaysMonotonic(t *testing.T) {
	pm := pacemaker.NewPacemaker([]uint64{0, 1, 2, 3}, 20)
	pm.Start()

	var firstTimeout uint64
	select {
	case firstTimeout = <-pm.TimeoutChan():
	case <-time.After(500 * time.Millisecond):
		t.Fatalf("did not receive first timeout")
	}

	if pm.GetCurrentView() <= firstTimeout {
		pm.AdvanceView(firstTimeout)
	}
	if pm.GetCurrentView() != 2 {
		t.Fatalf("expected view 2 after first timeout handling, got %d", pm.GetCurrentView())
	}

	if pm.GetCurrentView() <= firstTimeout {
		pm.AdvanceView(firstTimeout)
	}
	if pm.GetCurrentView() != 2 {
		t.Fatalf("stale timeout must not regress or advance view, got %d", pm.GetCurrentView())
	}

	var secondTimeout uint64
	select {
	case secondTimeout = <-pm.TimeoutChan():
	case <-time.After(500 * time.Millisecond):
		t.Fatalf("did not receive second timeout")
	}

	if pm.GetCurrentView() <= secondTimeout {
		pm.AdvanceView(secondTimeout)
	}
	if pm.GetCurrentView() != 3 {
		t.Fatalf("expected view 3 after second timeout handling, got %d", pm.GetCurrentView())
	}
}

func TestNextBlockDataFromMempoolReturnsNilWithoutCerts(t *testing.T) {
	engine, _ := buildEngineVoteHarness(t)
	data := engine.nextBlockDataFromMempool(nil)
	if data != nil {
		t.Fatalf("expected nil block data without certificates, got %q", string(data))
	}
}

func TestNextBlockDataFromMempoolReturnsNilForMissingVertex(t *testing.T) {
	engine, _ := buildEngineVoteHarness(t)
	engine.mempool = mempool.NewMempoolWithOptions(engine.nodeID, engine.keypair, engine.valSet, nil, mempool.Options{})
	certs := []*types.VertexCertificate{
		{VertexHash: types.Hash{1, 2, 3}},
	}
	data := engine.nextBlockDataFromMempool(certs)
	if data != nil {
		t.Fatalf("expected nil block data for missing vertex, got %q", string(data))
	}
}

func TestNextBlockDataFromMempoolUsesRealTransactionPayload(t *testing.T) {
	engine, _ := buildEngineVoteHarness(t)
	engine.mempool = mempool.NewMempoolWithOptions(
		engine.nodeID,
		engine.keypair,
		engine.valSet,
		nil,
		mempool.Options{
			ProposalInterval: 5 * time.Millisecond,
		},
	)
	engine.mempool.Start()

	tx := &types.Transaction{
		Type:    types.TxTypeNormal,
		Payload: []byte("real-payload"),
	}
	if err := engine.mempool.SubmitTransaction(tx); err != nil {
		t.Fatalf("submit transaction failed: %v", err)
	}

	var certs []*types.VertexCertificate
	select {
	case certs = <-engine.mempool.GetProposalChan():
	case <-time.After(500 * time.Millisecond):
		t.Fatalf("did not receive proposal certificates from mempool")
	}
	if len(certs) == 0 {
		t.Fatalf("expected proposal certificates to be non-empty")
	}

	data := engine.nextBlockDataFromMempool(certs)
	if len(data) == 0 {
		t.Fatalf("expected non-empty block data")
	}

	want, err := json.Marshal(tx)
	if err != nil {
		t.Fatalf("marshal expected tx failed: %v", err)
	}
	if string(data) != string(want) {
		t.Fatalf("unexpected block data, got %s want %s", string(data), string(want))
	}
}

func TestEngineProposalBoundaryRejectsInvalidViewWindow(t *testing.T) {
	engine, _ := buildEngineVoteHarness(t)
	engine.pacemaker.AdvanceView(4)
	qc := types.NewQuorumCertificate([]byte("p"), 4, 1, types.PhasePrepare)

	stale := stampProposalIdentity(types.NewBlock(1, qc.BlockHash, []byte("stale"), 2, 1, 1, int64(engine.rankState.ExpectedRank(1)), qc, nil), 1, 0)
	engine.handleProposal(stale)

	futureQC := types.NewQuorumCertificate([]byte("q"), 9, 1, types.PhasePrepare)
	future := stampProposalIdentity(types.NewBlock(2, futureQC.BlockHash, []byte("future"), 10, 1, 1, int64(engine.rankState.ExpectedRank(2)), futureQC, nil), 1, 0)
	engine.handleProposal(future)

	stats := engine.GetRejectedStats()
	if stats["proposal_stale_view"] != 1 {
		t.Fatalf("expected stale-view rejection count 1, got %d", stats["proposal_stale_view"])
	}
	if stats["proposal_future_view"] != 1 {
		t.Fatalf("expected future-view rejection count 1, got %d", stats["proposal_future_view"])
	}
}

func TestEngineProposalBoundaryRejectsWrongLeaderAndQCInconsistency(t *testing.T) {
	engine, _ := buildEngineVoteHarness(t)
	qc := types.NewQuorumCertificate([]byte("p"), 1, 1, types.PhasePrepare)

	wrongLeader := stampProposalIdentity(types.NewBlock(1, qc.BlockHash, []byte("wrong-leader"), 2, 1, 3, int64(engine.rankState.ExpectedRank(1)), qc, nil), 1, 0)
	engine.handleProposal(wrongLeader)

	wrongEpochQC := types.NewQuorumCertificate([]byte("p2"), 1, 2, types.PhasePrepare)
	epochMismatch := stampProposalIdentity(types.NewBlock(2, wrongEpochQC.BlockHash, []byte("epoch"), 2, 1, 1, int64(engine.rankState.ExpectedRank(2)), wrongEpochQC, nil), 1, 0)
	engine.handleProposal(epochMismatch)

	parentMismatch := stampProposalIdentity(types.NewBlock(3, []byte("other-parent"), []byte("parent"), 2, 1, 1, int64(engine.rankState.ExpectedRank(3)), qc, nil), 1, 0)
	engine.handleProposal(parentMismatch)

	stats := engine.GetRejectedStats()
	if stats["proposal_wrong_leader"] != 1 {
		t.Fatalf("expected wrong-leader rejection count 1, got %d", stats["proposal_wrong_leader"])
	}
	if stats["proposal_qc_epoch_mismatch"] != 1 {
		t.Fatalf("expected qc-epoch rejection count 1, got %d", stats["proposal_qc_epoch_mismatch"])
	}
	if stats["proposal_parent_qc_mismatch"] != 1 {
		t.Fatalf("expected parent-qc rejection count 1, got %d", stats["proposal_parent_qc_mismatch"])
	}
}

func TestPacemakerLeaderBoundaryAtZeroView(t *testing.T) {
	pm := pacemaker.NewPacemaker([]uint64{11, 13, 17}, 20)
	if leader := pm.GetLeader(0); leader != 11 {
		t.Fatalf("expected leader 11 for view 0 boundary, got %d", leader)
	}
}

func TestEngineResetPacemakerSortsLargeValidatorIDs(t *testing.T) {
	engine := &Engine{timeoutMs: 20}
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		105: {ID: 105, PublicKey: []byte("v105"), Power: 1, IsActive: true},
		7:   {ID: 7, PublicKey: []byte("v7"), Power: 1, IsActive: true},
		42:  {ID: 42, PublicKey: []byte("v42"), Power: 1, IsActive: true},
	})

	engine.resetPacemaker(valSet)
	engine.pacemaker.SetLeaderSelector(nil)

	if leader := engine.pacemaker.GetLeader(1); leader != 7 {
		t.Fatalf("expected first leader id 7, got %d", leader)
	}
	if leader := engine.pacemaker.GetLeader(2); leader != 42 {
		t.Fatalf("expected second leader id 42, got %d", leader)
	}
	if leader := engine.pacemaker.GetLeader(3); leader != 105 {
		t.Fatalf("expected third leader id 105, got %d", leader)
	}
}

func TestEngineHydraAllowedLeadersDrivePacemakerSelection(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	engine.beacon = nil
	validators := make(map[uint64]*hydra.Validator, len(engine.valSet.Validators))
	for id, v := range engine.valSet.Validators {
		copyVal := *v
		validators[id] = &copyVal
	}
	hm, err := hydra.NewHydraManager(0, validators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	engine.SetHydraManager(hm)
	engine.resetPacemaker(engine.valSet)
	engine.pacemaker.SetLeaderSelector(nil)

	hm.LSetManager.InstallConfiguration(&hydra.Configuration{
		ID: 1,
		Validators: map[uint64]*hydra.Validator{
			1: {ID: 1, PublicKey: keypairs[1].PublicKey, Power: 1, IsActive: true},
			3: {ID: 3, PublicKey: keypairs[3].PublicKey, Power: 1, IsActive: true},
		},
		QuorumSize: 2,
	})
	engine.refreshLeaderSelector()
	engine.pacemaker.SetLeaderSelector(nil)

	if leader := engine.pacemaker.GetLeader(1); leader != 1 {
		t.Fatalf("expected hydra-constrained leader 1, got %d", leader)
	}
	if leader := engine.pacemaker.GetLeader(2); leader != 3 {
		t.Fatalf("expected hydra-constrained leader 3, got %d", leader)
	}
}

func TestEngineProposalBoundaryRejectsHydraDisallowedLeader(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	validators := make(map[uint64]*hydra.Validator, len(engine.valSet.Validators))
	for id, v := range engine.valSet.Validators {
		copyVal := *v
		validators[id] = &copyVal
	}
	hm, err := hydra.NewHydraManager(0, validators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	engine.SetHydraManager(hm)
	hm.LSetManager.InstallConfiguration(&hydra.Configuration{
		ID: 1,
		Validators: map[uint64]*hydra.Validator{
			1: {ID: 1, PublicKey: keypairs[1].PublicKey, Power: 1, IsActive: true},
		},
		QuorumSize: 1,
	})
	engine.refreshLeaderSelector()

	qc := types.NewQuorumCertificate([]byte("p"), 1, 1, types.PhasePrepare)
	block := stampProposalIdentity(types.NewBlock(1, qc.BlockHash, []byte("hydra"), 2, 1, 2, int64(engine.rankState.ExpectedRank(1)), qc, nil), 1, 0)
	if ok, reason := engine.validateProposalConstraint(block); ok || reason != "proposal_wrong_leader" {
		t.Fatalf("expected proposal_wrong_leader due to committed hydra leader set, got ok=%v reason=%s", ok, reason)
	}
}

func TestEngineHydraMarksDoNotChangePacemakerSelection(t *testing.T) {
	engine, _ := buildEngineVoteHarness(t)
	engine.beacon = nil
	validators := make(map[uint64]*hydra.Validator, len(engine.valSet.Validators))
	for id, v := range engine.valSet.Validators {
		copyVal := *v
		validators[id] = &copyVal
	}
	hm, err := hydra.NewHydraManager(0, validators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	engine.SetHydraManager(hm)
	hm.LSetManager.MarkFault(1, hydra.FaultClassUnavailable)
	hm.LSetManager.MarkFault(2, hydra.FaultClassDegraded)
	engine.refreshLeaderSelector()
	engine.pacemaker.SetLeaderSelector(nil)

	if leader := engine.pacemaker.GetLeader(1); leader != 0 {
		t.Fatalf("expected committed leader 0 despite local hydra marks, got %d", leader)
	}
	if leader := engine.pacemaker.GetLeader(2); leader != 1 {
		t.Fatalf("expected committed leader 1 despite local hydra marks, got %d", leader)
	}
	if !hm.IsAllowedLeader(1) {
		t.Fatalf("local hydra mark must not disallow committed leader 1")
	}
	if hm.LSetManager.IsAllowedToPropose(1) {
		t.Fatalf("typed fault marks should still block local proposal eligibility")
	}
	if hm.LSetManager.HasEvictableFault(2) {
		t.Fatalf("degraded mark should not become evictable")
	}
}

func TestEngineUpdateValidatorSetDoesNotDeadlock(t *testing.T) {
	engine, _ := buildEngineVoteHarness(t)
	newValSet := types.NewValidatorSet(2, map[uint64]*types.Validator{
		7:  {ID: 7, PublicKey: []byte("v7"), Power: 1, IsActive: true},
		42: {ID: 42, PublicKey: []byte("v42"), Power: 1, IsActive: true},
	})

	done := make(chan struct{})
	go func() {
		engine.UpdateValidatorSet(newValSet)
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(500 * time.Millisecond):
		t.Fatalf("UpdateValidatorSet deadlocked")
	}

	if engine.pacemaker.GetCurrentView() != 1 {
		t.Fatalf("expected pacemaker reset to view 1, got %d", engine.pacemaker.GetCurrentView())
	}
	leader := engine.pacemaker.GetLeader(1)
	if leader != 7 && leader != 42 {
		t.Fatalf("expected leader from updated validator set, got %d", leader)
	}
}

func TestHandleTimeoutVoteRejectsCrossConfigReplay(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	engine.vcm = viewchange.NewViewChangeManager(engine.nodeID, 20*time.Millisecond, func() uint64 {
		return engine.valSet.QuorumSize
	})
	engine.pacemaker.AdvanceView(1)
	vote := &types.TimeoutVote{
		View:     2,
		Epoch:    1,
		ConfigID: 99,
		Lane:     0,
		VoterID:  1,
	}
	vote.Signature = crypto.Sign(types.TimeoutSigningBytes(vote.View, vote.Epoch, vote.ConfigID, vote.Lane), keypairs[1].PrivateKey)

	engine.handleTimeoutVote(vote)

	stats := engine.GetRejectedStats()
	if stats["timeout_vote_config_mismatch"] != 1 {
		t.Fatalf("expected timeout_vote_config_mismatch rejection, got %d", stats["timeout_vote_config_mismatch"])
	}
}

func TestHandleTimeoutVoteRejectsFutureView(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	engine.vcm = viewchange.NewViewChangeManager(engine.nodeID, 20*time.Millisecond, func() uint64 {
		return engine.valSet.QuorumSize
	})
	vote := &types.TimeoutVote{
		View:     3,
		Epoch:    1,
		ConfigID: 1,
		Lane:     0,
		VoterID:  1,
	}
	vote.Signature = crypto.Sign(types.TimeoutSigningBytes(vote.View, vote.Epoch, vote.ConfigID, vote.Lane), keypairs[1].PrivateKey)

	engine.handleTimeoutVote(vote)

	stats := engine.GetRejectedStats()
	if stats["future_timeout_vote"] != 1 {
		t.Fatalf("expected future_timeout_vote rejection, got %#v", stats)
	}
	if engine.pacemaker.GetCurrentView() != 1 {
		t.Fatalf("future timeout vote must not advance view, got %d", engine.pacemaker.GetCurrentView())
	}
}

func TestEngineHandleMessageRejectsFutureTimeoutVote(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	engine.vcm = viewchange.NewViewChangeManager(engine.nodeID, 20*time.Millisecond, func() uint64 {
		return engine.valSet.QuorumSize
	})
	vote := &types.TimeoutVote{View: 3, Epoch: 1, ConfigID: 1, Lane: 0, VoterID: 1}
	vote.Signature = crypto.Sign(types.TimeoutSigningBytes(vote.View, vote.Epoch, vote.ConfigID, vote.Lane), keypairs[1].PrivateKey)
	msg := &types.Message{Type: types.MsgTimeout, SenderID: 1, View: 3, Epoch: 1, ConfigID: 1, Lane: 0, LeaderSetHash: engine.leaderSetHashSnapshot(), BarrierView: 3, Instance: 0, TimeoutVote: vote}
	if err := msg.Sign(keypairs[1].PrivateKey); err != nil {
		t.Fatalf("sign timeout message: %v", err)
	}

	engine.handleMessage(msg)

	stats := engine.GetRejectedStats()
	if stats["future_timeout_vote"] != 1 {
		t.Fatalf("expected future_timeout_vote rejection, got %#v", stats)
	}
	if engine.pacemaker.GetCurrentView() != 1 {
		t.Fatalf("future timeout vote must not advance view, got %d", engine.pacemaker.GetCurrentView())
	}
}

func TestEngineHandleMessageRejectsNilTimeoutVotePayload(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	engine.vcm = viewchange.NewViewChangeManager(engine.nodeID, 20*time.Millisecond, func() uint64 {
		return engine.valSet.QuorumSize
	})
	msg := &types.Message{
		Type:          types.MsgTimeout,
		SenderID:      1,
		View:          2,
		Epoch:         1,
		ConfigID:      1,
		Lane:          0,
		LeaderSetHash: engine.leaderSetHashSnapshot(),
		BarrierView:   2,
		Instance:      0,
	}
	if err := msg.Sign(keypairs[1].PrivateKey); err != nil {
		t.Fatalf("sign timeout message: %v", err)
	}
	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["nil_timeout_vote_payload"]; got != 1 {
		t.Fatalf("expected nil_timeout_vote_payload rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsTimeoutVoteSenderMismatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	engine.vcm = viewchange.NewViewChangeManager(engine.nodeID, 20*time.Millisecond, func() uint64 {
		return engine.valSet.QuorumSize
	})
	vote := &types.TimeoutVote{View: 2, Epoch: 1, ConfigID: 1, Lane: 0, VoterID: 1}
	vote.Signature = crypto.Sign(types.TimeoutSigningBytes(vote.View, vote.Epoch, vote.ConfigID, vote.Lane), keypairs[1].PrivateKey)
	msg := &types.Message{Type: types.MsgTimeout, SenderID: 2, View: 2, Epoch: 1, ConfigID: 1, Lane: 0, LeaderSetHash: engine.leaderSetHashSnapshot(), BarrierView: 2, Instance: 0, TimeoutVote: vote}
	if err := msg.Sign(keypairs[2].PrivateKey); err != nil {
		t.Fatalf("sign timeout message: %v", err)
	}
	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["timeout_vote_sender_mismatch"]; got != 1 {
		t.Fatalf("expected timeout_vote_sender_mismatch rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsTimeoutVoteViewMismatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	engine.vcm = viewchange.NewViewChangeManager(engine.nodeID, 20*time.Millisecond, func() uint64 {
		return engine.valSet.QuorumSize
	})
	vote := &types.TimeoutVote{View: 2, Epoch: 1, ConfigID: 1, Lane: 0, VoterID: 1}
	vote.Signature = crypto.Sign(types.TimeoutSigningBytes(vote.View, vote.Epoch, vote.ConfigID, vote.Lane), keypairs[1].PrivateKey)
	msg := &types.Message{Type: types.MsgTimeout, SenderID: 1, View: 3, Epoch: 1, ConfigID: 1, Lane: 0, LeaderSetHash: engine.leaderSetHashSnapshot(), BarrierView: 3, Instance: 0, TimeoutVote: vote}
	if err := msg.Sign(keypairs[1].PrivateKey); err != nil {
		t.Fatalf("sign timeout message: %v", err)
	}
	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["timeout_vote_view_wrapper_mismatch"]; got != 1 {
		t.Fatalf("expected timeout_vote_view_wrapper_mismatch rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsTimeoutVoteEpochMismatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	engine.vcm = viewchange.NewViewChangeManager(engine.nodeID, 20*time.Millisecond, func() uint64 {
		return engine.valSet.QuorumSize
	})
	vote := &types.TimeoutVote{View: 2, Epoch: 1, ConfigID: 1, Lane: 0, VoterID: 1}
	vote.Signature = crypto.Sign(types.TimeoutSigningBytes(vote.View, vote.Epoch, vote.ConfigID, vote.Lane), keypairs[1].PrivateKey)
	msg := &types.Message{Type: types.MsgTimeout, SenderID: 1, View: 2, Epoch: 2, ConfigID: 1, Lane: 0, LeaderSetHash: engine.leaderSetHashSnapshot(), BarrierView: 2, Instance: 0, TimeoutVote: vote}
	if err := msg.Sign(keypairs[1].PrivateKey); err != nil {
		t.Fatalf("sign timeout message: %v", err)
	}
	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["timeout_vote_epoch_wrapper_mismatch"]; got != 1 {
		t.Fatalf("expected timeout_vote_epoch_wrapper_mismatch rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsTimeoutVoteConfigMismatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	engine.vcm = viewchange.NewViewChangeManager(engine.nodeID, 20*time.Millisecond, func() uint64 {
		return engine.valSet.QuorumSize
	})
	vote := &types.TimeoutVote{View: 2, Epoch: 1, ConfigID: 2, Lane: 0, VoterID: 1}
	vote.Signature = crypto.Sign(types.TimeoutSigningBytes(vote.View, vote.Epoch, vote.ConfigID, vote.Lane), keypairs[1].PrivateKey)
	msg := &types.Message{Type: types.MsgTimeout, SenderID: 1, View: 2, Epoch: 1, ConfigID: 1, Lane: 0, LeaderSetHash: engine.leaderSetHashSnapshot(), BarrierView: 2, Instance: 0, TimeoutVote: vote}
	if err := msg.Sign(keypairs[1].PrivateKey); err != nil {
		t.Fatalf("sign timeout message: %v", err)
	}
	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["timeout_vote_config_wrapper_mismatch"]; got != 1 {
		t.Fatalf("expected timeout_vote_config_wrapper_mismatch rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsTimeoutVoteLaneMismatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	engine.vcm = viewchange.NewViewChangeManager(engine.nodeID, 20*time.Millisecond, func() uint64 {
		return engine.valSet.QuorumSize
	})
	vote := &types.TimeoutVote{View: 2, Epoch: 1, ConfigID: 1, Lane: 1, VoterID: 1}
	vote.Signature = crypto.Sign(types.TimeoutSigningBytes(vote.View, vote.Epoch, vote.ConfigID, vote.Lane), keypairs[1].PrivateKey)
	msg := &types.Message{Type: types.MsgTimeout, SenderID: 1, View: 2, Epoch: 1, ConfigID: 1, Lane: 0, LeaderSetHash: engine.leaderSetHashSnapshot(), BarrierView: 2, Instance: 0, TimeoutVote: vote}
	if err := msg.Sign(keypairs[1].PrivateKey); err != nil {
		t.Fatalf("sign timeout message: %v", err)
	}
	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["timeout_vote_lane_wrapper_mismatch"]; got != 1 {
		t.Fatalf("expected timeout_vote_lane_wrapper_mismatch rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsProposalBarrierViewMismatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	block := stampProposalIdentity(types.NewBlock(1, make([]byte, 32), nil, 2, 1, 1, 1, nil, nil), 1, 0)
	msg := signProposalMessageForTest(t, engine, keypairs[1], 1, block)
	msg.BarrierView = 3
	if err := msg.Sign(keypairs[1].PrivateKey); err != nil {
		t.Fatalf("re-sign proposal message: %v", err)
	}

	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["proposal_barrier_view_mismatch"]; got != 1 {
		t.Fatalf("expected proposal_barrier_view_mismatch rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsVoteBarrierViewMismatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	vote := newSignedVote(t, keypairs[1], []byte("vote-barrier-view"), 1)
	msg := signVoteMessageForTest(t, engine, keypairs[1], 1, vote)
	msg.BarrierView = 2
	if err := msg.Sign(keypairs[1].PrivateKey); err != nil {
		t.Fatalf("re-sign vote message: %v", err)
	}

	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["vote_barrier_view_mismatch"]; got != 1 {
		t.Fatalf("expected vote_barrier_view_mismatch rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsTimeoutVoteBarrierViewMismatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	engine.vcm = viewchange.NewViewChangeManager(engine.nodeID, 20*time.Millisecond, func() uint64 {
		return engine.valSet.QuorumSize
	})
	vote := &types.TimeoutVote{View: 2, Epoch: 1, ConfigID: 1, Lane: 0, VoterID: 1}
	vote.Signature = crypto.Sign(types.TimeoutSigningBytes(vote.View, vote.Epoch, vote.ConfigID, vote.Lane), keypairs[1].PrivateKey)
	msg := &types.Message{Type: types.MsgTimeout, SenderID: 1, View: 2, Epoch: 1, ConfigID: 1, Lane: 0, LeaderSetHash: engine.leaderSetHashSnapshot(), BarrierView: 3, Instance: 0, TimeoutVote: vote}
	if err := msg.Sign(keypairs[1].PrivateKey); err != nil {
		t.Fatalf("sign timeout message: %v", err)
	}

	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["timeout_vote_barrier_view_mismatch"]; got != 1 {
		t.Fatalf("expected timeout_vote_barrier_view_mismatch rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestVerifyTCRejectsConfigMismatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	tc := &types.TimeoutCertificate{
		View:     2,
		Epoch:    1,
		ConfigID: 2,
		Lane:     0,
		Signatures: map[uint64][]byte{
			1: crypto.Sign(types.TimeoutSigningBytes(2, 1, 2, 0), keypairs[1].PrivateKey),
			2: crypto.Sign(types.TimeoutSigningBytes(2, 1, 2, 0), keypairs[2].PrivateKey),
			3: crypto.Sign(types.TimeoutSigningBytes(2, 1, 2, 0), keypairs[3].PrivateKey),
		},
		NumVoters: 3,
	}

	if engine.verifyTC(tc) {
		t.Fatalf("expected verifyTC to reject config-mismatched TC")
	}
}

func TestHandleNewViewRejectsInvalidTCHighQC(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	badQC := types.NewQuorumCertificateWithIdentity([]byte("bad-qc"), 2, 1, 1, 0, types.PhasePrepare)
	badQC.AddSignature(1, []byte("bad-sig-1"))
	badQC.AddSignature(2, []byte("bad-sig-2"))
	badQC.AddSignature(3, []byte("bad-sig-3"))
	msg := &types.Message{
		Type:     types.MsgNewView,
		SenderID: 1,
		View:     3,
		Epoch:    1,
		ConfigID: 1,
		Lane:     0,
		Instance: 0,
		TC: &types.TimeoutCertificate{
			View:      2,
			Epoch:     1,
			ConfigID:  1,
			Lane:      0,
			HighestQC: badQC,
			Signatures: map[uint64][]byte{
				1: crypto.Sign(types.TimeoutSigningBytes(2, 1, 1, 0), keypairs[1].PrivateKey),
				2: crypto.Sign(types.TimeoutSigningBytes(2, 1, 1, 0), keypairs[2].PrivateKey),
				3: crypto.Sign(types.TimeoutSigningBytes(2, 1, 1, 0), keypairs[3].PrivateKey),
			},
			NumVoters: 3,
		},
	}

	engine.handleNewView(msg)

	stats := engine.GetRejectedStats()
	if stats["newview_invalid_tc_highqc"] != 1 {
		t.Fatalf("expected newview_invalid_tc_highqc rejection, got %d", stats["newview_invalid_tc_highqc"])
	}
}

func TestHandleNewViewRejectsFutureTCHighQC(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	futureQC := newSignedQCForTest(t, keypairs, []byte("future-tc-highqc"), 3, 1, 1, 0)
	msg := &types.Message{
		Type:     types.MsgNewView,
		SenderID: 1,
		View:     3,
		Epoch:    1,
		ConfigID: 1,
		Lane:     0,
		Instance: 0,
		TC: &types.TimeoutCertificate{
			View:      2,
			Epoch:     1,
			ConfigID:  1,
			Lane:      0,
			HighestQC: futureQC,
			Signatures: map[uint64][]byte{
				1: crypto.Sign(types.TimeoutSigningBytes(2, 1, 1, 0), keypairs[1].PrivateKey),
				2: crypto.Sign(types.TimeoutSigningBytes(2, 1, 1, 0), keypairs[2].PrivateKey),
				3: crypto.Sign(types.TimeoutSigningBytes(2, 1, 1, 0), keypairs[3].PrivateKey),
			},
			NumVoters: 3,
		},
	}

	engine.handleNewView(msg)

	stats := engine.GetRejectedStats()
	if stats["newview_tc_highqc_future_view"] != 1 {
		t.Fatalf("expected newview_tc_highqc_future_view rejection, got %#v", stats)
	}
	if highQC := engine.blockTree.GetHighQC(); highQC == nil || highQC.View != 0 {
		t.Fatalf("expected future TC.HighestQC rejection not to change highQC, got %+v", highQC)
	}
}

func TestEngineHandleMessageRejectsFutureTCHighQC(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	futureQC := newSignedQCForTest(t, keypairs, []byte("future-tc-highqc-msg"), 3, 1, 1, 0)
	msg := signNewViewMessageForTest(t, engine, keypairs[1], &types.Message{
		Type:        types.MsgNewView,
		SenderID:    1,
		View:        3,
		Epoch:       1,
		ConfigID:    1,
		Lane:        0,
		BarrierView: 3,
		Instance:    0,
		TC: &types.TimeoutCertificate{
			View:      2,
			Epoch:     1,
			ConfigID:  1,
			Lane:      0,
			HighestQC: futureQC,
			Signatures: map[uint64][]byte{
				1: crypto.Sign(types.TimeoutSigningBytes(2, 1, 1, 0), keypairs[1].PrivateKey),
				2: crypto.Sign(types.TimeoutSigningBytes(2, 1, 1, 0), keypairs[2].PrivateKey),
				3: crypto.Sign(types.TimeoutSigningBytes(2, 1, 1, 0), keypairs[3].PrivateKey),
			},
			NumVoters: 3,
		},
	})

	engine.handleMessage(msg)

	stats := engine.GetRejectedStats()
	if stats["newview_tc_highqc_future_view"] != 1 {
		t.Fatalf("expected signed message path to reject future TC.HighestQC, got %#v", stats)
	}
	if view := engine.pacemaker.GetCurrentView(); view != 1 {
		t.Fatalf("expected signed message path rejection not to advance pacemaker, got %d", view)
	}
}

func TestHandleTimeoutVoteRejectsFutureHighestQC(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	engine.vcm = viewchange.NewViewChangeManager(engine.nodeID, 20*time.Millisecond, func() uint64 {
		return engine.valSet.QuorumSize
	})
	engine.pacemaker.AdvanceView(1)
	futureQC := newSignedQCForTest(t, keypairs, []byte("timeout-future-highqc"), 3, 1, 1, 0)
	vote := &types.TimeoutVote{
		View:      2,
		Epoch:     1,
		ConfigID:  1,
		Lane:      0,
		VoterID:   1,
		HighestQC: futureQC,
	}
	vote.Signature = crypto.Sign(types.TimeoutSigningBytes(vote.View, vote.Epoch, vote.ConfigID, vote.Lane), keypairs[1].PrivateKey)

	engine.handleTimeoutVote(vote)

	stats := engine.GetRejectedStats()
	if stats["timeout_vote_highqc_future_view"] != 1 {
		t.Fatalf("expected timeout_vote_highqc_future_view rejection, got %#v", stats)
	}
	if tc := engine.vcm.GetHighestTC(); tc != nil {
		t.Fatalf("expected future HighestQC timeout vote not to form/store TC, got %+v", tc)
	}
}

func TestEngineHandleMessageRejectsTimeoutVoteFutureHighestQC(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	engine.vcm = viewchange.NewViewChangeManager(engine.nodeID, 20*time.Millisecond, func() uint64 {
		return engine.valSet.QuorumSize
	})
	engine.pacemaker.AdvanceView(1)
	futureQC := newSignedQCForTest(t, keypairs, []byte("timeout-future-highqc-msg"), 3, 1, 1, 0)
	vote := &types.TimeoutVote{
		View:      2,
		Epoch:     1,
		ConfigID:  1,
		Lane:      0,
		VoterID:   1,
		HighestQC: futureQC,
	}
	vote.Signature = crypto.Sign(types.TimeoutSigningBytes(vote.View, vote.Epoch, vote.ConfigID, vote.Lane), keypairs[1].PrivateKey)
	msg := &types.Message{Type: types.MsgTimeout, SenderID: 1, View: 2, Epoch: 1, ConfigID: 1, Lane: 0, LeaderSetHash: engine.leaderSetHashSnapshot(), BarrierView: 2, Instance: 0, TimeoutVote: vote}
	if err := msg.Sign(keypairs[1].PrivateKey); err != nil {
		t.Fatalf("sign timeout message: %v", err)
	}

	engine.handleMessage(msg)

	stats := engine.GetRejectedStats()
	if stats["timeout_vote_highqc_future_view"] != 1 {
		t.Fatalf("expected signed timeout message path to reject future HighestQC, got %#v", stats)
	}
	if tc := engine.vcm.GetHighestTC(); tc != nil {
		t.Fatalf("expected signed timeout message path not to form/store TC, got %+v", tc)
	}
}

func TestHandleTimeoutVoteRejectsInvalidHighestQC(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	engine.vcm = viewchange.NewViewChangeManager(engine.nodeID, 20*time.Millisecond, func() uint64 {
		return engine.valSet.QuorumSize
	})
	engine.pacemaker.AdvanceView(1)
	badQC := types.NewQuorumCertificateWithIdentity([]byte("timeout-invalid-highqc"), 2, 1, 1, 0, types.PhasePrepare)
	badQC.AddSignature(1, []byte("bad-sig-1"))
	badQC.AddSignature(2, []byte("bad-sig-2"))
	badQC.AddSignature(3, []byte("bad-sig-3"))
	vote := &types.TimeoutVote{
		View:      2,
		Epoch:     1,
		ConfigID:  1,
		Lane:      0,
		VoterID:   1,
		HighestQC: badQC,
	}
	vote.Signature = crypto.Sign(types.TimeoutSigningBytes(vote.View, vote.Epoch, vote.ConfigID, vote.Lane), keypairs[1].PrivateKey)

	engine.handleTimeoutVote(vote)

	stats := engine.GetRejectedStats()
	if stats["timeout_vote_invalid_highqc"] != 1 {
		t.Fatalf("expected timeout_vote_invalid_highqc rejection, got %#v", stats)
	}
	if tc := engine.vcm.GetHighestTC(); tc != nil {
		t.Fatalf("expected invalid HighestQC timeout vote not to form/store TC, got %+v", tc)
	}
}

func TestEngineHandleMessageRejectsTimeoutVoteInvalidHighestQC(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	engine.vcm = viewchange.NewViewChangeManager(engine.nodeID, 20*time.Millisecond, func() uint64 {
		return engine.valSet.QuorumSize
	})
	engine.pacemaker.AdvanceView(1)
	badQC := types.NewQuorumCertificateWithIdentity([]byte("timeout-invalid-highqc-msg"), 2, 1, 1, 0, types.PhasePrepare)
	badQC.AddSignature(1, []byte("bad-sig-1"))
	badQC.AddSignature(2, []byte("bad-sig-2"))
	badQC.AddSignature(3, []byte("bad-sig-3"))
	vote := &types.TimeoutVote{
		View:      2,
		Epoch:     1,
		ConfigID:  1,
		Lane:      0,
		VoterID:   1,
		HighestQC: badQC,
	}
	vote.Signature = crypto.Sign(types.TimeoutSigningBytes(vote.View, vote.Epoch, vote.ConfigID, vote.Lane), keypairs[1].PrivateKey)
	msg := &types.Message{Type: types.MsgTimeout, SenderID: 1, View: 2, Epoch: 1, ConfigID: 1, Lane: 0, LeaderSetHash: engine.leaderSetHashSnapshot(), BarrierView: 2, Instance: 0, TimeoutVote: vote}
	if err := msg.Sign(keypairs[1].PrivateKey); err != nil {
		t.Fatalf("sign timeout message: %v", err)
	}

	engine.handleMessage(msg)

	stats := engine.GetRejectedStats()
	if stats["timeout_vote_invalid_highqc"] != 1 {
		t.Fatalf("expected signed timeout message path to reject invalid HighestQC, got %#v", stats)
	}
	if tc := engine.vcm.GetHighestTC(); tc != nil {
		t.Fatalf("expected signed timeout message path not to form/store TC, got %+v", tc)
	}
}

func TestHandleTimeoutVoteRejectsHighestQCMismatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	engine.vcm = viewchange.NewViewChangeManager(engine.nodeID, 20*time.Millisecond, func() uint64 {
		return engine.valSet.QuorumSize
	})
	engine.pacemaker.AdvanceView(1)
	mismatchQC := newSignedQCForTest(t, keypairs, []byte("timeout-highqc-mismatch"), 2, 1, 2, 0)
	vote := &types.TimeoutVote{
		View:      2,
		Epoch:     1,
		ConfigID:  1,
		Lane:      0,
		VoterID:   1,
		HighestQC: mismatchQC,
	}
	vote.Signature = crypto.Sign(types.TimeoutSigningBytes(vote.View, vote.Epoch, vote.ConfigID, vote.Lane), keypairs[1].PrivateKey)

	engine.handleTimeoutVote(vote)

	stats := engine.GetRejectedStats()
	if stats["timeout_vote_highqc_mismatch"] != 1 {
		t.Fatalf("expected timeout_vote_highqc_mismatch rejection, got %#v", stats)
	}
	if tc := engine.vcm.GetHighestTC(); tc != nil {
		t.Fatalf("expected mismatched HighestQC timeout vote not to form/store TC, got %+v", tc)
	}
}

func TestEngineHandleMessageRejectsTimeoutVoteHighestQCMismatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	engine.vcm = viewchange.NewViewChangeManager(engine.nodeID, 20*time.Millisecond, func() uint64 {
		return engine.valSet.QuorumSize
	})
	engine.pacemaker.AdvanceView(1)
	mismatchQC := newSignedQCForTest(t, keypairs, []byte("timeout-highqc-mismatch-msg"), 2, 1, 2, 0)
	vote := &types.TimeoutVote{
		View:      2,
		Epoch:     1,
		ConfigID:  1,
		Lane:      0,
		VoterID:   1,
		HighestQC: mismatchQC,
	}
	vote.Signature = crypto.Sign(types.TimeoutSigningBytes(vote.View, vote.Epoch, vote.ConfigID, vote.Lane), keypairs[1].PrivateKey)
	msg := &types.Message{Type: types.MsgTimeout, SenderID: 1, View: 2, Epoch: 1, ConfigID: 1, Lane: 0, LeaderSetHash: engine.leaderSetHashSnapshot(), BarrierView: 2, Instance: 0, TimeoutVote: vote}
	if err := msg.Sign(keypairs[1].PrivateKey); err != nil {
		t.Fatalf("sign timeout message: %v", err)
	}

	engine.handleMessage(msg)

	stats := engine.GetRejectedStats()
	if stats["timeout_vote_highqc_mismatch"] != 1 {
		t.Fatalf("expected signed timeout message path to reject HighestQC mismatch, got %#v", stats)
	}
	if tc := engine.vcm.GetHighestTC(); tc != nil {
		t.Fatalf("expected signed timeout message path not to form/store TC, got %+v", tc)
	}
}

func TestHandleTimeoutVoteAcceptsValidHighestQC(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	engine.vcm = viewchange.NewViewChangeManager(engine.nodeID, 20*time.Millisecond, func() uint64 {
		return engine.valSet.QuorumSize
	})
	engine.network = &libp2pnetwork.P2PNetwork{}
	engine.pacemaker.AdvanceView(1)
	validQC := newSignedQCForTest(t, keypairs, []byte("timeout-valid-highqc"), 2, 1, 1, 0)

	for _, voterID := range []uint64{1, 2, 3} {
		vote := &types.TimeoutVote{
			View:      2,
			Epoch:     1,
			ConfigID:  1,
			Lane:      0,
			VoterID:   voterID,
			HighestQC: validQC,
		}
		vote.Signature = crypto.Sign(types.TimeoutSigningBytes(vote.View, vote.Epoch, vote.ConfigID, vote.Lane), keypairs[voterID].PrivateKey)
		func() {
			defer func() { _ = recover() }()
			engine.handleTimeoutVote(vote)
		}()
	}

	stats := engine.GetRejectedStats()
	if stats["timeout_vote_highqc_mismatch"] != 0 || stats["timeout_vote_highqc_future_view"] != 0 || stats["timeout_vote_invalid_highqc"] != 0 {
		t.Fatalf("expected valid HighestQC timeout votes to bypass rejection, got %#v", stats)
	}
	tc := engine.vcm.GetHighestTC()
	if tc == nil {
		t.Fatalf("expected valid HighestQC timeout votes to form/store TC")
	}
	if tc.HighestQC == nil || tc.HighestQC.View != validQC.View {
		t.Fatalf("expected formed TC to retain valid HighestQC view %d, got %+v", validQC.View, tc.HighestQC)
	}
}

func TestEngineHandleMessageAcceptsValidTimeoutVoteHighestQC(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	engine.vcm = viewchange.NewViewChangeManager(engine.nodeID, 20*time.Millisecond, func() uint64 {
		return engine.valSet.QuorumSize
	})
	engine.network = &libp2pnetwork.P2PNetwork{}
	engine.pacemaker.AdvanceView(1)
	validQC := newSignedQCForTest(t, keypairs, []byte("timeout-valid-highqc-msg"), 2, 1, 1, 0)

	for _, voterID := range []uint64{1, 2, 3} {
		vote := &types.TimeoutVote{
			View:      2,
			Epoch:     1,
			ConfigID:  1,
			Lane:      0,
			VoterID:   voterID,
			HighestQC: validQC,
		}
		vote.Signature = crypto.Sign(types.TimeoutSigningBytes(vote.View, vote.Epoch, vote.ConfigID, vote.Lane), keypairs[voterID].PrivateKey)
		msg := &types.Message{Type: types.MsgTimeout, SenderID: voterID, View: 2, Epoch: 1, ConfigID: 1, Lane: 0, LeaderSetHash: engine.leaderSetHashSnapshot(), BarrierView: 2, Instance: 0, TimeoutVote: vote}
		if err := msg.Sign(keypairs[voterID].PrivateKey); err != nil {
			t.Fatalf("sign timeout message: %v", err)
		}
		func() {
			defer func() { _ = recover() }()
			engine.handleMessage(msg)
		}()
	}

	stats := engine.GetRejectedStats()
	if stats["timeout_vote_highqc_mismatch"] != 0 || stats["timeout_vote_highqc_future_view"] != 0 || stats["timeout_vote_invalid_highqc"] != 0 || stats["invalid_message_signature"] != 0 {
		t.Fatalf("expected valid HighestQC timeout messages to bypass rejection, got %#v", stats)
	}
	tc := engine.vcm.GetHighestTC()
	if tc == nil {
		t.Fatalf("expected valid HighestQC timeout messages to form/store TC")
	}
	if tc.HighestQC == nil || tc.HighestQC.View != validQC.View {
		t.Fatalf("expected formed TC to retain valid HighestQC view %d, got %+v", validQC.View, tc.HighestQC)
	}
}

func TestEngineHandleMessageRejectsNewViewTCViewMismatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	highestQC := newSignedQCForTest(t, keypairs, []byte("newview-tc-view"), 2, 1, 1, 0)
	tc := newSignedTCForTest(t, keypairs, 2, 1, 1, 0, highestQC)
	msg := signNewViewMessageForTest(t, engine, keypairs[1], &types.Message{
		Type:     types.MsgNewView,
		SenderID: 1,
		View:     4,
		Epoch:    1,
		ConfigID: 1,
		Lane:     0,
		BarrierView: 4,
		Instance: 0,
		TC:       tc,
	})

	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["newview_tc_view_wrapper_mismatch"]; got != 1 {
		t.Fatalf("expected newview_tc_view_wrapper_mismatch rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsNewViewTCEpochMismatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	highestQC := newSignedQCForTest(t, keypairs, []byte("newview-tc-epoch"), 2, 1, 1, 0)
	tc := newSignedTCForTest(t, keypairs, 2, 2, 1, 0, highestQC)
	msg := signNewViewMessageForTest(t, engine, keypairs[1], &types.Message{
		Type:     types.MsgNewView,
		SenderID: 1,
		View:     3,
		Epoch:    1,
		ConfigID: 1,
		Lane:     0,
		BarrierView: 3,
		Instance: 0,
		TC:       tc,
	})

	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["newview_tc_epoch_wrapper_mismatch"]; got != 1 {
		t.Fatalf("expected newview_tc_epoch_wrapper_mismatch rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsNewViewTCBarrierViewMismatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	highestQC := newSignedQCForTest(t, keypairs, []byte("newview-tc-barrier"), 2, 1, 1, 0)
	tc := newSignedTCForTest(t, keypairs, 2, 1, 1, 0, highestQC)
	msg := signNewViewMessageForTest(t, engine, keypairs[1], &types.Message{
		Type:     types.MsgNewView,
		SenderID: 1,
		View:     3,
		Epoch:    1,
		ConfigID: 1,
		Lane:     0,
		BarrierView: 4,
		Instance: 0,
		TC:       tc,
	})

	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["newview_tc_barrier_view_mismatch"]; got != 1 {
		t.Fatalf("expected newview_tc_barrier_view_mismatch rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsNewViewTCConfigMismatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	highestQC := newSignedQCForTest(t, keypairs, []byte("newview-tc-config"), 2, 1, 2, 0)
	tc := newSignedTCForTest(t, keypairs, 2, 1, 2, 0, highestQC)
	msg := signNewViewMessageForTest(t, engine, keypairs[1], &types.Message{
		Type:     types.MsgNewView,
		SenderID: 1,
		View:     3,
		Epoch:    1,
		ConfigID: 1,
		Lane:     0,
		BarrierView: 3,
		Instance: 0,
		TC:       tc,
	})

	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["newview_tc_config_wrapper_mismatch"]; got != 1 {
		t.Fatalf("expected newview_tc_config_wrapper_mismatch rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsNewViewTCLaneMismatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	highestQC := newSignedQCForTest(t, keypairs, []byte("newview-tc-lane"), 2, 1, 1, 1)
	tc := newSignedTCForTest(t, keypairs, 2, 1, 1, 1, highestQC)
	msg := signNewViewMessageForTest(t, engine, keypairs[1], &types.Message{
		Type:     types.MsgNewView,
		SenderID: 1,
		View:     3,
		Epoch:    1,
		ConfigID: 1,
		Lane:     0,
		BarrierView: 3,
		Instance: 0,
		TC:       tc,
	})

	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["newview_tc_lane_wrapper_mismatch"]; got != 1 {
		t.Fatalf("expected newview_tc_lane_wrapper_mismatch rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsNewViewQCViewMismatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	qc := newSignedQCForTest(t, keypairs, []byte("newview-qc-view"), 2, 1, 1, 0)
	msg := signNewViewMessageForTest(t, engine, keypairs[1], &types.Message{
		Type:     types.MsgNewView,
		SenderID: 1,
		View:     4,
		Epoch:    1,
		ConfigID: 1,
		Lane:     0,
		BarrierView: 4,
		Instance: 0,
		QC:       qc,
	})

	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["newview_qc_view_wrapper_mismatch"]; got != 1 {
		t.Fatalf("expected newview_qc_view_wrapper_mismatch rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsNewViewQCEpochMismatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	qc := newSignedQCForTest(t, keypairs, []byte("newview-qc-epoch"), 2, 2, 1, 0)
	msg := signNewViewMessageForTest(t, engine, keypairs[1], &types.Message{
		Type:     types.MsgNewView,
		SenderID: 1,
		View:     3,
		Epoch:    1,
		ConfigID: 1,
		Lane:     0,
		BarrierView: 3,
		Instance: 0,
		QC:       qc,
	})

	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["newview_qc_epoch_wrapper_mismatch"]; got != 1 {
		t.Fatalf("expected newview_qc_epoch_wrapper_mismatch rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsNewViewQCBarrierViewMismatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	qc := newSignedQCForTest(t, keypairs, []byte("newview-qc-barrier"), 2, 1, 1, 0)
	msg := signNewViewMessageForTest(t, engine, keypairs[1], &types.Message{
		Type:     types.MsgNewView,
		SenderID: 1,
		View:     3,
		Epoch:    1,
		ConfigID: 1,
		Lane:     0,
		BarrierView: 4,
		Instance: 0,
		QC:       qc,
	})

	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["newview_qc_barrier_view_mismatch"]; got != 1 {
		t.Fatalf("expected newview_qc_barrier_view_mismatch rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsNewViewQCConfigMismatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	qc := newSignedQCForTest(t, keypairs, []byte("newview-qc-config"), 2, 1, 2, 0)
	msg := signNewViewMessageForTest(t, engine, keypairs[1], &types.Message{
		Type:     types.MsgNewView,
		SenderID: 1,
		View:     3,
		Epoch:    1,
		ConfigID: 1,
		Lane:     0,
		BarrierView: 3,
		Instance: 0,
		QC:       qc,
	})

	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["newview_qc_config_wrapper_mismatch"]; got != 1 {
		t.Fatalf("expected newview_qc_config_wrapper_mismatch rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsNewViewQCLaneMismatch(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	qc := newSignedQCForTest(t, keypairs, []byte("newview-qc-lane"), 2, 1, 1, 1)
	msg := signNewViewMessageForTest(t, engine, keypairs[1], &types.Message{
		Type:     types.MsgNewView,
		SenderID: 1,
		View:     3,
		Epoch:    1,
		ConfigID: 1,
		Lane:     0,
		BarrierView: 3,
		Instance: 0,
		QC:       qc,
	})

	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["newview_qc_lane_wrapper_mismatch"]; got != 1 {
		t.Fatalf("expected newview_qc_lane_wrapper_mismatch rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsNewViewMissingCertificate(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	msg := signNewViewMessageForTest(t, engine, keypairs[1], &types.Message{
		Type:     types.MsgNewView,
		SenderID: 1,
		View:     3,
		Epoch:    1,
		ConfigID: 1,
		Lane:     0,
		BarrierView: 3,
		Instance: 0,
	})

	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["newview_missing_certificate"]; got != 1 {
		t.Fatalf("expected newview_missing_certificate rejection, got %#v", engine.GetRejectedStats())
	}
}

func TestEngineHandleMessageRejectsInvalidNewViewQC(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	qc := types.NewQuorumCertificateWithIdentity([]byte("newview-invalid-qc"), 2, 1, 1, 0, types.PhasePrepare)
	qc.AddSignature(1, []byte("bad-sig-1"))
	qc.AddSignature(2, []byte("bad-sig-2"))
	qc.AddSignature(3, []byte("bad-sig-3"))
	msg := signNewViewMessageForTest(t, engine, keypairs[1], &types.Message{
		Type:     types.MsgNewView,
		SenderID: 1,
		View:     3,
		Epoch:    1,
		ConfigID: 1,
		Lane:     0,
		BarrierView: 3,
		Instance: 0,
		QC:       qc,
	})

	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["newview_invalid_qc"]; got != 1 {
		t.Fatalf("expected newview_invalid_qc rejection, got %#v", engine.GetRejectedStats())
	}
	if view := engine.pacemaker.GetCurrentView(); view != 1 {
		t.Fatalf("expected invalid QC NewView not to advance pacemaker, got view %d", view)
	}
}

func TestEngineHandleMessageRejectsNewViewRedundantTopLevelQC(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	highestQC := newSignedQCForTest(t, keypairs, []byte("newview-tc-top-level-qc-highqc"), 2, 1, 1, 0)
	tc := newSignedTCForTest(t, keypairs, 2, 1, 1, 0, highestQC)
	topLevelQC := newSignedQCForTest(t, keypairs, []byte("newview-redundant-top-level-qc"), 2, 1, 1, 0)
	msg := signNewViewMessageForTest(t, engine, keypairs[1], &types.Message{
		Type:     types.MsgNewView,
		SenderID: 1,
		View:     3,
		Epoch:    1,
		ConfigID: 1,
		Lane:     0,
		BarrierView: 3,
		Instance: 0,
		TC:       tc,
		QC:       topLevelQC,
	})

	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["newview_redundant_top_level_qc"]; got != 1 {
		t.Fatalf("expected newview_redundant_top_level_qc rejection, got %#v", engine.GetRejectedStats())
	}
	if view := engine.pacemaker.GetCurrentView(); view != 1 {
		t.Fatalf("expected redundant-top-level-QC NewView not to advance pacemaker, got view %d", view)
	}
}

func TestEngineHandleMessageRejectsStaleNewViewTC(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	baseQC := newSignedQCForTest(t, keypairs, []byte("newview-stale-tc-baseqc"), 2, 1, 1, 0)
	engine.blockTree.OnVoteQC(baseQC)
	engine.pacemaker.AdvanceView(3)
	highestQC := newSignedQCForTest(t, keypairs, []byte("newview-stale-tc-highqc"), 2, 1, 1, 0)
	tc := newSignedTCForTest(t, keypairs, 2, 1, 1, 0, highestQC)
	msg := signNewViewMessageForTest(t, engine, engine.keypair, engine.newTCBackedNewViewMessage(tc))

	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["stale_newview_tc"]; got != 1 {
		t.Fatalf("expected stale_newview_tc rejection, got %#v", engine.GetRejectedStats())
	}
	if view := engine.pacemaker.GetCurrentView(); view != 4 {
		t.Fatalf("expected stale TC NewView not to change pacemaker view, got %d", view)
	}
	if highQC := engine.blockTree.GetHighQC(); highQC == nil || highQC.View != 2 {
		t.Fatalf("expected stale TC NewView not to change highQC, got %+v", highQC)
	}
}

func TestEngineHandleMessageAcceptsStaleNewViewTCHighQCCatchUp(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	baseQC := newSignedQCForTest(t, keypairs, []byte("newview-stale-tc-catchup-baseqc"), 2, 1, 1, 0)
	engine.blockTree.OnVoteQC(baseQC)
	engine.pacemaker.AdvanceView(4)
	beforeView := engine.pacemaker.GetCurrentView()
	highestQC := newSignedQCForTest(t, keypairs, []byte("newview-stale-tc-catchup-highqc"), 3, 1, 1, 0)
	tc := newSignedTCForTest(t, keypairs, 3, 1, 1, 0, highestQC)
	msg := signNewViewMessageForTest(t, engine, engine.keypair, engine.newTCBackedNewViewMessage(tc))

	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["stale_newview_tc"]; got != 0 {
		t.Fatalf("expected no stale_newview_tc rejection during highQC catch-up, got %#v", engine.GetRejectedStats())
	}
	if view := engine.pacemaker.GetCurrentView(); view != beforeView {
		t.Fatalf("expected stale TC catch-up not to change pacemaker view, got %d want %d", view, beforeView)
	}
	if highQC := engine.blockTree.GetHighQC(); highQC == nil || highQC.View != 3 {
		t.Fatalf("expected stale TC catch-up to advance highQC to view 3, got %+v", highQC)
	}
}

func TestEngineHandleMessageAcceptsValidNewViewTC(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	baseQC := newSignedQCForTest(t, keypairs, []byte("newview-valid-tc-baseqc"), 1, 1, 1, 0)
	engine.blockTree.OnVoteQC(baseQC)
	highestQC := newSignedQCForTest(t, keypairs, []byte("newview-valid-tc-highqc"), 2, 1, 1, 0)
	tc := newSignedTCForTest(t, keypairs, 2, 1, 1, 0, highestQC)
	msg := signNewViewMessageForTest(t, engine, engine.keypair, engine.newTCBackedNewViewMessage(tc))

	engine.handleMessage(msg)
	if view := engine.pacemaker.GetCurrentView(); view != 3 {
		t.Fatalf("expected valid TC NewView to advance pacemaker to view 3, got %d", view)
	}
	if highQC := engine.blockTree.GetHighQC(); highQC == nil || highQC.View != 2 {
		t.Fatalf("expected valid TC NewView to adopt highest QC view 2, got %+v", highQC)
	}
}

func TestEngineBuildsTCBackedNewViewWithoutTopLevelQC(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	highestQC := newSignedQCForTest(t, keypairs, []byte("newview-emit-tc-highqc"), 2, 1, 1, 0)
	tc := newSignedTCForTest(t, keypairs, 2, 1, 1, 0, highestQC)
	msg := engine.newTCBackedNewViewMessage(tc)
	if msg == nil {
		t.Fatalf("expected tc-backed newview message")
	}
	if msg.Type != types.MsgNewView {
		t.Fatalf("expected tc-backed newview type %v, got %v", types.MsgNewView, msg.Type)
	}
	if msg.SenderID != engine.nodeID {
		t.Fatalf("expected tc-backed newview sender %d, got %d", engine.nodeID, msg.SenderID)
	}
	if msg.QC != nil {
		t.Fatalf("expected tc-backed newview to omit top-level QC")
	}
	if msg.TC != tc {
		t.Fatalf("expected tc-backed newview to retain TC payload")
	}
	if msg.View != tc.View+1 {
		t.Fatalf("expected tc-backed newview view %d, got %d", tc.View+1, msg.View)
	}
	if msg.Epoch != tc.Epoch {
		t.Fatalf("expected tc-backed newview epoch %d, got %d", tc.Epoch, msg.Epoch)
	}
	if msg.ConfigID != tc.ConfigID {
		t.Fatalf("expected tc-backed newview config %d, got %d", tc.ConfigID, msg.ConfigID)
	}
	if msg.Lane != tc.Lane {
		t.Fatalf("expected tc-backed newview lane %d, got %d", tc.Lane, msg.Lane)
	}
	if msg.Instance != tc.Lane {
		t.Fatalf("expected tc-backed newview instance %d, got %d", tc.Lane, msg.Instance)
	}
	if msg.BarrierView != tc.View+1 {
		t.Fatalf("expected tc-backed newview barrier view %d, got %d", tc.View+1, msg.BarrierView)
	}
	if !types.LeaderSetHashEqual(msg.LeaderSetHash, engine.leaderSetHashForConfigID(tc.ConfigID)) {
		t.Fatalf("expected tc-backed newview leader-set hash to bind tc config %d", tc.ConfigID)
	}
}

func TestEngineBuildsTCBackedNewViewFromTCIdentityInsteadOfCurrentSnapshot(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	initialValidators := make(map[uint64]*hydra.Validator, len(engine.valSet.Validators))
	for id, v := range engine.valSet.Validators {
		copyVal := *v
		initialValidators[id] = &copyVal
	}
	hm, err := hydra.NewHydraManager(engine.nodeID, initialValidators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	newCommittedConfig := &types.Configuration{
		ID: 2,
		Validators: map[uint64]*types.Validator{
			0: {ID: 0, PublicKey: keypairs[0].PublicKey, Power: 1, IsActive: true},
			1: {ID: 1, PublicKey: keypairs[1].PublicKey, Power: 1, IsActive: true},
			2: {ID: 2, PublicKey: keypairs[2].PublicKey, Power: 1, IsActive: true},
		},
		QuorumSize: 3,
	}
	if err := hm.InstallCommittedConfiguration(newCommittedConfig); err != nil {
		t.Fatalf("install new committed configuration: %v", err)
	}
	engine.SetHydraManager(hm)
	engine.UpdateValidatorSet(newCommittedConfig.ToValidatorSet())

	highestQC := newSignedQCForTest(t, keypairs, []byte("newview-tc-snapshot-drift"), 2, 1, 1, 0)
	tc := newSignedTCForTest(t, keypairs, 2, 1, 1, 0, highestQC)
	msg := engine.newTCBackedNewViewMessage(tc)
	if msg != nil {
		t.Fatalf("expected tc-backed newview build to fail closed when tc config %d is not the current committed engine config", tc.ConfigID)
	}
}

func TestEngineRejectsOldConfigTimeoutStateAfterCommittedConfigAdvance(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	engine.vcm = viewchange.NewViewChangeManager(engine.nodeID, 20*time.Millisecond, func() uint64 {
		return engine.valSet.QuorumSize
	})

	initialValidators := make(map[uint64]*hydra.Validator, len(engine.valSet.Validators))
	for id, v := range engine.valSet.Validators {
		copyVal := *v
		initialValidators[id] = &copyVal
	}
	hm, err := hydra.NewHydraManager(engine.nodeID, initialValidators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	newCommittedConfig := &types.Configuration{
		ID: 2,
		Validators: map[uint64]*types.Validator{
			0: {ID: 0, PublicKey: keypairs[0].PublicKey, Power: 1, IsActive: true},
			1: {ID: 1, PublicKey: keypairs[1].PublicKey, Power: 1, IsActive: true},
			2: {ID: 2, PublicKey: keypairs[2].PublicKey, Power: 1, IsActive: true},
		},
		QuorumSize: 3,
	}
	if err := hm.InstallCommittedConfiguration(newCommittedConfig); err != nil {
		t.Fatalf("install new committed configuration: %v", err)
	}
	engine.SetHydraManager(hm)
	engine.UpdateValidatorSet(newCommittedConfig.ToValidatorSet())
	currentEpoch := engine.currentEpochSnapshot()
	currentConfigID := engine.currentConfigIDSnapshot()

	beforeView := engine.pacemaker.GetCurrentView()
	beforeHighQC := engine.blockTree.GetHighQC()
	vote := &types.TimeoutVote{
		View:     beforeView,
		Epoch:    currentEpoch,
		ConfigID: currentConfigID - 1,
		Lane:     0,
		VoterID:  1,
	}
	vote.Signature = crypto.Sign(types.TimeoutSigningBytes(vote.View, vote.Epoch, vote.ConfigID, vote.Lane), keypairs[1].PrivateKey)

	engine.handleTimeoutVote(vote)

	stats := engine.GetRejectedStats()
	if stats["timeout_vote_config_mismatch"] != 1 {
		t.Fatalf("expected timeout_vote_config_mismatch rejection after committed config advance, got %#v", stats)
	}
	if got := engine.vcm.GetHighestTC(); got != nil {
		t.Fatalf("expected old-config timeout vote not to form or publish any TC, got %+v", got)
	}
	if view := engine.pacemaker.GetCurrentView(); view != beforeView {
		t.Fatalf("expected old-config timeout vote not to advance pacemaker, got %d want %d", view, beforeView)
	}
	if highQC := engine.blockTree.GetHighQC(); highQC == nil || beforeHighQC == nil || highQC.View != beforeHighQC.View {
		t.Fatalf("expected old-config timeout vote not to change highQC, got %+v want %+v", highQC, beforeHighQC)
	}

	oldHighestQC := newSignedQCForTest(t, keypairs, []byte("old-config-timeout-state"), beforeView, currentEpoch, currentConfigID-1, 0)
	oldTC := newSignedTCForTest(t, keypairs, beforeView, currentEpoch, currentConfigID-1, 0, oldHighestQC)
	if msg := engine.newTCBackedNewViewMessage(oldTC); msg != nil {
		t.Fatalf("expected old-config tc-backed newview build to fail closed after committed config advance")
	}
}

func TestEngineHandleMessageRejectsOldConfigNewViewAfterCommittedConfigAdvance(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)

	initialValidators := make(map[uint64]*hydra.Validator, len(engine.valSet.Validators))
	for id, v := range engine.valSet.Validators {
		copyVal := *v
		initialValidators[id] = &copyVal
	}
	hm, err := hydra.NewHydraManager(engine.nodeID, initialValidators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	newCommittedConfig := &types.Configuration{
		ID: 2,
		Validators: map[uint64]*types.Validator{
			0: {ID: 0, PublicKey: keypairs[0].PublicKey, Power: 1, IsActive: true},
			1: {ID: 1, PublicKey: keypairs[1].PublicKey, Power: 1, IsActive: true},
			2: {ID: 2, PublicKey: keypairs[2].PublicKey, Power: 1, IsActive: true},
		},
		QuorumSize: 3,
	}
	if err := hm.InstallCommittedConfiguration(newCommittedConfig); err != nil {
		t.Fatalf("install new committed configuration: %v", err)
	}
	engine.SetHydraManager(hm)
	engine.UpdateValidatorSet(newCommittedConfig.ToValidatorSet())
	currentEpoch := engine.currentEpochSnapshot()
	currentConfigID := engine.currentConfigIDSnapshot()
	beforeView := engine.pacemaker.GetCurrentView()
	beforeHighQC := engine.blockTree.GetHighQC()

	oldHighestQC := newSignedQCForTest(t, keypairs, []byte("old-config-newview-highqc"), beforeView, currentEpoch, currentConfigID-1, 0)
	oldTC := newSignedTCForTest(t, keypairs, beforeView, currentEpoch, currentConfigID-1, 0, oldHighestQC)
	msg := signNewViewMessageForTest(t, engine, keypairs[1], &types.Message{
		Type:          types.MsgNewView,
		SenderID:      1,
		View:          oldTC.View + 1,
		Epoch:         currentEpoch,
		ConfigID:      currentConfigID - 1,
		Lane:          0,
		LeaderSetHash: engine.leaderSetHashSnapshot(),
		BarrierView:   oldTC.View + 1,
		Instance:      0,
		TC:            oldTC,
	})

	engine.handleMessage(msg)

	stats := engine.GetRejectedStats()
	if stats["message_config_mismatch"] != 1 {
		t.Fatalf("expected inbound old-config newview to be rejected with message_config_mismatch, got %#v", stats)
	}
	if view := engine.pacemaker.GetCurrentView(); view != beforeView {
		t.Fatalf("expected inbound old-config newview not to advance pacemaker, got %d want %d", view, beforeView)
	}
	if highQC := engine.blockTree.GetHighQC(); highQC == nil || beforeHighQC == nil || highQC.View != beforeHighQC.View {
		t.Fatalf("expected inbound old-config newview not to change highQC, got %+v want %+v", highQC, beforeHighQC)
	}
}

func TestEngineHandleMessageRejectsStaleNewViewQC(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	baseQC := newSignedQCForTest(t, keypairs, []byte("newview-stale-qc-base"), 2, 1, 1, 0)
	engine.blockTree.OnVoteQC(baseQC)
	engine.pacemaker.AdvanceView(3)
	qc := newSignedQCForTest(t, keypairs, []byte("newview-stale-qc"), 2, 1, 1, 0)
	msg := signNewViewMessageForTest(t, engine, keypairs[1], &types.Message{
		Type:        types.MsgNewView,
		SenderID:    1,
		View:        3,
		Epoch:       1,
		ConfigID:    1,
		Lane:        0,
		BarrierView: 3,
		Instance:    0,
		QC:          qc,
	})

	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["stale_newview_qc"]; got != 1 {
		t.Fatalf("expected stale_newview_qc rejection, got %#v", engine.GetRejectedStats())
	}
	if view := engine.pacemaker.GetCurrentView(); view != 4 {
		t.Fatalf("expected stale QC NewView not to change pacemaker view, got %d", view)
	}
	if highQC := engine.blockTree.GetHighQC(); highQC == nil || highQC.View != 2 {
		t.Fatalf("expected stale QC NewView not to change highQC, got %+v", highQC)
	}
}

func TestEngineHandleMessageAcceptsStaleNewViewQCHighQCCatchUp(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	baseQC := newSignedQCForTest(t, keypairs, []byte("newview-stale-catchup-base"), 2, 1, 1, 0)
	engine.blockTree.OnVoteQC(baseQC)
	engine.pacemaker.AdvanceView(4)
	beforeView := engine.pacemaker.GetCurrentView()
	qc := newSignedQCForTest(t, keypairs, []byte("newview-stale-catchup-qc"), 3, 1, 1, 0)
	msg := signNewViewMessageForTest(t, engine, keypairs[1], &types.Message{
		Type:        types.MsgNewView,
		SenderID:    1,
		View:        4,
		Epoch:       1,
		ConfigID:    1,
		Lane:        0,
		BarrierView: 4,
		Instance:    0,
		QC:          qc,
	})

	engine.handleMessage(msg)
	if got := engine.GetRejectedStats()["stale_newview_qc"]; got != 0 {
		t.Fatalf("expected no stale_newview_qc rejection during highQC catch-up, got %#v", engine.GetRejectedStats())
	}
	if view := engine.pacemaker.GetCurrentView(); view != beforeView {
		t.Fatalf("expected stale catch-up QC NewView not to change pacemaker view, got %d want %d", view, beforeView)
	}
	if highQC := engine.blockTree.GetHighQC(); highQC == nil || highQC.View != 3 {
		t.Fatalf("expected stale catch-up QC NewView to advance highQC to view 3, got %+v", highQC)
	}
}

func TestEngineHandleMessageAcceptsValidNewViewQC(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	baseQC := newSignedQCForTest(t, keypairs, []byte("newview-valid-qc-base"), 1, 1, 1, 0)
	engine.blockTree.OnVoteQC(baseQC)
	qc := newSignedQCForTest(t, keypairs, []byte("newview-valid-qc"), 2, 1, 1, 0)
	msg := signNewViewMessageForTest(t, engine, keypairs[1], &types.Message{
		Type:        types.MsgNewView,
		SenderID:    1,
		View:        3,
		Epoch:       1,
		ConfigID:    1,
		Lane:        0,
		BarrierView: 3,
		Instance:    0,
		QC:          qc,
	})

	engine.handleMessage(msg)
	if view := engine.pacemaker.GetCurrentView(); view != 3 {
		t.Fatalf("expected valid QC NewView to advance pacemaker to view 3, got %d", view)
	}
	if highQC := engine.blockTree.GetHighQC(); highQC == nil || highQC.View != 2 {
		t.Fatalf("expected valid QC NewView to adopt QC view 2, got %+v", highQC)
	}
}

func TestBuildProposalBlockUsesCaughtUpHighQC(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	baseQC := newSignedQCForTest(t, keypairs, []byte("proposal-catchup-base"), 2, 1, 1, 0)
	engine.blockTree.OnVoteQC(baseQC)
	engine.pacemaker.AdvanceView(4)
	caughtUpQC := newSignedQCForTest(t, keypairs, []byte("proposal-catchup-highqc"), 3, 1, 1, 0)
	msg := signNewViewMessageForTest(t, engine, keypairs[1], &types.Message{
		Type:        types.MsgNewView,
		SenderID:    1,
		View:        4,
		Epoch:       1,
		ConfigID:    1,
		Lane:        0,
		BarrierView: 4,
		Instance:    0,
		QC:          caughtUpQC,
	})

	engine.handleMessage(msg)

	block := engine.buildProposalBlock(nil)
	if block == nil {
		t.Fatalf("expected proposal block")
	}
	if block.Justify == nil || block.Justify.View != 3 {
		t.Fatalf("expected proposal block justify QC view 3 after catch-up, got %+v", block.Justify)
	}
	if string(block.Parent) != string(caughtUpQC.BlockHash) {
		t.Fatalf("expected proposal block parent to use caught-up QC hash")
	}
	if block.View != engine.pacemaker.GetCurrentView() {
		t.Fatalf("expected proposal block view %d, got %d", engine.pacemaker.GetCurrentView(), block.View)
	}
}

func TestEngineAdaptiveTuningUpdatesCommitteeAndTimeout(t *testing.T) {
	engine := &Engine{
		timeoutMs:     1000,
		committeeSize: 0,
		pacemaker:     pacemaker.NewPacemaker([]uint64{0, 1, 2, 3}, 1000),
		reputation:    NewLeaderReputation(DefaultReputationConfig()),
	}

	engine.SetAdaptiveTuning(AdaptiveTuning{
		CommitteeSize: 6,
		TimeoutMs:     1500,
	})
	got := engine.GetAdaptiveTuning()
	if got.CommitteeSize != 6 {
		t.Fatalf("unexpected committee size: %d", got.CommitteeSize)
	}
	if got.TimeoutMs != 1500 {
		t.Fatalf("unexpected timeout: %d", got.TimeoutMs)
	}
	if engine.GetCommitteeSize() != 6 {
		t.Fatalf("committee size not applied")
	}
}

func TestEngineHandleVoteRejectsMissingVRFPublicKeyWhenCommitteeActive(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	engine.committeeSize = 4
	engine.vrfPubKeys = map[uint64]kyber.Point{}
	blockHash := make([]byte, 32)
	copy(blockHash, []byte("vrf-missing-key"))
	vote := newSignedVote(t, keypairs[1], blockHash, 1)
	vote.VRFProof = []byte("proof-present")

	engine.handleVote(vote)

	stats := engine.GetRejectedStats()
	if stats["missing_vrf_pubkey"] != 1 {
		t.Fatalf("expected missing_vrf_pubkey rejection, got %#v", stats)
	}
}

func TestEngineSetLocalVRFKeypairOverridesGeneratedKeypair(t *testing.T) {
	_, keypairs := buildEngineVoteHarness(t)
	validators := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: keypairs[0].PublicKey, Power: 1, IsActive: true},
	}
	valSet := types.NewValidatorSet(1, validators)
	store := storage.NewStorageManager(99)
	engine := NewEngineWithInstanceAndOptions(0, keypairs[0], valSet, nil, store, 0, 1, "test-consensus", nil, DefaultEngineOptions())
	generatedPub := engine.GetVRFPubKey()
	if generatedPub == nil {
		t.Fatalf("expected generated VRF public key")
	}

	manifestPriv, manifestPub := crypto.GenerateVRFKey()
	engine.SetLocalVRFKeypair(manifestPriv, manifestPub)

	gotPub := engine.GetVRFPubKey()
	if gotPub == nil {
		t.Fatalf("expected manifest VRF public key")
	}
	generatedRaw, err := generatedPub.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal generated VRF public key: %v", err)
	}
	gotRaw, err := gotPub.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal current VRF public key: %v", err)
	}
	manifestRaw, err := manifestPub.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal manifest VRF public key: %v", err)
	}
	if string(gotRaw) != string(manifestRaw) {
		t.Fatalf("engine did not adopt manifest VRF public key")
	}
	if string(gotRaw) == string(generatedRaw) {
		t.Fatalf("engine still exposes generated VRF public key")
	}
	registered := engine.vrfPubKeys[engine.nodeID]
	if registered == nil {
		t.Fatalf("expected local VRF public key to be registered")
	}
	registeredRaw, err := registered.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal registered VRF public key: %v", err)
	}
	if string(registeredRaw) != string(manifestRaw) {
		t.Fatalf("registered VRF public key does not match manifest key")
	}
}

func TestEngineHandleVoteAcceptsRegisteredVRFProofWhenCommitteeActive(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	engine.committeeSize = int(engine.valSet.TotalPower)
	engine.vrfPubKeys = map[uint64]kyber.Point{}

	vrfPriv, vrfPub := crypto.GenerateVRFKey()
	engine.vrfPubKeys[1] = vrfPub
	selected, _, proof := engine.beacon.AmICommitteeMember(vrfPriv, engine.valSet.TotalPower, engine.committeeSize)
	if !selected {
		t.Fatalf("expected registered validator to be selected when committee size equals total power")
	}

	blockHash := make([]byte, 32)
	copy(blockHash, []byte("vrf-registered-key"))
	vote := newSignedVote(t, keypairs[1], blockHash, 1)
	vote.VRFProof = proof

	engine.handleVote(vote)

	stats := engine.GetRejectedStats()
	if stats["missing_vrf_pubkey"] != 0 {
		t.Fatalf("unexpected missing_vrf_pubkey rejection: %#v", stats)
	}
	if stats["invalid_vrf_proof"] != 0 {
		t.Fatalf("unexpected invalid_vrf_proof rejection: %#v", stats)
	}
	collector := engine.voteCollectors[engine.collectorKey(1, 1, 1, 0, blockHash)]
	if collector == nil {
		t.Fatalf("expected vote collector to be created")
	}
	if len(collector.signers) != 1 {
		t.Fatalf("expected 1 signer after accepted vote, got %d", len(collector.signers))
	}
}

func TestEngineUpdateValidatorSetPrunesVRFKeysForRemovedValidators(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	localPriv, localPub := crypto.GenerateVRFKey()
	_, removedPub := crypto.GenerateVRFKey()
	engine.SetLocalVRFKeypair(localPriv, localPub)
	engine.vrfPubKeys[1] = removedPub
	engine.committeeSize = 1

	nextValSet := types.NewValidatorSet(2, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: keypairs[0].PublicKey, Power: 1, IsActive: true},
	})
	engine.UpdateValidatorSet(nextValSet)

	if _, ok := engine.vrfPubKeys[1]; ok {
		t.Fatalf("expected removed validator VRF key to be pruned on epoch change")
	}
	if _, ok := engine.vrfPubKeys[0]; !ok {
		t.Fatalf("expected local validator VRF key to remain registered after epoch change")
	}
}

func TestEngineUpdateValidatorSetAllowsRefreshingVRFKeyForRetainedValidator(t *testing.T) {
	engine, keypairs := buildEngineVoteHarness(t)
	_, oldPub := crypto.GenerateVRFKey()
	newPriv, newPub := crypto.GenerateVRFKey()
	engine.RegisterVRFPubKey(1, oldPub)

	nextValSet := types.NewValidatorSet(2, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: keypairs[0].PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: keypairs[1].PublicKey, Power: 1, IsActive: true},
	})
	engine.committeeSize = int(nextValSet.TotalPower)
	engine.UpdateValidatorSetWithVRFKeys(nextValSet, map[uint64]kyber.Point{1: newPub})

	blockHash := make([]byte, 32)
	copy(blockHash, []byte("refreshed-retained-validator-vrf"))
	selected, _, proof := engine.beacon.AmICommitteeMember(newPriv, nextValSet.TotalPower, engine.committeeSize)
	if !selected {
		t.Fatalf("expected retained validator to be selected when committee size is one")
	}

	var blockID types.Hash
	copy(blockID[:], blockHash)
	vote, err := types.NewVoteWithIdentity(blockID, 1, 2, 2, 0, keypairs[1].PublicKey, keypairs[1].PrivateKey, proof)
	if err != nil {
		t.Fatalf("create refreshed epoch-2 vote failed: %v", err)
	}
	retained := engine.vrfPubKeys[1]
	if retained == nil {
		t.Fatalf("expected retained validator VRF key to remain registered")
	}
	retainedRaw, err := retained.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal retained validator VRF public key: %v", err)
	}
	newRaw, err := newPub.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal refreshed VRF public key: %v", err)
	}
	oldRaw, err := oldPub.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal original VRF public key: %v", err)
	}
	if string(retainedRaw) != string(newRaw) {
		t.Fatalf("expected retained validator VRF key to be refreshed on re-registration")
	}
	if string(retainedRaw) == string(oldRaw) {
		t.Fatalf("expected retained validator VRF key to stop using stale material")
	}

	engine.handleVote(vote)
	stats := engine.GetRejectedStats()
	if stats["missing_vrf_pubkey"] != 0 {
		t.Fatalf("unexpected missing_vrf_pubkey rejection after atomic VRF refresh: %#v", stats)
	}
	if stats["invalid_vrf_proof"] != 0 {
		t.Fatalf("unexpected invalid_vrf_proof rejection after atomic VRF refresh: %#v", stats)
	}
	collector := engine.voteCollectors[engine.collectorKey(1, 2, 2, 0, blockHash)]
	if collector == nil {
		t.Fatalf("expected refreshed retained-validator vote to be accepted into collector")
	}
	if len(collector.signers) != 1 {
		t.Fatalf("expected exactly one signer after refreshed retained-validator vote, got %d", len(collector.signers))
	}
}
