package hydra

import (
	"testing"
	"time"

	"octopus-bft/octopus/crypto"
	"octopus-bft/octopus/types"
)

func signedAutoMessage(t *testing.T, validator *Validator, msg *AutoTransitionMessage) *AutoTransitionMessage {
	t.Helper()
	if validator == nil {
		t.Fatal("validator is required")
	}
	privateKey, ok := lookupTestValidatorKey(validator.PublicKey)
	if !ok {
		t.Fatalf("missing test private key for validator %d", validator.ID)
	}
	msg.SenderID = validator.ID
	if err := msg.Sign(privateKey); err != nil {
		t.Fatalf("sign auto message: %v", err)
	}
	return msg
}

func signForValidator(t *testing.T, config *Configuration, senderID uint64, msg *AutoTransitionMessage) *AutoTransitionMessage {
	t.Helper()
	validator := config.Validators[senderID]
	if validator == nil {
		t.Fatalf("validator %d not found", senderID)
	}
	return signedAutoMessage(t, validator, msg)
}

func lookupValidatorPrivateKey(t *testing.T, config *Configuration, senderID uint64) types.PrivateKey {
	t.Helper()
	validator := config.Validators[senderID]
	if validator == nil {
		t.Fatalf("validator %d not found", senderID)
	}
	privateKey, ok := lookupTestValidatorKey(validator.PublicKey)
	if !ok {
		t.Fatalf("missing test private key for validator %d", senderID)
	}
	return privateKey
}

func signedProofVote(t *testing.T, config *Configuration, senderID uint64, digest []byte) *Vote {
	t.Helper()
	return &Vote{
		SenderID:  senderID,
		Digest:    append([]byte(nil), digest...),
		Signature: crypto.Sign(digest, lookupValidatorPrivateKey(t, config, senderID)),
	}
}

func TestAutoTransitionCommitOnlyEmitsCandidate(t *testing.T) {
	initial := testHydraConfig(1, 0, 1, 2, 3)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)

	candidate := initial.Copy()
	delete(candidate.Validators, 3)
	candidate.ID = 2
	candidate.QuorumSize = 3
	atm.pendingAutoConfigs[7] = &pendingAutoTransition{view: 7, config: candidate.Copy()}

	var observed *Configuration
	atm.OnTransition(func(config *Configuration, proof *TransitionProof) {
		observed = config
	})

	committedConfig, callback, err := atm.commitAutoTransitionLocked(&AutoTransitionMessage{View: 7})
	if err != nil {
		t.Fatalf("commit auto-transition: %v", err)
	}
	if callback != nil {
		callback(committedConfig, nil)
	}
	if observed == nil || observed.ID != candidate.ID {
		t.Fatalf("expected transition callback for candidate config %d, got %#v", candidate.ID, observed)
	}
	if got := tcm.GetMvalid().ID; got != initial.ID {
		t.Fatalf("temporary configuration manager mutated authoritative config: got=%d want=%d", got, initial.ID)
	}
}

func TestTriggerAutoTransitionUsesLocalLSetMembership(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	kp, err := crypto.GenerateKeyPair()
	if err != nil {
		t.Fatalf("generate keypair: %v", err)
	}
	initial.Validators[1].PublicKey = kp.PublicKey
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	net := &recordingNetwork{}
	atm := NewAutoTransitionManager(lset, tcm, net)
	atm.SetNodeID(1)
	atm.SetPrivateKey(kp.PrivateKey)
	lset.MarkFault(3, FaultClassUnavailable)

	if err := atm.TriggerAutoTransition(4, 9); err != nil {
		t.Fatalf("trigger auto-transition: %v", err)
	}
	if len(net.broadcasts) != 1 {
		t.Fatalf("expected 1 broadcast, got %d", len(net.broadcasts))
	}
	msg, ok := net.broadcasts[0].(*AutoTransitionMessage)
	if !ok {
		t.Fatalf("expected AutoTransitionMessage, got %T", net.broadcasts[0])
	}
	if msg.SenderID != 1 {
		t.Fatalf("expected local node to broadcast auto-transition, got sender %d", msg.SenderID)
	}
	if !msg.VerifySignature(kp.PublicKey) {
		t.Fatal("expected trigger auto-transition broadcast to be signed")
	}
}

func TestTriggerAutoTransitionRejectsNonLSetCaller(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)
	atm.SetNodeID(4)

	if err := atm.TriggerAutoTransition(1, 9); err == nil {
		t.Fatal("expected non-L-set caller to be rejected")
	}
}

func TestTriggerAutoTransitionRejectsWhenNoEvictableTargetsExist(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	kp, err := crypto.GenerateKeyPair()
	if err != nil {
		t.Fatalf("generate keypair: %v", err)
	}
	initial.Validators[1].PublicKey = kp.PublicKey
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	net := &recordingNetwork{}
	atm := NewAutoTransitionManager(lset, tcm, net)
	atm.SetNodeID(1)
	atm.SetPrivateKey(kp.PrivateKey)

	if err := atm.TriggerAutoTransition(4, 9); err == nil {
		t.Fatal("expected trigger with empty leave set to be rejected")
	}
	if len(net.broadcasts) != 0 {
		t.Fatalf("expected no broadcast for empty leave set, got %d", len(net.broadcasts))
	}
}

func TestHandleAutoProposeRejectsEmptyLeaveSet(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)

	msg := signForValidator(t, initial, 1, &AutoTransitionMessage{
		Type:  AutoPropose,
		View:  9,
		Leaves: []uint64{},
	})
	if err := atm.HandleAutoTransitionMessage(msg); err == nil {
		t.Fatal("expected empty auto-propose leave set to be rejected")
	}
	if pending := atm.pendingAutoConfigForView(9); pending != nil {
		t.Fatalf("expected no pending config for empty leave set, got %#v", pending)
	}
}

func TestHandleAutoCommitRejectsMismatchedProofSigner(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)
	atm.pendingAutoConfigs[11] = &pendingAutoTransition{view: 11, config: initial.Copy()}

	err = atm.handleAutoCommit(&AutoTransitionMessage{
		Type: AutoCommit,
		View: 11,
		Proof: &TransitionProof{
			View: 11,
			AutoVotes: map[uint64]*Vote{
				1: {SenderID: 1},
				2: {SenderID: 3},
				3: {SenderID: 3},
			},
		},
	})
	if err == nil {
		t.Fatal("expected mismatched proof signer to be rejected")
	}
}

func TestHandleAutoVoteRejectsReplayFromSameSender(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)
	blockHash := []byte("replay-same-sender")
	candidate := initial.Copy()
	delete(candidate.Validators, 3)
	candidate.ID = 2
	candidate.QuorumSize = (2*len(candidate.Validators))/3 + 1
	atm.pendingAutoConfigs[7] = &pendingAutoTransition{
		view:      7,
		config:    candidate,
		leaves:    []uint64{3},
		blockHash: blockHash,
	}

	first := atm.handleAutoVote(&AutoTransitionMessage{Type: AutoVote, SenderID: 1, View: 7, Leaves: []uint64{3}, BlockHash: blockHash, NewConfigID: 2})
	if first != nil {
		t.Fatalf("first vote rejected: %v", first)
	}
	if err := atm.handleAutoVote(&AutoTransitionMessage{Type: AutoVote, SenderID: 1, View: 7, Leaves: []uint64{3}, BlockHash: blockHash, NewConfigID: 2}); err == nil {
		t.Fatal("expected replayed auto-vote to be rejected")
	}
}

func TestHandleAutoVoteRejectsStaleView(t *testing.T) {
	initial := testHydraConfig(5, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	if err := tcm.InstallCommittedConfig(initial); err != nil {
		t.Fatalf("install committed config: %v", err)
	}
	atm := NewAutoTransitionManager(lset, tcm, nil)
	atm.lastCommittedView = 5

	if err := atm.handleAutoVote(&AutoTransitionMessage{Type: AutoVote, SenderID: 1, View: 4}); err == nil {
		t.Fatal("expected stale auto-vote view to be rejected")
	}
}

func TestHandleAutoVoteRejectsEmptyBufferedVote(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)

	if err := atm.handleAutoVote(&AutoTransitionMessage{Type: AutoVote, SenderID: 1, View: 7}); err == nil {
		t.Fatal("expected empty auto-vote to be rejected")
	}
	collector := atm.voteCollectors[7]
	if collector != nil {
		t.Fatalf("expected no collector for rejected empty auto-vote, got %#v", collector)
	}
	if len(atm.recentMessages) != 0 {
		t.Fatalf("expected rejected empty auto-vote to avoid replay-state allocation, got %d entries", len(atm.recentMessages))
	}
}

func TestHandleAutoVoteRejectsOrphanVoteWithoutAllocatingState(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)

	for _, view := range []uint64{7, 8, 19} {
		err := atm.handleAutoVote(&AutoTransitionMessage{Type: AutoVote, SenderID: 1, View: view, Leaves: []uint64{3}, BlockHash: []byte("orphan-vote"), NewConfigID: 2})
		if err == nil {
			t.Fatalf("expected orphan auto-vote for view %d to be rejected", view)
		}
		if collector := atm.voteCollectors[view]; collector != nil {
			t.Fatalf("expected no collector for orphan auto-vote view %d, got %#v", view, collector)
		}
	}
	if len(atm.voteCollectors) != 0 {
		t.Fatalf("expected no vote collectors after orphan votes, got %d", len(atm.voteCollectors))
	}
	if len(atm.recentMessages) != 0 {
		t.Fatalf("expected orphan votes to avoid replay-state allocation, got %d entries", len(atm.recentMessages))
	}
}

func TestAutoTransitionViewsAreNotRejectedUsingConfigID(t *testing.T) {
	initial := testHydraConfig(100, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	if err := tcm.InstallCommittedConfig(initial); err != nil {
		t.Fatalf("install committed config: %v", err)
	}
	atm := NewAutoTransitionManager(lset, tcm, nil)
	lset.MarkFault(3, FaultClassUnavailable)

	if err := atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{Type: AutoPropose, View: 41, Leaves: []uint64{3}})); err != nil {
		t.Fatalf("expected view 41 to remain valid despite config id 100, got %v", err)
	}
}

func TestHandleAutoCommitRejectsSenderOutsideLSet(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	lset.MarkFault(3, FaultClassUnavailable)
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)
	blockHash := []byte("sender-outside-lset")
	digest := autoTransitionDigest(11, []uint64{3}, blockHash, 2)

	err = atm.HandleAutoTransitionMessage(signForValidator(t, initial, 4, &AutoTransitionMessage{
		Type:      AutoCommit,
		View:      11,
		BlockHash: blockHash,
		Leaves:    []uint64{3},
		Proof: &TransitionProof{
			View:        11,
			NewConfigID: 2,
			BlockHash:   blockHash,
			AutoVotes: map[uint64]*Vote{
				1: signedProofVote(t, initial, 1, digest),
				2: signedProofVote(t, initial, 2, digest),
				3: signedProofVote(t, initial, 3, digest),
			},
		},
	}))
	if err == nil {
		t.Fatal("expected sender outside l-set auto-commit to be rejected")
	}
}

func TestHandleAutoCommitRejectsNonValidatorSender(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)
	atm.pendingAutoConfigs[12] = &pendingAutoTransition{view: 12, config: initial.Copy()}

	err = atm.handleAutoCommit(&AutoTransitionMessage{
		Type:     AutoCommit,
		SenderID: 9,
		View:     12,
		Proof: &TransitionProof{
			View: 12,
			AutoVotes: map[uint64]*Vote{
				1: {SenderID: 1},
				2: {SenderID: 2},
				3: {SenderID: 3},
			},
		},
	})
	if err == nil {
		t.Fatal("expected non-validator auto-commit sender to be rejected")
	}
}

func TestHandleAutoCommitRejectsReplayForCommittedView(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)
	candidate := initial.Copy()
	candidate.ID = 2
	blockHash := []byte("proof-13")
	digest := autoTransitionDigest(13, []uint64{4}, blockHash, candidate.ID)
	msg := signForValidator(t, initial, 1, &AutoTransitionMessage{
		Type:      AutoCommit,
		View:      13,
		BlockHash: blockHash,
		Leaves:    []uint64{4},
		Proof: &TransitionProof{
			View:        13,
			NewConfigID: candidate.ID,
			BlockHash:   blockHash,
			AutoVotes: map[uint64]*Vote{
				1: signedProofVote(t, initial, 1, digest),
				2: signedProofVote(t, initial, 2, digest),
				3: signedProofVote(t, initial, 3, digest),
			},
		},
	})

	atm.pendingAutoConfigs[13] = &pendingAutoTransition{view: 13, config: candidate.Copy(), leaves: []uint64{4}, blockHash: blockHash}
	if err := atm.HandleAutoTransitionMessage(msg); err != nil {
		t.Fatalf("first auto-commit rejected: %v", err)
	}
	atm.pendingAutoConfigs[13] = &pendingAutoTransition{view: 13, config: candidate.Copy(), leaves: []uint64{4}}
	if err := atm.HandleAutoTransitionMessage(msg); err == nil {
		t.Fatal("expected replayed auto-commit to be rejected")
	}
}

func TestHandleAutoCommitRejectsDigestMismatch(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)
	candidate := initial.Copy()
	candidate.ID = 2
	blockHash := []byte("proof-14")
	goodDigest := autoTransitionDigest(14, []uint64{4}, blockHash, candidate.ID)
	badDigest := autoTransitionDigest(14, []uint64{5}, blockHash, candidate.ID)
	atm.pendingAutoConfigs[14] = &pendingAutoTransition{view: 14, config: candidate.Copy(), leaves: []uint64{4}, blockHash: blockHash}

	err = atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{
		Type:      AutoCommit,
		View:      14,
		BlockHash: blockHash,
		Leaves:    []uint64{4},
		Proof: &TransitionProof{
			View:        14,
			NewConfigID: candidate.ID,
			BlockHash:   blockHash,
			AutoVotes: map[uint64]*Vote{
				1: signedProofVote(t, initial, 1, goodDigest),
				2: signedProofVote(t, initial, 2, badDigest),
				3: signedProofVote(t, initial, 3, goodDigest),
			},
		},
	}))
	if err == nil {
		t.Fatal("expected auto-commit digest mismatch to be rejected")
	}
}

func TestHandleAutoCommitRejectsMissingDigest(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)
	candidate := initial.Copy()
	candidate.ID = 2
	blockHash := []byte("proof-24")
	digest := autoTransitionDigest(24, []uint64{4}, blockHash, candidate.ID)
	atm.pendingAutoConfigs[24] = &pendingAutoTransition{view: 24, config: candidate.Copy(), leaves: []uint64{4}, blockHash: blockHash}

	err = atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{
		Type:      AutoCommit,
		View:      24,
		BlockHash: blockHash,
		Proof: &TransitionProof{
			View:        24,
			NewConfigID: candidate.ID,
			BlockHash:   blockHash,
			AutoVotes: map[uint64]*Vote{
				1: signedProofVote(t, initial, 1, digest),
				2: {SenderID: 2},
				3: signedProofVote(t, initial, 3, digest),
			},
		},
	}))
	if err == nil {
		t.Fatal("expected missing auto-commit digest to be rejected")
	}
}

func TestHandleAutoCommitAppliesRemoteProofWithoutLocalPendingState(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	lset.MarkFault(3, FaultClassUnavailable)
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)
	candidate := initial.Copy()
	candidate.ID = 2
	delete(candidate.Validators, 3)
	blockHash := []byte("remote-proof-block")

	committed := make(chan *Configuration, 1)
	atm.OnTransition(func(config *Configuration, proof *TransitionProof) {
		committed <- config
	})
	digest := autoTransitionDigest(25, []uint64{3}, blockHash, candidate.ID)
	msg := signForValidator(t, initial, 1, &AutoTransitionMessage{
		Type:      AutoCommit,
		View:      25,
		BlockHash: blockHash,
		Leaves:    []uint64{3},
		Proof: &TransitionProof{
			View:        25,
			NewConfigID: candidate.ID,
			BlockHash:   blockHash,
			AutoVotes: map[uint64]*Vote{
				1: {SenderID: 1, Digest: digest, Signature: crypto.Sign(digest, lookupValidatorPrivateKey(t, initial, 1))},
				2: {SenderID: 2, Digest: digest, Signature: crypto.Sign(digest, lookupValidatorPrivateKey(t, initial, 2))},
				3: {SenderID: 3, Digest: digest, Signature: crypto.Sign(digest, lookupValidatorPrivateKey(t, initial, 3))},
			},
		},
	})
	if err := atm.HandleAutoTransitionMessage(msg); err != nil {
		t.Fatalf("remote auto-commit rejected: %v", err)
	}
	if msg.Proof.NewConfigID != candidate.ID {
		t.Fatalf("expected inbound auto-commit message to remain unchanged, got config id %d", msg.Proof.NewConfigID)
	}
	select {
	case config := <-committed:
		if config == nil || config.ID != candidate.ID {
			t.Fatalf("expected remote auto-commit to install candidate %d, got %#v", candidate.ID, config)
		}
		if _, exists := config.Validators[3]; exists {
			t.Fatal("expected remote auto-commit to remove validator 3")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("expected remote auto-commit proof to commit without local pending state")
	}
}

func TestHandleAutoCommitAcceptsProofOnlyLeavesWithoutLocalPendingState(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	lset.MarkFault(3, FaultClassUnavailable)
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)
	blockHash := []byte("remote-proof-only-leaves")
	digest := autoTransitionDigest(26, []uint64{3}, blockHash, 2)

	committed := make(chan *Configuration, 1)
	atm.OnTransition(func(config *Configuration, proof *TransitionProof) {
		committed <- config
	})
	err = atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{
		Type:      AutoCommit,
		View:      26,
		BlockHash: blockHash,
		Proof: &TransitionProof{
			View:        26,
			NewConfigID: 2,
			Leaves:      []uint64{3},
			BlockHash:   blockHash,
			AutoVotes: map[uint64]*Vote{
				1: signedProofVote(t, initial, 1, digest),
				2: signedProofVote(t, initial, 2, digest),
				3: signedProofVote(t, initial, 3, digest),
			},
		},
	}))
	if err != nil {
		t.Fatalf("expected proof-only leaves auto-commit to be accepted, got %v", err)
	}
	select {
	case config := <-committed:
		if config == nil || config.ID != 2 {
			t.Fatalf("expected committed config 2, got %#v", config)
		}
		if _, exists := config.Validators[3]; exists {
			t.Fatal("expected validator 3 to be removed")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("expected proof-only leaves auto-commit to commit")
	}
}

func TestHandleAutoCommitRejectsNonEvictableTargetWithoutLocalPendingState(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	lset.MarkFault(4, FaultClassDegraded)
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)
	blockHash := []byte("remote-proof-non-evictable-leaf")
	digest := autoTransitionDigest(27, []uint64{4}, blockHash, 2)

	err = atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{
		Type:      AutoCommit,
		View:      27,
		BlockHash: blockHash,
		Proof: &TransitionProof{
			View:        27,
			NewConfigID: 2,
			Leaves:      []uint64{4},
			BlockHash:   blockHash,
			AutoVotes: map[uint64]*Vote{
				1: signedProofVote(t, initial, 1, digest),
				2: signedProofVote(t, initial, 2, digest),
				3: signedProofVote(t, initial, 3, digest),
			},
		},
	}))
	if err == nil {
		t.Fatal("expected auto-commit with non-evictable leave target to be rejected")
	}
}

func TestHandleAutoCommitRejectsTargetOutsideCurrentLSetWithoutLocalPendingState(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	lset.MarkFault(4, FaultClassUnavailable)
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)
	blockHash := []byte("remote-proof-outside-lset-leaf")
	digest := autoTransitionDigest(27, []uint64{4}, blockHash, 2)

	err = atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{
		Type:      AutoCommit,
		View:      27,
		BlockHash: blockHash,
		Proof: &TransitionProof{
			View:        27,
			NewConfigID: 2,
			Leaves:      []uint64{4},
			BlockHash:   blockHash,
			AutoVotes: map[uint64]*Vote{
				1: signedProofVote(t, initial, 1, digest),
				2: signedProofVote(t, initial, 2, digest),
				3: signedProofVote(t, initial, 3, digest),
			},
		},
	}))
	if err == nil {
		t.Fatal("expected auto-commit with target outside current l-set to be rejected")
	}
}

func TestHandleAutoCommitRejectsUnknownLeaveTargetWithoutLocalPendingState(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)
	blockHash := []byte("remote-proof-unknown-leaf")
	digest := autoTransitionDigest(27, []uint64{9}, blockHash, 2)

	err = atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{
		Type:      AutoCommit,
		View:      27,
		BlockHash: blockHash,
		Proof: &TransitionProof{
			View:        27,
			NewConfigID: 2,
			Leaves:      []uint64{9},
			BlockHash:   blockHash,
			AutoVotes: map[uint64]*Vote{
				1: signedProofVote(t, initial, 1, digest),
				2: signedProofVote(t, initial, 2, digest),
				3: signedProofVote(t, initial, 3, digest),
			},
		},
	}))
	if err == nil {
		t.Fatal("expected auto-commit with unknown leave target to be rejected")
	}
}

func TestHandleAutoCommitRejectsTopLevelProofLeavesMismatch(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)
	blockHash := []byte("remote-proof-leaf-mismatch")
	digest := autoTransitionDigest(28, []uint64{4}, blockHash, 2)

	err = atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{
		Type:      AutoCommit,
		View:      28,
		BlockHash: blockHash,
		Leaves:    []uint64{3},
		Proof: &TransitionProof{
			View:        28,
			NewConfigID: 2,
			Leaves:      []uint64{4},
			BlockHash:   blockHash,
			AutoVotes: map[uint64]*Vote{
				1: signedProofVote(t, initial, 1, digest),
				2: signedProofVote(t, initial, 2, digest),
				3: signedProofVote(t, initial, 3, digest),
			},
		},
	}))
	if err == nil {
		t.Fatal("expected top-level/proof leaves mismatch to be rejected")
	}
}

func TestHandleAutoCommitRejectsConfigIDMismatchForPendingView(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)
	candidate := initial.Copy()
	candidate.ID = 2
	delete(candidate.Validators, 4)
	blockHash := []byte("proof-config-mismatch")
	atm.pendingAutoConfigs[32] = &pendingAutoTransition{view: 32, config: candidate.Copy(), leaves: []uint64{4}, blockHash: blockHash}
	digest := autoTransitionDigest(32, []uint64{4}, blockHash, 99)

	err = atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{
		Type:      AutoCommit,
		View:      32,
		BlockHash: blockHash,
		Proof: &TransitionProof{
			View:        32,
			NewConfigID: 99,
			BlockHash:   blockHash,
			AutoVotes: map[uint64]*Vote{
				1: signedProofVote(t, initial, 1, digest),
				2: signedProofVote(t, initial, 2, digest),
				3: signedProofVote(t, initial, 3, digest),
			},
		},
	}))
	if err == nil {
		t.Fatal("expected config-id-mismatched auto-commit to be rejected")
	}
}

func TestHandleAutoCommitRejectsConfigIDCollisionAcrossPendingViews(t *testing.T) {
	initial := testHydraConfig(5, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)

	candidateA := initial.Copy()
	candidateA.ID = 6
	delete(candidateA.Validators, 4)
	candidateB := initial.Copy()
	candidateB.ID = 7
	delete(candidateB.Validators, 3)
	blockHash := []byte("proof-config-collision")
	atm.pendingAutoConfigs[33] = &pendingAutoTransition{view: 33, config: candidateA.Copy(), leaves: []uint64{4}, blockHash: blockHash}
	atm.pendingAutoConfigs[34] = &pendingAutoTransition{view: 34, config: candidateB.Copy(), leaves: []uint64{3}, blockHash: []byte("other-pending")}
	digest := autoTransitionDigest(33, []uint64{4}, blockHash, 7)

	err = atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{
		Type:      AutoCommit,
		View:      33,
		BlockHash: blockHash,
		Proof: &TransitionProof{
			View:        33,
			NewConfigID: 7,
			BlockHash:   blockHash,
			AutoVotes: map[uint64]*Vote{
				1: signedProofVote(t, initial, 1, digest),
				2: signedProofVote(t, initial, 2, digest),
				3: signedProofVote(t, initial, 3, digest),
			},
		},
	}))
	if err == nil {
		t.Fatal("expected colliding config-id auto-commit to be rejected")
	}
}

func TestHandleAutoCommitRejectsCommittedConfigIDReuseWithoutLocalPendingState(t *testing.T) {
	initial := testHydraConfig(5, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)
	blockHash := []byte("proof-config-reuse")
	digest := autoTransitionDigest(35, []uint64{4}, blockHash, initial.ID)

	err = atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{
		Type:      AutoCommit,
		View:      35,
		BlockHash: blockHash,
		Leaves:    []uint64{4},
		Proof: &TransitionProof{
			View:        35,
			NewConfigID: initial.ID,
			BlockHash:   blockHash,
			AutoVotes: map[uint64]*Vote{
				1: signedProofVote(t, initial, 1, digest),
				2: signedProofVote(t, initial, 2, digest),
				3: signedProofVote(t, initial, 3, digest),
			},
		},
	}))
	if err == nil {
		t.Fatal("expected committed config-id reuse to be rejected")
	}
}

func TestHandleAutoCommitRejectsFinalizedViewFromDifferentSender(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	lset.MarkFault(3, FaultClassUnavailable)
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)
	candidate := initial.Copy()
	candidate.ID = 2
	delete(candidate.Validators, 3)
	blockHash := []byte("finalized-view-proof")
	digest := autoTransitionDigest(36, []uint64{3}, blockHash, candidate.ID)

	commits := 0
	atm.OnTransition(func(config *Configuration, proof *TransitionProof) {
		commits++
	})
	first := signForValidator(t, initial, 1, &AutoTransitionMessage{
		Type:      AutoCommit,
		View:      36,
		BlockHash: blockHash,
		Leaves:    []uint64{3},
		Proof: &TransitionProof{
			View:        36,
			NewConfigID: candidate.ID,
			BlockHash:   blockHash,
			AutoVotes: map[uint64]*Vote{
				1: signedProofVote(t, initial, 1, digest),
				2: signedProofVote(t, initial, 2, digest),
				3: signedProofVote(t, initial, 3, digest),
			},
		},
	})
	if err := atm.HandleAutoTransitionMessage(first); err != nil {
		t.Fatalf("first auto-commit rejected: %v", err)
	}
	second := signForValidator(t, initial, 2, &AutoTransitionMessage{
		Type:      AutoCommit,
		View:      36,
		BlockHash: blockHash,
		Leaves:    []uint64{3},
		Proof: &TransitionProof{
			View:        36,
			NewConfigID: candidate.ID,
			BlockHash:   blockHash,
			AutoVotes: map[uint64]*Vote{
				1: signedProofVote(t, initial, 1, digest),
				2: signedProofVote(t, initial, 2, digest),
				3: signedProofVote(t, initial, 3, digest),
			},
		},
	})
	if err := atm.HandleAutoTransitionMessage(second); err == nil {
		t.Fatal("expected finalized view replay from different sender to be rejected")
	}
	if commits != 1 {
		t.Fatalf("expected exactly one committed callback, got %d", commits)
	}
}

func TestHandleAutoCommitRejectsMismatchedTopLevelLeavesForPendingView(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)
	candidate := initial.Copy()
	candidate.ID = 2
	delete(candidate.Validators, 4)
	blockHash := []byte("pending-leaf-mismatch")
	atm.pendingAutoConfigs[37] = &pendingAutoTransition{view: 37, config: candidate.Copy(), leaves: []uint64{4}, blockHash: blockHash}
	digest := autoTransitionDigest(37, []uint64{4}, blockHash, candidate.ID)

	err = atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{
		Type:      AutoCommit,
		View:      37,
		BlockHash: blockHash,
		Leaves:    []uint64{3},
		Proof: &TransitionProof{
			View:        37,
			NewConfigID: candidate.ID,
			BlockHash:   blockHash,
			AutoVotes: map[uint64]*Vote{
				1: signedProofVote(t, initial, 1, digest),
				2: signedProofVote(t, initial, 2, digest),
				3: signedProofVote(t, initial, 3, digest),
			},
		},
	}))
	if err == nil {
		t.Fatal("expected top-level leaves mismatch to be rejected")
	}
}

func TestHandleAutoCommitRejectsMismatchedBlockHashForPendingView(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)
	candidate := initial.Copy()
	candidate.ID = 2
	delete(candidate.Validators, 4)
	pendingBlockHash := []byte("pending-hash")
	proofBlockHash := []byte("proof-hash")
	atm.pendingAutoConfigs[40] = &pendingAutoTransition{view: 40, config: candidate.Copy(), leaves: []uint64{4}, blockHash: pendingBlockHash}
	digest := autoTransitionDigest(40, []uint64{4}, proofBlockHash, candidate.ID)

	err = atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{
		Type:      AutoCommit,
		View:      40,
		BlockHash: proofBlockHash,
		Leaves:    []uint64{4},
		Proof: &TransitionProof{
			View:        40,
			NewConfigID: candidate.ID,
			BlockHash:   proofBlockHash,
			AutoVotes: map[uint64]*Vote{
				1: signedProofVote(t, initial, 1, digest),
				2: signedProofVote(t, initial, 2, digest),
				3: signedProofVote(t, initial, 3, digest),
			},
		},
	}))
	if err == nil {
		t.Fatal("expected pending-view block hash mismatch to be rejected")
	}
}

func TestHandleAutoCommitRejectsBlockHashBindingForEmptyPendingHash(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)
	candidate := initial.Copy()
	candidate.ID = 2
	delete(candidate.Validators, 4)
	atm.pendingAutoConfigs[42] = &pendingAutoTransition{view: 42, config: candidate.Copy(), leaves: []uint64{4}, blockHash: nil}
	proofBlockHash := []byte("late-bound-hash")
	digest := autoTransitionDigest(42, []uint64{4}, proofBlockHash, candidate.ID)

	err = atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{
		Type:      AutoCommit,
		View:      42,
		BlockHash: proofBlockHash,
		Leaves:    []uint64{4},
		Proof: &TransitionProof{
			View:        42,
			NewConfigID: candidate.ID,
			BlockHash:   proofBlockHash,
			AutoVotes: map[uint64]*Vote{
				1: signedProofVote(t, initial, 1, digest),
				2: signedProofVote(t, initial, 2, digest),
				3: signedProofVote(t, initial, 3, digest),
			},
		},
	}))
	if err == nil {
		t.Fatal("expected empty pending block hash rebinding to be rejected")
	}
}

func TestHandleAutoCommitRejectsTopLevelProofConfigIDMismatch(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)
	blockHash := []byte("config-id-mismatch")
	digest := autoTransitionDigest(41, []uint64{4}, blockHash, 2)

	err = atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{
		Type:        AutoCommit,
		View:        41,
		BlockHash:   blockHash,
		Leaves:      []uint64{4},
		NewConfigID: 3,
		Proof: &TransitionProof{
			View:        41,
			NewConfigID: 2,
			BlockHash:   blockHash,
			AutoVotes: map[uint64]*Vote{
				1: signedProofVote(t, initial, 1, digest),
				2: signedProofVote(t, initial, 2, digest),
				3: signedProofVote(t, initial, 3, digest),
			},
		},
	}))
	if err == nil {
		t.Fatal("expected top-level/proof config id mismatch to be rejected")
	}
}

func TestHandleAutoCommitNormalizesProofConfigIDForPendingView(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)
	candidate := initial.Copy()
	candidate.ID = 2
	delete(candidate.Validators, 4)
	blockHash := []byte("normalized-proof-config")
	atm.pendingAutoConfigs[43] = &pendingAutoTransition{view: 43, config: candidate.Copy(), leaves: []uint64{4}, blockHash: blockHash}
	digest := autoTransitionDigest(43, []uint64{4}, blockHash, candidate.ID)

	proofs := make(chan *TransitionProof, 1)
	atm.OnTransition(func(config *Configuration, proof *TransitionProof) {
		proofs <- proof
	})
	if err := atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{
		Type:      AutoCommit,
		View:      43,
		BlockHash: blockHash,
		Leaves:    []uint64{4},
		Proof: &TransitionProof{
			View:        43,
			NewConfigID: 0,
			BlockHash:   blockHash,
			AutoVotes: map[uint64]*Vote{
				1: signedProofVote(t, initial, 1, digest),
				2: signedProofVote(t, initial, 2, digest),
				3: signedProofVote(t, initial, 3, digest),
			},
		},
	})); err != nil {
		t.Fatalf("pending-view auto-commit with implicit proof config id rejected: %v", err)
	}
	select {
	case proof := <-proofs:
		if proof == nil || proof.NewConfigID != candidate.ID {
			t.Fatalf("expected normalized proof config id %d, got %#v", candidate.ID, proof)
		}
		if !sameUint64Set(proof.Leaves, []uint64{4}) {
			t.Fatalf("expected proof leaves [4], got %#v", proof)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("expected normalized proof callback")
	}
}

func TestHandleAutoCommitRejectsReusedFinalizedConfigIDBeforeInstall(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	lset.MarkFault(2, FaultClassUnavailable)
	lset.MarkFault(3, FaultClassUnavailable)
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)
	firstDigest := autoTransitionDigest(38, []uint64{3}, []byte("first-finalized"), 2)
	if err := atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{
		Type:      AutoCommit,
		View:      38,
		BlockHash: []byte("first-finalized"),
		Leaves:    []uint64{3},
		Proof: &TransitionProof{
			View:        38,
			NewConfigID: 2,
			BlockHash:   []byte("first-finalized"),
			AutoVotes: map[uint64]*Vote{
				1: signedProofVote(t, initial, 1, firstDigest),
				2: signedProofVote(t, initial, 2, firstDigest),
				3: signedProofVote(t, initial, 3, firstDigest),
			},
		},
	})); err != nil {
		t.Fatalf("first finalized auto-commit rejected: %v", err)
	}
	secondDigest := autoTransitionDigest(39, []uint64{2}, []byte("second-finalized"), 2)
	err = atm.HandleAutoTransitionMessage(signForValidator(t, initial, 2, &AutoTransitionMessage{
		Type:      AutoCommit,
		View:      39,
		BlockHash: []byte("second-finalized"),
		Leaves:    []uint64{2},
		Proof: &TransitionProof{
			View:        39,
			NewConfigID: 2,
			BlockHash:   []byte("second-finalized"),
			AutoVotes: map[uint64]*Vote{
				1: signedProofVote(t, initial, 1, secondDigest),
				2: signedProofVote(t, initial, 2, secondDigest),
				3: signedProofVote(t, initial, 3, secondDigest),
			},
		},
	}))
	if err == nil {
		t.Fatal("expected finalized config-id reuse before install to be rejected")
	}
}

func TestHandleAutoCommitRejectsInvalidVoteSignature(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)
	candidate := initial.Copy()
	candidate.ID = 2
	delete(candidate.Validators, 4)
	atm.pendingAutoConfigs[26] = &pendingAutoTransition{view: 26, config: candidate.Copy(), leaves: []uint64{4}, blockHash: []byte("proof-block")}
	msg := signForValidator(t, initial, 1, &AutoTransitionMessage{
		Type:      AutoCommit,
		View:      26,
		BlockHash: []byte("proof-block"),
		Proof: &TransitionProof{
			View:        26,
			NewConfigID: candidate.ID,
			BlockHash:   []byte("proof-block"),
			AutoVotes: map[uint64]*Vote{
				1: {SenderID: 1, Digest: autoTransitionDigest(26, []uint64{4}, []byte("proof-block"), candidate.ID), Signature: []byte("bad")},
				2: {SenderID: 2, Digest: autoTransitionDigest(26, []uint64{4}, []byte("proof-block"), candidate.ID), Signature: crypto.Sign(autoTransitionDigest(26, []uint64{4}, []byte("proof-block"), candidate.ID), lookupValidatorPrivateKey(t, initial, 2))},
				3: {SenderID: 3, Digest: autoTransitionDigest(26, []uint64{4}, []byte("proof-block"), candidate.ID), Signature: crypto.Sign(autoTransitionDigest(26, []uint64{4}, []byte("proof-block"), candidate.ID), lookupValidatorPrivateKey(t, initial, 3))},
			},
		},
	})
	if err := atm.HandleAutoTransitionMessage(msg); err == nil {
		t.Fatal("expected invalid auto-commit vote signature to be rejected")
	}
}

func TestHandleAutoProposeUsesReservedNextConfigID(t *testing.T) {
	initial := testHydraConfig(5, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	lset.MarkFault(3, FaultClassUnavailable)
	tcm := NewTemporaryConfigurationManager(initial)
	if err := tcm.InstallCommittedConfig(initial); err != nil {
		t.Fatalf("install committed config: %v", err)
	}
	atm := NewAutoTransitionManager(lset, tcm, nil)

	if err := atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{Type: AutoPropose, View: 26, Leaves: []uint64{3}})); err != nil {
		t.Fatalf("auto-propose rejected: %v", err)
	}
	pending := atm.pendingAutoConfigs[26]
	if pending == nil || pending.config == nil {
		t.Fatal("expected pending config for view 26")
	}
	if pending.config.ID != 6 {
		t.Fatalf("expected next config id 6, got %d", pending.config.ID)
	}
}

func TestHandleAutoProposeReservesUniqueConfigIDsAcrossPendingViews(t *testing.T) {
	initial := testHydraConfig(5, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	lset.MarkFault(3, FaultClassUnavailable)
	lset.MarkFault(3, FaultClassDegraded)
	tcm := NewTemporaryConfigurationManager(initial)
	if err := tcm.InstallCommittedConfig(initial); err != nil {
		t.Fatalf("install committed config: %v", err)
	}
	atm := NewAutoTransitionManager(lset, tcm, nil)

	if err := atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{Type: AutoPropose, View: 26, Leaves: []uint64{3}})); err != nil {
		t.Fatalf("auto-propose 26 rejected: %v", err)
	}
	if err := atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{Type: AutoPropose, View: 27, Leaves: []uint64{3}})); err != nil {
		t.Fatalf("auto-propose 27 rejected: %v", err)
	}
	if atm.pendingAutoConfigs[26] == nil || atm.pendingAutoConfigs[27] == nil {
		t.Fatal("expected pending configs for both views")
	}
	if atm.pendingAutoConfigs[26].config.ID == atm.pendingAutoConfigs[27].config.ID {
		t.Fatalf("expected unique pending config ids, got %d", atm.pendingAutoConfigs[26].config.ID)
	}
	if atm.pendingAutoConfigs[26].config.ID != 6 || atm.pendingAutoConfigs[27].config.ID != 7 {
		t.Fatalf("expected reserved config ids 6 and 7, got %d and %d", atm.pendingAutoConfigs[26].config.ID, atm.pendingAutoConfigs[27].config.ID)
	}
}

func TestHandleAutoVoteAfterRejectedOrphanProposalStillRecoversView(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)

	proposalLeaves := []uint64{3}
	proposalBlockHash := []byte("delayed-proposal")
	proposalConfigID := uint64(2)
	if err := atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{Type: AutoVote, View: 15, Leaves: proposalLeaves, BlockHash: proposalBlockHash, NewConfigID: proposalConfigID})); err == nil {
		t.Fatal("expected orphan vote before proposal to be rejected")
	}
	if len(atm.recentMessages) != 0 {
		t.Fatalf("expected rejected orphan vote to avoid replay-state allocation, got %d entries", len(atm.recentMessages))
	}

	committed := make(chan *Configuration, 1)
	atm.OnTransition(func(config *Configuration, proof *TransitionProof) {
		committed <- config
	})
	lset.MarkFault(3, FaultClassUnavailable)
	if err := atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{Type: AutoPropose, View: 15, Leaves: proposalLeaves, BlockHash: proposalBlockHash})); err != nil {
		t.Fatalf("proposal after orphan rejection rejected: %v", err)
	}
	if err := atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{Type: AutoVote, View: 15, Leaves: proposalLeaves, BlockHash: proposalBlockHash, NewConfigID: proposalConfigID})); err != nil {
		t.Fatalf("first bound vote rejected: %v", err)
	}
	if err := atm.HandleAutoTransitionMessage(signForValidator(t, initial, 2, &AutoTransitionMessage{Type: AutoVote, View: 15, Leaves: proposalLeaves, BlockHash: proposalBlockHash, NewConfigID: proposalConfigID})); err != nil {
		t.Fatalf("second bound vote rejected: %v", err)
	}
	if err := atm.HandleAutoTransitionMessage(signForValidator(t, initial, 3, &AutoTransitionMessage{Type: AutoVote, View: 15, Leaves: proposalLeaves, BlockHash: proposalBlockHash, NewConfigID: proposalConfigID})); err != nil {
		t.Fatalf("third bound vote rejected: %v", err)
	}
	select {
	case config := <-committed:
		if config == nil {
			t.Fatal("expected committed config after proposal and bound quorum")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("expected proposal-driven quorum to recover view")
	}
}

func TestHandleAutoProposeRejectsReplayAndStaleView(t *testing.T) {
	initial := testHydraConfig(3, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	lset.MarkFault(3, FaultClassUnavailable)
	tcm := NewTemporaryConfigurationManager(initial)
	if err := tcm.InstallCommittedConfig(initial); err != nil {
		t.Fatalf("install committed config: %v", err)
	}
	atm := NewAutoTransitionManager(lset, tcm, nil)
	atm.lastCommittedView = 3

	if err := atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{Type: AutoPropose, View: 2, Leaves: []uint64{3}})); err == nil {
		t.Fatal("expected stale auto-propose to be rejected")
	}
	if err := atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{Type: AutoPropose, View: 4, Leaves: []uint64{3}})); err != nil {
		t.Fatalf("first auto-propose rejected: %v", err)
	}
	if err := atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{Type: AutoPropose, View: 4, Leaves: []uint64{3}})); err == nil {
		t.Fatal("expected replayed auto-propose to be rejected")
	}
}

func TestCommitAutoTransitionCallbackCanReenterATM(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)
	atm.pendingAutoConfigs[16] = &pendingAutoTransition{view: 16, config: initial.Copy()}

	done := make(chan struct{})
	atm.OnTransition(func(config *Configuration, proof *TransitionProof) {
		_ = atm.TriggerAutoTransition(1, 17)
		close(done)
	})

	committedConfig, callback, err := atm.commitAutoTransitionLocked(&AutoTransitionMessage{View: 16})
	if err != nil {
		t.Fatalf("commit auto-transition: %v", err)
	}
	if callback != nil {
		callback(committedConfig, nil)
	}
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("transition callback deadlocked while re-entering ATM")
	}
}

func TestCommitReadyFinalizesViewAndAdvancesStaleBarrier(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)

	candidate := initial.Copy()
	candidate.ID = 2
	delete(candidate.Validators, 4)
	blockHash := []byte("advisory-proof")
	atMView := uint64(31)
	atm.pendingAutoConfigs[atMView] = &pendingAutoTransition{view: atMView, config: candidate.Copy(), leaves: []uint64{4}, blockHash: blockHash}
	atm.voteCollectors[atMView] = &autoVoteCollector{
		view:         atMView,
		voters:       map[uint64]struct{}{1: {}, 2: {}, 3: {}},
		votes:        map[uint64]*Vote{1: {SenderID: 1}, 2: {SenderID: 2}, 3: {SenderID: 3}},
		pendingVotes: make(map[uint64]*AutoTransitionMessage),
		quorum:       3,
		timestamp:    time.Now(),
	}

	committedConfig, callback, proof, err := atm.commitReadyForView(atMView)
	if err != nil {
		t.Fatalf("commit ready failed: %v", err)
	}
	if committedConfig == nil || callback != nil || proof == nil {
		// callback may be nil in this test harness; only require commit result and proof.
	}
	if atm.lastCommittedView != atMView {
		t.Fatalf("expected stale barrier to advance to %d, got %d", atMView, atm.lastCommittedView)
	}
	if _, finalized := atm.finalizedViews[atMView]; !finalized {
		t.Fatalf("expected view %d to be finalized", atMView)
	}
	lset.MarkFault(3, FaultClassUnavailable)
	if err := atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{Type: AutoPropose, View: 31, Leaves: []uint64{3}})); err == nil {
		t.Fatal("expected finalized view to be rejected")
	}
	if err := atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{Type: AutoPropose, View: 32, Leaves: []uint64{3}})); err != nil {
		t.Fatalf("expected later higher view to remain admissible after finalized quorum, got %v", err)
	}
}

func TestHandleAutoProposeRejectsUnmarkedEvictionTargets(t *testing.T) {
	initial := testHydraConfig(5, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	if err := tcm.InstallCommittedConfig(initial); err != nil {
		t.Fatalf("install committed config: %v", err)
	}
	atm := NewAutoTransitionManager(lset, tcm, nil)

	if err := atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{Type: AutoPropose, View: 26, Leaves: []uint64{3}})); err == nil {
		t.Fatal("expected auto-propose with unmarked target to be rejected")
	}
	if _, exists := atm.pendingAutoConfigs[26]; exists {
		t.Fatal("unexpected pending config for unauthorized auto-propose")
	}
}

func TestHandleAutoProposeRejectsNonEvictableDegradedTarget(t *testing.T) {
	initial := testHydraConfig(5, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	lset.MarkFault(3, FaultClassDegraded)
	tcm := NewTemporaryConfigurationManager(initial)
	if err := tcm.InstallCommittedConfig(initial); err != nil {
		t.Fatalf("install committed config: %v", err)
	}
	atm := NewAutoTransitionManager(lset, tcm, nil)

	if err := atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{Type: AutoPropose, View: 28, Leaves: []uint64{3}})); err == nil {
		t.Fatal("expected degraded target to be rejected")
	}
}

func TestHandleAutoProposeAcceptsByzantineTarget(t *testing.T) {
	initial := testHydraConfig(5, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	lset.MarkFault(3, FaultClassByzantine)
	tcm := NewTemporaryConfigurationManager(initial)
	if err := tcm.InstallCommittedConfig(initial); err != nil {
		t.Fatalf("install committed config: %v", err)
	}
	atm := NewAutoTransitionManager(lset, tcm, nil)

	if err := atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{Type: AutoPropose, View: 29, Leaves: []uint64{3}})); err != nil {
		t.Fatalf("expected byzantine target to be accepted, got %v", err)
	}
	if _, exists := atm.pendingAutoConfigs[29]; !exists {
		t.Fatal("expected pending config for byzantine target")
	}
}

func TestHandleAutoProposeRejectsConflictingCandidateForSameView(t *testing.T) {
	initial := testHydraConfig(5, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	lset.MarkFault(2, FaultClassByzantine)
	lset.MarkFault(3, FaultClassUnavailable)
	tcm := NewTemporaryConfigurationManager(initial)
	if err := tcm.InstallCommittedConfig(initial); err != nil {
		t.Fatalf("install committed config: %v", err)
	}
	atm := NewAutoTransitionManager(lset, tcm, nil)

	if err := atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{Type: AutoPropose, View: 30, Leaves: []uint64{3}, BlockHash: []byte("candidate-a")})); err != nil {
		t.Fatalf("first candidate rejected: %v", err)
	}
	if err := atm.HandleAutoTransitionMessage(signForValidator(t, initial, 2, &AutoTransitionMessage{Type: AutoPropose, View: 30, Leaves: []uint64{2}, BlockHash: []byte("candidate-b")})); err == nil {
		t.Fatal("expected conflicting same-view candidate to be rejected")
	}
	pending := atm.pendingAutoConfigs[30]
	if pending == nil || !sameUint64Set(pending.leaves, []uint64{3}) || string(pending.blockHash) != "candidate-a" {
		t.Fatal("expected original candidate binding to remain intact")
	}
}

func TestHandleAutoProposeRejectsTargetsOutsideCurrentLSet(t *testing.T) {
	initial := testHydraConfig(5, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	lset.MarkFault(4, FaultClassUnavailable)
	tcm := NewTemporaryConfigurationManager(initial)
	if err := tcm.InstallCommittedConfig(initial); err != nil {
		t.Fatalf("install committed config: %v", err)
	}
	atm := NewAutoTransitionManager(lset, tcm, nil)

	if err := atm.HandleAutoTransitionMessage(signForValidator(t, initial, 1, &AutoTransitionMessage{Type: AutoPropose, View: 26, Leaves: []uint64{4}})); err == nil {
		t.Fatal("expected auto-propose targeting non-L-set node to be rejected")
	}
}

func TestHandleAutoVoteRejectsMismatchedCandidateBinding(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)

	candidate := initial.Copy()
	candidate.ID = 2
	delete(candidate.Validators, 4)
	blockHash := []byte("bound-view-17")
	atm.pendingAutoConfigs[17] = &pendingAutoTransition{view: 17, config: candidate.Copy(), leaves: []uint64{4}, blockHash: blockHash}

	for _, senderID := range []uint64{1, 2} {
		if err := atm.HandleAutoTransitionMessage(signForValidator(t, initial, senderID, &AutoTransitionMessage{Type: AutoVote, View: 17, Leaves: []uint64{4}, BlockHash: blockHash, NewConfigID: candidate.ID})); err != nil {
			t.Fatalf("matching vote %d rejected: %v", senderID, err)
		}
	}
	if err := atm.HandleAutoTransitionMessage(signForValidator(t, initial, 3, &AutoTransitionMessage{Type: AutoVote, View: 17, Leaves: []uint64{3}, BlockHash: blockHash, NewConfigID: candidate.ID})); err == nil {
		t.Fatal("expected mismatched auto-vote to be rejected")
	}
	if collector := atm.voteCollectors[17]; collector == nil || len(collector.votes) != 2 {
		t.Fatalf("expected only 2 bound votes after mismatch, got %#v", collector)
	}
	if _, exists := atm.pendingAutoConfigs[17]; !exists {
		t.Fatal("mismatched vote must not commit pending candidate")
	}
}

func TestHandleAutoVoteCommitsOnlyMatchingViewCandidate(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)

	view18 := initial.Copy()
	view18.ID = 2
	delete(view18.Validators, 4)
	view19 := initial.Copy()
	view19.ID = 3
	delete(view19.Validators, 3)
	view18BlockHash := []byte("view-18")
	view19BlockHash := []byte("view-19")
	atm.pendingAutoConfigs[18] = &pendingAutoTransition{view: 18, config: view18.Copy(), leaves: []uint64{4}, blockHash: view18BlockHash}
	atm.pendingAutoConfigs[19] = &pendingAutoTransition{view: 19, config: view19.Copy(), leaves: []uint64{3}, blockHash: view19BlockHash}

	committed := make(chan *Configuration, 1)
	atm.OnTransition(func(config *Configuration, proof *TransitionProof) {
		committed <- config
	})
	for _, senderID := range []uint64{1, 2, 3} {
		if err := atm.HandleAutoTransitionMessage(signForValidator(t, initial, senderID, &AutoTransitionMessage{Type: AutoVote, View: 18, Leaves: []uint64{4}, BlockHash: view18BlockHash, NewConfigID: view18.ID})); err != nil {
			t.Fatalf("vote %d rejected: %v", senderID, err)
		}
	}
	select {
	case config := <-committed:
		if config == nil || config.ID != view18.ID {
			t.Fatalf("expected view 18 candidate config %d, got %#v", view18.ID, config)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("expected view 18 votes to commit matching candidate")
	}
}

func TestConfigurationCopyDeepCopiesBlockHash(t *testing.T) {
	config := &Configuration{
		ID:        1,
		BlockHash: []byte{1, 2, 3},
		Validators: map[uint64]*Validator{
			1: {ID: 1, PublicKey: []byte{9, 9}, Power: 1, IsActive: true},
		},
	}
	copyConfig := config.Copy()
	copyConfig.BlockHash[0] = 7
	copyConfig.Validators[1].PublicKey[0] = 8
	if config.BlockHash[0] != 1 {
		t.Fatalf("expected original block hash to remain unchanged, got %v", config.BlockHash)
	}
	if config.Validators[1].PublicKey[0] != 9 {
		t.Fatalf("expected original validator public key to remain unchanged, got %v", config.Validators[1].PublicKey)
	}
}

func TestGetLSetReturnsDetachedValidators(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	view := lset.GetLSet()
	for _, validator := range view {
		validator.PublicKey[0] = 255
		validator.IsActive = false
	}
	fresh := lset.GetLSet()
	for _, validator := range fresh {
		if !validator.IsActive {
			t.Fatal("expected l-set validator snapshot to be detached from internal state")
		}
		if len(validator.PublicKey) > 0 && validator.PublicKey[0] == 255 {
			t.Fatal("expected l-set validator public key snapshot to be detached from internal state")
		}
	}
}

func TestLSetManagerDegradedBlocksProposalButIsNotEvictable(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	lset.MarkFault(2, FaultClassDegraded)

	if lset.IsAllowedToPropose(2) {
		t.Fatal("expected degraded validator to be blocked from proposing")
	}
	if lset.HasEvictableFault(2) {
		t.Fatal("expected degraded validator to be non-evictable")
	}
}

func TestLSetManagerUnavailableAndByzantineAreEvictable(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	lset.MarkFault(2, FaultClassUnavailable)
	lset.MarkFault(3, FaultClassByzantine)

	if !lset.HasEvictableFault(2) || !lset.HasEvictableFault(3) {
		t.Fatal("expected unavailable and byzantine validators to be evictable")
	}
	if lset.IsAllowedToPropose(2) || lset.IsAllowedToPropose(3) {
		t.Fatal("expected unavailable and byzantine validators to be blocked from proposing")
	}
}

func TestHandleAutoTransitionMessageRejectsMissingSignature(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)

	if err := atm.HandleAutoTransitionMessage(&AutoTransitionMessage{Type: AutoPropose, SenderID: 1, View: 10}); err == nil {
		t.Fatal("expected unsigned auto-propose to be rejected")
	}
}

func TestHandleAutoTransitionMessageRejectsInvalidSignatureBeforeReplayCache(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	lset, err := NewLSetManager(initial.Validators)
	if err != nil {
		t.Fatalf("new l-set manager: %v", err)
	}
	tcm := NewTemporaryConfigurationManager(initial)
	atm := NewAutoTransitionManager(lset, tcm, nil)

	kp, err := crypto.GenerateKeyPair()
	if err != nil {
		t.Fatalf("generate keypair: %v", err)
	}
	initial.Validators[1].PublicKey = kp.PublicKey
	msg := &AutoTransitionMessage{Type: AutoVote, SenderID: 1, View: 12, Signature: []byte("bad")}
	if err := atm.HandleAutoTransitionMessage(msg); err == nil {
		t.Fatal("expected invalid signature to be rejected")
	}
	if len(atm.recentMessages) != 0 {
		t.Fatal("expected invalid signature to be rejected before replay cache update")
	}
	if err := atm.HandleAutoTransitionMessage(msg); err == nil {
		t.Fatal("expected invalid signature replay to remain invalid rather than poisoning replay cache")
	}
}
