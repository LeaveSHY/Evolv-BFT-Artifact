package main

import (
	"bytes"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
	"unsafe"

	"github.com/libp2p/go-libp2p/core/peer"
	"go.dedis.ch/kyber/v3/suites"

	"evolvbft/evolvbft/adaptive"
	"evolvbft/evolvbft/bootstrap"
	"evolvbft/evolvbft/consensus/gbc"
	"evolvbft/evolvbft/consensus/hotstuff"
	"evolvbft/evolvbft/consensus/mempool"
	"evolvbft/evolvbft/crypto"
	"evolvbft/evolvbft/hydra"
	"evolvbft/evolvbft/membership"
	"evolvbft/evolvbft/storage"
	"evolvbft/evolvbft/types"
)

var testManifestVRFSuite = suites.MustFind("Ed25519")

func TestGBCReadViewReturnsLatestCheckpointFromStore(t *testing.T) {
	log := gbc.NewLog()
	block := types.NewBlock(7, nil, []byte("read-path"), 6, 8, 0, 0, nil, nil)
	out := hotstuff.InstanceOutput{
		InstanceID:  5,
		LocalHeight: block.Height,
		Rank:        21,
		BlockHash:   append([]byte(nil), block.Hash...),
		Block:       block,
	}
	if err := publishOrderedCheckpoint(log, nil, true, out); err != nil {
		t.Fatalf("publish ordered checkpoint: %v", err)
	}

	view := newGBCReadView(log)
	checkpoint, ok, err := view.LatestCheckpoint()
	if err != nil {
		t.Fatalf("read latest checkpoint: %v", err)
	}
	if !ok {
		t.Fatal("expected latest checkpoint through read view")
	}
	if checkpoint.InstanceID != out.InstanceID || checkpoint.LocalHeight != out.LocalHeight || checkpoint.Rank != out.Rank || checkpoint.Epoch != block.Epoch {
		t.Fatalf("unexpected checkpoint from read view: %+v", checkpoint)
	}
}

func TestGBCReadViewReturnsNotFoundWithoutStore(t *testing.T) {
	view := newGBCReadView(nil)
	if _, ok, err := view.LatestCheckpoint(); err != nil {
		t.Fatalf("read latest checkpoint: %v", err)
	} else if ok {
		t.Fatal("expected missing store to return not found")
	}
}

func TestPublishOrderedCheckpoint_AppendsCommittedCheckpointToGBC(t *testing.T) {
	log := gbc.NewLog()
	block := types.NewBlock(3, nil, []byte("checkpoint"), 2, 4, 0, 0, nil, nil)
	out := hotstuff.InstanceOutput{
		InstanceID:  1,
		LocalHeight: block.Height,
		Rank:        17,
		BlockHash:   append([]byte(nil), block.Hash...),
		Block:       block,
	}

	if err := publishOrderedCheckpoint(log, nil, true, out); err != nil {
		t.Fatalf("publish ordered checkpoint: %v", err)
	}
	entry, ok := log.Retrieve(1)
	if !ok {
		t.Fatal("expected gbc checkpoint entry")
	}
	if entry.Type != gbc.EntryCheckpoint {
		t.Fatalf("unexpected gbc entry type: %s", entry.Type)
	}
	var payload map[string]any
	if err := json.Unmarshal(entry.Payload, &payload); err != nil {
		t.Fatalf("unmarshal gbc checkpoint payload: %v", err)
	}
	if got := payload["instance_id"]; got != float64(1) {
		t.Fatalf("unexpected instance_id: %v", got)
	}
	if got := payload["local_height"]; got != float64(block.Height) {
		t.Fatalf("unexpected local_height: %v", got)
	}
	if got := payload["rank"]; got != float64(17) {
		t.Fatalf("unexpected rank: %v", got)
	}
	if got := payload["epoch"]; got != float64(4) {
		t.Fatalf("unexpected epoch: %v", got)
	}
	if got := payload["is_nil"]; got != false {
		t.Fatalf("unexpected is_nil: %v", got)
	}
	if got := payload["transition_count"]; got != float64(0) {
		t.Fatalf("unexpected transition_count: %v", got)
	}
	if got := payload["block_hash_hex"]; got != hex.EncodeToString(block.Hash) {
		t.Fatalf("unexpected block_hash_hex: %v", got)
	}
}

func TestPublishOrderedCheckpoint_SkipsNilAndMissingBlocks(t *testing.T) {
	cases := []hotstuff.InstanceOutput{
		{IsNil: true, Rank: 9},
		{Block: nil, Rank: 10},
	}
	for _, out := range cases {
		log := gbc.NewLog()
		if err := publishOrderedCheckpoint(log, nil, true, out); err != nil {
			t.Fatalf("publish ordered checkpoint: %v", err)
		}
		if _, ok := log.Retrieve(1); ok {
			t.Fatalf("expected no gbc entry for output %+v", out)
		}
	}
}

func TestPublishOrderedCheckpoint_DeduplicatesReplayedOutput(t *testing.T) {
	log := gbc.NewLog()
	block := types.NewBlock(4, nil, []byte("replay"), 3, 5, 0, 0, nil, nil)
	out := hotstuff.InstanceOutput{
		InstanceID:  2,
		LocalHeight: block.Height,
		Rank:        18,
		BlockHash:   append([]byte(nil), block.Hash...),
		Block:       block,
	}

	if err := publishOrderedCheckpoint(log, nil, true, out); err != nil {
		t.Fatalf("first publish ordered checkpoint: %v", err)
	}
	if err := publishOrderedCheckpoint(log, nil, true, out); err != nil {
		t.Fatalf("replayed publish ordered checkpoint: %v", err)
	}
	if _, ok := log.Retrieve(2); ok {
		t.Fatal("expected replayed ordered output to avoid duplicate gbc entry")
	}
	entry, ok := log.Retrieve(1)
	if !ok {
		t.Fatal("expected original gbc entry to remain")
	}
	if entry.Type != gbc.EntryCheckpoint {
		t.Fatalf("unexpected gbc entry type: %s", entry.Type)
	}
}

func TestPublishOrderedCheckpoint_WritesLatestCheckpointForInjectedStore(t *testing.T) {
	log := gbc.NewLog()
	block := types.NewBlock(5, nil, []byte("wired"), 4, 6, 0, 0, nil, nil)
	out := hotstuff.InstanceOutput{
		InstanceID:  3,
		LocalHeight: block.Height,
		Rank:        19,
		BlockHash:   append([]byte(nil), block.Hash...),
		Block:       block,
	}

	if err := publishOrderedCheckpoint(log, nil, true, out); err != nil {
		t.Fatalf("publish ordered checkpoint: %v", err)
	}
	checkpoint, ok, err := gbc.GetLatestCheckpoint(log)
	if err != nil {
		t.Fatalf("get latest checkpoint: %v", err)
	}
	if !ok {
		t.Fatal("expected latest checkpoint in injected gbc store")
	}
	if checkpoint.InstanceID != out.InstanceID || checkpoint.LocalHeight != out.LocalHeight || checkpoint.Rank != out.Rank {
		t.Fatalf("unexpected latest checkpoint: %+v", checkpoint)
	}
	if checkpoint.Epoch != block.Epoch {
		t.Fatalf("unexpected checkpoint epoch: %d", checkpoint.Epoch)
	}
}

func TestPublishOrderedCheckpoint_AcceptsNilStoreForNoOpPath(t *testing.T) {
	block := types.NewBlock(6, nil, []byte("nil-store"), 5, 7, 0, 0, nil, nil)
	out := hotstuff.InstanceOutput{
		InstanceID:  4,
		LocalHeight: block.Height,
		Rank:        20,
		BlockHash:   append([]byte(nil), block.Hash...),
		Block:       block,
	}

	if err := publishOrderedCheckpoint(nil, nil, true, out); err != nil {
		t.Fatalf("publish ordered checkpoint with nil store: %v", err)
	}
}

func TestPublishOrderedCheckpoint_SkipsLocalPublishWithoutFallback(t *testing.T) {
	log := gbc.NewLog()
	block := types.NewBlock(1, nil, []byte("no-local-fallback"), 1, 1, 0, 0, nil, nil)
	out := hotstuff.InstanceOutput{
		InstanceID:  1,
		LocalHeight: block.Height,
		Rank:        11,
		BlockHash:   append([]byte(nil), block.Hash...),
		Block:       block,
	}

	if err := publishOrderedCheckpoint(log, nil, false, out); err != nil {
		t.Fatalf("publish ordered checkpoint without fallback: %v", err)
	}
	if _, ok := log.Retrieve(1); ok {
		t.Fatal("non-GBC-primary path must not publish uncertified local checkpoint")
	}
}

func TestApplyOrderedOutput_ActivationPoint(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	initial := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
	}
	mm := membership.NewMembershipManager(initial)
	exec := hotstuff.NewExecutor(mm.GetCurrentConfig().ToValidatorSet())
	exec.SetReconfigAuthorizer(func(data *types.ReconfigData) bool { return data != nil && data.Type == types.ReconfigJoin && data.NodeID == 1 })
	exec.SetEpochChangeCallback(func(newValSet *types.ValidatorSet, transitions []*types.EpochTransition) error {
		if _, _, _, err := mm.InstallValidatorSetFromTransitions(newValSet, transitions); err != nil {
			t.Fatalf("install validator set: %v", err)
		}
		return nil
	})
	baseEpoch := mm.GetCurrentConfig().ID

	normalTx := &types.Transaction{Type: types.TxTypeNormal, Payload: []byte("x")}
	rawNormal, _ := json.Marshal(normalTx)
	normalBlock := types.NewBlock(1, nil, rawNormal, 1, baseEpoch, 0, 0, nil, nil)
	normalOut, err := applyOrderedOutput(hotstuff.InstanceOutput{
		InstanceID:  0,
		LocalHeight: normalBlock.Height,
		Rank:        7,
		BlockHash:   normalBlock.Hash,
		Block:       normalBlock,
	}, exec)
	if err != nil {
		t.Fatalf("apply committed normal block: %v", err)
	}
	if mm.GetCurrentConfig().ID != baseEpoch {
		t.Fatalf("epoch changed on non-reconfig block")
	}
	if len(normalOut.EpochTransitions) != 0 {
		t.Fatalf("expected no epoch transitions for normal block, got %d", len(normalOut.EpochTransitions))
	}

	kp1, _ := crypto.GenerateKeyPair()
	reconfig := types.ReconfigData{
		Type:      types.ReconfigJoin,
		NodeID:    1,
		PublicKey: kp1.PublicKey,
		Power:     1,
		Epoch:     baseEpoch,
	}
	reconfig.Sign(kp1.PrivateKey)
	reconfigPayload, _ := json.Marshal(reconfig)
	tx := &types.Transaction{Type: types.TxTypeReconfig, Payload: reconfigPayload}
	rawTx, _ := json.Marshal(tx)
	reconfigBlock := types.NewBlock(2, nil, rawTx, 2, baseEpoch, 0, 0, nil, nil)
	orderedOut, err := applyOrderedOutput(hotstuff.InstanceOutput{
		InstanceID:  0,
		LocalHeight: reconfigBlock.Height,
		Rank:        8,
		BlockHash:   reconfigBlock.Hash,
		Block:       reconfigBlock,
	}, exec)
	if err != nil {
		t.Fatalf("apply ordered reconfig block: %v", err)
	}
	if mm.GetCurrentConfig().ID == baseEpoch {
		t.Fatalf("epoch did not advance on reconfig block")
	}
	if len(orderedOut.EpochTransitions) != 1 {
		t.Fatalf("expected 1 epoch transition, got %d", len(orderedOut.EpochTransitions))
	}
	transition := orderedOut.EpochTransitions[0]
	if transition.OldEpoch != baseEpoch || transition.NewEpoch != baseEpoch+1 {
		t.Fatalf("unexpected epoch transition: %+v", transition)
	}
	if transition.ActivationHeight != reconfigBlock.Height {
		t.Fatalf("unexpected activation height: %d", transition.ActivationHeight)
	}
	if transition.ActivationRank != orderedOut.Rank {
		t.Fatalf("unexpected activation rank: %d", transition.ActivationRank)
	}
	if len(transition.Added) != 1 || transition.Added[0] != 1 {
		t.Fatalf("unexpected added set: %+v", transition.Added)
	}
}

func TestApplyOrderedOutput_DoesNotEmitQueuedFutureTransition(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	initial := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
	}
	mm := membership.NewMembershipManager(initial)
	exec := hotstuff.NewExecutor(mm.GetCurrentConfig().ToValidatorSet())
	exec.SetReconfigAuthorizer(func(data *types.ReconfigData) bool {
		return data != nil && data.Type == types.ReconfigJoin && (data.NodeID == 1 || data.NodeID == 2)
	})
	exec.SetEpochChangeCallback(func(newValSet *types.ValidatorSet, transitions []*types.EpochTransition) error {
		if _, _, _, err := mm.InstallValidatorSetFromTransitions(newValSet, transitions); err != nil {
			t.Fatalf("install validator set: %v", err)
		}
		return nil
	})

	kp1, _ := crypto.GenerateKeyPair()
	join1 := types.ReconfigData{Type: types.ReconfigJoin, NodeID: 1, PublicKey: kp1.PublicKey, Power: 1, Epoch: 1, TargetEpoch: 2}
	join1.Sign(kp1.PrivateKey)
	join1Payload, _ := json.Marshal(join1)
	join1Tx, _ := json.Marshal(&types.Transaction{Type: types.TxTypeReconfig, Payload: join1Payload})
	blockA := types.NewBlock(1, nil, join1Tx, 1, 1, 0, 0, nil, nil)

	kp2, _ := crypto.GenerateKeyPair()
	join2 := types.ReconfigData{Type: types.ReconfigJoin, NodeID: 2, PublicKey: kp2.PublicKey, Power: 1, Epoch: 2, TargetEpoch: 3}
	join2.Sign(kp2.PrivateKey)
	join2Payload, _ := json.Marshal(join2)
	join2Tx, _ := json.Marshal(&types.Transaction{Type: types.TxTypeReconfig, Payload: join2Payload})
	blockB := types.NewBlock(2, nil, join2Tx, 2, 2, 0, 0, nil, nil)

	outA, err := applyOrderedOutput(hotstuff.InstanceOutput{InstanceID: 0, LocalHeight: blockA.Height, Rank: 10, BlockHash: blockA.Hash, Block: blockA}, exec)
	if err != nil {
		t.Fatalf("apply ordered output A: %v", err)
	}
	if len(outA.EpochTransitions) != 1 {
		t.Fatalf("expected exactly one transition after first ordered output, got %d", len(outA.EpochTransitions))
	}
	if outA.EpochTransitions[0].NewEpoch != 2 {
		t.Fatalf("unexpected first transition epoch: %d", outA.EpochTransitions[0].NewEpoch)
	}
	if mm.GetCurrentConfig().ID != 2 {
		t.Fatalf("membership advanced too far after first ordered output: epoch=%d", mm.GetCurrentConfig().ID)
	}

	outB, err := applyOrderedOutput(hotstuff.InstanceOutput{InstanceID: 1, LocalHeight: blockB.Height, Rank: 11, BlockHash: blockB.Hash, Block: blockB}, exec)
	if err != nil {
		t.Fatalf("apply ordered output B: %v", err)
	}
	if len(outB.EpochTransitions) != 1 {
		t.Fatalf("expected one transition after second ordered output, got %d", len(outB.EpochTransitions))
	}
	if outB.EpochTransitions[0].NewEpoch != 3 {
		t.Fatalf("unexpected second transition epoch: %d", outB.EpochTransitions[0].NewEpoch)
	}
	if mm.GetCurrentConfig().ID != 3 {
		t.Fatalf("membership did not advance after second ordered output: epoch=%d", mm.GetCurrentConfig().ID)
	}
}

func TestApplyOrderedOutput_RetriesWithoutReexecutionAfterCallbackFailure(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	initial := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
	}
	mm := membership.NewMembershipManager(initial)
	exec := hotstuff.NewExecutor(mm.GetCurrentConfig().ToValidatorSet())
	exec.SetReconfigAuthorizer(func(data *types.ReconfigData) bool { return data != nil && data.Type == types.ReconfigJoin && data.NodeID == 1 })
	attempts := 0
	exec.SetEpochChangeCallback(func(newValSet *types.ValidatorSet, transitions []*types.EpochTransition) error {
		attempts++
		if attempts == 1 {
			return errors.New("transient install failure")
		}
		if _, _, _, err := mm.InstallValidatorSetFromTransitions(newValSet, transitions); err != nil {
			t.Fatalf("install validator set: %v", err)
		}
		return nil
	})

	kp1, _ := crypto.GenerateKeyPair()
	reconfig := types.ReconfigData{Type: types.ReconfigJoin, NodeID: 1, PublicKey: kp1.PublicKey, Power: 1, Epoch: 1, TargetEpoch: 2}
	reconfig.Sign(kp1.PrivateKey)
	reconfigPayload, _ := json.Marshal(reconfig)
	txPayload, _ := json.Marshal(&types.Transaction{Type: types.TxTypeReconfig, Payload: reconfigPayload})
	block := types.NewBlock(1, nil, txPayload, 1, 1, 0, 0, nil, nil)
	ordered := hotstuff.InstanceOutput{InstanceID: 0, LocalHeight: block.Height, Rank: 12, BlockHash: block.Hash, Block: block}

	first, err := applyOrderedOutput(ordered, exec)
	if err == nil {
		t.Fatalf("expected first apply to fail")
	}
	if len(first.EpochTransitions) != 0 {
		t.Fatalf("expected no published transitions on failed apply, got %d", len(first.EpochTransitions))
	}
	if mm.GetCurrentConfig().ID != 1 {
		t.Fatalf("membership changed despite failed apply: %d", mm.GetCurrentConfig().ID)
	}
	if pending := exec.PendingReconfigCount(); pending != 1 {
		t.Fatalf("expected one pending reconfig after failed apply, got %d", pending)
	}
	if blocksExecuted, _ := exec.Stats(); blocksExecuted != 1 {
		t.Fatalf("expected ordered block to execute once before retry, got %d", blocksExecuted)
	}

	second, err := applyOrderedOutput(ordered, exec)
	if err != nil {
		t.Fatalf("retry apply failed: %v", err)
	}
	if len(second.EpochTransitions) != 1 {
		t.Fatalf("expected one epoch transition on retry, got %d", len(second.EpochTransitions))
	}
	if mm.GetCurrentConfig().ID != 2 {
		t.Fatalf("membership did not advance after retry: %d", mm.GetCurrentConfig().ID)
	}
	if pending := exec.PendingReconfigCount(); pending != 0 {
		t.Fatalf("expected pending reconfigs cleared after retry, got %d", pending)
	}
	if blocksExecuted, _ := exec.Stats(); blocksExecuted != 1 {
		t.Fatalf("expected retry to avoid re-executing block, got %d executions", blocksExecuted)
	}
	if attempts != 2 {
		t.Fatalf("expected callback to be retried exactly once, got %d attempts", attempts)
	}
}

func TestApplyOrderedOutput_IsIdempotentForReplay(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	initial := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
	}
	mm := membership.NewMembershipManager(initial)
	exec := hotstuff.NewExecutor(mm.GetCurrentConfig().ToValidatorSet())
	exec.SetReconfigAuthorizer(func(data *types.ReconfigData) bool { return data != nil && data.Type == types.ReconfigJoin && data.NodeID == 1 })
	exec.SetEpochChangeCallback(func(newValSet *types.ValidatorSet, transitions []*types.EpochTransition) error {
		if _, _, _, err := mm.InstallValidatorSetFromTransitions(newValSet, transitions); err != nil {
			t.Fatalf("install validator set: %v", err)
		}
		return nil
	})

	kp1, _ := crypto.GenerateKeyPair()
	reconfig := types.ReconfigData{Type: types.ReconfigJoin, NodeID: 1, PublicKey: kp1.PublicKey, Power: 1, Epoch: 1, TargetEpoch: 2}
	reconfig.Sign(kp1.PrivateKey)
	reconfigPayload, _ := json.Marshal(reconfig)
	txPayload, _ := json.Marshal(&types.Transaction{Type: types.TxTypeReconfig, Payload: reconfigPayload})
	block := types.NewBlock(1, nil, txPayload, 1, 1, 0, 0, nil, nil)
	ordered := hotstuff.InstanceOutput{InstanceID: 0, LocalHeight: block.Height, Rank: 12, BlockHash: block.Hash, Block: block}

	first, err := applyOrderedOutput(ordered, exec)
	if err != nil {
		t.Fatalf("first apply failed: %v", err)
	}
	if len(first.EpochTransitions) != 1 {
		t.Fatalf("expected one epoch transition on first apply, got %d", len(first.EpochTransitions))
	}
	if mm.GetCurrentConfig().ID != 2 {
		t.Fatalf("unexpected epoch after first apply: %d", mm.GetCurrentConfig().ID)
	}

	second, err := applyOrderedOutput(ordered, exec)
	if err != nil {
		t.Fatalf("second apply failed: %v", err)
	}
	if len(second.EpochTransitions) != 0 {
		t.Fatalf("expected replay to be idempotent, got %d extra transitions", len(second.EpochTransitions))
	}
	if mm.GetCurrentConfig().ID != 2 {
		t.Fatalf("epoch changed after replay: %d", mm.GetCurrentConfig().ID)
	}
}

func TestInstallCommittedValidatorSetUsesExecutorTransitions(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	initial := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
	}
	mm := membership.NewMembershipManager(initial)
	newValSet := types.NewValidatorSet(2, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
	})
	transitions := []*types.EpochTransition{{
		OldEpoch:         1,
		NewEpoch:         2,
		ActivationHeight: 4,
		ActivationRank:   9,
		Added:            []uint64{1},
		QuorumSize:       newValSet.QuorumSize,
		ConfigHash:       newValSet.Hash(),
	}}

	if err := installCommittedValidatorSet(newValSet, transitions, mm, nil, nil, "", nil); err != nil {
		t.Fatalf("install committed validator set failed: %v", err)
	}
	if cfg := mm.GetCurrentConfig(); cfg == nil || cfg.ID != 2 {
		t.Fatalf("expected membership to advance to epoch 2, got %+v", cfg)
	}
	if err := installCommittedValidatorSet(newValSet, transitions, mm, nil, nil, "", nil); err != nil {
		t.Fatalf("replay committed validator set failed: %v", err)
	}
	cfg := mm.GetCurrentConfig()
	if cfg.ID != 2 {
		t.Fatalf("unexpected installed epoch: %d", cfg.ID)
	}
	event := mm.GetLatestEvent()
	if event == nil {
		t.Fatalf("expected latest event")
	}
	if event.OldEpoch != 1 || event.NewEpoch != 2 {
		t.Fatalf("unexpected event epochs: %+v", event)
	}
	if len(event.Added) != 1 || event.Added[0] != 1 {
		t.Fatalf("unexpected event added set: %+v", event.Added)
	}
}

func TestInstallCommittedValidatorSetRejectsInvalidVRFBytesBeforeApplying(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	initial := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
	}
	mm := membership.NewMembershipManager(initial)
	before := mm.GetCurrentConfig()
	newValSet := types.NewValidatorSet(2, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, VRFPublicKey: []byte("not-a-valid-kyber-point"), Power: 1, IsActive: true},
	})
	transitions := []*types.EpochTransition{{
		OldEpoch:         1,
		NewEpoch:         2,
		ActivationHeight: 4,
		ActivationRank:   9,
		Added:            []uint64{1},
		QuorumSize:       newValSet.QuorumSize,
		ConfigHash:       newValSet.Hash(),
	}}

	if err := installCommittedValidatorSet(newValSet, transitions, mm, nil, nil, "", nil); err == nil {
		t.Fatalf("expected install to fail for invalid vrf public key bytes")
	}
	after := mm.GetCurrentConfig()
	if after.ID != before.ID {
		t.Fatalf("membership should remain at epoch %d, got %d", before.ID, after.ID)
	}
	if _, ok := after.Validators[1]; ok {
		t.Fatalf("validator 1 should not be installed after invalid vrf public key")
	}
}

func TestWireHydraAutoLeaveInjectionInstallsCallbackBeforeStartWindow(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	validators := map[uint64]*hydra.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
	}
	hydraMgr, err := hydra.NewHydraManager(0, validators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	mm := membership.NewMembershipManager(map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
	})
	engine := buildHydraInjectionEngineForTest(t, kp0.PublicKey)

	before := hydraTransitionCallbackPointer(hydraMgr)
	if before != 0 {
		t.Fatalf("expected no hydra transition callback before wiring")
	}
	wireHydraAutoLeaveInjection(hydraMgr, mm, []*hotstuff.Engine{engine})
	after := hydraTransitionCallbackPointer(hydraMgr)
	if after == 0 {
		t.Fatalf("expected hydra transition callback after wiring")
	}
}

func TestInjectHydraAutoLeavesUsesSingleExplicitEngine(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	mm := membership.NewMembershipManager(map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
	})
	engineA := buildHydraInjectionEngineForTest(t, kp0.PublicKey)
	engineB := buildHydraInjectionEngineForTest(t, kp0.PublicKey)
	proof := &hydra.TransitionProof{
		View:        7,
		NewConfigID: 9,
		Leaves:      []uint64{1},
		BlockHash:   []byte("hydra-proof-block"),
		AutoVotes: map[uint64]*hydra.Vote{
			0: {SenderID: 0, Signature: []byte("sig0"), Digest: []byte("digest")},
			1: {SenderID: 1, Signature: []byte("sig1"), Digest: []byte("digest")},
		},
	}
	config := &hydra.Configuration{
		ID: 9,
		Validators: map[uint64]*hydra.Validator{
			0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		},
		QuorumSize: 1,
	}

	injectHydraAutoLeaves(config, proof, mm, []*hotstuff.Engine{engineA, engineB})

	txA := drainSingleReconfigTx(t, hydraInjectionMempool(engineA))
	if txA == nil {
		t.Fatalf("expected first engine to receive injected reconfig tx")
	}
	if txB := drainSingleReconfigTx(t, hydraInjectionMempool(engineB)); txB != nil {
		t.Fatalf("expected only one explicit engine to receive injected reconfig tx")
	}
	var data types.ReconfigData
	if err := json.Unmarshal(txA.Payload, &data); err != nil {
		t.Fatalf("unmarshal injected reconfig payload: %v", err)
	}
	if data.Type != types.ReconfigAutoLeave || data.NodeID != 1 {
		t.Fatalf("unexpected injected reconfig data: %+v", data)
	}
	if data.Epoch != mm.GetCurrentConfig().ID || data.TargetEpoch != mm.GetCurrentConfig().ID+1 {
		t.Fatalf("unexpected injected epochs: epoch=%d target=%d", data.Epoch, data.TargetEpoch)
	}
	if data.AutoLeaveProof == nil {
		t.Fatalf("expected injected auto-leave tx to carry Hydra quorum proof")
	}
	if data.AutoLeaveProof.NewConfigID != proof.NewConfigID {
		t.Fatalf("unexpected injected proof config id: %d", data.AutoLeaveProof.NewConfigID)
	}
	if !reflect.DeepEqual(data.AutoLeaveProof.Leaves, proof.Leaves) {
		t.Fatalf("unexpected injected proof leaves: %v", data.AutoLeaveProof.Leaves)
	}
	if string(data.AutoLeaveProof.BlockHash) != string(proof.BlockHash) {
		t.Fatalf("unexpected injected proof block hash: %x", data.AutoLeaveProof.BlockHash)
	}
	if len(data.AutoLeaveProof.AutoVotes) != len(proof.AutoVotes) {
		t.Fatalf("unexpected injected proof vote count: %d", len(data.AutoLeaveProof.AutoVotes))
	}
}

func TestInstallCommittedValidatorSetUpdatesHydraBeforeEngineLeaderRefresh(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	kp2, _ := crypto.GenerateKeyPair()
	initial := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: kp2.PublicKey, Power: 1, IsActive: true},
	}
	mm := membership.NewMembershipManager(initial)
	hydraValidators := map[uint64]*hydra.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: kp2.PublicKey, Power: 1, IsActive: true},
	}
	hydraMgr, err := hydra.NewHydraManager(0, hydraValidators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	engine := buildHydraInjectionEngineForTest(t, kp0.PublicKey)
	engine.SetHydraManager(hydraMgr)
	engine.GetLeaderReputation().RecordTimeout(0)
	hydraMgr.LSetManager.InstallConfiguration(&hydra.Configuration{
		ID: 1,
		Validators: map[uint64]*hydra.Validator{
			0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
			1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
			2: {ID: 2, PublicKey: kp2.PublicKey, Power: 1, IsActive: true},
		},
		QuorumSize: 3,
	})
	engine.UpdateValidatorSet(types.NewValidatorSet(1, initial))

	newValSet := types.NewValidatorSet(2, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
	})
	transitions := []*types.EpochTransition{{
		OldEpoch:         1,
		NewEpoch:         2,
		ActivationHeight: 5,
		ActivationRank:   11,
		Removed:          []uint64{2},
		QuorumSize:       newValSet.QuorumSize,
		ConfigHash:       newValSet.Hash(),
	}}

	if err := installCommittedValidatorSet(newValSet, transitions, mm, []*hotstuff.Engine{engine}, nil, "", hydraMgr); err != nil {
		t.Fatalf("install committed validator set: %v", err)
	}
	if !reflect.DeepEqual(hydraMgr.AllowedLeaders(), []uint64{0, 1}) {
		t.Fatalf("unexpected hydra allowed leaders after committed install: %v", hydraMgr.AllowedLeaders())
	}
	leader := engine.GetCurrentValidatorSet()
	if leader == nil || leader.Epoch != 2 {
		t.Fatalf("expected engine validator set epoch 2, got %+v", leader)
	}
	if got := hydraMgr.IsAllowedLeader(2); got {
		t.Fatalf("expected removed validator 2 to be disallowed after committed install")
	}
}

func TestInstallCommittedValidatorSetIgnoresStaleLowerEpochBeforeHydraMutation(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	kp2, _ := crypto.GenerateKeyPair()
	initial := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
	}
	mm := membership.NewMembershipManager(initial)
	hydraMgr, err := hydra.NewHydraManager(0, map[uint64]*hydra.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
	}, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}

	newerValSet := types.NewValidatorSet(2, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: kp2.PublicKey, Power: 1, IsActive: true},
	})
	newerTransitions := []*types.EpochTransition{{
		OldEpoch:         1,
		NewEpoch:         2,
		ActivationHeight: 5,
		ActivationRank:   11,
		Added:            []uint64{2},
		QuorumSize:       newerValSet.QuorumSize,
		ConfigHash:       newerValSet.Hash(),
	}}
	if err := installCommittedValidatorSet(newerValSet, newerTransitions, mm, nil, nil, "", hydraMgr); err != nil {
		t.Fatalf("install newer validator set: %v", err)
	}
	beforeHydraHistory := len(hydraMgr.DiscoveryManager.GetHistory())

	staleValSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
	})
	staleTransitions := []*types.EpochTransition{{
		OldEpoch:         0,
		NewEpoch:         1,
		ActivationHeight: 4,
		ActivationRank:   9,
		Added:            []uint64{1},
		QuorumSize:       staleValSet.QuorumSize,
		ConfigHash:       staleValSet.Hash(),
	}}
	if err := installCommittedValidatorSet(staleValSet, staleTransitions, mm, nil, nil, "", hydraMgr); err != nil {
		t.Fatalf("stale committed validator set should be ignored, got error: %v", err)
	}
	cfg := mm.GetCurrentConfig()
	if cfg == nil || cfg.ID != 2 {
		t.Fatalf("expected membership to remain at epoch 2, got %+v", cfg)
	}
	if _, ok := cfg.Validators[2]; !ok {
		t.Fatalf("expected newer validator to remain installed")
	}
	if _, ok := cfg.Validators[1]; ok {
		t.Fatalf("did not expect stale validator to be installed")
	}
	hydraCurrent := hydraMgr.GetCurrentConfiguration()
	if hydraCurrent == nil || hydraCurrent.ID != 2 {
		t.Fatalf("expected hydra committed config to remain at id 2, got %+v", hydraCurrent)
	}
	if got := len(hydraMgr.DiscoveryManager.GetHistory()); got != beforeHydraHistory {
		t.Fatalf("expected stale committed install to avoid hydra history append, got %d entries", got)
	}
}

func TestInstallCommittedValidatorSetRejectsConflictingSameEpochBeforeHydraMutation(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	kp2, _ := crypto.GenerateKeyPair()
	initial := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
	}
	mm := membership.NewMembershipManager(initial)
	hydraMgr, err := hydra.NewHydraManager(0, map[uint64]*hydra.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
	}, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	installed := types.NewValidatorSet(2, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
	})
	installedTransitions := []*types.EpochTransition{{
		OldEpoch:         1,
		NewEpoch:         2,
		ActivationHeight: 5,
		ActivationRank:   11,
		Added:            []uint64{1},
		QuorumSize:       installed.QuorumSize,
		ConfigHash:       installed.Hash(),
	}}
	if err := installCommittedValidatorSet(installed, installedTransitions, mm, nil, nil, "", hydraMgr); err != nil {
		t.Fatalf("install initial epoch-2 validator set: %v", err)
	}
	beforeHydraHistory := len(hydraMgr.DiscoveryManager.GetHistory())

	conflicting := types.NewValidatorSet(2, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: kp2.PublicKey, Power: 1, IsActive: true},
	})
	conflictingTransitions := []*types.EpochTransition{{
		OldEpoch:         1,
		NewEpoch:         2,
		ActivationHeight: 6,
		ActivationRank:   12,
		Added:            []uint64{2},
		QuorumSize:       conflicting.QuorumSize,
		ConfigHash:       conflicting.Hash(),
	}}
	if err := installCommittedValidatorSet(conflicting, conflictingTransitions, mm, nil, nil, "", hydraMgr); err == nil {
		t.Fatalf("expected conflicting same-epoch validator set to fail")
	}
	cfg := mm.GetCurrentConfig()
	if cfg == nil || cfg.ID != 2 {
		t.Fatalf("expected membership to remain at epoch 2, got %+v", cfg)
	}
	if _, ok := cfg.Validators[1]; !ok {
		t.Fatalf("expected original epoch-2 validator to remain installed")
	}
	if _, ok := cfg.Validators[2]; ok {
		t.Fatalf("did not expect conflicting epoch-2 validator to replace installed config")
	}
	hydraCurrent := hydraMgr.GetCurrentConfiguration()
	if hydraCurrent == nil || hydraCurrent.ID != 2 {
		t.Fatalf("expected hydra committed config to remain at id 2, got %+v", hydraCurrent)
	}
	if _, ok := hydraCurrent.Validators[1]; !ok {
		t.Fatalf("expected original hydra epoch-2 validator to remain installed")
	}
	if _, ok := hydraCurrent.Validators[2]; ok {
		t.Fatalf("did not expect conflicting hydra epoch-2 validator to be installed")
	}
	if got := len(hydraMgr.DiscoveryManager.GetHistory()); got != beforeHydraHistory {
		t.Fatalf("expected conflicting same-epoch install to avoid hydra history append, got %d entries", got)
	}
}

func TestInstallCommittedValidatorSetIgnoresStaleLowerEpochWithoutClearingHydraPendingIntents(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	kp2, _ := crypto.GenerateKeyPair()
	joinerKey, _ := crypto.GenerateKeyPair()
	initial := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
	}
	mm := membership.NewMembershipManager(initial)
	engine := buildHydraInjectionEngineForTest(t, kp0.PublicKey)
	hydraMgr, err := hydra.NewHydraManager(0, map[uint64]*hydra.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
	}, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}

	newerValSet := types.NewValidatorSet(2, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: kp2.PublicKey, Power: 1, IsActive: true},
	})
	newerTransitions := []*types.EpochTransition{{
		OldEpoch:         1,
		NewEpoch:         2,
		ActivationHeight: 5,
		ActivationRank:   11,
		Added:            []uint64{2},
		QuorumSize:       newerValSet.QuorumSize,
		ConfigHash:       newerValSet.Hash(),
	}}
	if err := installCommittedValidatorSet(newerValSet, newerTransitions, mm, []*hotstuff.Engine{engine}, nil, "", hydraMgr); err != nil {
		t.Fatalf("install newer validator set: %v", err)
	}
	beforeHydraCurrent := hydraMgr.GetCurrentConfiguration()
	beforeHydraHighestKnown := hydraMgr.GetHighestKnownConfiguration()
	beforeHydraLSet := hydraMgr.LSetManager.GetLSet()
	beforeHydraHistory := hydraMgr.DiscoveryManager.GetHistory()
	if err := hydraMgr.SubmitJoinRequest(5, joinerKey.PublicKey, 1); err != nil {
		t.Fatalf("submit join request: %v", err)
	}
	if err := hydraMgr.SubmitLeaveRequest(0); err != nil {
		t.Fatalf("submit leave request: %v", err)
	}
	beforePendingJoins := hydraMgr.TempConfigManager.GetPendingJoins()
	beforePendingLeaves := hydraMgr.TempConfigManager.GetPendingLeaves()
	beforeMembership := mm.GetCurrentConfig()
	beforeMembershipHash := beforeMembership.Hash()
	beforeTempHistory := hydraMgr.TempConfigManager.GetHistory()
	beforeEngine := engine.GetCurrentValidatorSet().Copy()

	staleValSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
	})
	staleTransitions := []*types.EpochTransition{{
		OldEpoch:         0,
		NewEpoch:         1,
		ActivationHeight: 4,
		ActivationRank:   9,
		Added:            []uint64{1},
		QuorumSize:       staleValSet.QuorumSize,
		ConfigHash:       staleValSet.Hash(),
	}}
	if err := installCommittedValidatorSet(staleValSet, staleTransitions, mm, []*hotstuff.Engine{engine}, nil, "", hydraMgr); err != nil {
		t.Fatalf("stale committed validator set should be ignored, got error: %v", err)
	}
	if pending := hydraMgr.TempConfigManager.GetPendingJoins(); !reflect.DeepEqual(pending, beforePendingJoins) {
		t.Fatalf("expected stale wrapper path to preserve hydra pending joins, got %+v", pending)
	}
	if pending := hydraMgr.TempConfigManager.GetPendingLeaves(); !reflect.DeepEqual(pending, beforePendingLeaves) {
		t.Fatalf("expected stale wrapper path to preserve hydra pending leaves, got %+v", pending)
	}
	afterMembership := mm.GetCurrentConfig()
	if afterMembership == nil || beforeMembership == nil || afterMembership.ID != beforeMembership.ID {
		t.Fatalf("expected stale wrapper path to preserve membership epoch %d, got %+v", beforeMembership.ID, afterMembership)
	}
	if !reflect.DeepEqual(afterMembership.Validators, beforeMembership.Validators) {
		t.Fatalf("expected stale wrapper path to preserve membership validators")
	}
	if !bytes.Equal(afterMembership.Hash(), beforeMembershipHash) {
		t.Fatalf("expected stale wrapper path to preserve membership hash %x, got %x", beforeMembershipHash, afterMembership.Hash())
	}
	afterEngine := engine.GetCurrentValidatorSet()
	if afterEngine == nil || beforeEngine == nil || afterEngine.Epoch != beforeEngine.Epoch {
		t.Fatalf("expected stale wrapper path to preserve engine epoch %d, got %+v", beforeEngine.Epoch, afterEngine)
	}
	if !reflect.DeepEqual(afterEngine.Validators, beforeEngine.Validators) {
		t.Fatalf("expected stale wrapper path to preserve engine validators")
	}
	hydraCurrent := hydraMgr.GetCurrentConfiguration()
	if hydraCurrent == nil || beforeHydraCurrent == nil || hydraCurrent.ID != beforeHydraCurrent.ID {
		t.Fatalf("expected stale wrapper path to preserve hydra current config id %d, got %+v", beforeHydraCurrent.ID, hydraCurrent)
	}
	if !reflect.DeepEqual(hydraCurrent.Validators, beforeHydraCurrent.Validators) {
		t.Fatalf("expected stale wrapper path to preserve hydra current validators")
	}
	if highest := hydraMgr.GetHighestKnownConfiguration(); !reflect.DeepEqual(highest, beforeHydraHighestKnown) {
		t.Fatalf("expected stale wrapper path to preserve hydra highest-known config")
	}
	if lset := hydraMgr.LSetManager.GetLSet(); !reflect.DeepEqual(lset, beforeHydraLSet) {
		t.Fatalf("expected stale wrapper path to preserve hydra lset")
	}
	if history := hydraMgr.DiscoveryManager.GetHistory(); !reflect.DeepEqual(history, beforeHydraHistory) {
		t.Fatalf("expected stale wrapper path to preserve hydra history")
	}
	if history := hydraMgr.TempConfigManager.GetHistory(); !reflect.DeepEqual(history, beforeTempHistory) {
		t.Fatalf("expected stale wrapper path to preserve hydra temp history")
	}
}

func TestInstallCommittedValidatorSetRejectsConflictingSameEpochWithoutClearingHydraPendingIntents(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	kp2, _ := crypto.GenerateKeyPair()
	joinerKey, _ := crypto.GenerateKeyPair()
	initial := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
	}
	mm := membership.NewMembershipManager(initial)
	engine := buildHydraInjectionEngineForTest(t, kp0.PublicKey)
	hydraMgr, err := hydra.NewHydraManager(0, map[uint64]*hydra.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
	}, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}

	installed := types.NewValidatorSet(2, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
	})
	installedTransitions := []*types.EpochTransition{{
		OldEpoch:         1,
		NewEpoch:         2,
		ActivationHeight: 5,
		ActivationRank:   11,
		Added:            []uint64{1},
		QuorumSize:       installed.QuorumSize,
		ConfigHash:       installed.Hash(),
	}}
	if err := installCommittedValidatorSet(installed, installedTransitions, mm, []*hotstuff.Engine{engine}, nil, "", hydraMgr); err != nil {
		t.Fatalf("install initial epoch-2 validator set: %v", err)
	}
	beforeHydraCurrent := hydraMgr.GetCurrentConfiguration()
	beforeHydraHighestKnown := hydraMgr.GetHighestKnownConfiguration()
	beforeHydraLSet := hydraMgr.LSetManager.GetLSet()
	beforeHydraHistory := hydraMgr.DiscoveryManager.GetHistory()
	if err := hydraMgr.SubmitJoinRequest(5, joinerKey.PublicKey, 1); err != nil {
		t.Fatalf("submit join request: %v", err)
	}
	if err := hydraMgr.SubmitLeaveRequest(0); err != nil {
		t.Fatalf("submit leave request: %v", err)
	}
	beforePendingJoins := hydraMgr.TempConfigManager.GetPendingJoins()
	beforePendingLeaves := hydraMgr.TempConfigManager.GetPendingLeaves()
	beforeMembership := mm.GetCurrentConfig()
	beforeMembershipHash := beforeMembership.Hash()
	beforeTempHistory := hydraMgr.TempConfigManager.GetHistory()
	beforeEngine := engine.GetCurrentValidatorSet().Copy()

	conflicting := types.NewValidatorSet(2, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: kp2.PublicKey, Power: 1, IsActive: true},
	})
	conflictingTransitions := []*types.EpochTransition{{
		OldEpoch:         1,
		NewEpoch:         2,
		ActivationHeight: 6,
		ActivationRank:   12,
		Added:            []uint64{2},
		QuorumSize:       conflicting.QuorumSize,
		ConfigHash:       conflicting.Hash(),
	}}
	if err := installCommittedValidatorSet(conflicting, conflictingTransitions, mm, []*hotstuff.Engine{engine}, nil, "", hydraMgr); err == nil {
		t.Fatalf("expected conflicting same-epoch validator set to fail")
	} else if !strings.Contains(err.Error(), "conflicting committed validator set") {
		t.Fatalf("expected conflicting same-epoch validator set error, got: %v", err)
	}
	if pending := hydraMgr.TempConfigManager.GetPendingJoins(); !reflect.DeepEqual(pending, beforePendingJoins) {
		t.Fatalf("expected conflicting wrapper path to preserve hydra pending joins, got %+v", pending)
	}
	if pending := hydraMgr.TempConfigManager.GetPendingLeaves(); !reflect.DeepEqual(pending, beforePendingLeaves) {
		t.Fatalf("expected conflicting wrapper path to preserve hydra pending leaves, got %+v", pending)
	}
	afterMembership := mm.GetCurrentConfig()
	if afterMembership == nil || beforeMembership == nil || afterMembership.ID != beforeMembership.ID {
		t.Fatalf("expected conflicting wrapper path to preserve membership epoch %d, got %+v", beforeMembership.ID, afterMembership)
	}
	if !reflect.DeepEqual(afterMembership.Validators, beforeMembership.Validators) {
		t.Fatalf("expected conflicting wrapper path to preserve membership validators")
	}
	if !bytes.Equal(afterMembership.Hash(), beforeMembershipHash) {
		t.Fatalf("expected conflicting wrapper path to preserve membership hash %x, got %x", beforeMembershipHash, afterMembership.Hash())
	}
	afterEngine := engine.GetCurrentValidatorSet()
	if afterEngine == nil || beforeEngine == nil || afterEngine.Epoch != beforeEngine.Epoch {
		t.Fatalf("expected conflicting wrapper path to preserve engine epoch %d, got %+v", beforeEngine.Epoch, afterEngine)
	}
	if !reflect.DeepEqual(afterEngine.Validators, beforeEngine.Validators) {
		t.Fatalf("expected conflicting wrapper path to preserve engine validators")
	}
	hydraCurrent := hydraMgr.GetCurrentConfiguration()
	if hydraCurrent == nil || beforeHydraCurrent == nil || hydraCurrent.ID != beforeHydraCurrent.ID {
		t.Fatalf("expected conflicting wrapper path to preserve hydra current config id %d, got %+v", beforeHydraCurrent.ID, hydraCurrent)
	}
	if !reflect.DeepEqual(hydraCurrent.Validators, beforeHydraCurrent.Validators) {
		t.Fatalf("expected conflicting wrapper path to preserve hydra current validators")
	}
	if highest := hydraMgr.GetHighestKnownConfiguration(); !reflect.DeepEqual(highest, beforeHydraHighestKnown) {
		t.Fatalf("expected conflicting wrapper path to preserve hydra highest-known config")
	}
	if lset := hydraMgr.LSetManager.GetLSet(); !reflect.DeepEqual(lset, beforeHydraLSet) {
		t.Fatalf("expected conflicting wrapper path to preserve hydra lset")
	}
	if history := hydraMgr.DiscoveryManager.GetHistory(); !reflect.DeepEqual(history, beforeHydraHistory) {
		t.Fatalf("expected conflicting wrapper path to preserve hydra history")
	}
	if history := hydraMgr.TempConfigManager.GetHistory(); !reflect.DeepEqual(history, beforeTempHistory) {
		t.Fatalf("expected conflicting wrapper path to preserve hydra temp history")
	}
}

func TestInstallCommittedValidatorSetReconcilesHydraAndEngineOnReplayWithoutClearingUnrelatedPendingIntents(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	joinerKey, _ := crypto.GenerateKeyPair()
	initial := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
	}
	mm := membership.NewMembershipManager(initial)
	engine := buildHydraInjectionEngineForTest(t, kp0.PublicKey)
	hydraMgr, err := hydra.NewHydraManager(0, map[uint64]*hydra.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
	}, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}

	newValSet := types.NewValidatorSet(2, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
	})
	transitions := []*types.EpochTransition{{
		OldEpoch:         1,
		NewEpoch:         2,
		ActivationHeight: 4,
		ActivationRank:   9,
		Added:            []uint64{1},
		QuorumSize:       newValSet.QuorumSize,
		ConfigHash:       newValSet.Hash(),
	}}
	if _, _, changed, err := mm.InstallValidatorSetFromTransitions(newValSet, transitions); err != nil || !changed {
		t.Fatalf("preinstall membership config failed: changed=%v err=%v", changed, err)
	}
	current := mm.GetCurrentConfig()
	if current == nil || current.ID != 2 {
		t.Fatalf("expected membership to already be at epoch 2 before replay, got %+v", current)
	}
	if !bytes.Equal(current.Hash(), newValSet.Hash()) {
		t.Fatalf("expected replay precondition to be same-hash with committed membership")
	}
	if hydraCurrent := hydraMgr.GetCurrentConfiguration(); hydraCurrent == nil || hydraCurrent.ID != 0 {
		t.Fatalf("expected hydra to remain behind before replay reconcile, got %+v", hydraCurrent)
	}
	beforeEngine := engine.GetCurrentValidatorSet()
	if beforeEngine == nil || beforeEngine.Epoch == 2 {
		t.Fatalf("expected engine to remain stale before replay reconcile, got %+v", beforeEngine)
	}
	if err := hydraMgr.SubmitJoinRequest(5, joinerKey.PublicKey, 1); err != nil {
		t.Fatalf("submit join request: %v", err)
	}
	if err := hydraMgr.SubmitLeaveRequest(0); err != nil {
		t.Fatalf("submit leave request: %v", err)
	}
	beforePendingJoins := hydraMgr.TempConfigManager.GetPendingJoins()
	beforePendingLeaves := hydraMgr.TempConfigManager.GetPendingLeaves()
	beforeTempHistory := hydraMgr.TempConfigManager.GetHistory()
	if err := installCommittedValidatorSet(newValSet, transitions, mm, []*hotstuff.Engine{engine}, nil, "", hydraMgr); err != nil {
		t.Fatalf("replay committed validator set should reconcile hydra and engine: %v", err)
	}
	hydraCurrent := hydraMgr.GetCurrentConfiguration()
	if hydraCurrent == nil || hydraCurrent.ID != 2 {
		t.Fatalf("expected hydra committed config to catch up to epoch 2, got %+v", hydraCurrent)
	}
	if _, ok := hydraCurrent.Validators[1]; !ok {
		t.Fatalf("expected replay reconcile to install validator 1 into hydra")
	}
	afterEngine := engine.GetCurrentValidatorSet()
	if afterEngine == nil || afterEngine.Epoch != 2 {
		t.Fatalf("expected engine validator set to catch up to epoch 2, got %+v", afterEngine)
	}
	if _, ok := afterEngine.Validators[1]; !ok {
		t.Fatalf("expected replay reconcile to install validator 1 into engine")
	}
	if pending := hydraMgr.TempConfigManager.GetPendingJoins(); !reflect.DeepEqual(pending, beforePendingJoins) {
		t.Fatalf("expected replay reconcile to preserve unrelated hydra pending joins, got %+v", pending)
	}
	if pending := hydraMgr.TempConfigManager.GetPendingLeaves(); !reflect.DeepEqual(pending, beforePendingLeaves) {
		t.Fatalf("expected replay reconcile to preserve unrelated hydra pending leaves, got %+v", pending)
	}
	if history := hydraMgr.TempConfigManager.GetHistory(); len(history) != len(beforeTempHistory)+1 {
		t.Fatalf("expected replay reconcile to append exactly one hydra temp-history entry, got len=%d want=%d", len(history), len(beforeTempHistory)+1)
	} else {
		if !reflect.DeepEqual(history[:len(beforeTempHistory)], beforeTempHistory) {
			t.Fatalf("expected replay reconcile to preserve prior hydra temp-history prefix")
		}
		last := history[len(history)-1]
		if last == nil || last.ID != 2 {
			t.Fatalf("expected replay reconcile to append hydra temp-history entry for epoch 2, got %+v", last)
		}
		if !reflect.DeepEqual(last.Validators, hydraCurrent.Validators) {
			t.Fatalf("expected replay reconcile to append hydra temp-history entry matching committed hydra validators")
		}
	}
}

func TestInstallCommittedValidatorSetRejectsHydraMirrorFailureBeforeMembershipAdvance(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	initial := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
	}
	mm := membership.NewMembershipManager(initial)
	exec := hotstuff.NewExecutor(mm.GetCurrentConfig().ToValidatorSet())
	exec.SetReconfigAuthorizer(func(data *types.ReconfigData) bool { return data != nil && data.Type == types.ReconfigJoin && data.NodeID == 1 })
	engine := buildHydraInjectionEngineForTest(t, kp0.PublicKey)
	beforeMembership := mm.GetCurrentConfig()
	beforeEngine := engine.GetCurrentValidatorSet().Copy()
	beforeExecutor := exec.GetCurrentValidatorSet().Copy()

	brokenHydraMgr, err := hydra.NewHydraManager(0, map[uint64]*hydra.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
	}, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	brokenHydraMgr.TempConfigManager = nil

	exec.SetEpochChangeCallback(func(newValSet *types.ValidatorSet, transitions []*types.EpochTransition) error {
		return installCommittedValidatorSet(newValSet, transitions, mm, []*hotstuff.Engine{engine}, nil, "", brokenHydraMgr)
	})

	block := types.NewBlock(4, nil, []byte("payload"), 1, 1, 0, 0, nil, nil)
	reconfig := types.ReconfigData{Type: types.ReconfigJoin, NodeID: 1, PublicKey: kp1.PublicKey, Power: 1, Epoch: 1, TargetEpoch: 2}
	reconfig.Sign(kp1.PrivateKey)
	reconfigPayload, err := json.Marshal(reconfig)
	if err != nil {
		t.Fatalf("marshal reconfig data: %v", err)
	}
	txPayload, err := json.Marshal(&types.Transaction{Type: types.TxTypeReconfig, Payload: reconfigPayload})
	if err != nil {
		t.Fatalf("marshal tx: %v", err)
	}
	block.Data = txPayload
	block.Hash = []byte("hydra-failure-block")

	if err := exec.ExecuteBlock(block); err != nil {
		t.Fatalf("execute block: %v", err)
	}
	if got, err := exec.CommitReconfigs(block, 9); err == nil || len(got) != 0 {
		t.Fatalf("expected hydra mirror failure to abort committed install, got transitions=%v err=%v", got, err)
	}
	afterMembership := mm.GetCurrentConfig()
	if afterMembership.ID != beforeMembership.ID {
		t.Fatalf("membership advanced despite hydra failure: before=%d after=%d", beforeMembership.ID, afterMembership.ID)
	}
	if _, ok := afterMembership.Validators[1]; ok {
		t.Fatalf("membership installed validator 1 despite hydra failure")
	}
	afterEngine := engine.GetCurrentValidatorSet()
	if afterEngine == nil || afterEngine.Epoch != beforeEngine.Epoch {
		t.Fatalf("engine validator set changed despite hydra failure: before=%+v after=%+v", beforeEngine, afterEngine)
	}
	if _, ok := afterEngine.Validators[1]; ok {
		t.Fatalf("engine installed validator 1 despite hydra failure")
	}
	afterExecutor := exec.GetCurrentValidatorSet()
	if afterExecutor == nil || afterExecutor.Epoch != beforeExecutor.Epoch {
		t.Fatalf("executor validator set changed despite hydra failure: before=%+v after=%+v", beforeExecutor, afterExecutor)
	}
	if _, ok := afterExecutor.Validators[1]; ok {
		t.Fatalf("executor installed validator 1 despite hydra failure")
	}
	if got, err := exec.CommitReconfigs(block, 9); err == nil || len(got) != 0 {
		t.Fatalf("expected repeated commit to continue surfacing hydra failure until reconciled, got transitions=%v err=%v", got, err)
	}
	if pending := exec.PendingReconfigCount(); pending != 1 {
		t.Fatalf("expected executor to retain pending reconfig after callback failure, got %d", pending)
	}
}

func TestBlockHashChangesWithPayloadCertificates(t *testing.T) {
	base := types.NewBlock(1, nil, []byte("same-data"), 1, 1, 0, 0, nil, nil)
	base.ConfigID = 1
	base.LaneID = 0
	base.Timestamp = 123
	base.Payload = []*types.VertexCertificate{{VertexHash: types.Hash{1}, Epoch: 1, Round: 1}}
	base.Hash = base.ComputeHash()

	changed := types.NewBlock(1, nil, []byte("same-data"), 1, 1, 0, 0, nil, nil)
	changed.ConfigID = 1
	changed.LaneID = 0
	changed.Timestamp = 123
	changed.Payload = []*types.VertexCertificate{{VertexHash: types.Hash{2}, Epoch: 1, Round: 1}}
	changed.Hash = changed.ComputeHash()

	if string(base.Hash) == string(changed.Hash) {
		t.Fatalf("expected block hash to change when payload certificates change")
	}
}

func TestLoadBootstrapState_UsesManifestIdentity(t *testing.T) {
	node0 := makeManifestNodeForTest(t, 0)
	node1 := makeManifestNodeForTest(t, 1)
	manifestPath := writeManifestForTest(t, node0, node1)

	cfg := &bootstrap.EngineConfig{
		NodeID:   1,
		Manifest: manifestPath,
	}
	manifest, err := bootstrap.LoadGenesisManifest(manifestPath)
	if err != nil {
		t.Fatalf("load manifest: %v", err)
	}
	keypair, validators, peerMap, err := loadBootstrapState(cfg, manifest)
	if err != nil {
		t.Fatalf("load bootstrap state: %v", err)
	}
	if got := hex.EncodeToString(keypair.PublicKey); got != node1.PublicKeyHex {
		t.Fatalf("unexpected keypair public key: %s", got)
	}
	if len(validators) != 2 {
		t.Fatalf("unexpected validator count: %d", len(validators))
	}
	if len(peerMap) != 2 {
		t.Fatalf("unexpected peer map size: %d", len(peerMap))
	}
	if peerMap[0].ID.String() != node0.PeerID {
		t.Fatalf("unexpected node 0 peer id: %s", peerMap[0].ID)
	}
}

func TestLoadManifestVRFState_UsesManifestKeys(t *testing.T) {
	node0 := makeManifestNodeForTest(t, 0)
	node1 := makeManifestNodeForTest(t, 1)
	manifestPath := writeManifestForTest(t, node0, node1)

	cfg := &bootstrap.EngineConfig{
		NodeID:   1,
		Manifest: manifestPath,
	}
	manifest, err := bootstrap.LoadGenesisManifest(manifestPath)
	if err != nil {
		t.Fatalf("load manifest: %v", err)
	}
	vrfPriv, vrfPub, vrfPubKeys, err := loadManifestVRFState(cfg, manifest)
	if err != nil {
		t.Fatalf("load manifest vrf state: %v", err)
	}
	if len(vrfPubKeys) != 2 {
		t.Fatalf("unexpected vrf pubkey count: %d", len(vrfPubKeys))
	}
	vrfPrivRaw, err := vrfPriv.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal local vrf private key: %v", err)
	}
	vrfPubRaw, err := vrfPub.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal local vrf public key: %v", err)
	}
	if got := hex.EncodeToString(vrfPrivRaw); got != node1.VRFPrivateKeyHex {
		t.Fatalf("unexpected local vrf private key: %s", got)
	}
	if got := hex.EncodeToString(vrfPubRaw); got != node1.VRFPublicKeyHex {
		t.Fatalf("unexpected local vrf public key: %s", got)
	}
	pub0Raw, err := vrfPubKeys[0].MarshalBinary()
	if err != nil {
		t.Fatalf("marshal validator 0 vrf public key: %v", err)
	}
	if got := hex.EncodeToString(pub0Raw); got != node0.VRFPublicKeyHex {
		t.Fatalf("unexpected validator 0 vrf public key: %s", got)
	}
}

func TestLoadManifestVRFState_RejectsMissingLocalVRFPrivateKey(t *testing.T) {
	node0 := makeManifestNodeForTest(t, 0)
	node1 := makeManifestNodeForTest(t, 1)
	node1.VRFPrivateKeyHex = ""
	manifestPath := writeManifestForTest(t, node0, node1)

	cfg := &bootstrap.EngineConfig{
		NodeID:   1,
		Manifest: manifestPath,
	}
	manifest, err := bootstrap.LoadGenesisManifest(manifestPath)
	if err != nil {
		t.Fatalf("load manifest: %v", err)
	}
	if _, _, _, err := loadManifestVRFState(cfg, manifest); err == nil {
		t.Fatalf("expected manifest vrf state load to fail without local vrf private key")
	}
}

func TestLoadManifest_RejectsMissingPeerMultiaddrForMultiNode(t *testing.T) {
	node0 := makeManifestNodeForTest(t, 0)
	node1 := makeManifestNodeForTest(t, 1)
	node1.P2PMultiaddr = ""
	manifestPath := writeManifestForTest(t, node0, node1)

	cfg := &bootstrap.EngineConfig{
		NodeID:   1,
		Manifest: manifestPath,
	}
	if _, err := loadManifest(cfg); err == nil {
		t.Fatalf("expected manifest load to fail without multi-node peer multiaddr")
	}
}

func TestLoadManifest_RejectsMissingVRFMaterial(t *testing.T) {
	node0 := makeManifestNodeForTest(t, 0)
	node1 := makeManifestNodeForTest(t, 1)
	node0.VRFPublicKeyHex = ""
	node0.VRFPrivateKeyHex = ""
	manifestPath := writeManifestForTest(t, node0, node1)

	cfg := &bootstrap.EngineConfig{
		NodeID:   1,
		Manifest: manifestPath,
	}
	if _, err := loadManifest(cfg); err == nil {
		t.Fatalf("expected manifest load to fail without validator vrf material")
	}
}

func TestLoadManifest_RejectsMissingLocalPrivateKey(t *testing.T) {
	node0 := makeManifestNodeForTest(t, 0)
	node1 := makeManifestNodeForTest(t, 1)
	node1.PrivateKeyHex = ""
	manifestPath := writeManifestForTest(t, node0, node1)

	cfg := &bootstrap.EngineConfig{
		NodeID:   1,
		Manifest: manifestPath,
	}
	if _, err := loadManifest(cfg); err == nil {
		t.Fatalf("expected manifest load to fail without local private key")
	}
}

func TestManifestVRFState_WiresAllInstances(t *testing.T) {
	node0 := makeManifestNodeForTest(t, 0)
	node1 := makeManifestNodeForTest(t, 1)
	manifestPath := writeManifestForTest(t, node0, node1)
	cfg := &bootstrap.EngineConfig{
		NodeID:   1,
		Manifest: manifestPath,
	}
	manifest, err := bootstrap.LoadGenesisManifest(manifestPath)
	if err != nil {
		t.Fatalf("load manifest: %v", err)
	}

	keypair, validators, _, err := loadBootstrapState(cfg, manifest)
	if err != nil {
		t.Fatalf("load bootstrap state: %v", err)
	}
	vrfPriv, vrfPub, vrfPubKeys, err := loadManifestVRFState(cfg, manifest)
	if err != nil {
		t.Fatalf("load manifest vrf state: %v", err)
	}
	if vrfPriv == nil {
		t.Fatalf("expected local vrf private key")
	}
	if vrfPub == nil {
		t.Fatalf("expected local vrf public key")
	}
	if len(validators) != 2 {
		t.Fatalf("unexpected validator count: %d", len(validators))
	}

	for instanceID := uint64(0); instanceID < 2; instanceID++ {
		store := storage.NewStorageManager(1000 + instanceID)
		defer store.Close()
		engine := hotstuff.NewEngineWithInstanceAndOptions(1, keypair, types.NewValidatorSet(1, validators), nil, store, instanceID, 2, "test-consensus", nil, hotstuff.DefaultEngineOptions())
		engine.SetLocalVRFKeypair(vrfPriv, vrfPub)
		for validatorID, pubKey := range vrfPubKeys {
			engine.RegisterVRFPubKey(validatorID, pubKey)
		}

		gotPub := engine.GetVRFPubKey()
		if gotPub == nil {
			t.Fatalf("instance %d missing local vrf pubkey", instanceID)
		}
		gotPubRaw, err := gotPub.MarshalBinary()
		if err != nil {
			t.Fatalf("marshal instance %d local vrf pubkey: %v", instanceID, err)
		}
		if got := hex.EncodeToString(gotPubRaw); got != node1.VRFPublicKeyHex {
			t.Fatalf("instance %d unexpected local vrf public key: %s", instanceID, got)
		}
		pub0Raw, err := vrfPubKeys[0].MarshalBinary()
		if err != nil {
			t.Fatalf("marshal validator 0 vrf public key: %v", err)
		}
		if got := hex.EncodeToString(pub0Raw); got != node0.VRFPublicKeyHex {
			t.Fatalf("instance %d unexpected validator 0 vrf public key: %s", instanceID, got)
		}
	}
}

func TestLoadBootstrapState_RejectsEphemeralMultiNodeBootstrap(t *testing.T) {
	cfg := &bootstrap.EngineConfig{
		NodeID:            0,
		TotalNodes:        4,
		InitialValidators: 4,
	}

	_, _, _, err := loadBootstrapState(cfg, nil)
	if err == nil {
		t.Fatalf("expected multi-node bootstrap without manifest to fail")
	}
}

func TestLoadBootstrapState_AllowsSingleNodeEphemeralBootstrap(t *testing.T) {
	cfg := &bootstrap.EngineConfig{
		NodeID:            0,
		TotalNodes:        1,
		InitialValidators: 1,
	}

	keypair, validators, peerMap, err := loadBootstrapState(cfg, nil)
	if err != nil {
		t.Fatalf("single-node bootstrap failed: %v", err)
	}
	if len(keypair.PublicKey) == 0 || len(keypair.PrivateKey) == 0 {
		t.Fatalf("expected local keypair to be generated")
	}
	if len(validators) != 1 {
		t.Fatalf("unexpected validator count: %d", len(validators))
	}
	if validator := validators[0]; validator == nil || len(validator.PublicKey) == 0 {
		t.Fatalf("expected validator 0 public key to be populated")
	}
	if len(peerMap) != 0 {
		t.Fatalf("expected empty peer map in single-node ephemeral mode, got %d entries", len(peerMap))
	}
}

func TestBuildAdminMux_RejectsWrongMethodsOnPrivilegedRoutes(t *testing.T) {
	engine := buildAdaptiveEngineForTest(t)
	memberMgr := membership.NewMembershipManager(engine.GetCurrentValidatorSet().Validators)
	handler := buildAdminMux(
		0,
		&types.Keypair{PublicKey: engine.GetCurrentValidatorSet().Validators[0].PublicKey},
		engine,
		memberMgr,
		[]*hotstuff.Engine{engine},
		hotstuff.NewGlobalOrderer(1, 50*time.Millisecond),
		hotstuff.NewGlobalConfirmedMetrics(50*time.Millisecond),
		nil,
		nil,
		nil,
		&evolvbftAdaptiveRuntime{},
		nil,
		false,
	)

	cases := []struct {
		method string
		path   string
	}{
		{method: http.MethodGet, path: "/join"},
		{method: http.MethodGet, path: "/leave"},
		{method: http.MethodGet, path: "/tx"},
		{method: http.MethodGet, path: "/debug/gc"},
		{method: http.MethodPost, path: "/config"},
		{method: http.MethodPost, path: "/metrics"},
		{method: http.MethodPost, path: "/adaptive"},
	}

	for _, tc := range cases {
		req := httptest.NewRequest(tc.method, tc.path, nil)
		rr := httptest.NewRecorder()
		handler.ServeHTTP(rr, req)
		if rr.Code != http.StatusMethodNotAllowed {
			t.Fatalf("expected %s %s to return 405, got %d", tc.method, tc.path, rr.Code)
		}
	}
}

func TestBuildAdminMux_DisablesPprofByDefault(t *testing.T) {
	engine := buildAdaptiveEngineForTest(t)
	memberMgr := membership.NewMembershipManager(engine.GetCurrentValidatorSet().Validators)
	handler := buildAdminMux(
		0,
		&types.Keypair{PublicKey: engine.GetCurrentValidatorSet().Validators[0].PublicKey},
		engine,
		memberMgr,
		[]*hotstuff.Engine{engine},
		hotstuff.NewGlobalOrderer(1, 50*time.Millisecond),
		hotstuff.NewGlobalConfirmedMetrics(50*time.Millisecond),
		nil,
		nil,
		nil,
		&evolvbftAdaptiveRuntime{},
		nil,
		false,
	)

	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, httptest.NewRequest(http.MethodGet, "/debug/pprof/", nil))
	if resp.Code != http.StatusNotFound {
		t.Fatalf("expected /debug/pprof/ to be disabled by default, got %d", resp.Code)
	}
}

func TestBuildAdminMux_EnablesPprofWhenConfigured(t *testing.T) {
	engine := buildAdaptiveEngineForTest(t)
	memberMgr := membership.NewMembershipManager(engine.GetCurrentValidatorSet().Validators)
	handler := buildAdminMux(
		0,
		&types.Keypair{PublicKey: engine.GetCurrentValidatorSet().Validators[0].PublicKey},
		engine,
		memberMgr,
		[]*hotstuff.Engine{engine},
		hotstuff.NewGlobalOrderer(1, 50*time.Millisecond),
		hotstuff.NewGlobalConfirmedMetrics(50*time.Millisecond),
		nil,
		nil,
		nil,
		&evolvbftAdaptiveRuntime{},
		nil,
		true,
	)

	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, httptest.NewRequest(http.MethodGet, "/debug/pprof/", nil))
	if resp.Code != http.StatusOK {
		t.Fatalf("expected /debug/pprof/ to be enabled when configured, got %d", resp.Code)
	}
}

func setMempoolMaxTxSize(engine *hotstuff.Engine, size int) {
	mp := hydraInjectionMempool(engine)
	if mp == nil {
		return
	}
	mpValue := reflect.ValueOf(mp).Elem().FieldByName("maxTxSize")
	reflect.NewAt(mpValue.Type(), unsafe.Pointer(mpValue.UnsafeAddr())).Elem().SetInt(int64(size))
}

func TestBuildAdminMux_RejectsOversizedBodies(t *testing.T) {
	engine := buildAdaptiveEngineForTest(t)
	memberMgr := membership.NewMembershipManager(engine.GetCurrentValidatorSet().Validators)
	handler := buildAdminMux(
		0,
		&types.Keypair{PublicKey: engine.GetCurrentValidatorSet().Validators[0].PublicKey},
		engine,
		memberMgr,
		[]*hotstuff.Engine{engine},
		hotstuff.NewGlobalOrderer(1, 50*time.Millisecond),
		hotstuff.NewGlobalConfirmedMetrics(50*time.Millisecond),
		nil,
		nil,
		nil,
		&evolvbftAdaptiveRuntime{},
		nil,
		false,
	)

	txReq := httptest.NewRequest(http.MethodPost, "/tx", strings.NewReader(strings.Repeat("x", adminMaxTxBodyBytes+1)))
	txResp := httptest.NewRecorder()
	handler.ServeHTTP(txResp, txReq)
	if txResp.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("expected oversized /tx body to return 413, got %d", txResp.Code)
	}

	ctxReq := httptest.NewRequest(http.MethodPost, "/adaptive/context", strings.NewReader(`{"heterogeneity_score":"`+strings.Repeat("x", adminMaxAdaptiveContextBodyBytes+1)+`"}`))
	ctxResp := httptest.NewRecorder()
	handler.ServeHTTP(ctxResp, ctxReq)
	if ctxResp.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("expected oversized /adaptive/context body to return 413, got %d", ctxResp.Code)
	}
}

func TestBuildAdminMux_JoinFailureRollsBackHydraPendingIntent(t *testing.T) {
	engine := buildAdaptiveEngineForTest(t)
	setMempoolMaxTxSize(engine, 1)
	memberMgr := membership.NewMembershipManager(engine.GetCurrentValidatorSet().Validators)
	hydraValidators := make(map[uint64]*hydra.Validator, len(engine.GetCurrentValidatorSet().Validators))
	for id, v := range engine.GetCurrentValidatorSet().Validators {
		copyVal := *v
		hydraValidators[id] = &hydra.Validator{ID: copyVal.ID, PublicKey: copyVal.PublicKey, Power: copyVal.Power, IsActive: copyVal.IsActive}
	}
	hydraMgr, err := hydra.NewHydraManager(4, hydraValidators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	kp, err := crypto.GenerateKeyPair()
	if err != nil {
		t.Fatalf("generate keypair: %v", err)
	}
	handler := buildAdminMux(
		4,
		&types.Keypair{PublicKey: kp.PublicKey, PrivateKey: kp.PrivateKey},
		engine,
		memberMgr,
		[]*hotstuff.Engine{engine},
		hotstuff.NewGlobalOrderer(1, 50*time.Millisecond),
		hotstuff.NewGlobalConfirmedMetrics(50*time.Millisecond),
		nil,
		hydraMgr,
		nil,
		&evolvbftAdaptiveRuntime{},
		nil,
		false,
	)

	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, httptest.NewRequest(http.MethodPost, "/join", nil))
	if resp.Code != http.StatusInternalServerError {
		t.Fatalf("expected POST /join to return 500 when submission fails, got %d", resp.Code)
	}
	if pending := hydraMgr.TempConfigManager.GetPendingJoins(); len(pending) != 0 {
		t.Fatalf("expected hydra pending join rollback after admin enqueue failure, got %+v", pending)
	}
}

func TestBuildAdminMux_JoinReturns500WhenSubmissionFails(t *testing.T) {
	engine := buildAdaptiveEngineForTest(t)
	setMempoolMaxTxSize(engine, 1)
	memberMgr := membership.NewMembershipManager(engine.GetCurrentValidatorSet().Validators)
	kp, err := crypto.GenerateKeyPair()
	if err != nil {
		t.Fatalf("generate keypair: %v", err)
	}
	handler := buildAdminMux(
		4,
		&types.Keypair{PublicKey: kp.PublicKey, PrivateKey: kp.PrivateKey},
		engine,
		memberMgr,
		[]*hotstuff.Engine{engine},
		hotstuff.NewGlobalOrderer(1, 50*time.Millisecond),
		hotstuff.NewGlobalConfirmedMetrics(50*time.Millisecond),
		nil,
		nil,
		nil,
		&evolvbftAdaptiveRuntime{},
		nil,
		false,
	)

	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, httptest.NewRequest(http.MethodPost, "/join", nil))
	if resp.Code != http.StatusInternalServerError {
		t.Fatalf("expected POST /join to return 500 when submission fails, got %d", resp.Code)
	}
}

func TestBuildAdminMux_LeaveReturns500WhenSubmissionFails(t *testing.T) {
	engine := buildAdaptiveEngineForTest(t)
	setMempoolMaxTxSize(engine, 1)
	memberMgr := membership.NewMembershipManager(engine.GetCurrentValidatorSet().Validators)
	kp, err := crypto.GenerateKeyPair()
	if err != nil {
		t.Fatalf("generate keypair: %v", err)
	}
	handler := buildAdminMux(
		0,
		&types.Keypair{PublicKey: kp.PublicKey, PrivateKey: kp.PrivateKey},
		engine,
		memberMgr,
		[]*hotstuff.Engine{engine},
		hotstuff.NewGlobalOrderer(1, 50*time.Millisecond),
		hotstuff.NewGlobalConfirmedMetrics(50*time.Millisecond),
		nil,
		nil,
		nil,
		&evolvbftAdaptiveRuntime{},
		nil,
		false,
	)

	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, httptest.NewRequest(http.MethodPost, "/leave", nil))
	if resp.Code != http.StatusInternalServerError {
		t.Fatalf("expected POST /leave to return 500 when submission fails, got %d", resp.Code)
	}
}

func TestBuildAdminMux_JoinUsesHydraPendingIntentPath(t *testing.T) {
	engine := buildAdaptiveEngineForTest(t)
	memberMgr := membership.NewMembershipManager(engine.GetCurrentValidatorSet().Validators)
	hydraValidators := make(map[uint64]*hydra.Validator, len(engine.GetCurrentValidatorSet().Validators))
	for id, v := range engine.GetCurrentValidatorSet().Validators {
		copyVal := *v
		hydraValidators[id] = &hydra.Validator{ID: copyVal.ID, PublicKey: copyVal.PublicKey, Power: copyVal.Power, IsActive: copyVal.IsActive}
	}
	hydraMgr, err := hydra.NewHydraManager(4, hydraValidators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	kp, err := crypto.GenerateKeyPair()
	if err != nil {
		t.Fatalf("generate keypair: %v", err)
	}
	handler := buildAdminMux(
		4,
		&types.Keypair{PublicKey: kp.PublicKey, PrivateKey: kp.PrivateKey},
		engine,
		memberMgr,
		[]*hotstuff.Engine{engine},
		hotstuff.NewGlobalOrderer(1, 50*time.Millisecond),
		hotstuff.NewGlobalConfirmedMetrics(50*time.Millisecond),
		nil,
		hydraMgr,
		nil,
		&evolvbftAdaptiveRuntime{},
		nil,
		false,
	)

	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, httptest.NewRequest(http.MethodPost, "/join", nil))
	if resp.Code != http.StatusOK {
		t.Fatalf("expected POST /join to succeed, got %d", resp.Code)
	}
	pending := hydraMgr.TempConfigManager.GetPendingJoins()
	if len(pending) != 1 || pending[0].ID != 4 {
		t.Fatalf("expected hydra pending join for node 4, got %+v", pending)
	}
	if string(pending[0].PublicKey) != string(kp.PublicKey) {
		t.Fatalf("unexpected pending join public key")
	}
	tx := drainSingleReconfigTx(t, hydraInjectionMempool(engine))
	if tx == nil {
		t.Fatalf("expected queued join transaction")
	}
	var data types.ReconfigData
	if err := json.Unmarshal(tx.Payload, &data); err != nil {
		t.Fatalf("unmarshal queued join payload: %v", err)
	}
	if data.Type != types.ReconfigJoin || data.NodeID != 4 {
		t.Fatalf("unexpected queued join data: %+v", data)
	}
	if !reflect.DeepEqual(data.PublicKey, kp.PublicKey) {
		t.Fatalf("unexpected queued join public key")
	}
	if data.Epoch != memberMgr.GetCurrentConfig().ID || data.TargetEpoch != memberMgr.GetCurrentConfig().ID+1 {
		t.Fatalf("unexpected queued join epochs: epoch=%d target=%d", data.Epoch, data.TargetEpoch)
	}
}

func TestBuildAdminMux_LeaveUsesHydraPendingIntentPath(t *testing.T) {
	engine := buildAdaptiveEngineForTest(t)
	memberMgr := membership.NewMembershipManager(engine.GetCurrentValidatorSet().Validators)
	hydraValidators := make(map[uint64]*hydra.Validator, len(engine.GetCurrentValidatorSet().Validators))
	for id, v := range engine.GetCurrentValidatorSet().Validators {
		copyVal := *v
		hydraValidators[id] = &hydra.Validator{ID: copyVal.ID, PublicKey: copyVal.PublicKey, Power: copyVal.Power, IsActive: copyVal.IsActive}
	}
	hydraMgr, err := hydra.NewHydraManager(0, hydraValidators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	kp, err := crypto.GenerateKeyPair()
	if err != nil {
		t.Fatalf("generate keypair: %v", err)
	}
	handler := buildAdminMux(
		0,
		&types.Keypair{PublicKey: kp.PublicKey, PrivateKey: kp.PrivateKey},
		engine,
		memberMgr,
		[]*hotstuff.Engine{engine},
		hotstuff.NewGlobalOrderer(1, 50*time.Millisecond),
		hotstuff.NewGlobalConfirmedMetrics(50*time.Millisecond),
		nil,
		hydraMgr,
		nil,
		&evolvbftAdaptiveRuntime{},
		nil,
		false,
	)

	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, httptest.NewRequest(http.MethodPost, "/leave", nil))
	if resp.Code != http.StatusOK {
		t.Fatalf("expected POST /leave to succeed, got %d", resp.Code)
	}
	pending := hydraMgr.TempConfigManager.GetPendingLeaves()
	if len(pending) != 1 || pending[0].ID != 0 {
		t.Fatalf("expected hydra pending leave for node 0, got %+v", pending)
	}
	tx := drainSingleReconfigTx(t, hydraInjectionMempool(engine))
	if tx == nil {
		t.Fatalf("expected queued leave transaction")
	}
	var data types.ReconfigData
	if err := json.Unmarshal(tx.Payload, &data); err != nil {
		t.Fatalf("unmarshal queued leave payload: %v", err)
	}
	if data.Type != types.ReconfigLeave || data.NodeID != 0 {
		t.Fatalf("unexpected queued leave data: %+v", data)
	}
	if data.Epoch != memberMgr.GetCurrentConfig().ID || data.TargetEpoch != memberMgr.GetCurrentConfig().ID+1 {
		t.Fatalf("unexpected queued leave epochs: epoch=%d target=%d", data.Epoch, data.TargetEpoch)
	}
}

func TestReconfigAuthorizedByHydraPendingIntent_RequiresPendingJoin(t *testing.T) {
	hydraMgr, err := hydra.NewHydraManager(0, map[uint64]*hydra.Validator{
		0: {ID: 0, PublicKey: []byte("v0"), Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: []byte("v1"), Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: []byte("v2"), Power: 1, IsActive: true},
		3: {ID: 3, PublicKey: []byte("v3"), Power: 1, IsActive: true},
	}, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	joinerKey, err := crypto.GenerateKeyPair()
	if err != nil {
		t.Fatalf("generate joiner key: %v", err)
	}
	data := &types.ReconfigData{Type: types.ReconfigJoin, NodeID: 4, PublicKey: joinerKey.PublicKey, Power: 1}
	if reconfigAuthorizedByHydraPendingIntent(data, hydraMgr) {
		t.Fatal("expected join without pending hydra intent to be rejected")
	}
	if err := hydraMgr.SubmitJoinRequest(4, joinerKey.PublicKey, 1); err != nil {
		t.Fatalf("submit join request: %v", err)
	}
	if !reconfigAuthorizedByHydraPendingIntent(data, hydraMgr) {
		t.Fatal("expected join with pending hydra intent to be authorized")
	}
}

func TestReconfigAuthorizedByHydraPendingIntent_RequiresPendingLeave(t *testing.T) {
	hydraMgr, err := hydra.NewHydraManager(0, map[uint64]*hydra.Validator{
		0: {ID: 0, PublicKey: []byte("v0"), Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: []byte("v1"), Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: []byte("v2"), Power: 1, IsActive: true},
		3: {ID: 3, PublicKey: []byte("v3"), Power: 1, IsActive: true},
	}, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	data := &types.ReconfigData{Type: types.ReconfigLeave, NodeID: 0, PublicKey: []byte("v0")}
	if reconfigAuthorizedByHydraPendingIntent(data, hydraMgr) {
		t.Fatal("expected leave without pending hydra intent to be rejected")
	}
	if err := hydraMgr.SubmitLeaveRequest(0); err != nil {
		t.Fatalf("submit leave request: %v", err)
	}
	if !reconfigAuthorizedByHydraPendingIntent(data, hydraMgr) {
		t.Fatal("expected leave with pending hydra intent to be authorized")
	}
}

func TestReconfigAuthorizedByHydraPendingIntent_RejectsJoinAfterCommittedInstallClearsPendingIntent(t *testing.T) {
	hydraMgr, err := hydra.NewHydraManager(0, map[uint64]*hydra.Validator{
		0: {ID: 0, PublicKey: []byte("v0"), Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: []byte("v1"), Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: []byte("v2"), Power: 1, IsActive: true},
		3: {ID: 3, PublicKey: []byte("v3"), Power: 1, IsActive: true},
	}, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	joinerKey, err := crypto.GenerateKeyPair()
	if err != nil {
		t.Fatalf("generate joiner key: %v", err)
	}
	data := &types.ReconfigData{Type: types.ReconfigJoin, NodeID: 4, PublicKey: joinerKey.PublicKey, Power: 1}
	if err := hydraMgr.SubmitJoinRequest(4, joinerKey.PublicKey, 1); err != nil {
		t.Fatalf("submit join request: %v", err)
	}
	if !reconfigAuthorizedByHydraPendingIntent(data, hydraMgr) {
		t.Fatal("expected join with pending hydra intent to be authorized")
	}

	committed := &types.Configuration{
		ID: 1,
		Validators: map[uint64]*types.Validator{
			0: {ID: 0, PublicKey: []byte("v0"), Power: 1, IsActive: true},
			1: {ID: 1, PublicKey: []byte("v1"), Power: 1, IsActive: true},
			2: {ID: 2, PublicKey: []byte("v2"), Power: 1, IsActive: true},
			3: {ID: 3, PublicKey: []byte("v3"), Power: 1, IsActive: true},
			4: {ID: 4, PublicKey: joinerKey.PublicKey, Power: 1, IsActive: true},
		},
		QuorumSize: 4,
	}
	if err := hydraMgr.InstallCommittedConfiguration(committed); err != nil {
		t.Fatalf("install committed configuration: %v", err)
	}
	if reconfigAuthorizedByHydraPendingIntent(data, hydraMgr) {
		t.Fatal("expected committed join to clear pending authorization and reject replay")
	}
}

func TestReconfigAuthorizedByHydraPendingIntent_RejectsLeaveAfterCommittedInstallClearsPendingIntent(t *testing.T) {
	hydraMgr, err := hydra.NewHydraManager(0, map[uint64]*hydra.Validator{
		0: {ID: 0, PublicKey: []byte("v0"), Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: []byte("v1"), Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: []byte("v2"), Power: 1, IsActive: true},
		3: {ID: 3, PublicKey: []byte("v3"), Power: 1, IsActive: true},
	}, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	data := &types.ReconfigData{Type: types.ReconfigLeave, NodeID: 0, PublicKey: []byte("v0")}
	if err := hydraMgr.SubmitLeaveRequest(0); err != nil {
		t.Fatalf("submit leave request: %v", err)
	}
	if !reconfigAuthorizedByHydraPendingIntent(data, hydraMgr) {
		t.Fatal("expected leave with pending hydra intent to be authorized")
	}

	committed := &types.Configuration{
		ID: 1,
		Validators: map[uint64]*types.Validator{
			1: {ID: 1, PublicKey: []byte("v1"), Power: 1, IsActive: true},
			2: {ID: 2, PublicKey: []byte("v2"), Power: 1, IsActive: true},
			3: {ID: 3, PublicKey: []byte("v3"), Power: 1, IsActive: true},
		},
		QuorumSize: 3,
	}
	if err := hydraMgr.InstallCommittedConfiguration(committed); err != nil {
		t.Fatalf("install committed configuration: %v", err)
	}
	if reconfigAuthorizedByHydraPendingIntent(data, hydraMgr) {
		t.Fatal("expected committed leave to clear pending authorization and reject replay")
	}
}

func TestApplyOrderedOutput_ReplayedManualJoinIsRejectedAfterCommittedInstallClearsPendingIntent(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	kp2, _ := crypto.GenerateKeyPair()
	kp3, _ := crypto.GenerateKeyPair()
	joinerKey, _ := crypto.GenerateKeyPair()
	initial := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: kp2.PublicKey, Power: 1, IsActive: true},
		3: {ID: 3, PublicKey: kp3.PublicKey, Power: 1, IsActive: true},
	}
	memberMgr := membership.NewMembershipManager(initial)
	baseEpoch := memberMgr.GetCurrentConfig().ID
	hydraValidators := map[uint64]*hydra.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: kp2.PublicKey, Power: 1, IsActive: true},
		3: {ID: 3, PublicKey: kp3.PublicKey, Power: 1, IsActive: true},
	}
	hydraMgr, err := hydra.NewHydraManager(0, hydraValidators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	exec := hotstuff.NewExecutor(memberMgr.GetCurrentConfig().ToValidatorSet())
	exec.SetReconfigAuthorizer(func(data *types.ReconfigData) bool {
		return reconfigAuthorizedByHydraPendingIntent(data, hydraMgr)
	})
	exec.SetEpochChangeCallback(func(newValSet *types.ValidatorSet, transitions []*types.EpochTransition) error {
		return installCommittedValidatorSet(newValSet, transitions, memberMgr, nil, nil, "", hydraMgr)
	})

	join := types.ReconfigData{
		Type:        types.ReconfigJoin,
		NodeID:      4,
		PublicKey:   joinerKey.PublicKey,
		Power:       1,
		Epoch:       baseEpoch,
		TargetEpoch: baseEpoch + 1,
	}
	if err := hydraMgr.SubmitJoinRequest(4, joinerKey.PublicKey, 1); err != nil {
		t.Fatalf("submit join request: %v", err)
	}
	if !reconfigAuthorizedByHydraPendingIntent(&join, hydraMgr) {
		t.Fatal("expected join with pending hydra intent to be authorized before commit")
	}
	join.Sign(joinerKey.PrivateKey)
	joinPayload, _ := json.Marshal(join)
	txPayload, _ := json.Marshal(&types.Transaction{Type: types.TxTypeReconfig, Payload: joinPayload})

	firstBlock := types.NewBlock(1, nil, txPayload, 1, baseEpoch, 0, 0, nil, nil)
	ordered := hotstuff.InstanceOutput{
		InstanceID:  0,
		LocalHeight: firstBlock.Height,
		Rank:        28,
		BlockHash:   firstBlock.Hash,
		Block:       firstBlock,
	}
	first, err := applyOrderedOutput(ordered, exec)
	if err != nil {
		t.Fatalf("apply first ordered join: %v", err)
	}
	if len(first.EpochTransitions) != 1 {
		t.Fatalf("expected committed join to produce one transition, got %d", len(first.EpochTransitions))
	}
	if memberMgr.GetCurrentConfig().ID != baseEpoch+1 {
		t.Fatalf("expected membership to advance to epoch %d, got %d", baseEpoch+1, memberMgr.GetCurrentConfig().ID)
	}
	if _, ok := memberMgr.GetCurrentConfig().Validators[4]; !ok {
		t.Fatal("expected committed join to add validator 4")
	}
	if pending := hydraMgr.TempConfigManager.GetPendingJoins(); len(pending) != 0 {
		t.Fatalf("expected committed install path to clear pending joins, got %+v", pending)
	}
	if reconfigAuthorizedByHydraPendingIntent(&join, hydraMgr) {
		t.Fatal("expected committed install path to clear helper-level join authorization")
	}

	sameOutputReplay, err := applyOrderedOutput(ordered, exec)
	if err != nil {
		t.Fatalf("replay same ordered join output: %v", err)
	}
	if len(sameOutputReplay.EpochTransitions) != 0 {
		t.Fatalf("expected same ordered join output replay to be idempotent, got %d transitions", len(sameOutputReplay.EpochTransitions))
	}
	if pending := exec.PendingReconfigCount(); pending != 0 {
		t.Fatalf("expected same ordered join output replay to leave no queued reconfigs, got %d", pending)
	}

	replayBlock := types.NewBlock(2, nil, txPayload, 2, baseEpoch, 0, 0, nil, nil)
	replay, err := applyOrderedOutput(hotstuff.InstanceOutput{
		InstanceID:  0,
		LocalHeight: replayBlock.Height,
		Rank:        29,
		BlockHash:   replayBlock.Hash,
		Block:       replayBlock,
	}, exec)
	if err != nil {
		t.Fatalf("apply replayed join block: %v", err)
	}
	if len(replay.EpochTransitions) != 0 {
		t.Fatalf("expected replayed manual join payload to be rejected after committed install, got %d transitions", len(replay.EpochTransitions))
	}
	if memberMgr.GetCurrentConfig().ID != baseEpoch+1 {
		t.Fatalf("expected replay to leave membership at epoch %d, got %d", baseEpoch+1, memberMgr.GetCurrentConfig().ID)
	}
	if _, ok := memberMgr.GetCurrentConfig().Validators[4]; !ok {
		t.Fatal("expected replay to keep validator 4 installed")
	}
	if pending := exec.PendingReconfigCount(); pending != 0 {
		t.Fatalf("expected replay rejection to leave no queued reconfigs, got %d", pending)
	}
}

func TestApplyOrderedOutput_ReplayedManualLeaveIsRejectedAfterCommittedInstallClearsPendingIntent(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	kp2, _ := crypto.GenerateKeyPair()
	kp3, _ := crypto.GenerateKeyPair()
	kp4, _ := crypto.GenerateKeyPair()
	// Need 5 validators so that after leave n=4 >= 3f+1=4 (BFT guard)
	initial := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: kp2.PublicKey, Power: 1, IsActive: true},
		3: {ID: 3, PublicKey: kp3.PublicKey, Power: 1, IsActive: true},
		4: {ID: 4, PublicKey: kp4.PublicKey, Power: 1, IsActive: true},
	}
	memberMgr := membership.NewMembershipManager(initial)
	baseEpoch := memberMgr.GetCurrentConfig().ID
	hydraValidators := map[uint64]*hydra.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: kp2.PublicKey, Power: 1, IsActive: true},
		3: {ID: 3, PublicKey: kp3.PublicKey, Power: 1, IsActive: true},
		4: {ID: 4, PublicKey: kp4.PublicKey, Power: 1, IsActive: true},
	}
	hydraMgr, err := hydra.NewHydraManager(0, hydraValidators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	exec := hotstuff.NewExecutor(memberMgr.GetCurrentConfig().ToValidatorSet())
	exec.SetReconfigAuthorizer(func(data *types.ReconfigData) bool {
		return reconfigAuthorizedByHydraPendingIntent(data, hydraMgr)
	})
	exec.SetEpochChangeCallback(func(newValSet *types.ValidatorSet, transitions []*types.EpochTransition) error {
		return installCommittedValidatorSet(newValSet, transitions, memberMgr, nil, nil, "", hydraMgr)
	})

	leave := types.ReconfigData{
		Type:        types.ReconfigLeave,
		NodeID:      0,
		PublicKey:   kp0.PublicKey,
		Power:       1,
		Epoch:       baseEpoch,
		TargetEpoch: baseEpoch + 1,
	}
	if err := hydraMgr.SubmitLeaveRequest(0); err != nil {
		t.Fatalf("submit leave request: %v", err)
	}
	if !reconfigAuthorizedByHydraPendingIntent(&leave, hydraMgr) {
		t.Fatal("expected leave with pending hydra intent to be authorized before commit")
	}
	leave.Sign(kp0.PrivateKey)
	leavePayload, _ := json.Marshal(leave)
	txPayload, _ := json.Marshal(&types.Transaction{Type: types.TxTypeReconfig, Payload: leavePayload})

	firstBlock := types.NewBlock(1, nil, txPayload, 1, baseEpoch, 0, 0, nil, nil)
	first, err := applyOrderedOutput(hotstuff.InstanceOutput{
		InstanceID:  0,
		LocalHeight: firstBlock.Height,
		Rank:        30,
		BlockHash:   firstBlock.Hash,
		Block:       firstBlock,
	}, exec)
	if err != nil {
		t.Fatalf("apply first ordered leave: %v", err)
	}
	if len(first.EpochTransitions) != 1 {
		t.Fatalf("expected committed leave to produce one transition, got %d", len(first.EpochTransitions))
	}
	if memberMgr.GetCurrentConfig().ID != baseEpoch+1 {
		t.Fatalf("expected membership to advance to epoch %d, got %d", baseEpoch+1, memberMgr.GetCurrentConfig().ID)
	}
	if _, ok := memberMgr.GetCurrentConfig().Validators[0]; ok {
		t.Fatal("expected committed leave to remove validator 0")
	}
	if pending := hydraMgr.TempConfigManager.GetPendingLeaves(); len(pending) != 0 {
		t.Fatalf("expected committed install path to clear pending leaves, got %+v", pending)
	}
	if reconfigAuthorizedByHydraPendingIntent(&leave, hydraMgr) {
		t.Fatal("expected committed install path to clear helper-level leave authorization")
	}

	replayBlock := types.NewBlock(2, nil, txPayload, 2, baseEpoch, 0, 0, nil, nil)
	replay, err := applyOrderedOutput(hotstuff.InstanceOutput{
		InstanceID:  0,
		LocalHeight: replayBlock.Height,
		Rank:        31,
		BlockHash:   replayBlock.Hash,
		Block:       replayBlock,
	}, exec)
	if err != nil {
		t.Fatalf("apply replayed leave block: %v", err)
	}
	if len(replay.EpochTransitions) != 0 {
		t.Fatalf("expected replayed manual leave to be rejected after committed install, got %d transitions", len(replay.EpochTransitions))
	}
	if memberMgr.GetCurrentConfig().ID != baseEpoch+1 {
		t.Fatalf("expected replay to leave membership at epoch %d, got %d", baseEpoch+1, memberMgr.GetCurrentConfig().ID)
	}
	if _, ok := memberMgr.GetCurrentConfig().Validators[0]; ok {
		t.Fatal("expected replay to keep validator 0 removed")
	}
	if pending := exec.PendingReconfigCount(); pending != 0 {
		t.Fatalf("expected replay rejection to leave no queued reconfigs, got %d", pending)
	}
}

func TestBuildAdminMux_JoinRollsBackHydraPendingIntentWhenEngineMissing(t *testing.T) {
	engine := buildAdaptiveEngineForTest(t)
	memberMgr := membership.NewMembershipManager(engine.GetCurrentValidatorSet().Validators)
	hydraValidators := make(map[uint64]*hydra.Validator, len(engine.GetCurrentValidatorSet().Validators))
	for id, v := range engine.GetCurrentValidatorSet().Validators {
		copyVal := *v
		hydraValidators[id] = &hydra.Validator{ID: copyVal.ID, PublicKey: copyVal.PublicKey, Power: copyVal.Power, IsActive: copyVal.IsActive}
	}
	hydraMgr, err := hydra.NewHydraManager(4, hydraValidators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	kp, err := crypto.GenerateKeyPair()
	if err != nil {
		t.Fatalf("generate keypair: %v", err)
	}
	handler := buildAdminMux(
		4,
		&types.Keypair{PublicKey: kp.PublicKey, PrivateKey: kp.PrivateKey},
		nil,
		memberMgr,
		nil,
		hotstuff.NewGlobalOrderer(1, 50*time.Millisecond),
		hotstuff.NewGlobalConfirmedMetrics(50*time.Millisecond),
		nil,
		hydraMgr,
		nil,
		&evolvbftAdaptiveRuntime{},
		nil,
		false,
	)

	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, httptest.NewRequest(http.MethodPost, "/join", nil))
	if resp.Code != http.StatusInternalServerError {
		t.Fatalf("expected POST /join to return 500 when engine is missing, got %d", resp.Code)
	}
	if pending := hydraMgr.TempConfigManager.GetPendingJoins(); len(pending) != 0 {
		t.Fatalf("expected hydra pending join rollback when engine is missing, got %+v", pending)
	}
}

func TestBuildAdminMux_LeaveRollsBackHydraPendingIntentWhenEngineMissing(t *testing.T) {
	engine := buildAdaptiveEngineForTest(t)
	memberMgr := membership.NewMembershipManager(engine.GetCurrentValidatorSet().Validators)
	hydraValidators := make(map[uint64]*hydra.Validator, len(engine.GetCurrentValidatorSet().Validators))
	for id, v := range engine.GetCurrentValidatorSet().Validators {
		copyVal := *v
		hydraValidators[id] = &hydra.Validator{ID: copyVal.ID, PublicKey: copyVal.PublicKey, Power: copyVal.Power, IsActive: copyVal.IsActive}
	}
	hydraMgr, err := hydra.NewHydraManager(0, hydraValidators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	kp, err := crypto.GenerateKeyPair()
	if err != nil {
		t.Fatalf("generate keypair: %v", err)
	}
	handler := buildAdminMux(
		0,
		&types.Keypair{PublicKey: kp.PublicKey, PrivateKey: kp.PrivateKey},
		nil,
		memberMgr,
		nil,
		hotstuff.NewGlobalOrderer(1, 50*time.Millisecond),
		hotstuff.NewGlobalConfirmedMetrics(50*time.Millisecond),
		nil,
		hydraMgr,
		nil,
		&evolvbftAdaptiveRuntime{},
		nil,
		false,
	)

	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, httptest.NewRequest(http.MethodPost, "/leave", nil))
	if resp.Code != http.StatusInternalServerError {
		t.Fatalf("expected POST /leave to return 500 when engine is missing, got %d", resp.Code)
	}
	if pending := hydraMgr.TempConfigManager.GetPendingLeaves(); len(pending) != 0 {
		t.Fatalf("expected hydra pending leave rollback when engine is missing, got %+v", pending)
	}
}

func TestNewAdminHTTPServer_ConfiguresTimeouts(t *testing.T) {
	handler := http.NewServeMux()
	srv := newAdminHTTPServer("127.0.0.1:8080", handler)
	if srv.Addr != "127.0.0.1:8080" {
		t.Fatalf("unexpected server addr: %q", srv.Addr)
	}
	if srv.Handler != handler {
		t.Fatalf("expected server handler to be preserved")
	}
	if srv.ReadHeaderTimeout != adminReadHeaderTimeout {
		t.Fatalf("unexpected ReadHeaderTimeout: %s", srv.ReadHeaderTimeout)
	}
	if srv.ReadTimeout != adminReadTimeout {
		t.Fatalf("unexpected ReadTimeout: %s", srv.ReadTimeout)
	}
	if srv.WriteTimeout != adminWriteTimeout {
		t.Fatalf("unexpected WriteTimeout: %s", srv.WriteTimeout)
	}
	if srv.IdleTimeout != adminIdleTimeout {
		t.Fatalf("unexpected IdleTimeout: %s", srv.IdleTimeout)
	}
}

func TestBuildAdminMux_HydraEndpointUsesCommittedConfigForCurrentAndHydraForHighestKnown(t *testing.T) {
	engine := buildAdaptiveEngineForTest(t)
	memberMgr := membership.NewMembershipManager(engine.GetCurrentValidatorSet().Validators)
	validators := make(map[uint64]*hydra.Validator, len(engine.GetCurrentValidatorSet().Validators))
	for id, v := range engine.GetCurrentValidatorSet().Validators {
		copyVal := *v
		validators[id] = &hydra.Validator{ID: copyVal.ID, PublicKey: copyVal.PublicKey, Power: copyVal.Power, IsActive: copyVal.IsActive}
	}
	hydraMgr, err := hydra.NewHydraManager(0, validators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	if err := hydraMgr.InstallCommittedConfiguration(&types.Configuration{ID: 1, Validators: engine.GetCurrentValidatorSet().Validators, QuorumSize: 3}); err != nil {
		t.Fatalf("install committed config: %v", err)
	}
	if err := hydraMgr.DiscoveryManager.AddConfiguration(&hydra.Configuration{ID: 2, Validators: validators, QuorumSize: 3}); err != nil {
		t.Fatalf("add discovered config: %v", err)
	}
	handler := buildAdminMux(
		0,
		&types.Keypair{PublicKey: engine.GetCurrentValidatorSet().Validators[0].PublicKey},
		engine,
		memberMgr,
		[]*hotstuff.Engine{engine},
		hotstuff.NewGlobalOrderer(1, 50*time.Millisecond),
		hotstuff.NewGlobalConfirmedMetrics(50*time.Millisecond),
		nil,
		hydraMgr,
		nil,
		&evolvbftAdaptiveRuntime{},
		nil,
		false,
	)

	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, httptest.NewRequest(http.MethodGet, "/hydra", nil))
	if resp.Code != http.StatusOK {
		t.Fatalf("expected GET /hydra to succeed, got %d", resp.Code)
	}
	var body struct {
		ConfigID             uint64 `json:"config_id"`
		HighestKnownConfigID uint64 `json:"highest_known_config_id"`
	}
	if err := json.Unmarshal(resp.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode hydra response: %v", err)
	}
	if body.ConfigID != 1 {
		t.Fatalf("expected committed config_id 1, got %d", body.ConfigID)
	}
	if body.HighestKnownConfigID != 2 {
		t.Fatalf("expected highest_known_config_id 2, got %d", body.HighestKnownConfigID)
	}
}

func TestBuildAdminMux_HydraEndpointToleratesMissingHydraSubcomponents(t *testing.T) {
	engine := buildAdaptiveEngineForTest(t)
	memberMgr := membership.NewMembershipManager(engine.GetCurrentValidatorSet().Validators)
	hydraMgr, err := hydra.NewHydraManager(0, map[uint64]*hydra.Validator{
		0: {ID: 0, PublicKey: engine.GetCurrentValidatorSet().Validators[0].PublicKey, Power: 1, IsActive: true},
	}, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	hydraMgr.LSetManager = nil
	hydraMgr.TempConfigManager = nil

	handler := buildAdminMux(
		0,
		&types.Keypair{PublicKey: engine.GetCurrentValidatorSet().Validators[0].PublicKey},
		engine,
		memberMgr,
		[]*hotstuff.Engine{engine},
		hotstuff.NewGlobalOrderer(1, 50*time.Millisecond),
		hotstuff.NewGlobalConfirmedMetrics(50*time.Millisecond),
		nil,
		hydraMgr,
		nil,
		&evolvbftAdaptiveRuntime{},
		nil,
		false,
	)

	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, httptest.NewRequest(http.MethodGet, "/hydra", nil))
	if resp.Code != http.StatusOK {
		t.Fatalf("expected GET /hydra to succeed, got %d", resp.Code)
	}
	var body struct {
		CanParticipate bool `json:"can_participate"`
		PendingJoins   int  `json:"pending_joins"`
		PendingLeaves  int  `json:"pending_leaves"`
		LSetSize       int  `json:"lset_size"`
	}
	if err := json.Unmarshal(resp.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode hydra response: %v", err)
	}
	if body.PendingJoins != 0 || body.PendingLeaves != 0 || body.LSetSize != 0 {
		t.Fatalf("expected zeroed hydra subcomponent stats when managers are nil, got %+v", body)
	}
	if !body.CanParticipate {
		t.Fatalf("expected can_participate to remain available when hydra subcomponents are nil")
	}
}

func TestBuildAdminMux_AdaptiveContextPostRoundTrips(t *testing.T) {
	runtime := &evolvbftAdaptiveRuntime{}
	engine := buildAdaptiveEngineForTest(t)
	memberMgr := membership.NewMembershipManager(engine.GetCurrentValidatorSet().Validators)
	handler := buildAdminMux(
		0,
		&types.Keypair{PublicKey: engine.GetCurrentValidatorSet().Validators[0].PublicKey},
		engine,
		memberMgr,
		[]*hotstuff.Engine{engine},
		hotstuff.NewGlobalOrderer(1, 50*time.Millisecond),
		hotstuff.NewGlobalConfirmedMetrics(50*time.Millisecond),
		nil,
		nil,
		nil,
		runtime,
		nil,
		false,
	)

	body := `{"heterogeneity_score":0.7,"churn_rate":0.2,"adversary_score":0.4,"network_jitter_ms":15,"ai_load_score":0.5}`
	req := httptest.NewRequest(http.MethodPost, "/adaptive/context", strings.NewReader(body))
	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, req)
	if resp.Code != http.StatusOK {
		t.Fatalf("expected POST /adaptive/context to succeed, got %d", resp.Code)
	}

	got := runtime.GetScenarioContext()
	if got.HeterogeneityScore != 0.7 || got.ChurnRate != 0.2 || got.AdversaryScore != 0.4 || got.NetworkJitterMs != 15 || got.AILoadScore != 0.5 {
		t.Fatalf("unexpected adaptive context after POST: %+v", got)
	}
}

func TestBuildAdminMux_AdaptiveGetReturnsTruthfulContractWithoutController(t *testing.T) {
	runtime := &evolvbftAdaptiveRuntime{}
	runtime.SetScenarioContext(adaptive.ScenarioContext{
		HeterogeneityScore: 0.6,
		ChurnRate:          0.2,
		AdversaryScore:     0.1,
		NetworkJitterMs:    12,
		AILoadScore:        0.4,
	})
	engine := buildAdaptiveEngineForTest(t)
	memberMgr := membership.NewMembershipManager(engine.GetCurrentValidatorSet().Validators)
	handler := buildAdminMux(
		0,
		&types.Keypair{PublicKey: engine.GetCurrentValidatorSet().Validators[0].PublicKey},
		engine,
		memberMgr,
		[]*hotstuff.Engine{engine},
		hotstuff.NewGlobalOrderer(1, 50*time.Millisecond),
		hotstuff.NewGlobalConfirmedMetrics(50*time.Millisecond),
		nil,
		nil,
		nil,
		runtime,
		nil,
		false,
	)

	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, httptest.NewRequest(http.MethodGet, "/adaptive", nil))
	if resp.Code != http.StatusOK {
		t.Fatalf("expected GET /adaptive to succeed, got %d", resp.Code)
	}
	if got := resp.Header().Get("Content-Type"); !strings.Contains(got, "application/json") {
		t.Fatalf("expected JSON content type, got %q", got)
	}

	var body struct {
		Enabled         bool                     `json:"enabled"`
		SchemaVersion   string                   `json:"schema_version"`
		Schema          map[string]any           `json:"schema"`
		HasLastDecision bool                     `json:"has_last_decision"`
		LastDecision    adaptive.Decision        `json:"last_decision"`
		ClaimBoundary         string                       `json:"claim_boundary"`
		OrganizationSemantics adaptive.OrganizationSemantics `json:"organization_semantics"`
		Context               adaptive.ScenarioContext     `json:"context"`
	}
	if err := json.Unmarshal(resp.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode adaptive response: %v", err)
	}
	if body.Enabled {
		t.Fatalf("expected adaptive to report disabled when controller is nil")
	}
	if got := body.SchemaVersion; got != adaptive.SchemaVersion {
		t.Fatalf("unexpected schema version: %q", got)
	}
	if got := body.Schema["schema_version"]; got != adaptive.SchemaVersion {
		t.Fatalf("expected schema snapshot to carry schema_version, got %v", got)
	}
	if got := body.Schema["decision_fields"]; !reflect.DeepEqual(normalizeJSONValue(got), normalizeJSONValue(adaptive.SchemaSnapshot()["decision_fields"])) {
		t.Fatalf("unexpected decision_fields snapshot: %v", got)
	}
	if got := body.Schema["admin_response_fields"]; !reflect.DeepEqual(normalizeJSONValue(got), normalizeJSONValue([]string{"enabled", "schema_version", "schema", "has_last_decision", "last_decision", "claim_boundary", "organization_semantics", "context"})) {
		t.Fatalf("unexpected admin_response_fields snapshot: %v", got)
	}
	if got := body.Schema["organization_semantics_fields"]; !reflect.DeepEqual(normalizeJSONValue(got), normalizeJSONValue([]string{"status", "claim_boundary"})) {
		t.Fatalf("unexpected organization_semantics_fields snapshot: %v", got)
	}
	if got := body.Schema["decision_action_stage_fields"]; !reflect.DeepEqual(normalizeJSONValue(got), normalizeJSONValue(adaptive.SchemaSnapshot()["decision_action_stage_fields"])) {
		t.Fatalf("unexpected decision_action_stage_fields snapshot: %v", got)
	}
	if got := body.Schema["trace_fields"]; !reflect.DeepEqual(normalizeJSONValue(got), normalizeJSONValue(adaptive.SchemaSnapshot()["trace_fields"])) {
		t.Fatalf("unexpected trace_fields snapshot: %v", got)
	}
	if body.Context != runtime.GetScenarioContext() {
		t.Fatalf("unexpected adaptive context: %+v", body.Context)
	}
	if body.HasLastDecision {
		t.Fatalf("expected has_last_decision=false when controller is nil")
	}
	if body.ClaimBoundary == "" {
		t.Fatalf("expected claim_boundary to be present")
	}
	if !strings.Contains(body.ClaimBoundary, "sanitized SFAC actions") || !strings.Contains(body.ClaimBoundary, "not trust-estimator authority") {
		t.Fatalf("unexpected claim_boundary: %q", body.ClaimBoundary)
	}
	if body.OrganizationSemantics.Status != adaptive.OrganizationSemanticsAbsent {
		t.Fatalf("expected organization semantics to be absent, got %+v", body.OrganizationSemantics)
	}
	if !strings.Contains(body.OrganizationSemantics.ClaimBoundary, "not equivalent to MOISE+ role decomposition") {
		t.Fatalf("unexpected organization semantics boundary: %q", body.OrganizationSemantics.ClaimBoundary)
	}
	if !reflect.DeepEqual(body.LastDecision, adaptive.Decision{}) {
		t.Fatalf("expected zero last decision when controller is nil, got %+v", body.LastDecision)
	}
}

func TestBuildAdminMux_AdaptiveGetReturnsTruthfulContractBeforeFirstDecision(t *testing.T) {
	engine := buildAdaptiveEngineForTest(t)
	memberMgr := membership.NewMembershipManager(engine.GetCurrentValidatorSet().Validators)
	runtime := &evolvbftAdaptiveRuntime{}
	controller := adaptive.NewController(adaptive.Config{Enabled: true, Interval: time.Second}, runtime, runtime, adaptive.SafeBaselinePolicy{}, adaptive.DefaultGuardrails())

	handler := buildAdminMux(
		0,
		&types.Keypair{PublicKey: engine.GetCurrentValidatorSet().Validators[0].PublicKey},
		engine,
		memberMgr,
		[]*hotstuff.Engine{engine},
		hotstuff.NewGlobalOrderer(1, 50*time.Millisecond),
		hotstuff.NewGlobalConfirmedMetrics(50*time.Millisecond),
		nil,
		nil,
		controller,
		runtime,
		nil,
		false,
	)

	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, httptest.NewRequest(http.MethodGet, "/adaptive", nil))
	if resp.Code != http.StatusOK {
		t.Fatalf("expected GET /adaptive to succeed, got %d", resp.Code)
	}

	var body struct {
		Enabled         bool                     `json:"enabled"`
		SchemaVersion   string                   `json:"schema_version"`
		Schema          map[string]any           `json:"schema"`
		HasLastDecision bool                     `json:"has_last_decision"`
		LastDecision    adaptive.Decision        `json:"last_decision"`
		ClaimBoundary         string                       `json:"claim_boundary"`
		OrganizationSemantics adaptive.OrganizationSemantics `json:"organization_semantics"`
		Context               adaptive.ScenarioContext     `json:"context"`
	}
	if err := json.Unmarshal(resp.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode adaptive response: %v", err)
	}
	if !body.Enabled {
		t.Fatalf("expected adaptive to report enabled when controller is present")
	}
	if body.HasLastDecision {
		t.Fatalf("expected has_last_decision=false before first decision")
	}
	if body.ClaimBoundary == "" {
		t.Fatalf("expected claim_boundary to be present")
	}
	if body.OrganizationSemantics.Status != adaptive.OrganizationSemanticsAbsent {
		t.Fatalf("expected organization semantics to be absent before first decision, got %+v", body.OrganizationSemantics)
	}
	if !reflect.DeepEqual(body.LastDecision, adaptive.Decision{}) {
		t.Fatalf("expected zero last decision before first tick, got %+v", body.LastDecision)
	}
}

func TestBuildAdminMux_AdaptiveGetReturnsStageDecisionAndRedactedBoundary(t *testing.T) {
	engine := buildAdaptiveEngineForTest(t)
	memberMgr := membership.NewMembershipManager(engine.GetCurrentValidatorSet().Validators)
	runtime := &evolvbftAdaptiveRuntime{}
	runtime.SetScenarioContext(adaptive.ScenarioContext{
		HeterogeneityScore: 0.8,
		ChurnRate:          0.3,
		AdversaryScore:     0.2,
		NetworkJitterMs:    18,
		AILoadScore:        0.6,
	})
	controller := adaptive.NewController(adaptive.Config{Enabled: true, Interval: time.Second}, runtime, runtime, adaptive.SafeBaselinePolicy{}, adaptive.DefaultGuardrails())
	decision := adaptive.Decision{
		Timestamp:  time.Unix(1710000000, 0).UTC(),
		PolicyName: "safe-baseline",
		Observation: adaptive.Observation{
			ValidatorCount:     4,
			CommitteeSize:      4,
			PacemakerTimeoutMs: 1000,
			Agents: []adaptive.AgentObservation{{
				InstanceID:                0,
				Epoch:                     2,
				ValidatorCount:            4,
				CommitteeSize:             4,
				PacemakerTimeoutMs:        900,
				MempoolMaxBatchTxs:        32,
				MempoolProposalIntervalMs: 40,
			}},
			TrustSnapshots: []adaptive.TrustSnapshot{{
				NodeID:             1,
				SampleCount:        5,
				SuccessRate:        0.8,
				FailureProbability: 0.2,
			}},
		},
		Candidate: adaptive.DecisionActionStage{
			Action: adaptive.Action{
				CommitteeSize:             5,
				PacemakerTimeoutMs:        900,
				MempoolMaxBatchTxs:        64,
				MempoolProposalIntervalMs: 50,
				Reason:                    "policy-proposal",
				AgentActions: []adaptive.AgentAction{{
					InstanceID:                0,
					CommitteeSize:             5,
					PacemakerTimeoutMs:        900,
					MempoolMaxBatchTxs:        64,
					MempoolProposalIntervalMs: 50,
				}},
			},
			Present:       true,
			Reason:        "candidate-stage",
			BlockedFields: []string{"submit_join"},
			Notes:         []string{"candidate-note"},
		},
		Governed: adaptive.DecisionActionStage{
			Action: adaptive.Action{
				CommitteeSize:             4,
				PacemakerTimeoutMs:        1000,
				MempoolMaxBatchTxs:        48,
				MempoolProposalIntervalMs: 60,
			},
			Present:       true,
			Mutated:       true,
			BlockedFields: []string{"committee_size"},
			Notes:         []string{"governed-note"},
		},
		Masked: adaptive.DecisionActionStage{
			Action: adaptive.Action{
				CommitteeSize:             4,
				PacemakerTimeoutMs:        1100,
				MempoolMaxBatchTxs:        48,
				MempoolProposalIntervalMs: 60,
			},
			Present: true,
			Mutated: true,
		},
		Applied: adaptive.DecisionActionStage{
			Action: adaptive.Action{
				CommitteeSize:             4,
				PacemakerTimeoutMs:        1100,
				MempoolMaxBatchTxs:        48,
				MempoolProposalIntervalMs: 60,
			},
			Present: true,
		},
		Reward:     1.5,
		TeamReward: 1.25,
		RoleRewards: map[string]float64{
			"lane_tuner": 0.7,
		},
		Trace: adaptive.TraceStatus{
			Enabled:        true,
			WriteFailed:    true,
			CloseFailed:    false,
			DroppedSamples: 2,
		},
	}
	setAdaptiveControllerLastDecision(t, controller, decision)

	handler := buildAdminMux(
		0,
		&types.Keypair{PublicKey: engine.GetCurrentValidatorSet().Validators[0].PublicKey},
		engine,
		memberMgr,
		[]*hotstuff.Engine{engine},
		hotstuff.NewGlobalOrderer(1, 50*time.Millisecond),
		hotstuff.NewGlobalConfirmedMetrics(50*time.Millisecond),
		nil,
		nil,
		controller,
		runtime,
		nil,
		false,
	)

	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, httptest.NewRequest(http.MethodGet, "/adaptive", nil))
	if resp.Code != http.StatusOK {
		t.Fatalf("expected GET /adaptive to succeed, got %d", resp.Code)
	}

	var body struct {
		Enabled         bool                     `json:"enabled"`
		SchemaVersion   string                   `json:"schema_version"`
		Schema          map[string]any           `json:"schema"`
		HasLastDecision bool                     `json:"has_last_decision"`
		LastDecision    adaptive.Decision        `json:"last_decision"`
		ClaimBoundary         string                       `json:"claim_boundary"`
		OrganizationSemantics adaptive.OrganizationSemantics `json:"organization_semantics"`
		Context               adaptive.ScenarioContext     `json:"context"`
	}
	if err := json.Unmarshal(resp.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode adaptive response: %v", err)
	}
	if !body.Enabled {
		t.Fatalf("expected adaptive to report enabled when controller is present")
	}
	if got := body.SchemaVersion; got != adaptive.SchemaVersion {
		t.Fatalf("unexpected schema version: %q", got)
	}
	if got := body.Schema["schema_version"]; got != adaptive.SchemaVersion {
		t.Fatalf("expected schema snapshot to carry schema_version, got %v", got)
	}
	if got := body.Schema["decision_fields"]; !reflect.DeepEqual(normalizeJSONValue(got), normalizeJSONValue(adaptive.SchemaSnapshot()["decision_fields"])) {
		t.Fatalf("unexpected decision_fields snapshot: %v", got)
	}
	if got := body.Schema["admin_response_fields"]; !reflect.DeepEqual(normalizeJSONValue(got), normalizeJSONValue([]string{"enabled", "schema_version", "schema", "has_last_decision", "last_decision", "claim_boundary", "organization_semantics", "context"})) {
		t.Fatalf("unexpected admin_response_fields snapshot: %v", got)
	}
	if got := body.Schema["organization_semantics_fields"]; !reflect.DeepEqual(normalizeJSONValue(got), normalizeJSONValue([]string{"status", "claim_boundary"})) {
		t.Fatalf("unexpected organization_semantics_fields snapshot: %v", got)
	}
	if got := body.Schema["decision_action_stage_fields"]; !reflect.DeepEqual(normalizeJSONValue(got), normalizeJSONValue(adaptive.SchemaSnapshot()["decision_action_stage_fields"])) {
		t.Fatalf("unexpected decision_action_stage_fields snapshot: %v", got)
	}
	if got := body.Schema["trace_fields"]; !reflect.DeepEqual(normalizeJSONValue(got), normalizeJSONValue(adaptive.SchemaSnapshot()["trace_fields"])) {
		t.Fatalf("unexpected trace_fields snapshot: %v", got)
	}
	if body.Context != runtime.GetScenarioContext() {
		t.Fatalf("unexpected adaptive context: %+v", body.Context)
	}
	if !body.HasLastDecision {
		t.Fatalf("expected has_last_decision=true when controller has recorded a decision")
	}
	if body.ClaimBoundary == "" {
		t.Fatalf("expected claim_boundary to be present")
	}
	if body.OrganizationSemantics.Status != adaptive.OrganizationSemanticsAbsent {
		t.Fatalf("expected organization semantics to remain absent with a recorded decision, got %+v", body.OrganizationSemantics)
	}
	if !strings.Contains(body.OrganizationSemantics.ClaimBoundary, "not equivalent to MOISE+ role decomposition") {
		t.Fatalf("unexpected organization semantics boundary: %q", body.OrganizationSemantics.ClaimBoundary)
	}
	if !body.LastDecision.Candidate.Present || !body.LastDecision.Governed.Present || !body.LastDecision.Masked.Present || !body.LastDecision.Applied.Present {
		t.Fatalf("expected stage-based decision chain in last_decision, got %+v", body.LastDecision)
	}
	if body.LastDecision.Candidate.Action.Reason != "policy-proposal" {
		t.Fatalf("expected admin last_decision to retain candidate action reason, got %+v", body.LastDecision.Candidate.Action)
	}
	if len(body.LastDecision.Candidate.Action.AgentActions) != 1 {
		t.Fatalf("expected admin last_decision to retain agent_actions, got %+v", body.LastDecision.Candidate.Action.AgentActions)
	}
	if len(body.LastDecision.Observation.Agents) != 1 || len(body.LastDecision.Observation.TrustSnapshots) != 1 {
		t.Fatalf("expected admin last_decision to retain nested observation details, got %+v", body.LastDecision.Observation)
	}
	if body.LastDecision.Trace.Enabled != true || body.LastDecision.Trace.WriteFailed != true || body.LastDecision.Trace.CloseFailed != false || body.LastDecision.Trace.DroppedSamples != 2 {
		t.Fatalf("unexpected trace status in admin last_decision: %+v", body.LastDecision.Trace)
	}
	if body.LastDecision.RoleRewards["lane_tuner"] != 0.7 {
		t.Fatalf("unexpected role rewards in admin last_decision: %+v", body.LastDecision.RoleRewards)
	}
}

func TestBuildAdminMux_AdaptiveContextRejectsUnknownFields(t *testing.T) {
	runtime := &evolvbftAdaptiveRuntime{}
	engine := buildAdaptiveEngineForTest(t)
	memberMgr := membership.NewMembershipManager(engine.GetCurrentValidatorSet().Validators)
	handler := buildAdminMux(
		0,
		&types.Keypair{PublicKey: engine.GetCurrentValidatorSet().Validators[0].PublicKey},
		engine,
		memberMgr,
		[]*hotstuff.Engine{engine},
		hotstuff.NewGlobalOrderer(1, 50*time.Millisecond),
		hotstuff.NewGlobalConfirmedMetrics(50*time.Millisecond),
		nil,
		nil,
		nil,
		runtime,
		nil,
		false,
	)

	req := httptest.NewRequest(http.MethodPost, "/adaptive/context", strings.NewReader(`{"heterogeneity_score":0.5,"unexpected":1}`))
	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, req)
	if resp.Code != http.StatusBadRequest {
		t.Fatalf("expected POST /adaptive/context with unknown field to return 400, got %d", resp.Code)
	}
	if got := runtime.GetScenarioContext(); got != (adaptive.ScenarioContext{}) {
		t.Fatalf("unexpected adaptive context after invalid POST: %+v", got)
	}
}

func TestBuildAdminMux_AdaptiveContextRejectsInvalidValues(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{name: "negative heterogeneity", body: `{"heterogeneity_score":-0.1,"churn_rate":0.2,"adversary_score":0.4,"network_jitter_ms":15,"ai_load_score":0.5}`},
		{name: "churn too large", body: `{"heterogeneity_score":0.1,"churn_rate":1.1,"adversary_score":0.4,"network_jitter_ms":15,"ai_load_score":0.5}`},
		{name: "adversary too large", body: `{"heterogeneity_score":0.1,"churn_rate":0.2,"adversary_score":1.1,"network_jitter_ms":15,"ai_load_score":0.5}`},
		{name: "negative jitter", body: `{"heterogeneity_score":0.1,"churn_rate":0.2,"adversary_score":0.4,"network_jitter_ms":-1,"ai_load_score":0.5}`},
		{name: "jitter too large", body: `{"heterogeneity_score":0.1,"churn_rate":0.2,"adversary_score":0.4,"network_jitter_ms":60001,"ai_load_score":0.5}`},
		{name: "negative ai load", body: `{"heterogeneity_score":0.1,"churn_rate":0.2,"adversary_score":0.4,"network_jitter_ms":15,"ai_load_score":-0.1}`},
		{name: "missing field", body: `{"heterogeneity_score":0.1,"churn_rate":0.2,"adversary_score":0.4,"network_jitter_ms":15}`},
		{name: "null body", body: `null`},
		{name: "duplicate key", body: `{"heterogeneity_score":0.1,"heterogeneity_score":0.2,"churn_rate":0.2,"adversary_score":0.4,"network_jitter_ms":15,"ai_load_score":0.5}`},
		{name: "trailing json", body: `{"heterogeneity_score":0.1,"churn_rate":0.2,"adversary_score":0.4,"network_jitter_ms":15,"ai_load_score":0.5}{}`},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			runtime := &evolvbftAdaptiveRuntime{}
			engine := buildAdaptiveEngineForTest(t)
			memberMgr := membership.NewMembershipManager(engine.GetCurrentValidatorSet().Validators)
			handler := buildAdminMux(
				0,
				&types.Keypair{PublicKey: engine.GetCurrentValidatorSet().Validators[0].PublicKey},
				engine,
				memberMgr,
				[]*hotstuff.Engine{engine},
				hotstuff.NewGlobalOrderer(1, 50*time.Millisecond),
				hotstuff.NewGlobalConfirmedMetrics(50*time.Millisecond),
				nil,
				nil,
				nil,
				runtime,
				nil,
				false,
			)

			resp := httptest.NewRecorder()
			handler.ServeHTTP(resp, httptest.NewRequest(http.MethodPost, "/adaptive/context", strings.NewReader(tc.body)))
			if resp.Code != http.StatusBadRequest {
				t.Fatalf("expected invalid adaptive context to return 400, got %d", resp.Code)
			}
			if got := runtime.GetScenarioContext(); got != (adaptive.ScenarioContext{}) {
				t.Fatalf("unexpected adaptive context after invalid POST: %+v", got)
			}
		})
	}
}

func normalizeJSONValue(value any) any {
	encoded, err := json.Marshal(value)
	if err != nil {
		return value
	}
	var out any
	if err := json.Unmarshal(encoded, &out); err != nil {
		return value
	}
	return out
}

func setAdaptiveControllerLastDecision(t *testing.T, controller *adaptive.Controller, decision adaptive.Decision) {
	t.Helper()
	field := reflect.ValueOf(controller).Elem().FieldByName("last")
	reflect.NewAt(field.Type(), unsafe.Pointer(field.UnsafeAddr())).Elem().Set(reflect.ValueOf(decision))
}

func writeManifestForTest(t *testing.T, nodes ...bootstrap.ManifestNode) string {
	t.Helper()
	dir := t.TempDir()
	manifestPath := filepath.Join(dir, "genesis.json")
	payload, err := json.MarshalIndent(&bootstrap.GenesisManifest{
		Version: 1,
		Nodes:   nodes,
	}, "", "  ")
	if err != nil {
		t.Fatalf("marshal manifest: %v", err)
	}
	if err := os.WriteFile(manifestPath, payload, 0o600); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	return manifestPath
}

func makeManifestNodeForTest(t *testing.T, nodeID uint64) bootstrap.ManifestNode {
	t.Helper()
	seed := sha256.Sum256([]byte{byte(nodeID), byte(nodeID >> 8), 0xA5})
	priv := ed25519.NewKeyFromSeed(seed[:])
	pub := priv.Public().(ed25519.PublicKey)
	peerID, err := peerIDFromTestPublicKey(pub)
	if err != nil {
		t.Fatalf("peer id: %v", err)
	}
	vrfSeed := sha256.Sum256([]byte{0x56, 0x52, 0x46, byte(nodeID), byte(nodeID >> 8), 0xA5})
	vrfPriv := testManifestVRFSuite.Scalar().Pick(testManifestVRFSuite.XOF(vrfSeed[:]))
	vrfPub := testManifestVRFSuite.Point().Mul(vrfPriv, nil)
	vrfPrivRaw, err := vrfPriv.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal vrf private key: %v", err)
	}
	vrfPubRaw, err := vrfPub.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal vrf public key: %v", err)
	}
	return bootstrap.ManifestNode{
		ID:               nodeID,
		Power:            1,
		IsActive:         true,
		PublicKeyHex:     hex.EncodeToString(pub),
		PrivateKeyHex:    hex.EncodeToString(priv),
		VRFPublicKeyHex:  hex.EncodeToString(vrfPubRaw),
		VRFPrivateKeyHex: hex.EncodeToString(vrfPrivRaw),
		PeerID:           peerID.String(),
		P2PMultiaddr:     "/ip4/127.0.0.1/tcp/8080/p2p/" + peerID.String(),
	}
}

func buildHydraInjectionEngineForTest(t *testing.T, pubKey types.PublicKey) *hotstuff.Engine {
	t.Helper()
	validatorSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: pubKey, Power: 1, IsActive: true},
	})
	store := storage.NewStorageManager(uint64(time.Now().UnixNano()))
	t.Cleanup(func() {
		_ = store.Close()
	})
	return hotstuff.NewEngineWithInstanceAndOptions(
		0,
		&types.Keypair{PublicKey: pubKey, PrivateKey: types.PrivateKey(make([]byte, ed25519.PrivateKeySize))},
		validatorSet,
		nil,
		store,
		0,
		1,
		"test-consensus",
		nil,
		hotstuff.DefaultEngineOptions(),
	)
}

func hydraTransitionCallbackPointer(hydraMgr *hydra.HydraManager) uintptr {
	if hydraMgr == nil || hydraMgr.AutoTransitionManager == nil {
		return 0
	}
	atmValue := reflect.ValueOf(hydraMgr.AutoTransitionManager).Elem().FieldByName("onTransition")
	return atmValue.Pointer()
}

func hydraInjectionMempool(engine *hotstuff.Engine) *mempool.Mempool {
	if engine == nil {
		return nil
	}
	engineValue := reflect.ValueOf(engine).Elem().FieldByName("mempool")
	return *(**mempool.Mempool)(unsafe.Pointer(engineValue.UnsafeAddr()))
}

func drainSingleReconfigTx(t *testing.T, mp *mempool.Mempool) *types.Transaction {
	t.Helper()
	if mp == nil {
		return nil
	}
	mpValue := reflect.ValueOf(mp).Elem().FieldByName("txQueue")
	if mpValue.Len() == 0 {
		return nil
	}
	first := mpValue.Index(0)
	txField := first.Elem().FieldByName("tx")
	tx := *(**types.Transaction)(unsafe.Pointer(txField.UnsafeAddr()))
	if tx == nil {
		return nil
	}
	dup := &types.Transaction{Type: tx.Type, Payload: append([]byte(nil), tx.Payload...)}
	return dup
}

func peerIDFromTestPublicKey(pub ed25519.PublicKey) (peer.ID, error) {
	manifest := &bootstrap.GenesisManifest{
		Version: 1,
		Nodes: []bootstrap.ManifestNode{
			{
				ID:           0,
				IsActive:     true,
				Power:        1,
				PublicKeyHex: hex.EncodeToString(pub),
			},
		},
	}
	if err := manifest.Normalize(); err != nil {
		return "", err
	}
	return peer.Decode(manifest.Nodes[0].PeerID)
}
