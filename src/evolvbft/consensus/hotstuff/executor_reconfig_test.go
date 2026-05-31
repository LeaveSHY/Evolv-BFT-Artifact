package hotstuff

import (
	"bytes"
	"encoding/json"
	"sort"
	"sync"
	"sync/atomic"
	"testing"

	"evolvbft/evolvbft/crypto"
	"evolvbft/evolvbft/types"
)

// makeReconfigBlock creates a Block with a single ReconfigTx in the Data field
// (Path 2 in executor: raw Data with no Payload certs).
func makeReconfigBlock(t *testing.T, height uint64, hash []byte, reconfigData *types.ReconfigData, signer types.PrivateKey) *types.Block {
	t.Helper()
	if signer != nil {
		reconfigData.Sign(signer)
	}
	payload, err := json.Marshal(reconfigData)
	if err != nil {
		t.Fatalf("marshal reconfig data: %v", err)
	}
	tx := types.Transaction{Type: types.TxTypeReconfig, Payload: payload}
	data, err := json.Marshal(tx)
	if err != nil {
		t.Fatalf("marshal tx: %v", err)
	}
	return &types.Block{
		Height: height,
		Hash:   hash,
		Data:   data,
		Epoch:  1,
	}
}

func makeAutoLeaveProof(t *testing.T, view uint64, targetEpoch uint64, blockHash []byte, leaves []uint64, voters map[uint64]*crypto.Keypair) *types.HydraTransitionProof {
	t.Helper()
	sortedLeaves := append([]uint64(nil), leaves...)
	sort.Slice(sortedLeaves, func(i, j int) bool { return sortedLeaves[i] < sortedLeaves[j] })
	digest := autoLeaveProofDigest(view, sortedLeaves, blockHash, targetEpoch)
	autoVotes := make(map[uint64]*types.HydraAutoVote, len(voters))
	for voterID, kp := range voters {
		autoVotes[voterID] = &types.HydraAutoVote{
			SenderID:  voterID,
			Signature: crypto.Sign(digest, kp.PrivateKey),
			Digest:    append([]byte(nil), digest...),
		}
	}
	return &types.HydraTransitionProof{
		View:        view,
		NewConfigID: targetEpoch,
		Leaves:      sortedLeaves,
		BlockHash:   append([]byte(nil), blockHash...),
		AutoVotes:   autoVotes,
	}
}

type stubVertexResolver struct {
	mu       sync.RWMutex
	vertices map[types.Hash]*types.Vertex
}

func (s *stubVertexResolver) GetVertex(hash types.Hash) *types.Vertex {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.vertices[hash]
}

func (s *stubVertexResolver) setVertex(hash types.Hash, vertex *types.Vertex) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.vertices == nil {
		s.vertices = make(map[types.Hash]*types.Vertex)
	}
	s.vertices[hash] = vertex
}

func TestExecutorApplyReconfigAutoLeave(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	kp2, _ := crypto.GenerateKeyPair()
	kp3, _ := crypto.GenerateKeyPair()
	kp4, _ := crypto.GenerateKeyPair()
	blockHash := []byte("test-block-hash")
	// Need 5 validators so that after auto-leave n=4 >= 3f+1=4 (BFT guard)
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: kp2.PublicKey, Power: 1, IsActive: true},
		3: {ID: 3, PublicKey: kp3.PublicKey, Power: 1, IsActive: true},
		4: {ID: 4, PublicKey: kp4.PublicKey, Power: 1, IsActive: true},
	})
	exec := NewExecutor(valSet)

	// n=5 requires quorum of 4 signers for auto-leave proof
	block := makeReconfigBlock(t, 10, blockHash, &types.ReconfigData{
		Type:        types.ReconfigAutoLeave,
		NodeID:      2,
		Epoch:       1,
		TargetEpoch: 2,
		AutoLeaveProof: makeAutoLeaveProof(t, 17, 2, blockHash, []uint64{2}, map[uint64]*crypto.Keypair{
			0: kp0,
			1: kp1,
			3: kp3,
			4: kp4,
		}),
	}, nil)

	if err := exec.ExecuteBlock(block); err != nil {
		t.Fatalf("execute block: %v", err)
	}

	transitions, err := exec.CommitReconfigs(block, 100)
	if err != nil {
		t.Fatalf("commit reconfigs: %v", err)
	}
	if len(transitions) != 1 {
		t.Fatalf("expected 1 transition, got %d", len(transitions))
	}

	tr := transitions[0]
	if tr.OldEpoch != 1 || tr.NewEpoch != 2 {
		t.Fatalf("expected epoch 1→2, got %d→%d", tr.OldEpoch, tr.NewEpoch)
	}
	if len(tr.Removed) != 1 || tr.Removed[0] != 2 {
		t.Fatalf("expected node 2 removed, got %v", tr.Removed)
	}

	// Verify validator set was updated
	exec.mu.RLock()
	_, stillPresent := exec.currentValSet.Validators[2]
	numValidators := len(exec.currentValSet.Validators)
	exec.mu.RUnlock()

	if stillPresent {
		t.Fatalf("node 2 should be removed from validator set")
	}
	if numValidators != 4 {
		t.Fatalf("expected 4 validators, got %d", numValidators)
	}
}

func TestExecutorAutoLeaveAndManualLeaveBehaveIdentically(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	kp2, _ := crypto.GenerateKeyPair()
	kp3, _ := crypto.GenerateKeyPair()
	kp4, _ := crypto.GenerateKeyPair()
	// Need 5 validators so that after leave n=4 >= 3f+1=4 (BFT guard)
	makeValSet := func() *types.ValidatorSet {
		return types.NewValidatorSet(1, map[uint64]*types.Validator{
			0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
			1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
			2: {ID: 2, PublicKey: kp2.PublicKey, Power: 1, IsActive: true},
			3: {ID: 3, PublicKey: kp3.PublicKey, Power: 1, IsActive: true},
			4: {ID: 4, PublicKey: kp4.PublicKey, Power: 1, IsActive: true},
		})
	}

	// Manual leave
	execManual := NewExecutor(makeValSet())
	execManual.SetReconfigAuthorizer(func(data *types.ReconfigData) bool {
		return data != nil && data.Type == types.ReconfigLeave && data.NodeID == 2
	})
	manualBlock := makeReconfigBlock(t, 10, []byte("manual-block"), &types.ReconfigData{
		Type: types.ReconfigLeave, NodeID: 2, PublicKey: kp2.PublicKey, Epoch: 1, TargetEpoch: 2,
	}, kp2.PrivateKey)
	if err := execManual.ExecuteBlock(manualBlock); err != nil {
		t.Fatalf("execute manual block: %v", err)
	}
	manualTr, err := execManual.CommitReconfigs(manualBlock, 100)
	if err != nil {
		t.Fatalf("commit manual reconfigs: %v", err)
	}

	// Auto leave
	execAuto := NewExecutor(makeValSet())
	autoBlockHash := []byte("auto-block")
	autoBlock := makeReconfigBlock(t, 10, autoBlockHash, &types.ReconfigData{
		Type:        types.ReconfigAutoLeave,
		NodeID:      2,
		Epoch:       1,
		TargetEpoch: 2,
		AutoLeaveProof: makeAutoLeaveProof(t, 18, 2, autoBlockHash, []uint64{2}, map[uint64]*crypto.Keypair{
			0: kp0,
			1: kp1,
			3: kp3,
			4: kp4,
		}),
	}, nil)
	if err := execAuto.ExecuteBlock(autoBlock); err != nil {
		t.Fatalf("execute auto block: %v", err)
	}
	autoTr, err := execAuto.CommitReconfigs(autoBlock, 100)
	if err != nil {
		t.Fatalf("commit auto reconfigs: %v", err)
	}

	if len(manualTr) != 1 || len(autoTr) != 1 {
		t.Fatalf("both should produce 1 transition: manual=%d auto=%d", len(manualTr), len(autoTr))
	}

	if manualTr[0].NewEpoch != autoTr[0].NewEpoch {
		t.Fatalf("epoch mismatch: manual=%d auto=%d", manualTr[0].NewEpoch, autoTr[0].NewEpoch)
	}
	if manualTr[0].QuorumSize != autoTr[0].QuorumSize {
		t.Fatalf("quorum mismatch: manual=%d auto=%d", manualTr[0].QuorumSize, autoTr[0].QuorumSize)
	}

	execManual.mu.RLock()
	execAuto.mu.RLock()
	manualHash := execManual.currentValSet.Hash()
	autoHash := execAuto.currentValSet.Hash()
	execManual.mu.RUnlock()
	execAuto.mu.RUnlock()

	if !bytes.Equal(manualHash, autoHash) {
		t.Fatalf("validator set hash should be identical after manual vs auto leave")
	}
}

func TestExecutorCommitReconfigs_RetryDoesNotPublishStateOnCallbackFailure(t *testing.T) {
	kp1, _ := crypto.GenerateKeyPair()
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: []byte("pk0"), Power: 1, IsActive: true},
	})
	exec := NewExecutor(valSet)
	exec.SetReconfigAuthorizer(func(data *types.ReconfigData) bool { return data != nil && data.Type == types.ReconfigJoin && data.NodeID == 1 })
	block := makeReconfigBlock(t, 10, []byte("retry-block"), &types.ReconfigData{
		Type:        types.ReconfigJoin,
		NodeID:      1,
		PublicKey:   kp1.PublicKey,
		Power:       1,
		Epoch:       1,
		TargetEpoch: 2,
	}, kp1.PrivateKey)
	if err := exec.ExecuteBlock(block); err != nil {
		t.Fatalf("execute block: %v", err)
	}

	var failOnce bool = true
	exec.SetEpochChangeCallback(func(newValSet *types.ValidatorSet, transitions []*types.EpochTransition) error {
		if failOnce {
			failOnce = false
			return bytes.ErrTooLarge
		}
		return nil
	})

	if got, err := exec.CommitReconfigs(block, 100); err == nil || len(got) != 0 {
		t.Fatalf("expected first commit to fail, got transitions=%v err=%v", got, err)
	}
	if exec.GetCurrentValidatorSet().Epoch != 1 {
		t.Fatalf("validator set should remain unpublished after callback failure, got epoch %d", exec.GetCurrentValidatorSet().Epoch)
	}
	if exec.PendingReconfigCount() != 1 {
		t.Fatalf("expected pending reconfig retained after callback failure, got %d", exec.PendingReconfigCount())
	}

	got, err := exec.CommitReconfigs(block, 100)
	if err != nil {
		t.Fatalf("retry commit reconfigs: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("expected exactly one transition after retry, got %d", len(got))
	}
	if exec.GetCurrentValidatorSet().Epoch != 2 {
		t.Fatalf("expected validator set epoch 2 after successful retry, got %d", exec.GetCurrentValidatorSet().Epoch)
	}
	if exec.PendingReconfigCount() != 0 {
		t.Fatalf("expected no pending reconfigs after successful retry, got %d", exec.PendingReconfigCount())
	}
}

func TestExecutorCommitReconfigs_DoesNotMarkUnexecutedBlockApplied(t *testing.T) {
	kp1, _ := crypto.GenerateKeyPair()
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: []byte("pk0"), Power: 1, IsActive: true},
	})
	exec := NewExecutor(valSet)
	exec.SetReconfigAuthorizer(func(data *types.ReconfigData) bool { return data != nil && data.Type == types.ReconfigJoin && data.NodeID == 1 })
	block := makeReconfigBlock(t, 10, []byte("commit-before-exec"), &types.ReconfigData{
		Type:        types.ReconfigJoin,
		NodeID:      1,
		PublicKey:   kp1.PublicKey,
		Power:       1,
		Epoch:       1,
		TargetEpoch: 2,
	}, kp1.PrivateKey)

	transitions, err := exec.CommitReconfigs(block, 100)
	if err != nil {
		t.Fatalf("commit before execute: %v", err)
	}
	if len(transitions) != 0 {
		t.Fatalf("expected no transitions before execution, got %d", len(transitions))
	}

	transitions, err = exec.ApplyOrderedBlock(block, 100)
	if err != nil {
		t.Fatalf("apply ordered block after premature commit attempt: %v", err)
	}
	if len(transitions) != 1 {
		t.Fatalf("expected one transition after ordered apply, got %d", len(transitions))
	}
	if exec.GetCurrentValidatorSet().Epoch != 2 {
		t.Fatalf("expected epoch 2 after ordered apply, got %d", exec.GetCurrentValidatorSet().Epoch)
	}
}

func TestExecutorCommitReconfigs_MatchesCommittedBlockByHeightAndHash(t *testing.T) {
	kp1, _ := crypto.GenerateKeyPair()
	kp2, _ := crypto.GenerateKeyPair()
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: []byte("pk0"), Power: 1, IsActive: true},
	})
	exec := NewExecutor(valSet)
	exec.SetReconfigAuthorizer(func(data *types.ReconfigData) bool {
		return data != nil && data.Type == types.ReconfigJoin && (data.NodeID == 1 || data.NodeID == 2)
	})

	blockA := makeReconfigBlock(t, 10, []byte("block-a"), &types.ReconfigData{
		Type:         types.ReconfigJoin,
		NodeID:       1,
		PublicKey:    kp1.PublicKey,
		VRFPublicKey: []byte("vrf1"),
		Power:        1,
		Epoch:        1,
		TargetEpoch:  2,
	}, kp1.PrivateKey)
	blockB := makeReconfigBlock(t, 10, []byte("block-b"), &types.ReconfigData{
		Type:        types.ReconfigJoin,
		NodeID:      2,
		PublicKey:   kp2.PublicKey,
		Power:       1,
		Epoch:       1,
		TargetEpoch: 3,
	}, kp2.PrivateKey)

	if err := exec.ExecuteBlock(blockA); err != nil {
		t.Fatalf("execute block A: %v", err)
	}
	if err := exec.ExecuteBlock(blockB); err != nil {
		t.Fatalf("execute block B: %v", err)
	}

	first, err := exec.CommitReconfigs(blockA, 100)
	if err != nil {
		t.Fatalf("commit block A reconfigs: %v", err)
	}
	if len(first) != 1 {
		t.Fatalf("expected exactly one transition for block A, got %d", len(first))
	}
	if len(first[0].Added) != 1 || first[0].Added[0] != 1 {
		t.Fatalf("unexpected first transition: %+v", first[0])
	}
	if exec.GetCurrentValidatorSet().Epoch != 2 {
		t.Fatalf("unexpected epoch after block A commit: %d", exec.GetCurrentValidatorSet().Epoch)
	}
	if exec.GetCurrentValidatorSet().TotalPower != 2 {
		t.Fatalf("unexpected total power after block A commit: %d", exec.GetCurrentValidatorSet().TotalPower)
	}
	if got := string(exec.GetCurrentValidatorSet().Validators[1].VRFPublicKey); got != "vrf1" {
		t.Fatalf("unexpected validator 1 vrf public key after commit: %q", got)
	}
	if _, ok := exec.GetCurrentValidatorSet().Validators[2]; ok {
		t.Fatalf("block B reconfig should remain pending until its own hash is committed")
	}

	second, err := exec.CommitReconfigs(blockB, 101)
	if err != nil {
		t.Fatalf("commit block B reconfigs: %v", err)
	}
	if len(second) != 1 {
		t.Fatalf("expected exactly one transition for block B, got %d", len(second))
	}
	if len(second[0].Added) != 1 || second[0].Added[0] != 2 {
		t.Fatalf("unexpected second transition: %+v", second[0])
	}
	if exec.GetCurrentValidatorSet().Epoch != 3 {
		t.Fatalf("unexpected epoch after block B commit: %d", exec.GetCurrentValidatorSet().Epoch)
	}
	if exec.GetCurrentValidatorSet().TotalPower != 3 {
		t.Fatalf("unexpected total power after block B commit: %d", exec.GetCurrentValidatorSet().TotalPower)
	}
}

func TestExecutorCommitReconfigs_DoesNotApplyDifferentHeightOrHash(t *testing.T) {
	kp1, _ := crypto.GenerateKeyPair()
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: []byte("pk0"), Power: 1, IsActive: true},
	})
	exec := NewExecutor(valSet)
	exec.SetReconfigAuthorizer(func(data *types.ReconfigData) bool {
		return data != nil && data.Type == types.ReconfigJoin && data.NodeID == 1
	})

	originalBlock := makeReconfigBlock(t, 10, []byte("target-block"), &types.ReconfigData{
		Type:        types.ReconfigJoin,
		NodeID:      1,
		PublicKey:   kp1.PublicKey,
		Power:       1,
		Epoch:       1,
		TargetEpoch: 2,
	}, kp1.PrivateKey)
	if err := exec.ExecuteBlock(originalBlock); err != nil {
		t.Fatalf("execute original block: %v", err)
	}

	wrongHeight := &types.Block{Height: 11, Hash: []byte("target-block")}
	transitions, err := exec.CommitReconfigs(wrongHeight, 100)
	if err != nil {
		t.Fatalf("commit wrong-height block: %v", err)
	}
	if len(transitions) != 0 {
		t.Fatalf("expected no transitions for wrong-height block, got %d", len(transitions))
	}
	if exec.GetCurrentValidatorSet().Epoch != 1 {
		t.Fatalf("wrong-height commit should not advance epoch, got %d", exec.GetCurrentValidatorSet().Epoch)
	}
	if exec.PendingReconfigCount() != 1 {
		t.Fatalf("wrong-height commit should preserve pending reconfig, got %d", exec.PendingReconfigCount())
	}

	wrongHash := &types.Block{Height: 10, Hash: []byte("different-hash")}
	transitions, err = exec.CommitReconfigs(wrongHash, 101)
	if err != nil {
		t.Fatalf("commit wrong-hash block: %v", err)
	}
	if len(transitions) != 0 {
		t.Fatalf("expected no transitions for wrong-hash block, got %d", len(transitions))
	}
	if exec.GetCurrentValidatorSet().Epoch != 1 {
		t.Fatalf("wrong-hash commit should not advance epoch, got %d", exec.GetCurrentValidatorSet().Epoch)
	}
	if exec.PendingReconfigCount() != 1 {
		t.Fatalf("wrong-hash commit should preserve pending reconfig, got %d", exec.PendingReconfigCount())
	}

	transitions, err = exec.CommitReconfigs(wrongHeight, 103)
	if err != nil {
		t.Fatalf("retry wrong-height block after mismatch: %v", err)
	}
	if len(transitions) != 0 {
		t.Fatalf("expected retrying wrong-height block to stay inert, got %d", len(transitions))
	}
	if exec.GetCurrentValidatorSet().Epoch != 1 {
		t.Fatalf("retrying wrong-height block should preserve epoch 1, got %d", exec.GetCurrentValidatorSet().Epoch)
	}
	if exec.PendingReconfigCount() != 1 {
		t.Fatalf("retrying wrong-height block should still preserve pending reconfig, got %d", exec.PendingReconfigCount())
	}

	transitions, err = exec.CommitReconfigs(originalBlock, 102)
	if err != nil {
		t.Fatalf("commit matching block: %v", err)
	}
	if len(transitions) != 1 {
		t.Fatalf("expected one transition after matching commit, got %d", len(transitions))
	}
	if exec.GetCurrentValidatorSet().Epoch != 2 {
		t.Fatalf("matching commit should advance epoch to 2, got %d", exec.GetCurrentValidatorSet().Epoch)
	}
	if exec.PendingReconfigCount() != 0 {
		t.Fatalf("expected pending reconfig queue to drain after matching commit, got %d", exec.PendingReconfigCount())
	}
}

func TestExecutorCommitReconfigs_IdempotentAfterSuccessfulApply(t *testing.T) {
	kp1, _ := crypto.GenerateKeyPair()
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: []byte("pk0"), Power: 1, IsActive: true},
	})
	exec := NewExecutor(valSet)
	exec.SetReconfigAuthorizer(func(data *types.ReconfigData) bool {
		return data != nil && data.Type == types.ReconfigJoin && data.NodeID == 1
	})

	block := makeReconfigBlock(t, 10, []byte("idempotent-block"), &types.ReconfigData{
		Type:        types.ReconfigJoin,
		NodeID:      1,
		PublicKey:   kp1.PublicKey,
		Power:       1,
		Epoch:       1,
		TargetEpoch: 2,
	}, kp1.PrivateKey)
	if err := exec.ExecuteBlock(block); err != nil {
		t.Fatalf("execute block: %v", err)
	}

	first, err := exec.CommitReconfigs(block, 100)
	if err != nil {
		t.Fatalf("first commit reconfigs: %v", err)
	}
	if len(first) != 1 {
		t.Fatalf("expected exactly one transition on first commit, got %d", len(first))
	}
	if exec.GetCurrentValidatorSet().Epoch != 2 {
		t.Fatalf("expected epoch 2 after first commit, got %d", exec.GetCurrentValidatorSet().Epoch)
	}

	second, err := exec.CommitReconfigs(block, 100)
	if err != nil {
		t.Fatalf("second commit reconfigs: %v", err)
	}
	if len(second) != 0 {
		t.Fatalf("expected idempotent second commit to return no transitions, got %d", len(second))
	}
	if exec.GetCurrentValidatorSet().Epoch != 2 {
		t.Fatalf("second commit should preserve epoch 2, got %d", exec.GetCurrentValidatorSet().Epoch)
	}
	if exec.PendingReconfigCount() != 0 {
		t.Fatalf("expected no pending reconfigs after idempotent second commit, got %d", exec.PendingReconfigCount())
	}
}
func TestExecutorRejectsUnsignedManualReconfig(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
	})
	exec := NewExecutor(valSet)

	block := makeReconfigBlock(t, 10, []byte("unsigned-block"), &types.ReconfigData{
		Type:        types.ReconfigJoin,
		NodeID:      1,
		PublicKey:   kp1.PublicKey,
		Power:       1,
		Epoch:       1,
		TargetEpoch: 2,
	}, nil)

	if err := exec.ExecuteBlock(block); err != nil {
		t.Fatalf("execute block: %v", err)
	}
	transitions, err := exec.CommitReconfigs(block, 100)
	if err != nil {
		t.Fatalf("commit reconfigs: %v", err)
	}
	if len(transitions) != 0 {
		t.Fatalf("expected unsigned reconfig to be ignored, got %d transitions", len(transitions))
	}
	if exec.GetCurrentValidatorSet().Epoch != 1 {
		t.Fatalf("unexpected epoch after unsigned reconfig: %d", exec.GetCurrentValidatorSet().Epoch)
	}
	if _, ok := exec.GetCurrentValidatorSet().Validators[1]; ok {
		t.Fatalf("unsigned reconfig should not install validator 1")
	}
}

func TestExecutorRejectsManualLeaveSignedByWrongKey(t *testing.T) {
	victim, _ := crypto.GenerateKeyPair()
	attacker, _ := crypto.GenerateKeyPair()
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		2: {ID: 2, PublicKey: victim.PublicKey, Power: 1, IsActive: true},
	})
	exec := NewExecutor(valSet)
	exec.SetReconfigAuthorizer(func(data *types.ReconfigData) bool {
		return data != nil && data.Type == types.ReconfigLeave && data.NodeID == 2
	})
	block := makeReconfigBlock(t, 10, []byte("forged-manual-leave"), &types.ReconfigData{
		Type:        types.ReconfigLeave,
		NodeID:      2,
		PublicKey:   attacker.PublicKey,
		Epoch:       1,
		TargetEpoch: 2,
	}, attacker.PrivateKey)
	if err := exec.ExecuteBlock(block); err != nil {
		t.Fatalf("execute block: %v", err)
	}
	transitions, err := exec.CommitReconfigs(block, 100)
	if err != nil {
		t.Fatalf("commit reconfigs: %v", err)
	}
	if len(transitions) != 0 {
		t.Fatalf("expected forged manual leave to be rejected, got %d transitions", len(transitions))
	}
	if _, ok := exec.GetCurrentValidatorSet().Validators[2]; !ok {
		t.Fatal("forged manual leave should not remove validator 2")
	}
}

func TestExecutorRejectsUnsignedManualLeaveWhenAuthorizerMissing(t *testing.T) {
	victim, _ := crypto.GenerateKeyPair()
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		2: {ID: 2, PublicKey: victim.PublicKey, Power: 1, IsActive: true},
	})
	exec := NewExecutor(valSet)
	block := makeReconfigBlock(t, 10, []byte("missing-authorizer-manual-leave"), &types.ReconfigData{
		Type:        types.ReconfigLeave,
		NodeID:      2,
		PublicKey:   victim.PublicKey,
		Epoch:       1,
		TargetEpoch: 2,
	}, victim.PrivateKey)
	if err := exec.ExecuteBlock(block); err != nil {
		t.Fatalf("execute block: %v", err)
	}
	transitions, err := exec.CommitReconfigs(block, 100)
	if err != nil {
		t.Fatalf("commit reconfigs: %v", err)
	}
	if len(transitions) != 0 {
		t.Fatalf("expected manual leave without authorizer to be rejected, got %d transitions", len(transitions))
	}
	if _, ok := exec.GetCurrentValidatorSet().Validators[2]; !ok {
		t.Fatal("manual leave without authorizer should not remove validator 2")
	}
}

func TestExecutorRejectsUnauthorizedManualLeaveWhenAuthorizerConfigured(t *testing.T) {
	victim, _ := crypto.GenerateKeyPair()
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		2: {ID: 2, PublicKey: victim.PublicKey, Power: 1, IsActive: true},
	})
	exec := NewExecutor(valSet)
	exec.SetReconfigAuthorizer(func(data *types.ReconfigData) bool { return false })
	block := makeReconfigBlock(t, 10, []byte("unauthorized-manual-leave"), &types.ReconfigData{
		Type:        types.ReconfigLeave,
		NodeID:      2,
		PublicKey:   victim.PublicKey,
		Epoch:       1,
		TargetEpoch: 2,
	}, victim.PrivateKey)
	if err := exec.ExecuteBlock(block); err != nil {
		t.Fatalf("execute block: %v", err)
	}
	transitions, err := exec.CommitReconfigs(block, 100)
	if err != nil {
		t.Fatalf("commit reconfigs: %v", err)
	}
	if len(transitions) != 0 {
		t.Fatalf("expected unauthorized manual leave to be rejected, got %d transitions", len(transitions))
	}
	if _, ok := exec.GetCurrentValidatorSet().Validators[2]; !ok {
		t.Fatal("unauthorized manual leave should not remove validator 2")
	}
}

func TestExecutorAllowsAuthorizedManualLeaveWhenAuthorizerConfigured(t *testing.T) {
	victim, _ := crypto.GenerateKeyPair()
	kpA, _ := crypto.GenerateKeyPair()
	kpB, _ := crypto.GenerateKeyPair()
	kpC, _ := crypto.GenerateKeyPair()
	kpD, _ := crypto.GenerateKeyPair()
	// Need 5 validators so that after leave n=4 >= 3f+1=4 (BFT guard)
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		2:  {ID: 2, PublicKey: victim.PublicKey, Power: 1, IsActive: true},
		10: {ID: 10, PublicKey: kpA.PublicKey, Power: 1, IsActive: true},
		11: {ID: 11, PublicKey: kpB.PublicKey, Power: 1, IsActive: true},
		12: {ID: 12, PublicKey: kpC.PublicKey, Power: 1, IsActive: true},
		13: {ID: 13, PublicKey: kpD.PublicKey, Power: 1, IsActive: true},
	})
	exec := NewExecutor(valSet)
	exec.SetReconfigAuthorizer(func(data *types.ReconfigData) bool {
		return data != nil && data.Type == types.ReconfigLeave && data.NodeID == 2
	})
	block := makeReconfigBlock(t, 10, []byte("authorized-manual-leave"), &types.ReconfigData{
		Type:        types.ReconfigLeave,
		NodeID:      2,
		PublicKey:   victim.PublicKey,
		Epoch:       1,
		TargetEpoch: 2,
	}, victim.PrivateKey)
	if err := exec.ExecuteBlock(block); err != nil {
		t.Fatalf("execute block: %v", err)
	}
	transitions, err := exec.CommitReconfigs(block, 100)
	if err != nil {
		t.Fatalf("commit reconfigs: %v", err)
	}
	if len(transitions) != 1 {
		t.Fatalf("expected authorized manual leave to produce one transition, got %d", len(transitions))
	}
	if _, ok := exec.GetCurrentValidatorSet().Validators[2]; ok {
		t.Fatal("authorized manual leave should remove validator 2")
	}
}

func TestExecutorRejectsStaleSignedReconfig(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	valSet := types.NewValidatorSet(2, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
	})
	exec := NewExecutor(valSet)

	block := makeReconfigBlock(t, 10, []byte("stale-block"), &types.ReconfigData{
		Type:        types.ReconfigJoin,
		NodeID:      1,
		PublicKey:   kp1.PublicKey,
		Power:       1,
		Epoch:       1,
		TargetEpoch: 2,
	}, kp1.PrivateKey)

	if err := exec.ExecuteBlock(block); err != nil {
		t.Fatalf("execute block: %v", err)
	}
	transitions, err := exec.CommitReconfigs(block, 100)
	if err != nil {
		t.Fatalf("commit reconfigs: %v", err)
	}
	if len(transitions) != 0 {
		t.Fatalf("expected stale signed reconfig to be ignored, got %d transitions", len(transitions))
	}
	if exec.GetCurrentValidatorSet().Epoch != 2 {
		t.Fatalf("unexpected epoch after stale reconfig: %d", exec.GetCurrentValidatorSet().Epoch)
	}
	if _, ok := exec.GetCurrentValidatorSet().Validators[1]; ok {
		t.Fatalf("stale reconfig should not install validator 1")
	}
}

func TestExecutorIgnoresNoOpReconfig(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
	})
	exec := NewExecutor(valSet)

	block := makeReconfigBlock(t, 10, []byte("noop-block"), &types.ReconfigData{
		Type:        types.ReconfigJoin,
		NodeID:      1,
		PublicKey:   kp1.PublicKey,
		Power:       1,
		Epoch:       1,
		TargetEpoch: 2,
	}, kp1.PrivateKey)

	if err := exec.ExecuteBlock(block); err != nil {
		t.Fatalf("execute block: %v", err)
	}
	transitions, err := exec.CommitReconfigs(block, 100)
	if err != nil {
		t.Fatalf("commit reconfigs: %v", err)
	}
	if len(transitions) != 0 {
		t.Fatalf("expected duplicate join to be ignored, got %d transitions", len(transitions))
	}
	if exec.GetCurrentValidatorSet().Epoch != 1 {
		t.Fatalf("unexpected epoch after duplicate join: %d", exec.GetCurrentValidatorSet().Epoch)
	}
	if exec.GetCurrentValidatorSet().TotalPower != 2 {
		t.Fatalf("unexpected total power after duplicate join: %d", exec.GetCurrentValidatorSet().TotalPower)
	}
}

func TestExecutorRejectsAutoLeaveWithoutProof(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	kp2, _ := crypto.GenerateKeyPair()
	kp3, _ := crypto.GenerateKeyPair()
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: kp2.PublicKey, Power: 1, IsActive: true},
		3: {ID: 3, PublicKey: kp3.PublicKey, Power: 1, IsActive: true},
	})
	exec := NewExecutor(valSet)
	block := makeReconfigBlock(t, 10, []byte("auto-no-proof"), &types.ReconfigData{
		Type:        types.ReconfigAutoLeave,
		NodeID:      2,
		Epoch:       1,
		TargetEpoch: 2,
	}, nil)
	if err := exec.ExecuteBlock(block); err != nil {
		t.Fatalf("execute block: %v", err)
	}
	transitions, err := exec.CommitReconfigs(block, 100)
	if err != nil {
		t.Fatalf("commit reconfigs: %v", err)
	}
	if len(transitions) != 0 {
		t.Fatalf("expected unsigned auto-leave to be ignored, got %d transitions", len(transitions))
	}
}

func TestExecutorRejectsAutoLeaveWithInsufficientQuorum(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	kp2, _ := crypto.GenerateKeyPair()
	kp3, _ := crypto.GenerateKeyPair()
	blockHash := []byte("auto-insufficient-quorum")
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: kp2.PublicKey, Power: 1, IsActive: true},
		3: {ID: 3, PublicKey: kp3.PublicKey, Power: 1, IsActive: true},
	})
	exec := NewExecutor(valSet)
	block := makeReconfigBlock(t, 10, blockHash, &types.ReconfigData{
		Type:        types.ReconfigAutoLeave,
		NodeID:      2,
		Epoch:       1,
		TargetEpoch: 2,
		AutoLeaveProof: makeAutoLeaveProof(t, 19, 2, blockHash, []uint64{2}, map[uint64]*crypto.Keypair{
			0: kp0,
			1: kp1,
		}),
	}, nil)
	if err := exec.ExecuteBlock(block); err != nil {
		t.Fatalf("execute block: %v", err)
	}
	transitions, err := exec.CommitReconfigs(block, 100)
	if err != nil {
		t.Fatalf("commit reconfigs: %v", err)
	}
	if len(transitions) != 0 {
		t.Fatalf("expected insufficient-quorum auto-leave to be ignored, got %d transitions", len(transitions))
	}
}

func TestExecutorRejectsAutoLeaveWithDigestMismatch(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	kp2, _ := crypto.GenerateKeyPair()
	kp3, _ := crypto.GenerateKeyPair()
	blockHash := []byte("auto-digest-mismatch")
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: kp2.PublicKey, Power: 1, IsActive: true},
		3: {ID: 3, PublicKey: kp3.PublicKey, Power: 1, IsActive: true},
	})
	exec := NewExecutor(valSet)
	proof := makeAutoLeaveProof(t, 20, 2, blockHash, []uint64{2}, map[uint64]*crypto.Keypair{
		0: kp0,
		1: kp1,
		2: kp2,
	})
	proof.AutoVotes[1].Digest = []byte("bad-digest")
	block := makeReconfigBlock(t, 10, blockHash, &types.ReconfigData{
		Type:           types.ReconfigAutoLeave,
		NodeID:         2,
		Epoch:          1,
		TargetEpoch:    2,
		AutoLeaveProof: proof,
	}, nil)
	if err := exec.ExecuteBlock(block); err != nil {
		t.Fatalf("execute block: %v", err)
	}
	transitions, err := exec.CommitReconfigs(block, 100)
	if err != nil {
		t.Fatalf("commit reconfigs: %v", err)
	}
	if len(transitions) != 0 {
		t.Fatalf("expected digest-mismatched auto-leave to be ignored, got %d transitions", len(transitions))
	}
}

func TestExecutorAcceptsWeightedAutoLeaveQuorumByPower(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	kp2, _ := crypto.GenerateKeyPair()
	kp3, _ := crypto.GenerateKeyPair()
	kp4, _ := crypto.GenerateKeyPair()
	blockHash := []byte("auto-weighted-quorum")
	// Need 5 validators (by count) so that after leave n=4 >= 3f+1=4 (BFT guard)
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 3, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 2, IsActive: true},
		2: {ID: 2, PublicKey: kp2.PublicKey, Power: 1, IsActive: true},
		3: {ID: 3, PublicKey: kp3.PublicKey, Power: 1, IsActive: true},
		4: {ID: 4, PublicKey: kp4.PublicKey, Power: 1, IsActive: true},
	})
	exec := NewExecutor(valSet)
	block := makeReconfigBlock(t, 10, blockHash, &types.ReconfigData{
		Type:        types.ReconfigAutoLeave,
		NodeID:      2,
		Epoch:       1,
		TargetEpoch: 2,
		AutoLeaveProof: makeAutoLeaveProof(t, 21, 2, blockHash, []uint64{2}, map[uint64]*crypto.Keypair{
			0: kp0,
			1: kp1,
			3: kp3,
			4: kp4,
		}),
	}, nil)
	if err := exec.ExecuteBlock(block); err != nil {
		t.Fatalf("execute block: %v", err)
	}
	transitions, err := exec.CommitReconfigs(block, 100)
	if err != nil {
		t.Fatalf("commit reconfigs: %v", err)
	}
	if len(transitions) != 1 {
		t.Fatalf("expected weighted auto-leave to be accepted, got %d transitions", len(transitions))
	}
	if len(transitions[0].Removed) != 1 || transitions[0].Removed[0] != 2 {
		t.Fatalf("expected node 2 removed, got %v", transitions[0].Removed)
	}
}

func TestExecutorAcceptsAutoLeaveProofWhenConfigIDDiffersFromTargetEpoch(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	kp2, _ := crypto.GenerateKeyPair()
	kp3, _ := crypto.GenerateKeyPair()
	kp4, _ := crypto.GenerateKeyPair()
	// Need 5 validators so that after leave n=4 >= 3f+1=4 (BFT guard)
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: kp2.PublicKey, Power: 1, IsActive: true},
		3: {ID: 3, PublicKey: kp3.PublicKey, Power: 1, IsActive: true},
		4: {ID: 4, PublicKey: kp4.PublicKey, Power: 1, IsActive: true},
	})
	exec := NewExecutor(valSet)
	proof := makeAutoLeaveProof(t, 22, 9, []byte("ordered-block-hash"), []uint64{2}, map[uint64]*crypto.Keypair{
		0: kp0,
		1: kp1,
		3: kp3,
		4: kp4,
	})
	block := makeReconfigBlock(t, 10, []byte("ordered-block-hash"), &types.ReconfigData{
		Type:           types.ReconfigAutoLeave,
		NodeID:         2,
		Epoch:          1,
		TargetEpoch:    2,
		AutoLeaveProof: proof,
	}, nil)
	if err := exec.ExecuteBlock(block); err != nil {
		t.Fatalf("execute block: %v", err)
	}
	transitions, err := exec.CommitReconfigs(block, 100)
	if err != nil {
		t.Fatalf("commit reconfigs: %v", err)
	}
	if len(transitions) != 1 {
		t.Fatalf("expected auto-leave with distinct config id to be accepted, got %d transitions", len(transitions))
	}
	if len(transitions[0].Removed) != 1 || transitions[0].Removed[0] != 2 {
		t.Fatalf("expected node 2 removed, got %v", transitions[0].Removed)
	}
	if transitions[0].NewEpoch != 2 {
		t.Fatalf("expected epoch transition to remain driven by TargetEpoch=2, got %d", transitions[0].NewEpoch)
	}
}

func TestExecutorRejectsAutoLeaveProofMissingConfigID(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	kp2, _ := crypto.GenerateKeyPair()
	kp3, _ := crypto.GenerateKeyPair()
	blockHash := []byte("auto-missing-config-id")
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: kp2.PublicKey, Power: 1, IsActive: true},
		3: {ID: 3, PublicKey: kp3.PublicKey, Power: 1, IsActive: true},
	})
	exec := NewExecutor(valSet)
	proof := makeAutoLeaveProof(t, 23, 9, blockHash, []uint64{2}, map[uint64]*crypto.Keypair{
		0: kp0,
		1: kp1,
		2: kp2,
	})
	proof.NewConfigID = 0
	block := makeReconfigBlock(t, 10, blockHash, &types.ReconfigData{
		Type:           types.ReconfigAutoLeave,
		NodeID:         2,
		Epoch:          1,
		TargetEpoch:    2,
		AutoLeaveProof: proof,
	}, nil)
	if err := exec.ExecuteBlock(block); err != nil {
		t.Fatalf("execute block: %v", err)
	}
	transitions, err := exec.CommitReconfigs(block, 100)
	if err != nil {
		t.Fatalf("commit reconfigs: %v", err)
	}
	if len(transitions) != 0 {
		t.Fatalf("expected missing-config-id auto-leave proof to be rejected, got %d transitions", len(transitions))
	}
}

func TestExecutorRejectsAutoLeaveProofWithMismatchedBlockHash(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	kp2, _ := crypto.GenerateKeyPair()
	kp3, _ := crypto.GenerateKeyPair()
	proofBlockHash := []byte("proof-bound-block")
	orderedBlockHash := []byte("different-ordered-block")
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: kp2.PublicKey, Power: 1, IsActive: true},
		3: {ID: 3, PublicKey: kp3.PublicKey, Power: 1, IsActive: true},
	})
	exec := NewExecutor(valSet)
	block := makeReconfigBlock(t, 10, orderedBlockHash, &types.ReconfigData{
		Type:        types.ReconfigAutoLeave,
		NodeID:      2,
		Epoch:       1,
		TargetEpoch: 2,
		AutoLeaveProof: makeAutoLeaveProof(t, 24, 2, proofBlockHash, []uint64{2}, map[uint64]*crypto.Keypair{
			0: kp0,
			1: kp1,
			2: kp2,
		}),
	}, nil)
	if err := exec.ExecuteBlock(block); err != nil {
		t.Fatalf("execute block: %v", err)
	}
	transitions, err := exec.CommitReconfigs(block, 100)
	if err != nil {
		t.Fatalf("commit reconfigs: %v", err)
	}
	if len(transitions) != 0 {
		t.Fatalf("expected mismatched proof/block hash auto-leave to be rejected, got %d transitions", len(transitions))
	}
}

func TestExecutorRejectsJoinWithoutAuthorizerApproval(t *testing.T) {
	kp1, _ := crypto.GenerateKeyPair()
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: []byte("pk0"), Power: 1, IsActive: true},
	})
	exec := NewExecutor(valSet)
	block := makeReconfigBlock(t, 10, []byte("unauthorized-join"), &types.ReconfigData{
		Type:        types.ReconfigJoin,
		NodeID:      1,
		PublicKey:   kp1.PublicKey,
		Power:       1,
		Epoch:       1,
		TargetEpoch: 2,
	}, kp1.PrivateKey)
	if err := exec.ExecuteBlock(block); err != nil {
		t.Fatalf("execute block: %v", err)
	}
	transitions, err := exec.CommitReconfigs(block, 100)
	if err != nil {
		t.Fatalf("commit reconfigs: %v", err)
	}
	if len(transitions) != 0 {
		t.Fatalf("expected unauthorized join to be rejected, got %d transitions", len(transitions))
	}
	if exec.GetCurrentValidatorSet().Epoch != 1 {
		t.Fatalf("unexpected epoch after unauthorized join: %d", exec.GetCurrentValidatorSet().Epoch)
	}
}

func TestExecutorRejectsAutoLeaveProofSignedOnlyByInactiveValidators(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	kp2, _ := crypto.GenerateKeyPair()
	kp3, _ := crypto.GenerateKeyPair()
	blockHash := []byte("inactive-auto-voter")
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: kp2.PublicKey, Power: 1, IsActive: true},
		3: {ID: 3, PublicKey: kp3.PublicKey, Power: 5, IsActive: false},
	})
	exec := NewExecutor(valSet)
	block := makeReconfigBlock(t, 10, blockHash, &types.ReconfigData{
		Type:        types.ReconfigAutoLeave,
		NodeID:      2,
		Epoch:       1,
		TargetEpoch: 2,
		AutoLeaveProof: makeAutoLeaveProof(t, 25, 2, blockHash, []uint64{2}, map[uint64]*crypto.Keypair{
			3: kp3,
		}),
	}, nil)
	if err := exec.ExecuteBlock(block); err != nil {
		t.Fatalf("execute block: %v", err)
	}
	transitions, err := exec.CommitReconfigs(block, 100)
	if err != nil {
		t.Fatalf("commit reconfigs: %v", err)
	}
	if len(transitions) != 0 {
		t.Fatalf("expected inactive-only auto-leave proof to be rejected, got %d transitions", len(transitions))
	}
}

func TestExecutorCommitReconfigs_RejectsOrderedBlockWithoutHash(t *testing.T) {
	kp1, _ := crypto.GenerateKeyPair()
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: []byte("pk0"), Power: 1, IsActive: true},
	})
	exec := NewExecutor(valSet)
	exec.SetReconfigAuthorizer(func(data *types.ReconfigData) bool { return data != nil && data.Type == types.ReconfigJoin && data.NodeID == 1 })
	block := makeReconfigBlock(t, 10, nil, &types.ReconfigData{
		Type:        types.ReconfigJoin,
		NodeID:      1,
		PublicKey:   kp1.PublicKey,
		Power:       1,
		Epoch:       1,
		TargetEpoch: 2,
	}, kp1.PrivateKey)
	if err := exec.ExecuteBlock(block); err != nil {
		t.Fatalf("execute block: %v", err)
	}
	before := exec.GetCurrentValidatorSet()
	transitions, err := exec.CommitReconfigs(block, 100)
	if err == nil {
		t.Fatalf("expected commit reconfigs to reject block without hash")
	}
	if len(transitions) != 0 {
		t.Fatalf("expected no transitions for missing-hash commit, got %d", len(transitions))
	}
	if exec.PendingReconfigCount() != 1 {
		t.Fatalf("expected pending reconfig retained after missing-hash commit, got %d", exec.PendingReconfigCount())
	}
	after := exec.GetCurrentValidatorSet()
	if after == nil || before == nil || after.Epoch != before.Epoch {
		t.Fatalf("validator set changed despite missing-hash commit failure: before=%+v after=%+v", before, after)
	}
}

func TestExecutorApplyOrderedBlock_RejectsMissingHashWithoutStateChange(t *testing.T) {
	kp1, _ := crypto.GenerateKeyPair()
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: []byte("pk0"), Power: 1, IsActive: true},
	})
	exec := NewExecutor(valSet)
	exec.SetReconfigAuthorizer(func(data *types.ReconfigData) bool { return data != nil && data.Type == types.ReconfigJoin && data.NodeID == 1 })
	block := makeReconfigBlock(t, 10, nil, &types.ReconfigData{
		Type:        types.ReconfigJoin,
		NodeID:      1,
		PublicKey:   kp1.PublicKey,
		Power:       1,
		Epoch:       1,
		TargetEpoch: 2,
	}, kp1.PrivateKey)
	before := exec.GetCurrentValidatorSet()
	transitions, err := exec.ApplyOrderedBlock(block, 100)
	if err == nil {
		t.Fatalf("expected ordered apply to reject block without hash")
	}
	if len(transitions) != 0 {
		t.Fatalf("expected no transitions for missing-hash apply, got %d", len(transitions))
	}
	if exec.PendingReconfigCount() != 0 {
		t.Fatalf("expected no pending reconfigs after missing-hash apply, got %d", exec.PendingReconfigCount())
	}
	after := exec.GetCurrentValidatorSet()
	if after == nil || before == nil || after.Epoch != before.Epoch {
		t.Fatalf("validator set changed despite missing-hash apply failure: before=%+v after=%+v", before, after)
	}
	if blocksExecuted, txsExecuted := exec.Stats(); blocksExecuted != 0 || txsExecuted != 0 {
		t.Fatalf("expected zero execution stats after missing-hash apply, got blocks=%d txs=%d", blocksExecuted, txsExecuted)
	}
}

func TestExecutorApplyOrderedBlock_FailsMissingVertexWithoutPartialState(t *testing.T) {
	kp1, _ := crypto.GenerateKeyPair()
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: []byte("pk0"), Power: 1, IsActive: true},
	})
	exec := NewExecutor(valSet)
	exec.SetReconfigAuthorizer(func(data *types.ReconfigData) bool { return data != nil && data.Type == types.ReconfigJoin && data.NodeID == 1 })
	resolver := &stubVertexResolver{}
	exec.SetVertexResolver(resolver)

	var missingHash types.Hash
	copy(missingHash[:], []byte("missing-vertex-hash-0000000000000"))
	block := &types.Block{
		Height: 10,
		Hash:   []byte("missing-vertex-block"),
		Epoch:  1,
		Payload: []*types.VertexCertificate{
			{VertexHash: missingHash, Epoch: 1, Round: 1},
		},
	}

	transitions, err := exec.ApplyOrderedBlock(block, 100)
	if err == nil {
		t.Fatalf("expected missing vertex to fail ordered apply")
	}
	if len(transitions) != 0 {
		t.Fatalf("expected no transitions on failed ordered apply, got %d", len(transitions))
	}
	if exec.PendingReconfigCount() != 0 {
		t.Fatalf("expected no pending reconfigs after failed ordered apply, got %d", exec.PendingReconfigCount())
	}
	if exec.GetCurrentValidatorSet().Epoch != 1 {
		t.Fatalf("validator set changed despite failed ordered apply: epoch=%d", exec.GetCurrentValidatorSet().Epoch)
	}
	if blocksExecuted, txsExecuted := exec.Stats(); blocksExecuted != 0 || txsExecuted != 0 {
		t.Fatalf("expected zero execution stats after failed ordered apply, got blocks=%d txs=%d", blocksExecuted, txsExecuted)
	}

	vertex := &types.Vertex{Txs: []*types.Transaction{{Type: types.TxTypeReconfig, Payload: mustMarshalReconfigTxPayload(t, &types.ReconfigData{Type: types.ReconfigJoin, NodeID: 1, PublicKey: kp1.PublicKey, Power: 1, Epoch: 1, TargetEpoch: 2}, kp1.PrivateKey)}}}
	resolver.setVertex(missingHash, vertex)

	transitions, err = exec.ApplyOrderedBlock(block, 100)
	if err != nil {
		t.Fatalf("retry ordered apply after vertex restore: %v", err)
	}
	if len(transitions) != 1 {
		t.Fatalf("expected one transition after retry, got %d", len(transitions))
	}
	if exec.GetCurrentValidatorSet().Epoch != 2 {
		t.Fatalf("expected epoch 2 after successful retry, got %d", exec.GetCurrentValidatorSet().Epoch)
	}
	if exec.PendingReconfigCount() != 0 {
		t.Fatalf("expected no pending reconfigs after successful retry, got %d", exec.PendingReconfigCount())
	}
	if blocksExecuted, txsExecuted := exec.Stats(); blocksExecuted != 1 || txsExecuted != 1 {
		t.Fatalf("expected single successful execution after retry, got blocks=%d txs=%d", blocksExecuted, txsExecuted)
	}
}

func TestExecutorApplyOrderedBlock_ConcurrentReplayExecutesOnce(t *testing.T) {
	kp1, _ := crypto.GenerateKeyPair()
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: []byte("pk0"), Power: 1, IsActive: true},
	})
	exec := NewExecutor(valSet)
	exec.SetReconfigAuthorizer(func(data *types.ReconfigData) bool { return data != nil && data.Type == types.ReconfigJoin && data.NodeID == 1 })
	var callbackCount atomic.Int32
	exec.SetEpochChangeCallback(func(newValSet *types.ValidatorSet, transitions []*types.EpochTransition) error {
		callbackCount.Add(1)
		return nil
	})
	block := makeReconfigBlock(t, 10, []byte("concurrent-replay-block"), &types.ReconfigData{
		Type:        types.ReconfigJoin,
		NodeID:      1,
		PublicKey:   kp1.PublicKey,
		Power:       1,
		Epoch:       1,
		TargetEpoch: 2,
	}, kp1.PrivateKey)

	const goroutines = 8
	results := make(chan []*types.EpochTransition, goroutines)
	errs := make(chan error, goroutines)
	var wg sync.WaitGroup
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			transitions, err := exec.ApplyOrderedBlock(block, 100)
			results <- transitions
			errs <- err
		}()
	}
	wg.Wait()
	close(results)
	close(errs)

	nonEmpty := 0
	for err := range errs {
		if err != nil {
			t.Fatalf("concurrent ordered apply failed: %v", err)
		}
	}
	for transitions := range results {
		if len(transitions) > 0 {
			nonEmpty++
			if len(transitions) != 1 {
				t.Fatalf("expected winning apply to publish exactly one transition, got %d", len(transitions))
			}
		}
	}
	if nonEmpty != 1 {
		t.Fatalf("expected exactly one caller to publish a transition, got %d", nonEmpty)
	}
	if exec.GetCurrentValidatorSet().Epoch != 2 {
		t.Fatalf("expected epoch 2 after concurrent replay, got %d", exec.GetCurrentValidatorSet().Epoch)
	}
	if blocksExecuted, txsExecuted := exec.Stats(); blocksExecuted != 1 || txsExecuted != 1 {
		t.Fatalf("expected block to execute exactly once, got blocks=%d txs=%d", blocksExecuted, txsExecuted)
	}
	if got := callbackCount.Load(); got != 1 {
		t.Fatalf("expected callback to run exactly once, got %d", got)
	}
}

func TestExecutorApplyOrderedBlock_RetryAfterExecutionFailureReexecutesCleanly(t *testing.T) {
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: []byte("pk0"), Power: 1, IsActive: true},
	})
	exec := NewExecutor(valSet)
	missingVertexHash := types.Hash{}
	copy(missingVertexHash[:], []byte("missing-vertex"))
	block := &types.Block{
		Height: 10,
		Hash:   []byte("ordered-exec-retry"),
		Epoch:  1,
		Payload: []*types.VertexCertificate{{
			VertexHash: missingVertexHash,
		}},
	}
	resolver := &stubVertexResolver{}
	exec.SetVertexResolver(resolver)

	if got, err := exec.ApplyOrderedBlock(block, 100); err == nil || len(got) != 0 {
		t.Fatalf("expected first ordered apply to fail on missing vertex, got transitions=%v err=%v", got, err)
	}
	if blocksExecuted, txsExecuted := exec.Stats(); blocksExecuted != 0 || txsExecuted != 0 {
		t.Fatalf("expected failed execution to leave stats untouched, got blocks=%d txs=%d", blocksExecuted, txsExecuted)
	}
	if exec.PendingReconfigCount() != 0 {
		t.Fatalf("expected no pending reconfigs after failed execution, got %d", exec.PendingReconfigCount())
	}

	kp1, _ := crypto.GenerateKeyPair()
	reconfig := &types.ReconfigData{
		Type:        types.ReconfigJoin,
		NodeID:      1,
		PublicKey:   kp1.PublicKey,
		Power:       1,
		Epoch:       1,
		TargetEpoch: 2,
	}
	txPayload := mustMarshalReconfigTxPayload(t, reconfig, kp1.PrivateKey)
	vertex := &types.Vertex{Txs: []*types.Transaction{{Type: types.TxTypeReconfig, Payload: txPayload}}}
	resolver.setVertex(missingVertexHash, vertex)
	exec.SetReconfigAuthorizer(func(data *types.ReconfigData) bool {
		return data != nil && data.Type == types.ReconfigJoin && data.NodeID == 1
	})

	transitions, err := exec.ApplyOrderedBlock(block, 100)
	if err != nil {
		t.Fatalf("retry ordered apply failed: %v", err)
	}
	if len(transitions) != 1 {
		t.Fatalf("expected one transition after retry, got %d", len(transitions))
	}
	if exec.GetCurrentValidatorSet().Epoch != 2 {
		t.Fatalf("expected epoch 2 after retry, got %d", exec.GetCurrentValidatorSet().Epoch)
	}
	if blocksExecuted, txsExecuted := exec.Stats(); blocksExecuted != 1 || txsExecuted != 1 {
		t.Fatalf("expected retry to execute block exactly once successfully, got blocks=%d txs=%d", blocksExecuted, txsExecuted)
	}
}

func TestExecutorCommitReconfigs_DifferentActivationRankDoesNotRepublishTransition(t *testing.T) {
	kp1, _ := crypto.GenerateKeyPair()
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: []byte("pk0"), Power: 1, IsActive: true},
	})
	exec := NewExecutor(valSet)
	exec.SetReconfigAuthorizer(func(data *types.ReconfigData) bool { return data != nil && data.Type == types.ReconfigJoin && data.NodeID == 1 })
	block := makeReconfigBlock(t, 10, []byte("activation-rank-replay"), &types.ReconfigData{
		Type:        types.ReconfigJoin,
		NodeID:      1,
		PublicKey:   kp1.PublicKey,
		Power:       1,
		Epoch:       1,
		TargetEpoch: 2,
	}, kp1.PrivateKey)
	if err := exec.ExecuteBlock(block); err != nil {
		t.Fatalf("execute block: %v", err)
	}

	first, err := exec.CommitReconfigs(block, 100)
	if err != nil {
		t.Fatalf("first commit reconfigs: %v", err)
	}
	if len(first) != 1 {
		t.Fatalf("expected one transition on first commit, got %d", len(first))
	}
	if first[0].ActivationRank != 100 {
		t.Fatalf("unexpected activation rank on first commit: %d", first[0].ActivationRank)
	}

	second, err := exec.CommitReconfigs(block, 101)
	if err != nil {
		t.Fatalf("second commit reconfigs: %v", err)
	}
	if len(second) != 0 {
		t.Fatalf("expected replay with different activation rank to stay idempotent, got %d transitions", len(second))
	}
	if exec.GetCurrentValidatorSet().Epoch != 2 {
		t.Fatalf("expected epoch to remain 2 after replay, got %d", exec.GetCurrentValidatorSet().Epoch)
	}
}

func mustMarshalReconfigTxPayload(t *testing.T, reconfig *types.ReconfigData, signer types.PrivateKey) []byte {
	t.Helper()
	if signer != nil {
		reconfig.Sign(signer)
	}
	payload, err := json.Marshal(reconfig)
	if err != nil {
		t.Fatalf("marshal reconfig data: %v", err)
	}
	return payload
}
