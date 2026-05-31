package hotstuff

import (
	"testing"

	"evolvbft/evolvbft/storage"
	"evolvbft/evolvbft/types"
)

func TestBlockTreeCommitCallbackAndQCUpdate(t *testing.T) {
	validators := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: []byte("v0"), Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: []byte("v1"), Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: []byte("v2"), Power: 1, IsActive: true},
		3: {ID: 3, PublicKey: []byte("v3"), Power: 1, IsActive: true},
	}
	valSet := types.NewValidatorSet(1, validators)
	store := storage.NewStorageManager(0)
	exec := NewExecutor(valSet)
	bt := NewBlockTree(store, exec)

	var committed uint64
	bt.SetCommitCallback(func(block *types.Block) {
		committed = block.Height
	})

	b1 := types.NewBlock(1, make([]byte, 32), []byte("b1"), 1, 1, 0, 0, types.NewQuorumCertificate(nil, 0, 0, types.PhaseDecide), nil)
	qc1 := types.NewQuorumCertificate(b1.Hash, 1, 1, types.PhasePrepare)
	b2 := types.NewBlock(2, b1.Hash, []byte("b2"), 2, 1, 0, 0, qc1, nil)
	qc2 := types.NewQuorumCertificate(b2.Hash, 2, 1, types.PhasePrepare)
	b3 := types.NewBlock(3, b2.Hash, []byte("b3"), 3, 1, 0, 0, qc2, nil)
	qc3 := types.NewQuorumCertificate(b3.Hash, 3, 1, types.PhasePrepare)
	b4 := types.NewBlock(4, b3.Hash, []byte("b4"), 4, 1, 0, 0, qc3, nil)

	if err := bt.ProcessBlock(b1); err != nil {
		t.Fatalf("process b1 failed: %v", err)
	}
	if err := bt.ProcessBlock(b2); err != nil {
		t.Fatalf("process b2 failed: %v", err)
	}
	if err := bt.ProcessBlock(b3); err != nil {
		t.Fatalf("process b3 failed: %v", err)
	}
	if err := bt.ProcessBlock(b4); err != nil {
		t.Fatalf("process b4 failed: %v", err)
	}
	if committed != 1 {
		t.Fatalf("expected committed height 1, got %d", committed)
	}

	bt.OnVoteQC(types.NewQuorumCertificate(b4.Hash, 4, 1, types.PhasePrepare))
	if bt.GetHighQC().View != 4 {
		t.Fatalf("expected highQC view 4, got %d", bt.GetHighQC().View)
	}
}

func TestSafeNodeRejectsConflictingProposal(t *testing.T) {
	validators := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: []byte("v0"), Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: []byte("v1"), Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: []byte("v2"), Power: 1, IsActive: true},
		3: {ID: 3, PublicKey: []byte("v3"), Power: 1, IsActive: true},
	}
	valSet := types.NewValidatorSet(1, validators)
	store := storage.NewStorageManager(0)
	exec := NewExecutor(valSet)
	bt := NewBlockTree(store, exec)

	b1 := types.NewBlock(1, make([]byte, 32), []byte("b1"), 1, 1, 0, 0, types.NewQuorumCertificate(nil, 0, 0, types.PhaseDecide), nil)
	qc1 := types.NewQuorumCertificate(b1.Hash, 1, 1, types.PhasePrepare)
	b2 := types.NewBlock(2, b1.Hash, []byte("b2"), 2, 1, 0, 0, qc1, nil)
	qc2 := types.NewQuorumCertificate(b2.Hash, 2, 1, types.PhasePrepare)
	b3 := types.NewBlock(3, b2.Hash, []byte("b3"), 3, 1, 0, 0, qc2, nil)
	qc3 := types.NewQuorumCertificate(b3.Hash, 3, 1, types.PhasePrepare)
	b4 := types.NewBlock(4, b3.Hash, []byte("b4"), 4, 1, 0, 0, qc3, nil)

	if err := bt.ProcessBlock(b1); err != nil {
		t.Fatalf("process b1 failed: %v", err)
	}
	if err := bt.ProcessBlock(b2); err != nil {
		t.Fatalf("process b2 failed: %v", err)
	}
	if err := bt.ProcessBlock(b3); err != nil {
		t.Fatalf("process b3 failed: %v", err)
	}
	if err := bt.ProcessBlock(b4); err != nil {
		t.Fatalf("process b4 failed: %v", err)
	}

	branch1 := types.NewBlock(5, b1.Hash, []byte("branch1"), 5, 1, 0, 0, qc1, nil)
	if err := bt.ProcessBlock(branch1); err != nil {
		t.Fatalf("process branch1 failed: %v", err)
	}
	conflictQC := types.NewQuorumCertificate(branch1.Hash, 1, 1, types.PhasePrepare)
	conflict := types.NewBlock(6, branch1.Hash, []byte("conflict"), 100, 1, 0, 0, conflictQC, nil)

	if bt.SafeNode(conflict) {
		t.Fatalf("expected conflicting proposal to be rejected by SafeNode")
	}
}

func TestBlockTreeInvariantChecksForLockAndCommit(t *testing.T) {
	validators := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: []byte("v0"), Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: []byte("v1"), Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: []byte("v2"), Power: 1, IsActive: true},
		3: {ID: 3, PublicKey: []byte("v3"), Power: 1, IsActive: true},
	}
	valSet := types.NewValidatorSet(1, validators)
	store := storage.NewStorageManager(0)
	exec := NewExecutor(valSet)
	bt := NewBlockTree(store, exec)

	var committed uint64
	bt.SetCommitCallback(func(block *types.Block) {
		committed = block.Height
	})

	b1 := types.NewBlock(1, make([]byte, 32), []byte("b1"), 1, 1, 0, 0, types.NewQuorumCertificate(nil, 0, 0, types.PhaseDecide), nil)
	qc1 := types.NewQuorumCertificate(b1.Hash, 1, 1, types.PhasePrepare)
	b2 := types.NewBlock(2, b1.Hash, []byte("b2"), 2, 1, 0, 0, qc1, nil)
	qc2 := types.NewQuorumCertificate(b2.Hash, 2, 1, types.PhasePrepare)
	b3 := types.NewBlock(3, b2.Hash, []byte("b3"), 3, 1, 0, 0, qc2, nil)
	qc3 := types.NewQuorumCertificate(b3.Hash, 3, 1, types.PhasePrepare)
	b4 := types.NewBlock(4, b3.Hash, []byte("b4"), 4, 1, 0, 0, qc3, nil)

	if err := bt.ProcessBlock(b1); err != nil {
		t.Fatalf("process b1 failed: %v", err)
	}
	if err := bt.ProcessBlock(b2); err != nil {
		t.Fatalf("process b2 failed: %v", err)
	}
	if err := bt.ProcessBlock(b3); err != nil {
		t.Fatalf("process b3 failed: %v", err)
	}
	if err := bt.ProcessBlock(b4); err != nil {
		t.Fatalf("process b4 failed: %v", err)
	}

	if committed != 1 {
		t.Fatalf("expected committed height 1, got %d", committed)
	}
	lockedViewBefore := bt.GetLockedQC().View

	bad1 := types.NewBlock(5, b2.Hash, []byte("bad1"), 2, 1, 0, 0, qc2, nil)
	badQC1 := types.NewQuorumCertificate(bad1.Hash, 2, 1, types.PhasePrepare)
	bad2 := types.NewBlock(6, bad1.Hash, []byte("bad2"), 2, 1, 0, 0, badQC1, nil)
	badQC2 := types.NewQuorumCertificate(bad2.Hash, 2, 1, types.PhasePrepare)
	bad3 := types.NewBlock(7, bad2.Hash, []byte("bad3"), 2, 1, 0, 0, badQC2, nil)

	if err := bt.ProcessBlock(bad1); err != nil {
		t.Fatalf("process bad1 failed: %v", err)
	}
	if err := bt.ProcessBlock(bad2); err != nil {
		t.Fatalf("process bad2 failed: %v", err)
	}
	if err := bt.ProcessBlock(bad3); err != nil {
		t.Fatalf("process bad3 failed: %v", err)
	}

	if bt.GetLockedQC().View != lockedViewBefore {
		t.Fatalf("expected lockedQC view to remain %d, got %d", lockedViewBefore, bt.GetLockedQC().View)
	}
	if committed != 1 {
		t.Fatalf("expected committed height to remain 1, got %d", committed)
	}
}

func TestBlockTreeOnVoteQCBoundaryAcrossEpoch(t *testing.T) {
	validators := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: []byte("v0"), Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: []byte("v1"), Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: []byte("v2"), Power: 1, IsActive: true},
		3: {ID: 3, PublicKey: []byte("v3"), Power: 1, IsActive: true},
	}
	valSet := types.NewValidatorSet(1, validators)
	store := storage.NewStorageManager(0)
	bt := NewBlockTree(store, NewExecutor(valSet))

	bt.OnVoteQC(types.NewQuorumCertificate([]byte("e1-v7"), 7, 1, types.PhasePrepare))
	if bt.GetHighQC().Epoch != 1 || bt.GetHighQC().View != 7 {
		t.Fatalf("expected highQC to move to epoch 1 view 7, got epoch=%d view=%d", bt.GetHighQC().Epoch, bt.GetHighQC().View)
	}

	bt.OnVoteQC(types.NewQuorumCertificate([]byte("e0-v99"), 99, 0, types.PhasePrepare))
	if bt.GetHighQC().Epoch != 1 || bt.GetHighQC().View != 7 {
		t.Fatalf("expected stale-epoch qc ignored, got epoch=%d view=%d", bt.GetHighQC().Epoch, bt.GetHighQC().View)
	}

	bt.OnVoteQC(types.NewQuorumCertificate([]byte("e2-v1"), 1, 2, types.PhasePrepare))
	if bt.GetHighQC().Epoch != 2 || bt.GetHighQC().View != 1 {
		t.Fatalf("expected newer-epoch qc accepted, got epoch=%d view=%d", bt.GetHighQC().Epoch, bt.GetHighQC().View)
	}
}

func TestBlockTreeFastCommitWith2Chain(t *testing.T) {
	validators := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: []byte("v0"), Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: []byte("v1"), Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: []byte("v2"), Power: 1, IsActive: true},
		3: {ID: 3, PublicKey: []byte("v3"), Power: 1, IsActive: true},
	}
	valSet := types.NewValidatorSet(1, validators)
	store := storage.NewStorageManager(0)
	exec := NewExecutor(valSet)

	// Test 1: Fast commit (2-chain) when NumSignatures == 4
	bt := NewBlockTree(store, exec)
	bt.SetFastPathThreshold(4)

	var committed uint64
	bt.SetCommitCallback(func(block *types.Block) {
		committed = block.Height
	})

	b1 := types.NewBlock(1, make([]byte, 32), []byte("b1"), 1, 1, 0, 0, types.NewQuorumCertificate(nil, 0, 0, types.PhaseDecide), nil)
	qc1 := types.NewQuorumCertificate(b1.Hash, 1, 1, types.PhasePrepare)

	b2 := types.NewBlock(2, b1.Hash, []byte("b2"), 2, 1, 0, 0, qc1, nil)
	qc2 := types.NewQuorumCertificate(b2.Hash, 2, 1, types.PhasePrepare)
	qc2.NumSignatures = 4 // All validators signed!

	b3 := types.NewBlock(3, b2.Hash, []byte("b3"), 3, 1, 0, 0, qc2, nil)

	bt.ProcessBlock(b1)
	bt.ProcessBlock(b2)
	if committed != 0 {
		t.Fatalf("expected 0 commits after b2, got %d", committed)
	}

	bt.ProcessBlock(b3) // b3 has QC for b2. genericBlock=b2, lockedBlock=b1. Fast commits b1.
	if committed != 1 {
		t.Fatalf("expected b1 to be fast-committed (height 1), got %d", committed)
	}

	// Test 2: Fallback to 3-chain when NumSignatures < 4
	bt2 := NewBlockTree(storage.NewStorageManager(0), exec)
	bt2.SetFastPathThreshold(4)
	var committed2 uint64
	bt2.SetCommitCallback(func(block *types.Block) {
		committed2 = block.Height
	})

	b1_2 := types.NewBlock(1, make([]byte, 32), []byte("b1_2"), 1, 1, 0, 0, types.NewQuorumCertificate(nil, 0, 0, types.PhaseDecide), nil)
	qc1_2 := types.NewQuorumCertificate(b1_2.Hash, 1, 1, types.PhasePrepare)

	b2_2 := types.NewBlock(2, b1_2.Hash, []byte("b2_2"), 2, 1, 0, 0, qc1_2, nil)
	qc2_2 := types.NewQuorumCertificate(b2_2.Hash, 2, 1, types.PhasePrepare)
	qc2_2.NumSignatures = 2 // Below 90% threshold (3), no fast commit

	b3_2 := types.NewBlock(3, b2_2.Hash, []byte("b3_2"), 3, 1, 0, 0, qc2_2, nil)
	qc3_2 := types.NewQuorumCertificate(b3_2.Hash, 3, 1, types.PhasePrepare)

	b4_2 := types.NewBlock(4, b3_2.Hash, []byte("b4_2"), 4, 1, 0, 0, qc3_2, nil)

	bt2.ProcessBlock(b1_2)
	bt2.ProcessBlock(b2_2)
	bt2.ProcessBlock(b3_2)
	if committed2 != 0 {
		t.Fatalf("expected 0 commits because not full signatures, got %d", committed2)
	}

	bt2.ProcessBlock(b4_2) // b4 has QC for b3. generic=b3, locked=b2, commitBlock=b1.
	if committed2 != 1 {
		t.Fatalf("expected 3-chain fallback to commit height 1, got %d", committed2)
	}
}
