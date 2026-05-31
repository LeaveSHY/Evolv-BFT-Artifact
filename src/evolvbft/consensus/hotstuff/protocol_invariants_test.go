package hotstuff

import (
	"fmt"
	"testing"
	"time"

	"evolvbft/evolvbft/storage"
	"evolvbft/evolvbft/types"
)

func assertProtocolInvariants(orderLog []InstanceOutput, bt *BlockTree, numInstances uint64) error {
	if numInstances == 0 {
		return fmt.Errorf("numInstances must be > 0")
	}
	if len(orderLog) == 0 {
		return fmt.Errorf("order log must be non-empty")
	}
	prevRank := orderLog[0].Rank
	for i, entry := range orderLog {
		if i > 0 && entry.Rank != prevRank+1 {
			return fmt.Errorf("rank continuity violated at index %d: prev=%d curr=%d", i, prevRank, entry.Rank)
		}
		if entry.Rank%numInstances != entry.InstanceID {
			return fmt.Errorf("instance/rank mapping violated at rank %d", entry.Rank)
		}
		if !entry.IsNil {
			if entry.Block == nil {
				return fmt.Errorf("non-nil entry missing block at rank %d", entry.Rank)
			}
			expected := entry.Block.Height*numInstances + entry.InstanceID
			if expected != entry.Rank {
				return fmt.Errorf("block rank mismatch at rank %d: expected %d", entry.Rank, expected)
			}
			for _, transition := range entry.EpochTransitions {
				if transition == nil {
					return fmt.Errorf("nil epoch transition at rank %d", entry.Rank)
				}
				if transition.ActivationRank != entry.Rank {
					return fmt.Errorf("epoch transition rank mismatch at rank %d", entry.Rank)
				}
			}
		}
		prevRank = entry.Rank
	}
	if bt == nil {
		return nil
	}
	high := bt.GetHighQC()
	locked := bt.GetLockedQC()
	commit := bt.GetCommitQC()
	if !qcAtLeast(high, locked) {
		return fmt.Errorf("highQC must be >= lockedQC")
	}
	if !qcAtLeast(locked, commit) {
		return fmt.Errorf("lockedQC must be >= commitQC")
	}
	return nil
}

func qcAtLeast(a, b *types.QuorumCertificate) bool {
	if a == nil {
		return false
	}
	if b == nil {
		return true
	}
	if a.Epoch != b.Epoch {
		return a.Epoch > b.Epoch
	}
	return a.View >= b.View
}

func TestProtocolInvariants_NilEntriesPreserveRankAndInstanceMapping(t *testing.T) {
	in := make(chan InstanceOutput, 16)
	orderer := NewGlobalOrderer(2, 20*time.Millisecond)
	out := orderer.Start(in)

	in <- InstanceOutput{
		InstanceID:  0,
		LocalHeight: 1,
		Rank:        2,
		Block:       types.NewBlock(1, nil, []byte("a"), 1, 1, 0, 2, types.NewQuorumCertificate(nil, 0, 0, types.PhaseDecide), nil),
	}
	in <- InstanceOutput{
		InstanceID:  0,
		LocalHeight: 2,
		Rank:        4,
		Block:       types.NewBlock(2, nil, []byte("b"), 2, 1, 0, 4, types.NewQuorumCertificate(nil, 1, 1, types.PhasePrepare), nil),
	}

	collected := make([]InstanceOutput, 0, 3)
	timeout := time.After(500 * time.Millisecond)
	for len(collected) < 3 {
		select {
		case entry := <-out:
			collected = append(collected, entry)
		case <-timeout:
			t.Fatalf("timed out collecting ordered outputs with nil fill")
		}
	}
	orderer.Stop()

	if !collected[1].IsNil {
		t.Fatalf("expected middle rank to be nil-filled, got %+v", collected[1])
	}
	if collected[1].Rank != 3 {
		t.Fatalf("expected nil-filled rank 3, got %d", collected[1].Rank)
	}
	if collected[1].InstanceID != 1 {
		t.Fatalf("expected nil-filled instance 1 at rank 3, got %d", collected[1].InstanceID)
	}
	if collected[1].LocalHeight != 1 {
		t.Fatalf("expected nil-filled local height 1 at rank 3, got %d", collected[1].LocalHeight)
	}

	if err := assertProtocolInvariants(collected, nil, 2); err != nil {
		t.Fatalf("protocol invariants failed for nil-filled sequence: %v", err)
	}
}
func TestProtocolInvariants_LateOutputIgnoredAfterNilFill(t *testing.T) {
	in := make(chan InstanceOutput, 16)
	orderer := NewGlobalOrderer(2, 20*time.Millisecond)
	out := orderer.Start(in)

	in <- InstanceOutput{
		InstanceID:  0,
		LocalHeight: 1,
		Rank:        2,
		Block:       types.NewBlock(1, nil, []byte("a"), 1, 1, 0, 2, types.NewQuorumCertificate(nil, 0, 0, types.PhaseDecide), nil),
	}
	in <- InstanceOutput{
		InstanceID:  0,
		LocalHeight: 2,
		Rank:        4,
		Block:       types.NewBlock(2, nil, []byte("b"), 2, 1, 0, 4, types.NewQuorumCertificate(nil, 1, 1, types.PhasePrepare), nil),
	}

	collected := make([]InstanceOutput, 0, 3)
	timeout := time.After(500 * time.Millisecond)
	for len(collected) < 3 {
		select {
		case entry := <-out:
			collected = append(collected, entry)
		case <-timeout:
			t.Fatalf("timed out collecting outputs before late arrival")
		}
	}

	in <- InstanceOutput{
		InstanceID:  1,
		LocalHeight: 1,
		Rank:        3,
		Block:       types.NewBlock(1, nil, []byte("late"), 1, 1, 1, 3, types.NewQuorumCertificate(nil, 0, 0, types.PhaseDecide), nil),
	}
	in <- InstanceOutput{
		InstanceID:  1,
		LocalHeight: 2,
		Rank:        5,
		Block:       types.NewBlock(2, nil, []byte("c"), 2, 1, 1, 5, types.NewQuorumCertificate(nil, 1, 1, types.PhasePrepare), nil),
	}

	select {
	case entry := <-out:
		collected = append(collected, entry)
		if entry.Rank != 5 {
			t.Fatalf("expected late rank 3 to be ignored and next emitted rank to be 5, got %d", entry.Rank)
		}
	case <-timeout:
		t.Fatalf("timed out collecting output after late arrival")
	}
	orderer.Stop()

	if !collected[1].IsNil || collected[1].Rank != 3 {
		t.Fatalf("expected rank 3 to remain nil-filled, got %+v", collected[1])
	}
	if err := assertProtocolInvariants(collected, nil, 2); err != nil {
		t.Fatalf("protocol invariants failed after late output arrival: %v", err)
	}

	emitted, nilled, late := orderer.Stats()
	if emitted != 3 {
		t.Fatalf("expected three non-nil/nil emissions recorded, got %d", emitted)
	}
	if nilled != 1 {
		t.Fatalf("expected one nil fill recorded, got %d", nilled)
	}
	if late != 1 {
		t.Fatalf("expected exactly one late output recorded, got %d", late)
	}
}
func TestProtocolInvariants_AutomatedCheck(t *testing.T) {
	in := make(chan InstanceOutput, 16)
	orderer := NewGlobalOrderer(2, 20*time.Millisecond)
	out := orderer.Start(in)

	in <- InstanceOutput{
		InstanceID:  0,
		LocalHeight: 1,
		Rank:        2,
		Block:       types.NewBlock(1, nil, []byte("a"), 1, 1, 0, 2, types.NewQuorumCertificate(nil, 0, 0, types.PhaseDecide), nil),
	}
	in <- InstanceOutput{
		InstanceID:  1,
		LocalHeight: 1,
		Rank:        3,
		Block:       types.NewBlock(1, nil, []byte("b"), 1, 1, 1, 3, types.NewQuorumCertificate(nil, 0, 0, types.PhaseDecide), nil),
	}
	in <- InstanceOutput{
		InstanceID:  1,
		LocalHeight: 2,
		Rank:        5,
		Block:       types.NewBlock(2, nil, []byte("c"), 2, 1, 1, 5, types.NewQuorumCertificate(nil, 1, 1, types.PhasePrepare), nil),
	}

	collected := make([]InstanceOutput, 0, 4)
	timeout := time.After(500 * time.Millisecond)
	for len(collected) < 4 {
		select {
		case entry := <-out:
			collected = append(collected, entry)
		case <-timeout:
			t.Fatalf("timed out collecting ordered outputs")
		}
	}
	orderer.Stop()

	validators := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: []byte("v0"), Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: []byte("v1"), Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: []byte("v2"), Power: 1, IsActive: true},
		3: {ID: 3, PublicKey: []byte("v3"), Power: 1, IsActive: true},
	}
	valSet := types.NewValidatorSet(1, validators)
	bt := NewBlockTree(storage.NewStorageManager(0), NewExecutor(valSet))

	b1 := types.NewBlock(1, make([]byte, 32), []byte("b1"), 1, 1, 0, 0, types.NewQuorumCertificate(nil, 0, 0, types.PhaseDecide), nil)
	qc1 := types.NewQuorumCertificate(b1.Hash, 1, 1, types.PhasePrepare)
	b2 := types.NewBlock(2, b1.Hash, []byte("b2"), 2, 1, 1, 0, qc1, nil)
	qc2 := types.NewQuorumCertificate(b2.Hash, 2, 1, types.PhasePrepare)
	b3 := types.NewBlock(3, b2.Hash, []byte("b3"), 3, 1, 0, 0, qc2, nil)
	qc3 := types.NewQuorumCertificate(b3.Hash, 3, 1, types.PhasePrepare)
	b4 := types.NewBlock(4, b3.Hash, []byte("b4"), 4, 1, 1, 0, qc3, nil)

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

	if err := assertProtocolInvariants(collected, bt, 2); err != nil {
		t.Fatalf("protocol invariants failed: %v", err)
	}
}
