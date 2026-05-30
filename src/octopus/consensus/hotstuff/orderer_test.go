package hotstuff

import (
	"testing"
	"time"

	"octopus-bft/octopus/types"
)

func TestGlobalOrderer_DeterministicOrdering(t *testing.T) {
	in := make(chan InstanceOutput, 16)
	o := NewGlobalOrderer(2, 50*time.Millisecond)
	out := o.Start(in)

	in <- InstanceOutput{InstanceID: 1, LocalHeight: 1, Rank: 3}
	in <- InstanceOutput{InstanceID: 0, LocalHeight: 1, Rank: 2}
	in <- InstanceOutput{InstanceID: 0, LocalHeight: 2, Rank: 4}
	in <- InstanceOutput{InstanceID: 1, LocalHeight: 2, Rank: 5}

	got := make([]uint64, 0, 4)
	timeout := time.After(500 * time.Millisecond)
	for len(got) < 4 {
		select {
		case e := <-out:
			got = append(got, e.Rank)
		case <-timeout:
			t.Fatalf("timeout, got=%v", got)
		}
	}

	want := []uint64{2, 3, 4, 5}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("rank mismatch at %d: got=%v want=%v", i, got, want)
		}
	}
}

func TestGlobalOrderer_NilAndLateEntry(t *testing.T) {
	in := make(chan InstanceOutput, 16)
	o := NewGlobalOrderer(2, 30*time.Millisecond)
	out := o.Start(in)

	in <- InstanceOutput{InstanceID: 0, LocalHeight: 1, Rank: 2}
	in <- InstanceOutput{InstanceID: 1, LocalHeight: 1, Rank: 3}
	in <- InstanceOutput{InstanceID: 1, LocalHeight: 2, Rank: 5}

	var seenNil bool
	var nilRank uint64
	timeout := time.After(500 * time.Millisecond)
	for !seenNil {
		select {
		case e := <-out:
			if e.IsNil {
				seenNil = true
				nilRank = e.Rank
			}
		case <-timeout:
			t.Fatal("timeout waiting for nil")
		}
	}
	if nilRank != 4 {
		t.Fatalf("unexpected nil rank: %d", nilRank)
	}

	in <- InstanceOutput{InstanceID: 0, LocalHeight: 2, Rank: 4}

	deadline := time.Now().Add(500 * time.Millisecond)
	for {
		_, nilled, late := o.Stats()
		if nilled > 0 && late > 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("expected nilled>0 and late>0, got nilled=%d late=%d", nilled, late)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func TestGlobalOrderer_PendingCapPrefersLowerRank(t *testing.T) {
	in := make(chan InstanceOutput, 16)
	o := NewGlobalOrdererWithLimit(2, 40*time.Millisecond, 2)
	out := o.Start(in)

	in <- InstanceOutput{InstanceID: 1, LocalHeight: 3, Rank: 7}
	in <- InstanceOutput{InstanceID: 0, LocalHeight: 3, Rank: 6}
	in <- InstanceOutput{InstanceID: 1, LocalHeight: 2, Rank: 5}
	in <- InstanceOutput{InstanceID: 0, LocalHeight: 1, Rank: 2}
	in <- InstanceOutput{InstanceID: 1, LocalHeight: 1, Rank: 3}

	got := make([]uint64, 0, 3)
	timeout := time.After(600 * time.Millisecond)
	for len(got) < 3 {
		select {
		case e := <-out:
			got = append(got, e.Rank)
		case <-timeout:
			t.Fatalf("timeout waiting ordered outputs, got=%v", got)
		}
	}
	if got[0] != 2 || got[1] != 3 {
		t.Fatalf("unexpected ordered prefix: %v", got)
	}
	if o.Dropped() == 0 {
		t.Fatalf("expected pending cap to drop at least one entry")
	}
}

func TestGlobalOrderer_StaleDuplicateAfterRealEmitDoesNotAlterLogPrefix(t *testing.T) {
	in := make(chan InstanceOutput, 16)
	o := NewGlobalOrderer(2, 30*time.Millisecond)
	out := o.Start(in)

	in <- InstanceOutput{InstanceID: 0, LocalHeight: 1, Rank: 2}
	in <- InstanceOutput{InstanceID: 1, LocalHeight: 1, Rank: 3}
	in <- InstanceOutput{InstanceID: 0, LocalHeight: 2, Rank: 4}

	got := make([]InstanceOutput, 0, 3)
	timeout := time.After(500 * time.Millisecond)
	for len(got) < 3 {
		select {
		case entry := <-out:
			got = append(got, entry)
		case <-timeout:
			t.Fatalf("timeout waiting for initial ordered outputs, got=%+v", got)
		}
	}

	prefixBefore := o.Log()
	if len(prefixBefore) != 3 {
		t.Fatalf("expected ordered log prefix of length 3, got %d", len(prefixBefore))
	}

	in <- InstanceOutput{InstanceID: 1, LocalHeight: 1, Rank: 3, Block: types.NewBlock(1, nil, []byte("stale-3"), 1, 1, 1, 3, types.NewQuorumCertificate(nil, 0, 0, types.PhaseDecide), nil)}
	in <- InstanceOutput{InstanceID: 0, LocalHeight: 2, Rank: 4, Block: types.NewBlock(2, nil, []byte("stale-4"), 2, 1, 0, 4, types.NewQuorumCertificate(nil, 1, 1, types.PhasePrepare), nil)}
	in <- InstanceOutput{InstanceID: 1, LocalHeight: 2, Rank: 5}

	select {
	case entry := <-out:
		if entry.Rank != 5 {
			t.Fatalf("expected next emitted rank to remain 5 after stale duplicates, got %d", entry.Rank)
		}
	case <-timeout:
		t.Fatalf("timeout waiting for next ordered output after stale duplicates")
	}

	prefixAfter := o.Log()
	if len(prefixAfter) != 4 {
		t.Fatalf("expected log length 4 after new rank 5 emit, got %d", len(prefixAfter))
	}
	for i := 0; i < len(prefixBefore); i++ {
		if prefixAfter[i].Rank != prefixBefore[i].Rank || prefixAfter[i].IsNil != prefixBefore[i].IsNil {
			t.Fatalf("stale duplicates altered ordered log prefix at index %d: before=%+v after=%+v", i, prefixBefore[i], prefixAfter[i])
		}
	}

	emitted, nilled, late := o.Stats()
	if emitted != 4 {
		t.Fatalf("expected four emitted entries after rank 5, got %d", emitted)
	}
	if nilled != 0 {
		t.Fatalf("expected no nil fills in stale duplicate test, got %d", nilled)
	}
	if late != 2 {
		t.Fatalf("expected two stale duplicates recorded as late, got %d", late)
	}
}
func TestGlobalOrderer_MultipleLateArrivalsAfterConsecutiveNilFillsKeepAppendOnlyLog(t *testing.T) {
	in := make(chan InstanceOutput, 16)
	o := NewGlobalOrderer(2, 25*time.Millisecond)
	out := o.Start(in)

	in <- InstanceOutput{InstanceID: 0, LocalHeight: 1, Rank: 2}
	in <- InstanceOutput{InstanceID: 1, LocalHeight: 3, Rank: 7}

	got := make([]InstanceOutput, 0, 6)
	timeout := time.After(800 * time.Millisecond)
	for len(got) < 6 {
		select {
		case entry := <-out:
			got = append(got, entry)
		case <-timeout:
			t.Fatalf("timeout waiting for nil-filled prefix, got=%+v", got)
		}
	}

	for i, rank := range []uint64{2, 3, 4, 5, 6, 7} {
		if got[i].Rank != rank {
			t.Fatalf("unexpected rank continuity before late arrivals at index %d: got=%d want=%d full=%+v", i, got[i].Rank, rank, got)
		}
	}
	for _, idx := range []int{1, 2, 3, 4} {
		if !got[idx].IsNil {
			t.Fatalf("expected rank %d to remain nil-filled before late arrivals, got %+v", got[idx].Rank, got[idx])
		}
	}

	prefixBefore := o.Log()
	if len(prefixBefore) != 6 {
		t.Fatalf("expected ordered log prefix of length 6, got %d", len(prefixBefore))
	}

	for _, lateEntry := range []InstanceOutput{
		{InstanceID: 1, LocalHeight: 1, Rank: 3, Block: types.NewBlock(1, nil, []byte("late-3"), 1, 1, 1, 3, types.NewQuorumCertificate(nil, 0, 0, types.PhaseDecide), nil)},
		{InstanceID: 0, LocalHeight: 2, Rank: 4, Block: types.NewBlock(2, nil, []byte("late-4"), 2, 1, 0, 4, types.NewQuorumCertificate(nil, 1, 1, types.PhasePrepare), nil)},
		{InstanceID: 1, LocalHeight: 2, Rank: 5, Block: types.NewBlock(2, nil, []byte("late-5"), 2, 1, 1, 5, types.NewQuorumCertificate(nil, 1, 1, types.PhasePrepare), nil)},
		{InstanceID: 0, LocalHeight: 3, Rank: 6, Block: types.NewBlock(3, nil, []byte("late-6"), 3, 1, 0, 6, types.NewQuorumCertificate(nil, 2, 1, types.PhasePrepare), nil)},
	} {
		in <- lateEntry
	}
	in <- InstanceOutput{InstanceID: 0, LocalHeight: 4, Rank: 8}

	select {
	case entry := <-out:
		if entry.Rank != 8 {
			t.Fatalf("expected next emitted rank to remain 8 after batched late arrivals, got %d", entry.Rank)
		}
	case <-timeout:
		t.Fatalf("timeout waiting for next ordered output after batched late arrivals")
	}

	prefixAfter := o.Log()
	if len(prefixAfter) != 7 {
		t.Fatalf("expected log length 7 after rank 8 emit, got %d", len(prefixAfter))
	}
	for i := 0; i < len(prefixBefore); i++ {
		if prefixAfter[i].Rank != prefixBefore[i].Rank || prefixAfter[i].IsNil != prefixBefore[i].IsNil {
			t.Fatalf("batched late arrivals altered ordered log prefix at index %d: before=%+v after=%+v", i, prefixBefore[i], prefixAfter[i])
		}
	}

	emitted, nilled, late := o.Stats()
	if emitted != 3 {
		t.Fatalf("expected three real emissions (ranks 2, 7, 8), got %d", emitted)
	}
	if nilled != 4 {
		t.Fatalf("expected four nil fills (ranks 3-6), got %d", nilled)
	}
	if late != 4 {
		t.Fatalf("expected four batched late arrivals recorded, got %d", late)
	}
}
func TestGlobalOrderer_RejectsInconsistentInstanceRankMapping(t *testing.T) {
	in := make(chan InstanceOutput, 16)
	o := NewGlobalOrderer(2, 30*time.Millisecond)
	out := o.Start(in)

	in <- InstanceOutput{InstanceID: 1, LocalHeight: 1, Rank: 2}
	in <- InstanceOutput{InstanceID: 1, LocalHeight: 1, Rank: 3}
	in <- InstanceOutput{InstanceID: 0, LocalHeight: 2, Rank: 4}

	got := make([]InstanceOutput, 0, 3)
	timeout := time.After(600 * time.Millisecond)
	for len(got) < 3 {
		select {
		case entry := <-out:
			got = append(got, entry)
		case <-timeout:
			t.Fatalf("timeout waiting ordered outputs, got=%+v", got)
		}
	}

	if !got[0].IsNil || got[0].Rank != 2 {
		t.Fatalf("expected rank 2 to nil-fill after inconsistent mapping is rejected, got %+v", got[0])
	}
	if got[1].Rank != 3 || got[1].InstanceID != 1 || got[1].LocalHeight != 1 || got[1].IsNil {
		t.Fatalf("expected valid rank 3 entry to survive, got %+v", got[1])
	}
	if got[2].Rank != 4 || got[2].InstanceID != 0 || got[2].LocalHeight != 2 || got[2].IsNil {
		t.Fatalf("expected valid rank 4 entry to survive, got %+v", got[2])
	}
}

func TestGlobalOrderer_RejectsInconsistentLocalHeightRankMapping(t *testing.T) {
	in := make(chan InstanceOutput, 16)
	o := NewGlobalOrderer(2, 30*time.Millisecond)
	out := o.Start(in)

	in <- InstanceOutput{InstanceID: 0, LocalHeight: 2, Rank: 2}
	in <- InstanceOutput{InstanceID: 1, LocalHeight: 1, Rank: 3}
	in <- InstanceOutput{InstanceID: 0, LocalHeight: 2, Rank: 4}

	got := make([]InstanceOutput, 0, 3)
	timeout := time.After(600 * time.Millisecond)
	for len(got) < 3 {
		select {
		case entry := <-out:
			got = append(got, entry)
		case <-timeout:
			t.Fatalf("timeout waiting ordered outputs, got=%+v", got)
		}
	}

	if !got[0].IsNil || got[0].Rank != 2 {
		t.Fatalf("expected rank 2 to nil-fill after inconsistent local height is rejected, got %+v", got[0])
	}
	if got[1].Rank != 3 || got[1].InstanceID != 1 || got[1].LocalHeight != 1 || got[1].IsNil {
		t.Fatalf("expected valid rank 3 entry to survive, got %+v", got[1])
	}
	if got[2].Rank != 4 || got[2].InstanceID != 0 || got[2].LocalHeight != 2 || got[2].IsNil {
		t.Fatalf("expected valid rank 4 entry to survive, got %+v", got[2])
	}
}

func TestGlobalOrderer_PreservesEpochTransitionsInLog(t *testing.T) {
	in := make(chan InstanceOutput, 4)
	o := NewGlobalOrderer(2, 50*time.Millisecond)
	out := o.Start(in)

	transition := &types.EpochTransition{
		OldEpoch:         1,
		NewEpoch:         2,
		ActivationHeight: 1,
		ActivationRank:   2,
	}
	in <- InstanceOutput{
		InstanceID:       0,
		LocalHeight:      1,
		Rank:             2,
		EpochTransitions: []*types.EpochTransition{transition},
	}
	in <- InstanceOutput{InstanceID: 1, LocalHeight: 1, Rank: 3}

	select {
	case got := <-out:
		if len(got.EpochTransitions) != 1 || got.EpochTransitions[0].NewEpoch != 2 {
			t.Fatalf("epoch transitions not preserved in output: %+v", got.EpochTransitions)
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("timeout waiting for ordered output with epoch transition")
	}
}

func TestGlobalOrderer_StartsAtFirstCommittedWaveAndNilMapsToExpectedSlot(t *testing.T) {
	in := make(chan InstanceOutput, 8)
	o := NewGlobalOrderer(3, 30*time.Millisecond)
	out := o.Start(in)

	in <- InstanceOutput{InstanceID: 0, LocalHeight: 1, Rank: 3}
	in <- InstanceOutput{InstanceID: 1, LocalHeight: 1, Rank: 4}

	var entries []InstanceOutput
	timeout := time.After(500 * time.Millisecond)
	for len(entries) < 3 {
		select {
		case e := <-out:
			entries = append(entries, e)
		case <-timeout:
			t.Fatalf("timeout waiting for first wave, got=%+v", entries)
		}
	}

	if entries[0].Rank != 3 || entries[1].Rank != 4 {
		t.Fatalf("unexpected first ordered prefix: %+v", entries)
	}
	if !entries[2].IsNil {
		t.Fatalf("expected third entry to be nil-filled, got %+v", entries[2])
	}
	if entries[2].Rank != 5 || entries[2].InstanceID != 2 || entries[2].LocalHeight != 1 {
		t.Fatalf("unexpected nil slot mapping: %+v", entries[2])
	}
}

func TestGlobalOrderer_NilFillMaintainsRankContinuityAcrossGaps(t *testing.T) {
	in := make(chan InstanceOutput, 8)
	o := NewGlobalOrderer(2, 25*time.Millisecond)
	out := o.Start(in)

	in <- InstanceOutput{InstanceID: 0, LocalHeight: 1, Rank: 2}
	in <- InstanceOutput{InstanceID: 1, LocalHeight: 2, Rank: 5}

	got := make([]InstanceOutput, 0, 4)
	timeout := time.After(700 * time.Millisecond)
	for len(got) < 4 {
		select {
		case entry := <-out:
			got = append(got, entry)
		case <-timeout:
			t.Fatalf("timeout waiting for nil-filled sequence, got=%+v", got)
		}
	}

	for i, rank := range []uint64{2, 3, 4, 5} {
		if got[i].Rank != rank {
			t.Fatalf("rank continuity violated at index %d: got=%d want=%d full=%+v", i, got[i].Rank, rank, got)
		}
	}
	if got[1].IsNil != true || got[2].IsNil != true || got[3].IsNil {
		t.Fatalf("expected ranks 3 and 4 to be nil-filled only, got=%+v", got)
	}
	if got[1].InstanceID != 1 || got[1].LocalHeight != 1 {
		t.Fatalf("unexpected nil slot mapping at rank 3: %+v", got[1])
	}
	if got[2].InstanceID != 0 || got[2].LocalHeight != 2 {
		t.Fatalf("unexpected nil slot mapping at rank 4: %+v", got[2])
	}
	if got[3].InstanceID != 1 || got[3].LocalHeight != 2 {
		t.Fatalf("unexpected resumed real entry at rank 5: %+v", got[3])
	}
}
