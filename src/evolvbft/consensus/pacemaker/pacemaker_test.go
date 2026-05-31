package pacemaker

import (
	"testing"
)

func TestPacemakerLaneOffsetDiversifiesLeaders(t *testing.T) {
	validators := []uint64{10, 20, 30, 40}
	const view = 1

	// Create 4 pacemakers on lanes 0-3.  Without beacon (leaderSelector=nil),
	// round-robin with lane offset should give different leaders.
	leaders := make(map[uint64]bool)
	for lane := uint64(0); lane < 4; lane++ {
		pm := NewPacemakerWithLane(validators, 1000, lane)
		l := pm.GetLeader(view)
		leaders[l] = true
	}
	// 4 lanes, 4 validators → all 4 should be different
	if len(leaders) != 4 {
		t.Errorf("expected 4 distinct leaders for 4 lanes with 4 validators, got %d", len(leaders))
	}
}

func TestPacemakerLane0BackwardCompatible(t *testing.T) {
	validators := []uint64{1, 2, 3}

	pm0 := NewPacemaker(validators, 1000)     // lane=0 (default)
	pmL := NewPacemakerWithLane(validators, 1000, 0) // explicit lane=0

	for v := uint64(1); v <= 10; v++ {
		a := pm0.GetLeader(v)
		b := pmL.GetLeader(v)
		if a != b {
			t.Errorf("view %d: NewPacemaker and NewPacemakerWithLane(0) disagree: %d vs %d", v, a, b)
		}
	}
}

func TestPacemakerLeaderSelectorOverridesRoundRobin(t *testing.T) {
	validators := []uint64{10, 20, 30}
	pm := NewPacemakerWithLane(validators, 1000, 1)

	// Set a custom leader selector that always returns 99.
	pm.SetLeaderSelector(func(view uint64) uint64 { return 99 })

	for v := uint64(1); v <= 5; v++ {
		if l := pm.GetLeader(v); l != 99 {
			t.Errorf("view %d: expected custom selector result 99, got %d", v, l)
		}
	}
}
