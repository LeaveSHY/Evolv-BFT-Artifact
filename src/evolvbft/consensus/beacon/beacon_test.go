package beacon

import (
	"testing"
)

func TestSelectLeader_DifferentLanesDifferentLeaders(t *testing.T) {
	rb := NewRandomBeacon([]byte("test-seed-for-lane-diversity"))
	validators := []uint64{1, 2, 3, 4, 5, 6, 7}

	// For a given view, different lanes should (statistically) produce
	// different leaders. With 4 lanes and 7 validators the probability
	// of all 4 colliding is (1/7)^3 ≈ 0.3% — negligible.
	const view = 10
	const numLanes = 4
	leaders := make(map[uint64]bool)
	for lane := uint64(0); lane < numLanes; lane++ {
		l := rb.SelectLeader(view, lane, validators)
		leaders[l] = true
	}
	if len(leaders) < 2 {
		t.Errorf("expected at least 2 distinct leaders across %d lanes, got %d", numLanes, len(leaders))
	}
}

func TestSelectLeader_SameLaneDeterministic(t *testing.T) {
	rb := NewRandomBeacon([]byte("deterministic-seed"))
	validators := []uint64{10, 20, 30, 40}

	l1 := rb.SelectLeader(5, 2, validators)
	l2 := rb.SelectLeader(5, 2, validators)
	if l1 != l2 {
		t.Errorf("same (view,lane) should return same leader, got %d vs %d", l1, l2)
	}
}

func TestSelectLeader_DifferentViewsDifferentLeaders(t *testing.T) {
	rb := NewRandomBeacon([]byte("view-diversity-seed"))
	validators := []uint64{1, 2, 3, 4, 5}

	leaders := make(map[uint64]bool)
	for v := uint64(1); v <= 10; v++ {
		leaders[rb.SelectLeader(v, 0, validators)] = true
	}
	// Over 10 views with 5 validators, at least 2 distinct leaders expected.
	if len(leaders) < 2 {
		t.Errorf("expected at least 2 distinct leaders across 10 views, got %d", len(leaders))
	}
}

func TestSelectLeader_EdgeCases(t *testing.T) {
	rb := NewRandomBeacon([]byte("edge"))

	// Empty validator list → 0
	if l := rb.SelectLeader(1, 0, nil); l != 0 {
		t.Errorf("empty validators should return 0, got %d", l)
	}

	// Single validator → always that validator
	if l := rb.SelectLeader(1, 0, []uint64{42}); l != 42 {
		t.Errorf("single validator should return 42, got %d", l)
	}
	if l := rb.SelectLeader(99, 5, []uint64{42}); l != 42 {
		t.Errorf("single validator should return 42 regardless of view/lane, got %d", l)
	}
}
