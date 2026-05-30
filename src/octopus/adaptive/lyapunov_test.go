package adaptive

import (
	"math"
	"sync"
	"testing"
)

func TestLyapunov_ZeroStateReturnsZero(t *testing.T) {
	lm := NewLyapunovMonitor(DefaultLyapunovConfig())
	snap := lm.Evaluate(LyapunovState{
		InstanceFaults:   []int{0, 0},
		InstanceSizes:    []int{10, 10},
		NormalizedRegret: 0,
	})
	if snap.Value != 0 {
		t.Errorf("all-safe state should have V=0, got %f", snap.Value)
	}
	if snap.IsViolation {
		t.Error("first step should never be a violation")
	}
}

func TestLyapunov_SafetyTermComputation(t *testing.T) {
	cfg := LyapunovConfig{
		WeightSafety:       1.0,
		WeightRegret:       0,
		DeltaS:             1,
		ViolationThreshold: 0.01,
	}
	lm := NewLyapunovMonitor(cfg)

	// Instance with n=4, f=1: margin = 3*1+1+1-4 = 1 (violation!)
	// Instance with n=7, f=2: margin = 3*2+1+1-7 = 1 (violation!)
	snap := lm.Evaluate(LyapunovState{
		InstanceFaults: []int{1, 2},
		InstanceSizes:  []int{4, 7},
	})
	// Safety = max(0,1) + max(0,1) = 2
	if math.Abs(snap.SafetyTerm-2.0) > 1e-9 {
		t.Errorf("expected safety_term=2.0, got %f", snap.SafetyTerm)
	}
	if math.Abs(snap.Value-2.0) > 1e-9 {
		t.Errorf("expected V=2.0 (weight_regret=0), got %f", snap.Value)
	}
}

func TestLyapunov_SafetyTermZeroWhenQuorumSafe(t *testing.T) {
	cfg := LyapunovConfig{
		WeightSafety:       1.0,
		WeightRegret:       0,
		DeltaS:             1,
		ViolationThreshold: 0.01,
	}
	lm := NewLyapunovMonitor(cfg)

	// Instance with n=10, f=2: margin = 3*2+1+1-10 = -2 (safe)
	snap := lm.Evaluate(LyapunovState{
		InstanceFaults: []int{2},
		InstanceSizes:  []int{10},
	})
	if snap.SafetyTerm != 0 {
		t.Errorf("safe instance should have safety_term=0, got %f", snap.SafetyTerm)
	}
}

func TestLyapunov_RegretTermContributes(t *testing.T) {
	cfg := LyapunovConfig{
		WeightSafety:       0,
		WeightRegret:       1.0,
		DeltaS:             1,
		ViolationThreshold: 0.01,
	}
	lm := NewLyapunovMonitor(cfg)

	snap := lm.Evaluate(LyapunovState{
		InstanceFaults:   []int{0},
		InstanceSizes:    []int{10},
		NormalizedRegret: 0.5,
	})
	if math.Abs(snap.Value-0.5) > 1e-9 {
		t.Errorf("expected V=0.5 from regret alone, got %f", snap.Value)
	}
}

func TestLyapunov_MonotonicDecreaseNoViolation(t *testing.T) {
	lm := NewLyapunovMonitor(DefaultLyapunovConfig())

	// Decreasing sequence: V = 3, 2, 1, 0
	states := []LyapunovState{
		{InstanceFaults: []int{2}, InstanceSizes: []int{4}, NormalizedRegret: 0},  // 3*2+1+1-4=4
		{InstanceFaults: []int{1}, InstanceSizes: []int{4}, NormalizedRegret: 0},  // 3*1+1+1-4=1
		{InstanceFaults: []int{0}, InstanceSizes: []int{4}, NormalizedRegret: 0},  // 0
		{InstanceFaults: []int{0}, InstanceSizes: []int{10}, NormalizedRegret: 0}, // 0
	}

	for i, s := range states {
		snap := lm.Evaluate(s)
		if snap.IsViolation {
			t.Errorf("step %d should not be a violation (monotonic decrease)", i)
		}
	}
	if lm.Violations() != 0 {
		t.Errorf("expected 0 violations, got %d", lm.Violations())
	}
	if lm.MonotonicStreak() != 4 {
		t.Errorf("expected monotonic streak=4, got %d", lm.MonotonicStreak())
	}
}

func TestLyapunov_ViolationDetected(t *testing.T) {
	cfg := LyapunovConfig{
		WeightSafety:       1.0,
		WeightRegret:       0,
		DeltaS:             1,
		ViolationThreshold: 0.01,
	}
	lm := NewLyapunovMonitor(cfg)

	// Start with safe state: V=0
	lm.Evaluate(LyapunovState{
		InstanceFaults: []int{0},
		InstanceSizes:  []int{10},
	})

	// Now V increases (safety degraded): violation!
	snap := lm.Evaluate(LyapunovState{
		InstanceFaults: []int{3},
		InstanceSizes:  []int{4}, // margin = 3*3+1+1-4 = 7
	})

	if !snap.IsViolation {
		t.Error("V increase should trigger violation")
	}
	if lm.Violations() != 1 {
		t.Errorf("expected 1 violation, got %d", lm.Violations())
	}
	if snap.Delta <= 0 {
		t.Errorf("expected positive delta, got %f", snap.Delta)
	}
}

func TestLyapunov_ViolationThresholdTolerance(t *testing.T) {
	cfg := LyapunovConfig{
		WeightSafety:       1.0,
		WeightRegret:       1.0,
		DeltaS:             1,
		ViolationThreshold: 0.1, // tolerate small increases
	}
	lm := NewLyapunovMonitor(cfg)

	lm.Evaluate(LyapunovState{
		InstanceFaults:   []int{0},
		InstanceSizes:    []int{10},
		NormalizedRegret: 0.5,
	})

	// Tiny increase in regret (0.05 < threshold 0.1): should NOT be violation
	snap := lm.Evaluate(LyapunovState{
		InstanceFaults:   []int{0},
		InstanceSizes:    []int{10},
		NormalizedRegret: 0.55,
	})
	if snap.IsViolation {
		t.Error("increase within threshold should not be violation")
	}
}

func TestLyapunov_Stats(t *testing.T) {
	lm := NewLyapunovMonitor(DefaultLyapunovConfig())

	// 10 decreasing steps
	for i := 10; i > 0; i-- {
		lm.Evaluate(LyapunovState{
			InstanceFaults:   []int{i},
			InstanceSizes:    []int{40},
			NormalizedRegret: float64(i) * 0.01,
		})
	}

	stats := lm.Stats()
	if stats.TotalSteps != 10 {
		t.Errorf("expected 10 steps, got %d", stats.TotalSteps)
	}
	if stats.Violations != 0 {
		t.Errorf("expected 0 violations, got %d", stats.Violations)
	}
	if !stats.IsStable {
		t.Error("expected system to be stable")
	}
	if stats.MonotonicStreak != 10 {
		t.Errorf("expected streak=10, got %d", stats.MonotonicStreak)
	}
}

func TestLyapunov_ConcurrentSafe(t *testing.T) {
	lm := NewLyapunovMonitor(DefaultLyapunovConfig())

	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			lm.Evaluate(LyapunovState{
				InstanceFaults:   []int{idx % 3},
				InstanceSizes:    []int{10},
				NormalizedRegret: float64(idx) * 0.001,
			})
			lm.Stats()
		}(i)
	}
	wg.Wait()

	if lm.TotalSteps() != 100 {
		t.Errorf("expected 100 steps, got %d", lm.TotalSteps())
	}
}
