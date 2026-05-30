package adaptive

import (
	"math"
	"testing"
)

func TestRegretTracker_BasicAccumulation(t *testing.T) {
	rt := NewRegretTracker(RegretConfig{RewardUpperBound: 10.0})

	// Observe rewards below upper bound → regret accumulates
	for i := 0; i < 100; i++ {
		rt.Observe(5.0) // regret = 10-5 = 5 per step
	}

	if rt.T() != 100 {
		t.Fatalf("expected T=100, got %d", rt.T())
	}
	if abs(rt.CumulativeRegret()-500.0) > 1e-9 {
		t.Fatalf("expected regret=500, got %f", rt.CumulativeRegret())
	}
	if abs(rt.NormalizedRegret()-5.0) > 1e-9 {
		t.Fatalf("expected normalized=5.0, got %f", rt.NormalizedRegret())
	}
}

func TestRegretTracker_RegretPerSqrtT(t *testing.T) {
	rt := NewRegretTracker(RegretConfig{RewardUpperBound: 10.0})

	// 100 steps with regret=5 per step → R(100)=500, R(T)/√T = 500/10 = 50
	for i := 0; i < 100; i++ {
		rt.Observe(5.0)
	}

	expected := 500.0 / math.Sqrt(100.0) // = 50
	if abs(rt.RegretPerSqrtT()-expected) > 1e-9 {
		t.Fatalf("expected regret/√T=%f, got %f", expected, rt.RegretPerSqrtT())
	}
}

func TestRegretTracker_ZeroRegretWhenOptimal(t *testing.T) {
	rt := NewRegretTracker(RegretConfig{RewardUpperBound: 5.0})

	// Reward = upper bound → zero regret
	for i := 0; i < 50; i++ {
		rt.Observe(5.0)
	}

	if rt.CumulativeRegret() != 0 {
		t.Fatalf("expected zero regret at optimal, got %f", rt.CumulativeRegret())
	}
}

func TestRegretTracker_ExceedingUpperBoundClamps(t *testing.T) {
	rt := NewRegretTracker(RegretConfig{RewardUpperBound: 5.0})

	// Reward > upper bound → instantaneous regret clamped to 0
	rt.Observe(8.0)
	if rt.CumulativeRegret() != 0 {
		t.Fatalf("expected zero regret when exceeding bound, got %f", rt.CumulativeRegret())
	}
}

func TestRegretTracker_AdaptiveUpperBound(t *testing.T) {
	// No explicit upper bound → uses max observed
	rt := NewRegretTracker(RegretConfig{})

	rt.Observe(10.0) // max=10, regret=0
	rt.Observe(3.0)  // max=10, regret=7
	rt.Observe(5.0)  // max=10, regret=5
	rt.Observe(10.0) // max=10, regret=0

	if abs(rt.CumulativeRegret()-12.0) > 1e-9 {
		t.Fatalf("expected regret=12 (7+5), got %f", rt.CumulativeRegret())
	}
}

func TestRegretTracker_WindowAverage(t *testing.T) {
	rt := NewRegretTracker(RegretConfig{WindowSize: 5})

	rewards := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	for _, r := range rewards {
		rt.Observe(r)
	}

	// Window holds last 5: [6,7,8,9,10] → avg=8
	avg := rt.WindowAverageReward()
	if abs(avg-8.0) > 1e-9 {
		t.Fatalf("expected window avg=8.0, got %f", avg)
	}
}

func TestRegretTracker_Snapshot(t *testing.T) {
	rt := NewRegretTracker(RegretConfig{RewardUpperBound: 10.0})
	rt.Observe(3.0)
	rt.Observe(7.0)

	snap := rt.Snapshot()
	if snap.T != 2 {
		t.Fatalf("expected T=2, got %d", snap.T)
	}
	// Regret: (10-3) + (10-7) = 7+3 = 10
	if abs(snap.CumulativeRegret-10.0) > 1e-9 {
		t.Fatalf("expected regret=10, got %f", snap.CumulativeRegret)
	}
	if snap.MaxObservedReward != 7.0 {
		t.Fatalf("expected max=7, got %f", snap.MaxObservedReward)
	}
}

func TestRegretTracker_ConcurrentSafety(t *testing.T) {
	rt := NewRegretTracker(RegretConfig{RewardUpperBound: 10.0})
	done := make(chan struct{})

	// Concurrent writes
	go func() {
		for i := 0; i < 1000; i++ {
			rt.Observe(float64(i % 10))
		}
		done <- struct{}{}
	}()
	// Concurrent reads
	go func() {
		for i := 0; i < 1000; i++ {
			_ = rt.CumulativeRegret()
			_ = rt.NormalizedRegret()
			_ = rt.Snapshot()
		}
		done <- struct{}{}
	}()

	<-done
	<-done

	if rt.T() != 1000 {
		t.Fatalf("expected T=1000, got %d", rt.T())
	}
}
