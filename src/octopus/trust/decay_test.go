package trust

import (
	"math"
	"sync"
	"testing"
	"time"
)

func TestTrustDecay_FreshObservationNoDecay(t *testing.T) {
	td := NewTrustDecay(DefaultDecayConfig())
	now := time.Now()
	td.Touch(1, now)

	// Query immediately: no decay (within MinFreshness=5s)
	result := td.Decay(1, 0.1, now.Add(2*time.Second))
	if result != 0.1 {
		t.Errorf("expected 0.1 within freshness window, got %f", result)
	}
}

func TestTrustDecay_UnknownNodeReturnsBaseline(t *testing.T) {
	td := NewTrustDecay(DefaultDecayConfig())
	now := time.Now()

	result := td.Decay(99, 0.2, now)
	if result != 0.5 {
		t.Errorf("unknown node should return baseline=0.5, got %f", result)
	}
}

func TestTrustDecay_ExponentialDecayTowardBaseline(t *testing.T) {
	cfg := DecayConfig{
		HalfLife:     10 * time.Second,
		Baseline:     0.5,
		MinFreshness: 0, // no grace period for this test
	}
	td := NewTrustDecay(cfg)
	t0 := time.Now()
	td.Touch(1, t0)

	// Initial prob: 0.1 (trusted node)
	rawProb := 0.1

	// After 1 half-life (10s): should be halfway to baseline
	// f_eff = 0.5 + (0.1 - 0.5) * exp(-ln2) = 0.5 + (-0.4)*0.5 = 0.5 - 0.2 = 0.3
	result := td.Decay(1, rawProb, t0.Add(10*time.Second))
	expected := 0.5 + (0.1-0.5)*0.5
	if math.Abs(result-expected) > 1e-9 {
		t.Errorf("after 1 half-life expected %f, got %f", expected, result)
	}

	// After 2 half-lives (20s): 0.5 + (-0.4)*0.25 = 0.4
	result2 := td.Decay(1, rawProb, t0.Add(20*time.Second))
	expected2 := 0.5 + (0.1-0.5)*0.25
	if math.Abs(result2-expected2) > 1e-9 {
		t.Errorf("after 2 half-lives expected %f, got %f", expected2, result2)
	}

	// After many half-lives: should approach baseline
	resultInf := td.Decay(1, rawProb, t0.Add(600*time.Second))
	if math.Abs(resultInf-0.5) > 0.001 {
		t.Errorf("after 60 half-lives should be ~baseline 0.5, got %f", resultInf)
	}
}

func TestTrustDecay_HighFaultDecaysDown(t *testing.T) {
	cfg := DecayConfig{
		HalfLife:     10 * time.Second,
		Baseline:     0.5,
		MinFreshness: 0,
	}
	td := NewTrustDecay(cfg)
	t0 := time.Now()
	td.Touch(1, t0)

	// High fault prob (Byzantine suspect): 0.9
	// After 1 half-life: 0.5 + (0.9-0.5)*0.5 = 0.5 + 0.2 = 0.7
	result := td.Decay(1, 0.9, t0.Add(10*time.Second))
	expected := 0.5 + (0.9-0.5)*0.5
	if math.Abs(result-expected) > 1e-9 {
		t.Errorf("high-fault after 1 half-life expected %f, got %f", expected, result)
	}
}

func TestTrustDecay_TouchResetsDecay(t *testing.T) {
	cfg := DecayConfig{
		HalfLife:     10 * time.Second,
		Baseline:     0.5,
		MinFreshness: 2 * time.Second,
	}
	td := NewTrustDecay(cfg)
	t0 := time.Now()
	td.Touch(1, t0)

	// After 30s without touch: heavy decay
	result1 := td.Decay(1, 0.1, t0.Add(30*time.Second))
	if result1 <= 0.4 {
		t.Errorf("expected significant decay after 30s, got %f", result1)
	}

	// Touch again at 30s
	td.Touch(1, t0.Add(30*time.Second))

	// Query at 31s: within freshness window, no decay
	result2 := td.Decay(1, 0.1, t0.Add(31*time.Second))
	if result2 != 0.1 {
		t.Errorf("after re-touch, expected 0.1 within freshness, got %f", result2)
	}
}

func TestTrustDecay_DecayAll(t *testing.T) {
	cfg := DecayConfig{
		HalfLife:     10 * time.Second,
		Baseline:     0.5,
		MinFreshness: 0,
	}
	td := NewTrustDecay(cfg)
	t0 := time.Now()
	td.Touch(1, t0)
	td.Touch(2, t0)

	probs := map[uint64]float64{
		1: 0.1,
		2: 0.9,
		3: 0.3, // node 3 never touched
	}

	// After 10s (1 half-life)
	results := td.DecayAll(probs, t0.Add(10*time.Second))

	// Node 1: 0.5 + (0.1-0.5)*0.5 = 0.3
	if math.Abs(results[1]-0.3) > 1e-9 {
		t.Errorf("node 1 expected 0.3, got %f", results[1])
	}
	// Node 2: 0.5 + (0.9-0.5)*0.5 = 0.7
	if math.Abs(results[2]-0.7) > 1e-9 {
		t.Errorf("node 2 expected 0.7, got %f", results[2])
	}
	// Node 3: never touched → baseline 0.5
	if results[3] != 0.5 {
		t.Errorf("node 3 (unknown) expected baseline 0.5, got %f", results[3])
	}
}

func TestTrustDecay_Staleness(t *testing.T) {
	td := NewTrustDecay(DefaultDecayConfig())
	now := time.Now()
	td.Touch(1, now.Add(-30*time.Second))

	stale := td.Staleness(1, now)
	if stale != 30*time.Second {
		t.Errorf("expected 30s staleness, got %v", stale)
	}

	// Unknown node
	stale2 := td.Staleness(99, now)
	if stale2 != -1 {
		t.Errorf("unknown node should return -1, got %v", stale2)
	}
}

func TestTrustDecay_Remove(t *testing.T) {
	td := NewTrustDecay(DefaultDecayConfig())
	now := time.Now()
	td.Touch(1, now)
	td.Remove(1)

	// After removal, should return baseline
	result := td.Decay(1, 0.1, now)
	if result != 0.5 {
		t.Errorf("after removal expected baseline 0.5, got %f", result)
	}
	if td.TrackedNodes() != 0 {
		t.Errorf("expected 0 tracked nodes after removal, got %d", td.TrackedNodes())
	}
}

func TestTrustDecay_ConcurrentSafe(t *testing.T) {
	td := NewTrustDecay(DefaultDecayConfig())
	now := time.Now()

	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(id uint64) {
			defer wg.Done()
			td.Touch(id, now)
			td.Decay(id, 0.3, now.Add(time.Second))
			td.Staleness(id, now)
		}(uint64(i))
	}
	wg.Wait()

	if td.TrackedNodes() != 100 {
		t.Errorf("expected 100 tracked nodes, got %d", td.TrackedNodes())
	}
}
