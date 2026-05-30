package hotstuff

import (
	"testing"
	"time"

	"octopus-bft/octopus/types"
)

func TestGlobalConfirmedMetrics_ObserveAndSnapshot(t *testing.T) {
	orderer := NewGlobalOrderer(2, 100*time.Millisecond)

	m := NewGlobalConfirmedMetrics(2 * time.Second)
	now := time.Now()

	// Observe a non-nil output with a timestamp
	block := &types.Block{Height: 1, Timestamp: now.Add(-50 * time.Millisecond).UnixNano()}
	m.ObserveGlobalConfirmed(InstanceOutput{Block: block}, now)

	// Observe a nil output
	m.ObserveGlobalConfirmed(InstanceOutput{IsNil: true}, now.Add(10*time.Millisecond))

	snap := m.Snapshot(orderer, map[string]uint64{"equivocation": 2})

	if snap.GlobalConfirmedTotal != 2 {
		t.Errorf("expected total 2, got %d", snap.GlobalConfirmedTotal)
	}
	if snap.GlobalConfirmedNil != 1 {
		t.Errorf("expected nil 1, got %d", snap.GlobalConfirmedNil)
	}
	if snap.RejectTotal != 2 {
		t.Errorf("expected reject total 2, got %d", snap.RejectTotal)
	}
	if snap.LatencyP50Ms <= 0 {
		t.Errorf("expected positive P50 latency, got %.2f", snap.LatencyP50Ms)
	}
}

func TestGlobalConfirmedMetrics_RecoveryDetection(t *testing.T) {
	m := NewGlobalConfirmedMetrics(100 * time.Millisecond)
	orderer := NewGlobalOrderer(1, 100*time.Millisecond)

	now := time.Now()
	m.ObserveGlobalConfirmed(InstanceOutput{Block: &types.Block{Height: 1}}, now)

	// Second observation after a gap larger than barTimeout
	m.ObserveGlobalConfirmed(InstanceOutput{Block: &types.Block{Height: 2}}, now.Add(200*time.Millisecond))

	snap := m.Snapshot(orderer, nil)
	if snap.RecoveryP50Ms <= 0 {
		t.Errorf("expected recovery sample after gap > barTimeout, got P50=%.2f", snap.RecoveryP50Ms)
	}
}

func TestQuantile_EdgeCases(t *testing.T) {
	if q := quantile(nil, 0.5); q != 0 {
		t.Errorf("nil slice should return 0, got %f", q)
	}
	if q := quantile([]float64{42}, 0.5); q != 42 {
		t.Errorf("single element should return 42, got %f", q)
	}
	samples := []float64{10, 20, 30, 40, 50}
	if q := quantile(samples, 0); q != 10 {
		t.Errorf("q=0 should return min, got %f", q)
	}
	if q := quantile(samples, 1); q != 50 {
		t.Errorf("q=1 should return max, got %f", q)
	}
}
