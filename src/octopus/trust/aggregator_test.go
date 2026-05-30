package trust

import (
	"testing"
)

func TestAggregator_IngestAndGlobalProb(t *testing.T) {
	agg := NewAggregator()

	// Instance 1 reports node 42 as suspicious
	ok := agg.Ingest(TrustReport{
		InstanceID: 1,
		Epoch:      5,
		FaultProbs: map[uint64]float64{42: 0.8, 43: 0.1},
	})
	if !ok {
		t.Fatal("expected ingest to succeed")
	}

	// Instance 2 reports node 42 with lower suspicion
	ok = agg.Ingest(TrustReport{
		InstanceID: 2,
		Epoch:      3,
		FaultProbs: map[uint64]float64{42: 0.3, 44: 0.9},
	})
	if !ok {
		t.Fatal("expected ingest to succeed")
	}

	// Global fault prob for node 42 should be max(0.8, 0.3) = 0.8
	prob, found := agg.GlobalFaultProb(42)
	if !found {
		t.Fatal("expected node 42 to be found")
	}
	if prob != 0.8 {
		t.Fatalf("expected global prob 0.8, got %f", prob)
	}

	// Node 44 only reported by instance 2
	prob, found = agg.GlobalFaultProb(44)
	if !found {
		t.Fatal("expected node 44 to be found")
	}
	if prob != 0.9 {
		t.Fatalf("expected global prob 0.9 for node 44, got %f", prob)
	}

	// Unknown node
	_, found = agg.GlobalFaultProb(999)
	if found {
		t.Fatal("expected unknown node not to be found")
	}
}

func TestAggregator_MonotonicEpoch(t *testing.T) {
	agg := NewAggregator()

	// Report epoch 5
	agg.Ingest(TrustReport{InstanceID: 1, Epoch: 5, FaultProbs: map[uint64]float64{10: 0.9}})

	// Try to report epoch 3 (stale) — should be rejected
	ok := agg.Ingest(TrustReport{InstanceID: 1, Epoch: 3, FaultProbs: map[uint64]float64{10: 0.1}})
	if ok {
		t.Fatal("expected stale report to be rejected")
	}

	// Same epoch (5) should also be rejected (not strictly newer)
	ok = agg.Ingest(TrustReport{InstanceID: 1, Epoch: 5, FaultProbs: map[uint64]float64{10: 0.1}})
	if ok {
		t.Fatal("expected same-epoch report to be rejected")
	}

	// Global prob should still be 0.9 (original report)
	prob, _ := agg.GlobalFaultProb(10)
	if prob != 0.9 {
		t.Fatalf("expected 0.9 after stale rejection, got %f", prob)
	}

	// Epoch 6 should succeed and override
	ok = agg.Ingest(TrustReport{InstanceID: 1, Epoch: 6, FaultProbs: map[uint64]float64{10: 0.2}})
	if !ok {
		t.Fatal("expected newer epoch to succeed")
	}
	prob, _ = agg.GlobalFaultProb(10)
	if prob != 0.2 {
		t.Fatalf("expected 0.2 after epoch 6, got %f", prob)
	}
}

func TestAggregator_FusedFaultProb(t *testing.T) {
	agg := NewAggregator()
	agg.Ingest(TrustReport{InstanceID: 1, Epoch: 1, FaultProbs: map[uint64]float64{7: 0.6}})

	// Local prob lower than global → uses global
	fused := agg.FusedFaultProb(7, 0.3)
	if fused != 0.6 {
		t.Fatalf("expected fused 0.6 (global > local), got %f", fused)
	}

	// Local prob higher than global → uses local
	fused = agg.FusedFaultProb(7, 0.9)
	if fused != 0.9 {
		t.Fatalf("expected fused 0.9 (local > global), got %f", fused)
	}

	// Unknown node → uses local as-is
	fused = agg.FusedFaultProb(999, 0.4)
	if fused != 0.4 {
		t.Fatalf("expected fused 0.4 (no global data), got %f", fused)
	}
}

func TestAggregator_GlobalFaultProbs(t *testing.T) {
	agg := NewAggregator()
	agg.Ingest(TrustReport{InstanceID: 1, Epoch: 1, FaultProbs: map[uint64]float64{1: 0.3, 2: 0.8}})
	agg.Ingest(TrustReport{InstanceID: 2, Epoch: 1, FaultProbs: map[uint64]float64{1: 0.7, 3: 0.5}})

	global := agg.GlobalFaultProbs()
	if global[1] != 0.7 {
		t.Fatalf("expected max(0.3, 0.7) = 0.7 for node 1, got %f", global[1])
	}
	if global[2] != 0.8 {
		t.Fatalf("expected 0.8 for node 2, got %f", global[2])
	}
	if global[3] != 0.5 {
		t.Fatalf("expected 0.5 for node 3, got %f", global[3])
	}
}

func TestAggregator_InstanceCount(t *testing.T) {
	agg := NewAggregator()
	if agg.InstanceCount() != 0 {
		t.Fatal("expected 0 instances initially")
	}
	agg.Ingest(TrustReport{InstanceID: 1, Epoch: 1, FaultProbs: map[uint64]float64{}})
	agg.Ingest(TrustReport{InstanceID: 2, Epoch: 1, FaultProbs: map[uint64]float64{}})
	agg.Ingest(TrustReport{InstanceID: 3, Epoch: 1, FaultProbs: map[uint64]float64{}})
	if agg.InstanceCount() != 3 {
		t.Fatalf("expected 3 instances, got %d", agg.InstanceCount())
	}
}

func TestAggregator_AttackMigration(t *testing.T) {
	// Scenario: Byzantine node 99 is detected by instance 1, then migrates to instance 2.
	// Instance 2 should see node 99's bad reputation via the aggregator.
	agg := NewAggregator()

	// Instance 1 detects node 99 as Byzantine (high fault prob)
	agg.Ingest(TrustReport{
		InstanceID: 1,
		Epoch:      10,
		FaultProbs: map[uint64]float64{99: 0.95},
	})

	// Node 99 migrates to instance 2. Instance 2 has no local data yet.
	// It queries the aggregator for global reputation.
	localProb := 0.5 // neutral prior for new node
	fused := agg.FusedFaultProb(99, localProb)

	// Fused should be 0.95 (global knowledge prevents attack migration)
	if fused != 0.95 {
		t.Fatalf("attack migration defense failed: expected fused 0.95, got %f", fused)
	}
}

func TestAggregator_ClampsProbabilities(t *testing.T) {
	agg := NewAggregator()

	// Malformed report with out-of-range values should be clamped
	agg.Ingest(TrustReport{
		InstanceID: 1,
		Epoch:      1,
		FaultProbs: map[uint64]float64{1: 1.5, 2: -0.3},
	})
	prob, _ := agg.GlobalFaultProb(1)
	if prob != 1.0 {
		t.Fatalf("expected clamped prob 1.0, got %f", prob)
	}
	prob, _ = agg.GlobalFaultProb(2)
	if prob != 0.0 {
		t.Fatalf("expected clamped prob 0.0, got %f", prob)
	}
}
