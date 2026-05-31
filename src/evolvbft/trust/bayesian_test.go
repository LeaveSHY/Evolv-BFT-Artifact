package trust

import (
	"math"
	"testing"
)

// --- FeatureVector tests ---

func TestFeatureVectorFaultProb_ZeroWeights(t *testing.T) {
	fv := FeatureVector{0.5, 0.1, 0.2, 100.0, 20.0}
	// σ(0) = 0.5
	p := fv.FaultProb([5]float64{}, 0)
	if math.Abs(p-0.5) > 1e-9 {
		t.Fatalf("expected 0.5 for zero weights, got %f", p)
	}
}

func TestFeatureVectorFaultProb_LargeBias(t *testing.T) {
	fv := FeatureVector{0, 0, 0, 0, 0}
	// σ(10) ≈ 1
	p := fv.FaultProb([5]float64{}, 10)
	if p < 0.999 {
		t.Fatalf("expected near 1 for large positive bias, got %f", p)
	}
	// σ(-10) ≈ 0
	p = fv.FaultProb([5]float64{}, -10)
	if p > 0.001 {
		t.Fatalf("expected near 0 for large negative bias, got %f", p)
	}
}

func TestFeatureVectorFaultProb_WeightedFeatures(t *testing.T) {
	// High timeout rate with positive timeout weight → high fault prob
	fv := FeatureVector{1.0, 0, 0, 0, 0}
	w := [5]float64{5.0, 0, 0, 0, 0}
	p := fv.FaultProb(w, 0)
	// σ(5) ≈ 0.9933
	if p < 0.99 {
		t.Fatalf("expected high fault prob with high timeout weight, got %f", p)
	}
}

// --- BayesianEstimator tests ---

func TestBayesianEstimator_BasicObserveAndFeatures(t *testing.T) {
	be := NewBayesianEstimator(BayesianConfig{WindowSize: 4, MinSamples: 2})
	be.ObserveEpoch(1, EpochEvent{Timeouts: 1, Equivocations: 0, ViewChanges: 0, LatencyMs: 100})
	be.ObserveEpoch(1, EpochEvent{Timeouts: 0, Equivocations: 1, ViewChanges: 0, LatencyMs: 200})

	fv, ok := be.Features(1)
	if !ok {
		t.Fatal("expected features after 2 observations")
	}
	// d/W = 1/2 = 0.5
	if math.Abs(fv[0]-0.5) > 1e-9 {
		t.Fatalf("timeout rate: expected 0.5, got %f", fv[0])
	}
	// e/W = 1/2 = 0.5
	if math.Abs(fv[1]-0.5) > 1e-9 {
		t.Fatalf("equivocation rate: expected 0.5, got %f", fv[1])
	}
	// v/W = 0
	if fv[2] != 0 {
		t.Fatalf("view-change rate: expected 0, got %f", fv[2])
	}
	// mean latency = 150ms, normalized: 150/1000 = 0.15
	if math.Abs(fv[3]-0.15) > 1e-9 {
		t.Fatalf("mean latency: expected 0.15 (normalized), got %f", fv[3])
	}
	// std = 50ms, normalized: 50/1000 = 0.05
	if math.Abs(fv[4]-0.05) > 1e-9 {
		t.Fatalf("latency std: expected 0.05 (normalized), got %f", fv[4])
	}
}

func TestBayesianEstimator_InsufficientSamples(t *testing.T) {
	be := NewBayesianEstimator(BayesianConfig{WindowSize: 4, MinSamples: 3})
	be.ObserveEpoch(1, EpochEvent{LatencyMs: 100})
	be.ObserveEpoch(1, EpochEvent{LatencyMs: 200})

	_, ok := be.Features(1)
	if ok {
		t.Fatal("should not return features with only 2/3 samples")
	}
	_, ok = be.FaultProbability(1)
	if ok {
		t.Fatal("should not return fault prob with insufficient samples")
	}
}

func TestBayesianEstimator_SlidingWindowDropsOld(t *testing.T) {
	be := NewBayesianEstimator(BayesianConfig{WindowSize: 2, MinSamples: 1})
	be.ObserveEpoch(1, EpochEvent{Timeouts: 3, LatencyMs: 100})
	be.ObserveEpoch(1, EpochEvent{Timeouts: 0, LatencyMs: 200})
	be.ObserveEpoch(1, EpochEvent{Timeouts: 0, LatencyMs: 300})

	fv, ok := be.Features(1)
	if !ok {
		t.Fatal("expected features")
	}
	// Only last 2 epochs: timeouts=0, latency=[200,300]
	if fv[0] != 0 {
		t.Fatalf("timeout rate should be 0 after old epoch dropped, got %f", fv[0])
	}
	if math.Abs(fv[3]-0.25) > 1e-9 {
		t.Fatalf("mean latency: expected 0.25 (normalized), got %f", fv[3])
	}
}

func TestBayesianEstimator_FaultProbabilityUsesWeights(t *testing.T) {
	be := NewBayesianEstimator(BayesianConfig{
		WindowSize: 4,
		MinSamples: 1,
		Weights: ClassifierWeights{
			W: [5]float64{10.0, 5.0, 3.0, 0.01, 0.01},
			B: -2.0,
		},
	})

	// Honest agent: no faults, low latency
	be.ObserveEpoch(1, EpochEvent{Timeouts: 0, Equivocations: 0, ViewChanges: 0, LatencyMs: 50})
	p, ok := be.FaultProbability(1)
	if !ok {
		t.Fatal("expected fault probability")
	}
	// w^T x + b = 10*0 + 5*0 + 3*0 + 0.01*50 + 0.01*0 - 2 = -1.5
	// σ(-1.5) ≈ 0.1824
	if p > 0.3 {
		t.Fatalf("honest agent should have low fault prob, got %f", p)
	}

	// Byzantine agent: timeouts and equivocations
	be.ObserveEpoch(2, EpochEvent{Timeouts: 1, Equivocations: 1, ViewChanges: 0, LatencyMs: 500})
	p2, ok := be.FaultProbability(2)
	if !ok {
		t.Fatal("expected fault probability")
	}
	// w^T x + b = 10*1 + 5*1 + 3*0 + 0.01*500 + 0.01*0 - 2 = 18
	// σ(18) ≈ 1.0
	if p2 < 0.9 {
		t.Fatalf("byzantine agent should have high fault prob, got %f", p2)
	}

	// Separation
	if p2 <= p {
		t.Fatalf("byzantine should have higher fault prob than honest: %f <= %f", p2, p)
	}
}

func TestBayesianEstimator_UpdateWeights(t *testing.T) {
	be := NewBayesianEstimator(BayesianConfig{WindowSize: 4, MinSamples: 1})
	be.ObserveEpoch(1, EpochEvent{Timeouts: 1, LatencyMs: 100})

	// Default weights (all zero) → σ(0) = 0.5
	p1, _ := be.FaultProbability(1)
	if math.Abs(p1-0.5) > 1e-6 {
		t.Fatalf("expected 0.5 with zero weights, got %f", p1)
	}

	// Update weights to penalize timeouts
	be.UpdateWeights(ClassifierWeights{
		W: [5]float64{10.0, 0, 0, 0, 0},
		B: 0,
	})

	p2, _ := be.FaultProbability(1)
	// σ(10*1) ≈ 1.0
	if p2 < 0.99 {
		t.Fatalf("after weight update, high timeout should give high prob, got %f", p2)
	}
}

func TestBayesianEstimator_WindowLen(t *testing.T) {
	be := NewBayesianEstimator(BayesianConfig{WindowSize: 3, MinSamples: 1})
	if be.WindowLen(1) != 0 {
		t.Fatal("empty window should be 0")
	}
	be.ObserveEpoch(1, EpochEvent{})
	be.ObserveEpoch(1, EpochEvent{})
	if be.WindowLen(1) != 2 {
		t.Fatalf("expected 2, got %d", be.WindowLen(1))
	}
	be.ObserveEpoch(1, EpochEvent{})
	be.ObserveEpoch(1, EpochEvent{})
	if be.WindowLen(1) != 3 {
		t.Fatalf("expected capped at 3, got %d", be.WindowLen(1))
	}
}

func TestBayesianEstimator_Reset(t *testing.T) {
	be := NewBayesianEstimator(BayesianConfig{WindowSize: 4, MinSamples: 1})
	be.ObserveEpoch(1, EpochEvent{LatencyMs: 100})
	be.Reset(1)
	_, ok := be.Features(1)
	if ok {
		t.Fatal("after reset, features should not be available")
	}
}

func TestBayesianEstimator_AllNodes(t *testing.T) {
	be := NewBayesianEstimator(BayesianConfig{WindowSize: 4, MinSamples: 1})
	be.ObserveEpoch(1, EpochEvent{})
	be.ObserveEpoch(5, EpochEvent{})
	be.ObserveEpoch(9, EpochEvent{})

	nodes := be.AllNodes()
	if len(nodes) != 3 {
		t.Fatalf("expected 3 nodes, got %d", len(nodes))
	}
	seen := make(map[uint64]bool)
	for _, n := range nodes {
		seen[n] = true
	}
	for _, expected := range []uint64{1, 5, 9} {
		if !seen[expected] {
			t.Fatalf("missing node %d", expected)
		}
	}
}

func TestBayesianEstimator_MultipleAgentsConcurrent(t *testing.T) {
	be := NewBayesianEstimator(BayesianConfig{
		WindowSize: 4,
		MinSamples: 1,
		Weights: ClassifierWeights{
			W: [5]float64{5.0, 5.0, 0, 0, 0},
			B: -1.0,
		},
	})

	// Agent 1: honest
	be.ObserveEpoch(1, EpochEvent{Timeouts: 0, Equivocations: 0, LatencyMs: 50})
	// Agent 2: byzantine
	be.ObserveEpoch(2, EpochEvent{Timeouts: 1, Equivocations: 1, LatencyMs: 300})

	p1, _ := be.FaultProbability(1)
	p2, _ := be.FaultProbability(2)

	if p1 >= p2 {
		t.Fatalf("honest agent fault prob (%f) should be less than byzantine (%f)", p1, p2)
	}
}

func TestBayesianEstimator_LatencyStdComputation(t *testing.T) {
	be := NewBayesianEstimator(BayesianConfig{WindowSize: 4, MinSamples: 3})
	// Constant latency → std = 0
	be.ObserveEpoch(1, EpochEvent{LatencyMs: 100})
	be.ObserveEpoch(1, EpochEvent{LatencyMs: 100})
	be.ObserveEpoch(1, EpochEvent{LatencyMs: 100})

	fv, ok := be.Features(1)
	if !ok {
		t.Fatal("expected features")
	}
	if fv[4] != 0 {
		t.Fatalf("constant latency should have std 0, got %f", fv[4])
	}
}

// --- DetectionBound tests ---

func TestDetectionBound_HighMisbehaviorRate(t *testing.T) {
	// W=8, ρ=0.5 → exp(-8*0.25/2) = exp(-1) ≈ 0.3679
	fnr := DetectionBound(8, 0.5)
	expected := math.Exp(-1.0)
	if math.Abs(fnr-expected) > 1e-9 {
		t.Fatalf("expected %f, got %f", expected, fnr)
	}
}

func TestDetectionBound_LargeWindowLowRate(t *testing.T) {
	// W=100, ρ=0.3 → exp(-100*0.09/2) = exp(-4.5) ≈ 0.0111
	fnr := DetectionBound(100, 0.3)
	if fnr > 0.02 {
		t.Fatalf("large window should give very low FNR bound, got %f", fnr)
	}
}

func TestDetectionBound_ZeroMisbehavior(t *testing.T) {
	// ρ=0 → exp(0) = 1 (no misbehavior, cannot detect)
	fnr := DetectionBound(8, 0)
	if fnr != 1.0 {
		t.Fatalf("zero misbehavior rate should give FNR=1, got %f", fnr)
	}
}

func TestDetectionBound_DecreasesWithWindow(t *testing.T) {
	fnr4 := DetectionBound(4, 0.3)
	fnr16 := DetectionBound(16, 0.3)
	if fnr16 >= fnr4 {
		t.Fatalf("larger window should decrease FNR: W=4 → %f, W=16 → %f", fnr4, fnr16)
	}
}
