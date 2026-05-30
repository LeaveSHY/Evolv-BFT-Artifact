package trust

import (
	"math"
	"testing"
)

func TestEWMAEstimator_Basic(t *testing.T) {
	e := NewEWMAEstimator(0.3)

	// First observation: EWMA = raw
	v := e.Update(1, 0.5)
	if v != 0.5 {
		t.Fatalf("first update should equal raw: got %f", v)
	}

	// Second observation: EWMA = 0.3*0.8 + 0.7*0.5 = 0.24 + 0.35 = 0.59
	v = e.Update(1, 0.8)
	expected := 0.3*0.8 + 0.7*0.5
	if math.Abs(v-expected) > 1e-10 {
		t.Fatalf("second update: got %f, want %f", v, expected)
	}

	// Score should match
	s, ok := e.Score(1)
	if !ok || math.Abs(s-expected) > 1e-10 {
		t.Fatalf("Score: got %f/%v, want %f/true", s, ok, expected)
	}
}

func TestEWMAEstimator_Reset(t *testing.T) {
	e := NewEWMAEstimator(0.2)
	e.Update(1, 0.5)
	e.Reset(1)
	_, ok := e.Score(1)
	if ok {
		t.Fatal("Score should return false after Reset")
	}
}

func TestEWMAEstimator_DefaultAlpha(t *testing.T) {
	e := NewEWMAEstimator(0) // invalid, should default to 0.10 (Paper §III-D)
	if e.alpha != 0.10 {
		t.Fatalf("expected default alpha 0.10, got %f", e.alpha)
	}
}

func TestCombinedEstimator_FusionWeights(t *testing.T) {
	cfg := BayesianConfig{
		WindowSize: 4,
		MinSamples: 1,
		Weights: ClassifierWeights{
			W: [5]float64{2, 2, 1, 0.5, 0.3},
			B: -1.0,
		},
	}
	ce := NewCombinedEstimator(cfg, 0.3, 0.7)

	// Feed several epochs of Byzantine behavior
	for i := 0; i < 4; i++ {
		ce.ObserveEpoch(42, EpochEvent{
			Timeouts:      1,
			Equivocations: 1,
			ViewChanges:   0,
			LatencyMs:     200,
		})
	}

	prob, ok := ce.FaultProbability(42)
	if !ok {
		t.Fatal("FaultProbability should succeed after observations")
	}
	if prob < 0.5 {
		t.Fatalf("Byzantine agent should have high fault prob, got %f", prob)
	}

	// Also check that features delegate correctly
	fv, ok := ce.Features(42)
	if !ok {
		t.Fatal("Features should succeed")
	}
	if fv[0] != 1.0 { // all epochs had 1 timeout, window=4, so d/W = 4/4 = 1.0
		t.Fatalf("expected timeout rate 1.0, got %f", fv[0])
	}
}

func TestCombinedEstimator_HonestAgent(t *testing.T) {
	cfg := BayesianConfig{
		WindowSize: 4,
		MinSamples: 1,
		Weights: ClassifierWeights{
			W: [5]float64{2, 2, 1, 0.5, 0.3},
			B: -3.0, // bias toward honest
		},
	}
	ce := NewCombinedEstimator(cfg, 0.2, 0.6)

	for i := 0; i < 4; i++ {
		ce.ObserveEpoch(1, EpochEvent{
			Timeouts:      0,
			Equivocations: 0,
			ViewChanges:   0,
			LatencyMs:     50,
		})
	}

	prob, ok := ce.FaultProbability(1)
	if !ok {
		t.Fatal("should succeed")
	}
	if prob > 0.3 {
		t.Fatalf("honest agent should have low fault prob, got %f", prob)
	}
}
