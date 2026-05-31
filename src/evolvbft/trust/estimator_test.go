package trust

import "testing"

func TestEstimatorUpdatesTrustFromSlidingWindow(t *testing.T) {
	estimator := NewEstimator(Config{WindowSize: 4, MinSamples: 2})
	estimator.Observe(7, Observation{Success: true})
	estimator.Observe(7, Observation{Timeout: true})
	estimator.Observe(7, Observation{Success: true})

	score, ok := estimator.Score(7)
	if !ok {
		t.Fatalf("expected score for node 7")
	}
	if score.SampleCount != 3 {
		t.Fatalf("unexpected sample count: %d", score.SampleCount)
	}
	if score.SuccessRate != 2.0/3.0 {
		t.Fatalf("unexpected success rate: %f", score.SuccessRate)
	}
	if score.FailureProbability != 1.0/3.0 {
		t.Fatalf("unexpected failure probability: %f", score.FailureProbability)
	}
}

func TestEstimatorDropsOldObservationsBeyondWindow(t *testing.T) {
	estimator := NewEstimator(Config{WindowSize: 2, MinSamples: 1})
	estimator.Observe(9, Observation{Success: true})
	estimator.Observe(9, Observation{Timeout: true})
	estimator.Observe(9, Observation{Timeout: true})

	score, ok := estimator.Score(9)
	if !ok {
		t.Fatalf("expected score for node 9")
	}
	if score.SampleCount != 2 {
		t.Fatalf("unexpected sample count: %d", score.SampleCount)
	}
	if score.FailureProbability != 1.0 {
		t.Fatalf("unexpected failure probability: %f", score.FailureProbability)
	}
}

func TestEstimatorRequiresMinimumSamples(t *testing.T) {
	estimator := NewEstimator(Config{WindowSize: 4, MinSamples: 2})
	estimator.Observe(3, Observation{Success: true})
	if _, ok := estimator.Score(3); ok {
		t.Fatalf("expected insufficient samples to return no score")
	}
}
