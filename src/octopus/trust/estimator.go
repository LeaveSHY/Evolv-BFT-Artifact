package trust

// ═══════════════════════════════════════════════════════════════════════════════
// DEPRECATED: This file contains a minimal sliding-window trust scaffold used
// only during early prototyping. The production implementation is
// CombinedEstimator in combined_estimator.go (Bayesian + EWMA, Eq. 5-6).
//
// Retained for reference only. All integration paths use CombinedEstimator.
// ═══════════════════════════════════════════════════════════════════════════════

import "sync"

// Estimator is a minimal sliding-window trust scaffold.
// It is not the full paper-grade Trust Estimation Model and only maintains
// bounded in-memory behavioral counters for local aggregation.
type Estimator struct {
	mu      sync.RWMutex
	config  Config
	history map[uint64][]Observation
}

// NewEstimator constructs a scaffold with bounded per-node in-memory history.
func NewEstimator(config Config) *Estimator {
	if config.WindowSize <= 0 {
		config.WindowSize = 8
	}
	if config.MinSamples <= 0 {
		config.MinSamples = 1
	}
	if config.MinSamples > config.WindowSize {
		config.MinSamples = config.WindowSize
	}
	return &Estimator{
		config:  config,
		history: make(map[uint64][]Observation),
	}
}

// Observe appends one behavioral sample into the node's local sliding window.
func (e *Estimator) Observe(nodeID uint64, observation Observation) {
	e.mu.Lock()
	defer e.mu.Unlock()

	history := append(e.history[nodeID], observation)
	if len(history) > e.config.WindowSize {
		history = append([]Observation(nil), history[len(history)-e.config.WindowSize:]...)
	} else {
		history = append([]Observation(nil), history...)
	}
	e.history[nodeID] = history
}

// Score returns the current scaffold-level estimate if enough samples exist.
func (e *Estimator) Score(nodeID uint64) (Score, bool) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	history := e.history[nodeID]
	if len(history) < e.config.MinSamples {
		return Score{}, false
	}

	successes := 0
	failures := 0
	for _, observation := range history {
		if observation.Success {
			successes++
		}
		if observation.Timeout {
			failures++
		}
	}

	total := len(history)
	return Score{
		NodeID:             nodeID,
		SampleCount:        total,
		SuccessRate:        float64(successes) / float64(total),
		FailureProbability: float64(failures) / float64(total),
	}, true
}
