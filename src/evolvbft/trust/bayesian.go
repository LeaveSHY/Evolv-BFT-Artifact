package trust

import (
	"math"
	"sync"
)

// BayesianEstimator implements the paper-grade trust estimation model
// (§III-D, Eq. 5–6): a 5-feature sliding-window with sigmoid classifier.
//
// For each agent R_k, the estimator maintains a window of W EpochEvents and
// computes the feature vector x_t^k = (d/W, e/W, v/W, τ̄, σ_τ), then outputs
// f̂_t^k = σ(w^T x + b).
type BayesianEstimator struct {
	mu      sync.RWMutex
	config  BayesianConfig
	history map[uint64][]EpochEvent // nodeID → sliding window
}

// NewBayesianEstimator creates a paper-grade trust estimator.
func NewBayesianEstimator(config BayesianConfig) *BayesianEstimator {
	if config.WindowSize <= 0 {
		config.WindowSize = 8
	}
	if config.MinSamples <= 0 {
		config.MinSamples = 1
	}
	if config.MinSamples > config.WindowSize {
		config.MinSamples = config.WindowSize
	}
	if config.MaxLatencyMs <= 0 {
		config.MaxLatencyMs = 1000 // default 1s normalization cap
	}
	return &BayesianEstimator{
		config:  config,
		history: make(map[uint64][]EpochEvent),
	}
}

// ObserveEpoch appends one epoch of consensus metadata for the given agent.
func (be *BayesianEstimator) ObserveEpoch(nodeID uint64, event EpochEvent) {
	be.mu.Lock()
	defer be.mu.Unlock()

	h := append(be.history[nodeID], event)
	if len(h) > be.config.WindowSize {
		h = append([]EpochEvent(nil), h[len(h)-be.config.WindowSize:]...)
	} else {
		h = append([]EpochEvent(nil), h...)
	}
	be.history[nodeID] = h
}

// Features computes the 5-dimensional feature vector (Eq. 5) from the
// agent's sliding window. Returns false if insufficient samples.
func (be *BayesianEstimator) Features(nodeID uint64) (FeatureVector, bool) {
	be.mu.RLock()
	defer be.mu.RUnlock()

	h := be.history[nodeID]
	if len(h) < be.config.MinSamples {
		return FeatureVector{}, false
	}

	w := float64(len(h))
	var totalD, totalE, totalV int
	var sumLat, sumLat2 float64

	for _, ev := range h {
		totalD += ev.Timeouts
		totalE += ev.Equivocations
		totalV += ev.ViewChanges
		sumLat += ev.LatencyMs
		sumLat2 += ev.LatencyMs * ev.LatencyMs
	}

	meanLat := sumLat / w
	// σ = sqrt(E[X²] - E[X]²), clamped to 0
	variance := sumLat2/w - meanLat*meanLat
	if variance < 0 {
		variance = 0
	}
	stdLat := math.Sqrt(variance)

	// Normalize latency features to [0,1] as per paper: "feature space bounded in [0,1]^5"
	maxLat := be.config.MaxLatencyMs
	normMean := math.Min(1.0, meanLat/maxLat)
	normStd := math.Min(1.0, stdLat/maxLat)

	return FeatureVector{
		float64(totalD) / w,
		float64(totalE) / w,
		float64(totalV) / w,
		normMean,
		normStd,
	}, true
}

// FaultProbability computes f̂_t^k = σ(w^T x + b) per Eq. 6.
// Returns the probability and false if insufficient samples.
func (be *BayesianEstimator) FaultProbability(nodeID uint64) (float64, bool) {
	fv, ok := be.Features(nodeID)
	if !ok {
		return 0, false
	}
	be.mu.RLock()
	w := be.config.Weights
	be.mu.RUnlock()
	return fv.FaultProb(w.W, w.B), true
}

// UpdateWeights replaces the classifier weights (w, b).
// Called by the MARL policy during end-to-end training.
func (be *BayesianEstimator) UpdateWeights(weights ClassifierWeights) {
	be.mu.Lock()
	defer be.mu.Unlock()
	be.config.Weights = weights
}

// Weights returns the current classifier weights.
func (be *BayesianEstimator) Weights() ClassifierWeights {
	be.mu.RLock()
	defer be.mu.RUnlock()
	return be.config.Weights
}

// WindowLen returns the current window length for the given agent.
func (be *BayesianEstimator) WindowLen(nodeID uint64) int {
	be.mu.RLock()
	defer be.mu.RUnlock()
	return len(be.history[nodeID])
}

// Reset clears all history for the given agent.
func (be *BayesianEstimator) Reset(nodeID uint64) {
	be.mu.Lock()
	defer be.mu.Unlock()
	delete(be.history, nodeID)
}

// AllNodes returns all node IDs with recorded history.
func (be *BayesianEstimator) AllNodes() []uint64 {
	be.mu.RLock()
	defer be.mu.RUnlock()
	nodes := make([]uint64, 0, len(be.history))
	for id := range be.history {
		nodes = append(nodes, id)
	}
	return nodes
}

// DetectionBound returns the theoretical FNR upper bound from
// Theorem bounded-detection (Eq. kappa-det): FNR ≤ exp(-2Wη²),
// where η is the metadata-separation margin (distance from the threshold
// to either the honest baseline rate ρ₀ or the misbehavior rate ρ_min).
// The margin η satisfies: ρ₀ + η ≤ θ_high ≤ ρ_min - η.
// For backward compatibility, if called with (W, ρ) directly, we compute
// η = ρ/2 assuming a symmetric threshold placement (θ = ρ/2).
func DetectionBound(windowSize int, misbehaviorRate float64) float64 {
	return DetectionBoundMargin(windowSize, misbehaviorRate/2.0)
}

// DetectionBoundMargin returns exp(-2Wη²) directly from the separation margin η.
func DetectionBoundMargin(windowSize int, margin float64) float64 {
	return math.Exp(-2.0 * float64(windowSize) * margin * margin)
}
