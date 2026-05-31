package trust

import (
	"math"
	"sync"
)

// EWMAEstimator implements an asymmetric exponentially weighted moving average
// (EWMA) trust estimation as described in §III-D and Appendix Hyperparams.
//
// For each agent R_k, the asymmetric EWMA applies:
//
//	if f_raw >= prev:  ewma_t^k = (1 - α)·ewma_{t-1}^k + α·f_raw        (rising)
//	otherwise:         ewma_t^k = (1 - α_down)·ewma_{t-1}^k + α_down·f_raw (falling)
//
// where α_down = min(2.5·α, 0.35) accelerates decay during dormant phases
// to suppress lingering false positives.
type EWMAEstimator struct {
	mu        sync.RWMutex
	alpha     float64            // base smoothing factor ∈ (0, 1) for rising signals
	alphaDown float64            // accelerated coefficient for falling signals
	state     map[uint64]float64 // nodeID → current EWMA value
}

// NewEWMAEstimator creates an asymmetric EWMA estimator with the given base
// smoothing factor. The falling coefficient is α_down = min(2.5·α, 0.35).
// Typical α ∈ [0.1, 0.3]: lower α = smoother, higher α = more reactive.
func NewEWMAEstimator(alpha float64) *EWMAEstimator {
	if alpha <= 0 || alpha > 1 {
		alpha = 0.10 // Paper §III-D default: α=0.10 for rising signals
	}
	alphaDown := math.Min(2.5*alpha, 0.35)
	return &EWMAEstimator{
		alpha:     alpha,
		alphaDown: alphaDown,
		state:     make(map[uint64]float64),
	}
}

// Update incorporates a new raw fault signal for the given agent using
// asymmetric smoothing: base α for rising signals, accelerated α_down for
// falling signals (Appendix Eq. asymmetric EWMA).
func (e *EWMAEstimator) Update(nodeID uint64, rawFault float64) float64 {
	e.mu.Lock()
	defer e.mu.Unlock()

	prev, exists := e.state[nodeID]
	if !exists {
		e.state[nodeID] = rawFault
		return rawFault
	}

	// Asymmetric: use alphaDown when signal is falling (attack departure)
	alpha := e.alpha
	if rawFault < prev {
		alpha = e.alphaDown
	}
	smoothed := (1-alpha)*prev + alpha*rawFault
	e.state[nodeID] = smoothed
	return smoothed
}

// Score returns the current EWMA fault estimate for the given agent.
func (e *EWMAEstimator) Score(nodeID uint64) (float64, bool) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	v, ok := e.state[nodeID]
	return v, ok
}

// Reset clears the EWMA state for the given agent.
func (e *EWMAEstimator) Reset(nodeID uint64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	delete(e.state, nodeID)
}

// AllScores returns a snapshot of all EWMA scores.
func (e *EWMAEstimator) AllScores() map[uint64]float64 {
	e.mu.RLock()
	defer e.mu.RUnlock()
	out := make(map[uint64]float64, len(e.state))
	for k, v := range e.state {
		out[k] = v
	}
	return out
}

// CombinedEstimator fuses BayesianEstimator (sliding-window sigmoid) with
// EWMAEstimator (exponential smoothing) to produce a robust, recency-aware
// fault probability. This implements the dual-mechanism trust estimation
// described in §III-D.
//
// Combined fault probability:
//
//	f_combined = γ * f_bayesian + (1 - γ) * f_ewma
//
// where γ ∈ [0, 1] controls the balance between the two estimators.
type CombinedEstimator struct {
	Bayesian *BayesianEstimator
	EWMA     *EWMAEstimator
	Gamma    float64 // fusion weight: γ * bayesian + (1-γ) * ewma
}

// NewCombinedEstimator creates a fused trust estimator with both mechanisms.
func NewCombinedEstimator(bayesianCfg BayesianConfig, ewmaAlpha float64, gamma float64) *CombinedEstimator {
	if gamma < 0 || gamma > 1 {
		gamma = 0.7 // default: 70% bayesian, 30% ewma
	}
	return &CombinedEstimator{
		Bayesian: NewBayesianEstimator(bayesianCfg),
		EWMA:     NewEWMAEstimator(ewmaAlpha),
		Gamma:    gamma,
	}
}

// ObserveEpoch feeds epoch data to both estimators.
func (ce *CombinedEstimator) ObserveEpoch(nodeID uint64, event EpochEvent) {
	ce.Bayesian.ObserveEpoch(nodeID, event)

	// Compute raw fault prob from bayesian and feed to EWMA
	if faultProb, ok := ce.Bayesian.FaultProbability(nodeID); ok {
		ce.EWMA.Update(nodeID, faultProb)
	}
}

// FaultProbability returns the fused fault probability.
func (ce *CombinedEstimator) FaultProbability(nodeID uint64) (float64, bool) {
	bayesProb, bayesOk := ce.Bayesian.FaultProbability(nodeID)
	ewmaProb, ewmaOk := ce.EWMA.Score(nodeID)

	if !bayesOk && !ewmaOk {
		return 0, false
	}
	if !bayesOk {
		return ewmaProb, true
	}
	if !ewmaOk {
		return bayesProb, true
	}

	combined := ce.Gamma*bayesProb + (1-ce.Gamma)*ewmaProb
	return math.Min(1.0, math.Max(0.0, combined)), true
}

// Features delegates to the BayesianEstimator.
func (ce *CombinedEstimator) Features(nodeID uint64) (FeatureVector, bool) {
	return ce.Bayesian.Features(nodeID)
}

// UpdateWeights updates the sigmoid classifier weights in the BayesianEstimator.
func (ce *CombinedEstimator) UpdateWeights(weights ClassifierWeights) {
	ce.Bayesian.UpdateWeights(weights)
}
