package trust

import "math"

// Observation is one minimal behavioral sample consumed by the in-memory scaffold.
// Success and Timeout are currently independent counters rather than a complete,
// mutually exclusive event taxonomy.
type Observation struct {
	Success bool
	Timeout bool
}

// Config defines the bounded sliding-window behavior of the scaffold.
type Config struct {
	WindowSize int
	MinSamples int
}

// Score summarizes the current scaffold-level estimate for one node.
// FailureProbability currently means timeout/failure ratio within the local window,
// not a validated probabilistic fault model from the paper.
type Score struct {
	NodeID             uint64
	SampleCount        int
	SuccessRate        float64
	FailureProbability float64
}

// --- Paper-grade trust model (§III-D, Eq. 5–6) ---

// EpochEvent captures one epoch of consensus metadata for an agent.
// These five counters map directly to the feature vector x_t^k in Eq. 5.
type EpochEvent struct {
	Timeouts      int     // d: timeout failures this epoch
	Equivocations int     // e: equivocation events this epoch
	ViewChanges   int     // v: view-change initiations this epoch
	LatencyMs     float64 // round-trip latency this epoch (ms)
}

// FeatureVector is the 5-dimensional trust feature (Eq. 5).
//
//	x = (d/W, e/W, v/W, τ̄, σ_τ)
type FeatureVector [5]float64

// FaultProb returns σ(w^T x + b) per Eq. 6.
func (x FeatureVector) FaultProb(w [5]float64, b float64) float64 {
	dot := b
	for i := 0; i < 5; i++ {
		dot += w[i] * x[i]
	}
	return 1.0 / (1.0 + math.Exp(-dot))
}

// ClassifierWeights holds the linear classifier parameters (w, b).
// Trained end-to-end with the MARL policy.
type ClassifierWeights struct {
	W [5]float64
	B float64
}

// BayesianConfig configures the paper-grade trust estimator.
type BayesianConfig struct {
	WindowSize   int               // W: sliding window length (epochs)
	MinSamples   int               // minimum epochs before producing a score
	Weights      ClassifierWeights // (w, b) for sigmoid classifier
	MaxLatencyMs float64           // normalization cap for latency features (default 1000ms)
}
