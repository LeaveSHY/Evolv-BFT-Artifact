package trust

import (
	"math"
	"sync"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════════
// Trust Decay (Temporal Forgetting) — trust/decay.go
//
// Paper Mapping:
//   - §III-D Adaptive Trust Management: "adapts to evolving adversaries"
//   - Threat Model: Byzantine nodes may accumulate trust during dormant phases
//     then exploit it during attack windows (wolf-in-sheep strategy).
//
// Mechanism:
//   When a node has no fresh observations for duration t, its effective fault
//   probability drifts toward an "uncertain" baseline (default 0.5) at rate:
//
//     f_eff(node) = baseline + (f_raw - baseline) · exp(-λ·t)
//
//   where λ = ln(2) / halfLife controls the decay speed.
//
// Properties:
//   - Dormant nodes gradually lose accumulated trust (conservative)
//   - Active nodes with fresh data remain at their true estimate
//   - Prevents wolf-in-sheep attacks where Byzantine nodes go silent before strike
//   - Decay is monotonic toward uncertainty (never increases trust without evidence)
// ═══════════════════════════════════════════════════════════════════════════════

// DecayConfig controls the temporal trust decay behavior.
type DecayConfig struct {
	// HalfLife is the time after which trust decays halfway toward baseline.
	// Shorter = more aggressive forgetting. Typical: 30s–120s for vehicle platoons.
	HalfLife time.Duration

	// Baseline is the "maximally uncertain" fault probability target.
	// Trust decays toward this value. Default: 0.5 (maximum entropy).
	Baseline float64

	// MinFreshness is the grace period during which no decay applies.
	// Prevents micro-decay from jittery observation timing.
	MinFreshness time.Duration
}

// DefaultDecayConfig returns conservative defaults suitable for V2X platoons.
func DefaultDecayConfig() DecayConfig {
	return DecayConfig{
		HalfLife:     60 * time.Second,
		Baseline:     0.5,
		MinFreshness: 5 * time.Second,
	}
}

// TrustDecay tracks observation timestamps and applies temporal decay.
type TrustDecay struct {
	mu       sync.RWMutex
	config   DecayConfig
	lambda   float64 // ln(2) / halfLife (precomputed)
	lastSeen map[uint64]time.Time
}

// NewTrustDecay creates a temporal decay tracker.
func NewTrustDecay(config DecayConfig) *TrustDecay {
	if config.HalfLife <= 0 {
		config.HalfLife = 60 * time.Second
	}
	if config.Baseline < 0 || config.Baseline > 1 {
		config.Baseline = 0.5
	}
	if config.MinFreshness < 0 {
		config.MinFreshness = 0
	}
	lambda := math.Ln2 / config.HalfLife.Seconds()
	return &TrustDecay{
		config:   config,
		lambda:   lambda,
		lastSeen: make(map[uint64]time.Time),
	}
}

// Touch records that we received a fresh observation for the node.
func (td *TrustDecay) Touch(nodeID uint64, at time.Time) {
	td.mu.Lock()
	defer td.mu.Unlock()
	td.lastSeen[nodeID] = at
}

// Decay applies temporal forgetting to a raw fault probability.
// If the node has fresh observations (within MinFreshness), rawProb is returned unchanged.
// Otherwise, rawProb decays toward Baseline exponentially.
func (td *TrustDecay) Decay(nodeID uint64, rawProb float64, now time.Time) float64 {
	td.mu.RLock()
	lastSeen, exists := td.lastSeen[nodeID]
	td.mu.RUnlock()

	if !exists {
		// Never observed: return baseline (maximally uncertain)
		return td.config.Baseline
	}

	elapsed := now.Sub(lastSeen)
	if elapsed <= td.config.MinFreshness {
		// Within grace period: no decay
		return rawProb
	}

	// Exponential decay toward baseline:
	//   f_eff = baseline + (rawProb - baseline) * exp(-λ * elapsed)
	decayFactor := math.Exp(-td.lambda * elapsed.Seconds())
	return td.config.Baseline + (rawProb-td.config.Baseline)*decayFactor
}

// DecayAll applies decay to a map of fault probabilities (e.g., from Aggregator).
func (td *TrustDecay) DecayAll(probs map[uint64]float64, now time.Time) map[uint64]float64 {
	result := make(map[uint64]float64, len(probs))
	for nodeID, prob := range probs {
		result[nodeID] = td.Decay(nodeID, prob, now)
	}
	return result
}

// Staleness returns how long since the last observation for a node.
// Returns -1 if the node was never observed.
func (td *TrustDecay) Staleness(nodeID uint64, now time.Time) time.Duration {
	td.mu.RLock()
	defer td.mu.RUnlock()
	lastSeen, exists := td.lastSeen[nodeID]
	if !exists {
		return -1
	}
	return now.Sub(lastSeen)
}

// Remove stops tracking a node (e.g., after it leaves the system).
func (td *TrustDecay) Remove(nodeID uint64) {
	td.mu.Lock()
	defer td.mu.Unlock()
	delete(td.lastSeen, nodeID)
}

// TrackedNodes returns the number of nodes being tracked.
func (td *TrustDecay) TrackedNodes() int {
	td.mu.RLock()
	defer td.mu.RUnlock()
	return len(td.lastSeen)
}
