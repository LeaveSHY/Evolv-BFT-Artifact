package adaptive

import (
	"math"
	"sync"
)

// ═══════════════════════════════════════════════════════════════════════════════
// Policy Entropy Monitor — adaptive/entropy.go
//
// Paper Mapping:
//   - Definition 1 (P3, §IV): Self-evolving non-triviality condition (ii)
//     ρ_evol > 0 — "the protocol continues to adapt (non-degenerate)"
//   - Theorem regret-bound: requires that exploration does not collapse
//
// Mechanism:
//   Tracks the empirical action distribution entropy H(π) over a sliding
//   window. If entropy drops below a threshold, the policy has converged
//   prematurely (i.e., always picking the same action regardless of state).
//   This violates the self-evolving requirement.
//
//   H(π) = -Σ p(a) log₂ p(a)
//
//   High entropy: diverse actions (exploring / adapting)
//   Low entropy: deterministic (converged, possibly prematurely)
//   Zero entropy: degenerate (single action always selected)
//
//   The monitor uses the action bucket frequencies from a ring buffer of
//   recent decisions. It does NOT interfere with the policy — it only
//   observes and reports. A low-entropy alarm can trigger exploration reset.
// ═══════════════════════════════════════════════════════════════════════════════

// EntropyConfig controls the policy entropy monitor.
type EntropyConfig struct {
	// WindowSize: number of recent actions to compute entropy over.
	// Larger windows give more stable estimates. Default: 100.
	WindowSize int

	// LowEntropyThreshold: H(π) below this triggers a low-entropy alarm.
	// For K action buckets, max entropy = log₂(K). A reasonable threshold
	// is ~10% of theoretical max. Default: 0.5 bits.
	LowEntropyThreshold float64

	// HighEntropyThreshold: above this indicates healthy exploration.
	// Default: 2.0 bits (≈ uniform over 4 distinct actions).
	HighEntropyThreshold float64
}

// DefaultEntropyConfig returns production defaults.
func DefaultEntropyConfig() EntropyConfig {
	return EntropyConfig{
		WindowSize:           100,
		LowEntropyThreshold:  0.5,
		HighEntropyThreshold: 2.0,
	}
}

// EntropyMonitor tracks the action distribution entropy.
type EntropyMonitor struct {
	mu     sync.Mutex
	config EntropyConfig
	window []actionBucket
	pos    int
	full   bool
	// Bucket discretization (same as ExplorationBonus)
	committeeBucket int
	timeoutBucket   int
	batchBucket     int
	// Stats
	totalObservations int
	lowAlarmCount     int
}

// NewEntropyMonitor creates a policy entropy monitor.
func NewEntropyMonitor(config EntropyConfig) *EntropyMonitor {
	if config.WindowSize <= 0 {
		config.WindowSize = 100
	}
	if config.LowEntropyThreshold <= 0 {
		config.LowEntropyThreshold = 0.5
	}
	if config.HighEntropyThreshold <= 0 {
		config.HighEntropyThreshold = 2.0
	}
	return &EntropyMonitor{
		config:          config,
		window:          make([]actionBucket, config.WindowSize),
		committeeBucket: 5,
		timeoutBucket:   100,
		batchBucket:     512,
	}
}

// Observe records an action and returns the current entropy (bits).
func (em *EntropyMonitor) Observe(action Action) float64 {
	em.mu.Lock()
	defer em.mu.Unlock()

	bucket := em.bucketize(action)
	em.window[em.pos] = bucket
	em.pos = (em.pos + 1) % em.config.WindowSize
	if !em.full && em.pos == 0 {
		em.full = true
	}
	em.totalObservations++

	entropy := em.computeEntropy()
	if entropy < em.config.LowEntropyThreshold && em.effectiveSize() >= em.config.WindowSize/2 {
		em.lowAlarmCount++
	}
	return entropy
}

// Entropy returns the current action distribution entropy without observing.
func (em *EntropyMonitor) Entropy() float64 {
	em.mu.Lock()
	defer em.mu.Unlock()
	return em.computeEntropy()
}

// IsLowEntropy returns true if current entropy is below the alarm threshold.
func (em *EntropyMonitor) IsLowEntropy() bool {
	em.mu.Lock()
	defer em.mu.Unlock()
	return em.computeEntropy() < em.config.LowEntropyThreshold && em.effectiveSize() >= em.config.WindowSize/2
}

// EntropyStats provides monitoring diagnostics.
type EntropyStats struct {
	Entropy           float64 `json:"entropy_bits"`
	IsLow             bool    `json:"is_low_entropy"`
	WindowFill        int     `json:"window_fill"`
	WindowSize        int     `json:"window_size"`
	UniqueBuckets     int     `json:"unique_buckets"`
	TotalObservations int     `json:"total_observations"`
	LowAlarmCount     int     `json:"low_alarm_count"`
}

// Stats returns current entropy monitoring diagnostics.
func (em *EntropyMonitor) Stats() EntropyStats {
	em.mu.Lock()
	defer em.mu.Unlock()
	entropy := em.computeEntropy()
	size := em.effectiveSize()
	counts := em.bucketCounts()
	return EntropyStats{
		Entropy:           entropy,
		IsLow:             entropy < em.config.LowEntropyThreshold && size >= em.config.WindowSize/2,
		WindowFill:        size,
		WindowSize:        em.config.WindowSize,
		UniqueBuckets:     len(counts),
		TotalObservations: em.totalObservations,
		LowAlarmCount:     em.lowAlarmCount,
	}
}

// bucketize maps action to discrete bucket (same scheme as ExplorationBonus).
func (em *EntropyMonitor) bucketize(action Action) actionBucket {
	return actionBucket{
		CommitteeBucket: action.CommitteeSize / em.committeeBucket,
		TimeoutBucket:   action.PacemakerTimeoutMs / em.timeoutBucket,
		BatchBucket:     action.MempoolMaxBatchTxs / em.batchBucket,
	}
}

// effectiveSize returns how many entries are valid in the ring buffer.
func (em *EntropyMonitor) effectiveSize() int {
	if em.full {
		return em.config.WindowSize
	}
	return em.pos
}

// bucketCounts returns frequency map for current window. Caller must hold mu.
func (em *EntropyMonitor) bucketCounts() map[actionBucket]int {
	size := em.effectiveSize()
	counts := make(map[actionBucket]int)
	for i := 0; i < size; i++ {
		counts[em.window[i]]++
	}
	return counts
}

// computeEntropy calculates Shannon entropy in bits. Caller must hold mu.
func (em *EntropyMonitor) computeEntropy() float64 {
	size := em.effectiveSize()
	if size <= 1 {
		return 0
	}
	counts := em.bucketCounts()
	n := float64(size)
	var entropy float64
	for _, count := range counts {
		if count == 0 {
			continue
		}
		p := float64(count) / n
		entropy -= p * math.Log2(p)
	}
	return entropy
}
