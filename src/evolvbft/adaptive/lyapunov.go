package adaptive

import (
	"math"
	"sync"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════════
// Lyapunov Stability Monitor — adaptive/lyapunov.go
//
// Paper Mapping:
//   - §IV Safety Invariant: "safety is preserved under all adaptations"
//   - Theorem safety-preservation: V(s_{t+1}) ≤ V(s_t) after safety-filtered actions
//
// The Lyapunov function V(state) provides a runtime certificate that the
// system is not degrading after each adaptation step. When V is non-increasing,
// we have continuous assurance that:
//   (1) BFT quorum margins are maintained (safety term)
//   (2) Policy performance is converging (regret term)
//
// V(s) = w_safety · Σᵢ max(0, 3fᵢ+1+δ - nᵢ) + w_regret · R(T)/√T
//
// where:
//   - fᵢ = estimated Byzantine count in instance i
//   - nᵢ = validator count in instance i
//   - δ = safety margin (DeltaS from SafetyFilter)
//   - R(T)/√T = normalized cumulative regret from RegretTracker
//
// A violation (V increases) indicates the safety filter may need tightening
// or the policy is exploring a suboptimal region. Violations are logged but
// do not halt the system (the safety filter independently guarantees safety).
// ═══════════════════════════════════════════════════════════════════════════════

// LyapunovConfig controls the stability monitor weights.
type LyapunovConfig struct {
	// WeightSafety scales the quorum margin violation term.
	WeightSafety float64

	// WeightRegret scales the normalized regret term.
	WeightRegret float64

	// DeltaS is the safety margin parameter (from SafetyFilter).
	DeltaS int

	// ViolationThreshold: V increase above this triggers a violation event.
	// Small increases from exploration noise are tolerated.
	ViolationThreshold float64
}

// DefaultLyapunovConfig returns production defaults.
func DefaultLyapunovConfig() LyapunovConfig {
	return LyapunovConfig{
		WeightSafety:       1.0,
		WeightRegret:       0.1,
		DeltaS:             1,
		ViolationThreshold: 0.01,
	}
}

// LyapunovState captures the inputs for V(state) computation.
type LyapunovState struct {
	// InstanceFaults[i] = estimated Byzantine count in instance i
	InstanceFaults []int
	// InstanceSizes[i] = validator count in instance i
	InstanceSizes []int
	// NormalizedRegret = R(T)/√T from RegretTracker (0 if no tracker)
	NormalizedRegret float64
}

// LyapunovSnapshot captures one V(state) observation.
type LyapunovSnapshot struct {
	Timestamp       time.Time `json:"timestamp"`
	Value           float64   `json:"value"`
	SafetyTerm      float64   `json:"safety_term"`
	RegretTerm      float64   `json:"regret_term"`
	IsViolation     bool      `json:"is_violation"`
	Delta           float64   `json:"delta"` // V(t) - V(t-1)
	MonotonicStreak int       `json:"monotonic_streak"`
}

// LyapunovMonitor tracks the Lyapunov function over time.
type LyapunovMonitor struct {
	mu              sync.Mutex
	config          LyapunovConfig
	prevValue       float64
	initialized     bool
	violations      int
	monotonicStreak int // consecutive non-increasing steps
	totalSteps      int
	history         []LyapunovSnapshot
	maxHistory      int
}

// NewLyapunovMonitor creates a stability monitor.
func NewLyapunovMonitor(config LyapunovConfig) *LyapunovMonitor {
	if config.WeightSafety <= 0 {
		config.WeightSafety = 1.0
	}
	if config.WeightRegret < 0 {
		config.WeightRegret = 0.1
	}
	if config.ViolationThreshold <= 0 {
		config.ViolationThreshold = 0.01
	}
	return &LyapunovMonitor{
		config:     config,
		maxHistory: 1000, // keep last 1000 snapshots
	}
}

// Evaluate computes V(state) and checks the monotonicity invariant.
// Returns the snapshot with violation info.
func (lm *LyapunovMonitor) Evaluate(state LyapunovState) LyapunovSnapshot {
	lm.mu.Lock()
	defer lm.mu.Unlock()

	// Compute safety term: Σᵢ max(0, 3fᵢ + 1 + δ - nᵢ)
	var safetyTerm float64
	for i := range state.InstanceFaults {
		if i >= len(state.InstanceSizes) {
			break
		}
		margin := 3*state.InstanceFaults[i] + 1 + lm.config.DeltaS - state.InstanceSizes[i]
		if margin > 0 {
			safetyTerm += float64(margin)
		}
	}

	// Compute regret term: R(T)/√T (already normalized by caller)
	regretTerm := math.Max(0, state.NormalizedRegret)

	// V(s) = w_safety * safetyTerm + w_regret * regretTerm
	value := lm.config.WeightSafety*safetyTerm + lm.config.WeightRegret*regretTerm

	snap := LyapunovSnapshot{
		Timestamp:  time.Now(),
		Value:      value,
		SafetyTerm: safetyTerm,
		RegretTerm: regretTerm,
	}

	if lm.initialized {
		snap.Delta = value - lm.prevValue
		if snap.Delta > lm.config.ViolationThreshold {
			snap.IsViolation = true
			lm.violations++
			lm.monotonicStreak = 0
		} else {
			lm.monotonicStreak++
		}
	} else {
		lm.initialized = true
		lm.monotonicStreak = 1
	}
	snap.MonotonicStreak = lm.monotonicStreak

	lm.prevValue = value
	lm.totalSteps++

	// Append to history (ring buffer)
	if len(lm.history) >= lm.maxHistory {
		lm.history = append(lm.history[1:], snap)
	} else {
		lm.history = append(lm.history, snap)
	}

	return snap
}

// CurrentValue returns the last computed V(state).
func (lm *LyapunovMonitor) CurrentValue() float64 {
	lm.mu.Lock()
	defer lm.mu.Unlock()
	return lm.prevValue
}

// Violations returns the total number of monotonicity violations.
func (lm *LyapunovMonitor) Violations() int {
	lm.mu.Lock()
	defer lm.mu.Unlock()
	return lm.violations
}

// MonotonicStreak returns consecutive non-increasing steps.
func (lm *LyapunovMonitor) MonotonicStreak() int {
	lm.mu.Lock()
	defer lm.mu.Unlock()
	return lm.monotonicStreak
}

// TotalSteps returns the total number of evaluations.
func (lm *LyapunovMonitor) TotalSteps() int {
	lm.mu.Lock()
	defer lm.mu.Unlock()
	return lm.totalSteps
}

// ViolationRate returns the fraction of steps that were violations.
func (lm *LyapunovMonitor) ViolationRate() float64 {
	lm.mu.Lock()
	defer lm.mu.Unlock()
	if lm.totalSteps == 0 {
		return 0
	}
	return float64(lm.violations) / float64(lm.totalSteps)
}

// LyapunovStats provides a summary for monitoring/admin endpoints.
type LyapunovStats struct {
	CurrentValue    float64 `json:"current_value"`
	TotalSteps      int     `json:"total_steps"`
	Violations      int     `json:"violations"`
	ViolationRate   float64 `json:"violation_rate"`
	MonotonicStreak int     `json:"monotonic_streak"`
	IsStable        bool    `json:"is_stable"` // true if violation rate < 5%
}

// Stats returns current stability statistics.
func (lm *LyapunovMonitor) Stats() LyapunovStats {
	lm.mu.Lock()
	defer lm.mu.Unlock()
	vRate := 0.0
	if lm.totalSteps > 0 {
		vRate = float64(lm.violations) / float64(lm.totalSteps)
	}
	return LyapunovStats{
		CurrentValue:    lm.prevValue,
		TotalSteps:      lm.totalSteps,
		Violations:      lm.violations,
		ViolationRate:   vRate,
		MonotonicStreak: lm.monotonicStreak,
		IsStable:        vRate < 0.05, // <5% violation rate = stable
	}
}
