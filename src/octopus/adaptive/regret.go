package adaptive

import (
	"math"
	"sync"
)

// ═══════════════════════════════════════════════════════════════════════════════
// Cumulative Regret Tracker (§IV, Theorem regret-bound)
//
// Tracks R(T) = Σ_{t=1}^{T} [r*(s_t) - r_π(s_t)] where:
//   - r*(s_t) is the best achievable reward at state s_t (hindsight optimum)
//   - r_π(s_t) is the reward actually received by the learned policy
//
// The paper proves R(T) ∈ O(√T) under Assumption 1 (bounded rewards) and
// Assumption 2 (mixing time). This tracker provides empirical verification.
//
// Usage:
//   tracker := NewRegretTracker(RegretConfig{RewardUpperBound: 10.0})
//   tracker.Observe(actualReward)
//   fmt.Printf("Regret at T=%d: %.2f (normalized: %.4f)\n",
//       tracker.T(), tracker.CumulativeRegret(), tracker.NormalizedRegret())
// ═══════════════════════════════════════════════════════════════════════════════

// RegretConfig configures the regret tracker.
type RegretConfig struct {
	// RewardUpperBound is the maximum achievable reward per step (r_max).
	// Used as the hindsight optimum when no oracle is available.
	// Default: estimated from observed maximum.
	RewardUpperBound float64

	// WindowSize for sliding-window regret analysis (0 = track all).
	WindowSize int
}

// RegretTracker computes cumulative and normalized regret over time.
type RegretTracker struct {
	mu             sync.Mutex
	config         RegretConfig
	totalReward    float64
	maxObserved    float64
	steps          int
	cumulRegret    float64
	windowRewards  []float64
	windowIdx      int
	windowFull     bool
	hasUpperBound  bool
	sqrtTHistory   []float64 // optional: record regret/√T for convergence plots
}

// NewRegretTracker creates a regret tracker. If config.RewardUpperBound > 0,
// it uses that as the hindsight optimum. Otherwise, it uses the running
// maximum observed reward (conservative lower bound on regret).
func NewRegretTracker(config RegretConfig) *RegretTracker {
	rt := &RegretTracker{
		config:        config,
		maxObserved:   math.Inf(-1),
		hasUpperBound: config.RewardUpperBound > 0,
	}
	if config.WindowSize > 0 {
		rt.windowRewards = make([]float64, config.WindowSize)
	}
	return rt
}

// Observe records a reward observation and updates regret.
func (rt *RegretTracker) Observe(reward float64) {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	rt.steps++
	rt.totalReward += reward

	if reward > rt.maxObserved {
		rt.maxObserved = reward
	}

	// Compute instantaneous regret: r* - r_t
	rStar := rt.upperBound()
	instantRegret := rStar - reward
	if instantRegret < 0 {
		instantRegret = 0 // policy can exceed conservative upper bound
	}
	rt.cumulRegret += instantRegret

	// Window tracking
	if rt.windowRewards != nil {
		rt.windowRewards[rt.windowIdx] = reward
		rt.windowIdx = (rt.windowIdx + 1) % rt.config.WindowSize
		if rt.windowIdx == 0 {
			rt.windowFull = true
		}
	}
}

// CumulativeRegret returns R(T) = Σ (r* - r_t).
func (rt *RegretTracker) CumulativeRegret() float64 {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	return rt.cumulRegret
}

// NormalizedRegret returns R(T)/T (average per-step regret).
// Approaches 0 for no-regret algorithms.
func (rt *RegretTracker) NormalizedRegret() float64 {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rt.steps == 0 {
		return 0
	}
	return rt.cumulRegret / float64(rt.steps)
}

// RegretPerSqrtT returns R(T)/√T. Should remain bounded for O(√T) regret.
func (rt *RegretTracker) RegretPerSqrtT() float64 {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rt.steps == 0 {
		return 0
	}
	return rt.cumulRegret / math.Sqrt(float64(rt.steps))
}

// T returns the number of observed steps.
func (rt *RegretTracker) T() int {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	return rt.steps
}

// AverageReward returns the mean reward observed so far.
func (rt *RegretTracker) AverageReward() float64 {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rt.steps == 0 {
		return 0
	}
	return rt.totalReward / float64(rt.steps)
}

// WindowAverageReward returns the average reward over the last WindowSize steps.
func (rt *RegretTracker) WindowAverageReward() float64 {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rt.windowRewards == nil {
		return rt.totalReward / math.Max(float64(rt.steps), 1)
	}
	count := rt.config.WindowSize
	if !rt.windowFull {
		count = rt.windowIdx
	}
	if count == 0 {
		return 0
	}
	sum := 0.0
	for i := 0; i < count; i++ {
		sum += rt.windowRewards[i]
	}
	return sum / float64(count)
}

// Snapshot returns a point-in-time regret summary for logging/plotting.
func (rt *RegretTracker) Snapshot() RegretSnapshot {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	return RegretSnapshot{
		T:                rt.steps,
		CumulativeRegret: rt.cumulRegret,
		NormalizedRegret: rt.normalizedLocked(),
		RegretPerSqrtT:   rt.regretPerSqrtTLocked(),
		AverageReward:    rt.avgRewardLocked(),
		MaxObservedReward: rt.maxObserved,
	}
}

// RegretSnapshot is a serializable regret summary.
type RegretSnapshot struct {
	T                 int     `json:"t"`
	CumulativeRegret  float64 `json:"cumulative_regret"`
	NormalizedRegret  float64 `json:"normalized_regret"`
	RegretPerSqrtT    float64 `json:"regret_per_sqrt_t"`
	AverageReward     float64 `json:"average_reward"`
	MaxObservedReward float64 `json:"max_observed_reward"`
}

func (rt *RegretTracker) upperBound() float64 {
	if rt.hasUpperBound {
		return rt.config.RewardUpperBound
	}
	return rt.maxObserved
}

func (rt *RegretTracker) normalizedLocked() float64 {
	if rt.steps == 0 {
		return 0
	}
	return rt.cumulRegret / float64(rt.steps)
}

func (rt *RegretTracker) regretPerSqrtTLocked() float64 {
	if rt.steps == 0 {
		return 0
	}
	return rt.cumulRegret / math.Sqrt(float64(rt.steps))
}

func (rt *RegretTracker) avgRewardLocked() float64 {
	if rt.steps == 0 {
		return 0
	}
	return rt.totalReward / float64(rt.steps)
}
