package adaptive

import (
	"math"
	"sync"
)

// ═══════════════════════════════════════════════════════════════════════════════
// Value Baseline & Advantage Estimation — adaptive/advantage.go
//
// Paper Mapping:
//   - SFAC (Safe Factored Actor-Critic, §III-C): the "Critic" component
//   - Eq. policy-gradient: ∇J ∝ Σ A(s,a) · ∇log π(a|s)
//   - Advantage A(s,a) = R(s,a) - V(s) reduces gradient variance
//
// Mechanism:
//   The Go-side maintains a lightweight EMA baseline V̂(s) as a running
//   estimate of expected reward. When computing the feedback reward for
//   the Python SFAC policy, we subtract the baseline:
//     feedback_advantage = reward + exploration_bonus - baseline
//
//   This has two key effects:
//   (1) Reduces variance in policy gradient updates (faster convergence)
//   (2) Differentiates truly good actions from average performance
//
// The EMA baseline adapts to non-stationary reward distributions (which
// arise naturally as the adversary evolves and the system reconfigures).
// ═══════════════════════════════════════════════════════════════════════════════

// AdvantageConfig controls the value baseline estimator.
type AdvantageConfig struct {
	// Alpha is the EMA smoothing factor for the baseline.
	// Larger α → faster adaptation to reward changes.
	// Typical: 0.01–0.1. Default: 0.05.
	Alpha float64

	// WarmupSteps: number of steps before the baseline is used.
	// During warmup, advantage = raw reward (no baseline subtraction).
	WarmupSteps int

	// NormalizeAdvantage: if true, divide advantage by running std dev.
	// Prevents gradient explosion when reward magnitudes change.
	NormalizeAdvantage bool

	// StdEpsilon: small constant added to std dev to prevent division by zero.
	StdEpsilon float64
}

// DefaultAdvantageConfig returns production defaults.
func DefaultAdvantageConfig() AdvantageConfig {
	return AdvantageConfig{
		Alpha:              0.05,
		WarmupSteps:        20,
		NormalizeAdvantage: true,
		StdEpsilon:         1e-8,
	}
}

// ValueBaseline maintains an EMA estimate of expected reward (the "Critic").
type ValueBaseline struct {
	mu       sync.Mutex
	config   AdvantageConfig
	baseline float64 // EMA of reward (V̂)
	variance float64 // EMA of (reward - baseline)²
	steps    int
	warmedUp bool
	totalAdv float64 // sum of advantages (for diagnostics)
	maxAdv   float64
	minAdv   float64
}

// NewValueBaseline creates an advantage estimator.
func NewValueBaseline(config AdvantageConfig) *ValueBaseline {
	if config.Alpha <= 0 || config.Alpha > 1 {
		config.Alpha = 0.05
	}
	if config.WarmupSteps < 0 {
		config.WarmupSteps = 0
	}
	if config.StdEpsilon <= 0 {
		config.StdEpsilon = 1e-8
	}
	return &ValueBaseline{
		config: config,
		minAdv: math.Inf(1),
		maxAdv: math.Inf(-1),
	}
}

// Advantage computes A = reward - V̂ and updates the baseline.
// Returns the advantage (optionally normalized) for the feedback signal.
func (vb *ValueBaseline) Advantage(reward float64) float64 {
	vb.mu.Lock()
	defer vb.mu.Unlock()

	vb.steps++

	// During warmup, just accumulate baseline without subtracting
	if vb.steps <= vb.config.WarmupSteps {
		if vb.steps == 1 {
			vb.baseline = reward
		} else {
			vb.baseline += (reward - vb.baseline) / float64(vb.steps)
		}
		vb.warmedUp = vb.steps >= vb.config.WarmupSteps
		return reward // raw reward during warmup
	}

	// Compute advantage: A = R - V̂
	advantage := reward - vb.baseline

	// Update baseline EMA
	vb.baseline += vb.config.Alpha * (reward - vb.baseline)

	// Update variance EMA for normalization
	vb.variance += vb.config.Alpha * (advantage*advantage - vb.variance)

	// Track diagnostics
	vb.totalAdv += advantage
	if advantage > vb.maxAdv {
		vb.maxAdv = advantage
	}
	if advantage < vb.minAdv {
		vb.minAdv = advantage
	}

	// Optionally normalize by std dev
	if vb.config.NormalizeAdvantage {
		std := math.Sqrt(vb.variance) + vb.config.StdEpsilon
		advantage /= std
	}

	return advantage
}

// Baseline returns the current V̂ estimate.
func (vb *ValueBaseline) Baseline() float64 {
	vb.mu.Lock()
	defer vb.mu.Unlock()
	return vb.baseline
}

// Steps returns total observations processed.
func (vb *ValueBaseline) Steps() int {
	vb.mu.Lock()
	defer vb.mu.Unlock()
	return vb.steps
}

// IsWarmedUp returns whether the baseline has enough data.
func (vb *ValueBaseline) IsWarmedUp() bool {
	vb.mu.Lock()
	defer vb.mu.Unlock()
	return vb.warmedUp
}

// AdvantageStats provides diagnostics for monitoring.
type AdvantageStats struct {
	Baseline float64 `json:"baseline"`
	Variance float64 `json:"variance"`
	StdDev   float64 `json:"std_dev"`
	Steps    int     `json:"steps"`
	WarmedUp bool    `json:"warmed_up"`
	MeanAdv  float64 `json:"mean_advantage"`
	MaxAdv   float64 `json:"max_advantage"`
	MinAdv   float64 `json:"min_advantage"`
}

// Stats returns current advantage estimation diagnostics.
func (vb *ValueBaseline) Stats() AdvantageStats {
	vb.mu.Lock()
	defer vb.mu.Unlock()
	std := math.Sqrt(vb.variance)
	var meanAdv float64
	effectiveSteps := vb.steps - vb.config.WarmupSteps
	if effectiveSteps > 0 {
		meanAdv = vb.totalAdv / float64(effectiveSteps)
	}
	return AdvantageStats{
		Baseline: vb.baseline,
		Variance: vb.variance,
		StdDev:   std,
		Steps:    vb.steps,
		WarmedUp: vb.warmedUp,
		MeanAdv:  meanAdv,
		MaxAdv:   vb.maxAdv,
		MinAdv:   vb.minAdv,
	}
}
