package adaptive

import (
	"math"
	"sync"
)

// ═══════════════════════════════════════════════════════════════════════════════
// Exploration Bonus (UCB-style) — adaptive/exploration.go
//
// Paper Mapping:
//   - Definition 1 (P3, §IV): Self-evolving non-triviality condition (i)
//     "evidence-sensitivity" — the protocol must respond to uncertainty by
//     actively exploring less-visited parameter configurations.
//   - Theorem regret-bound: UCB exploration yields O(√T) cumulative regret
//     when combined with SFAC policy updates.
//
// Mechanism:
//   Actions are discretized into buckets (CommitteeSize, Timeout, BatchSize).
//   The exploration bonus β/√N(bucket) is added to the reward fed back to
//   the Python SFAC policy, encouraging optimistic value estimation for
//   under-explored regions of the action space.
//
// The bonus decays as 1/√N, ensuring convergence: as the policy converges
// to the optimal action, the bonus vanishes and reward reflects true utility.
// ═══════════════════════════════════════════════════════════════════════════════

// ExplorationConfig controls the UCB exploration bonus.
type ExplorationConfig struct {
	// Beta is the initial exploration coefficient (confidence width).
	// Larger values encourage more exploration. Typical: 0.5–2.0.
	Beta float64

	// BetaDecay controls how Beta decreases over time.
	// "none" = constant Beta (default)
	// "sqrt" = Beta(t) = Beta / √t  (standard UCB convergence)
	// "log"  = Beta(t) = Beta / ln(t+1) (slower decay)
	BetaDecay string

	// CommitteeBucketSize discretizes CommitteeSize into buckets.
	// E.g., bucket=5 means {5,10,15,...} are distinct buckets.
	CommitteeBucketSize int

	// TimeoutBucketMs discretizes PacemakerTimeoutMs.
	TimeoutBucketMs int

	// BatchBucketSize discretizes MempoolMaxBatchTxs.
	BatchBucketSize int

	// MinBonus is the floor for the exploration bonus (prevents zero).
	MinBonus float64

	// MaxBonus caps the bonus to prevent reward distortion.
	MaxBonus float64
}

// DefaultExplorationConfig returns sensible defaults.
func DefaultExplorationConfig() ExplorationConfig {
	return ExplorationConfig{
		Beta:                1.0,
		CommitteeBucketSize: 5,
		TimeoutBucketMs:     100,
		BatchBucketSize:     512,
		MinBonus:            0.0,
		MaxBonus:            1.0,
	}
}

// actionBucket is a discretized representation of the continuous action space.
type actionBucket struct {
	CommitteeBucket int
	TimeoutBucket   int
	BatchBucket     int
}

// ExplorationBonus tracks action visitation and computes UCB-style bonuses.
type ExplorationBonus struct {
	mu      sync.Mutex
	config  ExplorationConfig
	visits  map[actionBucket]int
	totalN  int
}

// NewExplorationBonus creates an exploration bonus tracker.
func NewExplorationBonus(config ExplorationConfig) *ExplorationBonus {
	if config.CommitteeBucketSize <= 0 {
		config.CommitteeBucketSize = 5
	}
	if config.TimeoutBucketMs <= 0 {
		config.TimeoutBucketMs = 100
	}
	if config.BatchBucketSize <= 0 {
		config.BatchBucketSize = 512
	}
	if config.Beta <= 0 {
		config.Beta = 1.0
	}
	if config.MaxBonus <= 0 {
		config.MaxBonus = 1.0
	}
	return &ExplorationBonus{
		config: config,
		visits: make(map[actionBucket]int),
	}
}

// bucketize maps a continuous action to its discrete bucket.
func (eb *ExplorationBonus) bucketize(action Action) actionBucket {
	return actionBucket{
		CommitteeBucket: action.CommitteeSize / eb.config.CommitteeBucketSize,
		TimeoutBucket:   action.PacemakerTimeoutMs / eb.config.TimeoutBucketMs,
		BatchBucket:     action.MempoolMaxBatchTxs / eb.config.BatchBucketSize,
	}
}

// Bonus computes the exploration bonus for the given action WITHOUT recording.
// Use this when you want to query the bonus without side effects.
func (eb *ExplorationBonus) Bonus(action Action) float64 {
	eb.mu.Lock()
	defer eb.mu.Unlock()
	bucket := eb.bucketize(action)
	n := eb.visits[bucket]
	return eb.computeBonus(n)
}

// ObserveAndBonus records the action visit and returns the exploration bonus.
// This should be called once per tick after the action is applied.
func (eb *ExplorationBonus) ObserveAndBonus(action Action) float64 {
	eb.mu.Lock()
	defer eb.mu.Unlock()
	bucket := eb.bucketize(action)
	eb.visits[bucket]++
	eb.totalN++
	n := eb.visits[bucket]
	return eb.computeBonus(n)
}

// computeBonus implements UCB: β(t) / √N(a). Caller must hold mu.
func (eb *ExplorationBonus) computeBonus(n int) float64 {
	if n == 0 {
		// Unvisited action: return max bonus (optimistic initialization).
		return eb.config.MaxBonus
	}
	beta := eb.effectiveBeta()
	bonus := beta / math.Sqrt(float64(n))
	if bonus < eb.config.MinBonus {
		bonus = eb.config.MinBonus
	}
	if bonus > eb.config.MaxBonus {
		bonus = eb.config.MaxBonus
	}
	return bonus
}

// effectiveBeta computes β(t) based on the decay schedule. Caller must hold mu.
func (eb *ExplorationBonus) effectiveBeta() float64 {
	t := eb.totalN
	if t <= 1 {
		return eb.config.Beta
	}
	switch eb.config.BetaDecay {
	case "sqrt":
		// β(t) = β₀ / √t — standard UCB convergence rate
		return eb.config.Beta / math.Sqrt(float64(t))
	case "log":
		// β(t) = β₀ / ln(t+1) — slower decay, more exploration
		return eb.config.Beta / math.Log(float64(t+1))
	default:
		// "none" or empty: constant beta
		return eb.config.Beta
	}
}

// TotalVisits returns the total number of actions observed.
func (eb *ExplorationBonus) TotalVisits() int {
	eb.mu.Lock()
	defer eb.mu.Unlock()
	return eb.totalN
}

// BucketVisits returns the visit count for the action bucket.
func (eb *ExplorationBonus) BucketVisits(action Action) int {
	eb.mu.Lock()
	defer eb.mu.Unlock()
	bucket := eb.bucketize(action)
	return eb.visits[bucket]
}

// UniqueBuckets returns the number of distinct action buckets visited.
func (eb *ExplorationBonus) UniqueBuckets() int {
	eb.mu.Lock()
	defer eb.mu.Unlock()
	return len(eb.visits)
}

// ExplorationStats provides a snapshot of exploration state.
type ExplorationStats struct {
	TotalVisits   int     `json:"total_visits"`
	UniqueBuckets int     `json:"unique_buckets"`
	Beta          float64 `json:"beta"`          // initial β₀
	EffectiveBeta float64 `json:"effective_beta"` // β(t) after decay
	BetaDecay     string  `json:"beta_decay"`
	AvgBonus      float64 `json:"avg_bonus"`
}

// Stats returns current exploration statistics.
func (eb *ExplorationBonus) Stats() ExplorationStats {
	eb.mu.Lock()
	defer eb.mu.Unlock()
	unique := len(eb.visits)
	var avgBonus float64
	if unique > 0 {
		var sum float64
		for _, n := range eb.visits {
			sum += eb.computeBonus(n)
		}
		avgBonus = sum / float64(unique)
	}
	return ExplorationStats{
		TotalVisits:   eb.totalN,
		UniqueBuckets: unique,
		Beta:          eb.config.Beta,
		EffectiveBeta: eb.effectiveBeta(),
		BetaDecay:     eb.config.BetaDecay,
		AvgBonus:      avgBonus,
	}
}
