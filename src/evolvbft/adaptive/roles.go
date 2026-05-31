package adaptive

import "math"

// --- Byzantine-Aware Role Decomposition (§III-D P5, Appendix OrgSpecs) ---

// RoleName identifies one of the four per-agent MOISE+ organizational roles.
// These define RESPONSIBILITIES (what each sub-policy does).
type RoleName string

const (
	RoleSentinel  RoleName = "sentinel"  // ↦ rotate (detect Byzantine leaders)
	RoleCommander RoleName = "commander" // ↦ reconfig (manage membership)
	RoleTuner     RoleName = "tuner"     // ↦ param (adjust consensus parameters)
	RoleGuardian  RoleName = "guardian"  // safety filter (BFT invariant, ch=1.0)
)

// AllRoles returns the four roles assigned to every primary agent.
func AllRoles() []RoleName {
	return []RoleName{RoleSentinel, RoleCommander, RoleTuner, RoleGuardian}
}

// --- Cross-Language Role Mapping ---
//
// Two role systems coexist:
//   1. MOISE+ organizational roles (this file): sentinel/commander/tuner/guardian
//      → Define agent RESPONSIBILITIES and produce rrg/grg penalties/bonuses
//   2. MARL reward heads (reward.go): lane_tuner/recovery_tuner/membership_tuner/safety_guardian
//      → Define reward CREDIT ASSIGNMENT for multi-objective training
//
// Semantic mapping (MOISE+ role → primary reward head it influences):
//   sentinel   → recovery_tuner   (detection quality reduces recovery penalties)
//   commander  → membership_tuner (membership decisions drive join/leave rewards)
//   tuner      → lane_tuner       (parameter optimization drives throughput/latency)
//   guardian   → safety_guardian   (safety enforcement drives safety rewards)
//
// Python (marl/networks/role_critic.py) uses reward head names because it trains
// on reward signals. The rrg/grg from MOISE+ roles feed into OrganizationalReward
// which modifies the TOTAL scalar reward (additive to per-head decomposition).

// RoleToRewardHead maps each MOISE+ organizational role to its primary MARL reward head.
// This is the canonical cross-language mapping used by both Go and Python.
var RoleToRewardHead = map[RoleName]string{
	RoleSentinel:  "recovery_tuner",
	RoleCommander: "membership_tuner",
	RoleTuner:     "lane_tuner",
	RoleGuardian:  "safety_guardian",
}

// MissionName identifies one of the three organizational missions.
type MissionName string

const (
	MissionDefense     MissionName = "defense"     // {g_det →seq g_evict}
	MissionPerformance MissionName = "performance" // {g_tp ‖ g_stab}
	MissionSafety      MissionName = "safety"      // {g_safe}
)

// GoalName identifies one of the five organizational goals (Eq. missions).
type GoalName string

const (
	GoalDetect GoalName = "g_det"   // detect Byzantine agent within κ_det epochs
	GoalEvict  GoalName = "g_evict" // evict within τ epochs after detection
	GoalTP     GoalName = "g_tp"    // maintain throughput ≥ (1-ε)·tp_bar
	GoalStab   GoalName = "g_stab"  // avoid unnecessary reconfigurations
	GoalSafe   GoalName = "g_safe"  // keep safety margin ≥ δ_s
)

// GoalMission maps each goal to its mission.
func GoalMission(g GoalName) MissionName {
	switch g {
	case GoalDetect, GoalEvict:
		return MissionDefense
	case GoalTP, GoalStab:
		return MissionPerformance
	case GoalSafe:
		return MissionSafety
	default:
		return ""
	}
}

// RoleConfig holds hyperparameters for the role decomposition.
type RoleConfig struct {
	// FaultThresholdHigh (θ_high): fault probability above which Sentinel
	// should trigger a rotate action.
	FaultThresholdHigh float64

	// DetectionDeadline (κ_det): max epochs to detect a Byzantine agent.
	DetectionDeadline int

	// EvictionDeadline (τ): max epochs to evict after detection.
	EvictionDeadline int

	// StabilityThreshold (η_stab): max param change norm before Tuner penalty.
	StabilityThreshold float64

	// SafetyMarginTarget (δ_s): minimum acceptable safety margin.
	SafetyMarginTarget float64

	// Reward magnitudes for rrg penalties.
	RMiss  float64 // penalty for Sentinel missing detection
	RFalse float64 // penalty for Commander false eviction
	RChurn float64 // penalty for Tuner excessive param change

	// Bonus magnitude for grg goal achievement.
	RBonus float64

	// Lambda5 (λ₅): weight for goal achievement bonuses.
	Lambda5 float64

	// Lambda6 (λ₆): weight for role violation penalties.
	Lambda6 float64

	// Per-goal weights (w_g).
	GoalWeights map[GoalName]float64
}

// DefaultRoleConfig returns calibrated defaults matching Appendix Hyperparams.
func DefaultRoleConfig() RoleConfig {
	return RoleConfig{
		FaultThresholdHigh: 0.7,
		DetectionDeadline:  10,
		EvictionDeadline:   5,
		StabilityThreshold: 0.3,
		SafetyMarginTarget: 0.25,
		RMiss:              1.0,
		RFalse:             0.5,
		RChurn:             0.3,
		RBonus:             1.0,
		Lambda5:            0.1,
		Lambda6:            0.2,
		GoalWeights: map[GoalName]float64{
			GoalDetect: 1.0,
			GoalEvict:  1.0,
			GoalTP:     0.5,
			GoalStab:   0.5,
			GoalSafe:   1.0,
		},
	}
}

// --- Role Reward Guide (rrg, Eq. rrg) ---

// AgentRoleContext captures per-agent, per-epoch state needed by rrg/grg.
type AgentRoleContext struct {
	// Trust estimation outputs
	FaultProbs map[uint64]float64 // per-node fault probability f̂_t^k

	// Actions taken this epoch
	ReconfigTargets map[uint64]int // node → -1 (evict), +1 (join), 0 (no-op)
	RotateTriggered bool           // whether rotate was triggered
	ParamDelta      float64        // ‖param_t - param_{t-1}‖

	// Ground truth (for grg, available in training only)
	ByzantineSet map[uint64]bool // true Byzantine nodes (training signal)

	// Detection/eviction timing
	DetectionEpoch int // epoch when Byzantine agent first detected (-1 = not yet)
	EvictionEpoch  int // epoch when Byzantine agent evicted (-1 = not yet)
	CurrentEpoch   int

	// Performance metrics
	ThroughputRatio float64 // actual_tp / target_tp
	StableEpochs    int     // consecutive epochs without reconfig
	WindowSize      int     // W

	// Safety margin (from SafetyFilter.PerInstanceMargin)
	SafetyMargin float64
}

// RoleRewardGuide computes rrg penalties per role (Eq. rrg).
// Returns map[RoleName]float64 with non-positive values (penalties).
func RoleRewardGuide(ctx AgentRoleContext, cfg RoleConfig) map[RoleName]float64 {
	rrg := make(map[RoleName]float64)

	// Sentinel: missed detection penalty
	// If any node has f̂ > θ_high but was NOT evicted (reconfig != -1)
	for nodeID, fp := range ctx.FaultProbs {
		if fp > cfg.FaultThresholdHigh {
			rc, exists := ctx.ReconfigTargets[nodeID]
			if !exists || rc != -1 {
				rrg[RoleSentinel] -= cfg.RMiss
			}
		}
	}

	// Commander: false eviction penalty
	// If evicting a node that is NOT Byzantine
	for nodeID, rc := range ctx.ReconfigTargets {
		if rc == -1 {
			if ctx.ByzantineSet != nil && !ctx.ByzantineSet[nodeID] {
				rrg[RoleCommander] -= cfg.RFalse
			}
		}
	}

	// Tuner: excessive parameter churn penalty
	if ctx.ParamDelta > cfg.StabilityThreshold {
		rrg[RoleTuner] -= cfg.RChurn
	}

	// Guardian: no separate rrg penalty (handled by safety filter)
	return rrg
}

// GoalRewardGuide computes grg bonuses per goal (Eq. grg).
// Returns map[GoalName]float64 with non-negative values (bonuses).
func GoalRewardGuide(ctx AgentRoleContext, cfg RoleConfig) map[GoalName]float64 {
	grg := make(map[GoalName]float64)

	// g_det: detection speed bonus
	if ctx.DetectionEpoch >= 0 {
		dt := ctx.DetectionEpoch
		if dt <= cfg.DetectionDeadline {
			grg[GoalDetect] = cfg.RBonus * float64(cfg.DetectionDeadline-dt) / float64(cfg.DetectionDeadline)
		}
	}

	// g_evict: eviction speed bonus
	if ctx.EvictionEpoch >= 0 && ctx.DetectionEpoch >= 0 {
		dt := ctx.EvictionEpoch - ctx.DetectionEpoch
		if dt >= 0 && dt <= cfg.EvictionDeadline {
			grg[GoalEvict] = cfg.RBonus * float64(cfg.EvictionDeadline-dt) / float64(cfg.EvictionDeadline)
		}
	}

	// g_stab: stability bonus (W_stable / W)
	if ctx.WindowSize > 0 {
		ratio := float64(ctx.StableEpochs) / float64(ctx.WindowSize)
		if ratio > 1.0 {
			ratio = 1.0
		}
		grg[GoalStab] = cfg.RBonus * ratio
	}

	// g_tp and g_safe are implicit (base reward already handles throughput;
	// safety margin is in SafetyMarginPenalty). No separate grg bonus here
	// to avoid double-counting.

	return grg
}

// OrganizationalReward computes r_t^org = r_base + λ₅·Σ w_g·grg - λ₆·Σ rrg
// (Eq. reward-org).
func OrganizationalReward(baseReward float64, ctx AgentRoleContext, cfg RoleConfig) (float64, map[string]float64) {
	rrg := RoleRewardGuide(ctx, cfg)
	grg := GoalRewardGuide(ctx, cfg)

	// Goal achievement bonus: λ₅ Σ w_g · grg_g
	goalBonus := 0.0
	for g, bonus := range grg {
		w, ok := cfg.GoalWeights[g]
		if !ok {
			w = 1.0
		}
		goalBonus += w * bonus
	}
	goalBonus *= cfg.Lambda5

	// Role violation penalty: λ₆ Σ rrg_ς (already negative)
	rolePenalty := 0.0
	for _, penalty := range rrg {
		rolePenalty += penalty
	}
	rolePenalty *= cfg.Lambda6

	total := baseReward + goalBonus + rolePenalty

	// Flatten role rewards for trace logging
	details := make(map[string]float64)
	for role, val := range rrg {
		details["rrg_"+string(role)] = val
	}
	for goal, val := range grg {
		details["grg_"+string(goal)] = val
	}
	details["goal_bonus"] = goalBonus
	details["role_penalty"] = rolePenalty

	return total, details
}

// ConstraintHardness returns the constraint hardness ch for the Guardian role.
// When ch = 1.0, the role action guide is equivalent to the hard safety filter
// (Proposition rag-safety).
func ConstraintHardness(role RoleName) float64 {
	if role == RoleGuardian {
		return 1.0
	}
	return 0.0
}

// RoleActionGuide applies a soft role-specific action filter.
// For Guardian (ch=1.0), it delegates to SafetyFilter.MaskUnsafeAction.
// For other roles, it returns the action unchanged (soft guidance via rrg).
func RoleActionGuide(role RoleName, obs Observation, action Action, sf *SafetyFilter) (Action, bool) {
	if role == RoleGuardian && sf != nil {
		masked, _, anyMasked := sf.MaskUnsafeAction(obs, action)
		return masked, anyMasked
	}
	return action, false
}

// GradientVarianceReduction estimates the variance reduction factor from
// Proposition gradient-variance: Var[ĝ_org] / Var[ĝ_base].
// Returns a value in (0, 1] when role rewards positively correlate with
// the base gradient (covTerm > 0).
//
//	ratio = 1 - 2·λ₅·cov / (λ₅²·σ²_role + ε)  (clamped to [epsilon, 1])
func GradientVarianceReduction(lambda5, covTerm, sigmaRoleSq float64) float64 {
	if sigmaRoleSq <= 0 || lambda5 <= 0 {
		return 1.0
	}
	ratio := 1.0 - 2.0*lambda5*covTerm/(lambda5*lambda5*sigmaRoleSq+1e-12)
	return math.Max(ratio, 1e-6)
}
