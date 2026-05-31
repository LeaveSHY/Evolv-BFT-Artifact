package adaptive

// ═══════════════════════════════════════════════════════════════════════════════
// Deployment Reward Model (adaptive/reward.go)
//
// Paper Mapping:
//   - Paper Eq.14 (§III-D): base 4-term reward (tp, latency, vc, safety margin)
//   - Paper Eq.reward-org (Appendix B): extends with λ₅ (goal bonuses) + λ₆ (role penalties)
//   - This file: operational deployment extension with ~20 fine-grained signals
//
// Relationship to paper:
//   ThroughputGain       → λ₁ · Σ tp_t^i
//   LatencyPenalty       → λ₂ · Σ ℓ_t^i (extended with tail + recovery)
//   ChurnPenalty         → λ₃ · Σ vc_t^i (extended with jitter, adversary score)
//   SafetyViolationCost  → λ₄ · safety margin (Lyapunov continuous, not discrete δ_s)
//   RoleRewards map      → λ₅/λ₆ organizational reward (Eq.reward-org)
//
// The additional signals (backlog, reject, fairness, participation, discovery,
// membership gap, AI load, heterogeneity) refine production behavior without
// altering the O(√T) regret structure (bounded additive terms in constant C).
// ═══════════════════════════════════════════════════════════════════════════════

type RewardModel interface {
	Compute(observation Observation, action Action) RewardSignal
}

type RewardWeights struct {
	ThroughputGain      float64
	LatencyPenalty      float64
	TailLatencyPenalty  float64
	RecoveryPenalty     float64
	BacklogPenalty      float64
	RejectPenalty       float64
	AdversaryPenalty    float64
	ChurnPenalty        float64
	JitterPenalty       float64
	AILoadPenalty       float64
	HeterogeneityBonus  float64
	CommitteeCost       float64
	PendingMembership   float64
	ConfigGapPenalty    float64
	ParticipationBonus  float64
	DiscoveryBonus      float64
	JoinBonus           float64
	LeavePenalty        float64
	FairnessPenalty     float64
	SafetyViolationCost float64
}

type WeightedRewardModel struct {
	weights RewardWeights
}

func DefaultRewardModel() WeightedRewardModel {
	return WeightedRewardModel{
		weights: RewardWeights{
			ThroughputGain:      1.0 / 1000.0,
			LatencyPenalty:      1.0 / 250.0,
			TailLatencyPenalty:  1.0 / 150.0,
			RecoveryPenalty:     1.0 / 300.0,
			BacklogPenalty:      1.0 / 200.0,
			RejectPenalty:       0.12,
			AdversaryPenalty:    1.4,
			ChurnPenalty:        1.2,
			JitterPenalty:       1.0 / 500.0,
			AILoadPenalty:       0.8,
			HeterogeneityBonus:  0.2,
			CommitteeCost:       1.0 / 64.0,
			PendingMembership:   0.4,
			ConfigGapPenalty:    0.6,
			ParticipationBonus:  0.8,
			DiscoveryBonus:      0.1,
			JoinBonus:           0.3,
			LeavePenalty:        0.2,
			FairnessPenalty:     0.25,
			SafetyViolationCost: 1.5,
		},
	}
}

func (m WeightedRewardModel) Compute(observation Observation, action Action) RewardSignal {
	w := m.weights
	roleRewards := map[string]float64{
		"lane_tuner":       0,
		"recovery_tuner":   0,
		"membership_tuner": 0,
		"safety_guardian":  0,
	}

	throughputReward := observation.ThroughputTPS * w.ThroughputGain
	latencyPenalty := observation.LatencyP95Ms*w.LatencyPenalty + observation.LatencyP99Ms*w.TailLatencyPenalty
	recoveryPenalty := observation.RecoveryP95Ms * w.RecoveryPenalty
	backlogPenalty := float64(observation.BacklogPending+observation.BacklogMissing) * w.BacklogPenalty
	rejectPenalty := float64(observation.RejectTotal) * w.RejectPenalty
	stabilityPenalty := observation.AdversaryScore*w.AdversaryPenalty + observation.ChurnRate*w.ChurnPenalty + observation.NetworkJitterMs*w.JitterPenalty
	resourcePenalty := observation.AILoadScore*w.AILoadPenalty + float64(maxInt(action.CommitteeSize, 0))*w.CommitteeCost
	membershipPenalty := float64(observation.PendingJoins+observation.PendingLeaves) * w.PendingMembership
	if observation.HighestKnownConfigID > observation.CurrentConfigID {
		membershipPenalty += float64(observation.HighestKnownConfigID-observation.CurrentConfigID) * w.ConfigGapPenalty
	}
	fairnessPenalty := 0.0
	if observation.ValidatorCount > 0 {
		committee := action.CommitteeSize
		if committee <= 0 {
			committee = observation.CommitteeSize
		}
		utilization := float64(committee) / float64(observation.ValidatorCount)
		fairnessPenalty = absFloat64(utilization-observation.HeterogeneityScore) * w.FairnessPenalty
	}
	safetyPenalty := 0.0
	if action.SubmitJoin && action.SubmitLeave {
		safetyPenalty += w.SafetyViolationCost
	}
	if action.SubmitLeave && observation.ValidatorCount <= 3 {
		safetyPenalty += w.SafetyViolationCost
	}
	if action.CommitteeSize > 0 && observation.ValidatorCount > 0 && action.CommitteeSize > observation.ValidatorCount {
		safetyPenalty += w.SafetyViolationCost
	}

	teamReward := throughputReward - latencyPenalty - recoveryPenalty - backlogPenalty - rejectPenalty - stabilityPenalty - resourcePenalty - membershipPenalty - fairnessPenalty - safetyPenalty
	if observation.CanParticipate {
		teamReward += w.ParticipationBonus
	}
	if observation.LocalValidator {
		teamReward += 0.25
	}
	if action.SubmitJoin && !observation.LocalValidator {
		teamReward += w.JoinBonus
	}
	if action.SubmitLeave && observation.LocalValidator {
		teamReward -= w.LeavePenalty
	}
	if action.HydraDiscoveryTarget > 0 && !observation.CanParticipate {
		teamReward += w.DiscoveryBonus
	}
	teamReward += observation.HeterogeneityScore * w.HeterogeneityBonus

	roleRewards["lane_tuner"] = throughputReward - latencyPenalty - backlogPenalty - resourcePenalty - fairnessPenalty
	roleRewards["recovery_tuner"] = -recoveryPenalty - rejectPenalty - stabilityPenalty
	roleRewards["membership_tuner"] = -membershipPenalty
	if action.SubmitJoin && !observation.LocalValidator {
		roleRewards["membership_tuner"] += w.JoinBonus
	}
	if action.SubmitLeave && observation.LocalValidator {
		roleRewards["membership_tuner"] -= w.LeavePenalty
	}
	if action.HydraDiscoveryTarget > 0 && !observation.CanParticipate {
		roleRewards["membership_tuner"] += w.DiscoveryBonus
	}
	roleRewards["safety_guardian"] = -safetyPenalty

	// --- Organizational rrg penalties (Eq. reward-org, roles.go RoleToRewardHead) ---
	// These lightweight rrg-style penalties use observation data available at
	// Compute() time, aligning MOISE+ role responsibilities with reward heads.
	// Sentinel → recovery_tuner: missed-detection penalty when adversary score is
	// high but no eviction/leave was triggered.
	if observation.AdversaryScore >= 0.7 && !action.SubmitLeave {
		roleRewards["recovery_tuner"] -= 0.3 // RoleSentinel rrg proxy
	}
	// Commander → membership_tuner: excessive-churn penalty
	if observation.ChurnRate > 0.4 {
		roleRewards["membership_tuner"] -= observation.ChurnRate * 0.5 // RoleCommander rrg proxy
	}
	// Tuner → lane_tuner: param instability penalty (network jitter as proxy)
	if observation.NetworkJitterMs > 50 {
		roleRewards["lane_tuner"] -= 0.1 // RoleTuner rrg proxy
	}

	return RewardSignal{
		Total:       teamReward,
		TeamReward:  teamReward,
		RoleRewards: roleRewards,
	}
}

func absFloat64(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}

// PaperRewardWeights holds the 4 coefficients from paper Eq.14.
type PaperRewardWeights struct {
	Lambda1 float64 // throughput weight
	Lambda2 float64 // latency weight
	Lambda3 float64 // view-change/churn weight
	Lambda4 float64 // safety-margin penalty weight
}

// DefaultPaperRewardWeights returns the default Eq.14 coefficients.
func DefaultPaperRewardWeights() PaperRewardWeights {
	return PaperRewardWeights{
		Lambda1: 1.0 / 1000.0,
		Lambda2: 1.0 / 250.0,
		Lambda3: 1.2,
		Lambda4: 2.0,
	}
}

// PaperReward computes the exact 4-term reward from paper Eq.14 (§III-D):
//
//	r_t = λ₁·Σ tp_t^i  −  λ₂·Σ ℓ_t^i  −  λ₃·Σ vc_t^i  −  λ₄·Σ max(0, 3f_i+1+δ_s − |Ω_t^i|)
//
// Parameters:
//   - obs: current observation (throughput, latency, churn, validator count)
//   - deltaS: safety margin parameter (δ_s from Algorithm 5)
//   - weights: the 4 lambda coefficients
//
// This function exists for paper-code alignment verification. The production
// reward model (WeightedRewardModel.Compute) extends this with additional
// deployment signals while preserving the same regret structure.
func PaperReward(obs Observation, deltaS int, weights PaperRewardWeights) float64 {
	// Term 1: throughput reward (Σ tp_t^i)
	tp := obs.ThroughputTPS * weights.Lambda1

	// Term 2: latency penalty (Σ ℓ_t^i)
	latency := obs.LatencyP95Ms * weights.Lambda2

	// Term 3: view-change/churn penalty (Σ vc_t^i)
	churn := obs.ChurnRate * weights.Lambda3

	// Term 4: safety-margin violation penalty — sum over instances
	// Σ_i max(0, 3f_i+1+δ_s − |Ω_t^i|)
	safetyPenalty := 0.0
	if len(obs.Agents) > 0 {
		for _, agent := range obs.Agents {
			if agent.ValidatorCount <= 0 {
				continue
			}
			f := agent.FaultsEstimate
			if f < 1 {
				f = 1
			}
			threshold := 3*f + 1 + deltaS
			deficit := threshold - agent.ValidatorCount
			if deficit > 0 {
				safetyPenalty += float64(deficit) * weights.Lambda4
			}
		}
	} else if obs.ValidatorCount > 0 {
		// Fallback for single-instance observations
		f := (obs.ValidatorCount - 1) / 3
		if f < 1 {
			f = 1
		}
		threshold := 3*f + 1 + deltaS
		deficit := threshold - obs.ValidatorCount
		if deficit > 0 {
			safetyPenalty = float64(deficit) * weights.Lambda4
		}
	}

	return tp - latency - churn - safetyPenalty
}
