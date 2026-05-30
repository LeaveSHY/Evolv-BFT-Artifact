package adaptive

import "math"

// ═══════════════════════════════════════════════════════════════════════════════
// BFT Safety Constraints (Section III-D, Algorithm 1 lines 13-17)
//
// Implements the paper's pre-argmax safety filter and Lyapunov-inspired
// safety margin computation for the adaptive trust management layer.
//
// Three mechanisms, defense-in-depth:
//   1. Hard Mask (P3): n_v >= 3f_v + 1 quorum invariant — blocks unsafe evictions
//   2. Safety Margin: penalizes proximity to quorum boundary (λ₄ in reward)
//   3. Coupled Constraint: cross-instance total validator budget check
// ═══════════════════════════════════════════════════════════════════════════════

// SafetyFilter implements the pre-argmax safety mask from Algorithm 5 (Appendix).
// It enforces the BFT quorum invariant: n_after >= 3f + 1 + DeltaS for all instances.
type SafetyFilter struct {
	// DeltaS: additive safety margin above 3f+1 (paper: δ_s ≥ 0).
	// Algorithm 5 line 5: n_after >= 3f_i^rep + 1 + δ_s.
	// Default 1 ensures one-eviction headroom before reaching the BFT boundary.
	DeltaS int

	// GlobalBudgetMin: minimum total validators across all instances.
	// Prevents the system from shrinking below safe operation.
	GlobalBudgetMin int
}

// DefaultSafetyFilter returns a filter with standard BFT safety parameters.
// DeltaS=1 matches the paper's δ_s=1 used in experiments.
func DefaultSafetyFilter() SafetyFilter {
	return SafetyFilter{
		DeltaS:          1, // paper Algorithm 5: δ_s ≥ 1
		GlobalBudgetMin: 4, // minimum 4 validators globally
	}
}

// InstanceState captures the per-instance validator state for safety checking.
type InstanceState struct {
	InstanceID     uint64
	ValidatorCount int // current n_v
	CommitteeSize  int // current committee size
	PendingEvicts  int // number of evictions proposed in this action
	PendingAdmits  int // number of admissions proposed in this action
	// FaultsEstimate: trust system's estimate of Byzantine nodes remaining
	// AFTER the proposed evictions (paper: f_i^rep). Conservative default: 1.
	// The safety filter uses this as the f in "n_after >= 3f+1+δ_s".
	// This ensures monotonicity: with fixed f, more evictions only reduce
	// nAfter, making the check strictly harder to pass.
	FaultsEstimate int
}

// BFTQuorumThreshold returns the minimum validators needed for instance with f faults.
// For BFT: n >= 3f + 1, so min validators = 3f + 1 where f = (n-1)/3.
// This is the tightest bound: we need at least 4 for f=1.
func BFTQuorumThreshold(targetFaults int) int {
	return 3*targetFaults + 1
}

// MinValidatorsForSafety returns the minimum validators that must remain
// in an instance to maintain BFT safety with δ_s margin.
// For n validators, f = (n-1)/3, and we need n_after >= 3*f_after + 1 + δ_s.
func MinValidatorsForSafety(currentValidators int, deltaS int) int {
	if currentValidators <= 4+deltaS {
		return currentValidators // cannot reduce below minimum safe size
	}
	// Minimum: 3*1 + 1 + δ_s = 4 + δ_s (for f=1)
	return 4 + deltaS
}

// ComputePhi implements the Joint Safety Invariant Φ_t from §IV Definition 3:
//   Φ(Σ_t) = min( min_j(|Ω_t^j| - 3f_j - 1), m_GBC - 3f_GBC - 1 )
// Returns the minimum safety margin across all instances and GBC.
// Φ ≥ 0 is required for BFT safety; Φ > 0 indicates headroom.
func ComputePhi(instances []InstanceState, gbcMembers int, gbcFaults int) int {
	phi := gbcMembers - 3*gbcFaults - 1
	for _, inst := range instances {
		margin := inst.ValidatorCount - inst.PendingEvicts + inst.PendingAdmits - 3*inst.FaultsEstimate - 1
		if margin < phi {
			phi = margin
		}
	}
	return phi
}

// CheckQuorumInvariant verifies the BFT quorum invariant for a single instance.
// Returns true if the action is safe (n_after >= 3f_i^rep + 1 + δ_s).
//
// This implements Algorithm 5, line 5:
//   n_after = |Ω_t^i| - |{k : reconfig_t^{i,k} = -1}|
//   if n_after < 3f_i^rep + 1 + δ_s: block (mask action)
//
// Monotonicity guarantee: f is taken from FaultsEstimate (trust system's
// conservative estimate of remaining Byzantine), NOT derived from nAfter.
// With fixed f, reducing nAfter (more evictions) can only make the check
// harder — eliminating the non-monotonic "cliff effect" of threshold(nAfter).
//
// Special case: if no membership change is proposed (PendingEvicts==0 and
// PendingAdmits==0), the action cannot violate safety and is always permitted.
func (sf SafetyFilter) CheckQuorumInvariant(state InstanceState) bool {
	// No membership change → cannot worsen safety
	if state.PendingEvicts == 0 && state.PendingAdmits == 0 {
		return true
	}
	nAfter := state.ValidatorCount - state.PendingEvicts + state.PendingAdmits
	if nAfter < 1 {
		return false
	}
	// f from trust system's estimate (paper: f_i^rep).
	// Conservative default: at least 1 fault tolerance required.
	f := state.FaultsEstimate
	if f < 1 {
		f = 1
	}
	threshold := 3*f + 1 + sf.DeltaS
	return nAfter >= threshold
}

// CheckCoupledConstraint verifies the cross-instance coupled constraint.
// The total validator pool must remain above the global budget minimum,
// and each instance with pending evictions must maintain n >= 3f+1+δ_s.
// Uses FaultsEstimate (trust-derived f) for monotonicity.
func (sf SafetyFilter) CheckCoupledConstraint(instances []InstanceState) bool {
	totalAfter := 0
	for _, inst := range instances {
		nAfter := inst.ValidatorCount - inst.PendingEvicts + inst.PendingAdmits
		// Only check per-instance threshold if there are pending evictions
		if inst.PendingEvicts > 0 {
			f := inst.FaultsEstimate
			if f < 1 {
				f = 1
			}
			if nAfter < 3*f+1+sf.DeltaS {
				return false // instance would violate 3f+1+δ_s
			}
		}
		totalAfter += nAfter
	}
	return totalAfter >= sf.GlobalBudgetMin
}

// MaskUnsafeAction applies the pre-argmax safety mask to an Action.
// It blocks evictions that would violate the BFT quorum invariant and
// records which instances were masked.
//
// Returns the masked action, a list of masked instance IDs, and a boolean
// indicating whether any masking occurred.
func (sf SafetyFilter) MaskUnsafeAction(obs Observation, action Action) (Action, []uint64, bool) {
	masked := action
	var maskedInstances []uint64
	anyMasked := false

	// Build per-instance state from observation + action
	instances := sf.buildInstanceStates(obs, action)

	// Phase 1: Check each instance's quorum invariant independently
	for i, inst := range instances {
		if !sf.CheckQuorumInvariant(inst) {
			// Block: zero out evictions for this instance
			instances[i].PendingEvicts = 0
			anyMasked = true
			maskedInstances = append(maskedInstances, inst.InstanceID)
		}
	}

	// Phase 2: Check coupled cross-instance constraint
	if !sf.CheckCoupledConstraint(instances) {
		// Block all evictions globally
		maskedSet := make(map[uint64]bool, len(maskedInstances))
		for _, id := range maskedInstances {
			maskedSet[id] = true
		}
		for i := range instances {
			if instances[i].PendingEvicts > 0 {
				instances[i].PendingEvicts = 0
				anyMasked = true
				if !maskedSet[instances[i].InstanceID] {
					maskedInstances = append(maskedInstances, instances[i].InstanceID)
					maskedSet[instances[i].InstanceID] = true
				}
			}
		}
	}

	// Phase 3: Apply masking to the action
	if anyMasked {
		// Block global membership changes
		masked.SubmitLeave = false

		// Block per-instance reconfigurations for masked instances
		maskedSet := make(map[uint64]bool)
		for _, id := range maskedInstances {
			maskedSet[id] = true
		}
		if len(masked.AgentActions) > 0 {
			for idx, aa := range masked.AgentActions {
				if maskedSet[aa.InstanceID] {
					masked.AgentActions[idx].Reconfig = nil
					masked.AgentActions[idx].ReconfigEvictNodeIDs = nil
					masked.AgentActions[idx].ReconfigAdmitNodeIDs = nil
					// Preserve current configuration instead of applying change
					for _, agent := range obs.Agents {
						if agent.InstanceID == aa.InstanceID {
							masked.AgentActions[idx].CommitteeSize = agent.CommitteeSize
							masked.AgentActions[idx].PacemakerTimeoutMs = agent.PacemakerTimeoutMs
							masked.AgentActions[idx].MempoolMaxBatchTxs = agent.MempoolMaxBatchTxs
							masked.AgentActions[idx].MempoolProposalIntervalMs = agent.MempoolProposalIntervalMs
							break
						}
					}
				}
			}
		}
	}

	return masked, maskedInstances, anyMasked
}

// buildInstanceStates extracts per-instance state from observation and action.
func (sf SafetyFilter) buildInstanceStates(obs Observation, action Action) []InstanceState {
	// If no per-instance agents, treat as single instance
	if len(obs.Agents) == 0 {
		evicts := 0
		admits := 0
		if action.SubmitLeave {
			evicts = 1
		}
		if action.SubmitJoin {
			admits = 1
		}
		return []InstanceState{{
			InstanceID:     0,
			ValidatorCount: obs.ValidatorCount,
			CommitteeSize:  obs.CommitteeSize,
			PendingEvicts:  evicts,
			PendingAdmits:  admits,
		}}
	}

	states := make([]InstanceState, 0, len(obs.Agents))
	actionsByInstance := make(map[uint64]AgentAction, len(action.AgentActions))
	for _, agentAction := range action.AgentActions {
		actionsByInstance[agentAction.InstanceID] = agentAction
	}
	for _, agent := range obs.Agents {
		evicts := 0
		admits := 0
		if agentAction, ok := actionsByInstance[agent.InstanceID]; ok {
			evicts, admits = reconfigCounts(agentAction)
		}
		states = append(states, InstanceState{
			InstanceID:     agent.InstanceID,
			ValidatorCount: agent.ValidatorCount,
			CommitteeSize:  agent.CommitteeSize,
			PendingEvicts:  evicts,
			PendingAdmits:  admits,
			FaultsEstimate: agent.FaultsEstimate,
		})
	}
	return states
}

func reconfigCounts(action AgentAction) (int, int) {
	if len(action.ReconfigEvictNodeIDs) > 0 || len(action.ReconfigAdmitNodeIDs) > 0 {
		return len(action.ReconfigEvictNodeIDs), len(action.ReconfigAdmitNodeIDs)
	}
	evicts := 0
	admits := 0
	for _, decision := range action.Reconfig {
		switch {
		case decision < 0:
			evicts++
		case decision > 0:
			admits++
		}
	}
	return evicts, admits
}

// ═══════════════════════════════════════════════════════════════════════════════
// Safety Margin (Lyapunov-inspired drift measure)
//
// Computes how far each instance is from the quorum boundary.
// Used as the λ₄ penalty term in the reward function (Eq. reward in paper).
//
// Lyapunov function: V(s) = Σ_i max(0, threshold_i - n_after_i)²
// Safety margin:    margin_i = (n_v^i - (3f_v^i + 1)) / n_v^i
//
// When margin > 0, we are safe. When margin → 0, we approach the boundary.
// The reward penalty is -λ₄ * Σ_i max(0, 1 - margin_i/target_margin).
// ═══════════════════════════════════════════════════════════════════════════════

// SafetyMarginConfig configures the Lyapunov safety margin computation.
type SafetyMarginConfig struct {
	// TargetMargin: desired minimum fractional margin above quorum threshold.
	// Default 0.25 means we want n_v to be at least 25% above 3f+1.
	TargetMargin float64

	// Lambda: weight of the safety margin penalty in the reward.
	// Higher values make the policy more conservative.
	Lambda float64
}

// DefaultSafetyMarginConfig returns standard safety margin parameters.
func DefaultSafetyMarginConfig() SafetyMarginConfig {
	return SafetyMarginConfig{
		TargetMargin: 0.25,
		Lambda:       0.5,
	}
}

// PerInstanceMargin computes the safety margin for a single instance.
// Returns a value in [0, 1] where 0 = at threshold, 1 = far from threshold.
func PerInstanceMargin(validatorCount int) float64 {
	if validatorCount <= 0 {
		return 0
	}
	f := (validatorCount - 1) / 3
	threshold := 3*f + 1
	if validatorCount < threshold {
		return 0 // below threshold (should never happen in safe system)
	}
	surplus := float64(validatorCount - threshold)
	return surplus / float64(validatorCount)
}

// LyapunovDrift computes the Lyapunov drift for a proposed action.
// Negative drift means the system is moving toward safety (good).
// Positive drift means the system is moving toward the boundary (bad).
//
// V(s) = Σ_i max(0, target - margin_i)²
// drift = V(s') - V(s)
func LyapunovDrift(before []InstanceState, after []InstanceState, targetMargin float64) float64 {
	vBefore := lyapunovValue(before, targetMargin)
	vAfter := lyapunovValue(after, targetMargin)
	return vAfter - vBefore
}

func lyapunovValue(states []InstanceState, targetMargin float64) float64 {
	v := 0.0
	for _, s := range states {
		nv := s.ValidatorCount - s.PendingEvicts + s.PendingAdmits
		margin := PerInstanceMargin(nv)
		deficit := math.Max(0, targetMargin-margin)
		v += deficit * deficit
	}
	return v
}

// SafetyMarginPenalty computes the reward penalty for the safety margin.
// Returns a non-negative value that should be subtracted from the reward.
//
// penalty = λ₄ * Σ_i max(0, 1 - margin_i / target_margin)
func SafetyMarginPenalty(obs Observation, cfg SafetyMarginConfig) float64 {
	if cfg.Lambda <= 0 || cfg.TargetMargin <= 0 {
		return 0
	}

	penalty := 0.0
	if len(obs.Agents) == 0 {
		margin := PerInstanceMargin(obs.ValidatorCount)
		deficit := math.Max(0, 1.0-margin/cfg.TargetMargin)
		penalty += deficit
	} else {
		for _, agent := range obs.Agents {
			margin := PerInstanceMargin(agent.ValidatorCount)
			deficit := math.Max(0, 1.0-margin/cfg.TargetMargin)
			penalty += deficit
		}
	}

	return cfg.Lambda * penalty
}
