// Package adaptive implements runtime policy adaptation, guardrails, and SFAC bridges.
package adaptive

import "math"

func sfacActionFromTuple(instanceID uint64, reconfig []int, rotate bool, params []float64, obs Observation) AgentAction {
	aa := baseAgentAction(instanceID, obs)
	aa.Reconfig = append([]int(nil), reconfig...)
	aa.RotateLeader = rotate
	aa.ParamVector = append([]float64(nil), params...)
	applySFACReconfig(&aa, reconfig, obs.TrustSnapshots)
	applySFACParams(&aa, params)
	return aa
}

func baseAgentAction(instanceID uint64, obs Observation) AgentAction {
	for _, agent := range obs.Agents {
		if agent.InstanceID == instanceID {
			return AgentAction{
				InstanceID:                agent.InstanceID,
				CommitteeSize:             agent.CommitteeSize,
				PacemakerTimeoutMs:        agent.PacemakerTimeoutMs,
				MempoolMaxBatchTxs:        agent.MempoolMaxBatchTxs,
				MempoolProposalIntervalMs: agent.MempoolProposalIntervalMs,
			}
		}
	}
	return AgentAction{
		InstanceID:                instanceID,
		CommitteeSize:             obs.CommitteeSize,
		PacemakerTimeoutMs:        obs.PacemakerTimeoutMs,
		MempoolMaxBatchTxs:        obs.MempoolMaxBatchTxs,
		MempoolProposalIntervalMs: obs.MempoolProposalIntervalMs,
	}
}

func applySFACReconfig(action *AgentAction, reconfig []int, snapshots []TrustSnapshot) {
	if action == nil || len(reconfig) == 0 || len(snapshots) == 0 {
		return
	}
	for idx, decision := range reconfig {
		if decision == 0 || idx >= len(snapshots) {
			continue
		}
		nodeID := snapshots[idx].NodeID
		if decision < 0 {
			action.ReconfigEvictNodeIDs = append(action.ReconfigEvictNodeIDs, nodeID)
			continue
		}
		action.ReconfigAdmitNodeIDs = append(action.ReconfigAdmitNodeIDs, nodeID)
	}
}

func applySFACParams(action *AgentAction, params []float64) {
	if action == nil || len(params) == 0 {
		return
	}
	if len(params) == 1 {
		applySFACScalarParam(action, params[0])
		return
	}
	if len(params) > 0 {
		action.CommitteeSize = sfacParamInt(params[0], action.CommitteeSize, 2)
	}
	if len(params) > 1 {
		action.PacemakerTimeoutMs = sfacParamScaledInt(params[1], action.PacemakerTimeoutMs, 0.5)
	}
	if len(params) > 2 {
		action.MempoolMaxBatchTxs = sfacParamScaledInt(params[2], action.MempoolMaxBatchTxs, 0.5)
	}
	if len(params) > 3 {
		action.MempoolProposalIntervalMs = sfacParamScaledInt(params[3], action.MempoolProposalIntervalMs, 0.5)
	}
}

func applySFACScalarParam(action *AgentAction, raw float64) {
	if math.IsNaN(raw) || math.IsInf(raw, 0) {
		return
	}
	if math.Abs(raw) > 1 {
		action.PacemakerTimeoutMs = int(math.Round(raw))
		return
	}
	scalar := clampFloat(raw, -1, 1)
	action.PacemakerTimeoutMs = scaledInt(action.PacemakerTimeoutMs, 1+0.5*scalar)
	action.MempoolProposalIntervalMs = scaledInt(action.MempoolProposalIntervalMs, 1+0.5*scalar)
	action.MempoolMaxBatchTxs = scaledInt(action.MempoolMaxBatchTxs, 1-0.5*scalar)
}

func sfacParamInt(raw float64, current int, deltaScale float64) int {
	if math.IsNaN(raw) || math.IsInf(raw, 0) {
		return current
	}
	if math.Abs(raw) > 1 {
		return int(math.Round(raw))
	}
	return current + int(math.Round(raw*deltaScale))
}

func sfacParamScaledInt(raw float64, current int, maxFraction float64) int {
	if math.IsNaN(raw) || math.IsInf(raw, 0) {
		return current
	}
	if math.Abs(raw) > 1 {
		return int(math.Round(raw))
	}
	return scaledInt(current, 1+clampFloat(raw, -1, 1)*maxFraction)
}

func scaledInt(current int, factor float64) int {
	if current == 0 {
		current = 1
	}
	if factor < 0.1 {
		factor = 0.1
	}
	return int(math.Round(float64(current) * factor))
}

func clampFloat(value, minValue, maxValue float64) float64 {
	if value < minValue {
		return minValue
	}
	if value > maxValue {
		return maxValue
	}
	return value
}
