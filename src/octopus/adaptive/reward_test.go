package adaptive

import "testing"

func TestDefaultRewardModelPenalizesLatencyBacklogAndRejects(t *testing.T) {
	model := DefaultRewardModel()
	obs := Observation{
		ThroughputTPS:  2000,
		LatencyP95Ms:   800,
		BacklogPending: 500,
		RejectTotal:    20,
		AdversaryScore: 0.7,
		ChurnRate:      0.4,
	}
	reward := model.Compute(obs, Action{})
	if reward.Total >= 0 {
		t.Fatalf("expected degraded conditions to produce negative reward, got %f", reward.Total)
	}
	if reward.TeamReward != reward.Total {
		t.Fatalf("expected team reward to match total reward, got %+v", reward)
	}
	if reward.RoleRewards["recovery_tuner"] >= 0 {
		t.Fatalf("expected recovery_tuner reward to be negative, got %+v", reward.RoleRewards)
	}
}

func TestDefaultRewardModelRewardsHealthyOperation(t *testing.T) {
	model := DefaultRewardModel()
	obs := Observation{
		ThroughputTPS:  8000,
		LatencyP95Ms:   60,
		BacklogPending: 5,
		RejectTotal:    0,
		AdversaryScore: 0.0,
		ChurnRate:      0.0,
	}
	reward := model.Compute(obs, Action{})
	if reward.Total <= 0 {
		t.Fatalf("expected healthy conditions to produce positive reward, got %f", reward.Total)
	}
	if reward.RoleRewards["lane_tuner"] <= reward.RoleRewards["recovery_tuner"] {
		t.Fatalf("expected lane_tuner reward to dominate recovery reward, got %+v", reward.RoleRewards)
	}
}

func TestDefaultRewardModelPenalizesUnsafeMembershipActions(t *testing.T) {
	model := DefaultRewardModel()
	reward := model.Compute(Observation{ValidatorCount: 3, LocalValidator: true}, Action{SubmitLeave: true})
	if reward.RoleRewards["safety_guardian"] >= 0 {
		t.Fatalf("expected unsafe leave to be penalized by safety_guardian, got %+v", reward.RoleRewards)
	}
}

func TestPaperReward_MultiInstanceSafetyPenalty(t *testing.T) {
	weights := DefaultPaperRewardWeights()

	// Two instances: one safe (n=7, f=1 → threshold=3+1+0=4 → safe),
	// one unsafe (n=3, f=1 → threshold=4 → deficit=1)
	obs := Observation{
		ThroughputTPS: 1000,
		LatencyP95Ms:  100,
		ChurnRate:     0.0,
		Agents: []AgentObservation{
			{InstanceID: 1, ValidatorCount: 7, FaultsEstimate: 1},
			{InstanceID: 2, ValidatorCount: 3, FaultsEstimate: 1},
		},
	}

	r := PaperReward(obs, 0, weights)
	// Expected: tp=1.0, lat=-0.4, churn=0, safety=-1*2.0=-2.0 → r = 1.0-0.4-2.0 = -1.4
	expected := 1000*weights.Lambda1 - 100*weights.Lambda2 - 0 - 1*weights.Lambda4
	if abs(r-expected) > 1e-9 {
		t.Fatalf("expected PaperReward=%f, got %f", expected, r)
	}
}

func TestPaperReward_FallbackToGlobalWhenNoAgents(t *testing.T) {
	weights := DefaultPaperRewardWeights()

	// No per-instance agents → uses global ValidatorCount
	obs := Observation{
		ThroughputTPS:  500,
		LatencyP95Ms:   50,
		ChurnRate:      0.0,
		ValidatorCount: 4,
	}

	r := PaperReward(obs, 0, weights)
	// n=4, f=1, threshold=4, deficit=0 → no safety penalty
	expected := 500*weights.Lambda1 - 50*weights.Lambda2
	if abs(r-expected) > 1e-9 {
		t.Fatalf("expected PaperReward=%f (no penalty), got %f", expected, r)
	}
}

func TestPaperReward_AllInstancesSafe(t *testing.T) {
	weights := DefaultPaperRewardWeights()

	obs := Observation{
		ThroughputTPS: 2000,
		LatencyP95Ms:  80,
		ChurnRate:     0.1,
		Agents: []AgentObservation{
			{InstanceID: 1, ValidatorCount: 10, FaultsEstimate: 2},
			{InstanceID: 2, ValidatorCount: 10, FaultsEstimate: 2},
			{InstanceID: 3, ValidatorCount: 10, FaultsEstimate: 2},
		},
	}

	r := PaperReward(obs, 0, weights)
	// n=10, f=2, threshold=7 → 10>=7 → safe for all. No safety penalty.
	expected := 2000*weights.Lambda1 - 80*weights.Lambda2 - 0.1*weights.Lambda3
	if abs(r-expected) > 1e-9 {
		t.Fatalf("expected PaperReward=%f, got %f", expected, r)
	}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
