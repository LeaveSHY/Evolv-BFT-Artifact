package adaptive

import (
	"math"
	"testing"
)

func TestAllRolesReturnsFour(t *testing.T) {
	roles := AllRoles()
	if len(roles) != 4 {
		t.Fatalf("expected 4 roles, got %d", len(roles))
	}
	expected := map[RoleName]bool{RoleSentinel: true, RoleCommander: true, RoleTuner: true, RoleGuardian: true}
	for _, r := range roles {
		if !expected[r] {
			t.Fatalf("unexpected role: %s", r)
		}
	}
}

func TestGoalMission(t *testing.T) {
	tests := []struct {
		goal    GoalName
		mission MissionName
	}{
		{GoalDetect, MissionDefense},
		{GoalEvict, MissionDefense},
		{GoalTP, MissionPerformance},
		{GoalStab, MissionPerformance},
		{GoalSafe, MissionSafety},
	}
	for _, tt := range tests {
		if m := GoalMission(tt.goal); m != tt.mission {
			t.Errorf("GoalMission(%s) = %s, want %s", tt.goal, m, tt.mission)
		}
	}
}

func TestRRG_SentinelMissedDetection(t *testing.T) {
	cfg := DefaultRoleConfig()
	ctx := AgentRoleContext{
		FaultProbs:      map[uint64]float64{1: 0.9}, // high fault prob
		ReconfigTargets: map[uint64]int{1: 0},       // NOT evicting
	}
	rrg := RoleRewardGuide(ctx, cfg)
	if rrg[RoleSentinel] >= 0 {
		t.Fatalf("sentinel should be penalized for missed detection, got %f", rrg[RoleSentinel])
	}
	if rrg[RoleSentinel] != -cfg.RMiss {
		t.Fatalf("expected penalty -%f, got %f", cfg.RMiss, rrg[RoleSentinel])
	}
}

func TestRRG_SentinelNoFalseAlarm(t *testing.T) {
	cfg := DefaultRoleConfig()
	ctx := AgentRoleContext{
		FaultProbs:      map[uint64]float64{1: 0.3}, // below threshold
		ReconfigTargets: map[uint64]int{},
	}
	rrg := RoleRewardGuide(ctx, cfg)
	if rrg[RoleSentinel] != 0 {
		t.Fatalf("sentinel should have no penalty for low fault prob, got %f", rrg[RoleSentinel])
	}
}

func TestRRG_CommanderFalseEviction(t *testing.T) {
	cfg := DefaultRoleConfig()
	ctx := AgentRoleContext{
		FaultProbs:      map[uint64]float64{2: 0.5},
		ReconfigTargets: map[uint64]int{2: -1},     // evicting node 2
		ByzantineSet:    map[uint64]bool{2: false}, // node 2 is honest
	}
	rrg := RoleRewardGuide(ctx, cfg)
	if rrg[RoleCommander] >= 0 {
		t.Fatalf("commander should be penalized for false eviction, got %f", rrg[RoleCommander])
	}
}

func TestRRG_CommanderCorrectEviction(t *testing.T) {
	cfg := DefaultRoleConfig()
	ctx := AgentRoleContext{
		FaultProbs:      map[uint64]float64{3: 0.9},
		ReconfigTargets: map[uint64]int{3: -1},    // evicting node 3
		ByzantineSet:    map[uint64]bool{3: true}, // node 3 IS Byzantine
	}
	rrg := RoleRewardGuide(ctx, cfg)
	if rrg[RoleCommander] != 0 {
		t.Fatalf("commander should have no penalty for correct eviction, got %f", rrg[RoleCommander])
	}
}

func TestRRG_TunerChurnPenalty(t *testing.T) {
	cfg := DefaultRoleConfig()
	ctx := AgentRoleContext{
		ParamDelta: 0.5, // exceeds StabilityThreshold (0.3)
	}
	rrg := RoleRewardGuide(ctx, cfg)
	if rrg[RoleTuner] >= 0 {
		t.Fatalf("tuner should be penalized for churn, got %f", rrg[RoleTuner])
	}
}

func TestRRG_TunerStable(t *testing.T) {
	cfg := DefaultRoleConfig()
	ctx := AgentRoleContext{
		ParamDelta: 0.1, // below threshold
	}
	rrg := RoleRewardGuide(ctx, cfg)
	if rrg[RoleTuner] != 0 {
		t.Fatalf("tuner should have no penalty for stable params, got %f", rrg[RoleTuner])
	}
}

func TestGRG_DetectionSpeedBonus(t *testing.T) {
	cfg := DefaultRoleConfig()
	ctx := AgentRoleContext{
		DetectionEpoch: 3, // detected at epoch 3 (within κ_det=10)
	}
	grg := GoalRewardGuide(ctx, cfg)
	// bonus = r_b * (10-3)/10 = 1.0 * 7/10 = 0.7
	expected := cfg.RBonus * 7.0 / 10.0
	if math.Abs(grg[GoalDetect]-expected) > 1e-9 {
		t.Fatalf("expected g_det bonus %f, got %f", expected, grg[GoalDetect])
	}
}

func TestGRG_EvictionSpeedBonus(t *testing.T) {
	cfg := DefaultRoleConfig()
	ctx := AgentRoleContext{
		DetectionEpoch: 3,
		EvictionEpoch:  5, // evicted 2 epochs after detection (within τ=5)
	}
	grg := GoalRewardGuide(ctx, cfg)
	// bonus = r_b * (5-2)/5 = 0.6
	expected := cfg.RBonus * 3.0 / 5.0
	if math.Abs(grg[GoalEvict]-expected) > 1e-9 {
		t.Fatalf("expected g_evict bonus %f, got %f", expected, grg[GoalEvict])
	}
}

func TestGRG_StabilityBonus(t *testing.T) {
	cfg := DefaultRoleConfig()
	ctx := AgentRoleContext{
		StableEpochs: 6,
		WindowSize:   8,
	}
	grg := GoalRewardGuide(ctx, cfg)
	// bonus = r_b * 6/8 = 0.75
	expected := cfg.RBonus * 6.0 / 8.0
	if math.Abs(grg[GoalStab]-expected) > 1e-9 {
		t.Fatalf("expected g_stab bonus %f, got %f", expected, grg[GoalStab])
	}
}

func TestGRG_NoDetectionNoBonus(t *testing.T) {
	cfg := DefaultRoleConfig()
	ctx := AgentRoleContext{
		DetectionEpoch: -1, // not detected
		EvictionEpoch:  -1,
	}
	grg := GoalRewardGuide(ctx, cfg)
	if grg[GoalDetect] != 0 {
		t.Fatalf("no detection → no g_det bonus, got %f", grg[GoalDetect])
	}
	if grg[GoalEvict] != 0 {
		t.Fatalf("no eviction → no g_evict bonus, got %f", grg[GoalEvict])
	}
}

func TestOrganizationalReward_BaseCase(t *testing.T) {
	cfg := DefaultRoleConfig()
	cfg.Lambda5 = 0
	cfg.Lambda6 = 0

	ctx := AgentRoleContext{}
	total, _ := OrganizationalReward(5.0, ctx, cfg)
	// With λ₅=λ₆=0, r_org = r_base
	if total != 5.0 {
		t.Fatalf("with zero lambdas, r_org should equal r_base, got %f", total)
	}
}

func TestOrganizationalReward_WithPenaltiesAndBonuses(t *testing.T) {
	cfg := DefaultRoleConfig()
	ctx := AgentRoleContext{
		FaultProbs:      map[uint64]float64{1: 0.9},
		ReconfigTargets: map[uint64]int{1: 0}, // missed eviction
		ParamDelta:      0.5,                  // churn
		DetectionEpoch:  2,                    // fast detection
		StableEpochs:    8,
		WindowSize:      8,
	}
	total, details := OrganizationalReward(1.0, ctx, cfg)

	// Should have sentinel and tuner penalties, detection and stability bonuses
	if details["rrg_sentinel"] >= 0 {
		t.Fatal("expected sentinel penalty")
	}
	if details["rrg_tuner"] >= 0 {
		t.Fatal("expected tuner penalty")
	}
	if details["grg_g_det"] <= 0 {
		t.Fatal("expected detection bonus")
	}
	if details["grg_g_stab"] <= 0 {
		t.Fatal("expected stability bonus")
	}
	// total should differ from base
	if total == 1.0 {
		t.Fatal("r_org should differ from r_base with active penalties/bonuses")
	}
}

func TestConstraintHardness_Guardian(t *testing.T) {
	if ConstraintHardness(RoleGuardian) != 1.0 {
		t.Fatal("Guardian should have ch=1.0")
	}
	for _, r := range []RoleName{RoleSentinel, RoleCommander, RoleTuner} {
		if ConstraintHardness(r) != 0.0 {
			t.Fatalf("%s should have ch=0.0", r)
		}
	}
}

func TestRoleActionGuide_GuardianDelegatesToSafety(t *testing.T) {
	sf := DefaultSafetyFilter()
	obs := Observation{
		Agents: []AgentObservation{
			{InstanceID: 1, ValidatorCount: 4, CommitteeSize: 4, PacemakerTimeoutMs: 500},
		},
	}
	action := Action{
		AgentActions: []AgentAction{
			{InstanceID: 1, CommitteeSize: 3, ReconfigEvictNodeIDs: []uint64{99}}, // evict 1 → unsafe at n=4
		},
	}
	masked, anyMasked := RoleActionGuide(RoleGuardian, obs, action, &sf)
	if !anyMasked {
		t.Fatal("Guardian should mask unsafe eviction")
	}
	// Masked action should preserve current committee size from obs
	if masked.AgentActions[0].CommitteeSize != 4 {
		t.Fatalf("expected committee size preserved at 4, got %d", masked.AgentActions[0].CommitteeSize)
	}
	// Eviction list should be cleared
	if len(masked.AgentActions[0].ReconfigEvictNodeIDs) != 0 {
		t.Fatalf("expected evict list cleared, got %v", masked.AgentActions[0].ReconfigEvictNodeIDs)
	}
}

func TestRoleActionGuide_NonGuardianPassesThrough(t *testing.T) {
	sf := DefaultSafetyFilter()
	obs := Observation{}
	action := Action{CommitteeSize: 7}
	result, masked := RoleActionGuide(RoleSentinel, obs, action, &sf)
	if masked {
		t.Fatal("non-Guardian roles should not mask")
	}
	if result.CommitteeSize != 7 {
		t.Fatal("action should pass through unchanged")
	}
}

func TestGradientVarianceReduction_PositiveCov(t *testing.T) {
	// cov > 0, σ²_role > 0 → ratio < 1 (variance reduced)
	ratio := GradientVarianceReduction(0.1, 0.5, 1.0)
	if ratio >= 1.0 {
		t.Fatalf("positive covariance should reduce variance, got ratio %f", ratio)
	}
}

func TestGradientVarianceReduction_ZeroLambda(t *testing.T) {
	ratio := GradientVarianceReduction(0, 0.5, 1.0)
	if ratio != 1.0 {
		t.Fatalf("zero lambda should give ratio 1.0, got %f", ratio)
	}
}

func TestGradientVarianceReduction_NegativeCovIncreasesVariance(t *testing.T) {
	// Negative covariance → ratio > 1 (would increase variance)
	ratio := GradientVarianceReduction(0.1, -0.5, 1.0)
	if ratio <= 1.0 {
		t.Fatalf("negative covariance should not reduce variance, got ratio %f", ratio)
	}
}
