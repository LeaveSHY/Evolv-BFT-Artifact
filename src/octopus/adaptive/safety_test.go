package adaptive

import (
	"math"
	"testing"
)

// ═══════════════════════════════════════════════════════════════════════════════
// Tests for BFT Safety Filter and Lyapunov Safety Margin
// ═══════════════════════════════════════════════════════════════════════════════

func TestBFTQuorumThreshold(t *testing.T) {
	cases := []struct {
		faults   int
		expected int
	}{
		{0, 1},
		{1, 4},
		{2, 7},
		{3, 10},
	}
	for _, tc := range cases {
		got := BFTQuorumThreshold(tc.faults)
		if got != tc.expected {
			t.Errorf("BFTQuorumThreshold(%d) = %d, want %d", tc.faults, got, tc.expected)
		}
	}
}

func TestCheckQuorumInvariant_SafeEviction(t *testing.T) {
	sf := DefaultSafetyFilter()
	// 7 validators, evict 1 → 6 remaining, f=1, need 4 → safe
	state := InstanceState{ValidatorCount: 7, PendingEvicts: 1}
	if !sf.CheckQuorumInvariant(state) {
		t.Fatal("evicting 1 from 7 should be safe")
	}
}

func TestCheckQuorumInvariant_UnsafeEviction(t *testing.T) {
	sf := DefaultSafetyFilter() // DeltaS=1
	// With default FaultsEstimate=0 → f=1, threshold=3*1+1+1=5

	// 4 validators, evict 1 → 3 remaining
	// nAfter=3 < threshold=5 → BLOCKS (can't safely tolerate 1 fault with 3 nodes)
	state := InstanceState{ValidatorCount: 4, PendingEvicts: 1}
	if sf.CheckQuorumInvariant(state) {
		t.Fatal("4→3 should be unsafe (3 < 3*1+1+1=5, default f=1)")
	}

	// 4 validators, evict 3 → 1 remaining → unsafe
	state2 := InstanceState{ValidatorCount: 4, PendingEvicts: 3}
	if sf.CheckQuorumInvariant(state2) {
		t.Fatal("4→1 should be unsafe")
	}

	// 4 validators, evict 4 → 0 remaining → unsafe
	state3 := InstanceState{ValidatorCount: 4, PendingEvicts: 4}
	if sf.CheckQuorumInvariant(state3) {
		t.Fatal("4→0 should be unsafe")
	}
}

func TestCheckQuorumInvariant_MinimumViable(t *testing.T) {
	sf := DefaultSafetyFilter() // DeltaS=1, default f=1, threshold=5

	// No eviction (no-op) → always safe regardless of current size
	state := InstanceState{ValidatorCount: 4, PendingEvicts: 0}
	if !sf.CheckQuorumInvariant(state) {
		t.Fatal("4 validators with 0 evictions should be safe (no-op)")
	}

	// 7 validators, evict 3 → 4 remaining
	// nAfter=4 < threshold=5 → BLOCKS
	state2 := InstanceState{ValidatorCount: 7, PendingEvicts: 3}
	if sf.CheckQuorumInvariant(state2) {
		t.Fatal("7→4 should be unsafe with default f=1 (4 < 5)")
	}

	// 7 validators, evict 2 → 5 remaining
	// nAfter=5 >= threshold=5 → safe
	state2b := InstanceState{ValidatorCount: 7, PendingEvicts: 2}
	if !sf.CheckQuorumInvariant(state2b) {
		t.Fatal("7→5 should be safe (5 >= 5)")
	}

	// 7 validators, evict 4 → 3 remaining
	// nAfter=3 < threshold=5 → BLOCKS (monotonic: evicting more is always harder)
	state3 := InstanceState{ValidatorCount: 7, PendingEvicts: 4}
	if sf.CheckQuorumInvariant(state3) {
		t.Fatal("7→3 should be unsafe (monotonicity: more evictions = harder)")
	}
}

func TestCheckQuorumInvariant_Monotonicity(t *testing.T) {
	// Critical property: for fixed n and FaultsEstimate, evicting more nodes
	// can NEVER make the check pass when fewer evictions failed (no cliff effect).
	sf := DefaultSafetyFilter()

	for n := 4; n <= 20; n++ {
		for f := 1; f <= (n-1)/3; f++ {
			lastResult := true
			for evicts := 0; evicts <= n; evicts++ {
				state := InstanceState{
					ValidatorCount: n,
					PendingEvicts:  evicts,
					FaultsEstimate: f,
				}
				result := sf.CheckQuorumInvariant(state)
				if !lastResult && result {
					t.Fatalf("monotonicity violation: n=%d, f=%d, evict=%d passes but evict=%d failed",
						n, f, evicts, evicts-1)
				}
				lastResult = result
			}
		}
	}
}

func TestCheckQuorumInvariant_WithExplicitFaults(t *testing.T) {
	sf := DefaultSafetyFilter() // DeltaS=1

	// 10 validators, trust system identifies 2 Byzantine, evict 2
	// f=2, threshold=3*2+1+1=8, nAfter=8, 8>=8 → safe
	state := InstanceState{ValidatorCount: 10, PendingEvicts: 2, FaultsEstimate: 2}
	if !sf.CheckQuorumInvariant(state) {
		t.Fatal("10→8 with f=2 should be safe (8 >= 8)")
	}

	// 10 validators, trust system identifies 2, evict 3
	// f=2, threshold=8, nAfter=7, 7<8 → BLOCKS
	state2 := InstanceState{ValidatorCount: 10, PendingEvicts: 3, FaultsEstimate: 2}
	if sf.CheckQuorumInvariant(state2) {
		t.Fatal("10→7 with f=2 should be unsafe (7 < 8)")
	}

	// 7 validators, f=1, evict 2 → nAfter=5, threshold=5 → safe
	state3 := InstanceState{ValidatorCount: 7, PendingEvicts: 2, FaultsEstimate: 1}
	if !sf.CheckQuorumInvariant(state3) {
		t.Fatal("7→5 with f=1 should be safe (5 >= 5)")
	}
}

func TestCheckCoupledConstraint(t *testing.T) {
	sf := SafetyFilter{DeltaS: 1, GlobalBudgetMin: 16}

	// 4 instances with 6 validators each → 24 total → safe (no evictions)
	instances := []InstanceState{
		{ValidatorCount: 6}, {ValidatorCount: 6},
		{ValidatorCount: 6}, {ValidatorCount: 6},
	}
	if !sf.CheckCoupledConstraint(instances) {
		t.Fatal("24 total validators, no evictions, should be safe")
	}

	// One instance evicts 2: 6→4, f=0→1, threshold=5, 4<5 → BLOCKS
	instances[0] = InstanceState{ValidatorCount: 6, PendingEvicts: 2}
	if sf.CheckCoupledConstraint(instances) {
		t.Fatal("instance 6→4 should be unsafe (4 < 5 with default f=1)")
	}

	// One instance evicts 1: 6→5, f=0→1, threshold=5, 5>=5 → safe per-instance
	// Total: 5+6+6+6=23 >= 16 → coupled also safe
	instances[0] = InstanceState{ValidatorCount: 6, PendingEvicts: 1}
	if !sf.CheckCoupledConstraint(instances) {
		t.Fatal("instance 6→5 should be safe (5 >= 5, total 23 >= 16)")
	}
}

func TestMaskUnsafeAction_NoMaskingNeeded(t *testing.T) {
	sf := DefaultSafetyFilter()
	obs := Observation{
		ValidatorCount: 10,
		CommitteeSize:  7,
	}
	action := Action{
		CommitteeSize:      7,
		PacemakerTimeoutMs: 500,
	}

	masked, instances, anyMasked := sf.MaskUnsafeAction(obs, action)
	if anyMasked {
		t.Fatal("no masking should be needed for safe action")
	}
	if len(instances) != 0 {
		t.Fatalf("expected no masked instances, got %v", instances)
	}
	if masked.CommitteeSize != 7 {
		t.Fatalf("expected committee size 7, got %d", masked.CommitteeSize)
	}
}

func TestMaskUnsafeAction_BlocksLeaveWhenUnsafe(t *testing.T) {
	sf := SafetyFilter{GlobalBudgetMin: 4}
	obs := Observation{
		ValidatorCount: 4,
		CommitteeSize:  4,
	}
	action := Action{
		SubmitLeave: true, // trying to leave with only 4 validators
	}

	masked, _, anyMasked := sf.MaskUnsafeAction(obs, action)
	// n_after = 3 (evict 1), f_after = 0, threshold = 1. Actually safe!
	// But global budget check: 3 < 4 → coupled constraint fails → masked
	if !anyMasked {
		t.Fatal("expected masking when global budget would be violated")
	}
	if masked.SubmitLeave {
		t.Fatal("SubmitLeave should be blocked")
	}
}

func TestMaskUnsafeAction_BlocksUnsafePerInstanceReconfig(t *testing.T) {
	sf := DefaultSafetyFilter()
	obs := Observation{
		Agents: []AgentObservation{{
			InstanceID:                7,
			ValidatorCount:            4,
			CommitteeSize:             4,
			PacemakerTimeoutMs:        1000,
			MempoolMaxBatchTxs:        2048,
			MempoolProposalIntervalMs: 100,
		}},
	}
	action := Action{AgentActions: []AgentAction{{
		InstanceID:                7,
		CommitteeSize:             3,
		PacemakerTimeoutMs:        1500,
		MempoolMaxBatchTxs:        1024,
		MempoolProposalIntervalMs: 150,
		Reconfig:                  []int{-1},
		ReconfigEvictNodeIDs:      []uint64{42},
	}}}

	masked, instances, anyMasked := sf.MaskUnsafeAction(obs, action)
	if !anyMasked || len(instances) != 1 || instances[0] != 7 {
		t.Fatalf("expected instance 7 to be masked, got instances=%v any=%v", instances, anyMasked)
	}
	if len(masked.AgentActions) != 1 {
		t.Fatalf("expected one masked agent action, got %+v", masked.AgentActions)
	}
	aa := masked.AgentActions[0]
	if len(aa.Reconfig) != 0 || len(aa.ReconfigEvictNodeIDs) != 0 || len(aa.ReconfigAdmitNodeIDs) != 0 {
		t.Fatalf("expected unsafe reconfiguration intent cleared, got %+v", aa)
	}
	if aa.CommitteeSize != 4 || aa.PacemakerTimeoutMs != 1000 || aa.MempoolMaxBatchTxs != 2048 || aa.MempoolProposalIntervalMs != 100 {
		t.Fatalf("expected masked action to preserve observed tuning, got %+v", aa)
	}
}

func TestBuildInstanceStatesCountsSFACReconfig(t *testing.T) {
	sf := DefaultSafetyFilter()
	obs := Observation{Agents: []AgentObservation{{InstanceID: 3, ValidatorCount: 7, CommitteeSize: 4}}}
	action := Action{AgentActions: []AgentAction{{InstanceID: 3, Reconfig: []int{-1, 0, 1, -1}}}}
	states := sf.buildInstanceStates(obs, action)
	if len(states) != 1 {
		t.Fatalf("expected one state, got %+v", states)
	}
	if states[0].PendingEvicts != 2 || states[0].PendingAdmits != 1 {
		t.Fatalf("unexpected reconfig counts: %+v", states[0])
	}
}

func TestPerInstanceMargin(t *testing.T) {
	cases := []struct {
		validators int
		expected   float64
	}{
		{0, 0.0},
		{4, 0.0},       // 4 validators: f=1, threshold=4, surplus=0
		{5, 0.2},       // 5 validators: f=1, threshold=4, surplus=1, margin=1/5=0.2
		{7, 3.0 / 7.0}, // 7 validators: f=2, threshold=7, surplus=0... wait
		// f=(7-1)/3=2, threshold=3*2+1=7. surplus=0, margin=0.
		{7, 0.0},
		{8, 1.0 / 8.0},   // f=2, threshold=7, surplus=1, margin=1/8
		{10, 3.0 / 10.0}, // f=3, threshold=10, surplus=0, margin=0
		{10, 0.0},        // f=(10-1)/3=3, threshold=3*3+1=10, surplus=0
		{13, 3.0 / 13.0}, // f=4, threshold=13, surplus=0
	}
	// Fix expected values
	cases = []struct {
		validators int
		expected   float64
	}{
		{0, 0.0},
		{1, 0.0},       // f=0, threshold=1, surplus=0
		{4, 0.0},       // f=1, threshold=4, surplus=0
		{5, 0.2},       // f=1, threshold=4, surplus=1, margin=1/5
		{6, 2.0 / 6.0}, // f=1, threshold=4, surplus=2
		{7, 0.0},       // f=2, threshold=7, surplus=0
		{8, 1.0 / 8.0}, // f=2, threshold=7, surplus=1
	}
	for _, tc := range cases {
		got := PerInstanceMargin(tc.validators)
		if math.Abs(got-tc.expected) > 1e-9 {
			t.Errorf("PerInstanceMargin(%d) = %f, want %f", tc.validators, got, tc.expected)
		}
	}
}

func TestLyapunovDrift_EvictionIncreasesRisk(t *testing.T) {
	targetMargin := 0.25

	before := []InstanceState{
		{ValidatorCount: 6}, // margin = 2/6 = 0.333 > target → deficit=0
	}
	after := []InstanceState{
		{ValidatorCount: 6, PendingEvicts: 2}, // n_after=4, margin=0 → deficit=0.25
	}

	drift := LyapunovDrift(before, after, targetMargin)
	if drift <= 0 {
		t.Fatalf("eviction should increase Lyapunov value (positive drift), got %f", drift)
	}
}

func TestLyapunovDrift_AdmissionDecreasesRisk(t *testing.T) {
	targetMargin := 0.25

	before := []InstanceState{
		{ValidatorCount: 4}, // margin = 0 → deficit = 0.25
	}
	after := []InstanceState{
		{ValidatorCount: 4, PendingAdmits: 2}, // n_after=6, margin=2/6=0.333 → deficit=0
	}

	drift := LyapunovDrift(before, after, targetMargin)
	if drift >= 0 {
		t.Fatalf("admission should decrease Lyapunov value (negative drift), got %f", drift)
	}
}

func TestLyapunovDrift_NoChangeZeroDrift(t *testing.T) {
	targetMargin := 0.25
	states := []InstanceState{
		{ValidatorCount: 6},
	}
	drift := LyapunovDrift(states, states, targetMargin)
	if math.Abs(drift) > 1e-12 {
		t.Fatalf("no change should produce zero drift, got %f", drift)
	}
}

func TestSafetyMarginPenalty_SafeState(t *testing.T) {
	cfg := SafetyMarginConfig{TargetMargin: 0.25, Lambda: 1.0}
	// 6 validators: margin = 2/6 = 0.333 > target 0.25 → deficit = 0
	obs := Observation{ValidatorCount: 6}
	penalty := SafetyMarginPenalty(obs, cfg)
	if penalty > 1e-9 {
		t.Fatalf("safe state should have zero penalty, got %f", penalty)
	}
}

func TestSafetyMarginPenalty_AtBoundary(t *testing.T) {
	cfg := SafetyMarginConfig{TargetMargin: 0.25, Lambda: 1.0}
	// 4 validators: margin = 0 → deficit = 1.0
	obs := Observation{ValidatorCount: 4}
	penalty := SafetyMarginPenalty(obs, cfg)
	if penalty < 0.9 {
		t.Fatalf("at-boundary state should have large penalty, got %f", penalty)
	}
}

func TestSafetyMarginPenalty_MultiInstance(t *testing.T) {
	cfg := SafetyMarginConfig{TargetMargin: 0.25, Lambda: 0.5}
	obs := Observation{
		Agents: []AgentObservation{
			{InstanceID: 0, ValidatorCount: 4}, // margin=0, deficit=1.0
			{InstanceID: 1, ValidatorCount: 6}, // margin=0.333, deficit=0
			{InstanceID: 2, ValidatorCount: 4}, // margin=0, deficit=1.0
			{InstanceID: 3, ValidatorCount: 8}, // margin=0.125, deficit=0.5
		},
	}
	penalty := SafetyMarginPenalty(obs, cfg)
	// penalty = 0.5 * (1.0 + 0.0 + 1.0 + 0.5) = 0.5 * 2.5 = 1.25
	if math.Abs(penalty-1.25) > 0.01 {
		t.Fatalf("multi-instance penalty expected ~1.25, got %f", penalty)
	}
}

func TestSafetyMarginPenalty_DisabledLambdaZero(t *testing.T) {
	cfg := SafetyMarginConfig{TargetMargin: 0.25, Lambda: 0}
	obs := Observation{ValidatorCount: 4}
	penalty := SafetyMarginPenalty(obs, cfg)
	if penalty != 0 {
		t.Fatalf("zero lambda should produce zero penalty, got %f", penalty)
	}
}
