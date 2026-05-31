package adaptive

import (
	"testing"
)

// --- Controller SetSafetyFilter / HasLastDecision ---

func TestControllerSetSafetyFilter(t *testing.T) {
	ctrl := NewController(Config{}, nil, nil, SafeBaselinePolicy{}, DefaultGuardrails())
	sf := DefaultSafetyFilter()
	ctrl.SetSafetyFilter(&sf)
}

func TestControllerHasLastDecisionFalse(t *testing.T) {
	ctrl := NewController(Config{}, nil, nil, SafeBaselinePolicy{}, DefaultGuardrails())
	if ctrl.HasLastDecision() {
		t.Fatal("expected no last decision initially")
	}
}

// --- actionHasPayload ---

func TestActionHasPayloadEmpty(t *testing.T) {
	if actionHasPayload(Action{}) {
		t.Fatal("empty action should have no payload")
	}
}

func TestActionHasPayloadWithCommitteeSize(t *testing.T) {
	if !actionHasPayload(Action{CommitteeSize: 10}) {
		t.Fatal("action with committee size should have payload")
	}
}

func TestActionHasPayloadWithSubmitLeave(t *testing.T) {
	if !actionHasPayload(Action{SubmitLeave: true}) {
		t.Fatal("action with submit leave should have payload")
	}
}

func TestActionHasPayloadWithReason(t *testing.T) {
	if !actionHasPayload(Action{Reason: "test"}) {
		t.Fatal("action with reason should have payload")
	}
}

func TestActionHasPayloadWithAgentActions(t *testing.T) {
	if !actionHasPayload(Action{AgentActions: []AgentAction{{InstanceID: 1}}}) {
		t.Fatal("action with agent actions should have payload")
	}
}

// --- Policy Name() methods ---

func TestSafeBaselinePolicyName(t *testing.T) {
	p := SafeBaselinePolicy{}
	if p.Name() != "safe-baseline" {
		t.Fatalf("expected 'safe-baseline', got %q", p.Name())
	}
}

func TestScriptedPolicyName(t *testing.T) {
	p := NewScriptedPolicy("/tmp/script.json")
	if p.Name() != "scripted" {
		t.Fatalf("expected 'scripted', got %q", p.Name())
	}
}

// --- SafetyFilter: MinValidatorsForSafety ---

func TestMinValidatorsForSafetySmall(t *testing.T) {
	// With δ_s=1 (default), minimum for f=1 is 3*1+1+1=5
	if got := MinValidatorsForSafety(5, 1); got != 5 {
		t.Fatalf("expected 5, got %d", got)
	}
	if got := MinValidatorsForSafety(3, 1); got != 3 {
		t.Fatalf("expected 3, got %d", got)
	}
	if got := MinValidatorsForSafety(1, 1); got != 1 {
		t.Fatalf("expected 1, got %d", got)
	}
	// With δ_s=0, minimum for f=1 is 3*1+1+0=4
	if got := MinValidatorsForSafety(4, 0); got != 4 {
		t.Fatalf("expected 4, got %d", got)
	}
}

func TestMinValidatorsForSafetyLarge(t *testing.T) {
	// With δ_s=1, minimum is 4+1=5
	if got := MinValidatorsForSafety(10, 1); got != 5 {
		t.Fatalf("expected 5 for large set with δ_s=1, got %d", got)
	}
	// With δ_s=0, minimum is 4
	if got := MinValidatorsForSafety(10, 0); got != 4 {
		t.Fatalf("expected 4 for large set with δ_s=0, got %d", got)
	}
}

// --- DefaultSafetyMarginConfig ---

func TestDefaultSafetyMarginConfig(t *testing.T) {
	cfg := DefaultSafetyMarginConfig()
	if cfg.TargetMargin != 0.25 {
		t.Fatalf("expected target margin 0.25, got %f", cfg.TargetMargin)
	}
	if cfg.Lambda != 0.5 {
		t.Fatalf("expected lambda 0.5, got %f", cfg.Lambda)
	}
}

// --- SFACPolicy unit tests (no server needed for constructor and Name) ---

func TestNewSFACPolicy(t *testing.T) {
	p := NewSFACPolicy("http://localhost:8321", 0)
	if p == nil {
		t.Fatal("NewSFACPolicy returned nil")
	}
	if p.Name() != "sfac" {
		t.Fatalf("expected 'sfac', got %q", p.Name())
	}
}

func TestSFACPolicyDecideNoServer(t *testing.T) {
	p := NewSFACPolicy("http://localhost:99999", 100)
	obs := Observation{
		CommitteeSize:      10,
		PacemakerTimeoutMs: 500,
	}
	action := p.Decide(obs)
	// Should fall back gracefully (no server running)
	if action.Reason == "" {
		t.Fatal("expected fallback reason when no server available")
	}
}

func TestSFACPolicyBuildRequest(t *testing.T) {
	p := NewSFACPolicy("http://localhost:8321", 1000)
	obs := Observation{
		Agents: []AgentObservation{
			{InstanceID: 0, ValidatorCount: 10},
			{InstanceID: 1, ValidatorCount: 8},
		},
		ThroughputTPS: 1000.0,
		LatencyP95Ms:  50,
		TrustSnapshots: []TrustSnapshot{
			{NodeID: 1, TimeoutRate: 0.1, EquivocationRate: 0.0},
		},
	}
	req := p.buildSFACRequest(obs)
	if req.NumInstances != 2 {
		t.Fatalf("expected 2 instances, got %d", req.NumInstances)
	}
	if len(req.Instances) != 2 {
		t.Fatalf("expected 2 instance requests, got %d", len(req.Instances))
	}
}

func TestSFACPolicyConvertResponse(t *testing.T) {
	p := NewSFACPolicy("http://localhost:8321", 1000)
	obs := Observation{
		CommitteeSize:             10,
		PacemakerTimeoutMs:        500,
		MempoolMaxBatchTxs:        2048,
		MempoolProposalIntervalMs: 100,
		Agents: []AgentObservation{{
			InstanceID:                0,
			ValidatorCount:            7,
			CommitteeSize:             5,
			PacemakerTimeoutMs:        1000,
			MempoolMaxBatchTxs:        2048,
			MempoolProposalIntervalMs: 100,
		}},
		TrustSnapshots: []TrustSnapshot{
			{NodeID: 11},
			{NodeID: 12},
			{NodeID: 13},
		},
	}
	resp := SFACResponse{
		Actions: []SFACAgentAction{
			{InstanceID: 0, Reconfig: []int{-1, 0, 1}, Rotate: true, Params: []float64{6, 1500, 512, 75}},
		},
	}
	action := p.convertSFACResponse(resp, obs)
	if action.Reason != "sfac-policy" {
		t.Fatalf("expected 'sfac-policy' reason, got %q", action.Reason)
	}
	if action.SubmitLeave {
		t.Fatal("leader rotation must not be translated into local validator leave")
	}
	if len(action.AgentActions) != 1 {
		t.Fatalf("expected one agent action, got %+v", action.AgentActions)
	}
	aa := action.AgentActions[0]
	if !aa.RotateLeader || aa.CommitteeSize != 6 || aa.PacemakerTimeoutMs != 1500 || aa.MempoolMaxBatchTxs != 512 || aa.MempoolProposalIntervalMs != 75 {
		t.Fatalf("unexpected SFAC agent action: %+v", aa)
	}
	if len(aa.ReconfigEvictNodeIDs) != 1 || aa.ReconfigEvictNodeIDs[0] != 11 || len(aa.ReconfigAdmitNodeIDs) != 1 || aa.ReconfigAdmitNodeIDs[0] != 13 {
		t.Fatalf("unexpected SFAC reconfig mapping: %+v", aa)
	}
	if len(aa.ParamVector) != 4 || len(aa.Reconfig) != 3 {
		t.Fatalf("expected raw SFAC tuple retained for trace: %+v", aa)
	}
}
