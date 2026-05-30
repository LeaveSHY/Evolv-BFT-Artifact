package adaptive

import (
	"errors"
	"sync"
	"testing"
	"time"
)

var errTestTraceWrite = errors.New("trace-write")

type staticObserver struct {
	observation Observation
}

func (s staticObserver) Observe() Observation {
	return s.observation
}

type recordingActuator struct {
	mu      sync.Mutex
	applied []Action
}

func (r *recordingActuator) Apply(action Action) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.applied = append(r.applied, action)
	return nil
}

func (r *recordingActuator) Applied() []Action {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]Action(nil), r.applied...)
}

type recordingTraceWriter struct {
	mu      sync.Mutex
	samples []TrajectorySample
	err     error
	closed  int
}

func (r *recordingTraceWriter) Write(sample TrajectorySample) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.samples = append(r.samples, sample)
	return r.err
}

func (r *recordingTraceWriter) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.closed++
	return r.err
}

func (r *recordingTraceWriter) Samples() []TrajectorySample {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]TrajectorySample(nil), r.samples...)
}

func (r *recordingTraceWriter) Closed() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.closed
}

type staticPolicy struct {
	name   string
	action Action
}

func (s staticPolicy) Name() string {
	return s.name
}

func (s staticPolicy) Decide(observation Observation) Action {
	return s.action
}

func TestGuardrailsSanitizeClampsUnsafeAction(t *testing.T) {
	obs := Observation{
		ValidatorCount:            16,
		CommitteeSize:             0,
		PacemakerTimeoutMs:        1000,
		MempoolMaxBatchTxs:        2048,
		MempoolProposalIntervalMs: 500,
	}

	guard := DefaultGuardrails()
	raw := Action{
		CommitteeSize:             -9,
		PacemakerTimeoutMs:        5,
		MempoolMaxBatchTxs:        999999,
		MempoolProposalIntervalMs: 1,
	}

	got := guard.Sanitize(obs, raw)

	if got.CommitteeSize != 0 {
		t.Fatalf("committee size should clamp to 0 disable, got %d", got.CommitteeSize)
	}
	if got.PacemakerTimeoutMs != guard.MinPacemakerTimeoutMs {
		t.Fatalf("timeout should clamp to min %d, got %d", guard.MinPacemakerTimeoutMs, got.PacemakerTimeoutMs)
	}
	if got.MempoolMaxBatchTxs != guard.MaxMempoolBatchTxs {
		t.Fatalf("batch size should clamp to max %d, got %d", guard.MaxMempoolBatchTxs, got.MempoolMaxBatchTxs)
	}
	if got.MempoolProposalIntervalMs != guard.MinProposalIntervalMs {
		t.Fatalf("proposal interval should clamp to min %d, got %d", guard.MinProposalIntervalMs, got.MempoolProposalIntervalMs)
	}
}

func TestControllerTickAppliesSanitizedPolicyAction(t *testing.T) {
	obs := Observation{
		ValidatorCount:            8,
		CommitteeSize:             0,
		PacemakerTimeoutMs:        1000,
		MempoolMaxBatchTxs:        2048,
		MempoolProposalIntervalMs: 250,
	}
	actuator := &recordingActuator{}
	controller := NewController(
		Config{Enabled: true, Interval: 50 * time.Millisecond},
		staticObserver{observation: obs},
		actuator,
		staticPolicy{
			name: "test",
			action: Action{
				CommitteeSize:             3,
				PacemakerTimeoutMs:        200,
				MempoolMaxBatchTxs:        1024,
				MempoolProposalIntervalMs: 100,
			},
		},
		DefaultGuardrails(),
	)

	if err := controller.Tick(); err != nil {
		t.Fatalf("tick failed: %v", err)
	}
	applied := actuator.Applied()
	if len(applied) != 1 {
		t.Fatalf("expected 1 action applied, got %d", len(applied))
	}
	got := applied[0]
	if got.CommitteeSize != 4 {
		t.Fatalf("committee size should clamp to minimum safe value 4, got %d", got.CommitteeSize)
	}
	if controller.LastDecision().PolicyName != "test" {
		t.Fatalf("unexpected policy name: %q", controller.LastDecision().PolicyName)
	}
}

func TestGuardrailsSanitizeClampsPerAgentActions(t *testing.T) {
	obs := Observation{
		ValidatorCount: 8,
		Agents: []AgentObservation{
			{InstanceID: 0, ValidatorCount: 8},
			{InstanceID: 1, ValidatorCount: 6},
		},
	}
	guard := DefaultGuardrails()
	got := guard.Sanitize(obs, Action{
		AgentActions: []AgentAction{
			{InstanceID: 0, CommitteeSize: 2, PacemakerTimeoutMs: 100, MempoolMaxBatchTxs: 999999, MempoolProposalIntervalMs: 1},
			{InstanceID: 1, CommitteeSize: 10, PacemakerTimeoutMs: 99999, MempoolMaxBatchTxs: -1, MempoolProposalIntervalMs: 99999},
		},
	})
	if len(got.AgentActions) != 2 {
		t.Fatalf("expected 2 sanitized agent actions, got %d", len(got.AgentActions))
	}
	if got.AgentActions[0].CommitteeSize != 4 {
		t.Fatalf("unexpected agent0 committee size: %+v", got.AgentActions[0])
	}
	if got.AgentActions[1].CommitteeSize != 6 {
		t.Fatalf("unexpected agent1 committee size: %+v", got.AgentActions[1])
	}
	if got.AgentActions[0].MempoolMaxBatchTxs != guard.MaxMempoolBatchTxs {
		t.Fatalf("agent0 batch size not clamped")
	}
	if got.AgentActions[1].MempoolMaxBatchTxs != guard.MinMempoolBatchTxs {
		t.Fatalf("agent1 batch size not clamped")
	}
}

func TestControllerTickWritesDecisionStagesToTrace(t *testing.T) {
	obs := Observation{
		ValidatorCount:             8,
		GlobalConfirmedTotal:       5,
		GlobalConfirmedNil:         1,
		LastOrderedRank:            9,
		LastOrderedHeight:          4,
		LastOrderedLaneID:          0,
		LastOrderedConfigID:        0,
		LastOrderedTransitionCount: 2,
		LastReconfigEpoch:          3,
		Agents:                     []AgentObservation{{InstanceID: 1, ValidatorCount: 8}},
		TrustSnapshots:             []TrustSnapshot{{NodeID: 9, SampleCount: 2}},
	}
	actuator := &recordingActuator{}
	trace := &recordingTraceWriter{}
	controller := NewController(
		Config{Enabled: true, Interval: 50 * time.Millisecond},
		staticObserver{observation: obs},
		actuator,
		staticPolicy{
			name:   "test",
			action: Action{CommitteeSize: 3, PacemakerTimeoutMs: 200, Reason: "secret-candidate", AgentActions: []AgentAction{{InstanceID: 1, CommitteeSize: 3}}},
		},
		DefaultGuardrails(),
	)
	controller.SetTraceWriter(trace)
	controller.SetRewardModel(DefaultRewardModel())
	if err := controller.Tick(); err != nil {
		t.Fatalf("tick failed: %v", err)
	}
	samples := trace.Samples()
	if len(samples) != 1 {
		t.Fatalf("expected one trace sample, got %d", len(samples))
	}
	got := samples[0]
	if !got.GuardrailDelta {
		t.Fatalf("expected guardrail delta to be recorded, got %+v", got)
	}
	if got.Candidate.Action.CommitteeSize != 3 || got.Applied.Action.CommitteeSize != 4 {
		t.Fatalf("expected candidate/applied committee sizes to differ, got %+v", got)
	}
	if got.GovernanceDelta {
		t.Fatalf("did not expect governance delta in basic guardrail test, got %+v", got)
	}
	if got.Observation.GlobalConfirmedTotal != 5 || got.Observation.LastOrderedRank != 9 || got.Observation.LastOrderedLaneID != 0 || got.Observation.LastOrderedConfigID != 0 || got.Observation.LastReconfigEpoch != 3 {
		t.Fatalf("expected ordered/reconfig observation metadata to be preserved, got %+v", got.Observation)
	}
	if got.Observation.Agents != nil || got.Observation.TrustSnapshots != nil {
		t.Fatalf("expected trace observation to redact nested slices, got %+v", got.Observation)
	}
	if got.Candidate.Reason != "" || got.Candidate.Action.Reason != "" || got.Governed.Notes != nil || got.Candidate.Action.AgentActions != nil {
		t.Fatalf("expected trace stage details to be redacted, got %+v", got)
	}
	if !got.Trace.Enabled || got.Trace.WriteFailed || got.Trace.CloseFailed {
		t.Fatalf("expected successful trace status, got %+v", got.Trace)
	}
	if got.Provenance.PolicyName != "test" || got.Provenance.PolicyMode != "unknown" {
		t.Fatalf("expected runtime provenance policy identity, got %+v", got.Provenance)
	}
	if got.Provenance.SchemaVersion != SchemaVersion {
		t.Fatalf("expected provenance schema version, got %+v", got.Provenance)
	}
	if got.Provenance.TruthLevel != TraceTruthLevel {
		t.Fatalf("expected runtime trace truth level, got %+v", got.Provenance)
	}
	if got.Provenance.ClaimBoundary != TraceClaimBoundary {
		t.Fatalf("expected runtime trace claim boundary, got %+v", got.Provenance)
	}
}

func TestGovernanceSanitizeBlocksMembershipDuringElevation(t *testing.T) {
	obs := Observation{
		ValidatorCount:       4,
		CurrentConfigID:      1,
		HighestKnownConfigID: 2,
		LocalValidator:       false,
		CanParticipate:       true,
		RejectTotal:          1,
		AdversaryScore:       0.4,
	}
	governed, decision := DefaultGovernance().Sanitize(obs, Action{SubmitJoin: true, HydraDiscoveryTarget: 3})
	if !decision.FreezeMembership {
		t.Fatalf("expected membership freeze, got %+v", decision)
	}
	if governed.SubmitJoin {
		t.Fatalf("expected join to be blocked, got %+v", governed)
	}
	if governed.HydraDiscoveryTarget != 3 {
		t.Fatalf("expected discovery target to remain available, got %+v", governed)
	}
}

func TestControllerTickRecordsGovernedAction(t *testing.T) {
	obs := Observation{ValidatorCount: 4, LocalValidator: false, CanParticipate: true, RejectTotal: 1, AdversaryScore: 0.4}
	actuator := &recordingActuator{}
	trace := &recordingTraceWriter{}
	controller := NewController(
		Config{Enabled: true, Interval: 50 * time.Millisecond},
		staticObserver{observation: obs},
		actuator,
		staticPolicy{name: "test", action: Action{SubmitJoin: true, HydraDiscoveryTarget: 3}},
		DefaultGuardrails(),
	)
	controller.SetTraceWriter(trace)
	if err := controller.Tick(); err != nil {
		t.Fatalf("tick failed: %v", err)
	}
	got := trace.Samples()[0]
	if !got.GovernanceDelta {
		t.Fatalf("expected governance delta, got %+v", got)
	}
	if got.Candidate.Action.SubmitJoin == got.Governed.Action.SubmitJoin {
		t.Fatalf("expected governance to change membership action, got %+v", got)
	}
	if got.Governed.Action.HydraDiscoveryTarget != 3 {
		t.Fatalf("expected governance to preserve discovery target, got %+v", got)
	}
}

func TestGovernanceFreezeKeepsObservedLaneValues(t *testing.T) {
	obs := Observation{
		ValidatorCount:            8,
		CommitteeSize:             6,
		PacemakerTimeoutMs:        1400,
		MempoolMaxBatchTxs:        1024,
		MempoolProposalIntervalMs: 90,
		BacklogMissing:            12,
		AdversaryScore:            0.8,
	}
	governed, decision := DefaultGovernance().Sanitize(obs, Action{
		CommitteeSize:             4,
		PacemakerTimeoutMs:        600,
		MempoolMaxBatchTxs:        256,
		MempoolProposalIntervalMs: 20,
	})
	if !decision.FreezeLaneTuning {
		t.Fatalf("expected lane tuning freeze, got %+v", decision)
	}
	if governed.CommitteeSize != obs.CommitteeSize || governed.PacemakerTimeoutMs != obs.PacemakerTimeoutMs || governed.MempoolMaxBatchTxs != obs.MempoolMaxBatchTxs || governed.MempoolProposalIntervalMs != obs.MempoolProposalIntervalMs {
		t.Fatalf("expected governed values to hold observed settings, got %+v", governed)
	}
}

func TestLastDecisionReturnsDeepCopy(t *testing.T) {
	controller := NewController(
		Config{Enabled: true, Interval: 50 * time.Millisecond},
		staticObserver{observation: Observation{ValidatorCount: 8}},
		&recordingActuator{},
		staticPolicy{name: "test", action: Action{CommitteeSize: 4, Reason: "candidate"}},
		DefaultGuardrails(),
	)
	controller.last = Decision{
		PolicyName:  "test",
		Observation: Observation{Agents: []AgentObservation{{InstanceID: 1, ValidatorCount: 8}}, TrustSnapshots: []TrustSnapshot{{NodeID: 1, SampleCount: 3}}},
		Candidate:   DecisionActionStage{Action: Action{CommitteeSize: 4, AgentActions: []AgentAction{{InstanceID: 1, CommitteeSize: 4}}}},
		Governed:    DecisionActionStage{BlockedFields: []string{"committee_size"}},
		RoleRewards: map[string]float64{"lane_tuner": 1},
	}
	decision := controller.LastDecision()
	decision.Observation.Agents[0].ValidatorCount = 99
	decision.Observation.TrustSnapshots[0].SampleCount = 99
	decision.Candidate.Action.AgentActions[0].CommitteeSize = 99
	decision.Governed.BlockedFields = append(decision.Governed.BlockedFields, "forged")
	decision.RoleRewards["forged"] = 1

	fresh := controller.LastDecision()
	if fresh.Observation.Agents[0].ValidatorCount == 99 || fresh.Observation.TrustSnapshots[0].SampleCount == 99 {
		t.Fatalf("expected observation deep copy, got %+v", fresh.Observation)
	}
	if fresh.Candidate.Action.AgentActions[0].CommitteeSize == 99 {
		t.Fatalf("expected candidate action deep copy, got %+v", fresh.Candidate)
	}
	if len(fresh.Governed.BlockedFields) > 0 && fresh.Governed.BlockedFields[len(fresh.Governed.BlockedFields)-1] == "forged" {
		t.Fatalf("expected blocked fields deep copy, got %+v", fresh.Governed)
	}
	if _, ok := fresh.RoleRewards["forged"]; ok {
		t.Fatalf("expected role rewards deep copy, got %+v", fresh.RoleRewards)
	}
}

func TestLastDecisionReturnsRedactedDeepCopy(t *testing.T) {
	controller := NewController(
		Config{Enabled: true, Interval: 50 * time.Millisecond},
		staticObserver{observation: Observation{
			ValidatorCount: 8,
			Agents:         []AgentObservation{{InstanceID: 1, ValidatorCount: 8}},
			TrustSnapshots: []TrustSnapshot{{NodeID: 1, SampleCount: 3}},
		}},
		&recordingActuator{},
		staticPolicy{name: "test", action: Action{CommitteeSize: 4, AgentActions: []AgentAction{{InstanceID: 1, CommitteeSize: 4}}, Reason: "candidate"}},
		DefaultGuardrails(),
	)
	controller.SetRewardModel(DefaultRewardModel())
	if err := controller.Tick(); err != nil {
		t.Fatalf("tick failed: %v", err)
	}
	if controller.LastDecision().PolicyName != "test" {
		t.Fatalf("expected last decision to be populated, got %+v", controller.LastDecision())
	}
	decision := controller.LastDecision()
	if decision.RoleRewards == nil {
		t.Fatalf("expected role rewards in decision, got %+v", decision)
	}
	if decision.Provenance.PolicyName != "test" || decision.Provenance.PolicyMode != "unknown" {
		t.Fatalf("expected last decision provenance to preserve runtime policy identity, got %+v", decision.Provenance)
	}
	if decision.Provenance.TruthLevel != TraceTruthLevel || decision.Provenance.ClaimBoundary != TraceClaimBoundary {
		t.Fatalf("expected last decision provenance truth boundary, got %+v", decision.Provenance)
	}
	decision.Candidate.Action.CommitteeSize = 99
	decision.Governed.BlockedFields = append(decision.Governed.BlockedFields, "forged")
	decision.RoleRewards["forged"] = 1

	fresh := controller.LastDecision()
	if fresh.Observation.Agents != nil || fresh.Observation.TrustSnapshots != nil {
		t.Fatalf("expected last decision observation to be redacted, got %+v", fresh.Observation)
	}
	if fresh.Candidate.Action.CommitteeSize == 99 {
		t.Fatalf("expected candidate action deep copy, got %+v", fresh.Candidate)
	}
	if fresh.Candidate.Reason != "" || fresh.Candidate.Action.Reason != "" {
		t.Fatalf("expected last decision reasons to be redacted, got %+v", fresh.Candidate)
	}
	if len(fresh.Governed.BlockedFields) > 0 && fresh.Governed.BlockedFields[len(fresh.Governed.BlockedFields)-1] == "forged" {
		t.Fatalf("expected blocked fields deep copy, got %+v", fresh.Governed)
	}
	if _, ok := fresh.RoleRewards["forged"]; ok {
		t.Fatalf("expected role rewards deep copy, got %+v", fresh.RoleRewards)
	}
}

func TestControllerTickRedactsTraceExposureBoundary(t *testing.T) {
	trace := &recordingTraceWriter{}
	controller := NewController(
		Config{Enabled: true, Interval: 50 * time.Millisecond},
		staticObserver{observation: Observation{
			ValidatorCount: 8,
			Agents:         []AgentObservation{{InstanceID: 1, ValidatorCount: 8}},
			TrustSnapshots: []TrustSnapshot{{NodeID: 1, SampleCount: 3}},
		}},
		&recordingActuator{},
		staticPolicy{name: "test", action: Action{CommitteeSize: 4, AgentActions: []AgentAction{{InstanceID: 1, CommitteeSize: 4}}, Reason: "candidate-secret"}},
		DefaultGuardrails(),
	)
	controller.SetTraceWriter(trace)
	if err := controller.Tick(); err != nil {
		t.Fatalf("tick failed: %v", err)
	}
	got := trace.Samples()[0]
	if got.Candidate.Reason != "" || got.Governed.Reason != "" || got.Masked.Reason != "" || got.Applied.Reason != "" {
		t.Fatalf("expected reasons to be redacted from trace, got %+v", got)
	}
	if got.Observation.Agents != nil || got.Observation.TrustSnapshots != nil {
		t.Fatalf("expected trace observation details to be redacted, got %+v", got.Observation)
	}
	if len(got.Candidate.Action.AgentActions) != 0 || len(got.Applied.Action.AgentActions) != 0 {
		t.Fatalf("expected agent actions to be redacted from trace, got %+v", got)
	}
	if !got.Trace.Enabled {
		t.Fatalf("expected trace status enabled, got %+v", got.Trace)
	}
}

func TestControllerTickPreservesFacmacHTTPProvenanceAlias(t *testing.T) {
	trace := &recordingTraceWriter{}
	controller := NewController(
		Config{Enabled: true, Interval: 50 * time.Millisecond},
		staticObserver{observation: Observation{ValidatorCount: 8, CommitteeSize: 4, PacemakerTimeoutMs: 1000, MempoolMaxBatchTxs: 2048, MempoolProposalIntervalMs: 100}},
		&recordingActuator{},
		PolicyByName("facmac-http", "", ""),
		DefaultGuardrails(),
	)
	controller.SetTraceWriter(trace)
	if err := controller.Tick(); err != nil {
		t.Fatalf("tick failed: %v", err)
	}
	got := trace.Samples()[0]
	if got.Provenance.PolicyName != "facmac-http" || got.Provenance.PolicyMode != "facmac-http" {
		t.Fatalf("expected facmac-http alias to survive into provenance, got %+v", got.Provenance)
	}
}

func TestControllerTickTraceWriteFailureIsBounded(t *testing.T) {
	controller := NewController(
		Config{Enabled: true, Interval: 50 * time.Millisecond},
		staticObserver{observation: Observation{ValidatorCount: 8}},
		&recordingActuator{},
		staticPolicy{name: "test", action: Action{CommitteeSize: 4}},
		DefaultGuardrails(),
	)
	controller.SetTraceWriter(&recordingTraceWriter{err: errTestTraceWrite})
	if err := controller.Tick(); err != nil {
		t.Fatalf("expected bounded trace failure, got %v", err)
	}
	decision := controller.LastDecision()
	if !decision.Trace.Enabled || !decision.Trace.WriteFailed || decision.Trace.WriteError != "trace-write" {
		t.Fatalf("expected trace write failure to be recorded on decision, got %+v", decision.Trace)
	}
	if decision.Trace.DroppedSamples != 1 {
		t.Fatalf("expected dropped samples to increment, got %+v", decision.Trace)
	}
}

func TestControllerStartIsIdempotent(t *testing.T) {
	controller := NewController(
		Config{Enabled: true, Interval: 10 * time.Millisecond},
		staticObserver{observation: Observation{ValidatorCount: 8}},
		&recordingActuator{},
		staticPolicy{name: "test", action: Action{CommitteeSize: 4}},
		DefaultGuardrails(),
	)
	controller.Start()
	controller.Start()
	if !controller.started {
		t.Fatalf("expected controller to be marked started")
	}
	controller.Stop()
	if controller.started {
		t.Fatalf("expected controller to stop cleanly")
	}
}

func TestControllerCanRestartAfterStop(t *testing.T) {
	trace := &recordingTraceWriter{}
	controller := NewController(
		Config{Enabled: true, Interval: 10 * time.Millisecond},
		staticObserver{observation: Observation{ValidatorCount: 8}},
		&recordingActuator{},
		staticPolicy{name: "test", action: Action{CommitteeSize: 4}},
		DefaultGuardrails(),
	)
	controller.SetTraceWriter(trace)
	controller.Start()
	controller.Stop()
	controller.SetTraceWriter(trace)
	controller.Start()
	if !controller.started {
		t.Fatalf("expected controller to restart")
	}
	controller.Stop()
}

// --- Organizational Reward Integration Tests ---

// mockRoleContextProvider returns a fixed AgentRoleContext for testing.
type mockRoleContextProvider struct {
	ctx AgentRoleContext
}

func (m *mockRoleContextProvider) RoleContext(_ Observation, _ Action) AgentRoleContext {
	return m.ctx
}

func TestController_OrgReward_AugmentsBaseReward(t *testing.T) {
	obs := staticObserver{observation: Observation{
		ValidatorCount: 4,
		ThroughputTPS:  1000,
	}}
	act := &recordingActuator{}
	policy := staticPolicy{action: Action{CommitteeSize: 3}}
	ctrl := NewController(Config{Enabled: true}, obs, act, policy, DefaultGuardrails())
	ctrl.SetRewardModel(DefaultRewardModel())

	// First tick without org reward provider.
	if err := ctrl.Tick(); err != nil {
		t.Fatalf("tick without org reward: %v", err)
	}
	baseReward := ctrl.LastDecision().Reward

	// Now install org reward provider with penalties.
	provider := &mockRoleContextProvider{ctx: AgentRoleContext{
		FaultProbs:      map[uint64]float64{1: 0.9}, // sentinel: missed detection
		ReconfigTargets: map[uint64]int{},            // no eviction → rrg penalty
		ParamDelta:      0.5,                         // tuner: churn penalty (> 0.3 threshold)
		WindowSize:      10,
		StableEpochs:    8,
	}}
	cfg := DefaultRoleConfig()
	ctrl.SetRoleContextProvider(provider, cfg)

	// Second tick with org reward should differ from base.
	if err := ctrl.Tick(); err != nil {
		t.Fatalf("tick with org reward: %v", err)
	}
	orgReward := ctrl.LastDecision().Reward
	if orgReward == baseReward {
		t.Fatalf("expected org reward augmentation to change total reward, both = %f", baseReward)
	}
	// With penalties (sentinel miss + tuner churn), org reward should be lower.
	if orgReward >= baseReward {
		t.Fatalf("expected org reward (%f) < base reward (%f) due to rrg penalties", orgReward, baseReward)
	}
}

func TestController_OrgReward_NilProviderSkips(t *testing.T) {
	obs := staticObserver{observation: Observation{ValidatorCount: 4, ThroughputTPS: 500}}
	act := &recordingActuator{}
	policy := staticPolicy{action: Action{CommitteeSize: 3}}
	ctrl := NewController(Config{Enabled: true}, obs, act, policy, DefaultGuardrails())
	ctrl.SetRewardModel(DefaultRewardModel())

	// No org reward provider set → should not panic.
	if err := ctrl.Tick(); err != nil {
		t.Fatalf("tick: %v", err)
	}
	if ctrl.LastDecision().Reward == 0 {
		t.Fatalf("expected non-zero base reward even without org provider")
	}
}

func TestController_OrgReward_GoalBonusIncreasesReward(t *testing.T) {
	obs := staticObserver{observation: Observation{ValidatorCount: 4, ThroughputTPS: 1000}}
	act := &recordingActuator{}
	policy := staticPolicy{action: Action{CommitteeSize: 3}}
	ctrl := NewController(Config{Enabled: true}, obs, act, policy, DefaultGuardrails())
	ctrl.SetRewardModel(DefaultRewardModel())

	// Tick without org reward.
	if err := ctrl.Tick(); err != nil {
		t.Fatalf("tick: %v", err)
	}
	baseReward := ctrl.LastDecision().Reward

	// Install provider with goal bonuses (detection success, high stability) and no penalties.
	provider := &mockRoleContextProvider{ctx: AgentRoleContext{
		FaultProbs:      map[uint64]float64{},
		ReconfigTargets: map[uint64]int{},
		ParamDelta:      0.0,
		DetectionEpoch:  2, // fast detection
		EvictionEpoch:   4, // fast eviction
		WindowSize:      10,
		StableEpochs:    10,
	}}
	cfg := DefaultRoleConfig()
	ctrl.SetRoleContextProvider(provider, cfg)

	if err := ctrl.Tick(); err != nil {
		t.Fatalf("tick with bonuses: %v", err)
	}
	bonusReward := ctrl.LastDecision().Reward
	if bonusReward <= baseReward {
		t.Fatalf("expected bonus reward (%f) > base reward (%f) due to grg bonuses", bonusReward, baseReward)
	}
}

func TestStopRecordsTraceCloseErrorOnTraceStatus(t *testing.T) {
	controller := NewController(
		Config{Enabled: true, Interval: 50 * time.Millisecond},
		staticObserver{observation: Observation{ValidatorCount: 8}},
		&recordingActuator{},
		staticPolicy{name: "test", action: Action{CommitteeSize: 4, Reason: "candidate"}},
		DefaultGuardrails(),
	)
	if err := controller.Tick(); err != nil {
		t.Fatalf("tick failed: %v", err)
	}
	controller.SetTraceWriter(&recordingTraceWriter{err: errTestTraceWrite})
	controller.Start()
	controller.Stop()
	decision := controller.LastDecision()
	if !decision.Trace.Enabled || !decision.Trace.CloseFailed || decision.Trace.CloseError != "trace-write" {
		t.Fatalf("expected trace close error to be recorded, got %+v", decision.Trace)
	}
}

func TestStopClosesTraceOnce(t *testing.T) {
	trace := &recordingTraceWriter{}
	controller := NewController(
		Config{Enabled: true, Interval: 10 * time.Millisecond},
		staticObserver{observation: Observation{ValidatorCount: 8}},
		&recordingActuator{},
		staticPolicy{name: "test", action: Action{CommitteeSize: 4}},
		DefaultGuardrails(),
	)
	controller.SetTraceWriter(trace)
	controller.Start()
	controller.Stop()
	controller.Stop()
	if trace.Closed() != 1 {
		t.Fatalf("expected trace close exactly once, got %d", trace.Closed())
	}
}
