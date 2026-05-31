package adaptive

import (
	"fmt"
	"reflect"
	"sync"
	"time"
)

const maxTraceErrorLen = 256

type Observer interface {
	Observe() Observation
}

type Actuator interface {
	Apply(action Action) error
}

// RoleContextProvider supplies per-epoch MOISE+ organizational context for the
// OrganizationalReward computation (Eq. reward-org). Training harnesses implement
// this to supply ground-truth data (ByzantineSet, detection/eviction timing).
// Production runtimes leave it nil (org reward is disabled).
type RoleContextProvider interface {
	RoleContext(obs Observation, action Action) AgentRoleContext
}

type Controller struct {
	mu       sync.RWMutex
	tickMu   sync.Mutex
	config   Config
	observer Observer
	actuator Actuator
	policy   Policy

	governance   Governance
	guardrail    Guardrails
	safetyFilter *SafetyFilter
	reward       RewardModel
	regret       *RegretTracker
	exploration  *ExplorationBonus
	lyapunov     *LyapunovMonitor
	advantage    *ValueBaseline
	epochGate    *EpochGate
	entropy      *EntropyMonitor
	roleCtx      RoleContextProvider
	roleConfig   RoleConfig
	trace        TraceWriter
	last         Decision
	stopCh       chan struct{}
	started      bool

	// Cold-start: counts ticks elapsed; when < config.WarmupEpochs the
	// controller forces α_noop (§III-B cold-start conservative phase).
	epochCounter uint64
}

func NewController(config Config, observer Observer, actuator Actuator, policy Policy, guardrail Guardrails) *Controller {
	if config.Interval <= 0 {
		config.Interval = time.Second
	}
	return &Controller{
		config:     config,
		observer:   observer,
		actuator:   actuator,
		policy:     policy,
		governance: DefaultGovernance(),
		guardrail:  guardrail,
		stopCh:     make(chan struct{}),
	}
}

func (c *Controller) SetRewardModel(model RewardModel) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.reward = model
}

// SetRegretTracker attaches a cumulative regret tracker (§IV, Theorem regret-bound).
// If set, the controller feeds PaperReward into the tracker every tick.
func (c *Controller) SetRegretTracker(tracker *RegretTracker) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.regret = tracker
}

// RegretSnapshot returns the current regret statistics, or nil if no tracker.
func (c *Controller) RegretSnapshot() *RegretSnapshot {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.regret == nil {
		return nil
	}
	snap := c.regret.Snapshot()
	return &snap
}

// SetExplorationBonus attaches a UCB exploration bonus tracker (Def.1 evidence-sensitivity).
func (c *Controller) SetExplorationBonus(eb *ExplorationBonus) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.exploration = eb
}

// SetLyapunovMonitor attaches a Lyapunov stability certificate tracker (§IV safety invariant).
func (c *Controller) SetLyapunovMonitor(lm *LyapunovMonitor) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.lyapunov = lm
}

// LyapunovStats returns current stability statistics, or nil if no monitor.
func (c *Controller) LyapunovStats() *LyapunovStats {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.lyapunov == nil {
		return nil
	}
	stats := c.lyapunov.Stats()
	return &stats
}

// SetValueBaseline attaches an advantage estimator (Actor-Critic variance reduction).
// When set, feedback reward is transformed to advantage = reward - V̂(s) before
// sending to Python SFAC, reducing policy gradient variance.
func (c *Controller) SetValueBaseline(vb *ValueBaseline) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.advantage = vb
}

// AdvantageStats returns current advantage estimation diagnostics, or nil.
func (c *Controller) AdvantageStats() *AdvantageStats {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.advantage == nil {
		return nil
	}
	stats := c.advantage.Stats()
	return &stats
}

// SetEpochGate attaches an epoch-boundary parameter gate (§III-C liveness preservation).
// When set, actions are deferred to epoch boundaries rather than applied mid-round.
func (c *Controller) SetEpochGate(gate *EpochGate) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.epochGate = gate
}

// EpochGateStats returns epoch gate diagnostics, or nil if no gate.
func (c *Controller) EpochGateStats() *EpochGateStats {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.epochGate == nil {
		return nil
	}
	stats := c.epochGate.Stats()
	return &stats
}

// AdvanceEpoch notifies the controller that a new epoch has started.
// If the epoch gate has a pending action ready to release, it is applied.
func (c *Controller) AdvanceEpoch(epoch uint64) error {
	c.mu.RLock()
	gate := c.epochGate
	c.mu.RUnlock()
	if gate == nil {
		return nil
	}
	released := gate.AdvanceEpoch(epoch, time.Now())
	if released != nil && c.actuator != nil {
		return c.actuator.Apply(*released)
	}
	return nil
}

// SetEntropyMonitor attaches a policy entropy tracker (Def.1 non-degeneracy).
func (c *Controller) SetEntropyMonitor(em *EntropyMonitor) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.entropy = em
}

// EntropyStats returns policy entropy diagnostics, or nil if no monitor.
func (c *Controller) EntropyStats() *EntropyStats {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.entropy == nil {
		return nil
	}
	stats := c.entropy.Stats()
	return &stats
}

func (c *Controller) SetSafetyFilter(filter *SafetyFilter) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.safetyFilter = filter
}

func (c *Controller) SetTraceWriter(writer TraceWriter) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.trace = writer
}

// SetRoleContextProvider installs the MOISE+ organizational reward path.
// When set, each Tick augments the base reward with OrganizationalReward (Eq. reward-org).
func (c *Controller) SetRoleContextProvider(provider RoleContextProvider, cfg RoleConfig) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.roleCtx = provider
	c.roleConfig = cfg
}

func (c *Controller) Start() {
	if !c.config.Enabled || c.policy == nil || c.observer == nil || c.actuator == nil {
		return
	}
	c.mu.Lock()
	if c.started {
		c.mu.Unlock()
		return
	}
	c.started = true
	c.stopCh = make(chan struct{})
	stopCh := c.stopCh
	interval := c.config.Interval
	c.mu.Unlock()
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				_ = c.Tick()
			case <-stopCh:
				return
			}
		}
	}()
}

func (c *Controller) Stop() {
	c.mu.Lock()
	if !c.started {
		c.mu.Unlock()
		return
	}
	stopCh := c.stopCh
	c.started = false
	trace := c.trace
	c.trace = nil
	c.mu.Unlock()
	close(stopCh)
	if trace != nil {
		if err := trace.Close(); err != nil {
			c.mu.Lock()
			c.last.Trace.Enabled = true
			c.last.Trace.CloseFailed = true
			c.last.Trace.CloseError = truncateTraceError(err)
			c.mu.Unlock()
		}
	}
}

func (c *Controller) Tick() error {
	c.tickMu.Lock()
	defer c.tickMu.Unlock()
	if c.policy == nil {
		return nil
	}
	if c.observer == nil || c.actuator == nil {
		return fmt.Errorf("controller missing observer or actuator")
	}
	observation := c.observer.Observe()

	// Cold-start phase (§III-B): output α_noop until WarmupEpochs ticks have
	// passed. This ensures the trust estimator accumulates sufficient evidence
	// before the learned policy influences reconfiguration decisions.
	c.epochCounter++
	var candidateAction Action
	if c.config.WarmupEpochs > 0 && c.epochCounter <= c.config.WarmupEpochs {
		candidateAction = Action{Reason: "cold-start-noop"}
	} else {
		candidateAction = c.policy.Decide(observation)
	}
	governedAction, governanceDecision := c.governance.Sanitize(observation, candidateAction)
	maskedAction := c.guardrail.Sanitize(observation, governedAction)

	// BFT safety filter (P3): pre-argmax mask enforcing n_v >= 3f_v + 1
	c.mu.RLock()
	sf := c.safetyFilter
	c.mu.RUnlock()
	safetyMasked := false
	var safetyMaskedInstances []uint64
	if sf != nil {
		maskedAction, safetyMaskedInstances, safetyMasked = sf.MaskUnsafeAction(observation, maskedAction)
	}

	// Epoch-boundary gating (§III-C liveness): defer parameter changes to safe
	// transition points. If gate defers, we still compute reward/feedback for
	// the intended action (policy learns from intent, not from deferred no-ops).
	c.mu.RLock()
	gate := c.epochGate
	c.mu.RUnlock()
	actionApplied := true
	if gate != nil {
		result := gate.Submit(maskedAction, safetyMasked, time.Now())
		if result == GateDeferred || result == GateDropped {
			actionApplied = false // deferred to next epoch boundary
		}
	}
	if actionApplied {
		if err := c.actuator.Apply(maskedAction); err != nil {
			return err
		}
	}
	c.mu.RLock()
	rewardModel := c.reward
	traceWriter := c.trace
	roleProvider := c.roleCtx
	roleCfg := c.roleConfig
	explBonus := c.exploration
	lyapMon := c.lyapunov
	advEst := c.advantage
	entMon := c.entropy
	c.mu.RUnlock()
	rewardSignal := RewardSignal{}
	if rewardModel != nil {
		rewardSignal = rewardModel.Compute(observation, maskedAction)
	}
	// Augment with MOISE+ organizational reward (Eq. reward-org) when training
	// harness supplies a RoleContextProvider with ground-truth data.
	if roleProvider != nil {
		roleCtx := roleProvider.RoleContext(observation, maskedAction)
		orgTotal, orgDetails := OrganizationalReward(rewardSignal.Total, roleCtx, roleCfg)
		rewardSignal.Total = orgTotal
		rewardSignal.TeamReward = orgTotal
		// Merge org details into RoleRewards for feedback to Python
		if rewardSignal.RoleRewards == nil {
			rewardSignal.RoleRewards = make(map[string]float64)
		}
		for k, v := range orgDetails {
			rewardSignal.RoleRewards[k] = v
		}
	}
	// Compute paper Eq.14 reward in parallel for alignment verification.
	// This does NOT drive control decisions — it is logged for auditability.
	deltaS := 1
	if sf != nil {
		deltaS = sf.DeltaS
	}
	paperReward := PaperReward(observation, deltaS, DefaultPaperRewardWeights())
	// Feed into regret tracker for O(√T) convergence verification (§IV).
	if c.regret != nil {
		c.regret.Observe(paperReward)
	}
	// Lyapunov stability certificate (§IV safety invariant).
	if lyapMon != nil {
		var normRegret float64
		if c.regret != nil {
			normRegret = c.regret.RegretPerSqrtT()
		}
		faults := make([]int, len(observation.Agents))
		sizes := make([]int, len(observation.Agents))
		for i, ag := range observation.Agents {
			faults[i] = ag.FaultsEstimate
			sizes[i] = ag.ValidatorCount
		}
		lyapMon.Evaluate(LyapunovState{
			InstanceFaults:   faults,
			InstanceSizes:    sizes,
			NormalizedRegret: normRegret,
		})
	}
	// Policy entropy tracking (Def.1 non-degeneracy: ρ_evol > 0).
	if entMon != nil {
		entMon.Observe(maskedAction)
	}
	now := time.Now()
	candidateStage := DecisionActionStage{
		Action:  candidateAction,
		Present: true,
		Reason:  candidateAction.Reason,
	}
	governedStage := DecisionActionStage{
		Action:        governedAction,
		Present:       true,
		Mutated:       !reflect.DeepEqual(candidateAction, governedAction),
		Reason:        governedAction.Reason,
		BlockedFields: append([]string(nil), governanceDecision.BlockedFields...),
		Notes:         append([]string(nil), governanceDecision.Notes...),
	}
	maskedStage := DecisionActionStage{
		Action:  maskedAction,
		Present: true,
		Mutated: !reflect.DeepEqual(governedAction, maskedAction),
		Reason:  maskedAction.Reason,
	}
	if safetyMasked {
		maskedStage.Notes = append(maskedStage.Notes, "bft-safety-filter-active")
		for _, iid := range safetyMaskedInstances {
			maskedStage.Notes = append(maskedStage.Notes, fmt.Sprintf("masked-instance-%d", iid))
		}
	}
	appliedStage := DecisionActionStage{
		Action:  maskedAction,
		Present: true,
		Reason:  maskedAction.Reason,
	}
	provenance := TraceProvenance{
		PolicyName:    c.policy.Name(),
		PolicyMode:    policyModeForName(c.policy.Name()),
		SchemaVersion: SchemaVersion,
		TruthLevel:    TraceTruthLevel,
		ClaimBoundary: TraceClaimBoundary,
	}
	decision := Decision{
		Timestamp:   now,
		PolicyName:  c.policy.Name(),
		Observation: redactObservation(observation),
		Candidate:   redactDecisionActionStage(candidateStage),
		Governed:    redactDecisionActionStage(governedStage),
		Masked:      redactDecisionActionStage(maskedStage),
		Applied:     redactDecisionActionStage(appliedStage),
		Reward:      rewardSignal.Total,
		TeamReward:  rewardSignal.TeamReward,
		RoleRewards: cloneRoleRewards(rewardSignal.RoleRewards),
		Provenance:  provenance,
		Trace: TraceStatus{
			Enabled: traceWriter != nil,
		},
	}
	c.mu.Lock()
	c.last = decision
	c.mu.Unlock()
	if traceWriter != nil {
		sample := redactTrajectorySample(TrajectorySample{
			Timestamp:       now,
			PolicyName:      c.policy.Name(),
			Observation:     observation,
			Candidate:       candidateStage,
			Governed:        governedStage,
			Masked:          maskedStage,
			Applied:         appliedStage,
			GovernanceDelta: governedStage.Mutated,
			GuardrailDelta:  maskedStage.Mutated,
			Reward:          rewardSignal.Total,
			PaperReward:     paperReward,
			TeamReward:      rewardSignal.TeamReward,
			RoleRewards:     rewardSignal.RoleRewards,
			SchemaVersion:   SchemaVersion,
			Provenance:      provenance,
			Trace: TraceStatus{
				Enabled: true,
			},
		})
		if err := traceWriter.Write(sample); err != nil {
			c.mu.Lock()
			c.last.Trace.Enabled = true
			c.last.Trace.WriteFailed = true
			c.last.Trace.WriteError = truncateTraceError(err)
			c.last.Trace.DroppedSamples++
			c.mu.Unlock()
			return nil
		}
	}

	// Send feedback to the Python server for online training (non-blocking).
	if fp, ok := c.policy.(FeedbackPolicy); ok {
		// UCB exploration bonus (Def.1 evidence-sensitivity): augment reward
		// for under-explored action regions to encourage parameter space coverage.
		feedbackReward := rewardSignal.Total
		if explBonus != nil {
			feedbackReward += explBonus.ObserveAndBonus(maskedAction)
		}
		// Advantage estimation (Actor-Critic): subtract EMA baseline V̂(s)
		// to reduce policy gradient variance. After warmup, feedbackReward
		// becomes A(s,a) = (R + bonus) - V̂, optionally normalized by std.
		if advEst != nil {
			feedbackReward = advEst.Advantage(feedbackReward)
		}
		feedbackSample := TrajectorySample{
			Timestamp:       now,
			PolicyName:      c.policy.Name(),
			Observation:     observation,
			Candidate:       candidateStage,
			Governed:        governedStage,
			Masked:          maskedStage,
			Applied:         appliedStage,
			GovernanceDelta: governedStage.Mutated,
			GuardrailDelta:  maskedStage.Mutated,
			Reward:          feedbackReward,
			PaperReward:     paperReward,
			TeamReward:      rewardSignal.TeamReward,
			RoleRewards:     rewardSignal.RoleRewards,
			SchemaVersion:   SchemaVersion,
			Provenance:      provenance,
		}
		go func() {
			_ = fp.Feedback(feedbackSample)
		}()
	}

	return nil
}

func (c *Controller) LastDecision() Decision {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return cloneDecision(c.last)
}

func (c *Controller) HasLastDecision() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return !c.last.Timestamp.IsZero()
}

func cloneDecision(in Decision) Decision {
	out := in
	out.Observation = cloneObservation(in.Observation)
	out.Candidate = cloneDecisionActionStage(in.Candidate)
	out.Governed = cloneDecisionActionStage(in.Governed)
	out.Masked = cloneDecisionActionStage(in.Masked)
	out.Applied = cloneDecisionActionStage(in.Applied)
	out.RoleRewards = cloneRoleRewards(in.RoleRewards)
	return out
}

func cloneObservation(in Observation) Observation {
	out := in
	out.Agents = append([]AgentObservation(nil), in.Agents...)
	out.TrustSnapshots = append([]TrustSnapshot(nil), in.TrustSnapshots...)
	return out
}

func cloneDecisionActionStage(in DecisionActionStage) DecisionActionStage {
	out := in
	out.Action = cloneAction(in.Action)
	out.BlockedFields = append([]string(nil), in.BlockedFields...)
	out.Notes = append([]string(nil), in.Notes...)
	return out
}

func cloneAction(in Action) Action {
	out := in
	out.AgentActions = cloneAgentActions(in.AgentActions)
	return out
}

func cloneAgentActions(in []AgentAction) []AgentAction {
	if in == nil {
		return nil
	}
	out := make([]AgentAction, len(in))
	for idx, action := range in {
		out[idx] = action
		out[idx].Reconfig = append([]int(nil), action.Reconfig...)
		out[idx].ReconfigEvictNodeIDs = append([]uint64(nil), action.ReconfigEvictNodeIDs...)
		out[idx].ReconfigAdmitNodeIDs = append([]uint64(nil), action.ReconfigAdmitNodeIDs...)
		out[idx].ParamVector = append([]float64(nil), action.ParamVector...)
	}
	return out
}

func cloneRoleRewards(in map[string]float64) map[string]float64 {
	if in == nil {
		return nil
	}
	out := make(map[string]float64, len(in))
	for key, value := range in {
		out[key] = value
	}
	return out
}

func truncateTraceError(err error) string {
	if err == nil {
		return ""
	}
	msg := err.Error()
	if len(msg) <= maxTraceErrorLen {
		return msg
	}
	return msg[:maxTraceErrorLen]
}

func redactTrajectorySample(in TrajectorySample) TrajectorySample {
	out := in
	out.Observation = redactObservation(in.Observation)
	out.Candidate = redactDecisionActionStage(in.Candidate)
	out.Governed = redactDecisionActionStage(in.Governed)
	out.Masked = redactDecisionActionStage(in.Masked)
	out.Applied = redactDecisionActionStage(in.Applied)
	out.RoleRewards = cloneRoleRewards(in.RoleRewards)
	return out
}

func redactObservation(in Observation) Observation {
	out := in
	out.Agents = nil
	out.TrustSnapshots = nil
	return out
}

func redactDecisionActionStage(in DecisionActionStage) DecisionActionStage {
	out := cloneDecisionActionStage(in)
	out.Action = redactAction(out.Action)
	out.Reason = ""
	out.Notes = nil
	return out
}

func redactAction(in Action) Action {
	out := cloneAction(in)
	out.Reason = ""
	out.AgentActions = nil
	return out
}

func actionHasPayload(action Action) bool {
	return action.CommitteeSize != 0 ||
		action.PacemakerTimeoutMs != 0 ||
		action.MempoolMaxBatchTxs != 0 ||
		action.MempoolProposalIntervalMs != 0 ||
		action.SubmitJoin ||
		action.SubmitLeave ||
		action.HydraDiscoveryTarget != 0 ||
		action.Reason != "" ||
		len(action.AgentActions) > 0
}
