package adaptive

import "time"

const SchemaVersion = "octopus-adaptive-v1"

const (
	TraceTruthLevel               = "runtime_owned_trace"
	TraceClaimBoundary            = "authoritative Go runtime trace of candidate, governed, safety-masked, and applied adaptive actions; optimizer convergence remains delegated to the SFAC policy"
	AdminClaimBoundary            = "authoritative runtime adaptive status surface for sanitized SFAC actions; not trust-estimator authority or MOISE+ organization authority"
	TrustSnapshotBoundary         = "derived runtime evidence from LeaderReputation counters; not a first-class trust estimator, trust authority, or paper-grade probability model"
	OrganizationSemanticsAbsent   = "absent"
	OrganizationSemanticsBoundary = "organization-role semantics absent from the authoritative runtime adaptive surface; current governed outputs are not equivalent to MOISE+ role decomposition or organization authority"
)

type TrustSnapshot struct {
	// Snapshot of local leader outcome counters derived from LeaderReputation,
	// not a complete trust estimator output or paper-grade probability model.
	NodeID             uint64  `json:"node_id"`
	SampleCount        uint64  `json:"sample_count"`
	SuccessRate        float64 `json:"success_rate"`
	FailureProbability float64 `json:"failure_probability"`
	ClaimBoundary      string  `json:"claim_boundary"`

	// Trust feature vector (Eq. 5) — populated from LeaderReputation.TrustFeatureVector()
	TimeoutRate      float64 `json:"timeout_rate,omitempty"`      // d_t^k / W
	EquivocationRate float64 `json:"equivocation_rate,omitempty"` // e_t^k / W
	ViewChangeRate   float64 `json:"view_change_rate,omitempty"`  // v_t^k / W
	MeanLatency      float64 `json:"mean_latency,omitempty"`      // τ̄_t^k (normalized)
	StdLatency       float64 `json:"std_latency,omitempty"`       // σ_τ,t^k (normalized)
}

type Observation struct {
	Timestamp                  time.Time          `json:"timestamp"`
	NodeID                     uint64             `json:"node_id"`
	Epoch                      uint64             `json:"epoch"`
	ValidatorCount             int                `json:"validator_count"`
	CurrentConfigID            uint64             `json:"current_config_id"`
	HighestKnownConfigID       uint64             `json:"highest_known_config_id"`
	CommitteeSize              int                `json:"committee_size"`
	PacemakerTimeoutMs         int                `json:"pacemaker_timeout_ms"`
	MempoolMaxBatchTxs         int                `json:"mempool_max_batch_txs"`
	MempoolProposalIntervalMs  int                `json:"mempool_proposal_interval_ms"`
	ThroughputTPS              float64            `json:"throughput_tps"`
	LatencyP50Ms               float64            `json:"latency_p50_ms"`
	LatencyP95Ms               float64            `json:"latency_p95_ms"`
	LatencyP99Ms               float64            `json:"latency_p99_ms"`
	RecoveryP95Ms              float64            `json:"recovery_p95_ms"`
	BacklogPending             uint64             `json:"backlog_pending"`
	BacklogMissing             uint64             `json:"backlog_missing"`
	RejectTotal                uint64             `json:"reject_total"`
	ConnectedPeers             int                `json:"connected_peers"`
	KnownPeers                 int                `json:"known_peers"`
	PendingJoins               int                `json:"pending_joins"`
	PendingLeaves              int                `json:"pending_leaves"`
	LSetSize                   int                `json:"lset_size"`
	CanParticipate             bool               `json:"can_participate"`
	LocalValidator             bool               `json:"local_validator"`
	GlobalConfirmedTotal       uint64             `json:"global_confirmed_total"`
	GlobalConfirmedNil         uint64             `json:"global_confirmed_nil"`
	LastOrderedRank            uint64             `json:"last_ordered_rank"`
	LastOrderedHeight          uint64             `json:"last_ordered_height"`
	LastOrderedLaneID          uint64             `json:"last_ordered_lane_id"`
	LastOrderedConfigID        uint64             `json:"last_ordered_config_id"`
	LastOrderedNil             bool               `json:"last_ordered_nil"`
	LastOrderedTransitionCount int                `json:"last_ordered_transition_count"`
	LastReconfigEpoch          uint64             `json:"last_reconfig_epoch"`
	HeterogeneityScore         float64            `json:"heterogeneity_score"`
	ChurnRate                  float64            `json:"churn_rate"`
	AdversaryScore             float64            `json:"adversary_score"`
	NetworkJitterMs            float64            `json:"network_jitter_ms"`
	AILoadScore                float64            `json:"ai_load_score"`
	Agents                     []AgentObservation `json:"agents,omitempty"`
	TrustSnapshots             []TrustSnapshot    `json:"trust_snapshots,omitempty"`
}

type AgentObservation struct {
	InstanceID                uint64 `json:"instance_id"`
	Epoch                     uint64 `json:"epoch"`
	ValidatorCount            int    `json:"validator_count"`
	CommitteeSize             int    `json:"committee_size"`
	PacemakerTimeoutMs        int    `json:"pacemaker_timeout_ms"`
	MempoolMaxBatchTxs        int    `json:"mempool_max_batch_txs"`
	MempoolProposalIntervalMs int    `json:"mempool_proposal_interval_ms"`
	// FaultsEstimate: trust system's estimate of Byzantine count in this instance.
	// Used by SafetyFilter as f in "n_after >= 3f+1+δ_s". Defaults to 1 if unset.
	FaultsEstimate int `json:"faults_estimate,omitempty"`
}

type AgentAction struct {
	InstanceID                uint64    `json:"instance_id"`
	CommitteeSize             int       `json:"committee_size"`
	PacemakerTimeoutMs        int       `json:"pacemaker_timeout_ms"`
	MempoolMaxBatchTxs        int       `json:"mempool_max_batch_txs"`
	MempoolProposalIntervalMs int       `json:"mempool_proposal_interval_ms"`
	Reconfig                  []int     `json:"reconfig,omitempty"`
	ReconfigEvictNodeIDs      []uint64  `json:"reconfig_evict_node_ids,omitempty"`
	ReconfigAdmitNodeIDs      []uint64  `json:"reconfig_admit_node_ids,omitempty"`
	RotateLeader              bool      `json:"rotate_leader,omitempty"`
	ParamVector               []float64 `json:"param_vector,omitempty"`
}

type Action struct {
	CommitteeSize             int           `json:"committee_size"`
	PacemakerTimeoutMs        int           `json:"pacemaker_timeout_ms"`
	MempoolMaxBatchTxs        int           `json:"mempool_max_batch_txs"`
	MempoolProposalIntervalMs int           `json:"mempool_proposal_interval_ms"`
	SubmitJoin                bool          `json:"submit_join,omitempty"`
	SubmitLeave               bool          `json:"submit_leave,omitempty"`
	HydraDiscoveryTarget      int           `json:"hydra_discovery_target,omitempty"`
	Reason                    string        `json:"reason,omitempty"`
	AgentActions              []AgentAction `json:"agent_actions,omitempty"`
}

type TraceStatus struct {
	Enabled        bool   `json:"enabled"`
	WriteFailed    bool   `json:"write_failed,omitempty"`
	WriteError     string `json:"write_error,omitempty"`
	CloseFailed    bool   `json:"close_failed,omitempty"`
	CloseError     string `json:"close_error,omitempty"`
	DroppedSamples uint64 `json:"dropped_samples,omitempty"`
}

type TraceProvenance struct {
	PolicyName    string `json:"policy_name"`
	PolicyMode    string `json:"policy_mode"`
	SchemaVersion string `json:"schema_version"`
	TruthLevel    string `json:"truth_level"`
	ClaimBoundary string `json:"claim_boundary"`
}

type RewardSignal struct {
	Total       float64            `json:"total"`
	TeamReward  float64            `json:"team_reward"`
	RoleRewards map[string]float64 `json:"role_rewards,omitempty"`
}

type DecisionActionStage struct {
	Action        Action   `json:"action"`
	Present       bool     `json:"present"`
	Mutated       bool     `json:"mutated,omitempty"`
	Reason        string   `json:"reason,omitempty"`
	BlockedFields []string `json:"blocked_fields,omitempty"`
	Notes         []string `json:"notes,omitempty"`
}

type Decision struct {
	Timestamp   time.Time           `json:"timestamp"`
	PolicyName  string              `json:"policy_name"`
	Observation Observation         `json:"observation"`
	Candidate   DecisionActionStage `json:"candidate"`
	Governed    DecisionActionStage `json:"governed"`
	Masked      DecisionActionStage `json:"masked"`
	Applied     DecisionActionStage `json:"applied"`
	Reward      float64             `json:"reward"`
	TeamReward  float64             `json:"team_reward"`
	RoleRewards map[string]float64  `json:"role_rewards,omitempty"`
	Provenance  TraceProvenance     `json:"provenance"`
	Trace       TraceStatus         `json:"trace"`
}

type Config struct {
	Enabled  bool
	Interval time.Duration
	// WarmupEpochs: number of initial epochs during which the controller
	// outputs α_noop regardless of policy output (§III-B cold-start).
	// The safety mask remains active; warmup only suppresses learned actions
	// until the trust estimator accumulates sufficient evidence.
	// Default 0 means no warmup (backwards-compatible).
	WarmupEpochs uint64
}

type ScenarioContext struct {
	HeterogeneityScore float64 `json:"heterogeneity_score"`
	ChurnRate          float64 `json:"churn_rate"`
	AdversaryScore     float64 `json:"adversary_score"`
	NetworkJitterMs    float64 `json:"network_jitter_ms"`
	AILoadScore        float64 `json:"ai_load_score"`
}

type OrganizationSemantics struct {
	Status        string `json:"status"`
	ClaimBoundary string `json:"claim_boundary"`
}

type TrajectorySample struct {
	Timestamp       time.Time           `json:"timestamp"`
	PolicyName      string              `json:"policy_name"`
	Observation     Observation         `json:"observation"`
	Candidate       DecisionActionStage `json:"candidate"`
	Governed        DecisionActionStage `json:"governed"`
	Masked          DecisionActionStage `json:"masked"`
	Applied         DecisionActionStage `json:"applied"`
	GovernanceDelta bool                `json:"governance_delta"`
	GuardrailDelta  bool                `json:"guardrail_delta"`
	Reward          float64             `json:"reward"`
	PaperReward     float64             `json:"paper_reward"` // Eq.14 alignment metric
	TeamReward      float64             `json:"team_reward"`
	RoleRewards     map[string]float64  `json:"role_rewards,omitempty"`
	SchemaVersion   string              `json:"schema_version"`
	Provenance      TraceProvenance     `json:"provenance"`
	Trace           TraceStatus         `json:"trace"`
}

type GovernanceDecision struct {
	EscalationLevel  string   `json:"escalation_level"`
	FreezeMembership bool     `json:"freeze_membership"`
	FreezeLaneTuning bool     `json:"freeze_lane_tuning"`
	BlockedFields    []string `json:"blocked_fields,omitempty"`
	Notes            []string `json:"notes,omitempty"`
}

type Governance struct{}

type Guardrails struct {
	MinPacemakerTimeoutMs int
	MaxPacemakerTimeoutMs int
	MinMempoolBatchTxs    int
	MaxMempoolBatchTxs    int
	MinProposalIntervalMs int
	MaxProposalIntervalMs int
	MinCommitteeSize      int
}

func DefaultGovernance() Governance {
	return Governance{}
}

func DefaultGuardrails() Guardrails {
	return Guardrails{
		MinPacemakerTimeoutMs: 250,
		MaxPacemakerTimeoutMs: 5000,
		MinMempoolBatchTxs:    1,
		MaxMempoolBatchTxs:    8192,
		MinProposalIntervalMs: 10,
		MaxProposalIntervalMs: 2000,
		MinCommitteeSize:      4,
	}
}

func (g Guardrails) Sanitize(obs Observation, raw Action) Action {
	out := raw
	if out.CommitteeSize <= 0 {
		out.CommitteeSize = 0
	} else {
		if out.CommitteeSize < g.MinCommitteeSize {
			out.CommitteeSize = g.MinCommitteeSize
		}
		if obs.ValidatorCount > 0 && out.CommitteeSize > obs.ValidatorCount {
			out.CommitteeSize = obs.ValidatorCount
		}
	}
	out.PacemakerTimeoutMs = clampInt(out.PacemakerTimeoutMs, g.MinPacemakerTimeoutMs, g.MaxPacemakerTimeoutMs)
	out.MempoolMaxBatchTxs = clampInt(out.MempoolMaxBatchTxs, g.MinMempoolBatchTxs, g.MaxMempoolBatchTxs)
	out.MempoolProposalIntervalMs = clampInt(out.MempoolProposalIntervalMs, g.MinProposalIntervalMs, g.MaxProposalIntervalMs)
	if out.SubmitJoin && out.SubmitLeave {
		out.SubmitJoin = false
		out.SubmitLeave = false
	}
	if out.SubmitJoin && obs.LocalValidator {
		out.SubmitJoin = false
	}
	if out.SubmitLeave && (!obs.LocalValidator || obs.ValidatorCount <= 3) {
		out.SubmitLeave = false
	}
	if out.HydraDiscoveryTarget < 0 {
		out.HydraDiscoveryTarget = 0
	}
	if len(raw.AgentActions) > 0 {
		out.AgentActions = make([]AgentAction, 0, len(raw.AgentActions))
		agentValidators := make(map[uint64]int, len(obs.Agents))
		for _, agent := range obs.Agents {
			agentValidators[agent.InstanceID] = agent.ValidatorCount
		}
		for _, agentAction := range raw.AgentActions {
			sanitized := agentAction
			if sanitized.CommitteeSize <= 0 {
				sanitized.CommitteeSize = 0
			} else {
				if sanitized.CommitteeSize < g.MinCommitteeSize {
					sanitized.CommitteeSize = g.MinCommitteeSize
				}
				if validatorCount, ok := agentValidators[sanitized.InstanceID]; ok && validatorCount > 0 && sanitized.CommitteeSize > validatorCount {
					sanitized.CommitteeSize = validatorCount
				}
			}
			sanitized.PacemakerTimeoutMs = clampInt(sanitized.PacemakerTimeoutMs, g.MinPacemakerTimeoutMs, g.MaxPacemakerTimeoutMs)
			sanitized.MempoolMaxBatchTxs = clampInt(sanitized.MempoolMaxBatchTxs, g.MinMempoolBatchTxs, g.MaxMempoolBatchTxs)
			sanitized.MempoolProposalIntervalMs = clampInt(sanitized.MempoolProposalIntervalMs, g.MinProposalIntervalMs, g.MaxProposalIntervalMs)
			out.AgentActions = append(out.AgentActions, sanitized)
		}
	}
	return out
}

func (g Governance) Evaluate(obs Observation) GovernanceDecision {
	decision := GovernanceDecision{EscalationLevel: "nominal"}
	if obs.RejectTotal > 10 || obs.BacklogMissing > 10 || obs.AdversaryScore >= 0.7 {
		decision.EscalationLevel = "critical"
	} else if obs.RejectTotal > 0 || obs.BacklogMissing > 0 || obs.AdversaryScore >= 0.3 || obs.ChurnRate >= 0.4 {
		decision.EscalationLevel = "elevated"
	}
	if decision.EscalationLevel == "critical" || obs.BacklogMissing > 0 {
		decision.FreezeLaneTuning = true
		decision.Notes = append(decision.Notes, "lane-tuning-frozen")
	}
	if decision.EscalationLevel == "elevated" || decision.EscalationLevel == "critical" || obs.PendingJoins > 0 || obs.PendingLeaves > 0 {
		decision.FreezeMembership = true
		decision.Notes = append(decision.Notes, "membership-frozen")
	}
	if decision.FreezeMembership {
		decision.BlockedFields = append(decision.BlockedFields, "submit_join", "submit_leave")
		decision.BlockedFields = append(decision.BlockedFields, "agent_reconfig")
	}
	if decision.EscalationLevel == "critical" {
		decision.BlockedFields = append(decision.BlockedFields, "committee_size", "pacemaker_timeout_ms", "mempool_max_batch_txs", "mempool_proposal_interval_ms")
	}
	if !obs.CanParticipate {
		decision.BlockedFields = append(decision.BlockedFields, "submit_leave")
	}
	return decision
}

func (g Governance) Sanitize(obs Observation, raw Action) (Action, GovernanceDecision) {
	decision := g.Evaluate(obs)
	out := raw
	blocked := make(map[string]struct{}, len(decision.BlockedFields))
	for _, field := range decision.BlockedFields {
		blocked[field] = struct{}{}
	}
	if _, ok := blocked["committee_size"]; ok {
		out.CommitteeSize = obs.CommitteeSize
	}
	if _, ok := blocked["pacemaker_timeout_ms"]; ok {
		out.PacemakerTimeoutMs = obs.PacemakerTimeoutMs
	}
	if _, ok := blocked["mempool_max_batch_txs"]; ok {
		out.MempoolMaxBatchTxs = obs.MempoolMaxBatchTxs
	}
	if _, ok := blocked["mempool_proposal_interval_ms"]; ok {
		out.MempoolProposalIntervalMs = obs.MempoolProposalIntervalMs
	}
	if _, ok := blocked["submit_join"]; ok {
		out.SubmitJoin = false
	}
	if _, ok := blocked["submit_leave"]; ok {
		out.SubmitLeave = false
	}
	if decision.FreezeMembership {
		for idx := range out.AgentActions {
			out.AgentActions[idx].Reconfig = nil
			out.AgentActions[idx].ReconfigEvictNodeIDs = nil
			out.AgentActions[idx].ReconfigAdmitNodeIDs = nil
		}
	}
	if decision.FreezeLaneTuning {
		agentObserved := make(map[uint64]AgentObservation, len(obs.Agents))
		for _, agent := range obs.Agents {
			agentObserved[agent.InstanceID] = agent
		}
		for idx := range out.AgentActions {
			if agent, ok := agentObserved[out.AgentActions[idx].InstanceID]; ok {
				out.AgentActions[idx].CommitteeSize = agent.CommitteeSize
				out.AgentActions[idx].PacemakerTimeoutMs = agent.PacemakerTimeoutMs
				out.AgentActions[idx].MempoolMaxBatchTxs = agent.MempoolMaxBatchTxs
				out.AgentActions[idx].MempoolProposalIntervalMs = agent.MempoolProposalIntervalMs
				continue
			}
			out.AgentActions[idx].CommitteeSize = obs.CommitteeSize
			out.AgentActions[idx].PacemakerTimeoutMs = obs.PacemakerTimeoutMs
			out.AgentActions[idx].MempoolMaxBatchTxs = obs.MempoolMaxBatchTxs
			out.AgentActions[idx].MempoolProposalIntervalMs = obs.MempoolProposalIntervalMs
		}
	}
	if len(decision.Notes) > 0 {
		note := "governance:"
		for idx, item := range decision.Notes {
			if idx > 0 {
				note += ";"
			}
			note += item
		}
		if out.Reason != "" {
			out.Reason += " " + note
		} else {
			out.Reason = note
		}
	}
	return out, decision
}

func SchemaSnapshot() map[string]any {
	return map[string]any{
		"schema_version": SchemaVersion,
		"observation_fields": []string{
			"timestamp", "node_id", "epoch", "validator_count", "current_config_id", "highest_known_config_id",
			"committee_size", "pacemaker_timeout_ms", "mempool_max_batch_txs", "mempool_proposal_interval_ms",
			"throughput_tps", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "recovery_p95_ms",
			"backlog_pending", "backlog_missing", "reject_total", "connected_peers", "known_peers",
			"pending_joins", "pending_leaves", "lset_size", "can_participate", "local_validator",
			"global_confirmed_total", "global_confirmed_nil", "last_ordered_rank", "last_ordered_height", "last_ordered_nil", "last_ordered_transition_count", "last_reconfig_epoch",
			"heterogeneity_score", "churn_rate", "adversary_score", "network_jitter_ms", "ai_load_score", "agents", "trust_snapshots",
		},
		"agent_observation_fields": []string{
			"instance_id", "epoch", "validator_count", "committee_size", "pacemaker_timeout_ms", "mempool_max_batch_txs", "mempool_proposal_interval_ms",
		},
		"trust_snapshot_fields": []string{
			"node_id", "sample_count", "success_rate", "failure_probability", "claim_boundary",
		},
		"action_fields": []string{
			"committee_size", "pacemaker_timeout_ms", "mempool_max_batch_txs", "mempool_proposal_interval_ms",
			"submit_join", "submit_leave", "hydra_discovery_target", "reason", "agent_actions",
		},
		"agent_action_fields": []string{
			"instance_id", "committee_size", "pacemaker_timeout_ms", "mempool_max_batch_txs", "mempool_proposal_interval_ms", "reconfig", "reconfig_evict_node_ids", "reconfig_admit_node_ids", "rotate_leader", "param_vector",
		},
		"trace_fields": []string{
			"timestamp", "policy_name", "observation", "candidate", "governed", "masked", "applied", "governance_delta", "guardrail_delta", "reward", "team_reward", "role_rewards", "schema_version", "provenance", "trace",
		},
		"trace_provenance_fields": []string{
			"policy_name", "policy_mode", "schema_version", "truth_level", "claim_boundary",
		},
		"decision_fields": []string{
			"timestamp", "policy_name", "observation", "candidate", "governed", "masked", "applied", "reward", "team_reward", "role_rewards", "provenance", "trace",
		},
		"admin_response_fields": []string{
			"enabled", "schema_version", "schema", "has_last_decision", "last_decision", "claim_boundary", "organization_semantics", "context",
		},
		"organization_semantics_fields": []string{
			"status", "claim_boundary",
		},
		"decision_action_stage_fields": []string{
			"action", "present", "mutated", "reason", "blocked_fields", "notes",
		},
		"governance_fields": []string{
			"escalation_level", "freeze_membership", "freeze_lane_tuning", "blocked_fields", "notes",
		},
	}
}

func clampInt(v, minV, maxV int) int {
	if v < minV {
		return minV
	}
	if v > maxV {
		return maxV
	}
	return v
}
