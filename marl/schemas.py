from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Dict

SCHEMA_VERSION = "evolvbft-adaptive-v1"


def _ensure_schema_version(value: Any, path: str = "sample.schema_version", *, required: bool = False) -> str:
    if value is None:
        if required:
            raise ValueError(f"missing required field: {path}")
        return SCHEMA_VERSION
    if not isinstance(value, str) or not value:
        raise ValueError(f"expected non-empty string at {path}")
    if value != SCHEMA_VERSION:
        raise ValueError(f"unsupported trace schema_version: {value} != {SCHEMA_VERSION}")
    return value


def _ensure_known_fields(data: Dict[str, Any], allowed_fields: set[str], path: str) -> None:
    unknown_fields = sorted(set(data.keys()) - allowed_fields)
    if unknown_fields:
        raise ValueError(f"unknown field at {path}.{unknown_fields[0]}")


def _action_from_dict(data: Dict[str, Any], path: str) -> "Action":
    data = _ensure_object(data, path)
    _ensure_known_fields(data, set(Action.__dataclass_fields__), path)
    return Action(
        **{k: v for k, v in data.items() if k != "agent_actions"},
        agent_actions=[
            AgentAction.from_dict(item, path=f"{path}.agent_actions[{idx}]")
            for idx, item in enumerate(data.get("agent_actions", []))
        ],
    )


def _stage_from_dict(data: Dict[str, Any], path: str) -> "DecisionActionStage":
    data = _ensure_object(data, path)
    action_path = f"{path}.action"
    action_data = _ensure_object(data.get("action", {}), action_path)
    _ensure_known_fields(data, set(DecisionActionStage.__dataclass_fields__), path)
    _ensure_known_fields(action_data, set(Action.__dataclass_fields__), action_path)
    return DecisionActionStage(
        action=_action_from_dict(action_data, action_path),
        present=bool(data.get("present", False)),
        mutated=bool(data.get("mutated", False)),
        reason=data.get("reason"),
        blocked_fields=[str(item) for item in data.get("blocked_fields", [])],
        mutated_fields=[str(item) for item in data.get("mutated_fields", [])],
        affected_agent_ids=[int(item) for item in data.get("affected_agent_ids", [])],
        affected_agent_fields=[str(item) for item in data.get("affected_agent_fields", [])],
        notes=[str(item) for item in data.get("notes", [])],
        escalation_level=str(data.get("escalation_level", "") or ""),
        freeze_membership=bool(data.get("freeze_membership", False)),
        freeze_lane_tuning=bool(data.get("freeze_lane_tuning", False)),
        governance_summary=str(data.get("governance_summary", "") or ""),
        claim_safe_explanation=str(data.get("claim_safe_explanation", "") or ""),
    )


def _ensure_object(data: Any, path: str) -> Dict[str, Any]:
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be an object")
    return data


@dataclass
class AgentObservation:
    instance_id: int = 0
    epoch: int = 0
    validator_count: int = 0
    committee_size: int = 0
    pacemaker_timeout_ms: int = 1000
    mempool_max_batch_txs: int = 2048
    mempool_proposal_interval_ms: int = 100

    @classmethod
    def from_dict(cls, data: Dict[str, Any], path: str = "agent_observation") -> "AgentObservation":
        data = _ensure_object(data, path)
        _ensure_known_fields(data, set(cls.__dataclass_fields__), path)
        return cls(**data)


@dataclass
class AgentAction:
    instance_id: int = 0
    committee_size: int = 0
    pacemaker_timeout_ms: int = 1000
    mempool_max_batch_txs: int = 2048
    mempool_proposal_interval_ms: int = 100
    reconfig: list[int] = field(default_factory=list)
    reconfig_evict_node_ids: list[int] = field(default_factory=list)
    reconfig_admit_node_ids: list[int] = field(default_factory=list)
    rotate_leader: bool = False
    param_vector: list[float] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], path: str = "agent_action") -> "AgentAction":
        data = _ensure_object(data, path)
        _ensure_known_fields(data, set(cls.__dataclass_fields__), path)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "committee_size": self.committee_size,
            "pacemaker_timeout_ms": self.pacemaker_timeout_ms,
            "mempool_max_batch_txs": self.mempool_max_batch_txs,
            "mempool_proposal_interval_ms": self.mempool_proposal_interval_ms,
            "reconfig": list(self.reconfig),
            "reconfig_evict_node_ids": list(self.reconfig_evict_node_ids),
            "reconfig_admit_node_ids": list(self.reconfig_admit_node_ids),
            "rotate_leader": self.rotate_leader,
            "param_vector": list(self.param_vector),
        }


@dataclass
class TrustSnapshot:
    node_id: int = 0
    sample_count: int = 0
    success_rate: float = 0.0
    failure_probability: float = 0.0
    claim_boundary: str = ""
    # Trust feature vector (Eq. 5) — matches Go types.go TrustSnapshot
    timeout_rate: float = 0.0       # d_t^k / W
    equivocation_rate: float = 0.0  # e_t^k / W
    view_change_rate: float = 0.0   # v_t^k / W
    mean_latency: float = 0.0       # τ̄_t^k (normalized)
    std_latency: float = 0.0        # σ_τ,t^k (normalized)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], path: str = "trust_snapshot") -> "TrustSnapshot":
        data = _ensure_object(data, path)
        _ensure_known_fields(data, set(cls.__dataclass_fields__), path)
        return cls(**data)


@dataclass
class Observation:
    timestamp: str | None = None
    node_id: int = 0
    epoch: int = 0
    validator_count: int = 0
    current_config_id: int = 0
    highest_known_config_id: int = 0
    committee_size: int = 0
    pacemaker_timeout_ms: int = 1000
    mempool_max_batch_txs: int = 2048
    mempool_proposal_interval_ms: int = 100
    throughput_tps: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    recovery_p95_ms: float = 0.0
    backlog_pending: int = 0
    backlog_missing: int = 0
    reject_total: int = 0
    connected_peers: int = 0
    known_peers: int = 0
    pending_joins: int = 0
    pending_leaves: int = 0
    lset_size: int = 0
    can_participate: bool = True
    local_validator: bool = True
    global_confirmed_total: int = 0
    global_confirmed_nil: int = 0
    last_ordered_rank: int = 0
    last_ordered_height: int = 0
    last_ordered_lane_id: int = 0
    last_ordered_config_id: int = 0
    last_ordered_nil: bool = False
    last_ordered_transition_count: int = 0
    last_reconfig_epoch: int = 0
    heterogeneity_score: float = 0.0
    churn_rate: float = 0.0
    adversary_score: float = 0.0
    network_jitter_ms: float = 0.0
    ai_load_score: float = 0.0
    agents: list[AgentObservation] = field(default_factory=list)
    trust_snapshots: list[TrustSnapshot] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], path: str = "observation") -> "Observation":
        data = _ensure_object(data, path)
        _ensure_known_fields(data, set(cls.__dataclass_fields__), path)
        filtered = {k: v for k, v in data.items() if k not in {"agents", "trust_snapshots"}}
        filtered["agents"] = [AgentObservation.from_dict(item, path=f"{path}.agents[{idx}]") for idx, item in enumerate(data.get("agents", []))]
        filtered["trust_snapshots"] = [TrustSnapshot.from_dict(item, path=f"{path}.trust_snapshots[{idx}]") for idx, item in enumerate(data.get("trust_snapshots", []))]
        return cls(**filtered)


@dataclass
class Action:
    committee_size: int = 0
    pacemaker_timeout_ms: int = 1000
    mempool_max_batch_txs: int = 2048
    mempool_proposal_interval_ms: int = 100
    submit_join: bool = False
    submit_leave: bool = False
    hydra_discovery_target: int = 0
    reason: str | None = None
    agent_actions: list[AgentAction] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], path: str = "action") -> "Action":
        return _action_from_dict(data, path)

    def __post_init__(self) -> None:
        normalized = []
        for idx, agent in enumerate(self.agent_actions):
            if isinstance(agent, AgentAction):
                normalized.append(agent)
            else:
                normalized.append(AgentAction.from_dict(agent, path=f"action.agent_actions[{idx}]"))
        self.agent_actions = normalized

    def to_dict(self) -> Dict[str, Any]:
        return {
            "committee_size": self.committee_size,
            "pacemaker_timeout_ms": self.pacemaker_timeout_ms,
            "mempool_max_batch_txs": self.mempool_max_batch_txs,
            "mempool_proposal_interval_ms": self.mempool_proposal_interval_ms,
            "submit_join": self.submit_join,
            "submit_leave": self.submit_leave,
            "hydra_discovery_target": self.hydra_discovery_target,
            "reason": self.reason,
            "agent_actions": [agent.to_dict() for agent in self.agent_actions],
        }


def _action_has_payload(action: Action) -> bool:
    return any(
        [
            action.committee_size != 0,
            action.pacemaker_timeout_ms != 1000,
            action.mempool_max_batch_txs != 2048,
            action.mempool_proposal_interval_ms != 100,
            action.submit_join,
            action.submit_leave,
            action.hydra_discovery_target != 0,
            bool(action.agent_actions),
        ]
    )


@dataclass
class DecisionActionStage:
    action: Action = field(default_factory=Action)
    present: bool = False
    mutated: bool = False
    reason: str | None = None
    blocked_fields: list[str] = field(default_factory=list)
    mutated_fields: list[str] = field(default_factory=list)
    affected_agent_ids: list[int] = field(default_factory=list)
    affected_agent_fields: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    escalation_level: str = ""
    freeze_membership: bool = False
    freeze_lane_tuning: bool = False
    governance_summary: str = ""
    claim_safe_explanation: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionActionStage":
        return _stage_from_dict(data, "stage")


@dataclass
class TraceStatus:
    enabled: bool = False
    write_failed: bool = False
    write_error: str | None = None
    close_failed: bool = False
    close_error: str | None = None
    dropped_samples: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any], path: str = "trace") -> "TraceStatus":
        data = _ensure_object(data, path)
        _ensure_known_fields(data, set(cls.__dataclass_fields__), path)
        return cls(
            enabled=bool(data.get("enabled", False)),
            write_failed=bool(data.get("write_failed", False)),
            write_error=data.get("write_error") if isinstance(data.get("write_error"), str) else None,
            close_failed=bool(data.get("close_failed", False)),
            close_error=data.get("close_error") if isinstance(data.get("close_error"), str) else None,
            dropped_samples=int(data.get("dropped_samples", 0) or 0),
        )


@dataclass
class TrajectorySample:
    observation: Observation
    candidate: DecisionActionStage = field(default_factory=DecisionActionStage)
    governed: DecisionActionStage = field(default_factory=DecisionActionStage)
    masked: DecisionActionStage = field(default_factory=DecisionActionStage)
    applied: DecisionActionStage = field(default_factory=DecisionActionStage)
    reward: float = 0.0
    next_observation: Observation | None = None
    done: bool = False
    team_reward: float | None = None
    role_rewards: Dict[str, float] = field(default_factory=dict)
    governance_delta: bool = False
    guardrail_delta: bool = False
    schema_version: str = SCHEMA_VERSION
    policy_name: str = "unknown"
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    trace: TraceStatus = field(default_factory=TraceStatus)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectorySample":
        data = _ensure_object(data, "sample")
        _ensure_known_fields(data, set(cls.__dataclass_fields__), "sample")
        raw_ts = data.get("timestamp")
        ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00")) if raw_ts else datetime.now(UTC)
        return cls(
            timestamp=ts,
            policy_name=data.get("policy_name", "unknown"),
            observation=Observation.from_dict(data.get("observation", {}), path="observation"),
            next_observation=Observation.from_dict(data.get("next_observation", {}), path="next_observation") if data.get("next_observation") else None,
            done=bool(data.get("done", False)),
            team_reward=float(data["team_reward"]) if data.get("team_reward") is not None else None,
            role_rewards={str(k): float(v) for k, v in data.get("role_rewards", {}).items()},
            candidate=_stage_from_dict(data.get("candidate", {}), "candidate"),
            governed=_stage_from_dict(data.get("governed", {}), "governed"),
            masked=_stage_from_dict(data.get("masked", {}), "masked"),
            applied=_stage_from_dict(data.get("applied", {}), "applied"),
            reward=float(data.get("reward", 0.0)),
            governance_delta=bool(data.get("governance_delta", False)),
            guardrail_delta=bool(data.get("guardrail_delta", False)),
            schema_version=_ensure_schema_version(data.get("schema_version")),
            trace=TraceStatus.from_dict(data.get("trace", {}), path="trace"),
        )


def runtime_trace_sample_from_dict(data: Dict[str, Any]) -> TrajectorySample:
    return TrajectorySample.from_dict(data)


def schema_snapshot() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "observation_fields": list(Observation.__dataclass_fields__.keys()),
        "agent_observation_fields": list(AgentObservation.__dataclass_fields__.keys()),
        "trust_snapshot_fields": list(TrustSnapshot.__dataclass_fields__.keys()),
        "action_fields": list(Action.__dataclass_fields__.keys()),
        "agent_action_fields": list(AgentAction.__dataclass_fields__.keys()),
        "decision_action_stage_fields": list(DecisionActionStage.__dataclass_fields__.keys()),
        "decision_stages": ["candidate", "governed", "masked", "applied"],
        "decision_fields": [
            "observation",
            "candidate",
            "governed",
            "masked",
            "applied",
            "reward",
            "next_observation",
            "done",
            "team_reward",
            "role_rewards",
            "governance_delta",
            "guardrail_delta",
            "schema_version",
            "policy_name",
            "timestamp",
            "trace",
        ],
        "replay_summary_fields": [
            "replay_size",
            "governance_delta_rate",
            "guardrail_delta_rate",
            "divergence_rates",
            "reward_summary",
            "priority_summary",
        ],
        "training_summary_fields": [
            "last_training",
            "last_activation",
            "checkpoint",
            "model_ready",
            "active_model_source",
        ],
        "artifact_inventory_fields": [
            "schema_version",
            "active_model_source",
            "model_ready",
            "replay_ready",
            "replay_size",
            "checkpoint",
            "checkpoint_metadata",
            "training_metadata",
            "has_training_summary",
            "has_activation_summary",
            "claim_boundary",
            "trainer_family",
            "paper_grade_facmac",
            "target_action",
        ],
    }
