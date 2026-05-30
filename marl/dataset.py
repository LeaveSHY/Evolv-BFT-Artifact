from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None

from marl.schemas import Action, AgentAction, AgentObservation, Observation, SCHEMA_VERSION, TrajectorySample, runtime_trace_sample_from_dict

ACTION_DIM = 7
AGENT_ACTION_DIM = 4
AGENT_FEATURE_DIM = 7


FEATURE_ORDER = [
    "epoch",
    "validator_count",
    "current_config_id",
    "highest_known_config_id",
    "committee_size",
    "pacemaker_timeout_ms",
    "mempool_max_batch_txs",
    "mempool_proposal_interval_ms",
    "throughput_tps",
    "latency_p95_ms",
    "backlog_pending",
    "backlog_missing",
    "reject_total",
    "pending_joins",
    "pending_leaves",
    "lset_size",
    "can_participate",
    "local_validator",
    "heterogeneity_score",
    "churn_rate",
    "adversary_score",
    "network_jitter_ms",
    "ai_load_score",
    "agent_count",
    "agent_mean_committee_size",
    "agent_mean_timeout_ms",
    "agent_mean_batch_txs",
    "agent_mean_proposal_interval_ms",
]


@dataclass
class TrainingBatch:
    features: Any
    next_features: Any
    actions: Any
    candidate_actions: Any
    governed_actions: Any
    rewards: Any
    team_rewards: Any
    dones: Any
    governance_deltas: Any
    guardrail_deltas: Any
    agent_features: Any
    agent_actions: Any


class SimpleVector(list):
    @property
    def shape(self):
        return (len(self),)

    @property
    def ndim(self):
        return 1


class SimpleMatrix(list):
    def __init__(self, rows: list[list[float]], cols: int | None = None):
        super().__init__([list(row) for row in rows])
        self._cols = cols if cols is not None else (len(self[0]) if self else 0)

    @property
    def shape(self):
        return (len(self), self._cols)

    @property
    def ndim(self):
        return 2


def _as_vector(values: list[float]):
    if np is not None:
        return np.asarray(values, dtype=np.float64)
    return SimpleVector(float(v) for v in values)


def _as_matrix(rows: list, cols: int | None = None):
    if np is not None:
        width = cols if cols is not None else (len(rows[0]) if rows else 0)
        if not rows:
            return np.zeros((0, width), dtype=np.float64)
        return np.vstack(rows)
    return SimpleMatrix(rows, cols=cols)


def _zeros(rows: int, cols: int):
    if np is not None:
        return np.zeros((rows, cols), dtype=np.float64)
    return SimpleMatrix([[0.0 for _ in range(cols)] for _ in range(rows)], cols=cols)


def encode_observation(observation: Observation):
    agent_count = len(observation.agents)
    if agent_count > 0:
        mean_committee = sum(agent.committee_size for agent in observation.agents) / agent_count
        mean_timeout = sum(agent.pacemaker_timeout_ms for agent in observation.agents) / agent_count
        mean_batch = sum(agent.mempool_max_batch_txs for agent in observation.agents) / agent_count
        mean_interval = sum(agent.mempool_proposal_interval_ms for agent in observation.agents) / agent_count
    else:
        mean_committee = 0.0
        mean_timeout = 0.0
        mean_batch = 0.0
        mean_interval = 0.0

    values = {
        "epoch": observation.epoch,
        "validator_count": observation.validator_count,
        "current_config_id": observation.current_config_id,
        "highest_known_config_id": observation.highest_known_config_id,
        "committee_size": observation.committee_size,
        "pacemaker_timeout_ms": observation.pacemaker_timeout_ms,
        "mempool_max_batch_txs": observation.mempool_max_batch_txs,
        "mempool_proposal_interval_ms": observation.mempool_proposal_interval_ms,
        "throughput_tps": observation.throughput_tps,
        "latency_p95_ms": observation.latency_p95_ms,
        "backlog_pending": observation.backlog_pending,
        "backlog_missing": observation.backlog_missing,
        "reject_total": observation.reject_total,
        "pending_joins": observation.pending_joins,
        "pending_leaves": observation.pending_leaves,
        "lset_size": observation.lset_size,
        "can_participate": 1.0 if observation.can_participate else 0.0,
        "local_validator": 1.0 if observation.local_validator else 0.0,
        "heterogeneity_score": observation.heterogeneity_score,
        "churn_rate": observation.churn_rate,
        "adversary_score": observation.adversary_score,
        "network_jitter_ms": observation.network_jitter_ms,
        "ai_load_score": observation.ai_load_score,
        "agent_count": agent_count,
        "agent_mean_committee_size": mean_committee,
        "agent_mean_timeout_ms": mean_timeout,
        "agent_mean_batch_txs": mean_batch,
        "agent_mean_proposal_interval_ms": mean_interval,
    }
    return _as_vector([float(values[key]) for key in FEATURE_ORDER])


def encode_action(action: Action):
    return _as_vector(
        [
            float(action.committee_size),
            float(action.pacemaker_timeout_ms),
            float(action.mempool_max_batch_txs),
            float(action.mempool_proposal_interval_ms),
            float(1 if action.submit_join else 0),
            float(1 if action.submit_leave else 0),
            float(action.hydra_discovery_target),
        ]
    )


def action_diff_fields(left: Action, right: Action) -> list[str]:
    changed = []
    for field in [
        "committee_size",
        "pacemaker_timeout_ms",
        "mempool_max_batch_txs",
        "mempool_proposal_interval_ms",
        "submit_join",
        "submit_leave",
        "hydra_discovery_target",
    ]:
        if getattr(left, field) != getattr(right, field):
            changed.append(field)
    if [agent.to_dict() for agent in left.agent_actions] != [agent.to_dict() for agent in right.agent_actions]:
        changed.append("agent_actions")
    return changed


def action_has_divergence(left: Action, right: Action) -> bool:
    return bool(action_diff_fields(left, right))


def has_explicit_action(action: Action) -> bool:
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


def training_target_action(sample: TrajectorySample) -> Action:
    if sample.governed.present:
        return sample.governed.action
    return sample.applied.action


def training_target_agent_actions(sample: TrajectorySample) -> list[AgentAction]:
    if sample.governed.present:
        return training_target_action(sample).agent_actions
    return sample.applied.action.agent_actions


def encode_agent_observation(agent: AgentObservation):
    return _as_vector(
        [
            float(agent.instance_id),
            float(agent.epoch),
            float(agent.validator_count),
            float(agent.committee_size),
            float(agent.pacemaker_timeout_ms),
            float(agent.mempool_max_batch_txs),
            float(agent.mempool_proposal_interval_ms),
        ]
    )


def encode_agent_action(action: AgentAction):
    return _as_vector(
        [
            float(action.committee_size),
            float(action.pacemaker_timeout_ms),
            float(action.mempool_max_batch_txs),
            float(action.mempool_proposal_interval_ms),
        ]
    )


def load_trace_samples(path: str | Path) -> List[TrajectorySample]:
    samples: List[TrajectorySample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            sample = runtime_trace_sample_from_dict(json.loads(line))
            if sample.schema_version != SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported trace schema_version at line {line_number}: {sample.schema_version} != {SCHEMA_VERSION}"
                )
            samples.append(sample)
    return samples


def build_training_batch(samples: Iterable[TrajectorySample], infer_sequential_next_observation: bool = True) -> TrainingBatch:
    sample_list = list(samples)
    next_observations = []
    for idx, sample in enumerate(sample_list):
        if sample.next_observation is not None:
            next_observations.append(sample.next_observation)
        elif infer_sequential_next_observation and idx + 1 < len(sample_list):
            next_observations.append(sample_list[idx + 1].observation)
        else:
            next_observations.append(sample.observation)

    features = _as_matrix([encode_observation(sample.observation) for sample in sample_list], cols=len(FEATURE_ORDER))
    next_features = _as_matrix([encode_observation(obs) for obs in next_observations], cols=len(FEATURE_ORDER))
    actions = _as_matrix([encode_action(training_target_action(sample)) for sample in sample_list], cols=ACTION_DIM)
    candidate_actions = _as_matrix([encode_action(sample.candidate.action) for sample in sample_list], cols=ACTION_DIM)
    governed_actions = _as_matrix([encode_action(training_target_action(sample)) for sample in sample_list], cols=ACTION_DIM)
    rewards = _as_vector([sample.reward for sample in sample_list])
    team_rewards = _as_vector([sample.team_reward if sample.team_reward is not None else sample.reward for sample in sample_list])
    dones = _as_vector([1.0 if sample.done else 0.0 for sample in sample_list])
    governance_deltas = _as_vector([1.0 if sample.governance_delta else 0.0 for sample in sample_list])
    guardrail_deltas = _as_vector([1.0 if sample.guardrail_delta else 0.0 for sample in sample_list])
    agent_feature_rows = []
    agent_action_rows = []
    for sample in sample_list:
        target_agent_actions = training_target_agent_actions(sample)
        if sample.observation.agents and target_agent_actions:
            action_by_instance = {agent.instance_id: agent for agent in target_agent_actions}
            for agent in sample.observation.agents:
                if agent.instance_id not in action_by_instance:
                    continue
                agent_feature_rows.append(encode_agent_observation(agent))
                agent_action_rows.append(encode_agent_action(action_by_instance[agent.instance_id]))
    if agent_feature_rows:
        agent_features = _as_matrix(agent_feature_rows, cols=AGENT_FEATURE_DIM)
        agent_actions = _as_matrix(agent_action_rows, cols=AGENT_ACTION_DIM)
    else:
        agent_features = _zeros(0, AGENT_FEATURE_DIM)
        agent_actions = _zeros(0, AGENT_ACTION_DIM)
    return TrainingBatch(
        features=features,
        next_features=next_features,
        actions=actions,
        candidate_actions=candidate_actions,
        governed_actions=governed_actions,
        rewards=rewards,
        team_rewards=team_rewards,
        dones=dones,
        governance_deltas=governance_deltas,
        guardrail_deltas=guardrail_deltas,
        agent_features=agent_features,
        agent_actions=agent_actions,
    )
