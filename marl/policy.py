from __future__ import annotations

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None

from marl.dataset import encode_agent_observation, encode_observation
from marl.organization import ACTION_FIELDS, MOISEOrganizationModel, ROLE_ACTION_FIELDS
from marl.schemas import Action, AgentAction, Observation
from marl.trainer import SafeFACMACModel


def project_action(observation: Observation, action: Action) -> Action:
    committee_size = 0 if action.committee_size <= 0 else max(4, min(action.committee_size, max(observation.validator_count, 4)))
    pacemaker_timeout_ms = int(min(max(action.pacemaker_timeout_ms, 250), 5000))
    mempool_max_batch_txs = int(min(max(action.mempool_max_batch_txs, 1), 8192))
    mempool_proposal_interval_ms = int(min(max(action.mempool_proposal_interval_ms, 10), 2000))
    submit_join = bool(action.submit_join and not observation.local_validator)
    submit_leave = bool(action.submit_leave and observation.local_validator and observation.validator_count > 3)
    if submit_join and submit_leave:
        submit_join = False
        submit_leave = False
    hydra_discovery_target = int(max(action.hydra_discovery_target, 0))
    projected_agent_actions = []
    if action.agent_actions:
        validators_by_instance = {agent.instance_id: max(agent.validator_count, 4) for agent in observation.agents}
        for agent_action in action.agent_actions:
            max_validators = validators_by_instance.get(agent_action.instance_id, max(observation.validator_count, 4))
            projected_agent_actions.append(
                AgentAction(
                    instance_id=agent_action.instance_id,
                    committee_size=0 if agent_action.committee_size <= 0 else max(4, min(agent_action.committee_size, max_validators)),
                    pacemaker_timeout_ms=int(min(max(agent_action.pacemaker_timeout_ms, 250), 5000)),
                    mempool_max_batch_txs=int(min(max(agent_action.mempool_max_batch_txs, 1), 8192)),
                    mempool_proposal_interval_ms=int(min(max(agent_action.mempool_proposal_interval_ms, 10), 2000)),
                )
            )
    return Action(
        committee_size=committee_size,
        pacemaker_timeout_ms=pacemaker_timeout_ms,
        mempool_max_batch_txs=mempool_max_batch_txs,
        mempool_proposal_interval_ms=mempool_proposal_interval_ms,
        submit_join=submit_join,
        submit_leave=submit_leave,
        hydra_discovery_target=hydra_discovery_target,
        reason=action.reason,
        agent_actions=projected_agent_actions,
    )


class SafeFACMACPolicy:
    def __init__(self, model: SafeFACMACModel):
        self.model = model
        self.organization = MOISEOrganizationModel()

    def propose(self, observation: Observation) -> Action:
        action, _ = self.propose_with_role_attribution(observation)
        return action

    def propose_with_role_attribution(self, observation: Observation) -> tuple[Action, dict]:
        x = encode_observation(observation)
        raw = _affine(self.model.actor_weights, self.model.actor_bias, x)
        critic_value = float(_dot(self.model.critic_weights, x) + self.model.critic_bias)

        agent_actions = []
        if observation.agents and _matrix_rows(self.model.agent_actor_weights) > 0:
            for agent in observation.agents:
                agent_x = encode_agent_observation(agent)
                agent_raw = _affine(self.model.agent_actor_weights, self.model.agent_actor_bias, agent_x)
                agent_actions.append(
                    AgentAction(
                        instance_id=agent.instance_id,
                        committee_size=int(round(agent_raw[0])) if agent.validator_count else 0,
                        pacemaker_timeout_ms=int(round(agent_raw[1])),
                        mempool_max_batch_txs=int(round(agent_raw[2])),
                        mempool_proposal_interval_ms=int(round(agent_raw[3])),
                    )
                )

        action = Action(
            committee_size=int(round(raw[0])) if observation.validator_count else 0,
            pacemaker_timeout_ms=int(round(raw[1])),
            mempool_max_batch_txs=int(round(raw[2])),
            mempool_proposal_interval_ms=int(round(raw[3])),
            submit_join=bool(raw[4] >= 0.5),
            submit_leave=bool(raw[5] >= 0.5),
            hydra_discovery_target=int(round(raw[6])),
            reason=f"safe-facmac critic={critic_value:.3f}",
            agent_actions=agent_actions,
        )

        attribution = {"override_roles": [], "by_role": {}, "by_field": {}, "agent_count": len(agent_actions), "agent_instances": [agent.instance_id for agent in agent_actions]}
        role_weights = getattr(self.model, "role_actor_weights", {}) or {}
        role_bias = getattr(self.model, "role_actor_bias", {}) or {}
        if role_weights or role_bias:
            decision = self.organization.evaluate(observation)
            overrides = {}
            for role in decision.active_roles:
                weights = role_weights.get(role)
                bias = role_bias.get(role)
                fields = ROLE_ACTION_FIELDS.get(role, ())
                if weights is None or bias is None or not fields:
                    continue
                role_raw = _affine(weights, bias, x)
                values = {}
                for idx, field in enumerate(fields):
                    value = role_raw[idx]
                    overrides[field] = value
                    values[field] = value
                    attribution["by_field"][field] = {"role": role, "value": value}
                attribution["override_roles"].append(role)
                attribution["by_role"][role] = {"fields": list(fields), "values": values}
            if overrides:
                action = _compose_role_fields(action, overrides)
        return action, attribution

    def decide(self, observation: Observation) -> Action:
        return project_action(observation, self.organization.sanitize(observation, self.propose(observation)))


def build_default_action(observation: Observation) -> Action:
    return project_action(
        observation,
        Action(
            committee_size=observation.committee_size,
            pacemaker_timeout_ms=observation.pacemaker_timeout_ms,
            mempool_max_batch_txs=observation.mempool_max_batch_txs,
            mempool_proposal_interval_ms=observation.mempool_proposal_interval_ms,
            reason="default",
        ),
    )


def _matrix_rows(value) -> int:
    if hasattr(value, "shape"):
        shape = value.shape
        if len(shape) >= 1:
            return int(shape[0])
        return 0
    return len(value)


def _dot(weights, vector) -> float:
    if np is not None:
        w = np.asarray(weights)
        if w.size == 0:
            return 0.0
        return float(w @ vector)
    if not weights:
        return 0.0
    return float(sum(float(w) * float(v) for w, v in zip(weights, vector)))


def _compose_role_fields(base_action: Action, overrides: dict[str, float]) -> Action:
    return Action(
        committee_size=int(round(overrides.get("committee_size", base_action.committee_size))),
        pacemaker_timeout_ms=int(round(overrides.get("pacemaker_timeout_ms", base_action.pacemaker_timeout_ms))),
        mempool_max_batch_txs=int(round(overrides.get("mempool_max_batch_txs", base_action.mempool_max_batch_txs))),
        mempool_proposal_interval_ms=int(round(overrides.get("mempool_proposal_interval_ms", base_action.mempool_proposal_interval_ms))),
        submit_join=bool(overrides.get("submit_join", 1.0 if base_action.submit_join else 0.0) >= 0.5),
        submit_leave=bool(overrides.get("submit_leave", 1.0 if base_action.submit_leave else 0.0) >= 0.5),
        hydra_discovery_target=int(round(overrides.get("hydra_discovery_target", base_action.hydra_discovery_target))),
        reason=base_action.reason,
        agent_actions=base_action.agent_actions,
    )


def _affine(weights, bias, vector):
    if np is not None:
        w = np.asarray(weights)
        if w.size == 0:
            return np.asarray(bias, dtype=float).tolist()
        return (w @ vector + bias).tolist()
    if not weights:
        return [float(v) for v in bias]
    out = []
    for row, b in zip(weights, bias):
        out.append(_dot(row, vector) + float(b))
    return out
