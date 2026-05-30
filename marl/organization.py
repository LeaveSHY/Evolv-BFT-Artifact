from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from marl.schemas import Action, AgentAction, Observation, SCHEMA_VERSION

ACTION_FIELDS = (
    "committee_size",
    "pacemaker_timeout_ms",
    "mempool_max_batch_txs",
    "mempool_proposal_interval_ms",
    "submit_join",
    "submit_leave",
    "hydra_discovery_target",
)

ROLE_ACTION_FIELDS = {
    "lane_tuner": ("committee_size", "mempool_max_batch_txs", "mempool_proposal_interval_ms"),
    "recovery_tuner": ("pacemaker_timeout_ms",),
    "membership_tuner": ("submit_join", "submit_leave", "hydra_discovery_target"),
    "safety_guardian": tuple(),
}

# Cross-language mapping: Go MOISE+ roles → Python reward heads
# See Go adaptive/roles.go RoleToRewardHead for canonical definition.
MOISE_ROLE_TO_REWARD_HEAD = {
    "sentinel": "recovery_tuner",
    "commander": "membership_tuner",
    "tuner": "lane_tuner",
    "guardian": "safety_guardian",
}


@dataclass(frozen=True)
class RoleSpec:
    name: str
    allowed_fields: tuple[str, ...]
    missions: tuple[str, ...]


@dataclass
class OrganizationDecision:
    active_roles: List[str] = field(default_factory=list)
    missions: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    blocked_fields: List[str] = field(default_factory=list)
    blocked_field_reasons: Dict[str, List[str]] = field(default_factory=dict)
    escalation_level: str = "nominal"
    freeze_membership: bool = False
    freeze_lane_tuning: bool = False

    def block(self, fields: List[str] | tuple[str, ...], reason: str) -> None:
        for field in fields:
            self.blocked_fields.append(field)
            reasons = self.blocked_field_reasons.setdefault(field, [])
            if reason not in reasons:
                reasons.append(reason)
    evidence: Dict[str, float | int | bool] = field(default_factory=dict)
    role_field_map: Dict[str, List[str]] = field(default_factory=dict)


ROLE_PRIORITY = {
    "safety_guardian": 0,
    "recovery_tuner": 1,
    "membership_tuner": 2,
    "lane_tuner": 3,
}


class MOISEOrganizationModel:
    def __init__(self) -> None:
        self.roles: Dict[str, RoleSpec] = {
            "lane_tuner": RoleSpec(
                name="lane_tuner",
                allowed_fields=ROLE_ACTION_FIELDS["lane_tuner"],
                missions=("maximize_throughput", "reduce_tail_latency"),
            ),
            "recovery_tuner": RoleSpec(
                name="recovery_tuner",
                allowed_fields=ROLE_ACTION_FIELDS["recovery_tuner"],
                missions=("stabilize_recovery", "respect_safety_envelope"),
            ),
            "membership_tuner": RoleSpec(
                name="membership_tuner",
                allowed_fields=ROLE_ACTION_FIELDS["membership_tuner"],
                missions=("handle_membership_churn", "respect_safety_envelope"),
            ),
            "safety_guardian": RoleSpec(
                name="safety_guardian",
                allowed_fields=(),
                missions=("respect_safety_envelope",),
            ),
        }

    def evaluate(self, observation: Observation) -> OrganizationDecision:
        decision = OrganizationDecision()
        if observation.local_validator:
            decision.active_roles.append("lane_tuner")
        if observation.backlog_missing > 0 or observation.reject_total > 0 or observation.adversary_score > 0.2:
            decision.active_roles.append("recovery_tuner")
        if (
            not observation.local_validator
            or not observation.can_participate
            or observation.pending_joins > 0
            or observation.pending_leaves > 0
            or observation.highest_known_config_id > observation.current_config_id
        ):
            decision.active_roles.append("membership_tuner")
        if observation.reject_total > 0 or observation.backlog_missing > 0:
            decision.active_roles.append("safety_guardian")

        decision.active_roles = sorted(set(decision.active_roles), key=lambda role: ROLE_PRIORITY[role])
        decision.escalation_level = self._escalation_level(observation)
        decision.freeze_membership = self._freeze_membership(observation, decision.escalation_level)
        decision.freeze_lane_tuning = self._freeze_lane_tuning(observation, decision.escalation_level)
        decision.evidence = {
            "reject_total": observation.reject_total,
            "backlog_missing": observation.backlog_missing,
            "adversary_score": observation.adversary_score,
            "churn_rate": observation.churn_rate,
            "pending_joins": observation.pending_joins,
            "pending_leaves": observation.pending_leaves,
            "can_participate": observation.can_participate,
        }
        decision.role_field_map = {
            role: list(self.roles[role].allowed_fields)
            for role in decision.active_roles
        }
        if decision.freeze_membership:
            decision.block(["submit_join", "submit_leave", "hydra_discovery_target"], "membership-frozen")
            decision.notes.append("membership-frozen")
        if decision.freeze_lane_tuning:
            decision.block(ROLE_ACTION_FIELDS["lane_tuner"], "lane-tuning-frozen")
            decision.notes.append("lane-tuning-frozen")
        if "safety_guardian" in decision.active_roles:
            safety_reason = f"safety-escalation:{decision.escalation_level}"
            decision.block(self._safety_blocked_fields(observation, decision.escalation_level), safety_reason)
            if not observation.can_participate:
                decision.block(["submit_leave"], "cannot-participate")
            decision.notes.append(safety_reason)

        seen = set()
        for role in decision.active_roles:
            for mission in self.roles[role].missions:
                if mission not in seen:
                    decision.missions.append(mission)
                    seen.add(mission)
        if not decision.active_roles:
            decision.notes.append("no specialized role activated")
        return decision

    def sanitize(self, observation: Observation, action: Action) -> Action:
        decision = self.evaluate(observation)
        allowed = set()
        for role in decision.active_roles:
            allowed.update(self.roles[role].allowed_fields)
        allowed.difference_update(decision.blocked_fields)

        if not decision.active_roles:
            sanitized = Action(
                committee_size=action.committee_size,
                pacemaker_timeout_ms=action.pacemaker_timeout_ms,
                mempool_max_batch_txs=action.mempool_max_batch_txs,
                mempool_proposal_interval_ms=action.mempool_proposal_interval_ms,
                submit_join=action.submit_join,
                submit_leave=action.submit_leave,
                hydra_discovery_target=action.hydra_discovery_target,
                reason=action.reason or "organization:no-active-role",
                agent_actions=list(action.agent_actions),
            )
        else:
            sanitized = Action(
                committee_size=self._held_or_proposed(
                    allowed,
                    "committee_size",
                    action.committee_size,
                    observation.committee_size,
                ),
                pacemaker_timeout_ms=self._held_or_proposed(
                    allowed,
                    "pacemaker_timeout_ms",
                    action.pacemaker_timeout_ms,
                    observation.pacemaker_timeout_ms,
                ),
                mempool_max_batch_txs=self._held_or_proposed(
                    allowed,
                    "mempool_max_batch_txs",
                    action.mempool_max_batch_txs,
                    observation.mempool_max_batch_txs,
                ),
                mempool_proposal_interval_ms=self._held_or_proposed(
                    allowed,
                    "mempool_proposal_interval_ms",
                    action.mempool_proposal_interval_ms,
                    observation.mempool_proposal_interval_ms,
                ),
                submit_join=action.submit_join if "submit_join" in allowed else False,
                submit_leave=action.submit_leave if "submit_leave" in allowed else False,
                hydra_discovery_target=action.hydra_discovery_target if "hydra_discovery_target" in allowed else 0,
                reason=action.reason,
                agent_actions=self._sanitize_agent_actions(observation, action, allowed),
            )
        suffix = ",".join(decision.active_roles) if decision.active_roles else "none"
        note_suffix = ";".join(decision.notes) if decision.notes else "nominal"
        if sanitized.reason:
            sanitized.reason = f"{sanitized.reason} roles={suffix} escalation={decision.escalation_level} notes={note_suffix}"
        else:
            sanitized.reason = f"roles={suffix} escalation={decision.escalation_level} notes={note_suffix}"
        return sanitized

    def snapshot(self) -> dict:
        return {
            "name": "octopus-moise-marl",
            "schema_version": SCHEMA_VERSION,
            "action_fields": list(ACTION_FIELDS),
            "decision_fields": list(OrganizationDecision.__dataclass_fields__.keys()),
            "role_action_fields": {name: list(fields) for name, fields in ROLE_ACTION_FIELDS.items()},
            "role_priority": ROLE_PRIORITY,
            "role_activation_rules": {
                "lane_tuner": "activate when the local node is currently a validator",
                "recovery_tuner": "activate when backlog gaps, rejects, or adversary pressure indicate recovery stress",
                "membership_tuner": "activate when validator participation, config convergence, or membership churn needs coordination",
                "safety_guardian": "activate when rejects or backlog gaps indicate a safety-relevant disturbance",
            },
            "escalation_rules": {
                "critical": "reject_total > 10, backlog_missing > 10, or adversary_score >= 0.7",
                "elevated": "reject_total > 0, backlog_missing > 0, adversary_score >= 0.3, or churn_rate >= 0.4",
                "nominal": "all elevated and critical triggers inactive",
            },
            "freeze_rules": {
                "membership": "freeze when safety escalation is elevated/critical or node cannot safely participate",
                "lane_tuning": "freeze when elevated/critical instability is active",
            },
            "blocked_field_reasoning": {
                "membership-frozen": ["submit_join", "submit_leave", "hydra_discovery_target"],
                "lane-tuning-frozen": list(ROLE_ACTION_FIELDS["lane_tuner"]),
                "safety-escalation:elevated": ["submit_join", "submit_leave", "hydra_discovery_target"],
                "safety-escalation:critical": [
                    "submit_join",
                    "submit_leave",
                    "hydra_discovery_target",
                    "committee_size",
                    "mempool_max_batch_txs",
                    "mempool_proposal_interval_ms",
                ],
                "cannot-participate": ["submit_leave"],
            },
            "safety_blocking_rules": {
                "elevated_or_critical": ["submit_join", "submit_leave", "hydra_discovery_target"],
                "critical": ["committee_size", "mempool_max_batch_txs", "mempool_proposal_interval_ms"],
                "cannot_participate": ["submit_leave"],
            },
            "roles": {
                name: {
                    "allowed_fields": list(spec.allowed_fields),
                    "missions": list(spec.missions),
                }
                for name, spec in self.roles.items()
            },
        }

    def _escalation_level(self, observation: Observation) -> str:
        if observation.reject_total > 10 or observation.backlog_missing > 10 or observation.adversary_score >= 0.7:
            return "critical"
        if observation.reject_total > 0 or observation.backlog_missing > 0 or observation.adversary_score >= 0.3 or observation.churn_rate >= 0.4:
            return "elevated"
        return "nominal"

    def _freeze_membership(self, observation: Observation, escalation_level: str) -> bool:
        if escalation_level in {"elevated", "critical"}:
            return True
        return observation.pending_joins > 0 or observation.pending_leaves > 0

    def _freeze_lane_tuning(self, observation: Observation, escalation_level: str) -> bool:
        return escalation_level == "critical" or observation.backlog_missing > 0

    def _safety_blocked_fields(self, observation: Observation, escalation_level: str) -> List[str]:
        blocked: List[str] = []
        if escalation_level in {"elevated", "critical"}:
            blocked.extend(["submit_join", "submit_leave", "hydra_discovery_target"])
        if escalation_level == "critical":
            blocked.extend(["committee_size", "mempool_max_batch_txs", "mempool_proposal_interval_ms"])
        return blocked

    def _held_or_proposed(self, allowed: set[str], field: str, proposed: int, current: int) -> int:
        return proposed if field in allowed else current

    def _sanitize_agent_actions(self, observation: Observation, action: Action, allowed: set[str]) -> list[AgentAction]:
        if not action.agent_actions:
            return []
        observed_agents = {agent.instance_id: agent for agent in observation.agents}
        sanitized: list[AgentAction] = []
        for proposed in action.agent_actions:
            current = observed_agents.get(proposed.instance_id)
            if current is None:
                continue
            sanitized.append(
                AgentAction(
                    instance_id=proposed.instance_id,
                    committee_size=self._held_or_proposed(allowed, "committee_size", proposed.committee_size, current.committee_size),
                    pacemaker_timeout_ms=self._held_or_proposed(
                        allowed,
                        "pacemaker_timeout_ms",
                        proposed.pacemaker_timeout_ms,
                        current.pacemaker_timeout_ms,
                    ),
                    mempool_max_batch_txs=self._held_or_proposed(
                        allowed,
                        "mempool_max_batch_txs",
                        proposed.mempool_max_batch_txs,
                        current.mempool_max_batch_txs,
                    ),
                    mempool_proposal_interval_ms=self._held_or_proposed(
                        allowed,
                        "mempool_proposal_interval_ms",
                        proposed.mempool_proposal_interval_ms,
                        current.mempool_proposal_interval_ms,
                    ),
                )
            )
        return sanitized
