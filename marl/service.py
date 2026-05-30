from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from pathlib import Path, PureWindowsPath
from typing import Optional

from marl.dataset import action_diff_fields, action_has_divergence, build_training_batch, load_trace_samples
from marl.organization import MOISEOrganizationModel, ROLE_ACTION_FIELDS
from marl.policy import SafeFACMACPolicy, project_action
from marl.replay import ReplayBuffer, has_stress_signal
from marl.schemas import Action, Observation, SCHEMA_VERSION, TrajectorySample, schema_snapshot
from marl.trainer import SafeFACMACModel, SafeFACMACTrainer, load_checkpoint, save_checkpoint


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
TRACE_ROOT = WORKSPACE_ROOT
CHECKPOINT_ROOT = WORKSPACE_ROOT
EXPOSED_TRAINER_METADATA_FIELDS = {
    "trainer",
    "trainer_family",
    "paper_grade_facmac",
    "claim_boundary",
    "epochs",
    "actor_lr",
    "critic_lr",
    "bc_lr",
    "target_action",
    "governance_delta_rate",
    "guardrail_delta_rate",
    "role_head_coverage",
    "reward_summary",
    "history",
    "trainer_config",
}
CHECKPOINT_TRAINING_SUMMARY_FIELDS = {
    "mode",
    "samples",
    "batch_size",
    "trace_path",
    "trace_manifest_path",
    "scenario_label",
    "seed",
    "replay_size",
    "schema_version",
    "replay_priority_summary",
    "sample_summary",
    "trainer_metadata",
}
ARTIFACT_INVENTORY_CHECKPOINT_FIELDS = (
    "best_reward_mean",
    "best_saved_path",
    "last_loaded_path",
    "last_saved_path",
)
ADAPTIVE_SNAPSHOT_FIELDS = [
    "model_ready",
    "model",
    "organization",
    "schema",
    "replay",
    "training",
    "artifact_inventory",
]


def _validate_scoped_workspace_path(value: str | Path, field_name: str, allowed_root: Path) -> Path:
    raw = str(value).strip()
    if not raw:
        raise ValueError(f"{field_name} must not be empty")
    windows_path = PureWindowsPath(raw)
    if windows_path.is_absolute() or windows_path.drive or ".." in windows_path.parts:
        raise ValueError(f"{field_name} must be a relative path inside the workspace")
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field_name} must be a relative path inside the workspace")
    resolved = (WORKSPACE_ROOT / path).resolve()
    if not resolved.is_relative_to(WORKSPACE_ROOT):
        raise ValueError(f"{field_name} must be a relative path inside the workspace")
    if not resolved.is_relative_to(allowed_root.resolve()):
        raise ValueError(f"{field_name} must stay inside {allowed_root.relative_to(WORKSPACE_ROOT)}")
    return resolved


def _validate_workspace_path(value: str | Path, field_name: str) -> Path:
    return _validate_scoped_workspace_path(value, field_name=field_name, allowed_root=WORKSPACE_ROOT)


def _validate_trace_path(value: str | Path, field_name: str = "trace_path") -> Path:
    return _validate_scoped_workspace_path(value, field_name=field_name, allowed_root=TRACE_ROOT)


def _validate_checkpoint_path(value: str | Path, field_name: str = "path") -> Path:
    return _validate_scoped_workspace_path(value, field_name=field_name, allowed_root=CHECKPOINT_ROOT)


def _checkpoint_metadata_payload(checkpoint_summary: dict, last_training_summary: dict | None) -> dict:
    return {
        "best_saved_path": checkpoint_summary.get("best_saved_path"),
        "best_reward_mean": checkpoint_summary.get("best_reward_mean"),
        "last_training": json.loads(json.dumps(last_training_summary)) if last_training_summary is not None else None,
    }


def _parse_optional_finite_float(value) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _sanitize_workspace_relative_path(value, *, field_name: str = "checkpoint provenance path", allowed_root: Path = WORKSPACE_ROOT) -> str | None:
    if value is None or not isinstance(value, str):
        return None
    try:
        return str(_validate_scoped_workspace_path(value, field_name=field_name, allowed_root=allowed_root).relative_to(WORKSPACE_ROOT))
    except ValueError:
        return None


def _safe_int(value, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(parsed, 0)


def _filter_claim_disclosure_metadata(value: dict | None) -> dict | None:
    if not isinstance(value, dict):
        return value
    filtered = json.loads(json.dumps(value))
    for field in ("trainer_family", "paper_grade_facmac", "claim_boundary"):
        filtered.pop(field, None)
    return filtered


def _fallback_checkpoint_training_summary(replay_size: int, trainer_metadata: dict | None) -> dict:
    return {
        "mode": "checkpoint_origin_unknown",
        "samples": 0,
        "replay_size": replay_size,
        "schema_version": SCHEMA_VERSION,
        "replay_priority_summary": {
            "priority_sample_count": 0,
            "governance_delta_count": 0,
            "guardrail_delta_count": 0,
            "stress_signal_count": 0,
        },
        "sample_summary": {
            "requested_batch_size": None,
            "effective_batch_size": 0,
            "priority_sample_count": 0,
            "governance_delta_count": 0,
            "guardrail_delta_count": 0,
            "stress_signal_count": 0,
            "reward_mean": 0.0,
        },
        "trainer_metadata": _filter_claim_disclosure_metadata(trainer_metadata),
    }


def _summarize_sample_batch(batch: list) -> dict:
    if not batch:
        return {
            "effective_batch_size": 0,
            "priority_sample_count": 0,
            "governance_delta_count": 0,
            "guardrail_delta_count": 0,
            "stress_signal_count": 0,
            "reward_mean": 0.0,
        }
    reward_values = [float(sample.reward) for sample in batch]
    governance_delta_count = sum(1 for sample in batch if sample.governance_delta)
    guardrail_delta_count = sum(1 for sample in batch if sample.guardrail_delta)
    stress_signal_count = sum(1 for sample in batch if has_stress_signal(sample))
    return {
        "effective_batch_size": len(batch),
        "priority_sample_count": sum(
            1
            for sample in batch
            if sample.governance_delta or sample.guardrail_delta or has_stress_signal(sample)
        ),
        "governance_delta_count": governance_delta_count,
        "guardrail_delta_count": guardrail_delta_count,
        "stress_signal_count": stress_signal_count,
        "reward_mean": sum(reward_values) / len(reward_values),
    }


def _sanitize_role_head_coverage(value) -> dict[str, dict[str, float]]:
    if not isinstance(value, dict):
        return {}
    sanitized: dict[str, dict[str, float]] = {}
    for role, coverage in value.items():
        if role not in ROLE_ACTION_FIELDS or not isinstance(coverage, dict):
            continue
        sanitized[str(role)] = {
            "active_samples": _parse_optional_finite_float(coverage.get("active_samples")) or 0.0,
            "total_samples": _parse_optional_finite_float(coverage.get("total_samples")) or 0.0,
        }
    return sanitized


def _sanitize_history(value) -> dict[str, list[float]]:
    if not isinstance(value, dict):
        return {}
    sanitized: dict[str, list[float]] = {}
    for key, field_value in value.items():
        if not isinstance(field_value, list):
            continue
        sanitized_values = []
        for item in field_value:
            parsed = _parse_optional_finite_float(item)
            if parsed is not None:
                sanitized_values.append(parsed)
        sanitized[str(key)] = sanitized_values
    return sanitized


def _sanitize_trainer_config(value) -> dict[str, float | int]:
    if not isinstance(value, dict):
        return {}
    sanitized: dict[str, float | int] = {}
    int_fields = {"seed", "epochs"}
    float_fields = {"ridge", "actor_lr", "critic_lr", "bc_lr"}
    for field in int_fields:
        if field in value:
            sanitized[field] = _safe_int(value.get(field), default=0)
    for field in float_fields:
        parsed = _parse_optional_finite_float(value.get(field))
        if parsed is not None:
            sanitized[field] = parsed
    return sanitized


def _sanitize_reward_summary(value) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {
        "reward_mean": _parse_optional_finite_float(value.get("reward_mean")) or 0.0,
        "team_reward_mean": _parse_optional_finite_float(value.get("team_reward_mean")) or 0.0,
        "role_reward_mean": _parse_optional_finite_float(value.get("role_reward_mean")) or 0.0,
    }


def _sanitize_restored_trainer_metadata(value) -> dict | None:
    if not isinstance(value, dict):
        return None
    sanitized: dict = {}
    if "trainer" in value and isinstance(value.get("trainer"), str):
        sanitized["trainer"] = value["trainer"]
    if "trainer_family" in value and isinstance(value.get("trainer_family"), str):
        sanitized["trainer_family"] = value["trainer_family"]
    if "paper_grade_facmac" in value and isinstance(value.get("paper_grade_facmac"), bool):
        sanitized["paper_grade_facmac"] = value["paper_grade_facmac"]
    if "claim_boundary" in value and isinstance(value.get("claim_boundary"), str):
        sanitized["claim_boundary"] = value["claim_boundary"]
    for field in ("epochs",):
        if field in value:
            sanitized[field] = _safe_int(value.get(field), default=0)
    for field in ("actor_lr", "critic_lr", "bc_lr", "governance_delta_rate", "guardrail_delta_rate"):
        parsed = _parse_optional_finite_float(value.get(field))
        if parsed is not None:
            sanitized[field] = parsed
    if "target_action" in value and isinstance(value.get("target_action"), str):
        sanitized["target_action"] = value["target_action"]
    if "role_head_coverage" in value:
        sanitized["role_head_coverage"] = _sanitize_role_head_coverage(value.get("role_head_coverage"))
    if "reward_summary" in value:
        sanitized["reward_summary"] = _sanitize_reward_summary(value.get("reward_summary"))
    if "history" in value:
        sanitized["history"] = _sanitize_history(value.get("history"))
    if "trainer_config" in value:
        sanitized["trainer_config"] = _sanitize_trainer_config(value.get("trainer_config"))
    return sanitized


def _trusted_checkpoint_claim_metadata(model: SafeFACMACModel) -> dict[str, object] | None:
    metadata = model.metadata if isinstance(model.metadata, dict) else {}
    checkpoint_metadata = metadata.get("checkpoint_provenance") if isinstance(metadata, dict) else None
    if not isinstance(checkpoint_metadata, dict):
        return None
    trusted: dict[str, object] = {}
    for field in ("trainer_family", "paper_grade_facmac", "claim_boundary"):
        if field in metadata:
            trusted[field] = json.loads(json.dumps(metadata[field]))
    return trusted


def _sanitize_restored_training_summary(value: dict | None) -> dict | None:
    if not isinstance(value, dict):
        return None
    schema_version = value.get("schema_version")
    if schema_version is None:
        schema_version = SCHEMA_VERSION
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema_version: {schema_version}")
    replay_priority_summary = value.get("replay_priority_summary")
    if not isinstance(replay_priority_summary, dict):
        replay_priority_summary = {}
    sample_summary = value.get("sample_summary")
    has_sample_summary = isinstance(sample_summary, dict)
    if not has_sample_summary:
        sample_summary = {}
    batch_size = value.get("batch_size")
    sanitized_batch_size = _safe_int(batch_size, default=0) if batch_size is not None else None
    requested_batch_size = sample_summary.get("requested_batch_size")
    sanitized_requested_batch_size = _safe_int(requested_batch_size, default=0) if requested_batch_size is not None else sanitized_batch_size
    sanitized = {
        "mode": str(value.get("mode") or "checkpoint_origin_unknown"),
        "samples": _safe_int(value.get("samples"), default=0),
        "batch_size": sanitized_batch_size,
        "trace_path": _sanitize_workspace_relative_path(value.get("trace_path"), field_name="trace provenance path", allowed_root=TRACE_ROOT),
        "trace_manifest_path": _sanitize_workspace_relative_path(value.get("trace_manifest_path"), field_name="trace manifest provenance path", allowed_root=TRACE_ROOT),
        "scenario_label": str(value.get("scenario_label") or "") or None,
        "seed": str(value.get("seed") or "") or None,
        "replay_size": _safe_int(value.get("replay_size"), default=0),
        "schema_version": SCHEMA_VERSION,
        "replay_priority_summary": {
            "priority_sample_count": _safe_int(replay_priority_summary.get("priority_sample_count"), default=0),
            "governance_delta_count": _safe_int(replay_priority_summary.get("governance_delta_count"), default=0),
            "guardrail_delta_count": _safe_int(replay_priority_summary.get("guardrail_delta_count"), default=0),
            "stress_signal_count": _safe_int(replay_priority_summary.get("stress_signal_count"), default=0),
        },
        "trainer_metadata": _sanitize_restored_trainer_metadata(value.get("trainer_metadata")),
    }
    if has_sample_summary or sanitized["mode"] == "online":
        sanitized["sample_summary"] = {
            "requested_batch_size": sanitized_requested_batch_size,
            "effective_batch_size": _safe_int(sample_summary.get("effective_batch_size"), default=0),
            "priority_sample_count": _safe_int(sample_summary.get("priority_sample_count"), default=0),
            "governance_delta_count": _safe_int(sample_summary.get("governance_delta_count"), default=0),
            "guardrail_delta_count": _safe_int(sample_summary.get("guardrail_delta_count"), default=0),
            "stress_signal_count": _safe_int(sample_summary.get("stress_signal_count"), default=0),
            "reward_mean": _parse_optional_finite_float(sample_summary.get("reward_mean")) or 0.0,
        }
    return {
        field: sanitized[field]
        for field in CHECKPOINT_TRAINING_SUMMARY_FIELDS
        if field in sanitized
    }


def _restore_checkpoint_metadata(model: SafeFACMACModel) -> tuple[dict | None, dict | None]:
    metadata = model.metadata if isinstance(model.metadata, dict) else {}
    checkpoint_metadata = metadata.get("checkpoint_provenance")
    if not isinstance(checkpoint_metadata, dict):
        return None, None
    restored_checkpoint = {
        "best_saved_path": _sanitize_workspace_relative_path(
            checkpoint_metadata.get("best_saved_path"),
            field_name="checkpoint provenance path",
            allowed_root=CHECKPOINT_ROOT,
        ),
        "best_reward_mean": _parse_optional_finite_float(checkpoint_metadata.get("best_reward_mean")),
    }
    restored_training = _sanitize_restored_training_summary(checkpoint_metadata.get("last_training"))
    return restored_checkpoint, restored_training


def _stage_explanation(
    *,
    stage: str,
    changed: bool,
    fields: list[str],
    reason: str | None = None,
    role_override_attribution: dict | None = None,
    organization_decision: dict | None = None,
) -> dict:
    summary = f"{stage}: unchanged" if not changed else f"{stage}: changed"
    if changed and reason:
        summary = f"{summary} ({reason})"
    attributed_roles: list[str] = []
    if role_override_attribution and fields:
        by_field = role_override_attribution.get("by_field") if isinstance(role_override_attribution, dict) else {}
        attributed_roles = sorted({
            by_field[field]["role"]
            for field in fields
            if field in by_field and isinstance(by_field[field], dict) and "role" in by_field[field]
        })
    if organization_decision:
        governing_roles = organization_decision.get("active_roles") if isinstance(organization_decision, dict) else []
        if isinstance(governing_roles, list):
            attributed_roles = sorted(set(attributed_roles) | {role for role in governing_roles if isinstance(role, str)})
    blocked_by_guardrails: list[str] = []
    organization_notes: list[str] = []
    if organization_decision:
        blocked_fields = organization_decision.get("blocked_fields") if isinstance(organization_decision, dict) else []
        if isinstance(blocked_fields, list):
            blocked_by_guardrails = sorted({field for field in blocked_fields if isinstance(field, str)})
        notes = organization_decision.get("notes") if isinstance(organization_decision, dict) else []
        if isinstance(notes, list):
            organization_notes = [note for note in notes if isinstance(note, str)]
    return {
        "stage": stage,
        "changed": changed,
        "fields": list(fields),
        "reason": reason,
        "attributed_roles": attributed_roles,
        "blocked_by_guardrails": blocked_by_guardrails,
        "organization_notes": organization_notes,
        "summary": summary,
    }


def _organization_reasoning(decision) -> dict:
    blocked_field_reasons = getattr(decision, "blocked_field_reasons", {})
    return {
        "blocked_field_reasons": {field: list(reasons) for field, reasons in blocked_field_reasons.items()},
        "blocked_fields_without_reason": [
            field for field in decision.blocked_fields if field not in blocked_field_reasons
        ],
    }


class PolicyService:
    def __init__(self, replay_capacity: int = 4096, seed: int = 7):
        self._trainer = SafeFACMACTrainer()
        self._model: Optional[SafeFACMACModel] = None
        self._policy: Optional[SafeFACMACPolicy] = None
        self._replay = ReplayBuffer(capacity=replay_capacity)
        self._organization = MOISEOrganizationModel()
        self._last_training_summary: dict | None = None
        self._last_activation_summary: dict | None = None
        self._checkpoint_summary = {
            "last_saved_path": None,
            "last_loaded_path": None,
            "best_saved_path": None,
            "best_reward_mean": None,
        }
        self._online_sample_seed = seed
        self._online_sample_calls = 0
        self._active_model_source = "cold_start"

    def train_offline(self, trace_path: str | Path) -> dict:
        trace_path = _validate_trace_path(trace_path, field_name="trace_path")
        samples = load_trace_samples(trace_path)
        self._replay.extend(samples)
        self._model = self._trainer.fit(samples)
        self._policy = SafeFACMACPolicy(self._model)
        self._active_model_source = "offline_training"
        trace_relative_path = str(trace_path.relative_to(WORKSPACE_ROOT))
        trace_manifest_path = trace_path.with_name("runtime_trace_manifest.json")
        if trace_manifest_path.exists():
            manifest_payload = json.loads(trace_manifest_path.read_text(encoding="utf-8"))
            scenario_label = str(manifest_payload.get("scenario_label") or trace_path.parent.name)
            seed_label = str(manifest_payload.get("seed") or "unknown")
            trace_manifest_relative = str(trace_manifest_path.relative_to(WORKSPACE_ROOT))
        else:
            scenario_label = trace_path.parent.name if trace_path.parent != WORKSPACE_ROOT else "runtime_smoke"
            seed_label = "unknown"
            trace_manifest_relative = None
        self._last_training_summary = {
            "mode": "offline",
            "samples": len(samples),
            "trace_path": trace_relative_path,
            "trace_manifest_path": trace_manifest_relative,
            "scenario_label": scenario_label,
            "seed": seed_label,
            "replay_size": len(self._replay),
            "schema_version": SCHEMA_VERSION,
            "baseline_family": "runtime-backed-minimal",
            "claim_boundary": "minimal runtime-backed training baseline only",
            "replay_priority_summary": self._replay.priority_summary(),
            "trainer_metadata": self._trainer_metadata(),
        }
        self._last_activation_summary = {"mode": "offline_training"}
        return {"samples": len(samples), "model_ready": self._policy is not None, "training": self.training_summary()}

    def ingest(self, sample: TrajectorySample) -> dict:
        self._replay.push(sample)
        return {
            "replay_size": len(self._replay),
            "schema_version": sample.schema_version,
            "policy_name": sample.policy_name,
            "governance_delta": sample.governance_delta,
            "guardrail_delta": sample.guardrail_delta,
        }

    def train_online(self, batch_size: int) -> dict:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        batch = self._replay.sample(batch_size=batch_size, seed=self._online_sample_seed + self._online_sample_calls, mode="priority")
        if not batch:
            return {"samples": 0, "model_ready": self._policy is not None, "training": self.training_summary()}
        self._online_sample_calls += 1
        sample_summary = {
            "requested_batch_size": batch_size,
            **_summarize_sample_batch(list(batch)),
        }
        training_batch = build_training_batch(batch, infer_sequential_next_observation=False)
        self._model = self._trainer.fit_batch(training_batch, samples=list(batch))
        self._policy = SafeFACMACPolicy(self._model)
        self._active_model_source = "online_training"
        self._last_training_summary = {
            "mode": "online",
            "samples": len(batch),
            "batch_size": batch_size,
            "replay_size": len(self._replay),
            "schema_version": SCHEMA_VERSION,
            "replay_priority_summary": self._replay.priority_summary(),
            "sample_summary": sample_summary,
            "trainer_metadata": self._trainer_metadata(),
        }
        self._last_activation_summary = {"mode": "online_training"}
        return {"samples": len(batch), "model_ready": True, "training": self.training_summary()}

    def infer(self, observation: Observation) -> Action:
        _, _, applied = self._decision_flow_actions(observation)
        return applied

    def inspect_decision_flow(self, observation: Observation) -> dict:
        organization_decision, role_override_attribution, candidate, governed, applied = self._decision_flow_context(observation)
        organization_decision_payload = asdict(organization_decision)
        candidate_vs_governed_fields = action_diff_fields(candidate, governed)
        governed_vs_applied_fields = action_diff_fields(governed, applied)
        candidate_vs_applied_fields = action_diff_fields(candidate, applied)
        return {
            "model_ready": self._policy is not None,
            "organization_decision": organization_decision_payload,
            "organization_reasoning": _organization_reasoning(organization_decision),
            "role_override_attribution": role_override_attribution,
            "candidate": candidate.to_dict(),
            "governed": governed.to_dict(),
            "applied": applied.to_dict(),
            "divergence": {
                "candidate_vs_governed": bool(candidate_vs_governed_fields),
                "governed_vs_applied": bool(governed_vs_applied_fields),
                "candidate_vs_applied": bool(candidate_vs_applied_fields),
                "candidate_vs_governed_fields": candidate_vs_governed_fields,
                "governed_vs_applied_fields": governed_vs_applied_fields,
                "candidate_vs_applied_fields": candidate_vs_applied_fields,
            },
            "explanations": {
                "candidate_to_governed": _stage_explanation(
                    stage="organization",
                    changed=bool(candidate_vs_governed_fields),
                    fields=candidate_vs_governed_fields,
                    reason=governed.reason,
                    organization_decision=organization_decision_payload,
                )["summary"],
                "governed_to_applied": _stage_explanation(
                    stage="projection",
                    changed=bool(governed_vs_applied_fields),
                    fields=governed_vs_applied_fields,
                    reason=applied.reason,
                    organization_decision=organization_decision_payload,
                )["summary"],
                "candidate_to_applied": _stage_explanation(
                    stage="end_to_end",
                    changed=bool(candidate_vs_applied_fields),
                    fields=candidate_vs_applied_fields,
                    reason=applied.reason,
                    role_override_attribution=role_override_attribution,
                    organization_decision=organization_decision_payload,
                )["summary"],
            },
            "explanation_details": {
                "candidate_to_governed": _stage_explanation(
                    stage="organization",
                    changed=bool(candidate_vs_governed_fields),
                    fields=candidate_vs_governed_fields,
                    reason=governed.reason,
                    organization_decision=organization_decision_payload,
                ),
                "governed_to_applied": _stage_explanation(
                    stage="projection",
                    changed=bool(governed_vs_applied_fields),
                    fields=governed_vs_applied_fields,
                    reason=applied.reason,
                    organization_decision=organization_decision_payload,
                ),
                "candidate_to_applied": _stage_explanation(
                    stage="end_to_end",
                    changed=bool(candidate_vs_applied_fields),
                    fields=candidate_vs_applied_fields,
                    reason=applied.reason,
                    role_override_attribution=role_override_attribution,
                    organization_decision=organization_decision_payload,
                ),
            },
            "organization_explainability": {
                "active_roles": organization_decision.active_roles,
                "missions": organization_decision.missions,
                "blocked_fields": organization_decision.blocked_fields,
                "evidence": organization_decision.evidence,
                "role_field_map": organization_decision.role_field_map,
            },
            "artifact_inventory": self.artifact_inventory(),
        }

    def decision_envelope(self, observation: Observation) -> dict:
        organization_decision, role_override_attribution, candidate, governed, applied = self._decision_flow_context(observation)
        return {
            "action": candidate.to_dict(),
            "candidate": candidate.to_dict(),
            "governed": governed.to_dict(),
            "applied": applied.to_dict(),
            "organization_summary": {
                "active_roles": organization_decision.active_roles,
                "missions": organization_decision.missions,
                "blocked_fields": organization_decision.blocked_fields,
                "escalation_level": organization_decision.escalation_level,
                "freeze_membership": organization_decision.freeze_membership,
                "freeze_lane_tuning": organization_decision.freeze_lane_tuning,
            },
            "role_override_attribution": role_override_attribution,
            "claim_boundary": "co-runtime advisory decision envelope only; Go runtime remains final authority",
            "schema_version": SCHEMA_VERSION,
        }

    def _decision_flow_context(self, observation: Observation) -> tuple[object, dict, Action, Action, Action]:
        organization_decision = self._organization.evaluate(observation)
        role_override_attribution = {"override_roles": [], "by_role": {}, "by_field": {}}
        if self._policy is None:
            candidate = Action(
                committee_size=observation.committee_size,
                pacemaker_timeout_ms=observation.pacemaker_timeout_ms,
                mempool_max_batch_txs=observation.mempool_max_batch_txs,
                mempool_proposal_interval_ms=observation.mempool_proposal_interval_ms,
                reason="default",
            )
        else:
            candidate, role_override_attribution = self._policy.propose_with_role_attribution(observation)
        governed = self._organization.sanitize(observation, candidate)
        applied = project_action(observation, governed)
        return organization_decision, role_override_attribution, candidate, governed, applied

    def _decision_flow_actions(self, observation: Observation) -> tuple[Action, Action, Action]:
        _, _, candidate, governed, applied = self._decision_flow_context(observation)
        return candidate, governed, applied

    def artifact_inventory(self) -> dict:
        checkpoint = {
            field: self._checkpoint_summary.get(field)
            for field in ARTIFACT_INVENTORY_CHECKPOINT_FIELDS
        }
        trainer_metadata = self._trainer_metadata() or {}
        training_summary = dict(self._last_training_summary) if self._last_training_summary is not None else None
        checkpoint_trainer_metadata = trainer_metadata
        if not checkpoint_trainer_metadata and isinstance(training_summary, dict):
            checkpoint_trainer_metadata = training_summary.get("trainer_metadata") or {}
        checkpoint_metadata = {
            **_checkpoint_metadata_payload(self._checkpoint_summary, self._last_training_summary),
            "claim_boundary": checkpoint_trainer_metadata.get("claim_boundary"),
            "trainer_family": checkpoint_trainer_metadata.get("trainer_family"),
            "paper_grade_facmac": checkpoint_trainer_metadata.get("paper_grade_facmac"),
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "active_model_source": self._active_model_source,
            "model_ready": self._policy is not None,
            "replay_ready": len(self._replay) > 0,
            "replay_size": len(self._replay),
            "checkpoint": checkpoint,
            "checkpoint_metadata": checkpoint_metadata,
            "training_metadata": training_summary,
            "has_training_summary": self._last_training_summary is not None,
            "has_activation_summary": self._last_activation_summary is not None,
            "claim_boundary": trainer_metadata.get("claim_boundary"),
            "trainer_family": trainer_metadata.get("trainer_family"),
            "paper_grade_facmac": trainer_metadata.get("paper_grade_facmac"),
            "target_action": trainer_metadata.get("target_action"),
        }

    def model_snapshot(self) -> dict | None:
        if self._model is None:
            return None
        snapshot = self._model.to_dict()
        snapshot["metadata"] = self._trainer_metadata()
        snapshot["service_training"] = self.training_summary()
        snapshot["checkpoint"] = dict(self._checkpoint_summary)
        snapshot["metadata_provenance"] = "checkpoint" if self._active_model_source == "checkpoint_load" else "trainer"
        return snapshot

    def organization_snapshot(self) -> dict:
        return self._organization.snapshot()

    def schema_snapshot(self) -> dict:
        snapshot = schema_snapshot()
        snapshot["replay_summary_fields"] = [
            "replay_size",
            "governance_delta_rate",
            "guardrail_delta_rate",
            "divergence_rates",
            "reward_summary",
            "priority_summary",
        ]
        snapshot["training_summary_fields"] = [
            "last_training",
            "last_activation",
            "checkpoint",
            "model_ready",
            "active_model_source",
        ]
        snapshot["decision_fields"] = [
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
        ]
        snapshot["adaptive_snapshot_fields"] = list(ADAPTIVE_SNAPSHOT_FIELDS)
        snapshot["artifact_inventory_checkpoint_fields"] = sorted(ARTIFACT_INVENTORY_CHECKPOINT_FIELDS)
        snapshot["checkpoint_training_summary_fields"] = sorted(CHECKPOINT_TRAINING_SUMMARY_FIELDS)
        return snapshot

    def replay_summary(self) -> dict:
        samples = self._replay.snapshot()
        if not samples:
            return {
                "replay_size": 0,
                "governance_delta_rate": 0.0,
                "guardrail_delta_rate": 0.0,
                "divergence_rates": {
                "candidate_vs_governed": 0.0,
                "governed_vs_applied": 0.0,
                "candidate_vs_applied": 0.0,
                },
                "reward_summary": {
                    "reward_mean": 0.0,
                    "team_reward_mean": 0.0,
                    "role_reward_mean": 0.0,
                },
                "priority_summary": self._replay.priority_summary(),
            }
        replay_size = len(samples)
        governance_deltas = sum(1 for sample in samples if sample.governance_delta)
        guardrail_deltas = sum(1 for sample in samples if sample.guardrail_delta)
        governed_samples = [sample for sample in samples if sample.governed.present]
        candidate_vs_governed_count = sum(1 for sample in governed_samples if action_has_divergence(sample.candidate.action, sample.governed.action))
        governed_vs_applied_count = sum(1 for sample in governed_samples if action_has_divergence(sample.governed.action, sample.applied.action))
        candidate_vs_applied_count = sum(1 for sample in samples if action_has_divergence(sample.candidate.action, sample.applied.action))
        role_reward_means = [sum(float(value) for value in sample.role_rewards.values()) / len(sample.role_rewards) for sample in samples if sample.role_rewards]
        governed_count = len(governed_samples)
        return {
            "replay_size": replay_size,
            "governance_delta_rate": governance_deltas / replay_size,
            "guardrail_delta_rate": guardrail_deltas / replay_size,
            "divergence_rates": {
                "candidate_vs_governed": (candidate_vs_governed_count / governed_count) if governed_count else 0.0,
                "governed_vs_applied": (governed_vs_applied_count / governed_count) if governed_count else 0.0,
                "candidate_vs_applied": candidate_vs_applied_count / replay_size,
            },
            "reward_summary": {
                "reward_mean": sum(float(sample.reward) for sample in samples) / replay_size,
                "team_reward_mean": sum(float(sample.team_reward if sample.team_reward is not None else sample.reward) for sample in samples) / replay_size,
                "role_reward_mean": (sum(role_reward_means) / len(role_reward_means)) if role_reward_means else 0.0,
            },
            "priority_summary": self._replay.priority_summary(),
        }

    def inspect_replay_sample(self, index: int) -> dict:
        sample = self._replay.get(index)
        candidate, governed, applied = self._decision_flow_actions(sample.observation)
        current_flow = {
            "model_ready": self._policy is not None,
            "organization_decision": asdict(self._organization.evaluate(sample.observation)),
            "candidate": candidate.to_dict(),
            "governed": governed.to_dict(),
            "applied": applied.to_dict(),
        }
        recorded_flow = {
            "candidate_present": sample.candidate.present,
            "governed_present": sample.governed.present,
            "candidate": sample.candidate.action.to_dict(),
            "governed": sample.governed.action.to_dict() if sample.governed.present else None,
            "applied": sample.applied.action.to_dict(),
            "trace": {
                "enabled": sample.trace.enabled,
                "write_failed": sample.trace.write_failed,
                "close_failed": sample.trace.close_failed,
                "dropped_samples": sample.trace.dropped_samples,
            },
        }
        governed_divergence = None
        governed_divergence_fields = None
        if sample.governed.present:
            governed_divergence = action_has_divergence(sample.governed.action, governed)
            governed_divergence_fields = action_diff_fields(sample.governed.action, governed)
        return {
            "index": index,
            "recorded": recorded_flow,
            "current": current_flow,
            "divergence": {
                "recorded_candidate_vs_current_candidate": action_has_divergence(sample.candidate.action, candidate),
                "recorded_governed_vs_current_governed": governed_divergence,
                "recorded_applied_vs_current_applied": action_has_divergence(sample.applied.action, applied),
                "recorded_candidate_vs_current_candidate_fields": action_diff_fields(sample.candidate.action, candidate),
                "recorded_governed_vs_current_governed_fields": governed_divergence_fields,
                "recorded_applied_vs_current_applied_fields": action_diff_fields(sample.applied.action, applied),
            },
        }

    def adaptive_snapshot(self) -> dict:
        snapshot = {
            "model_ready": self._policy is not None,
            "model": self.model_snapshot(),
            "organization": self.organization_snapshot(),
            "schema": self.schema_snapshot(),
            "replay": self.replay_summary(),
            "training": self.training_summary(),
            "artifact_inventory": self.artifact_inventory(),
        }
        return {field: snapshot[field] for field in ADAPTIVE_SNAPSHOT_FIELDS}

    def _trainer_metadata(self) -> dict | None:
        if self._model is None:
            return None
        metadata = self._model.metadata or {}
        if not isinstance(metadata, dict):
            return None
        snapshot = {
            key: json.loads(json.dumps(value))
            for key, value in metadata.items()
            if key in EXPOSED_TRAINER_METADATA_FIELDS
        }
        if self._active_model_source == "checkpoint_load":
            trusted_claim_metadata = _trusted_checkpoint_claim_metadata(self._model)
            if trusted_claim_metadata is None:
                for field in ("trainer_family", "paper_grade_facmac", "claim_boundary"):
                    snapshot.pop(field, None)
            else:
                snapshot.update(trusted_claim_metadata)
        return snapshot

    def training_summary(self) -> dict:
        return {
            "last_training": dict(self._last_training_summary) if self._last_training_summary is not None else None,
            "last_activation": dict(self._last_activation_summary) if self._last_activation_summary is not None else None,
            "checkpoint": {
                **dict(self._checkpoint_summary),
                "active_checkpoint_path": self._checkpoint_summary["last_loaded_path"] if self._active_model_source == "checkpoint_load" else None,
            },
            "model_ready": self._policy is not None,
            "active_model_source": self._active_model_source,
        }

    def save_checkpoint(self, path: str | Path) -> dict:
        path = _validate_checkpoint_path(path, field_name="path")
        if self._model is None:
            return {"saved": False, "reason": "model_not_ready", "checkpoint": dict(self._checkpoint_summary)}
        relative_path = str(path.relative_to(WORKSPACE_ROOT))
        self._checkpoint_summary["last_saved_path"] = relative_path
        trainer_metadata = self._trainer_metadata() or {}
        reward_summary = trainer_metadata.get("reward_summary") if isinstance(trainer_metadata, dict) else None
        reward_mean = _parse_optional_finite_float(reward_summary.get("reward_mean") if isinstance(reward_summary, dict) else None)
        best_reward_mean = self._checkpoint_summary["best_reward_mean"]
        if reward_mean is not None and (best_reward_mean is None or reward_mean >= best_reward_mean):
            self._checkpoint_summary["best_saved_path"] = relative_path
            self._checkpoint_summary["best_reward_mean"] = reward_mean
        checkpoint_model = replace(
            self._model,
            metadata={
                **(self._model.metadata or {}),
                "checkpoint_provenance": _checkpoint_metadata_payload(self._checkpoint_summary, self._last_training_summary),
            },
        )
        save_checkpoint(checkpoint_model, path)
        return {"saved": True, "path": relative_path, "checkpoint": dict(self._checkpoint_summary), "training": self.training_summary()}

    def load_checkpoint(self, path: str | Path) -> dict:
        path = _validate_checkpoint_path(path, field_name="path")
        self._model = load_checkpoint(path)
        restored_checkpoint, restored_training = _restore_checkpoint_metadata(self._model)
        self._policy = SafeFACMACPolicy(self._model)
        self._active_model_source = "checkpoint_load"
        self._online_sample_calls = 0
        relative_path = str(path.relative_to(WORKSPACE_ROOT))
        self._checkpoint_summary["last_saved_path"] = None
        if restored_checkpoint is not None:
            self._checkpoint_summary["best_saved_path"] = restored_checkpoint.get("best_saved_path")
            self._checkpoint_summary["best_reward_mean"] = restored_checkpoint.get("best_reward_mean")
        else:
            self._checkpoint_summary["best_saved_path"] = None
            self._checkpoint_summary["best_reward_mean"] = None
        self._checkpoint_summary["last_loaded_path"] = relative_path
        self._last_activation_summary = {"mode": "checkpoint_load", "path": relative_path}
        if restored_training is not None:
            self._last_training_summary = restored_training
        else:
            self._last_training_summary = _fallback_checkpoint_training_summary(
                replay_size=len(self._replay),
                trainer_metadata=self._trainer_metadata(),
            )
        return {"loaded": True, "path": relative_path, "checkpoint": dict(self._checkpoint_summary), "training": self.training_summary()}
