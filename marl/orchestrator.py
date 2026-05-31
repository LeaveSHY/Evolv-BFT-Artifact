from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from marl.curriculum import CurriculumSchedule
from marl.schemas import TrajectorySample


def _post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8") or "{}")


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8") or "{}")


@dataclass
class ExperimentOrchestrator:
    evolvbft_base_url: str
    marl_service_url: str
    curriculum: CurriculumSchedule
    checkpoint_dir: Path

    def run(self, steps: int, train_every: int, checkpoint_every: int) -> dict:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for step in range(steps):
            context = self.curriculum.context_for_step(step)
            _post_json(f"{self.evolvbft_base_url}/adaptive/context", context)
            adaptive_snapshot = _get_json(f"{self.evolvbft_base_url}/adaptive")
            last_decision = adaptive_snapshot.get("last_decision", {})
            observation = last_decision.get("observation") or context
            candidate_stage = last_decision.get("candidate") or {}
            governed_stage = last_decision.get("governed") or {}
            applied_stage = last_decision.get("applied") or {}
            trace_status = last_decision.get("trace") or {}
            action = applied_stage.get("action") or {}
            reward = float(last_decision.get("reward", 0.0))
            governance_delta = bool(last_decision.get("governed", {}).get("mutated", False))
            guardrail_delta = bool(last_decision.get("masked", {}).get("mutated", False))
            trace_payload = {
                "policy_name": last_decision.get("policy_name", "unknown"),
                "observation": observation,
                "candidate": candidate_stage,
                "governed": governed_stage,
                "masked": last_decision.get("masked") or {},
                "applied": applied_stage,
                "reward": reward,
                "next_observation": last_decision.get("next_observation"),
                "done": bool(last_decision.get("done", False)),
                "team_reward": last_decision.get("team_reward"),
                "role_rewards": last_decision.get("role_rewards") or {},
                "governance_delta": governance_delta,
                "guardrail_delta": guardrail_delta,
                "schema_version": adaptive_snapshot.get("schema_version"),
                "trace": trace_status,
            }
            _post_json(f"{self.marl_service_url}/trace/ingest", trace_payload)

            if train_every > 0 and (step + 1) % train_every == 0:
                _post_json(f"{self.marl_service_url}/train/online", {"batch_size": train_every})

            if checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint-step-{step+1}.json"
                checkpoint_payload_path = checkpoint_path.name if checkpoint_path.is_absolute() else str(checkpoint_path)
                _post_json(f"{self.marl_service_url}/checkpoint/save", {"path": checkpoint_payload_path})

        return {"steps": steps}
