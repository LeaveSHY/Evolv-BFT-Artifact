from __future__ import annotations

import json
import random
import urllib.request
from dataclasses import dataclass


AIOT_NAMED_SCENARIOS = {
    "heterogeneous_steady_state": {
        "heterogeneity_score": 0.65,
        "churn_rate": 0.15,
        "adversary_score": 0.1,
        "network_jitter_ms": 20.0,
        "ai_load_score": 0.45,
    },
    "churn_reconfiguration_pressure": {
        "heterogeneity_score": 0.75,
        "churn_rate": 0.5,
        "adversary_score": 0.25,
        "network_jitter_ms": 35.0,
        "ai_load_score": 0.6,
    },
    "adversarial_safety_stress": {
        "heterogeneity_score": 0.8,
        "churn_rate": 0.45,
        "adversary_score": 0.75,
        "network_jitter_ms": 70.0,
        "ai_load_score": 0.8,
    },
    "jitter_recovery_stress": {
        "heterogeneity_score": 0.55,
        "churn_rate": 0.25,
        "adversary_score": 0.2,
        "network_jitter_ms": 65.0,
        "ai_load_score": 0.55,
    },
    "ai_load_throughput_stress": {
        "heterogeneity_score": 0.6,
        "churn_rate": 0.2,
        "adversary_score": 0.15,
        "network_jitter_ms": 30.0,
        "ai_load_score": 0.9,
    },
}

AIOT_SCENARIO_SEQUENCE = [
    "heterogeneous_steady_state",
    "churn_reconfiguration_pressure",
    "adversarial_safety_stress",
    "jitter_recovery_stress",
    "ai_load_throughput_stress",
]


@dataclass
class AIoTScenarioDriver:
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def next_context(self, step: int) -> dict:
        return self.named_context(AIOT_SCENARIO_SEQUENCE[step % len(AIOT_SCENARIO_SEQUENCE)])

    def named_context(self, scenario_name: str) -> dict:
        if scenario_name not in AIOT_NAMED_SCENARIOS:
            raise ValueError(f"unsupported scenario: {scenario_name}")
        scenario = AIOT_NAMED_SCENARIOS[scenario_name]
        return {
            "scenario_name": scenario_name,
            "heterogeneity_score": scenario["heterogeneity_score"],
            "churn_rate": scenario["churn_rate"],
            "adversary_score": scenario["adversary_score"],
            "network_jitter_ms": scenario["network_jitter_ms"],
            "ai_load_score": scenario["ai_load_score"],
        }

    def _bounded(self, center: float, spread: float) -> float:
        value = center + (self._rng.random() * 2 - 1) * spread
        if center > 1:
            return max(0.0, value)
        return min(max(value, 0.0), 1.0)


@dataclass
class ScenarioPublisher:
    endpoint: str

    def publish(self, context: dict) -> None:
        body = json.dumps(context).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            resp.read()
