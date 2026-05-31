from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import List

from marl.scenario import AIoTScenarioDriver


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
class ScenarioRolloutRunner:
    evolvbft_base_url: str
    driver: AIoTScenarioDriver

    def run_steps(self, steps: int) -> List[dict]:
        snapshots: List[dict] = []
        for step in range(steps):
            context = self.driver.next_context(step)
            _post_json(f"{self.evolvbft_base_url}/adaptive/context", context)
            snapshots.append(_get_json(f"{self.evolvbft_base_url}/adaptive"))
        return snapshots
