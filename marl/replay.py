from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, List

from marl.schemas import TrajectorySample


STRESS_ADVERSARY_THRESHOLD = 0.6
STRESS_CHURN_THRESHOLD = 0.4
STRESS_JITTER_THRESHOLD = 50.0


def has_stress_signal(sample: TrajectorySample) -> bool:
    observation = sample.observation
    return (
        observation.adversary_score >= STRESS_ADVERSARY_THRESHOLD
        or observation.churn_rate >= STRESS_CHURN_THRESHOLD
        or observation.network_jitter_ms >= STRESS_JITTER_THRESHOLD
    )


@dataclass
class ReplayBuffer:
    capacity: int
    _items: deque[TrajectorySample] = field(init=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._items = deque(maxlen=self.capacity)

    def push(self, sample: TrajectorySample) -> None:
        self._items.append(sample)

    def extend(self, samples: Iterable[TrajectorySample]) -> None:
        for sample in samples:
            self.push(sample)

    def sample(self, batch_size: int, seed: int | None = None, mode: str = "uniform") -> List[TrajectorySample]:
        if batch_size <= 0:
            return []
        rng = random.Random(seed)
        items = list(self._items)
        if mode == "uniform":
            if batch_size >= len(items):
                return items
            return rng.sample(items, batch_size)
        if mode != "priority":
            raise ValueError(f"unsupported replay sampling mode: {mode}")
        ordered = self._weighted_priority_order(items, rng)
        return ordered[: min(batch_size, len(ordered))]

    def priority_summary(self) -> dict[str, int]:
        items = list(self._items)
        return {
            "priority_sample_count": sum(1 for sample in items if self._priority_score(sample) > 0.0),
            "governance_delta_count": sum(1 for sample in items if sample.governance_delta),
            "guardrail_delta_count": sum(1 for sample in items if sample.guardrail_delta),
            "stress_signal_count": sum(1 for sample in items if has_stress_signal(sample)),
        }

    def snapshot(self) -> List[TrajectorySample]:
        return list(self._items)

    def get(self, index: int) -> TrajectorySample:
        if index < 0 or index >= len(self._items):
            raise IndexError(f"replay sample index out of range: {index}")
        return self.snapshot()[index]

    def _weighted_priority_order(self, items: list[TrajectorySample], rng: random.Random) -> list[TrajectorySample]:
        remaining = list(items)
        ordered: list[TrajectorySample] = []
        while remaining:
            weights = [1.0 + self._priority_score(sample) for sample in remaining]
            total = sum(weights)
            threshold = rng.random() * total
            cumulative = 0.0
            for index, weight in enumerate(weights):
                cumulative += weight
                if threshold <= cumulative:
                    ordered.append(remaining.pop(index))
                    break
        return ordered

    def _priority_score(self, sample: TrajectorySample) -> float:
        score = 0.0
        if sample.governance_delta:
            score += 2.0
        if sample.guardrail_delta:
            score += 4.0
        if has_stress_signal(sample):
            score += 3.0
        return score

    def __len__(self) -> int:
        return len(self._items)
