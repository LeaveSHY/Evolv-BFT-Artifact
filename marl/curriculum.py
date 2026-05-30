from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CurriculumPhase:
    name: str
    steps: int
    heterogeneity: float
    churn: float
    adversary: float
    jitter_ms: float
    ai_load: float

    def context(self) -> dict:
        return {
            "phase": self.name,
            "heterogeneity_score": self.heterogeneity,
            "churn_rate": self.churn,
            "adversary_score": self.adversary,
            "network_jitter_ms": self.jitter_ms,
            "ai_load_score": self.ai_load,
        }


@dataclass
class CurriculumSchedule:
    phases: list[CurriculumPhase]

    def context_for_step(self, step: int) -> dict:
        if not self.phases:
            return {}
        remaining = step
        for phase in self.phases:
            if remaining < phase.steps:
                return phase.context()
            remaining -= phase.steps
        return self.phases[-1].context()
