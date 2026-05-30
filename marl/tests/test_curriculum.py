import unittest

from marl.curriculum import CurriculumPhase, CurriculumSchedule


class CurriculumTests(unittest.TestCase):
    def test_schedule_transitions_between_phases(self):
        schedule = CurriculumSchedule(
            phases=[
                CurriculumPhase(name="warmup", steps=2, heterogeneity=0.2, churn=0.1, adversary=0.0, jitter_ms=10, ai_load=0.3),
                CurriculumPhase(name="stress", steps=3, heterogeneity=0.8, churn=0.4, adversary=0.7, jitter_ms=60, ai_load=0.9),
            ]
        )

        ctx0 = schedule.context_for_step(0)
        ctx2 = schedule.context_for_step(2)
        ctx4 = schedule.context_for_step(4)

        self.assertEqual(ctx0["phase"], "warmup")
        self.assertEqual(ctx2["phase"], "stress")
        self.assertEqual(ctx4["phase"], "stress")
        self.assertGreater(ctx2["adversary_score"], ctx0["adversary_score"])


if __name__ == "__main__":
    unittest.main()
