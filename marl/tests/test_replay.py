import unittest

from marl.replay import ReplayBuffer
from marl.schemas import Observation, Action, DecisionActionStage, TrajectorySample


class ReplayBufferTests(unittest.TestCase):
    def _sample(self, *, throughput_tps: float, governance_delta: bool = False, guardrail_delta: bool = False, adversary_score: float = 0.0, churn_rate: float = 0.0, network_jitter_ms: float = 0.0) -> TrajectorySample:
        action = DecisionActionStage(
            action=Action(pacemaker_timeout_ms=1000 + int(throughput_tps)),
            present=True,
        )
        return TrajectorySample(
            policy_name="safe-facmac",
            observation=Observation(
                validator_count=8,
                throughput_tps=throughput_tps,
                adversary_score=adversary_score,
                churn_rate=churn_rate,
                network_jitter_ms=network_jitter_ms,
            ),
            candidate=action,
            governed=action,
            masked=action,
            applied=action,
            reward=float(throughput_tps),
            governance_delta=governance_delta,
            guardrail_delta=guardrail_delta,
        )
    def test_push_and_sample(self):
        buf = ReplayBuffer(capacity=3)
        for i in range(5):
            buf.push(
                TrajectorySample(
                    policy_name="safe-facmac",
                    observation=Observation(validator_count=8, throughput_tps=1000 + i),
                    candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1000 + i), present=True),
                    governed=DecisionActionStage(action=Action(pacemaker_timeout_ms=1000 + i), present=True),
                    masked=DecisionActionStage(action=Action(pacemaker_timeout_ms=1000 + i), present=True),
                    applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1000 + i), present=True),
                    reward=float(i),
                )
            )

        self.assertEqual(len(buf), 3)
        batch = buf.sample(batch_size=2, seed=7)
        self.assertEqual(len(batch), 2)
        self.assertTrue(all(isinstance(item, TrajectorySample) for item in batch))

    def test_snapshot_and_get_expose_public_accessors(self):
        buf = ReplayBuffer(capacity=3)
        for i in range(3):
            buf.push(
                TrajectorySample(
                    policy_name="safe-facmac",
                    observation=Observation(validator_count=8, throughput_tps=1000 + i),
                    candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1000 + i), present=True),
                    governed=DecisionActionStage(action=Action(pacemaker_timeout_ms=1000 + i), present=True),
                    masked=DecisionActionStage(action=Action(pacemaker_timeout_ms=1000 + i), present=True),
                    applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1000 + i), present=True),
                    reward=float(i),
                )
            )

        snapshot = buf.snapshot()
        self.assertEqual(len(snapshot), 3)
        self.assertIsInstance(snapshot[0], TrajectorySample)
        self.assertEqual(buf.get(1).applied.action.pacemaker_timeout_ms, 1001)

    def test_priority_sample_prefers_stressed_samples(self):
        buf = ReplayBuffer(capacity=4)
        calm_sample = self._sample(throughput_tps=1.0)
        governance_sample = self._sample(throughput_tps=2.0, governance_delta=True)
        guardrail_sample = self._sample(throughput_tps=3.0, guardrail_delta=True)
        adversarial_sample = self._sample(
            throughput_tps=4.0,
            adversary_score=0.8,
            churn_rate=0.5,
            network_jitter_ms=60.0,
        )
        buf.extend([calm_sample, governance_sample, guardrail_sample, adversarial_sample])

        batch = buf.sample(batch_size=2, seed=7, mode="priority")

        self.assertEqual(len(batch), 2)
        self.assertIn(batch[0].observation.throughput_tps, {3.0, 4.0})
        self.assertNotIn(1.0, {item.observation.throughput_tps for item in batch})

    def test_priority_sample_is_deterministic_with_seed(self):
        buf = ReplayBuffer(capacity=5)
        buf.extend(
            [
                self._sample(throughput_tps=1.0),
                self._sample(throughput_tps=2.0, governance_delta=True),
                self._sample(throughput_tps=3.0, guardrail_delta=True),
                self._sample(throughput_tps=4.0, adversary_score=0.7),
                self._sample(throughput_tps=5.0, churn_rate=0.6, network_jitter_ms=80.0),
            ]
        )

        first = buf.sample(batch_size=3, seed=11, mode="priority")
        second = buf.sample(batch_size=3, seed=11, mode="priority")

        self.assertEqual(
            [item.observation.throughput_tps for item in first],
            [item.observation.throughput_tps for item in second],
        )

    def test_priority_summary_counts_priority_sources(self):
        buf = ReplayBuffer(capacity=4)
        buf.extend(
            [
                self._sample(throughput_tps=1.0),
                self._sample(throughput_tps=2.0, governance_delta=True),
                self._sample(throughput_tps=3.0, guardrail_delta=True),
                self._sample(throughput_tps=4.0, adversary_score=0.8, churn_rate=0.5, network_jitter_ms=60.0),
            ]
        )

        summary = buf.priority_summary()

        self.assertEqual(summary["priority_sample_count"], 3)
        self.assertEqual(summary["governance_delta_count"], 1)
        self.assertEqual(summary["guardrail_delta_count"], 1)
        self.assertEqual(summary["stress_signal_count"], 1)

    def test_priority_sample_orders_full_buffer_without_disabling_priority(self):
        buf = ReplayBuffer(capacity=4)
        buf.extend(
            [
                self._sample(throughput_tps=1.0),
                self._sample(throughput_tps=2.0, governance_delta=True),
                self._sample(throughput_tps=3.0, guardrail_delta=True),
                self._sample(throughput_tps=4.0, adversary_score=0.8, churn_rate=0.5, network_jitter_ms=60.0),
            ]
        )

        ordered = buf.sample(batch_size=4, seed=7, mode="priority")

        self.assertEqual(len(ordered), 4)
        self.assertIn(ordered[0].observation.throughput_tps, {3.0, 4.0})
        self.assertEqual({item.observation.throughput_tps for item in ordered}, {1.0, 2.0, 3.0, 4.0})


if __name__ == "__main__":
    unittest.main()
