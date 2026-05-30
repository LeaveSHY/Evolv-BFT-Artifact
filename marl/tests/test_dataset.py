import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from marl.dataset import action_diff_fields, action_has_divergence, load_trace_samples, build_training_batch
from marl.schemas import Action, AgentObservation, DecisionActionStage, Observation, TrajectorySample, SCHEMA_VERSION, runtime_trace_sample_from_dict


class DatasetTests(unittest.TestCase):
    def _write_single_sample(self, sample: dict) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "trace.jsonl"
        path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
        return path

    def test_load_trace_samples_uses_runtime_trace_adapter(self):
        path = self._write_single_sample(
            {
                "timestamp": "2026-04-01T00:00:00Z",
                "policy_name": "safe-baseline",
                "observation": {"validator_count": 8},
                "applied": {"action": {"pacemaker_timeout_ms": 1300}, "present": True},
                "reward": 0.5,
            }
        )

        with mock.patch("marl.dataset.runtime_trace_sample_from_dict", wraps=runtime_trace_sample_from_dict) as adapter:
            samples = load_trace_samples(path)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].schema_version, SCHEMA_VERSION)
        adapter.assert_called_once()

    def test_load_trace_samples_accepts_local_cluster_emitted_runtime_trace(self):
        path = self._write_single_sample(
            {
                "timestamp": "2026-04-11T00:00:00Z",
                "policy_name": "safe-baseline",
                "schema_version": "octopus-adaptive-v1",
                "observation": {
                    "validator_count": 4,
                    "global_confirmed_total": 0,
                    "committee_size": 4,
                },
                "candidate": {"present": True, "action": {"committee_size": 4}},
                "governed": {"present": True, "action": {"committee_size": 4}},
                "masked": {"present": True, "action": {"committee_size": 4}},
                "applied": {"present": True, "action": {"committee_size": 4}},
                "governance_delta": False,
                "guardrail_delta": False,
                "reward": 0.0,
                "team_reward": 0.0,
                "role_rewards": {},
                "trace": {"enabled": True, "dropped_samples": 0},
            }
        )

        with mock.patch("marl.dataset.runtime_trace_sample_from_dict", wraps=runtime_trace_sample_from_dict) as adapter:
            samples = load_trace_samples(path)

        adapter.assert_called_once()
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].schema_version, SCHEMA_VERSION)
        self.assertTrue(samples[0].trace.enabled)

    def test_load_trace_samples_and_build_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            sample = {
                "timestamp": "2026-04-01T00:00:00Z",
                "policy_name": "safe-baseline",
                "observation": {
                    "validator_count": 8,
                    "current_config_id": 1,
                    "highest_known_config_id": 2,
                    "committee_size": 0,
                    "pacemaker_timeout_ms": 1000,
                    "mempool_max_batch_txs": 2048,
                    "mempool_proposal_interval_ms": 100,
                    "throughput_tps": 5000,
                    "latency_p95_ms": 120,
                    "backlog_pending": 10,
                    "backlog_missing": 2,
                    "reject_total": 0,
                    "pending_joins": 1,
                    "pending_leaves": 0,
                    "lset_size": 3,
                    "can_participate": True,
                    "local_validator": False,
                    "heterogeneity_score": 0.6,
                    "churn_rate": 0.2,
                    "adversary_score": 0.1,
                    "network_jitter_ms": 15,
                    "ai_load_score": 0.5,
                    "agents": [
                        {
                            "instance_id": 0,
                            "committee_size": 4,
                            "pacemaker_timeout_ms": 1200,
                            "mempool_max_batch_txs": 1024,
                            "mempool_proposal_interval_ms": 80,
                            "validator_count": 8,
                            "epoch": 1,
                        }
                    ],
                },
                "candidate": {
                    "action": {
                        "committee_size": 2,
                        "pacemaker_timeout_ms": 100,
                        "mempool_max_batch_txs": 1024,
                        "mempool_proposal_interval_ms": 80,
                        "submit_join": True,
                        "hydra_discovery_target": 3,
                    },
                    "present": True,
                },
                "governed": {
                    "action": {
                        "committee_size": 2,
                        "pacemaker_timeout_ms": 100,
                        "mempool_max_batch_txs": 1024,
                        "mempool_proposal_interval_ms": 80,
                        "submit_join": False,
                        "hydra_discovery_target": 3,
                    },
                    "present": True,
                    "mutated": True,
                },
                "applied": {
                    "action": {
                        "committee_size": 4,
                        "pacemaker_timeout_ms": 1200,
                        "mempool_max_batch_txs": 1024,
                        "mempool_proposal_interval_ms": 80,
                        "submit_join": True,
                        "hydra_discovery_target": 3,
                    },
                    "present": True,
                    "mutated": True,
                },
                "governance_delta": True,
                "guardrail_delta": True,
                "reward": 1.5,
                "team_reward": 1.25,
                "role_rewards": {"lane_tuner": 0.8, "recovery_tuner": -0.2},
                "schema_version": "octopus-adaptive-v1",
            }
            path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

            samples = load_trace_samples(path)
            self.assertEqual(len(samples), 1)
            self.assertEqual(len(samples[0].observation.agents), 1)
            self.assertTrue(samples[0].governance_delta)
            self.assertTrue(samples[0].guardrail_delta)
            self.assertEqual(samples[0].candidate.action.pacemaker_timeout_ms, 100)
            self.assertEqual(samples[0].governed.action.submit_join, False)
            self.assertEqual(samples[0].schema_version, "octopus-adaptive-v1")

            batch = build_training_batch(samples)
            self.assertEqual(batch.features.shape[0], 1)
            self.assertEqual(batch.actions.shape[1], 7)
            self.assertEqual(float(batch.team_rewards[0]), 1.25)
            self.assertEqual(float(batch.governance_deltas[0]), 1.0)
            self.assertEqual(float(batch.guardrail_deltas[0]), 1.0)
            self.assertEqual(float(batch.candidate_actions[0][4]), 1.0)
            self.assertEqual(float(batch.governed_actions[0][4]), 0.0)
            self.assertEqual(float(batch.actions[0][4]), 0.0)

    def test_build_training_batch_derives_transition_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            first = {
                "timestamp": "2026-04-01T00:00:00Z",
                "policy_name": "safe-baseline",
                "observation": {"validator_count": 4, "throughput_tps": 1.0},
                "applied": {"action": {"pacemaker_timeout_ms": 1000}, "present": True},
                "reward": 1.0,
            }
            second = {
                "timestamp": "2026-04-01T00:00:01Z",
                "policy_name": "safe-baseline",
                "observation": {"validator_count": 4, "throughput_tps": 2.0},
                "applied": {"action": {"pacemaker_timeout_ms": 900}, "present": True},
                "reward": 2.0,
                "team_reward": 2.5,
                "done": True,
            }
            path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8")
            traces = load_trace_samples(path)
            batch = build_training_batch(traces)
            self.assertEqual(batch.next_features.shape[0], 2)
            self.assertEqual(batch.team_rewards.shape[0], 2)
            self.assertEqual(batch.dones.shape[0], 2)
            self.assertEqual(float(batch.dones[1]), 1.0)
            self.assertEqual(float(batch.team_rewards[1]), 2.5)

    def test_build_training_batch_uses_self_transition_when_next_observation_missing(self):
        samples = [
            load_trace_samples(self._write_single_sample({
                "timestamp": "2026-04-01T00:00:00Z",
                "policy_name": "safe-baseline",
                "observation": {"validator_count": 4, "throughput_tps": 1.0},
                "applied": {"action": {"pacemaker_timeout_ms": 1000}, "present": True},
                "reward": 1.0,
            }))[0],
            load_trace_samples(self._write_single_sample({
                "timestamp": "2026-04-01T00:00:01Z",
                "policy_name": "safe-baseline",
                "observation": {"validator_count": 4, "throughput_tps": 2.0},
                "applied": {"action": {"pacemaker_timeout_ms": 900}, "present": True},
                "reward": 2.0,
            }))[0],
        ]
        sequential_batch = build_training_batch(samples)
        self.assertEqual(float(sequential_batch.next_features[0][8]), 2.0)
        self.assertEqual(float(sequential_batch.next_features[1][8]), 2.0)
        self.assertEqual(float(sequential_batch.dones[0]), 0.0)
        self.assertEqual(float(sequential_batch.dones[1]), 0.0)

        replay_batch = build_training_batch(samples, infer_sequential_next_observation=False)
        self.assertEqual(float(replay_batch.next_features[0][8]), 1.0)
        self.assertEqual(float(replay_batch.next_features[1][8]), 2.0)
        self.assertEqual(float(replay_batch.dones[0]), 0.0)
        self.assertEqual(float(replay_batch.dones[1]), 0.0)

    def test_build_training_batch_prefers_governed_agent_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            sample = {
                "timestamp": "2026-04-01T00:00:00Z",
                "policy_name": "safe-baseline",
                "observation": {
                    "validator_count": 8,
                    "agents": [
                        {
                            "instance_id": 0,
                            "validator_count": 8,
                            "committee_size": 4,
                            "pacemaker_timeout_ms": 1200,
                            "mempool_max_batch_txs": 512,
                            "mempool_proposal_interval_ms": 80,
                        }
                    ],
                },
                "applied": {
                    "action": {
                        "agent_actions": [
                            {
                                "instance_id": 0,
                                "committee_size": 7,
                                "pacemaker_timeout_ms": 1700,
                                "mempool_max_batch_txs": 1536,
                                "mempool_proposal_interval_ms": 95,
                            }
                        ]
                    },
                    "present": True,
                },
                "governed": {
                    "action": {
                        "agent_actions": [
                            {
                                "instance_id": 0,
                                "committee_size": 4,
                                "pacemaker_timeout_ms": 1200,
                                "mempool_max_batch_txs": 512,
                                "mempool_proposal_interval_ms": 80,
                            }
                        ]
                    },
                    "present": True,
                    "mutated": True,
                },
                "governance_delta": True,
                "reward": 1.0,
            }
            path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            traces = load_trace_samples(path)
            batch = build_training_batch(traces)
            self.assertEqual(batch.agent_features.shape, (1, 7))
            self.assertEqual(float(batch.agent_features[0][0]), 0.0)
            self.assertEqual(float(batch.agent_features[0][1]), 0.0)
            self.assertEqual(float(batch.agent_features[0][2]), 8.0)
            self.assertEqual(float(batch.agent_features[0][3]), 4.0)
            self.assertEqual(float(batch.agent_features[0][4]), 1200.0)
            self.assertEqual(float(batch.agent_features[0][5]), 512.0)
            self.assertEqual(float(batch.agent_features[0][6]), 80.0)
            self.assertEqual(batch.agent_actions.shape[0], 1)
            self.assertEqual(float(batch.agent_actions[0][0]), 4.0)
            self.assertEqual(float(batch.agent_actions[0][1]), 1200.0)
            self.assertEqual(float(batch.agent_actions[0][2]), 512.0)
            self.assertEqual(float(batch.agent_actions[0][3]), 80.0)

    def test_build_training_batch_does_not_fall_back_when_governed_agent_actions_are_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            sample = {
                "timestamp": "2026-04-01T00:00:00Z",
                "policy_name": "safe-baseline",
                "observation": {
                    "validator_count": 8,
                    "agents": [
                        {
                            "instance_id": 0,
                            "validator_count": 8,
                            "committee_size": 4,
                            "pacemaker_timeout_ms": 1200,
                            "mempool_max_batch_txs": 512,
                            "mempool_proposal_interval_ms": 80,
                        }
                    ],
                },
                "applied": {
                    "action": {
                        "agent_actions": [
                            {
                                "instance_id": 0,
                                "committee_size": 7,
                                "pacemaker_timeout_ms": 1700,
                                "mempool_max_batch_txs": 1536,
                                "mempool_proposal_interval_ms": 95,
                            }
                        ]
                    },
                    "present": True,
                },
                "governed": {"action": {"agent_actions": []}, "present": True, "mutated": True},
                "governance_delta": True,
                "reward": 1.0,
            }
            path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            traces = load_trace_samples(path)
            batch = build_training_batch(traces)
            self.assertEqual(batch.agent_actions.shape[0], 0)

    def test_load_trace_samples_preserves_explicit_default_governed_stage_presence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            sample = {
                "timestamp": "2026-04-01T00:00:00Z",
                "policy_name": "safe-baseline",
                "observation": {"validator_count": 8},
                "applied": {"action": {"pacemaker_timeout_ms": 900}, "present": True},
                "governed": {"action": {}, "present": True},
                "reward": 1.0,
            }
            path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            traces = load_trace_samples(path)
            self.assertTrue(traces[0].governed.present)
            batch = build_training_batch(traces)
            self.assertEqual(float(batch.actions[0][1]), 1000.0)
            self.assertEqual(float(batch.governed_actions[0][1]), 1000.0)

    def test_build_training_batch_exposes_candidate_stage_actions(self):
        sample = TrajectorySample(
            policy_name="safe-baseline",
            observation=Observation(validator_count=8),
            candidate=DecisionActionStage(action=Action(submit_join=True), present=True),
            governed=DecisionActionStage(action=Action(submit_join=False), present=True),
            applied=DecisionActionStage(action=Action(submit_join=False), present=True),
            governance_delta=True,
            reward=1.0,
        )
        batch = build_training_batch([sample])
        self.assertEqual(float(batch.candidate_actions[0][4]), 1.0)
        self.assertEqual(float(batch.governed_actions[0][4]), 0.0)
        self.assertEqual(float(batch.actions[0][4]), 0.0)

    def test_build_training_batch_handles_empty_samples(self):
        batch = build_training_batch([])
        self.assertEqual(batch.features.shape, (0, 28))
        self.assertEqual(batch.next_features.shape, (0, 28))
        self.assertEqual(batch.actions.shape, (0, 7))
        self.assertEqual(batch.candidate_actions.shape, (0, 7))
        self.assertEqual(batch.governed_actions.shape, (0, 7))
        self.assertEqual(batch.agent_features.shape, (0, 7))
        self.assertEqual(batch.agent_actions.shape, (0, 4))
        self.assertEqual(batch.rewards.shape, (0,))
        self.assertEqual(batch.team_rewards.shape, (0,))
        self.assertEqual(batch.dones.shape, (0,))

    def test_build_training_batch_preserves_agent_feature_width(self):
        sample = TrajectorySample(
            policy_name="safe-baseline",
            observation=Observation(
                validator_count=8,
                agents=[
                    AgentObservation(
                        instance_id=0,
                        epoch=2,
                        validator_count=8,
                        committee_size=4,
                        pacemaker_timeout_ms=1200,
                        mempool_max_batch_txs=512,
                        mempool_proposal_interval_ms=80,
                    )
                ],
            ),
            governed=DecisionActionStage(
                action=Action(
                    agent_actions=[
                        {
                            "instance_id": 0,
                            "committee_size": 4,
                            "pacemaker_timeout_ms": 1200,
                            "mempool_max_batch_txs": 512,
                            "mempool_proposal_interval_ms": 80,
                        }
                    ]
                ),
                present=True,
            ),
            applied=DecisionActionStage(present=True),
            reward=1.0,
        )

        batch = build_training_batch([sample])

        self.assertEqual(batch.agent_features.shape, (1, 7))
        self.assertEqual(float(batch.agent_features[0][0]), 0.0)
        self.assertEqual(float(batch.agent_features[0][1]), 2.0)
        self.assertEqual(float(batch.agent_features[0][6]), 80.0)
        self.assertEqual(batch.agent_actions.shape, (1, 4))

    def test_load_trace_samples_rejects_unsupported_schema_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            sample = {
                "timestamp": "2026-04-01T00:00:00Z",
                "policy_name": "safe-baseline",
                "observation": {"validator_count": 4},
                "applied": {"action": {"pacemaker_timeout_ms": 1000}, "present": True},
                "reward": 1.0,
                "schema_version": "octopus-adaptive-v0",
            }
            path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported trace schema_version"):
                load_trace_samples(path)

    def test_load_trace_samples_rejects_unknown_observation_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            sample = {
                "timestamp": "2026-04-01T00:00:00Z",
                "policy_name": "safe-baseline",
                "observation": {"validator_count": 4, "unexpected": 1},
                "applied": {"action": {"pacemaker_timeout_ms": 1000}, "present": True},
                "reward": 1.0,
                "schema_version": "octopus-adaptive-v1",
            }
            path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown field at observation\.unexpected"):
                load_trace_samples(path)

    def test_load_trace_samples_rejects_unknown_stage_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            sample = {
                "timestamp": "2026-04-01T00:00:00Z",
                "policy_name": "safe-baseline",
                "observation": {"validator_count": 4},
                "candidate": {"action": {"pacemaker_timeout_ms": 900}, "present": True, "unexpected": 1},
                "applied": {"action": {"pacemaker_timeout_ms": 1000}, "present": True},
                "reward": 1.0,
                "schema_version": "octopus-adaptive-v1",
            }
            path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown field at candidate\.unexpected"):
                load_trace_samples(path)

    def test_load_trace_samples_rejects_unknown_action_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            sample = {
                "timestamp": "2026-04-01T00:00:00Z",
                "policy_name": "safe-baseline",
                "observation": {"validator_count": 4},
                "candidate": {"action": {"pacemaker_timeout_ms": 900, "unexpected": 1}, "present": True},
                "applied": {"action": {"pacemaker_timeout_ms": 1000}, "present": True},
                "reward": 1.0,
                "schema_version": "octopus-adaptive-v1",
            }
            path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown field at candidate\.action\.unexpected"):
                load_trace_samples(path)

    def test_load_trace_samples_rejects_unknown_agent_action_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            sample = {
                "timestamp": "2026-04-01T00:00:00Z",
                "policy_name": "safe-baseline",
                "observation": {"validator_count": 4},
                "candidate": {
                    "action": {
                        "agent_actions": [
                            {
                                "instance_id": 0,
                                "committee_size": 4,
                                "unexpected": 1,
                            }
                        ]
                    },
                    "present": True,
                },
                "applied": {"action": {"pacemaker_timeout_ms": 1000}, "present": True},
                "reward": 1.0,
                "schema_version": "octopus-adaptive-v1",
            }
            path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown field at candidate\.action\.agent_actions\[0\]\.unexpected"):
                load_trace_samples(path)

    def test_action_diff_helpers_detect_top_level_and_agent_divergence(self):
        left = Action(
            committee_size=4,
            pacemaker_timeout_ms=1000,
            agent_actions=[{"instance_id": 0, "committee_size": 4}],
        )
        right = Action(
            committee_size=6,
            pacemaker_timeout_ms=1000,
            agent_actions=[{"instance_id": 0, "committee_size": 5}],
        )
        diff_fields = action_diff_fields(left, right)
        self.assertIn("committee_size", diff_fields)
        self.assertIn("agent_actions", diff_fields)
        self.assertTrue(action_has_divergence(left, right))

    def test_action_diff_helpers_ignore_identical_actions(self):
        action = Action(committee_size=4, pacemaker_timeout_ms=1000)
        self.assertEqual(action_diff_fields(action, action), [])
        self.assertFalse(action_has_divergence(action, action))


if __name__ == "__main__":
    unittest.main()
