import json
import tempfile
import unittest
from pathlib import Path

from marl.policy import SafeFACMACPolicy
from marl.schemas import AgentObservation, Observation, Action, DecisionActionStage, TrajectorySample
from marl.trainer import FEATURE_DIM, MAX_CHECKPOINT_BYTES, SafeFACMACTrainer, load_checkpoint, save_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_save_and_load_checkpoint_roundtrip(self):
        samples = [
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(
                    current_config_id=1,
                    highest_known_config_id=2,
                    validator_count=8,
                    throughput_tps=5000,
                    latency_p95_ms=120,
                    can_participate=True,
                    local_validator=False,
                    agents=[
                        AgentObservation(
                            instance_id=0,
                            validator_count=8,
                            committee_size=4,
                            pacemaker_timeout_ms=1200,
                            mempool_max_batch_txs=512,
                            mempool_proposal_interval_ms=80,
                        )
                    ],
                ),
                candidate=DecisionActionStage(
                    action=Action(
                        pacemaker_timeout_ms=1200,
                        mempool_max_batch_txs=1024,
                        submit_join=True,
                        hydra_discovery_target=3,
                        agent_actions=[
                            {
                                "instance_id": 0,
                                "committee_size": 4,
                                "pacemaker_timeout_ms": 1100,
                                "mempool_max_batch_txs": 768,
                                "mempool_proposal_interval_ms": 70,
                            }
                        ],
                    ),
                    present=True,
                ),
                governed=DecisionActionStage(
                    action=Action(
                        pacemaker_timeout_ms=1200,
                        mempool_max_batch_txs=1024,
                        submit_join=True,
                        hydra_discovery_target=3,
                        agent_actions=[
                            {
                                "instance_id": 0,
                                "committee_size": 4,
                                "pacemaker_timeout_ms": 1100,
                                "mempool_max_batch_txs": 768,
                                "mempool_proposal_interval_ms": 70,
                            }
                        ],
                    ),
                    present=True,
                ),
                masked=DecisionActionStage(
                    action=Action(
                        pacemaker_timeout_ms=1200,
                        mempool_max_batch_txs=1024,
                        submit_join=True,
                        hydra_discovery_target=3,
                        agent_actions=[
                            {
                                "instance_id": 0,
                                "committee_size": 4,
                                "pacemaker_timeout_ms": 1100,
                                "mempool_max_batch_txs": 768,
                                "mempool_proposal_interval_ms": 70,
                            }
                        ],
                    ),
                    present=True,
                ),
                applied=DecisionActionStage(
                    action=Action(
                        pacemaker_timeout_ms=1200,
                        mempool_max_batch_txs=1024,
                        submit_join=True,
                        hydra_discovery_target=3,
                        agent_actions=[
                            {
                                "instance_id": 0,
                                "committee_size": 4,
                                "pacemaker_timeout_ms": 1100,
                                "mempool_max_batch_txs": 768,
                                "mempool_proposal_interval_ms": 70,
                            }
                        ],
                    ),
                    present=True,
                ),
                reward=1.0,
                role_rewards={"membership_tuner": 2.0},
                team_reward=1.25,
            )
        ]
        trainer = SafeFACMACTrainer()
        model = trainer.fit(samples)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            save_checkpoint(model, path)
            loaded = load_checkpoint(path)

            self.assertEqual(model.to_dict(), loaded.to_dict())
            obs = samples[0].observation
            action_a = SafeFACMACPolicy(model).decide(obs)
            action_b = SafeFACMACPolicy(loaded).decide(obs)
            self.assertEqual(action_a.to_dict(), action_b.to_dict())
            self.assertIn("membership_tuner", loaded.role_actor_bias)
            self.assertEqual(len(loaded.agent_actor_bias), 4)
            self.assertIn("role_head_coverage", loaded.metadata)
            self.assertEqual(loaded.metadata["trainer_config"]["seed"], 7)
            self.assertEqual(loaded.metadata["trainer_config"]["ridge"], 1e-3)
            self.assertEqual(loaded.metadata["trainer_config"]["epochs"], 24)

    def test_load_checkpoint_rejects_missing_required_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            path.write_text('{"actor_weights": [], "actor_bias": []}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required checkpoint fields"):
                load_checkpoint(path)

    def test_load_checkpoint_rejects_non_object_role_actor_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            path.write_text(
                json.dumps(
                    {
                        "actor_weights": [[0.0] * FEATURE_DIM for _ in range(7)],
                        "actor_bias": [0.0] * 7,
                        "critic_weights": [0.0] * FEATURE_DIM,
                        "critic_bias": 0.0,
                        "role_actor_weights": [],
                        "role_actor_bias": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "checkpoint role actor fields must be objects"):
                load_checkpoint(path)

    def test_load_checkpoint_rejects_actor_weight_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            path.write_text(
                json.dumps(
                    {
                        "actor_weights": [[0.0] * FEATURE_DIM for _ in range(6)],
                        "actor_bias": [0.0] * 7,
                        "critic_weights": [0.0] * FEATURE_DIM,
                        "critic_bias": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "checkpoint field actor_weights must have 7 rows"):
                load_checkpoint(path)

    def test_load_checkpoint_rejects_non_finite_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            path.write_text(
                json.dumps(
                    {
                        "actor_weights": [[0.0] * FEATURE_DIM for _ in range(7)],
                        "actor_bias": [0.0] * 7,
                        "critic_weights": [0.0] * (FEATURE_DIM - 1) + [float("nan")],
                        "critic_bias": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "checkpoint field critic_weights\\[27\\] must be finite"):
                load_checkpoint(path)

    def test_load_checkpoint_rejects_agent_head_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            path.write_text(
                json.dumps(
                    {
                        "actor_weights": [[0.0] * FEATURE_DIM for _ in range(7)],
                        "actor_bias": [0.0] * 7,
                        "critic_weights": [0.0] * FEATURE_DIM,
                        "critic_bias": 0.0,
                        "agent_actor_weights": [[0.0] * 7 for _ in range(3)],
                        "agent_actor_bias": [0.0] * 4,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "checkpoint field agent_actor_weights must have 4 rows"):
                load_checkpoint(path)

    def test_load_checkpoint_rejects_role_head_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            path.write_text(
                json.dumps(
                    {
                        "actor_weights": [[0.0] * FEATURE_DIM for _ in range(7)],
                        "actor_bias": [0.0] * 7,
                        "critic_weights": [0.0] * FEATURE_DIM,
                        "critic_bias": 0.0,
                        "role_actor_weights": {
                            "recovery_tuner": [[0.0] * FEATURE_DIM, [0.0] * FEATURE_DIM]
                        },
                        "role_actor_bias": {
                            "recovery_tuner": [0.0, 0.0]
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "checkpoint field role_actor_weights\\.recovery_tuner must have 1 rows"):
                load_checkpoint(path)

    def test_load_checkpoint_rejects_unsupported_role_heads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            path.write_text(
                json.dumps(
                    {
                        "actor_weights": [[0.0] * FEATURE_DIM for _ in range(7)],
                        "actor_bias": [0.0] * 7,
                        "critic_weights": [0.0] * FEATURE_DIM,
                        "critic_bias": 0.0,
                        "role_actor_weights": {
                            "rogue": [[0.0] * FEATURE_DIM]
                        },
                        "role_actor_bias": {
                            "rogue": [0.0]
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "checkpoint role actor fields contain unsupported roles: rogue"):
                load_checkpoint(path)

    def test_load_checkpoint_rejects_non_object_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            path.write_text(
                json.dumps(
                    {
                        "actor_weights": [[0.0] * FEATURE_DIM for _ in range(7)],
                        "actor_bias": [0.0] * 7,
                        "critic_weights": [0.0] * FEATURE_DIM,
                        "critic_bias": 0.0,
                        "metadata": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "checkpoint metadata must be an object"):
                load_checkpoint(path)

    def test_load_checkpoint_rejects_oversized_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            path.write_text("x" * (MAX_CHECKPOINT_BYTES + 1), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checkpoint file exceeds max size"):
                load_checkpoint(path)

    def test_loaded_checkpoint_ignores_malicious_role_head_coverage_metadata(self):
        model = SafeFACMACTrainer().fit(
            [
                TrajectorySample(
                    policy_name="safe-facmac",
                    observation=Observation(
                        current_config_id=1,
                        highest_known_config_id=1,
                        validator_count=8,
                        committee_size=4,
                        pacemaker_timeout_ms=900,
                        mempool_max_batch_txs=1500,
                        mempool_proposal_interval_ms=75,
                        can_participate=True,
                        local_validator=True,
                    ),
                    candidate=DecisionActionStage(
                        action=Action(
                            committee_size=6,
                            mempool_max_batch_txs=1500,
                            mempool_proposal_interval_ms=75,
                        ),
                        present=True,
                    ),
                    governed=DecisionActionStage(
                        action=Action(
                            committee_size=6,
                            mempool_max_batch_txs=1500,
                            mempool_proposal_interval_ms=75,
                        ),
                        present=True,
                    ),
                    masked=DecisionActionStage(
                        action=Action(
                            committee_size=6,
                            mempool_max_batch_txs=1500,
                            mempool_proposal_interval_ms=75,
                        ),
                        present=True,
                    ),
                    applied=DecisionActionStage(
                        action=Action(
                            committee_size=6,
                            mempool_max_batch_txs=1500,
                            mempool_proposal_interval_ms=75,
                        ),
                        present=True,
                    ),
                    reward=1.0,
                )
            ]
        )
        model.role_actor_weights = {"lane_tuner": [[0.0] * FEATURE_DIM for _ in range(3)]}
        model.role_actor_bias = {"lane_tuner": [7.0, 1800.0, 45.0]}
        payload = model.to_dict()
        payload["metadata"] = {"role_head_coverage": "malicious"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_checkpoint(path)
            action = SafeFACMACPolicy(loaded).decide(
                Observation(
                    current_config_id=1,
                    highest_known_config_id=1,
                    validator_count=8,
                    committee_size=6,
                    pacemaker_timeout_ms=900,
                    mempool_max_batch_txs=1500,
                    mempool_proposal_interval_ms=75,
                    can_participate=True,
                    local_validator=True,
                )
            )
            self.assertEqual(
                action.to_dict(),
                {
                    "committee_size": 7,
                    "pacemaker_timeout_ms": 900,
                    "mempool_max_batch_txs": 1800,
                    "mempool_proposal_interval_ms": 45,
                    "submit_join": False,
                    "submit_leave": False,
                    "hydra_discovery_target": 0,
                    "reason": action.reason,
                    "agent_actions": [],
                },
            )


if __name__ == "__main__":
    unittest.main()
