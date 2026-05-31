import json
import tempfile
import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient
    import marl.app as app_module
    from marl.app import app
except ModuleNotFoundError:  # pragma: no cover
    TestClient = None
    app = None
    app_module = None

from marl.organization import ROLE_ACTION_FIELDS
from marl.policy import SafeFACMACPolicy
from marl.scenario import AIOT_NAMED_SCENARIOS
from marl.schemas import Observation, Action, DecisionActionStage, TraceStatus, TrajectorySample, SCHEMA_VERSION, runtime_trace_sample_from_dict
from marl.service import PolicyService
from marl.trainer import SafeFACMACModel, SafeFACMACTrainer


class ServiceTests(unittest.TestCase):
    def _fresh_route_client(self):
        previous = app_module.service
        app_module.service = PolicyService()
        self.addCleanup(lambda: setattr(app_module, "service", previous))
        return TestClient(app)

    def _expected_role_override_value(self, field: str, value):
        if field in {
            "committee_size",
            "pacemaker_timeout_ms",
            "mempool_max_batch_txs",
            "mempool_proposal_interval_ms",
            "hydra_discovery_target",
        }:
            return int(round(value))
        if field in {"submit_join", "submit_leave"}:
            return bool(value >= 0.5)
        self.fail(f"unexpected attributed field: {field}")

    def test_service_accepts_go_style_observation_fields_in_trace_ingest(self):
        service = PolicyService()
        sample = TrajectorySample.from_dict(
            {
                "policy_name": "safe-facmac",
                "observation": {
                    "validator_count": 8,
                    "current_config_id": 1,
                    "highest_known_config_id": 2,
                    "global_confirmed_total": 5,
                    "global_confirmed_nil": 1,
                    "last_ordered_rank": 9,
                    "last_ordered_height": 4,
                    "last_ordered_lane_id": 0,
                    "last_ordered_config_id": 3,
                    "last_ordered_nil": False,
                    "last_ordered_transition_count": 2,
                    "last_reconfig_epoch": 7,
                    "trust_snapshots": [
                        {
                            "node_id": 9,
                            "sample_count": 2,
                            "success_rate": 0.75,
                            "failure_probability": 0.25,
                        }
                    ],
                },
                "applied": {"action": {"pacemaker_timeout_ms": 1300}, "present": True},
                "reward": 0.5,
            }
        )
        result = service.ingest(sample)
        self.assertEqual(result["replay_size"], 1)
        stored = service._replay.get(0)
        self.assertEqual(stored.observation.global_confirmed_total, 5)
        self.assertEqual(stored.observation.last_ordered_rank, 9)
        self.assertEqual(stored.observation.last_reconfig_epoch, 7)
        self.assertEqual(len(stored.observation.trust_snapshots), 1)
        self.assertEqual(stored.observation.trust_snapshots[0].node_id, 9)

        service = PolicyService()
        result = service.ingest(
            TrajectorySample.from_dict(
                {
                    "policy_name": "safe-facmac",
                    "observation": {"validator_count": 8},
                    "applied": {"action": {"pacemaker_timeout_ms": 1300}, "present": True},
                    "reward": 0.5,
                }
            )
        )
        self.assertEqual(result["replay_size"], 1)
        self.assertEqual(service._replay.get(0).schema_version, SCHEMA_VERSION)

    def test_service_ingest_returns_runtime_trace_ack(self):
        service = PolicyService()
        result = service.ingest(
            TrajectorySample.from_dict(
                {
                    "policy_name": "runtime-adaptive",
                    "observation": {"validator_count": 8},
                    "applied": {"action": {"pacemaker_timeout_ms": 1300}, "present": True},
                    "reward": 0.5,
                    "governance_delta": True,
                    "guardrail_delta": False,
                }
            )
        )

        self.assertEqual(result["replay_size"], 1)
        self.assertEqual(result["schema_version"], SCHEMA_VERSION)
        self.assertEqual(result["policy_name"], "runtime-adaptive")
        self.assertIs(result["governance_delta"], True)
        self.assertIs(result["guardrail_delta"], False)

    def test_service_ingest_accepts_full_go_runtime_observation_schema(self):
        service = PolicyService()
        result = service.ingest(
            TrajectorySample.from_dict(
                {
                    "policy_name": "safe-facmac",
                    "observation": {
                        "timestamp": "2026-04-01T00:00:00Z",
                        "node_id": 7,
                        "epoch": 3,
                        "validator_count": 8,
                        "current_config_id": 1,
                        "highest_known_config_id": 2,
                        "committee_size": 4,
                        "pacemaker_timeout_ms": 1200,
                        "mempool_max_batch_txs": 1024,
                        "mempool_proposal_interval_ms": 80,
                        "throughput_tps": 4000,
                        "latency_p50_ms": 90,
                        "latency_p95_ms": 140,
                        "latency_p99_ms": 180,
                        "recovery_p95_ms": 160,
                        "backlog_pending": 12,
                        "backlog_missing": 1,
                        "reject_total": 2,
                        "connected_peers": 6,
                        "known_peers": 8,
                        "pending_joins": 1,
                        "pending_leaves": 0,
                        "lset_size": 10,
                        "can_participate": True,
                        "local_validator": True,
                        "global_confirmed_total": 99,
                        "global_confirmed_nil": 3,
                        "last_ordered_rank": 120,
                        "last_ordered_height": 55,
                        "last_ordered_lane_id": 1,
                        "last_ordered_config_id": 4,
                        "last_ordered_nil": False,
                        "last_ordered_transition_count": 2,
                        "last_reconfig_epoch": 4,
                        "heterogeneity_score": 0.6,
                        "churn_rate": 0.2,
                        "adversary_score": 0.1,
                        "network_jitter_ms": 25,
                        "ai_load_score": 0.7,
                        "agents": [{"instance_id": 1, "epoch": 3, "validator_count": 8, "committee_size": 4, "pacemaker_timeout_ms": 1200, "mempool_max_batch_txs": 1024, "mempool_proposal_interval_ms": 80}],
                        "trust_snapshots": [{"node_id": 9, "sample_count": 10, "success_rate": 0.75, "failure_probability": 0.25}],
                    },
                    "applied": {"action": {"pacemaker_timeout_ms": 1300}, "present": True},
                    "reward": 0.5,
                }
            )
        )
        self.assertEqual(result["replay_size"], 1)
        sample = service._replay.get(0)
        self.assertEqual(sample.observation.node_id, 7)
        self.assertEqual(sample.observation.global_confirmed_total, 99)
        self.assertEqual(sample.observation.last_ordered_rank, 120)
        self.assertEqual(sample.observation.trust_snapshots[0].node_id, 9)

    def test_service_ingest_rejects_unsupported_trace_schema_version(self):
        service = PolicyService()
        with self.assertRaisesRegex(ValueError, "unsupported trace schema_version: evolvbft-adaptive-v0 != evolvbft-adaptive-v1"):
            service.ingest(
                TrajectorySample.from_dict(
                    {
                        "policy_name": "safe-facmac",
                        "observation": {"validator_count": 8},
                        "applied": {"action": {"pacemaker_timeout_ms": 1300}, "present": True},
                        "reward": 0.5,
                        "schema_version": "evolvbft-adaptive-v0",
                    }
                )
            )

    def test_trace_ingest_route_defaults_missing_schema_version(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        client = self._fresh_route_client()

        response = client.post(
            "/trace/ingest",
            json={
                "policy_name": "safe-facmac",
                "observation": {"validator_count": 8},
                "applied": {"action": {"pacemaker_timeout_ms": 1300}, "present": True},
                "reward": 0.5,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["schema_version"], SCHEMA_VERSION)

    def test_trace_ingest_route_rejects_unsupported_schema_version(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        client = self._fresh_route_client()

        response = client.post(
            "/trace/ingest",
            json={
                "policy_name": "safe-facmac",
                "observation": {"validator_count": 8},
                "applied": {"action": {"pacemaker_timeout_ms": 1300}, "present": True},
                "reward": 0.5,
                "schema_version": "evolvbft-adaptive-v0",
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json()["detail"],
            "unsupported trace schema_version: evolvbft-adaptive-v0 != evolvbft-adaptive-v1",
        )

    def test_runtime_trace_offline_training_preserves_runtime_metadata(self):
        workspace_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=workspace_root) as tmp:
            trace_path = Path(tmp) / "trace.jsonl"
            trace_path.write_text(
                json.dumps(
                    {
                        "timestamp": "2026-04-01T00:00:00Z",
                        "policy_name": "runtime-adaptive",
                        "observation": {
                            "validator_count": 8,
                            "current_config_id": 1,
                            "highest_known_config_id": 2,
                            "global_confirmed_total": 5,
                            "last_ordered_lane_id": 1,
                            "last_ordered_config_id": 2,
                            "trust_snapshots": [
                                {
                                    "node_id": 9,
                                    "sample_count": 2,
                                    "success_rate": 0.75,
                                    "failure_probability": 0.25,
                                }
                            ],
                        },
                        "candidate": {"action": {"pacemaker_timeout_ms": 1300}, "present": True},
                        "governed": {"action": {"pacemaker_timeout_ms": 1200}, "present": True, "mutated": True},
                        "applied": {"action": {"pacemaker_timeout_ms": 1200}, "present": True},
                        "reward": 0.5,
                        "governance_delta": True,
                        "guardrail_delta": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            service = PolicyService()
            summary = service.train_offline(trace_path.relative_to(workspace_root))
            self.assertEqual(summary["training"]["last_training"]["schema_version"], SCHEMA_VERSION)
            self.assertEqual(summary["training"]["last_training"]["trace_path"], str(trace_path.relative_to(workspace_root)))
            self.assertEqual(summary["training"]["last_training"]["replay_priority_summary"]["governance_delta_count"], 1)
            self.assertEqual(summary["training"]["last_training"]["replay_priority_summary"]["guardrail_delta_count"], 0)
            self.assertEqual(service._replay.get(0).observation.trust_snapshots[0].node_id, 9)

    def test_train_offline_records_runtime_backed_baseline_summary(self):
        workspace_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=workspace_root) as tmp:
            corpus_dir = Path(tmp) / "stable"
            corpus_dir.mkdir(parents=True)
            trace_path = corpus_dir / "runtime_trace.jsonl"
            trace_path.write_text(
                json.dumps(
                    {
                        "timestamp": "2026-04-11T00:00:00Z",
                        "policy_name": "safe-baseline",
                        "schema_version": "evolvbft-adaptive-v1",
                        "observation": {
                            "validator_count": 4,
                            "global_confirmed_total": 0,
                            "committee_size": 4,
                        },
                        "applied": {"present": True, "action": {"committee_size": 4}},
                        "reward": 0.0,
                        "trace": {"enabled": True, "dropped_samples": 0},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path = corpus_dir / "runtime_trace_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "producer": "deploy/run_local_cluster.sh",
                        "schema_version": "evolvbft-adaptive-v1",
                        "truth_level": "authoritative-runtime-trace",
                        "claim_boundary": "minimal runtime-backed trace corpus only",
                        "trace_path": str(trace_path),
                        "scenario_label": "stable",
                        "seed": "localtest-a",
                        "policy_name": "safe-baseline",
                        "trace_family": "authoritative-runtime-trace",
                    }
                ),
                encoding="utf-8",
            )

            service = PolicyService()
            result = service.train_offline(trace_path.relative_to(workspace_root))
            inventory = service.artifact_inventory()

        self.assertEqual(result["samples"], 1)
        self.assertEqual(inventory["training_metadata"]["mode"], "offline")
        self.assertEqual(inventory["training_metadata"]["baseline_family"], "runtime-backed-minimal")
        self.assertEqual(inventory["training_metadata"]["claim_boundary"], "minimal runtime-backed training baseline only")
        self.assertEqual(inventory["training_metadata"]["trace_path"], str(trace_path.relative_to(workspace_root)))
        self.assertEqual(inventory["training_metadata"]["scenario_label"], "stable")
        self.assertEqual(inventory["training_metadata"]["seed"], "localtest-a")
        self.assertEqual(inventory["training_metadata"]["trace_manifest_path"], str(manifest_path.relative_to(workspace_root)))

    def test_service_can_train_from_trace_and_infer(self):
        workspace_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=workspace_root) as tmp:
            trace_path = Path(tmp) / "trace.jsonl"
            trace_path.write_text(
                json.dumps(
                    {
                        "timestamp": "2026-04-01T00:00:00Z",
                        "policy_name": "safe-baseline",
                        "observation": {
                            "current_config_id": 1,
                            "highest_known_config_id": 2,
                            "validator_count": 8,
                            "committee_size": 0,
                            "pacemaker_timeout_ms": 1000,
                            "mempool_max_batch_txs": 2048,
                            "mempool_proposal_interval_ms": 100,
                            "throughput_tps": 4000,
                            "latency_p95_ms": 300,
                            "backlog_pending": 100,
                            "backlog_missing": 2,
                            "reject_total": 2,
                            "pending_joins": 1,
                            "pending_leaves": 0,
                            "lset_size": 3,
                            "can_participate": True,
                            "local_validator": False,
                            "heterogeneity_score": 0.5,
                            "churn_rate": 0.2,
                            "adversary_score": 0.4,
                            "network_jitter_ms": 25,
                            "ai_load_score": 0.6,
                        },
                        "candidate": {
                            "action": {
                                "committee_size": 4,
                                "pacemaker_timeout_ms": 1400,
                                "mempool_max_batch_txs": 1024,
                                "mempool_proposal_interval_ms": 120,
                                "submit_join": True,
                                "hydra_discovery_target": 3,
                            },
                            "present": True,
                        },
                        "governed": {
                            "action": {
                                "committee_size": 4,
                                "pacemaker_timeout_ms": 1400,
                                "mempool_max_batch_txs": 1024,
                                "mempool_proposal_interval_ms": 120,
                                "submit_join": False,
                                "hydra_discovery_target": 3,
                            },
                            "present": True,
                            "mutated": True,
                        },
                        "applied": {
                            "action": {
                                "committee_size": 4,
                                "pacemaker_timeout_ms": 1400,
                                "mempool_max_batch_txs": 1024,
                                "mempool_proposal_interval_ms": 120,
                                "submit_join": False,
                                "hydra_discovery_target": 3,
                            },
                            "present": True,
                        },
                        "governance_delta": True,
                        "reward": -0.5,
                        "schema_version": "evolvbft-adaptive-v1",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            service = PolicyService()
            summary = service.train_offline(trace_path.relative_to(workspace_root))
            self.assertEqual(summary["samples"], 1)
            self.assertEqual(summary["training"]["last_training"]["mode"], "offline")
            self.assertEqual(summary["training"]["last_training"]["trace_path"], str(trace_path.relative_to(workspace_root)))
            self.assertEqual(service.schema_snapshot()["schema_version"], SCHEMA_VERSION)
            self.assertEqual(service.model_snapshot()["metadata"]["target_action"], "governed_stage_preferred")
            self.assertEqual(service.model_snapshot()["service_training"]["last_training"]["mode"], "offline")
            self.assertNotIn("sample_summary", service.model_snapshot()["service_training"]["last_training"])

            action = service.infer(
                Observation(
                    current_config_id=1,
                    highest_known_config_id=2,
                    validator_count=8,
                    committee_size=0,
                    pacemaker_timeout_ms=1000,
                    mempool_max_batch_txs=2048,
                    mempool_proposal_interval_ms=100,
                    throughput_tps=5000,
                    latency_p95_ms=200,
                    backlog_pending=50,
                    backlog_missing=1,
                    reject_total=0,
                    pending_joins=0,
                    pending_leaves=0,
                    lset_size=3,
                    can_participate=True,
                    local_validator=False,
                    heterogeneity_score=0.4,
                    churn_rate=0.1,
                    adversary_score=0.2,
                    network_jitter_ms=15,
                    ai_load_score=0.7,
                )
            )
            self.assertIsNotNone(action.reason)

    def test_runtime_trace_checkpoint_round_trip_preserves_runtime_fields(self):
        service = PolicyService()
        sample = runtime_trace_sample_from_dict(
            {
                "policy_name": "runtime-adaptive",
                "observation": {
                    "validator_count": 8,
                    "current_config_id": 1,
                    "highest_known_config_id": 2,
                    "global_confirmed_total": 5,
                    "last_ordered_lane_id": 1,
                    "last_ordered_config_id": 2,
                    "trust_snapshots": [
                        {
                            "node_id": 9,
                            "sample_count": 2,
                            "success_rate": 0.75,
                            "failure_probability": 0.25,
                        }
                    ],
                },
                "candidate": {"action": {"pacemaker_timeout_ms": 1300}, "present": True},
                "governed": {"action": {"pacemaker_timeout_ms": 1200}, "present": True, "mutated": True},
                "applied": {"action": {"pacemaker_timeout_ms": 1200}, "present": True},
                "reward": 0.5,
                "governance_delta": True,
                "guardrail_delta": False,
            }
        )
        service.ingest(sample)
        summary = service.train_online(batch_size=1)
        self.assertEqual(summary["training"]["last_training"]["schema_version"], SCHEMA_VERSION)
        self.assertEqual(summary["training"]["last_training"]["sample_summary"]["governance_delta_count"], 1)
        self.assertEqual(summary["training"]["last_training"]["sample_summary"]["guardrail_delta_count"], 0)

        workspace_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=workspace_root) as tmp:
            path = Path(tmp) / "checkpoint.json"
            relative_path = path.relative_to(workspace_root)
            service.save_checkpoint(relative_path)
            reloaded = PolicyService()
            load_result = reloaded.load_checkpoint(relative_path)

        self.assertEqual(load_result["training"]["last_training"]["schema_version"], SCHEMA_VERSION)
        self.assertEqual(load_result["training"]["last_training"]["sample_summary"]["governance_delta_count"], 1)
        self.assertEqual(load_result["training"]["last_training"]["sample_summary"]["guardrail_delta_count"], 0)
        self.assertEqual(load_result["checkpoint"]["last_loaded_path"], str(relative_path))
        self.assertEqual(reloaded.model_snapshot()["metadata_provenance"], "checkpoint")

    def test_runtime_trace_checkpoint_round_trip_preserves_offline_provenance_fields(self):
        workspace_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=workspace_root) as tmp:
            trace_dir = Path(tmp) / "runtime_trace_matrix" / "stable"
            trace_dir.mkdir(parents=True)
            trace_path = trace_dir / "runtime_trace.jsonl"
            trace_path.write_text(
                json.dumps(
                    {
                        "timestamp": "2026-04-11T00:00:00Z",
                        "policy_name": "safe-baseline",
                        "schema_version": SCHEMA_VERSION,
                        "observation": {"validator_count": 4, "global_confirmed_total": 0, "committee_size": 4},
                        "applied": {"present": True, "action": {"committee_size": 4}},
                        "reward": 0.0,
                        "trace": {"enabled": True, "dropped_samples": 0},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path = trace_dir / "runtime_trace_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "producer": "deploy/run_local_cluster.sh",
                        "schema_version": SCHEMA_VERSION,
                        "truth_level": "authoritative-runtime-trace",
                        "claim_boundary": "minimal runtime-backed trace corpus only",
                        "trace_path": str(trace_path),
                        "scenario_label": "stable",
                        "seed": "localtest-a",
                        "policy_name": "safe-baseline",
                        "trace_family": "authoritative-runtime-trace",
                    }
                ),
                encoding="utf-8",
            )

            service = PolicyService()
            service.train_offline(trace_path.relative_to(workspace_root))
            checkpoint_path = Path(tmp) / "checkpoint.json"
            relative_path = checkpoint_path.relative_to(workspace_root)
            service.save_checkpoint(relative_path)

            reloaded = PolicyService()
            load_result = reloaded.load_checkpoint(relative_path)

        restored = load_result["training"]["last_training"]
        self.assertEqual(restored["mode"], "offline")
        self.assertEqual(restored["trace_path"], str(trace_path.relative_to(workspace_root)))
        self.assertEqual(restored["trace_manifest_path"], str(manifest_path.relative_to(workspace_root)))
        self.assertEqual(restored["scenario_label"], "stable")
        self.assertEqual(restored["seed"], "localtest-a")

    def test_service_online_replay_and_checkpoint(self):
        service = PolicyService()
        service.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=1, throughput_tps=4000, latency_p95_ms=200),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                reward=0.5,
            )
        )
        summary = service.train_online(batch_size=1)
        self.assertEqual(summary["samples"], 1)
        self.assertEqual(summary["training"]["last_training"]["mode"], "online")
        self.assertEqual(summary["training"]["last_training"]["batch_size"], 1)
        self.assertEqual(summary["training"]["last_activation"]["mode"], "online_training")
        observation = Observation(validator_count=8, current_config_id=1, highest_known_config_id=1, throughput_tps=4000, latency_p95_ms=200)
        role_observation = Observation(
            current_config_id=1,
            highest_known_config_id=2,
            validator_count=4,
            can_participate=True,
            local_validator=False,
        )
        before = service.infer(observation).to_dict()
        role_before = service.infer(role_observation).to_dict()
        snapshot_before = service.model_snapshot()

        workspace_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=workspace_root) as tmp:
            path = Path(tmp) / "checkpoint.json"
            relative_path = path.relative_to(workspace_root)
            save_result = service.save_checkpoint(relative_path)
            self.assertEqual(save_result["checkpoint"]["last_saved_path"], str(relative_path))
            self.assertEqual(save_result["checkpoint"]["best_saved_path"], str(relative_path))
            self.assertEqual(save_result["checkpoint"]["best_reward_mean"], 0.5)
            reloaded = PolicyService()
            load_result = reloaded.load_checkpoint(relative_path)
            self.assertEqual(load_result["checkpoint"]["last_loaded_path"], str(relative_path))
            self.assertEqual(load_result["checkpoint"]["last_saved_path"], None)
            self.assertEqual(load_result["checkpoint"]["best_saved_path"], str(relative_path))
            self.assertEqual(load_result["checkpoint"]["best_reward_mean"], 0.5)
            self.assertEqual(load_result["training"]["active_model_source"], "checkpoint_load")
            self.assertEqual(load_result["training"]["checkpoint"]["active_checkpoint_path"], str(relative_path))
            self.assertEqual(load_result["training"]["last_activation"]["mode"], "checkpoint_load")
            self.assertEqual(load_result["training"]["last_training"]["mode"], "online")
            self.assertEqual(load_result["training"]["last_training"]["schema_version"], SCHEMA_VERSION)
            self.assertIn("replay_priority_summary", load_result["training"]["last_training"])
            self.assertIn("sample_summary", load_result["training"]["last_training"])
            self.assertEqual(reloaded.model_snapshot()["metadata"]["trainer_config"]["seed"], 7)
            after = reloaded.infer(observation).to_dict()
            role_after = reloaded.infer(role_observation).to_dict()
            snapshot_after = reloaded.model_snapshot()
            self.assertEqual(reloaded.replay_summary()["replay_size"], 0)
            self.assertEqual(before, after)
            self.assertEqual(role_before, role_after)
            self.assertEqual(snapshot_before["actor_weights"], snapshot_after["actor_weights"])
            self.assertEqual(snapshot_before["actor_bias"], snapshot_after["actor_bias"])
            self.assertEqual(snapshot_before["role_actor_weights"], snapshot_after["role_actor_weights"])
            self.assertEqual(snapshot_before["role_actor_bias"], snapshot_after["role_actor_bias"])
            self.assertEqual(snapshot_before["agent_actor_weights"], snapshot_after["agent_actor_weights"])
            self.assertEqual(snapshot_before["agent_actor_bias"], snapshot_after["agent_actor_bias"])
            self.assertEqual(snapshot_before["critic_weights"], snapshot_after["critic_weights"])
            self.assertEqual(snapshot_before["critic_bias"], snapshot_after["critic_bias"])
            self.assertEqual(snapshot_before["metadata"], snapshot_after["metadata"])

            replay_before = service.replay_summary()
            self.assertEqual(replay_before["replay_size"], 1)
            service.load_checkpoint(relative_path)
            overwritten_summary = service.training_summary()
            overwritten_after = service.infer(observation).to_dict()
            overwritten_role_after = service.infer(role_observation).to_dict()
            overwritten_snapshot = service.model_snapshot()
            replay_after = service.replay_summary()
            self.assertEqual(overwritten_summary["active_model_source"], "checkpoint_load")
            self.assertEqual(overwritten_summary["last_training"]["mode"], "online")
            self.assertEqual(overwritten_summary["last_activation"]["mode"], "checkpoint_load")
            self.assertEqual(overwritten_summary["checkpoint"]["active_checkpoint_path"], str(relative_path))
            self.assertEqual(service.model_snapshot()["metadata_provenance"], "checkpoint")
            self.assertEqual(after, overwritten_after)
            self.assertEqual(role_after, overwritten_role_after)
            self.assertEqual(snapshot_after["actor_weights"], overwritten_snapshot["actor_weights"])
            self.assertEqual(snapshot_after["actor_bias"], overwritten_snapshot["actor_bias"])
            self.assertEqual(snapshot_after["role_actor_weights"], overwritten_snapshot["role_actor_weights"])
            self.assertEqual(snapshot_after["role_actor_bias"], overwritten_snapshot["role_actor_bias"])
            self.assertEqual(snapshot_after["agent_actor_weights"], overwritten_snapshot["agent_actor_weights"])
            self.assertEqual(snapshot_after["agent_actor_bias"], overwritten_snapshot["agent_actor_bias"])
            self.assertEqual(snapshot_after["critic_weights"], overwritten_snapshot["critic_weights"])
            self.assertEqual(snapshot_after["critic_bias"], overwritten_snapshot["critic_bias"])
            self.assertEqual(snapshot_after["metadata"], overwritten_snapshot["metadata"])
            self.assertEqual(replay_before, replay_after)

    def test_service_model_snapshot_filters_non_allowlisted_metadata(self):
        service = PolicyService()
        service.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=1, throughput_tps=4000, latency_p95_ms=200),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                reward=0.5,
            )
        )
        service.train_online(batch_size=1)
        service._model.metadata["unexpected_secret"] = "drop-me"
        service._model.metadata["claim_boundary"] = "research_asset_only"
        service._model.metadata["trainer_family"] = "facmac-compatible-research-service"
        service._model.metadata["paper_grade_facmac"] = False

        snapshot = service.model_snapshot()

        self.assertNotIn("unexpected_secret", snapshot["metadata"])
        self.assertEqual(snapshot["metadata"]["claim_boundary"], "research_asset_only")
        self.assertEqual(snapshot["metadata"]["trainer_family"], "facmac-compatible-research-service")
        self.assertFalse(snapshot["metadata"]["paper_grade_facmac"])
        self.assertEqual(snapshot["metadata_provenance"], "trainer")

    def test_service_exposes_unified_adaptive_snapshot(self):
        service = PolicyService()
        service.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=1, throughput_tps=4000, latency_p95_ms=200),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                reward=0.5,
                governance_delta=True,
            )
        )
        service.train_online(batch_size=1)

        snapshot = service.adaptive_snapshot()

        self.assertTrue(snapshot["model_ready"])
        self.assertIn("model", snapshot)
        self.assertIn("organization", snapshot)
        self.assertIn("schema", snapshot)
        self.assertIn("replay", snapshot)
        self.assertIn("training", snapshot)
        self.assertEqual(snapshot["model"]["service_training"], snapshot["training"])
        self.assertEqual(snapshot["replay"]["replay_size"], 1)
        self.assertEqual(snapshot["training"]["active_model_source"], "online_training")
        self.assertIn("decision_fields", snapshot["organization"])
        self.assertIn("role_activation_rules", snapshot["organization"])
        self.assertIn("escalation_rules", snapshot["organization"])
        self.assertIn("safety_blocking_rules", snapshot["organization"])
        self.assertIn("freeze_lane_tuning", snapshot["organization"]["decision_fields"])
        self.assertIn("artifact_inventory", snapshot)
        self.assertEqual(snapshot["artifact_inventory"]["schema_version"], SCHEMA_VERSION)
        self.assertEqual(snapshot["artifact_inventory"]["model_ready"], True)
        self.assertEqual(snapshot["artifact_inventory"]["replay_size"], 1)
        self.assertIn("checkpoint_metadata", snapshot["artifact_inventory"])
        self.assertEqual(
            snapshot["artifact_inventory"]["checkpoint_metadata"]["last_training"],
            snapshot["training"]["last_training"],
        )
        self.assertEqual(
            snapshot["artifact_inventory"]["checkpoint_metadata"]["claim_boundary"],
            snapshot["artifact_inventory"]["claim_boundary"],
        )
        self.assertEqual(
            snapshot["artifact_inventory"]["checkpoint_metadata"]["trainer_family"],
            snapshot["artifact_inventory"]["trainer_family"],
        )
        self.assertEqual(
            snapshot["artifact_inventory"]["checkpoint_metadata"]["paper_grade_facmac"],
            snapshot["artifact_inventory"]["paper_grade_facmac"],
        )
        self.assertIn("training_metadata", snapshot["artifact_inventory"])
        self.assertEqual(snapshot["artifact_inventory"]["training_metadata"], snapshot["training"]["last_training"])

    def test_artifact_inventory_exposes_claim_safe_checkpoint_metadata(self):
        service = PolicyService()
        service._checkpoint_summary["best_saved_path"] = "checkpoints/best.pt"
        service._checkpoint_summary["best_reward_mean"] = 0.75
        service._last_training_summary = {
            "trainer_metadata": {
                "trainer_family": "facmac-compatible-research-service",
                "claim_boundary": "research_co_runtime",
                "paper_grade_facmac": False,
            }
        }

        inventory = service.artifact_inventory()

        self.assertEqual(
            list(inventory["checkpoint"].keys()),
            ["best_reward_mean", "best_saved_path", "last_loaded_path", "last_saved_path"],
        )
        self.assertEqual(inventory["checkpoint"]["best_saved_path"], "checkpoints/best.pt")
        self.assertEqual(inventory["checkpoint_metadata"]["best_saved_path"], "checkpoints/best.pt")
        self.assertEqual(inventory["checkpoint_metadata"]["claim_boundary"], "research_co_runtime")
        self.assertEqual(inventory["checkpoint_metadata"]["trainer_family"], "facmac-compatible-research-service")
        self.assertIs(inventory["checkpoint_metadata"]["paper_grade_facmac"], False)

    def test_adaptive_snapshot_preserves_nested_checkpoint_metadata(self):
        service = PolicyService()
        service._checkpoint_summary["best_saved_path"] = "checkpoints/best.pt"
        service._checkpoint_summary["best_reward_mean"] = 0.75
        service._last_training_summary = {
            "trainer_metadata": {
                "trainer_family": "facmac-compatible-research-service",
                "claim_boundary": "research_co_runtime",
                "paper_grade_facmac": False,
            }
        }

        snapshot = service.adaptive_snapshot()

        self.assertEqual(snapshot["artifact_inventory"]["training_metadata"], snapshot["training"]["last_training"])
        self.assertEqual(snapshot["artifact_inventory"]["checkpoint_metadata"]["claim_boundary"], "research_co_runtime")

    def test_service_exposes_trainer_config_metadata_in_snapshot(self):
        service = PolicyService()
        service.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=1, throughput_tps=4000, latency_p95_ms=200),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                reward=0.5,
            )
        )
        service.train_online(batch_size=1)

        snapshot = service.model_snapshot()

        self.assertIn("trainer_config", snapshot["metadata"])
        self.assertEqual(snapshot["metadata"]["trainer_config"]["seed"], 7)
        self.assertIn("target_action", snapshot["metadata"])
        self.assertEqual(snapshot["metadata"]["target_action"], "governed_stage_preferred")

    def test_service_adaptive_snapshot_preserves_stage_contract_boundary(self):
        service = PolicyService()
        service.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=2, throughput_tps=4500, latency_p95_ms=180),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1400), present=True),
                governed=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300), present=True),
                masked=DecisionActionStage(action=Action(pacemaker_timeout_ms=1250), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1200), present=True),
                reward=1.0,
                governance_delta=True,
                guardrail_delta=True,
                trace=TraceStatus(enabled=True, write_failed=False, close_failed=False, dropped_samples=0),
            )
        )

        snapshot = service.adaptive_snapshot()

        self.assertIn("replay", snapshot)
        self.assertIn("training", snapshot)
        self.assertIn("schema", snapshot)
        self.assertIn("organization", snapshot)
        self.assertEqual(snapshot["replay"]["replay_size"], 1)
        self.assertIn("priority_summary", snapshot["replay"])
        self.assertEqual(snapshot["schema"]["schema_version"], SCHEMA_VERSION)

    def test_service_model_snapshot_is_none_before_training(self):
        service = PolicyService()

        self.assertIsNone(service.model_snapshot())

    def test_service_exposes_cold_start_adaptive_snapshot(self):
        service = PolicyService()

        snapshot = service.adaptive_snapshot()

        self.assertFalse(snapshot["model_ready"])
        self.assertIsNone(snapshot["model"])
        self.assertEqual(snapshot["training"]["active_model_source"], "cold_start")
        self.assertFalse(snapshot["training"]["model_ready"])
        self.assertIsNone(snapshot["training"]["last_training"])
        self.assertIsNone(snapshot["training"]["last_activation"])
        self.assertEqual(snapshot["replay"]["replay_size"], 0)
        self.assertEqual(snapshot["replay"]["priority_summary"]["priority_sample_count"], 0)
        self.assertEqual(snapshot["schema"]["schema_version"], SCHEMA_VERSION)
        self.assertEqual(snapshot["organization"]["schema_version"], SCHEMA_VERSION)
        self.assertIn("artifact_inventory", snapshot)
        self.assertEqual(snapshot["artifact_inventory"]["schema_version"], SCHEMA_VERSION)
        self.assertEqual(snapshot["artifact_inventory"]["active_model_source"], "cold_start")
        self.assertIsNone(snapshot["artifact_inventory"]["checkpoint_metadata"]["last_training"])
        self.assertIsNone(snapshot["artifact_inventory"]["checkpoint_metadata"]["claim_boundary"])
        self.assertIsNone(snapshot["artifact_inventory"]["checkpoint_metadata"]["trainer_family"])
        self.assertIsNone(snapshot["artifact_inventory"]["checkpoint_metadata"]["paper_grade_facmac"])
        self.assertIsNone(snapshot["artifact_inventory"]["training_metadata"])
        self.assertFalse(snapshot["artifact_inventory"]["has_training_summary"])
        self.assertFalse(snapshot["artifact_inventory"]["has_activation_summary"])

    def test_service_exposes_replay_shadow_summary(self):
        service = PolicyService()
        service.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(
                    validator_count=8,
                    current_config_id=1,
                    highest_known_config_id=2,
                    adversary_score=0.8,
                    churn_rate=0.5,
                    network_jitter_ms=60,
                ),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1200), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1200), present=True),
                reward=1.0,
                governance_delta=True,
            )
        )

        snapshot = service.adaptive_snapshot()

        self.assertEqual(snapshot["replay"]["replay_size"], 1)
        self.assertEqual(snapshot["replay"]["priority_summary"]["priority_sample_count"], 1)
        self.assertEqual(snapshot["replay"]["priority_summary"]["governance_delta_count"], 1)
        self.assertEqual(snapshot["training"]["active_model_source"], "cold_start")

    def test_service_inspects_decision_flow_cold_start_contract(self):
        service = PolicyService()
        observation = Observation(
            validator_count=8,
            current_config_id=1,
            highest_known_config_id=2,
            committee_size=4,
            pacemaker_timeout_ms=1000,
            mempool_max_batch_txs=1024,
            mempool_proposal_interval_ms=80,
            throughput_tps=4500,
            latency_p95_ms=180,
            local_validator=False,
        )

        flow = service.inspect_decision_flow(observation)

        self.assertFalse(flow["model_ready"])
        self.assertEqual(flow["candidate"]["committee_size"], observation.committee_size)
        self.assertEqual(flow["candidate"]["pacemaker_timeout_ms"], observation.pacemaker_timeout_ms)
        self.assertEqual(flow["candidate"]["mempool_max_batch_txs"], observation.mempool_max_batch_txs)
        self.assertEqual(flow["candidate"]["mempool_proposal_interval_ms"], observation.mempool_proposal_interval_ms)
        self.assertEqual(flow["candidate"]["reason"], "default")
        for field in (
            "committee_size",
            "pacemaker_timeout_ms",
            "mempool_max_batch_txs",
            "mempool_proposal_interval_ms",
            "submit_join",
            "submit_leave",
            "hydra_discovery_target",
        ):
            self.assertEqual(flow["candidate"][field], flow["governed"][field])
            self.assertEqual(flow["governed"][field], flow["applied"][field])
        self.assertIn("roles=membership_tuner", flow["governed"]["reason"])
        self.assertEqual(flow["organization_decision"]["active_roles"], ["membership_tuner"])
        self.assertFalse(flow["organization_decision"]["freeze_membership"])
        self.assertEqual(flow["organization_decision"]["blocked_fields"], [])
        self.assertFalse(flow["divergence"]["candidate_vs_governed"])
        self.assertFalse(flow["divergence"]["governed_vs_applied"])
        self.assertFalse(flow["divergence"]["candidate_vs_applied"])

    def test_service_inspects_decision_flow_checkpoint_loaded_contract(self):
        source = PolicyService()
        source.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=1, throughput_tps=4000, latency_p95_ms=200),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                reward=0.5,
            )
        )
        source.train_online(batch_size=1)

        workspace_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=workspace_root) as tmp:
            path = Path(tmp) / "checkpoint.json"
            relative_path = path.relative_to(workspace_root)
            source.save_checkpoint(relative_path)

            reloaded = PolicyService()
            reloaded.load_checkpoint(relative_path)
            observation = Observation(
                validator_count=8,
                current_config_id=1,
                highest_known_config_id=2,
                throughput_tps=4500,
                latency_p95_ms=180,
                local_validator=False,
            )

            flow = reloaded.inspect_decision_flow(observation)
            adaptive = reloaded.adaptive_snapshot()

            self.assertTrue(flow["model_ready"])
            self.assertEqual(adaptive["training"]["active_model_source"], "checkpoint_load")
            self.assertEqual(adaptive["training"]["checkpoint"]["active_checkpoint_path"], str(relative_path))
            self.assertTrue(adaptive["model_ready"])
            self.assertIsNotNone(adaptive["model"])
            self.assertEqual(adaptive["model"]["metadata_provenance"], "checkpoint")
            self.assertEqual(adaptive["model"]["service_training"]["active_model_source"], "checkpoint_load")
            self.assertEqual(set(flow["explanations"].keys()), {"candidate_to_governed", "governed_to_applied", "candidate_to_applied"})
            self.assertEqual(set(flow["explanation_details"].keys()), {"candidate_to_governed", "governed_to_applied", "candidate_to_applied"})
            for detail in flow["explanation_details"].values():
                self.assertIn("blocked_by_guardrails", detail)
                self.assertIn("organization_notes", detail)
            self.assertEqual(flow["divergence"]["candidate_vs_governed"], bool(flow["divergence"]["candidate_vs_governed_fields"]))
            self.assertEqual(flow["divergence"]["governed_vs_applied"], bool(flow["divergence"]["governed_vs_applied_fields"]))
            self.assertEqual(flow["divergence"]["candidate_vs_applied"], bool(flow["divergence"]["candidate_vs_applied_fields"]))
            self.assertEqual(flow["organization_reasoning"]["blocked_fields_without_reason"], [])
            self.assertIn("artifact_inventory", flow)
            self.assertEqual(flow["artifact_inventory"]["schema_version"], SCHEMA_VERSION)
            self.assertEqual(flow["artifact_inventory"]["model_ready"], True)
            self.assertEqual(flow["artifact_inventory"]["active_model_source"], "checkpoint_load")
            self.assertTrue(flow["artifact_inventory"]["has_training_summary"])
            self.assertTrue(flow["artifact_inventory"]["has_activation_summary"])

    def test_service_decision_flow_safety_matrix(self):
        service = PolicyService()
        cases = [
            {
                "name": "nominal-local-validator",
                "observation": Observation(
                    validator_count=8,
                    current_config_id=1,
                    highest_known_config_id=1,
                    committee_size=4,
                    pacemaker_timeout_ms=1000,
                    mempool_max_batch_txs=1024,
                    mempool_proposal_interval_ms=80,
                    throughput_tps=5000,
                    latency_p95_ms=150,
                    local_validator=True,
                    can_participate=True,
                ),
                "active_roles": ["lane_tuner"],
                "escalation_level": "nominal",
                "freeze_membership": False,
                "freeze_lane_tuning": False,
                "blocked_fields": [],
            },
            {
                "name": "elevated-membership-pressure",
                "observation": Observation(
                    validator_count=8,
                    current_config_id=1,
                    highest_known_config_id=2,
                    committee_size=4,
                    pacemaker_timeout_ms=1000,
                    mempool_max_batch_txs=1024,
                    mempool_proposal_interval_ms=80,
                    throughput_tps=4200,
                    latency_p95_ms=260,
                    churn_rate=0.45,
                    local_validator=False,
                    can_participate=True,
                ),
                "active_roles": ["membership_tuner"],
                "escalation_level": "elevated",
                "freeze_membership": True,
                "freeze_lane_tuning": False,
                "blocked_fields": ["submit_join", "submit_leave", "hydra_discovery_target"],
            },
            {
                "name": "critical-safety-stress",
                "observation": Observation(
                    validator_count=8,
                    current_config_id=1,
                    highest_known_config_id=2,
                    committee_size=4,
                    pacemaker_timeout_ms=1000,
                    mempool_max_batch_txs=1024,
                    mempool_proposal_interval_ms=80,
                    throughput_tps=3000,
                    latency_p95_ms=600,
                    backlog_missing=11,
                    reject_total=12,
                    adversary_score=0.8,
                    local_validator=False,
                    can_participate=True,
                ),
                "active_roles": ["safety_guardian", "recovery_tuner", "membership_tuner"],
                "escalation_level": "critical",
                "freeze_membership": True,
                "freeze_lane_tuning": True,
                "blocked_fields": [
                    "submit_join",
                    "submit_leave",
                    "hydra_discovery_target",
                    "committee_size",
                    "mempool_max_batch_txs",
                    "mempool_proposal_interval_ms",
                ],
            },
        ]

        for case in cases:
            with self.subTest(case=case["name"]):
                flow = service.inspect_decision_flow(case["observation"])
                decision = flow["organization_decision"]
                self.assertEqual(decision["active_roles"], case["active_roles"])
                self.assertEqual(decision["escalation_level"], case["escalation_level"])
                self.assertEqual(decision["freeze_membership"], case["freeze_membership"])
                self.assertEqual(decision["freeze_lane_tuning"], case["freeze_lane_tuning"])
                self.assertEqual(sorted(set(decision["blocked_fields"])), sorted(case["blocked_fields"]))
                self.assertEqual(flow["organization_reasoning"]["blocked_fields_without_reason"], [])
                for blocked_field in case["blocked_fields"]:
                    self.assertIn(blocked_field, flow["organization_reasoning"]["blocked_field_reasons"])

    def test_named_scenarios_form_acceptance_matrix(self):
        service = PolicyService()
        cases = [
            {
                "scenario_name": "heterogeneous_steady_state",
                "expected_escalation": "nominal",
                "expected_roles": ["membership_tuner"],
                "freeze_membership": False,
                "freeze_lane_tuning": False,
            },
            {
                "scenario_name": "churn_reconfiguration_pressure",
                "expected_escalation": "elevated",
                "expected_roles": ["recovery_tuner", "membership_tuner"],
                "freeze_membership": True,
                "freeze_lane_tuning": False,
            },
            {
                "scenario_name": "adversarial_safety_stress",
                "expected_escalation": "critical",
                "expected_roles": ["recovery_tuner", "membership_tuner"],
                "freeze_membership": True,
                "freeze_lane_tuning": True,
            },
            {
                "scenario_name": "jitter_recovery_stress",
                "expected_escalation": "nominal",
                "expected_roles": ["membership_tuner"],
                "freeze_membership": False,
                "freeze_lane_tuning": False,
            },
            {
                "scenario_name": "ai_load_throughput_stress",
                "expected_escalation": "nominal",
                "expected_roles": ["membership_tuner"],
                "freeze_membership": False,
                "freeze_lane_tuning": False,
            },
        ]

        for case in cases:
            with self.subTest(case=case["scenario_name"]):
                context = AIOT_NAMED_SCENARIOS[case["scenario_name"]]
                flow = service.inspect_decision_flow(
                    Observation(
                        validator_count=8,
                        current_config_id=1,
                        highest_known_config_id=2,
                        committee_size=4,
                        pacemaker_timeout_ms=1000,
                        mempool_max_batch_txs=1024,
                        mempool_proposal_interval_ms=80,
                        throughput_tps=4000,
                        latency_p95_ms=220,
                        local_validator=False,
                        can_participate=True,
                        heterogeneity_score=context["heterogeneity_score"],
                        churn_rate=context["churn_rate"],
                        adversary_score=context["adversary_score"],
                        network_jitter_ms=context["network_jitter_ms"],
                        ai_load_score=context["ai_load_score"],
                    )
                )
                decision = flow["organization_decision"]
                self.assertEqual(decision["escalation_level"], case["expected_escalation"])
                self.assertEqual(decision["active_roles"], case["expected_roles"])
                self.assertEqual(decision["freeze_membership"], case["freeze_membership"])
                self.assertEqual(decision["freeze_lane_tuning"], case["freeze_lane_tuning"])

    def test_service_inspect_replay_sample_marks_missing_governed_stage(self):
        service = PolicyService()
        sample = TrajectorySample(
            policy_name="safe-facmac",
            observation=Observation(
                validator_count=8,
                current_config_id=1,
                highest_known_config_id=2,
                committee_size=4,
                pacemaker_timeout_ms=1000,
                mempool_max_batch_txs=1024,
                mempool_proposal_interval_ms=80,
                throughput_tps=4500,
                latency_p95_ms=180,
                local_validator=False,
            ),
            candidate=DecisionActionStage(action=Action(committee_size=6, pacemaker_timeout_ms=1400), present=True),
            applied=DecisionActionStage(action=Action(committee_size=4, pacemaker_timeout_ms=1200), present=True),
            trace=TraceStatus(enabled=True, write_failed=True, close_failed=False, dropped_samples=2),
            reward=1.0,
        )
        service.ingest(sample)

        inspected = service.inspect_replay_sample(0)

        self.assertEqual(inspected["index"], 0)
        self.assertTrue(inspected["recorded"]["candidate_present"])
        self.assertFalse(inspected["recorded"]["governed_present"])
        self.assertEqual(inspected["recorded"]["candidate"]["committee_size"], 6)
        self.assertEqual(inspected["recorded"]["candidate"]["pacemaker_timeout_ms"], 1400)
        self.assertEqual(inspected["recorded"]["applied"]["committee_size"], 4)
        self.assertEqual(inspected["recorded"]["applied"]["pacemaker_timeout_ms"], 1200)
        self.assertIn("candidate", inspected["current"])
        self.assertIn("governed", inspected["current"])
        self.assertIn("applied", inspected["current"])
        self.assertIsNone(inspected["recorded"]["governed"])
        self.assertTrue(inspected["recorded"]["trace"]["enabled"])
        self.assertTrue(inspected["recorded"]["trace"]["write_failed"])
        self.assertFalse(inspected["recorded"]["trace"]["close_failed"])
        self.assertEqual(inspected["recorded"]["trace"]["dropped_samples"], 2)
        self.assertIsNone(inspected["divergence"]["recorded_governed_vs_current_governed"])
        self.assertIsNone(inspected["divergence"]["recorded_governed_vs_current_governed_fields"])

    def test_service_inspects_replay_sample_against_current_flow(self):
        service = PolicyService()
        service.train_online(batch_size=1)
        sample = TrajectorySample(
            policy_name="safe-facmac",
            observation=Observation(
                validator_count=8,
                current_config_id=1,
                highest_known_config_id=2,
                committee_size=4,
                pacemaker_timeout_ms=1000,
                mempool_max_batch_txs=1024,
                mempool_proposal_interval_ms=80,
                throughput_tps=4200,
                latency_p95_ms=640,
                backlog_missing=5,
                reject_total=7,
                local_validator=False,
                adversary_score=0.7,
                network_jitter_ms=70,
            ),
            candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1600, mempool_max_batch_txs=512), present=True),
            governed=DecisionActionStage(action=Action(pacemaker_timeout_ms=1500, mempool_max_batch_txs=512), present=True),
            applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1500, mempool_max_batch_txs=512), present=True),
            trace=TraceStatus(enabled=True, write_failed=False, close_failed=True, dropped_samples=1),
            reward=-1.0,
        )
        service.ingest(sample)

        inspected = service.inspect_replay_sample(0)

        self.assertEqual(inspected["index"], 0)
        self.assertTrue(inspected["recorded"]["governed_present"])
        self.assertEqual(inspected["recorded"]["candidate"]["pacemaker_timeout_ms"], 1600)
        self.assertEqual(inspected["recorded"]["governed"]["pacemaker_timeout_ms"], 1500)
        self.assertEqual(inspected["recorded"]["applied"]["pacemaker_timeout_ms"], 1500)
        self.assertEqual(inspected["current"]["organization_decision"]["escalation_level"], "critical")
        self.assertIn("recovery_tuner", inspected["current"]["organization_decision"]["active_roles"])
        self.assertIn("safety_guardian", inspected["current"]["organization_decision"]["active_roles"])
        self.assertTrue(inspected["current"]["governed"]["submit_join"] is False)
        self.assertTrue(inspected["recorded"]["trace"]["enabled"])
        self.assertFalse(inspected["recorded"]["trace"]["write_failed"])
        self.assertTrue(inspected["recorded"]["trace"]["close_failed"])
        self.assertEqual(inspected["recorded"]["trace"]["dropped_samples"], 1)
        self.assertEqual(
            inspected["divergence"]["recorded_governed_vs_current_governed"],
            bool(inspected["divergence"]["recorded_governed_vs_current_governed_fields"]),
        )
        self.assertEqual(
            inspected["divergence"]["recorded_candidate_vs_current_candidate"],
            bool(inspected["divergence"]["recorded_candidate_vs_current_candidate_fields"]),
        )
        self.assertEqual(
            inspected["divergence"]["recorded_applied_vs_current_applied"],
            bool(inspected["divergence"]["recorded_applied_vs_current_applied_fields"]),
        )

    def test_service_replay_sample_rejects_invalid_index(self):
        service = PolicyService()

        with self.assertRaises(IndexError):
            service.inspect_replay_sample(0)
        with self.assertRaises(IndexError):
            service.inspect_replay_sample(-1)

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_trace_ingest_route_rejects_non_object_trace(self):
        client = self._fresh_route_client()

        response = client.post(
            "/trace/ingest",
            json={
                "policy_name": "safe-facmac",
                "observation": {"validator_count": 4},
                "candidate": {},
                "governed": {},
                "masked": {},
                "applied": {},
                "trace": ["bad-shape"],
            },
        )

        self.assertEqual(response.status_code, 422)

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_trace_ingest_route_rejects_non_object_governed_stage(self):
        client = self._fresh_route_client()

        response = client.post(
            "/trace/ingest",
            json={
                "policy_name": "safe-facmac",
                "observation": {"validator_count": 4},
                "candidate": {},
                "governed": ["bad-shape"],
                "masked": {},
                "applied": {},
                "trace": {},
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_service_rejects_unsafe_paths(self):
        service = PolicyService()
        with self.assertRaisesRegex(ValueError, "must be a relative path inside the workspace"):
            service.train_offline("../trace.jsonl")
        with self.assertRaisesRegex(ValueError, "must be a relative path inside the workspace"):
            service.save_checkpoint("/tmp/checkpoint.json")
        with self.assertRaisesRegex(ValueError, "must be a relative path inside the workspace"):
            service.load_checkpoint("../checkpoint.json")

    def test_checkpoint_provenance_drops_trace_path_outside_trace_root(self):
        source = PolicyService()
        source.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=1, throughput_tps=4000, latency_p95_ms=200),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300), present=True),
                reward=0.5,
            )
        )
        source.train_online(batch_size=1)

        workspace_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=workspace_root) as tmp:
            path = Path(tmp) / "checkpoint.json"
            relative_path = path.relative_to(workspace_root)
            source.save_checkpoint(relative_path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["metadata"]["checkpoint_provenance"]["last_training"]["trace_path"] = "/tmp/outside-trace.jsonl"
            path.write_text(json.dumps(payload), encoding="utf-8")

            reloaded = PolicyService()
            load_result = reloaded.load_checkpoint(relative_path)
            self.assertIsNone(load_result["training"]["last_training"]["trace_path"])

    def test_checkpoint_provenance_drops_checkpoint_path_outside_checkpoint_root(self):
        source = PolicyService()
        source.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=1, throughput_tps=4000, latency_p95_ms=200),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300), present=True),
                reward=0.5,
            )
        )
        source.train_online(batch_size=1)

        workspace_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=workspace_root) as tmp:
            path = Path(tmp) / "checkpoint.json"
            relative_path = path.relative_to(workspace_root)
            source.save_checkpoint(relative_path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["metadata"]["checkpoint_provenance"]["best_saved_path"] = "/tmp/outside-checkpoint.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            reloaded = PolicyService()
            load_result = reloaded.load_checkpoint(relative_path)
            self.assertIsNone(load_result["checkpoint"]["best_saved_path"])

    def test_checkpoint_load_resets_online_sampler_state_for_reused_service(self):
        service = PolicyService(replay_capacity=2, seed=13)
        governance_sample = TrajectorySample(
            policy_name="safe-facmac",
            observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=2, throughput_tps=2000, adversary_score=0.8, churn_rate=0.5, network_jitter_ms=60),
            candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1200), present=True),
            applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1200), present=True),
            reward=1.0,
            governance_delta=True,
        )
        guardrail_sample = TrajectorySample(
            policy_name="safe-facmac",
            observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=3, throughput_tps=3000, adversary_score=0.7, churn_rate=0.45, network_jitter_ms=55),
            candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1500), present=True),
            applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1500), present=True),
            reward=2.0,
            guardrail_delta=True,
        )
        service.ingest(governance_sample)
        service.ingest(guardrail_sample)
        first_training = service.train_online(batch_size=1)
        first_sample_summary = first_training["training"]["last_training"]["sample_summary"]
        self.assertEqual(first_sample_summary["reward_mean"], 1.0)
        self.assertEqual(first_sample_summary["governance_delta_count"], 1)
        self.assertEqual(first_sample_summary["guardrail_delta_count"], 0)

        workspace_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=workspace_root) as tmp:
            path = Path(tmp) / "checkpoint.json"
            relative_path = path.relative_to(workspace_root)
            service.save_checkpoint(relative_path)
            service.load_checkpoint(relative_path)

            self.assertEqual(service.training_summary()["active_model_source"], "checkpoint_load")
            self.assertIsNone(service.training_summary()["checkpoint"]["last_saved_path"])

            service.ingest(governance_sample)
            service.ingest(guardrail_sample)

            resumed_training = service.train_online(batch_size=1)
            resumed_sample_summary = resumed_training["training"]["last_training"]["sample_summary"]
            self.assertEqual(resumed_sample_summary["reward_mean"], 1.0)
            self.assertEqual(resumed_sample_summary["governance_delta_count"], 1)
            self.assertEqual(resumed_sample_summary["guardrail_delta_count"], 0)
            self.assertEqual(resumed_sample_summary["priority_sample_count"], 1)
            self.assertEqual(resumed_sample_summary["effective_batch_size"], 1)

    def test_service_allows_nested_checkpoint_paths_inside_workspace(self):
        source = PolicyService()
        source.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=1, throughput_tps=4000, latency_p95_ms=200),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                reward=0.5,
            )
        )
        source.train_online(batch_size=1)

        workspace_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=workspace_root) as tmp:
            checkpoint_dir = Path(tmp) / "nested"
            checkpoint_dir.mkdir()
            path = checkpoint_dir / "checkpoint.json"
            relative_path = path.relative_to(workspace_root)

            save_result = source.save_checkpoint(relative_path)
            self.assertTrue(save_result["saved"])
            self.assertEqual(save_result["path"], str(relative_path))

            reloaded = PolicyService()
            load_result = reloaded.load_checkpoint(relative_path)
            self.assertTrue(load_result["loaded"])
            self.assertEqual(load_result["path"], str(relative_path))
            self.assertEqual(load_result["checkpoint"]["last_loaded_path"], str(relative_path))
            self.assertEqual(load_result["training"]["checkpoint"]["active_checkpoint_path"], str(relative_path))

    def test_checkpoint_load_drops_malicious_role_head_coverage_entries(self):
        source = PolicyService()
        source.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=1, throughput_tps=4000, latency_p95_ms=200),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                reward=0.5,
            )
        )
        source.train_online(batch_size=1)

        workspace_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=workspace_root) as tmp:
            path = Path(tmp) / "checkpoint.json"
            relative_path = path.relative_to(workspace_root)
            source.save_checkpoint(relative_path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["metadata"]["checkpoint_provenance"]["last_training"]["trainer_metadata"]["role_head_coverage"] = {
                "membership_tuner": {"active_samples": 3, "total_samples": 5},
                "malicious_role": {"active_samples": 99, "total_samples": 99},
                "lane_tuner": "bad-shape",
            }
            path.write_text(json.dumps(payload), encoding="utf-8")

            reloaded = PolicyService()
            load_result = reloaded.load_checkpoint(relative_path)
            restored_coverage = load_result["training"]["last_training"]["trainer_metadata"]["role_head_coverage"]
            self.assertEqual(restored_coverage, {"membership_tuner": {"active_samples": 3.0, "total_samples": 5.0}})
            self.assertNotIn("malicious_role", restored_coverage)
            self.assertEqual(reloaded.model_snapshot()["metadata_provenance"], "checkpoint")

    def test_checkpoint_load_legacy_online_summary_restores_requested_batch_without_sample_summary(self):
        source = PolicyService()
        source.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=1, throughput_tps=4000, latency_p95_ms=200),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                reward=0.5,
            )
        )
        source.train_online(batch_size=3)

        workspace_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=workspace_root) as tmp:
            path = Path(tmp) / "checkpoint.json"
            relative_path = path.relative_to(workspace_root)
            source.save_checkpoint(relative_path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            last_training = payload["metadata"]["checkpoint_provenance"]["last_training"]
            last_training["batch_size"] = 3
            last_training.pop("sample_summary", None)
            path.write_text(json.dumps(payload), encoding="utf-8")

            reloaded = PolicyService()
            load_result = reloaded.load_checkpoint(relative_path)
            restored = load_result["training"]["last_training"]
            self.assertEqual(restored["mode"], "online")
            self.assertEqual(restored["batch_size"], 3)
            self.assertIn("sample_summary", restored)
            self.assertEqual(restored["sample_summary"]["requested_batch_size"], 3)
            self.assertEqual(restored["sample_summary"]["effective_batch_size"], 0)
            self.assertEqual(restored["sample_summary"]["priority_sample_count"], 0)

    def test_checkpoint_load_restores_research_service_claim_metadata(self):
        source = PolicyService()
        source.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=1, throughput_tps=4000, latency_p95_ms=200),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                reward=0.5,
            )
        )
        source.train_online(batch_size=1)

        workspace_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=workspace_root) as tmp:
            path = Path(tmp) / "checkpoint.json"
            relative_path = path.relative_to(workspace_root)
            source.save_checkpoint(relative_path)

            reloaded = PolicyService()
            load_result = reloaded.load_checkpoint(relative_path)
            metadata = reloaded.model_snapshot()["metadata"]
            trainer_metadata = load_result["training"]["last_training"]["trainer_metadata"]
            artifact_inventory = reloaded.artifact_inventory()
            adaptive_snapshot = reloaded.adaptive_snapshot()

            self.assertEqual(metadata["trainer"], "safe-facmac-ctde")
            self.assertEqual(metadata["trainer_family"], "CTDE-monotonic-mixing-with-safety-filter")
            self.assertTrue(metadata["paper_grade_facmac"])
            self.assertEqual(metadata["claim_boundary"], "monotonic_mixing_igm_zero")
            self.assertEqual(trainer_metadata["trainer_family"], "CTDE-monotonic-mixing-with-safety-filter")
            self.assertTrue(trainer_metadata["paper_grade_facmac"])
            self.assertEqual(trainer_metadata["claim_boundary"], "monotonic_mixing_igm_zero")
            self.assertEqual(artifact_inventory["trainer_family"], "CTDE-monotonic-mixing-with-safety-filter")
            self.assertTrue(artifact_inventory["paper_grade_facmac"])
            self.assertEqual(artifact_inventory["claim_boundary"], "monotonic_mixing_igm_zero")
            self.assertEqual(artifact_inventory["checkpoint_metadata"]["best_saved_path"], artifact_inventory["checkpoint"]["best_saved_path"])
            self.assertEqual(artifact_inventory["checkpoint_metadata"]["best_reward_mean"], artifact_inventory["checkpoint"]["best_reward_mean"])
            self.assertEqual(artifact_inventory["checkpoint_metadata"]["last_training"], load_result["training"]["last_training"])
            self.assertEqual(artifact_inventory["checkpoint_metadata"]["trainer_family"], "CTDE-monotonic-mixing-with-safety-filter")
            self.assertTrue(artifact_inventory["checkpoint_metadata"]["paper_grade_facmac"])
            self.assertEqual(artifact_inventory["checkpoint_metadata"]["claim_boundary"], "monotonic_mixing_igm_zero")
            self.assertEqual(artifact_inventory["training_metadata"], load_result["training"]["last_training"])
            self.assertEqual(adaptive_snapshot["artifact_inventory"], artifact_inventory)
            self.assertEqual(
                adaptive_snapshot["schema"]["artifact_inventory_fields"],
                [
                    "schema_version",
                    "active_model_source",
                    "model_ready",
                    "replay_ready",
                    "replay_size",
                    "checkpoint",
                    "checkpoint_metadata",
                    "training_metadata",
                    "has_training_summary",
                    "has_activation_summary",
                    "claim_boundary",
                    "trainer_family",
                    "paper_grade_facmac",
                    "target_action",
                ],
            )
            self.assertEqual(
                adaptive_snapshot["schema"]["artifact_inventory_checkpoint_fields"],
                ["best_reward_mean", "best_saved_path", "last_loaded_path", "last_saved_path"],
            )
            self.assertEqual(
                adaptive_snapshot["schema"]["checkpoint_training_summary_fields"],
                [
                    "batch_size",
                    "mode",
                    "replay_priority_summary",
                    "replay_size",
                    "sample_summary",
                    "samples",
                    "scenario_label",
                    "schema_version",
                    "seed",
                    "trace_manifest_path",
                    "trace_path",
                    "trainer_metadata",
                ],
            )

    def test_checkpoint_load_sanitizes_restored_trainer_metadata_allowlist(self):
        source = PolicyService()
        source.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=1, throughput_tps=4000, latency_p95_ms=200),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                reward=0.5,
            )
        )
        source.train_online(batch_size=1)

        workspace_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=workspace_root) as tmp:
            path = Path(tmp) / "checkpoint.json"
            relative_path = path.relative_to(workspace_root)
            source.save_checkpoint(relative_path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            trainer_metadata = payload["metadata"]["checkpoint_provenance"]["last_training"]["trainer_metadata"]
            trainer_metadata.update(
                {
                    "trainer_family": "facmac",
                    "paper_grade_facmac": True,
                    "claim_boundary": "research_asset_only",
                    "epochs": "7",
                    "actor_lr": "0.001",
                    "critic_lr": "inf",
                    "bc_lr": "not-a-number",
                    "target_action": "governed_stage_preferred",
                    "history": {"loss": [0.5, "bad", "nan"]},
                    "reward_summary": {"reward_mean": 1.5, "team_reward_mean": "bad", "role_reward_mean": 0.2},
                    "trainer_config": {"seed": "11", "epochs": "9", "ridge": "0.01", "actor_lr": "0.0005", "bad": 1},
                    "unexpected_secret": "drop-me",
                }
            )
            path.write_text(json.dumps(payload), encoding="utf-8")

            reloaded = PolicyService()
            load_result = reloaded.load_checkpoint(relative_path)
            restored = load_result["training"]["last_training"]["trainer_metadata"]
            self.assertEqual(restored["trainer_family"], "facmac")
            self.assertTrue(restored["paper_grade_facmac"])
            self.assertEqual(restored["claim_boundary"], "research_asset_only")
            self.assertEqual(restored["epochs"], 7)
            self.assertEqual(restored["actor_lr"], 0.001)
            self.assertNotIn("critic_lr", restored)
            self.assertNotIn("bc_lr", restored)
            self.assertEqual(restored["target_action"], "governed_stage_preferred")
            self.assertEqual(restored["history"], {"loss": [0.5]})
            self.assertEqual(restored["reward_summary"], {"reward_mean": 1.5, "team_reward_mean": 0.0, "role_reward_mean": 0.2})
            self.assertEqual(restored["trainer_config"], {"seed": 11, "epochs": 9, "ridge": 0.01, "actor_lr": 0.0005})
            self.assertNotIn("unexpected_secret", restored)

    def test_checkpoint_load_without_provenance_uses_fallback_training_summary(self):
        source = PolicyService()
        source.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=1, throughput_tps=4000, latency_p95_ms=200),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                reward=0.5,
            )
        )
        source.train_online(batch_size=1)

        workspace_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=workspace_root) as tmp:
            path = Path(tmp) / "checkpoint.json"
            relative_path = path.relative_to(workspace_root)
            source.save_checkpoint(relative_path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["metadata"].pop("checkpoint_provenance", None)
            payload["metadata"]["trainer_family"] = "facmac"
            payload["metadata"]["paper_grade_facmac"] = True
            payload["metadata"]["claim_boundary"] = "research_asset_only"
            payload["metadata"]["unexpected_secret"] = "drop-me"
            path.write_text(json.dumps(payload), encoding="utf-8")

            reloaded = PolicyService()
            load_result = reloaded.load_checkpoint(relative_path)
            restored = load_result["training"]["last_training"]
            self.assertEqual(restored["mode"], "checkpoint_origin_unknown")
            self.assertEqual(restored["samples"], 0)
            self.assertEqual(restored["replay_size"], 0)
            self.assertEqual(restored["schema_version"], SCHEMA_VERSION)
            self.assertEqual(restored["replay_priority_summary"], {
                "priority_sample_count": 0,
                "governance_delta_count": 0,
                "guardrail_delta_count": 0,
                "stress_signal_count": 0,
            })
            self.assertEqual(restored["sample_summary"], {
                "requested_batch_size": None,
                "effective_batch_size": 0,
                "priority_sample_count": 0,
                "governance_delta_count": 0,
                "guardrail_delta_count": 0,
                "stress_signal_count": 0,
                "reward_mean": 0.0,
            })
            self.assertNotIn("trainer_family", restored["trainer_metadata"])
            self.assertNotIn("paper_grade_facmac", restored["trainer_metadata"])
            self.assertNotIn("claim_boundary", restored["trainer_metadata"])
            self.assertNotIn("unexpected_secret", restored["trainer_metadata"])
            self.assertIsNone(reloaded.model_snapshot()["metadata"].get("trainer_family"))
            self.assertIsNone(reloaded.model_snapshot()["metadata"].get("paper_grade_facmac"))
            self.assertIsNone(reloaded.model_snapshot()["metadata"].get("claim_boundary"))
            self.assertNotIn("unexpected_secret", reloaded.model_snapshot()["metadata"])
            self.assertEqual(load_result["training"]["active_model_source"], "checkpoint_load")
            self.assertEqual(load_result["training"]["checkpoint"]["active_checkpoint_path"], str(relative_path))

        source = PolicyService()
        source.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=1, throughput_tps=4000, latency_p95_ms=200),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                reward=0.5,
            )
        )
        source.train_online(batch_size=1)

        workspace_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=workspace_root) as tmp:
            path = Path(tmp) / "checkpoint.json"
            relative_path = path.relative_to(workspace_root)
            source.save_checkpoint(relative_path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["metadata"]["checkpoint_provenance"]["last_training"]["schema_version"] = "evolvbft-adaptive-v0"
            path.write_text(json.dumps(payload), encoding="utf-8")

            reloaded = PolicyService()
            with self.assertRaisesRegex(ValueError, "unsupported checkpoint schema_version: evolvbft-adaptive-v0"):
                reloaded.load_checkpoint(relative_path)

    def test_checkpoint_load_without_provenance_rejects_untrusted_claim_metadata(self):
        source = PolicyService()
        source.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=1, throughput_tps=4000, latency_p95_ms=200),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                reward=0.5,
            )
        )
        source.train_online(batch_size=1)

        workspace_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=workspace_root) as tmp:
            path = Path(tmp) / "checkpoint.json"
            relative_path = path.relative_to(workspace_root)
            source.save_checkpoint(relative_path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["metadata"].pop("checkpoint_provenance", None)
            payload["metadata"]["trainer_family"] = "facmac"
            payload["metadata"]["paper_grade_facmac"] = True
            payload["metadata"]["claim_boundary"] = "research_asset_only"
            payload["metadata"]["unexpected_secret"] = "drop-me"
            path.write_text(json.dumps(payload), encoding="utf-8")

            reloaded = PolicyService()
            load_result = reloaded.load_checkpoint(relative_path)
            restored = load_result["training"]["last_training"]
            self.assertEqual(restored["mode"], "checkpoint_origin_unknown")
            self.assertNotIn("trainer_family", restored["trainer_metadata"])
            self.assertNotIn("paper_grade_facmac", restored["trainer_metadata"])
            self.assertNotIn("claim_boundary", restored["trainer_metadata"])
            self.assertIsNone(reloaded.model_snapshot()["metadata"].get("trainer_family"))
            self.assertIsNone(reloaded.model_snapshot()["metadata"].get("paper_grade_facmac"))
            self.assertIsNone(reloaded.model_snapshot()["metadata"].get("claim_boundary"))
            self.assertNotIn("unexpected_secret", reloaded.model_snapshot()["metadata"])

    def test_checkpoint_load_rejects_future_training_schema_version(self):
        source = PolicyService()
        source.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=1, throughput_tps=4000, latency_p95_ms=200),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                reward=0.5,
            )
        )
        source.train_online(batch_size=1)

        workspace_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=workspace_root) as tmp:
            path = Path(tmp) / "checkpoint.json"
            relative_path = path.relative_to(workspace_root)
            source.save_checkpoint(relative_path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["metadata"].pop("checkpoint_provenance", None)
            payload["metadata"]["trainer_family"] = "facmac"
            payload["metadata"]["paper_grade_facmac"] = True
            payload["metadata"]["claim_boundary"] = "research_asset_only"
            payload["metadata"]["unexpected_secret"] = "drop-me"
            path.write_text(json.dumps(payload), encoding="utf-8")

            reloaded = PolicyService()
            load_result = reloaded.load_checkpoint(relative_path)
            restored = load_result["training"]["last_training"]
            self.assertEqual(restored["mode"], "checkpoint_origin_unknown")
            self.assertNotIn("trainer_family", restored["trainer_metadata"])
            self.assertNotIn("paper_grade_facmac", restored["trainer_metadata"])
            self.assertNotIn("claim_boundary", restored["trainer_metadata"])
            self.assertIsNone(reloaded.model_snapshot()["metadata"].get("trainer_family"))
            self.assertIsNone(reloaded.model_snapshot()["metadata"].get("paper_grade_facmac"))
            self.assertIsNone(reloaded.model_snapshot()["metadata"].get("claim_boundary"))
            self.assertNotIn("unexpected_secret", reloaded.model_snapshot()["metadata"])

    def test_checkpoint_load_rejects_future_training_schema_version(self):
        source = PolicyService()
        source.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=1, throughput_tps=4000, latency_p95_ms=200),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                reward=0.5,
            )
        )
        source.train_online(batch_size=1)

        workspace_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=workspace_root) as tmp:
            path = Path(tmp) / "checkpoint.json"
            relative_path = path.relative_to(workspace_root)
            source.save_checkpoint(relative_path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["metadata"]["checkpoint_provenance"]["last_training"]["schema_version"] = "evolvbft-adaptive-v2"
            path.write_text(json.dumps(payload), encoding="utf-8")

            reloaded = PolicyService()
            with self.assertRaisesRegex(ValueError, "unsupported checkpoint schema_version: evolvbft-adaptive-v2"):
                reloaded.load_checkpoint(relative_path)

    def test_checkpoint_load_defaults_missing_training_schema_version_to_current(self):
        source = PolicyService()
        source.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=1, throughput_tps=4000, latency_p95_ms=200),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                reward=0.5,
            )
        )
        source.train_online(batch_size=1)

        workspace_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=workspace_root) as tmp:
            path = Path(tmp) / "checkpoint.json"
            relative_path = path.relative_to(workspace_root)
            source.save_checkpoint(relative_path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            del payload["metadata"]["checkpoint_provenance"]["last_training"]["schema_version"]
            path.write_text(json.dumps(payload), encoding="utf-8")

            reloaded = PolicyService()
            load_result = reloaded.load_checkpoint(relative_path)
            self.assertEqual(load_result["training"]["last_training"]["schema_version"], SCHEMA_VERSION)

    def test_fresh_checkpoint_load_can_continue_with_new_replay(self):
        source = PolicyService()
        source.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=1, throughput_tps=4000, latency_p95_ms=200),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                reward=0.5,
            )
        )
        source.train_online(batch_size=1)

        workspace_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=workspace_root) as tmp:
            path = Path(tmp) / "checkpoint.json"
            relative_path = path.relative_to(workspace_root)
            source.save_checkpoint(relative_path)

            reloaded = PolicyService()
            reloaded.load_checkpoint(relative_path)
            self.assertEqual(reloaded.replay_summary()["replay_size"], 0)
            empty_train = reloaded.train_online(batch_size=1)
            self.assertEqual(empty_train["samples"], 0)
            self.assertEqual(empty_train["training"]["active_model_source"], "checkpoint_load")
            new_observation = Observation(validator_count=8, current_config_id=1, highest_known_config_id=2, throughput_tps=4500, latency_p95_ms=180)
            flow_before_online = reloaded.inspect_decision_flow(new_observation)
            behavior_before_online = reloaded.infer(new_observation).to_dict()
            snapshot_before_online = reloaded.model_snapshot()

            ingest_result = reloaded.ingest(
                TrajectorySample(
                    policy_name="safe-facmac",
                    observation=new_observation,
                    candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1200, mempool_max_batch_txs=1536, hydra_discovery_target=2), present=True),
                    applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1200, mempool_max_batch_txs=1536, hydra_discovery_target=2), present=True),
                    reward=1.0,
                )
            )
            self.assertEqual(ingest_result["replay_size"], 1)
            online_summary = reloaded.train_online(batch_size=1)
            flow_after_online = reloaded.inspect_decision_flow(new_observation)
            behavior_after_online = reloaded.infer(new_observation).to_dict()
            snapshot_after_online = reloaded.model_snapshot()
            self.assertEqual(online_summary["samples"], 1)
            self.assertEqual(online_summary["training"]["last_training"]["mode"], "online")
            self.assertEqual(online_summary["training"]["last_training"]["samples"], 1)
            self.assertEqual(online_summary["training"]["last_training"]["batch_size"], 1)
            self.assertEqual(online_summary["training"]["last_training"]["replay_size"], 1)
            self.assertEqual(online_summary["training"]["last_activation"]["mode"], "online_training")
            self.assertEqual(online_summary["training"]["active_model_source"], "online_training")
            self.assertIsNone(online_summary["training"]["checkpoint"]["active_checkpoint_path"])
            self.assertEqual(reloaded.replay_summary()["replay_size"], 1)
            self.assertNotEqual(snapshot_before_online["actor_bias"], snapshot_after_online["actor_bias"])
            control_fields = (
                "committee_size",
                "pacemaker_timeout_ms",
                "mempool_max_batch_txs",
                "mempool_proposal_interval_ms",
                "submit_join",
                "submit_leave",
                "hydra_discovery_target",
            )
            self.assertEqual(flow_before_online["applied"], behavior_before_online)
            self.assertEqual(flow_after_online["applied"], behavior_after_online)
            self.assertNotEqual(
                {field: behavior_before_online[field] for field in control_fields},
                {field: behavior_after_online[field] for field in control_fields},
            )
            self.assertNotEqual(
                {field: flow_before_online["candidate"][field] for field in control_fields},
                {field: flow_after_online["candidate"][field] for field in control_fields},
            )
            self.assertEqual(set(flow_before_online["divergence"]), set(flow_after_online["divergence"]))
            for flow in (flow_before_online, flow_after_online):
                self.assertEqual(flow["divergence"]["candidate_vs_governed"], bool(flow["divergence"]["candidate_vs_governed_fields"]))
                self.assertEqual(flow["divergence"]["governed_vs_applied"], bool(flow["divergence"]["governed_vs_applied_fields"]))
                self.assertEqual(flow["divergence"]["candidate_vs_applied"], bool(flow["divergence"]["candidate_vs_applied_fields"]))

    def test_checkpoint_loaded_service_noop_online_train_does_not_drift(self):
        source = PolicyService()
        source.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=1, throughput_tps=4000, latency_p95_ms=200),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                reward=0.5,
            )
        )
        source.train_online(batch_size=1)

        workspace_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=workspace_root) as tmp:
            path = Path(tmp) / "checkpoint.json"
            relative_path = path.relative_to(workspace_root)
            source.save_checkpoint(relative_path)

            reloaded = PolicyService()
            reloaded.load_checkpoint(relative_path)
            observation = Observation(
                validator_count=8,
                current_config_id=1,
                highest_known_config_id=2,
                throughput_tps=4500,
                latency_p95_ms=180,
            )
            flow_before = reloaded.inspect_decision_flow(observation)
            behavior_before = reloaded.infer(observation).to_dict()
            snapshot_before = reloaded.model_snapshot()

            empty_train = reloaded.train_online(batch_size=1)

            flow_after = reloaded.inspect_decision_flow(observation)
            behavior_after = reloaded.infer(observation).to_dict()
            snapshot_after = reloaded.model_snapshot()
            self.assertEqual(empty_train["samples"], 0)
            self.assertEqual(empty_train["training"]["active_model_source"], "checkpoint_load")
            self.assertEqual(empty_train["training"]["last_activation"]["mode"], "checkpoint_load")
            self.assertEqual(empty_train["training"]["checkpoint"]["active_checkpoint_path"], str(relative_path))
            control_fields = (
                "committee_size",
                "pacemaker_timeout_ms",
                "mempool_max_batch_txs",
                "mempool_proposal_interval_ms",
                "submit_join",
                "submit_leave",
                "hydra_discovery_target",
            )
            for stage in ("candidate", "governed", "applied"):
                self.assertEqual(
                    {field: flow_before[stage][field] for field in control_fields},
                    {field: flow_after[stage][field] for field in control_fields},
                )
            self.assertEqual(flow_before["divergence"], flow_after["divergence"])
            self.assertEqual(
                {field: behavior_before[field] for field in control_fields},
                {field: behavior_after[field] for field in control_fields},
            )
            for field in ("actor_weights", "actor_bias", "critic_weights", "critic_bias"):
                self.assertEqual(snapshot_before[field], snapshot_after[field])
            self.assertEqual(snapshot_after["service_training"]["active_model_source"], "checkpoint_load")

            priority_governance = TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=2, throughput_tps=2000, adversary_score=0.8, churn_rate=0.5, network_jitter_ms=60),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1200), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1200), present=True),
                reward=1.0,
                governance_delta=True,
            )
            priority_guardrail = TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=3, throughput_tps=3000, adversary_score=0.7, churn_rate=0.45, network_jitter_ms=55),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1500), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1500), present=True),
                reward=2.0,
                guardrail_delta=True,
            )

            control = PolicyService(replay_capacity=2, seed=13)
            control.load_checkpoint(relative_path)
            control.ingest(priority_governance)
            control.ingest(priority_guardrail)
            control_summary = control.train_online(batch_size=1)["training"]["last_training"]["sample_summary"]

            probe = PolicyService(replay_capacity=2, seed=13)
            probe.load_checkpoint(relative_path)
            empty_probe = probe.train_online(batch_size=1)
            self.assertEqual(empty_probe["samples"], 0)
            probe.ingest(priority_governance)
            probe.ingest(priority_guardrail)
            probe_summary = probe.train_online(batch_size=1)["training"]["last_training"]["sample_summary"]

            self.assertEqual(probe_summary, control_summary)

    def test_service_model_snapshot_exposes_scoped_service_metadata(self):
        service = PolicyService()
        service.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=1, throughput_tps=4000, latency_p95_ms=200),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                reward=0.5,
            )
        )
        service.train_online(batch_size=1)

        snapshot = service.model_snapshot()

        self.assertEqual(snapshot["metadata_provenance"], "trainer")
        self.assertEqual(snapshot["service_training"]["active_model_source"], "online_training")
        self.assertIn("trainer_config", snapshot["metadata"])
        self.assertIn("reward_summary", snapshot["metadata"])
        self.assertNotIn("checkpoint_provenance", snapshot["metadata"])
        self.assertEqual(snapshot["checkpoint"]["last_saved_path"], None)

    def test_schema_snapshot_exposes_bridge_contract_fields(self):
        snapshot = PolicyService().schema_snapshot()

        self.assertEqual(snapshot["schema_version"], SCHEMA_VERSION)
        self.assertIn("last_ordered_lane_id", snapshot["observation_fields"])
        self.assertIn("last_ordered_config_id", snapshot["observation_fields"])
        self.assertIn("hydra_discovery_target", snapshot["action_fields"])
        self.assertEqual(snapshot["decision_stages"], ["candidate", "governed", "masked", "applied"])
        self.assertIn("checkpoint_metadata", snapshot["artifact_inventory_fields"])
        self.assertIn("training_metadata", snapshot["artifact_inventory_fields"])

    def test_schema_snapshot_exposes_summary_field_contracts(self):
        service = PolicyService()

        snapshot = service.schema_snapshot()

        self.assertEqual(snapshot["replay_summary_fields"], [
            "replay_size",
            "governance_delta_rate",
            "guardrail_delta_rate",
            "divergence_rates",
            "reward_summary",
            "priority_summary",
        ])
        self.assertEqual(snapshot["training_summary_fields"], [
            "last_training",
            "last_activation",
            "checkpoint",
            "model_ready",
            "active_model_source",
        ])
        self.assertEqual(snapshot["decision_fields"], [
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
        ])

    def test_replay_summary_reports_zero_role_reward_mean_without_role_rewards(self):
        service = PolicyService()
        service.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=2),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=900), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=900), present=True),
                reward=1.0,
                team_reward=2.0,
            )
        )

        summary = service.replay_summary()

        self.assertEqual(summary["reward_summary"]["reward_mean"], 1.0)
        self.assertEqual(summary["reward_summary"]["team_reward_mean"], 2.0)
        self.assertEqual(summary["reward_summary"]["role_reward_mean"], 0.0)

    def test_replay_summary_aggregates_role_reward_means_across_samples(self):
        service = PolicyService()
        service.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=2),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=900), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=900), present=True),
                reward=1.0,
                role_rewards={"membership_tuner": 2.0, "lane_tuner": 4.0},
            )
        )
        service.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=2),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1000), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1000), present=True),
                reward=3.0,
                role_rewards={"membership_tuner": 6.0},
            )
        )

        summary = service.replay_summary()

        self.assertEqual(summary["reward_summary"]["role_reward_mean"], 4.5)

    def test_service_inspect_replay_sample_preserves_recorded_missing_governed_contract(self):
        service = PolicyService()
        service.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=2),
                candidate=DecisionActionStage(action=Action(committee_size=6, pacemaker_timeout_ms=900), present=True),
                applied=DecisionActionStage(action=Action(committee_size=4, pacemaker_timeout_ms=1200), present=True),
                reward=1.0,
            )
        )

        inspected = service.inspect_replay_sample(0)

        self.assertTrue(inspected["recorded"]["candidate_present"])
        self.assertFalse(inspected["recorded"]["governed_present"])
        self.assertIsNone(inspected["recorded"]["governed"])
        self.assertIsNone(inspected["divergence"]["recorded_governed_vs_current_governed"])
        self.assertIsNone(inspected["divergence"]["recorded_governed_vs_current_governed_fields"])

    def test_replay_summary_ignores_missing_governed_stage(self):
        service = PolicyService()
        service.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=2),
                candidate=DecisionActionStage(action=Action(committee_size=6, pacemaker_timeout_ms=900), present=True),
                applied=DecisionActionStage(action=Action(committee_size=4, pacemaker_timeout_ms=1200), present=True),
                reward=1.0,
            )
        )

        summary = service.replay_summary()

        self.assertEqual(summary["divergence_rates"]["candidate_vs_governed"], 0.0)
        self.assertEqual(summary["divergence_rates"]["governed_vs_applied"], 0.0)
        self.assertEqual(summary["divergence_rates"]["candidate_vs_applied"], 1.0)

    def test_replay_summary_counts_explicit_default_governed_stage(self):
        service = PolicyService()
        service.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=2),
                candidate=DecisionActionStage(action=Action(committee_size=6, pacemaker_timeout_ms=900), present=True),
                governed=DecisionActionStage(action=Action(), present=True),
                applied=DecisionActionStage(action=Action(committee_size=4, pacemaker_timeout_ms=1200), present=True),
                reward=1.0,
            )
        )

        summary = service.replay_summary()

        self.assertEqual(summary["divergence_rates"]["candidate_vs_governed"], 1.0)
        self.assertEqual(summary["divergence_rates"]["governed_vs_applied"], 1.0)

    def test_service_replay_summary_exposes_priority_composition(self):
        service = PolicyService()
        service.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(
                    validator_count=8,
                    current_config_id=1,
                    highest_known_config_id=2,
                    adversary_score=0.8,
                    churn_rate=0.5,
                    network_jitter_ms=60,
                ),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=900), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=900), present=True),
                reward=1.0,
                governance_delta=True,
            )
        )
        service.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=2),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1000), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1000), present=True),
                reward=0.5,
                guardrail_delta=True,
            )
        )
        service.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=2),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1100), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1100), present=True),
                reward=0.1,
            )
        )

        summary = service.replay_summary()

        self.assertEqual(summary["priority_summary"]["priority_sample_count"], 2)
        self.assertEqual(summary["priority_summary"]["governance_delta_count"], 1)
        self.assertEqual(summary["priority_summary"]["guardrail_delta_count"], 1)
        self.assertEqual(summary["priority_summary"]["stress_signal_count"], 1)

    def test_named_adversarial_scenario_drives_critical_organization_state(self):
        service = PolicyService()
        scenario = AIOT_NAMED_SCENARIOS["adversarial_safety_stress"]
        observation = Observation(
            validator_count=8,
            current_config_id=1,
            highest_known_config_id=1,
            committee_size=4,
            pacemaker_timeout_ms=1000,
            mempool_max_batch_txs=1024,
            mempool_proposal_interval_ms=80,
            throughput_tps=2500,
            latency_p95_ms=420,
            backlog_pending=0,
            backlog_missing=3,
            reject_total=2,
            pending_joins=0,
            pending_leaves=0,
            lset_size=3,
            can_participate=True,
            local_validator=True,
            heterogeneity_score=scenario["heterogeneity_score"],
            churn_rate=scenario["churn_rate"],
            adversary_score=scenario["adversary_score"],
            network_jitter_ms=scenario["network_jitter_ms"],
            ai_load_score=scenario["ai_load_score"],
        )

        flow = service.inspect_decision_flow(observation)
        organization_decision = flow["organization_decision"]
        expected_escalation = "critical" if (
            observation.reject_total > 10 or observation.backlog_missing > 10 or observation.adversary_score >= 0.7
        ) else "elevated"

        self.assertEqual(organization_decision["escalation_level"], expected_escalation)
        self.assertIn("recovery_tuner", organization_decision["active_roles"])
        self.assertIn("safety_guardian", organization_decision["active_roles"])
        self.assertTrue(organization_decision["freeze_membership"])
        self.assertTrue(organization_decision["freeze_lane_tuning"])
        self.assertIn("submit_join", organization_decision["blocked_fields"])
        self.assertIn("committee_size", organization_decision["blocked_fields"])
        self.assertEqual(flow["governed"]["committee_size"], observation.committee_size)
        self.assertEqual(flow["governed"]["hydra_discovery_target"], 0)
        self.assertFalse(flow["divergence"]["candidate_vs_governed"])
        self.assertEqual(flow["divergence"]["candidate_vs_governed_fields"], [])
        guarded_projection = flow["explanation_details"]["candidate_to_applied"]
        self.assertEqual(
            guarded_projection["blocked_by_guardrails"],
            sorted(set(organization_decision["blocked_fields"])),
        )
        self.assertIn("membership-frozen", guarded_projection["organization_notes"])
        self.assertIn("lane-tuning-frozen", guarded_projection["organization_notes"])
        self.assertIn("safety-escalation:critical", guarded_projection["organization_notes"])
        self.assertIn("lane_tuner", guarded_projection["attributed_roles"])
        self.assertIn("recovery_tuner", guarded_projection["attributed_roles"])
        self.assertIn("safety_guardian", guarded_projection["attributed_roles"])

    def test_fresh_checkpoint_load_supports_online_update_from_aiot_stress_replay(self):
        source = PolicyService()
        source.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(
                    validator_count=8,
                    current_config_id=1,
                    highest_known_config_id=1,
                    committee_size=4,
                    pacemaker_timeout_ms=1000,
                    mempool_max_batch_txs=1024,
                    mempool_proposal_interval_ms=80,
                    throughput_tps=9000,
                    latency_p95_ms=90,
                    backlog_pending=12,
                    backlog_missing=0,
                    reject_total=0,
                    pending_joins=0,
                    pending_leaves=0,
                    lset_size=3,
                    can_participate=True,
                    local_validator=False,
                    heterogeneity_score=0.2,
                    churn_rate=0.05,
                    adversary_score=0.0,
                    network_jitter_ms=8,
                    ai_load_score=0.3,
                ),
                candidate=DecisionActionStage(
                    action=Action(
                        committee_size=4,
                        pacemaker_timeout_ms=900,
                        mempool_max_batch_txs=1536,
                        mempool_proposal_interval_ms=60,
                        hydra_discovery_target=1,
                    ),
                    present=True,
                ),
                applied=DecisionActionStage(
                    action=Action(
                        committee_size=4,
                        pacemaker_timeout_ms=900,
                        mempool_max_batch_txs=1536,
                        mempool_proposal_interval_ms=60,
                        hydra_discovery_target=1,
                    ),
                    present=True,
                ),
                reward=1.0,
                team_reward=0.8,
            )
        )
        source.train_online(batch_size=1)

        workspace_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=workspace_root) as tmp:
            path = Path(tmp) / "checkpoint.json"
            relative_path = path.relative_to(workspace_root)
            source.save_checkpoint(relative_path)

            reloaded = PolicyService()
            reloaded.load_checkpoint(relative_path)
            pre_stress_observation = Observation(
                validator_count=8,
                current_config_id=1,
                highest_known_config_id=2,
                committee_size=4,
                pacemaker_timeout_ms=1000,
                mempool_max_batch_txs=1024,
                mempool_proposal_interval_ms=80,
                throughput_tps=8800,
                latency_p95_ms=95,
                backlog_pending=16,
                backlog_missing=0,
                reject_total=0,
                pending_joins=0,
                pending_leaves=0,
                lset_size=3,
                can_participate=True,
                local_validator=False,
                heterogeneity_score=0.25,
                churn_rate=0.08,
                adversary_score=0.0,
                network_jitter_ms=10,
                ai_load_score=0.35,
            )
            stress_observation = Observation(
                validator_count=8,
                current_config_id=1,
                highest_known_config_id=2,
                committee_size=4,
                pacemaker_timeout_ms=1000,
                mempool_max_batch_txs=1024,
                mempool_proposal_interval_ms=80,
                throughput_tps=4200,
                latency_p95_ms=640,
                backlog_pending=384,
                backlog_missing=5,
                reject_total=7,
                pending_joins=0,
                pending_leaves=0,
                lset_size=3,
                can_participate=True,
                local_validator=False,
                heterogeneity_score=0.45,
                churn_rate=0.25,
                adversary_score=0.7,
                network_jitter_ms=70,
                ai_load_score=0.85,
            )
            before_stress = reloaded.infer(pre_stress_observation).to_dict()
            flow_before_training = reloaded.inspect_decision_flow(stress_observation)

            reloaded.ingest(
                TrajectorySample(
                    policy_name="safe-facmac",
                    observation=stress_observation,
                    candidate=DecisionActionStage(
                        action=Action(
                            committee_size=4,
                            pacemaker_timeout_ms=1600,
                            mempool_max_batch_txs=512,
                            mempool_proposal_interval_ms=140,
                            hydra_discovery_target=2,
                        ),
                        present=True,
                    ),
                    applied=DecisionActionStage(
                        action=Action(
                            committee_size=4,
                            pacemaker_timeout_ms=1600,
                            mempool_max_batch_txs=512,
                            mempool_proposal_interval_ms=140,
                            hydra_discovery_target=2,
                        ),
                        present=True,
                    ),
                    reward=-1.0,
                    team_reward=-0.9,
                    governance_delta=True,
                    guardrail_delta=True,
                )
            )
            update_summary = reloaded.train_online(batch_size=1)
            flow_after_training = reloaded.inspect_decision_flow(stress_observation)
            after_stress = reloaded.infer(stress_observation).to_dict()
            self.assertEqual(update_summary["samples"], 1)
            self.assertEqual(update_summary["training"]["last_training"]["mode"], "online")
            self.assertEqual(update_summary["training"]["last_training"]["sample_summary"]["effective_batch_size"], 1)
            self.assertEqual(update_summary["training"]["last_training"]["sample_summary"]["priority_sample_count"], 1)
            self.assertEqual(update_summary["training"]["last_training"]["sample_summary"]["governance_delta_count"], 1)
            self.assertEqual(update_summary["training"]["last_training"]["sample_summary"]["guardrail_delta_count"], 1)
            self.assertEqual(flow_before_training["organization_decision"]["escalation_level"], "critical")
            self.assertEqual(flow_after_training["organization_decision"]["escalation_level"], "critical")
            self.assertTrue(flow_after_training["organization_decision"]["freeze_membership"])
            self.assertTrue(flow_after_training["organization_decision"]["freeze_lane_tuning"])
            self.assertEqual(flow_after_training["applied"], after_stress)
            self.assertNotEqual(before_stress["pacemaker_timeout_ms"], after_stress["pacemaker_timeout_ms"])
            self.assertNotEqual(
                {field: flow_before_training["candidate"][field] for field in ("committee_size", "pacemaker_timeout_ms", "mempool_max_batch_txs", "mempool_proposal_interval_ms", "hydra_discovery_target")},
                {field: flow_after_training["candidate"][field] for field in ("committee_size", "pacemaker_timeout_ms", "mempool_max_batch_txs", "mempool_proposal_interval_ms", "hydra_discovery_target")},
            )

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_schema_route_exposes_bounded_service_snapshot_fields(self):
        client = self._fresh_route_client()
        before_replay = app_module.service.replay_summary()
        before_training = app_module.service.training_summary()

        response = client.get("/schema")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
        self.assertEqual(
            payload["replay_summary_fields"],
            [
                "replay_size",
                "governance_delta_rate",
                "guardrail_delta_rate",
                "divergence_rates",
                "reward_summary",
                "priority_summary",
            ],
        )
        self.assertEqual(
            payload["training_summary_fields"],
            ["last_training", "last_activation", "checkpoint", "model_ready", "active_model_source"],
        )
        self.assertNotIn("routes", payload)
        self.assertEqual(app_module.service.replay_summary(), before_replay)
        self.assertEqual(app_module.service.training_summary(), before_training)

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_adaptive_route_exposes_unified_service_snapshot(self):
        client = self._fresh_route_client()
        app_module.service.ingest(
            TrajectorySample(
                policy_name="safe-facmac",
                observation=Observation(validator_count=8, current_config_id=1, highest_known_config_id=1, throughput_tps=4000, latency_p95_ms=200),
                candidate=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                applied=DecisionActionStage(action=Action(pacemaker_timeout_ms=1300, mempool_max_batch_txs=1024, hydra_discovery_target=1), present=True),
                reward=0.5,
            )
        )
        app_module.service.train_online(batch_size=1)

        response = client.get("/adaptive")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["model_ready"])
        self.assertIn("model", payload)
        self.assertIn("organization", payload)
        self.assertIn("schema", payload)
        self.assertIn("replay", payload)
        self.assertIn("training", payload)
        self.assertEqual(payload["model"]["service_training"], payload["training"])
        self.assertEqual(payload["replay"]["replay_size"], 1)
        self.assertEqual(payload["training"]["active_model_source"], "online_training")

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_inspect_route_returns_service_side_decision_flow_only(self):
        client = self._fresh_route_client()
        before_replay = app_module.service.replay_summary()
        before_training = app_module.service.training_summary()

        response = client.post(
            "/inspect",
            json={
                "validator_count": 8,
                "current_config_id": 1,
                "highest_known_config_id": 2,
                "committee_size": 4,
                "pacemaker_timeout_ms": 1000,
                "mempool_max_batch_txs": 1024,
                "mempool_proposal_interval_ms": 80,
                "throughput_tps": 4500,
                "latency_p95_ms": 180,
                "local_validator": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["model_ready"])
        self.assertIn("organization_decision", payload)
        self.assertIn("candidate", payload)
        self.assertIn("governed", payload)
        self.assertIn("applied", payload)
        self.assertIn("divergence", payload)
        self.assertIn("explanations", payload)
        self.assertNotIn("replay", payload)
        self.assertNotIn("trace", payload)
        self.assertEqual(app_module.service.replay_summary(), before_replay)
        self.assertEqual(app_module.service.training_summary(), before_training)

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_infer_route_returns_structured_decision_envelope(self):
        client = self._fresh_route_client()
        response = client.post(
            "/infer",
            json={
                "validator_count": 8,
                "current_config_id": 1,
                "highest_known_config_id": 2,
                "committee_size": 4,
                "pacemaker_timeout_ms": 1000,
                "mempool_max_batch_txs": 1024,
                "mempool_proposal_interval_ms": 80,
                "throughput_tps": 4500,
                "latency_p95_ms": 180,
                "local_validator": False,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("action", payload)
        self.assertIn("candidate", payload)
        self.assertIn("governed", payload)
        self.assertIn("applied", payload)
        self.assertIn("organization_summary", payload)
        self.assertIn("role_override_attribution", payload)
        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
        self.assertEqual(
            payload["claim_boundary"],
            "co-runtime advisory decision envelope only; Go runtime remains final authority",
        )

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_inspect_route_accepts_full_go_runtime_observation_contract(self):
        client = self._fresh_route_client()
        before_replay = app_module.service.replay_summary()
        before_training = app_module.service.training_summary()
        response = client.post(
            "/inspect",
            json={
                "timestamp": "2026-04-01T00:00:00Z",
                "node_id": 7,
                "epoch": 3,
                "validator_count": 8,
                "current_config_id": 1,
                "highest_known_config_id": 2,
                "committee_size": 4,
                "pacemaker_timeout_ms": 1200,
                "mempool_max_batch_txs": 1024,
                "mempool_proposal_interval_ms": 80,
                "throughput_tps": 4000,
                "latency_p50_ms": 90,
                "latency_p95_ms": 140,
                "latency_p99_ms": 180,
                "recovery_p95_ms": 160,
                "backlog_pending": 12,
                "backlog_missing": 1,
                "reject_total": 2,
                "connected_peers": 6,
                "known_peers": 8,
                "pending_joins": 1,
                "pending_leaves": 0,
                "lset_size": 10,
                "can_participate": True,
                "local_validator": True,
                "global_confirmed_total": 99,
                "global_confirmed_nil": 3,
                "last_ordered_rank": 120,
                "last_ordered_height": 55,
                "last_ordered_lane_id": 1,
                "last_ordered_config_id": 4,
                "last_ordered_nil": False,
                "last_ordered_transition_count": 2,
                "last_reconfig_epoch": 4,
                "heterogeneity_score": 0.6,
                "churn_rate": 0.2,
                "adversary_score": 0.1,
                "network_jitter_ms": 25,
                "ai_load_score": 0.7,
                "agents": [{"instance_id": 1, "epoch": 3, "validator_count": 8, "committee_size": 4, "pacemaker_timeout_ms": 1200, "mempool_max_batch_txs": 1024, "mempool_proposal_interval_ms": 80}],
                "trust_snapshots": [{"node_id": 9, "sample_count": 10, "success_rate": 0.75, "failure_probability": 0.25}],
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("organization_decision", payload)
        self.assertIn("candidate", payload)
        self.assertEqual(app_module.service.replay_summary(), before_replay)
        self.assertEqual(app_module.service.training_summary(), before_training)

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_checkpoint_save_route_returns_model_not_ready_contract(self):
        client = self._fresh_route_client()
        response = client.post("/checkpoint/save", json={"path": "tmp-checkpoint.json"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["saved"])
        self.assertEqual(payload["reason"], "model_not_ready")
        self.assertIn("checkpoint", payload)
        self.assertIsNone(payload["checkpoint"]["last_saved_path"])
        self.assertIsNone(payload["checkpoint"]["last_loaded_path"])
        self.assertIsNone(payload["checkpoint"]["best_saved_path"])
        self.assertIsNone(payload["checkpoint"]["best_reward_mean"])
        self.assertIsNone(payload["checkpoint"].get("active_checkpoint_path"))

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_checkpoint_load_route_rejects_nested_path(self):
        client = self._fresh_route_client()
        response = client.post("/checkpoint/load", json={"path": "nested/checkpoint.json"})
        self.assertEqual(response.status_code, 422)

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_checkpoint_load_route_rejects_missing_path(self):
        client = self._fresh_route_client()
        response = client.post("/checkpoint/load", json={})
        self.assertEqual(response.status_code, 422)

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_checkpoint_load_route_rejects_parent_traversal_path(self):
        client = self._fresh_route_client()
        response = client.post("/checkpoint/load", json={"path": "../checkpoint.json"})
        self.assertEqual(response.status_code, 422)

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_checkpoint_load_route_rejects_absolute_path(self):
        client = self._fresh_route_client()
        response = client.post("/checkpoint/load", json={"path": "/tmp/checkpoint.json"})
        self.assertEqual(response.status_code, 422)

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_checkpoint_load_route_sanitizes_workspace_file_errors(self):
        client = self._fresh_route_client()
        response = client.post("/checkpoint/load", json={"path": "missing.json"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "workspace file operation failed")

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_train_offline_route_rejects_missing_path(self):
        client = self._fresh_route_client()
        response = client.post("/train/offline", json={})
        self.assertEqual(response.status_code, 422)

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_train_offline_route_rejects_parent_traversal_path(self):
        client = self._fresh_route_client()
        response = client.post("/train/offline", json={"trace_path": "../trace.jsonl"})
        self.assertEqual(response.status_code, 422)

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_train_offline_route_rejects_absolute_path(self):
        client = self._fresh_route_client()
        response = client.post("/train/offline", json={"trace_path": "/tmp/trace.jsonl"})
        self.assertEqual(response.status_code, 422)

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_checkpoint_save_route_rejects_nested_path(self):
        client = self._fresh_route_client()
        response = client.post("/checkpoint/save", json={"path": "nested/checkpoint.json"})
        self.assertEqual(response.status_code, 422)

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_checkpoint_save_route_rejects_missing_path(self):
        client = self._fresh_route_client()
        response = client.post("/checkpoint/save", json={})
        self.assertEqual(response.status_code, 422)

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_checkpoint_save_route_rejects_parent_traversal_path(self):
        client = self._fresh_route_client()
        response = client.post("/checkpoint/save", json={"path": "../checkpoint.json"})
        self.assertEqual(response.status_code, 422)

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_checkpoint_save_route_rejects_absolute_path(self):
        client = self._fresh_route_client()
        response = client.post("/checkpoint/save", json={"path": "/tmp/checkpoint.json"})
        self.assertEqual(response.status_code, 422)

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_checkpoint_save_route_rejects_extra_fields(self):
        client = self._fresh_route_client()
        before = app_module.service.training_summary()
        response = client.post("/checkpoint/save", json={"path": "tmp-checkpoint.json", "mode": "overwrite"})
        self.assertEqual(response.status_code, 422)
        detail = response.json()["detail"]
        self.assertTrue(any(item["loc"][-1] == "mode" for item in detail))
        self.assertEqual(app_module.service.training_summary(), before)

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_checkpoint_load_route_rejects_extra_fields(self):
        client = self._fresh_route_client()
        before = app_module.service.training_summary()
        response = client.post("/checkpoint/load", json={"path": "checkpoint.json", "mode": "eager"})
        self.assertEqual(response.status_code, 422)
        detail = response.json()["detail"]
        self.assertTrue(any(item["loc"][-1] == "mode" for item in detail))
        self.assertEqual(app_module.service.training_summary(), before)

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_train_online_route_rejects_invalid_batch_size(self):
        client = self._fresh_route_client()
        response = client.post("/train/online", json={"batch_size": 0})
        self.assertEqual(response.status_code, 422)

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_train_online_route_rejects_extra_fields(self):
        client = self._fresh_route_client()
        response = client.post("/train/online", json={"batch_size": 1, "seed": 7})
        self.assertEqual(response.status_code, 422)

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_trace_ingest_route_rejects_non_object_next_observation(self):
        client = self._fresh_route_client()
        before_replay = app_module.service.replay_summary()
        before_training = app_module.service.training_summary()
        response = client.post(
            "/trace/ingest",
            json={
                "policy_name": "safe-facmac",
                "observation": {"validator_count": 8},
                "next_observation": ["bad-shape"],
                "candidate": {"action": {"pacemaker_timeout_ms": 900}},
                "reward": 1.0,
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("next_observation must be an object", response.json()["detail"])
        self.assertEqual(app_module.service.replay_summary(), before_replay)
        self.assertEqual(app_module.service.training_summary(), before_training)

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_trace_ingest_route_rejects_unknown_top_level_field(self):
        client = self._fresh_route_client()
        before_replay = app_module.service.replay_summary()
        before_training = app_module.service.training_summary()
        response = client.post(
            "/trace/ingest",
            json={
                "policy_name": "safe-facmac",
                "observation": {"validator_count": 8},
                "candidate": {"action": {"pacemaker_timeout_ms": 900}},
                "reward": 1.0,
                "unexpected": 1,
            },
        )
        self.assertEqual(response.status_code, 422)
        detail = response.json()["detail"]
        self.assertTrue(any(item["loc"][-1] == "unexpected" for item in detail))
        self.assertEqual(app_module.service.replay_summary(), before_replay)
        self.assertEqual(app_module.service.training_summary(), before_training)

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_trace_ingest_route_rejects_non_object_observation(self):
        client = self._fresh_route_client()
        response = client.post(
            "/trace/ingest",
            json={
                "policy_name": "safe-facmac",
                "observation": [],
                "candidate": {"action": {"pacemaker_timeout_ms": 900}},
                "reward": 1.0,
            },
        )
        self.assertEqual(response.status_code, 422)

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_trace_ingest_route_rejects_non_object_action(self):
        client = self._fresh_route_client()
        before_replay = app_module.service.replay_summary()
        before_training = app_module.service.training_summary()
        response = client.post(
            "/trace/ingest",
            json={
                "policy_name": "safe-facmac",
                "observation": {"validator_count": 8},
                "candidate": {"action": []},
                "reward": 1.0,
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("candidate.action must be an object", response.json()["detail"])
        self.assertEqual(app_module.service.replay_summary(), before_replay)
        self.assertEqual(app_module.service.training_summary(), before_training)

        client = self._fresh_route_client()
        before_replay = app_module.service.replay_summary()
        before_training = app_module.service.training_summary()
        response = client.post(
            "/trace/ingest",
            json={
                "policy_name": "safe-facmac",
                "observation": {"validator_count": 8, "unexpected": 1},
                "candidate": {"action": {"pacemaker_timeout_ms": 900}},
                "reward": 1.0,
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("unknown field", response.json()["detail"])
        self.assertEqual(app_module.service.replay_summary(), before_replay)
        self.assertEqual(app_module.service.training_summary(), before_training)

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_trace_ingest_route_rejects_unknown_stage_field(self):
        client = self._fresh_route_client()
        before_replay = app_module.service.replay_summary()
        before_training = app_module.service.training_summary()
        response = client.post(
            "/trace/ingest",
            json={
                "policy_name": "safe-facmac",
                "observation": {"validator_count": 8},
                "candidate": {"action": {"pacemaker_timeout_ms": 900}, "unexpected": 1},
                "reward": 1.0,
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("unknown field", response.json()["detail"])
        self.assertEqual(app_module.service.replay_summary(), before_replay)
        self.assertEqual(app_module.service.training_summary(), before_training)

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_trace_ingest_route_rejects_unknown_action_field(self):
        client = self._fresh_route_client()
        before_replay = app_module.service.replay_summary()
        before_training = app_module.service.training_summary()
        response = client.post(
            "/trace/ingest",
            json={
                "policy_name": "safe-facmac",
                "observation": {"validator_count": 8},
                "candidate": {"action": {"pacemaker_timeout_ms": 900, "unexpected": 1}},
                "reward": 1.0,
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("unknown field", response.json()["detail"])
        self.assertEqual(app_module.service.replay_summary(), before_replay)
        self.assertEqual(app_module.service.training_summary(), before_training)

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_trace_ingest_route_rejects_schema_version_mismatch(self):
        client = self._fresh_route_client()
        before_replay = app_module.service.replay_summary()
        before_training = app_module.service.training_summary()
        response = client.post(
            "/trace/ingest",
            json={
                "policy_name": "safe-facmac",
                "schema_version": "evolvbft-adaptive-v2",
                "observation": {"validator_count": 8},
                "candidate": {"action": {"pacemaker_timeout_ms": 900}},
                "reward": 1.0,
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("unsupported trace schema_version", response.json()["detail"])
        self.assertEqual(app_module.service.replay_summary(), before_replay)
        self.assertEqual(app_module.service.training_summary(), before_training)

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_trace_ingest_route_rejects_non_object_trace(self):
        client = self._fresh_route_client()
        before_replay = app_module.service.replay_summary()
        before_training = app_module.service.training_summary()
        response = client.post(
            "/trace/ingest",
            json={
                "policy_name": "safe-facmac",
                "observation": {"validator_count": 4},
                "candidate": {},
                "governed": {},
                "masked": {},
                "applied": {},
                "trace": ["bad-shape"],
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("trace must be an object", response.json()["detail"])
        self.assertEqual(app_module.service.replay_summary(), before_replay)
        self.assertEqual(app_module.service.training_summary(), before_training)

    @unittest.skipIf(TestClient is None or app is None, "fastapi test client unavailable")
    def test_trace_ingest_route_rejects_non_object_governed_stage(self):
        client = self._fresh_route_client()
        before_replay = app_module.service.replay_summary()
        before_training = app_module.service.training_summary()
        response = client.post(
            "/trace/ingest",
            json={
                "policy_name": "safe-facmac",
                "observation": {"validator_count": 4},
                "candidate": {},
                "governed": ["bad-shape"],
                "masked": {},
                "applied": {},
                "trace": {},
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("governed must be an object", response.json()["detail"])
        self.assertEqual(app_module.service.replay_summary(), before_replay)
        self.assertEqual(app_module.service.training_summary(), before_training)


if __name__ == "__main__":
    unittest.main()
