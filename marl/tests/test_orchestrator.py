import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from marl.curriculum import CurriculumPhase, CurriculumSchedule
from marl.orchestrator import ExperimentOrchestrator


class OrchestratorTests(unittest.TestCase):
    def test_orchestrator_sends_relative_checkpoint_path_when_checkpoint_dir_is_absolute(self):
        state = {"checkpoint_paths": []}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                body = self.rfile.read(length)
                payload = json.loads(body.decode("utf-8") or "{}")

                if self.path == "/adaptive/context":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(json.dumps(payload).encode("utf-8"))
                    return

                if self.path == "/trace/ingest":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"replay_size": 1}')
                    return

                if self.path == "/train/online":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"samples": 1, "model_ready": true}')
                    return

                if self.path == "/checkpoint/save":
                    state["checkpoint_paths"].append(payload.get("path"))
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"saved": true}')
                    return

                self.send_response(404)
                self.end_headers()

            def do_GET(self):
                if self.path == "/adaptive":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps(
                            {
                                "enabled": True,
                                "schema_version": "octopus-adaptive-v1",
                                "last_decision": {
                                    "policy_name": "facmac-http",
                                    "observation": {"validator_count": 8},
                                    "candidate": {"action": {"pacemaker_timeout_ms": 1200}, "present": True},
                                    "governed": {"action": {"pacemaker_timeout_ms": 1200}, "present": True},
                                    "applied": {"action": {"pacemaker_timeout_ms": 1200}, "present": True},
                                    "reward": 1.0,
                                },
                            }
                        ).encode("utf-8")
                    )
                    return
                self.send_response(404)
                self.end_headers()

            def log_message(self, format, *args):
                return

        octopus = HTTPServer(("127.0.0.1", 0), Handler)
        service = HTTPServer(("127.0.0.1", 0), Handler)
        octopus_thread = threading.Thread(target=octopus.serve_forever, daemon=True)
        service_thread = threading.Thread(target=service.serve_forever, daemon=True)
        octopus_thread.start()
        service_thread.start()

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            try:
                orchestrator = ExperimentOrchestrator(
                    octopus_base_url=f"http://127.0.0.1:{octopus.server_port}",
                    marl_service_url=f"http://127.0.0.1:{service.server_port}",
                    curriculum=CurriculumSchedule(
                        phases=[
                            CurriculumPhase(name="warmup", steps=1, heterogeneity=0.2, churn=0.1, adversary=0.0, jitter_ms=10, ai_load=0.3),
                        ]
                    ),
                    checkpoint_dir=checkpoint_dir,
                )
                orchestrator.run(steps=1, train_every=1, checkpoint_every=1)
            finally:
                octopus.shutdown()
                service.shutdown()
                octopus_thread.join(timeout=2)
                service_thread.join(timeout=2)
                octopus.server_close()
                service.server_close()

        self.assertEqual(state["checkpoint_paths"], ["checkpoint-step-1.json"])

    def test_orchestrator_forwards_runtime_schema_version_and_stage_payload(self):
        state = {"ingest_payloads": []}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                body = self.rfile.read(length)
                payload = json.loads(body.decode("utf-8") or "{}")

                if self.path == "/adaptive/context":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(json.dumps(payload).encode("utf-8"))
                    return

                if self.path == "/trace/ingest":
                    state["ingest_payloads"].append(payload)
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"replay_size": 1}')
                    return

                if self.path == "/train/online":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"samples": 1, "model_ready": true}')
                    return

                if self.path == "/checkpoint/save":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"saved": true}')
                    return

                self.send_response(404)
                self.end_headers()

            def do_GET(self):
                if self.path == "/adaptive":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps(
                            {
                                "enabled": True,
                                "schema_version": "octopus-adaptive-v1",
                                "last_decision": {
                                    "policy_name": "facmac-http",
                                    "observation": {
                                        "validator_count": 8,
                                        "trust_snapshots": [
                                            {
                                                "node_id": 3,
                                                "sample_count": 12,
                                                "success_rate": 0.75,
                                                "failure_probability": 0.25,
                                            }
                                        ],
                                    },
                                    "candidate": {
                                        "action": {
                                            "pacemaker_timeout_ms": 1300,
                                            "agent_actions": [
                                                {"instance_id": 0, "pacemaker_timeout_ms": 1250}
                                            ],
                                        },
                                        "present": True,
                                        "mutated": False,
                                    },
                                    "governed": {
                                        "action": {"pacemaker_timeout_ms": 1200},
                                        "present": True,
                                        "mutated": True,
                                    },
                                    "masked": {
                                        "action": {"pacemaker_timeout_ms": 1200},
                                        "present": True,
                                        "mutated": False,
                                    },
                                    "applied": {
                                        "action": {"pacemaker_timeout_ms": 1200},
                                        "present": True,
                                        "mutated": False,
                                    },
                                    "reward": 1.0,
                                    "trace": {"enabled": True, "dropped_samples": 0},
                                },
                            }
                        ).encode("utf-8")
                    )
                    return
                self.send_response(404)
                self.end_headers()

            def log_message(self, format, *args):
                return

        octopus = HTTPServer(("127.0.0.1", 0), Handler)
        service = HTTPServer(("127.0.0.1", 0), Handler)
        octopus_thread = threading.Thread(target=octopus.serve_forever, daemon=True)
        service_thread = threading.Thread(target=service.serve_forever, daemon=True)
        octopus_thread.start()
        service_thread.start()

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            try:
                orchestrator = ExperimentOrchestrator(
                    octopus_base_url=f"http://127.0.0.1:{octopus.server_port}",
                    marl_service_url=f"http://127.0.0.1:{service.server_port}",
                    curriculum=CurriculumSchedule(
                        phases=[
                            CurriculumPhase(name="warmup", steps=1, heterogeneity=0.2, churn=0.1, adversary=0.0, jitter_ms=10, ai_load=0.3),
                        ]
                    ),
                    checkpoint_dir=checkpoint_dir,
                )
                orchestrator.run(steps=1, train_every=1, checkpoint_every=0)
            finally:
                octopus.shutdown()
                service.shutdown()
                octopus_thread.join(timeout=2)
                service_thread.join(timeout=2)
                octopus.server_close()
                service.server_close()

        self.assertEqual(len(state["ingest_payloads"]), 1)
        payload = state["ingest_payloads"][0]
        self.assertEqual(payload["schema_version"], "octopus-adaptive-v1")
        self.assertIn("candidate", payload)
        self.assertIn("governed", payload)
        self.assertIn("masked", payload)
        self.assertIn("applied", payload)
        self.assertEqual(payload["candidate"]["action"]["agent_actions"][0]["instance_id"], 0)
        self.assertEqual(payload["observation"]["trust_snapshots"][0]["node_id"], 3)
        self.assertEqual(payload["trace"]["enabled"], True)

    def test_orchestrator_runs_training_loop_and_saves_checkpoint(self):
        state = {"contexts": 0, "ingests": 0, "online_trains": 0, "checkpoints": 0, "ingest_payloads": [], "checkpoint_payloads": []}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                body = self.rfile.read(length)
                payload = json.loads(body.decode("utf-8") or "{}")

                if self.path == "/adaptive/context":
                    state["contexts"] += 1
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(json.dumps(payload).encode("utf-8"))
                    return

                if self.path == "/trace/ingest":
                    state["ingests"] += 1
                    state["ingest_payloads"].append(payload)
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"replay_size": 1}')
                    return

                if self.path == "/train/online":
                    state["online_trains"] += 1
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"samples": 1, "model_ready": true}')
                    return

                if self.path == "/checkpoint/save":
                    state["checkpoints"] += 1
                    state["checkpoint_payloads"].append(payload)
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"saved": true}')
                    return

                self.send_response(404)
                self.end_headers()

            def do_GET(self):
                if self.path == "/adaptive":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps(
                            {
                                "enabled": True,
                                "last_decision": {
                                    "policy_name": "facmac-http",
                                    "observation": {
                                        "validator_count": 8,
                                        "committee_size": 4,
                                        "pacemaker_timeout_ms": 1200,
                                        "mempool_max_batch_txs": 1024,
                                        "mempool_proposal_interval_ms": 80,
                                        "throughput_tps": 6000,
                                        "latency_p95_ms": 120,
                                        "backlog_pending": 12,
                                        "reject_total": 0,
                                        "heterogeneity_score": 0.4,
                                        "churn_rate": 0.1,
                                        "adversary_score": 0.2,
                                        "network_jitter_ms": 10,
                                        "ai_load_score": 0.6,
                                    },
                                    "candidate": {
                                        "action": {
                                            "committee_size": 5,
                                            "pacemaker_timeout_ms": 1300,
                                            "mempool_max_batch_txs": 1200,
                                            "mempool_proposal_interval_ms": 90,
                                            "submit_join": True,
                                            "hydra_discovery_target": 2,
                                        },
                                        "present": True,
                                        "mutated": False,
                                    },
                                    "governed": {
                                        "action": {
                                            "committee_size": 4,
                                            "pacemaker_timeout_ms": 1300,
                                            "mempool_max_batch_txs": 1200,
                                            "mempool_proposal_interval_ms": 90,
                                            "submit_join": True,
                                            "hydra_discovery_target": 2,
                                        },
                                        "present": True,
                                        "mutated": True,
                                    },
                                    "masked": {
                                        "action": {
                                            "committee_size": 4,
                                            "pacemaker_timeout_ms": 1200,
                                            "mempool_max_batch_txs": 1024,
                                            "mempool_proposal_interval_ms": 80,
                                            "submit_join": True,
                                            "hydra_discovery_target": 2,
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
                                            "hydra_discovery_target": 2,
                                        },
                                        "present": True,
                                        "mutated": False,
                                    },
                                    "next_observation": {
                                        "validator_count": 8,
                                        "committee_size": 4,
                                        "pacemaker_timeout_ms": 1200,
                                        "mempool_max_batch_txs": 1024,
                                        "mempool_proposal_interval_ms": 80,
                                        "throughput_tps": 6500,
                                        "latency_p95_ms": 100,
                                    },
                                    "reward": 1.0,
                                    "team_reward": 1.25,
                                    "role_rewards": {"lane_tuner": 0.8, "recovery_tuner": -0.1},
                                    "governance_delta": True,
                                    "guardrail_delta": False,
                                    "done": False,
                                },
                            }
                        ).encode("utf-8")
                    )
                    return
                self.send_response(404)
                self.end_headers()

            def log_message(self, format, *args):
                return

        octopus = HTTPServer(("127.0.0.1", 0), Handler)
        service = HTTPServer(("127.0.0.1", 0), Handler)
        octopus_thread = threading.Thread(target=octopus.serve_forever, daemon=True)
        service_thread = threading.Thread(target=service.serve_forever, daemon=True)
        octopus_thread.start()
        service_thread.start()

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            try:
                orchestrator = ExperimentOrchestrator(
                    octopus_base_url=f"http://127.0.0.1:{octopus.server_port}",
                    marl_service_url=f"http://127.0.0.1:{service.server_port}",
                    curriculum=CurriculumSchedule(
                        phases=[
                            CurriculumPhase(name="warmup", steps=2, heterogeneity=0.2, churn=0.1, adversary=0.0, jitter_ms=10, ai_load=0.3),
                            CurriculumPhase(name="stress", steps=2, heterogeneity=0.7, churn=0.4, adversary=0.6, jitter_ms=50, ai_load=0.8),
                        ]
                    ),
                    checkpoint_dir=checkpoint_dir,
                )
                summary = orchestrator.run(steps=4, train_every=2, checkpoint_every=2)
            finally:
                octopus.shutdown()
                service.shutdown()
                octopus_thread.join(timeout=2)
                service_thread.join(timeout=2)
                octopus.server_close()
                service.server_close()

        self.assertEqual(summary["steps"], 4)
        self.assertEqual(state["contexts"], 4)
        self.assertEqual(state["ingests"], 4)
        self.assertEqual(state["online_trains"], 2)
        self.assertEqual(state["checkpoints"], 2)
        self.assertEqual(len(state["checkpoint_payloads"]), 2)
        self.assertEqual(state["checkpoint_payloads"][0]["path"], "checkpoint-step-2.json")
        self.assertEqual(state["checkpoint_payloads"][1]["path"], "checkpoint-step-4.json")
        self.assertEqual(len(state["ingest_payloads"]), 4)
        payload = state["ingest_payloads"][0]
        self.assertEqual(payload["policy_name"], "facmac-http")
        self.assertIn("observation", payload)
        self.assertIn("candidate", payload)
        self.assertIn("governed", payload)
        self.assertIn("masked", payload)
        self.assertIn("applied", payload)
        self.assertIn("next_observation", payload)
        self.assertEqual(payload["candidate"]["present"], True)
        self.assertEqual(payload["governed"]["present"], True)
        self.assertEqual(payload["applied"]["action"]["submit_join"], True)
        self.assertEqual(payload["candidate"]["action"]["committee_size"], 5)
        self.assertEqual(payload["governed"]["action"]["committee_size"], 4)
        self.assertEqual(payload["reward"], 1.0)
        self.assertEqual(payload["team_reward"], 1.25)
        self.assertEqual(payload["role_rewards"], {"lane_tuner": 0.8, "recovery_tuner": -0.1})
        self.assertEqual(payload["governance_delta"], True)
        self.assertEqual(payload["guardrail_delta"], True)
        self.assertEqual(payload["done"], False)

    def test_orchestrator_falls_back_when_rich_decision_fields_are_missing(self):
        state = {"ingest_payloads": []}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                body = self.rfile.read(length)
                payload = json.loads(body.decode("utf-8") or "{}")
                if self.path == "/adaptive/context":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(json.dumps(payload).encode("utf-8"))
                    return
                if self.path == "/trace/ingest":
                    state["ingest_payloads"].append(payload)
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"replay_size": 1}')
                    return
                if self.path == "/train/online":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"samples": 1, "model_ready": true}')
                    return
                if self.path == "/checkpoint/save":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"saved": true}')
                    return
                self.send_response(404)
                self.end_headers()

            def do_GET(self):
                if self.path == "/adaptive":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps(
                            {
                                "enabled": True,
                                "last_decision": {
                                    "policy_name": "facmac-http",
                                    "reward": 0.5,
                                },
                            }
                        ).encode("utf-8")
                    )
                    return
                self.send_response(404)
                self.end_headers()

            def log_message(self, format, *args):
                return

        octopus = HTTPServer(("127.0.0.1", 0), Handler)
        service = HTTPServer(("127.0.0.1", 0), Handler)
        octopus_thread = threading.Thread(target=octopus.serve_forever, daemon=True)
        service_thread = threading.Thread(target=service.serve_forever, daemon=True)
        octopus_thread.start()
        service_thread.start()

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            try:
                orchestrator = ExperimentOrchestrator(
                    octopus_base_url=f"http://127.0.0.1:{octopus.server_port}",
                    marl_service_url=f"http://127.0.0.1:{service.server_port}",
                    curriculum=CurriculumSchedule(
                        phases=[
                            CurriculumPhase(name="warmup", steps=1, heterogeneity=0.2, churn=0.1, adversary=0.0, jitter_ms=10, ai_load=0.3),
                        ]
                    ),
                    checkpoint_dir=checkpoint_dir,
                )
                orchestrator.run(steps=1, train_every=1, checkpoint_every=1)
            finally:
                octopus.shutdown()
                service.shutdown()
                octopus_thread.join(timeout=2)
                service_thread.join(timeout=2)
                octopus.server_close()
                service.server_close()

        payload = state["ingest_payloads"][0]
        self.assertEqual(payload["policy_name"], "facmac-http")
        self.assertIn("observation", payload)
        self.assertEqual(payload["observation"]["heterogeneity_score"], 0.2)
        self.assertEqual(payload["candidate"], {})
        self.assertEqual(payload["governed"], {})
        self.assertEqual(payload["masked"], {})
        self.assertEqual(payload["applied"], {})
        self.assertIsNone(payload["next_observation"])
        self.assertEqual(payload["reward"], 0.5)
        self.assertIsNone(payload["team_reward"])
        self.assertEqual(payload["role_rewards"], {})
        self.assertEqual(payload["governance_delta"], False)
        self.assertEqual(payload["guardrail_delta"], False)
        self.assertEqual(payload["done"], False)

    def test_orchestrator_keeps_presence_flags_false_when_only_applied_stage_exists(self):
        state = {"ingest_payloads": []}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                body = self.rfile.read(length)
                payload = json.loads(body.decode("utf-8") or "{}")
                if self.path == "/adaptive/context":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(json.dumps(payload).encode("utf-8"))
                    return
                if self.path == "/trace/ingest":
                    state["ingest_payloads"].append(payload)
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"replay_size": 1}')
                    return
                if self.path == "/train/online":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"samples": 1, "model_ready": true}')
                    return
                if self.path == "/checkpoint/save":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"saved": true}')
                    return
                self.send_response(404)
                self.end_headers()

            def do_GET(self):
                if self.path == "/adaptive":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps(
                            {
                                "enabled": True,
                                "last_decision": {
                                    "policy_name": "facmac-http",
                                    "observation": {"validator_count": 8},
                                    "applied": {
                                        "action": {"pacemaker_timeout_ms": 900},
                                        "present": True,
                                    },
                                    "reward": 0.5,
                                },
                            }
                        ).encode("utf-8")
                    )
                    return
                self.send_response(404)
                self.end_headers()

            def log_message(self, format, *args):
                return

        octopus = HTTPServer(("127.0.0.1", 0), Handler)
        service = HTTPServer(("127.0.0.1", 0), Handler)
        octopus_thread = threading.Thread(target=octopus.serve_forever, daemon=True)
        service_thread = threading.Thread(target=service.serve_forever, daemon=True)
        octopus_thread.start()
        service_thread.start()

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            try:
                orchestrator = ExperimentOrchestrator(
                    octopus_base_url=f"http://127.0.0.1:{octopus.server_port}",
                    marl_service_url=f"http://127.0.0.1:{service.server_port}",
                    curriculum=CurriculumSchedule(
                        phases=[
                            CurriculumPhase(name="warmup", steps=1, heterogeneity=0.2, churn=0.1, adversary=0.0, jitter_ms=10, ai_load=0.3),
                        ]
                    ),
                    checkpoint_dir=checkpoint_dir,
                )
                orchestrator.run(steps=1, train_every=1, checkpoint_every=1)
            finally:
                octopus.shutdown()
                service.shutdown()
                octopus_thread.join(timeout=2)
                service_thread.join(timeout=2)
                octopus.server_close()
                service.server_close()

        payload = state["ingest_payloads"][0]
        self.assertEqual(payload["applied"], {"action": {"pacemaker_timeout_ms": 900}, "present": True})
        self.assertEqual(payload["candidate"], {})
        self.assertEqual(payload["governed"], {})


if __name__ == "__main__":
    unittest.main()
