import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from marl.scenario import AIOT_NAMED_SCENARIOS, AIOT_SCENARIO_SEQUENCE, AIoTScenarioDriver, ScenarioPublisher


class ScenarioTests(unittest.TestCase):
    def test_driver_generates_bounded_context(self):
        driver = AIoTScenarioDriver(seed=7)
        ctx = driver.next_context(step=3)
        self.assertGreaterEqual(ctx["heterogeneity_score"], 0.0)
        self.assertLessEqual(ctx["heterogeneity_score"], 1.0)
        self.assertGreaterEqual(ctx["adversary_score"], 0.0)
        self.assertLessEqual(ctx["adversary_score"], 1.0)

    def test_named_scenarios_cover_adversarial_aiot_regimes(self):
        required = {
            "heterogeneous_steady_state",
            "churn_reconfiguration_pressure",
            "adversarial_safety_stress",
            "jitter_recovery_stress",
            "ai_load_throughput_stress",
        }
        self.assertTrue(required.issubset(set(AIOT_NAMED_SCENARIOS)))
        ordered = [
            "heterogeneous_steady_state",
            "churn_reconfiguration_pressure",
            "adversarial_safety_stress",
            "jitter_recovery_stress",
            "ai_load_throughput_stress",
        ]
        indexes = [AIOT_SCENARIO_SEQUENCE.index(name) for name in ordered]
        self.assertEqual(indexes, sorted(indexes))
        self.assertGreater(AIOT_NAMED_SCENARIOS["churn_reconfiguration_pressure"]["churn_rate"], 0.4)
        self.assertGreaterEqual(AIOT_NAMED_SCENARIOS["adversarial_safety_stress"]["adversary_score"], 0.7)
        self.assertGreaterEqual(AIOT_NAMED_SCENARIOS["jitter_recovery_stress"]["network_jitter_ms"], 50)
        self.assertGreaterEqual(AIOT_NAMED_SCENARIOS["ai_load_throughput_stress"]["ai_load_score"], 0.85)

    def test_driver_can_emit_named_scenario(self):
        driver = AIoTScenarioDriver(seed=7)
        ctx = driver.named_context("adversarial_safety_stress")
        self.assertEqual(ctx["scenario_name"], "adversarial_safety_stress")
        self.assertEqual(ctx["adversary_score"], AIOT_NAMED_SCENARIOS["adversarial_safety_stress"]["adversary_score"])
        self.assertEqual(ctx["network_jitter_ms"], AIOT_NAMED_SCENARIOS["adversarial_safety_stress"]["network_jitter_ms"])

    def test_named_scenario_acceptance_matrix_is_canonical(self):
        steady = AIOT_NAMED_SCENARIOS["heterogeneous_steady_state"]
        churn = AIOT_NAMED_SCENARIOS["churn_reconfiguration_pressure"]
        adversarial = AIOT_NAMED_SCENARIOS["adversarial_safety_stress"]
        jitter = AIOT_NAMED_SCENARIOS["jitter_recovery_stress"]
        ai_load = AIOT_NAMED_SCENARIOS["ai_load_throughput_stress"]

        self.assertGreater(adversarial["adversary_score"], churn["adversary_score"])
        self.assertGreater(adversarial["adversary_score"], jitter["adversary_score"])
        self.assertGreater(adversarial["adversary_score"], ai_load["adversary_score"])
        self.assertGreater(churn["churn_rate"], steady["churn_rate"])
        self.assertGreaterEqual(churn["churn_rate"], adversarial["churn_rate"])
        self.assertGreater(jitter["network_jitter_ms"], steady["network_jitter_ms"])
        self.assertGreater(adversarial["network_jitter_ms"], steady["network_jitter_ms"])
        self.assertGreater(ai_load["ai_load_score"], steady["ai_load_score"])
        self.assertGreater(ai_load["ai_load_score"], churn["ai_load_score"])
        self.assertGreater(ai_load["ai_load_score"], jitter["ai_load_score"])

    def test_publisher_posts_context(self):
        received = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                body = self.rfile.read(length)
                received["payload"] = json.loads(body.decode("utf-8"))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, format, *args):
                return

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            publisher = ScenarioPublisher(f"http://127.0.0.1:{server.server_port}/adaptive/context")
            publisher.publish({"heterogeneity_score": 0.5, "churn_rate": 0.2})
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()
        self.assertIn("payload", received)
        self.assertEqual(received["payload"]["churn_rate"], 0.2)


if __name__ == "__main__":
    unittest.main()
