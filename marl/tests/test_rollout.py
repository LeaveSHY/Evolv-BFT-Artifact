import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from marl.rollout import ScenarioRolloutRunner
from marl.scenario import AIoTScenarioDriver


class RolloutTests(unittest.TestCase):
    def test_runner_pushes_context_and_collects_snapshots(self):
        state = {"contexts": [], "adaptive_reads": 0}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path != "/adaptive/context":
                    self.send_response(404)
                    self.end_headers()
                    return
                length = int(self.headers["Content-Length"])
                body = self.rfile.read(length)
                state["contexts"].append(json.loads(body.decode("utf-8")))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path != "/adaptive":
                    self.send_response(404)
                    self.end_headers()
                    return
                state["adaptive_reads"] += 1
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"enabled": True, "last_decision": {}}).encode("utf-8"))

            def log_message(self, format, *args):
                return

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            runner = ScenarioRolloutRunner(
                evolvbft_base_url=f"http://127.0.0.1:{server.server_port}",
                driver=AIoTScenarioDriver(seed=1),
            )
            snapshots = runner.run_steps(steps=3)
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()

        self.assertEqual(len(snapshots), 3)
        self.assertEqual(len(state["contexts"]), 3)
        self.assertEqual(state["adaptive_reads"], 3)


if __name__ == "__main__":
    unittest.main()
