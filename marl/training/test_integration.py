"""Integration test: start FastAPI server, call torch endpoints."""
import sys
sys.path.insert(0, ".")

import threading
import time
import json

import uvicorn
import requests

from marl.app import app

PORT = 18901

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")

# Start server in background thread
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(2)  # wait for startup

BASE = f"http://127.0.0.1:{PORT}"

def test_health():
    r = requests.get(f"{BASE}/torch/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["backend"] == "pytorch"
    print(f"[PASS] /torch/health: device={data['device']}, m={data['m_instances']}")
    return data

def test_decide():
    payload = {
        "epoch": 1,
        "num_instances": 4,
        "instances": [
            {"validator_count": 4, "trust_features": [
                {"timeout_rate": 0.1, "equivocation_rate": 0.0, "view_change_rate": 0.05, "mean_latency": 50, "std_latency": 10}
            ]},
            {"validator_count": 4, "trust_features": [
                {"timeout_rate": 0.8, "equivocation_rate": 0.5, "view_change_rate": 0.3, "mean_latency": 200, "std_latency": 80}
            ]},
            {"validator_count": 3, "trust_features": []},
            {"validator_count": 5, "trust_features": [
                {"timeout_rate": 0.0, "equivocation_rate": 0.0, "view_change_rate": 0.0, "mean_latency": 30, "std_latency": 5}
            ]},
        ],
        "global_state": [float(i) for i in range(28)],
    }
    r = requests.post(f"{BASE}/torch/decide", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "actions" in data
    assert len(data["actions"]) == 4
    print(f"[PASS] /torch/decide: {len(data['actions'])} actions returned, train_steps={data['train_steps']}")
    return data

def test_feedback():
    payload = {
        "per_instance_rewards": [1.0, -0.5, 0.3, 0.8],
        "done": False,
    }
    r = requests.post(f"{BASE}/torch/feedback", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "stored"
    print(f"[PASS] /torch/feedback: buffer_size={data['buffer_size']}")
    return data

def test_metrics():
    r = requests.get(f"{BASE}/torch/metrics")
    assert r.status_code == 200
    data = r.json()
    print(f"[PASS] /torch/metrics: train_steps={data['train_steps']}, buffer={data['buffer_size']}")
    return data

def test_reset():
    r = requests.post(f"{BASE}/torch/reset")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "reset"
    print(f"[PASS] /torch/reset")

def test_training_loop():
    """Simulate 200 decide+feedback cycles to trigger training."""
    for i in range(250):
        payload = {
            "epoch": i,
            "num_instances": 4,
            "instances": [
                {"validator_count": 4, "trust_features": [
                    {"timeout_rate": 0.1 * (i % 5), "equivocation_rate": 0.05, "view_change_rate": 0.02, "mean_latency": 50, "std_latency": 10}
                ]} for _ in range(4)
            ],
            "global_state": [float(i + j) for j in range(28)],
        }
        r = requests.post(f"{BASE}/torch/decide", json=payload)
        assert r.status_code == 200

        fb = {
            "per_instance_rewards": [0.5 + 0.1 * (i % 3), -0.2, 0.3, 0.1],
            "done": (i % 50 == 49),
        }
        r = requests.post(f"{BASE}/torch/feedback", json=fb)
        assert r.status_code == 200

    # Check metrics after training
    r = requests.get(f"{BASE}/torch/metrics")
    data = r.json()
    assert data["train_steps"] > 0, "Training should have occurred"
    print(f"[PASS] Training loop: {data['train_steps']} train steps, "
          f"critic_loss={data['recent_critic_loss']:.4f}, "
          f"buffer={data['buffer_size']}")

def test_export():
    r = requests.get(f"{BASE}/torch/export")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "exported"
    assert data["n_params"] > 0
    print(f"[PASS] /torch/export: {data['n_params']:,} params exported")

print("=" * 60)
print("Integration Test: PyTorch SFAC Endpoints")
print("=" * 60)

test_health()
test_decide()
test_feedback()
test_metrics()
test_training_loop()
test_export()
test_reset()

print("\n=== ALL INTEGRATION TESTS PASSED ===")
