"""SFAC Bridge: Integrates the Python SFAC controller into the unified marl/ service.

This module wraps experiments/sfac_facmac_aligned.py (SFACFACMACController) to provide
SFAC-specific trust management decisions through the consolidated marl/app.py FastAPI server.

Endpoints added to app.py:
    POST /sfac/decide   - SFAC-specific typed request/response (Go SFACPolicy)
    POST /sfac/feedback - Reward feedback for online SFAC training
    POST /sfac/reset    - Reset SFAC episode state
    GET  /sfac/health   - SFAC-specific health check
"""
from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

log = logging.getLogger("sfac-bridge")

# Add experiments/ to path so we can import sfac_ppo
_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

try:
    from sfac_facmac import FACMACConfig, FACMACController
    SFACConfig = FACMACConfig
    SFACController = FACMACController
except ImportError:
    try:
        from sfac_facmac_aligned import FACMACConfig as SFACConfig, SFACFACMACController as SFACController
        log.warning("sfac_facmac not available; falling back to sfac_facmac_aligned")
    except ImportError:
        SFACConfig = None
        SFACController = None
        log.warning("No SFAC controller available; bridge endpoints will return 503")


class SFACBridge:
    """Thread-safe wrapper around SFACController for HTTP serving."""

    def __init__(
        self,
        m_instances: int = 4,
        model_path: Optional[str] = None,
        train_mode: bool = False,
    ):
        self.available = SFACController is not None
        if not self.available:
            self.m = m_instances
            self.epoch = 0
            self.decisions = 0
            return

        cfg = SFACConfig(m_instances=m_instances)
        self.lock = threading.Lock()
        self.controller = SFACController(cfg)
        self.m = m_instances
        self.epoch = 0
        self.decisions = 0
        self.train_mode = train_mode

        if model_path:
            try:
                if hasattr(self.controller, 'load'):
                    self.controller.load(model_path)
                    log.info("Loaded SFAC model from %s", model_path)
                else:
                    log.info("Controller does not support load(); using fresh weights")
            except Exception as e:
                log.warning("Failed to load SFAC model: %s", e)

        if not train_mode:
            self.controller.set_eval()
        else:
            self.controller.set_train()

    def decide(self, request: dict) -> dict:
        """Process an SFACRequest and return an SFACResponse."""
        if not self.available:
            return {"actions": [], "error": "sfac_ppo not available"}

        with self.lock:
            instances = request.get("instances", [])
            m = len(instances) if instances else self.m
            obs = np.zeros((m, 5), dtype=np.float64)
            instance_sizes = np.ones(m) * 4
            feature_scores_by_instance = []

            for i, inst in enumerate(instances):
                if i >= m:
                    break
                features = inst.get("trust_features", [])
                scores = []
                if features:
                    # Pass real 5-dim features per instance (max across agents)
                    obs[i, 0] = max(f.get("timeout_rate", 0) for f in features)
                    obs[i, 1] = max(f.get("equivocation_rate", 0) for f in features)
                    obs[i, 2] = max(f.get("view_change_rate", 0) for f in features)
                    obs[i, 3] = np.mean([f.get("mean_latency", 0) for f in features])
                    obs[i, 4] = np.mean([f.get("std_latency", 0) for f in features])
                    scores = [
                        f.get("timeout_rate", 0) + f.get("equivocation_rate", 0)
                        for f in features
                    ]
                instance_sizes[i] = max(1, inst.get("validator_count", 4))
                feature_scores_by_instance.append(scores)

            # Use real 5-dim observation path (no synthetic reconstruction)
            decision = self.controller.decide_from_obs(
                obs, instance_sizes, epoch=self.epoch,
            )

            self.epoch += 1
            self.decisions += 1

            # Convert to SFACResponse format
            actions = []
            detected = decision.get("detected_instances", np.zeros(m))
            for k in range(m):
                scores = feature_scores_by_instance[k] if k < len(feature_scores_by_instance) else []
                reconfig = [0] * len(scores)
                if detected[k]:
                    if scores:
                        target_idx = int(np.argmax(scores))
                        reconfig[target_idx] = -1
                    else:
                        reconfig = [-1]
                action = {
                    "instance_id": k,
                    "reconfig": reconfig if reconfig else [0],
                    "rotate": bool(detected[k]),
                    "params": [],
                }
                actions.append(action)

            return {
                "actions": actions,
                "value": float(decision.get("value", 0.0)),
            }

    def feedback(self, reward_data: dict) -> dict:
        """Process reward feedback for online training."""
        if not self.available:
            return {"status": "unavailable"}

        with self.lock:
            if not self.train_mode:
                return {"status": "ignored", "reason": "eval-mode"}

            per_inst_rewards = np.array(
                reward_data.get("per_instance_rewards", [0.0] * self.m),
                dtype=np.float64,
            )
            done = reward_data.get("done", False)

            self.controller.store_transition(per_inst_rewards, done)
            result = {"status": "stored"}

            if self.controller.buffer.n_steps >= 64:
                losses = self.controller.train_step()
                result["train_losses"] = {
                    k: round(v, 6) for k, v in losses.items()
                }

            return result

    def reset(self) -> dict:
        """Reset episode state."""
        if not self.available:
            return {"status": "unavailable"}

        with self.lock:
            self.controller.reset()
            self.epoch = 0
            self.decisions = 0
            return {"status": "reset"}

    def health(self) -> dict:
        """Return SFAC bridge health status."""
        return {
            "status": "ok" if self.available else "unavailable",
            "sfac_available": self.available,
            "decisions": self.decisions,
            "epoch": self.epoch,
            "train_mode": getattr(self, "train_mode", False),
            "m_instances": self.m,
        }
