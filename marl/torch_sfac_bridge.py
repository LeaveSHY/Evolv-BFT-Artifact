"""PyTorch SFAC Bridge: GPU-accelerated FACMAC integration for FastAPI.

Replaces the numpy-based SFACBridge with PyTorch GPU inference and training.
Maintains the same API contract (decide/feedback/reset) so Go side is unchanged.

Architecture:
  - Online GPU training via SafeRoleFACMACTrainer (Lagrangian safety + role-decomposed)
  - Decentralized execution: each agent uses local actor π_i(o_i)
  - Background training: every N decide calls, run a training step
  - Safety constraint: λ auto-tunes to bound violation cost ≤ d
  - Export: convert PyTorch weights to numpy for lightweight serving fallback
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from marl.networks import ActorNetwork, AgentActorNetwork, CriticNetwork, QMIXMixer
from marl.networks.safety_guardian import (
    SafetyGuardianConfig,
    SafetyGuardianPolicy,
    SafetyModulation,
    apply_safety_modulation,
    pre_argmax_safety_mask,
)
from marl.training import (
    FACMACTrainer,
    SafeRoleFACMACTrainer,
    SafeTrainingConfig,
    TrainingConfig,
    Transition,
)

log = logging.getLogger("torch-sfac-bridge")


class TorchSFACBridge:
    """Thread-safe PyTorch FACMAC bridge for HTTP serving.

    Drop-in replacement for SFACBridge with GPU acceleration.
    Same request/response format; Go client code unchanged.
    """

    def __init__(
        self,
        m_instances: int = 10,
        model_path: Optional[str] = None,
        train_mode: bool = True,
        device: Optional[str] = None,
        train_every: int = 4,
        safe_training: bool = True,
        cost_limit: float = 0.1,
    ):
        self.m = m_instances
        self.train_mode = train_mode
        self.train_every = train_every
        self.epoch = 0
        self.decisions = 0
        self.lock = threading.Lock()
        self.safe_training = safe_training

        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if safe_training:
            self.safe_config = SafeTrainingConfig(
                state_dim=28,
                agent_obs_dim=7,
                agent_action_dim=4,
                n_agents=m_instances,
                mixing_hidden=32,
                trunk_dims=(256, 128),
                head_hidden=64,
                role_weights=(1.0, 1.0, 1.0, 1.5),
                actor_lr=3e-4,
                critic_lr=1e-3,
                mixer_lr=1e-3,
                gamma=0.95,
                tau=0.005,
                batch_size=64,
                buffer_capacity=100_000,
                warmup_steps=200,
                total_steps=500_000,
                device=device,
                cost_limit=cost_limit,
                lambda_init=0.1,
                lambda_lr=5e-3,
                lambda_max=10.0,
                cost_window_size=1000,
            )
            self.trainer = SafeRoleFACMACTrainer(self.safe_config)
            self.config = self.safe_config  # alias for compatibility
        else:
            self.config = TrainingConfig(
                state_dim=28,
                agent_obs_dim=7,
                agent_action_dim=4,
                action_dim=7,
                n_agents=m_instances,
                mixing_hidden=32,
                actor_lr=3e-4,
                critic_lr=1e-3,
                gamma=0.95,
                tau=0.005,
                batch_size=64,
                buffer_capacity=100_000,
                warmup_steps=200,
                total_steps=500_000,
                device=device,
            )
            self.trainer = FACMACTrainer(self.config)

        self.available = True

        # Safety Guardian active policy (proactive modulation)
        if safe_training:
            self.safety_guardian = SafetyGuardianPolicy(SafetyGuardianConfig(
                state_dim=28,
                hidden_dims=(128, 64),
                lr=1e-3,
                tau=0.01,
                gamma=0.95,
                conservative_penalty=0.1,
                device=device,
            ))
        else:
            self.safety_guardian = None

        # State tracking for transitions
        self._last_state: Optional[np.ndarray] = None
        self._last_agent_obs: Optional[np.ndarray] = None
        self._last_actions: Optional[np.ndarray] = None
        self._last_modulation: Optional[SafetyModulation] = None

        if model_path and Path(model_path).exists():
            try:
                self.trainer.load(model_path)
                log.info("Loaded PyTorch SFAC model from %s", model_path)
            except Exception as e:
                log.warning("Failed to load model: %s", e)

        log.info(
            "TorchSFACBridge initialized: device=%s, m=%d, train=%s, safe=%s",
            device, m_instances, train_mode, safe_training,
        )

    def decide(self, request: dict) -> dict:
        """Process observation and return per-instance actions.

        Matches SFACBridge.decide() contract for Go compatibility.
        """
        with self.lock:
            instances = request.get("instances", [])
            m = len(instances) if instances else self.m
            global_state = request.get("global_state", [])

            # Build state vector (28-dim)
            state = np.zeros(28, dtype=np.float32)
            if global_state and len(global_state) >= 28:
                state[:28] = np.array(global_state[:28], dtype=np.float32)
            else:
                # Reconstruct from instance data
                state[0] = float(request.get("epoch", self.epoch))
                state[1] = float(m)
                for i, inst in enumerate(instances):
                    if i < m:
                        state[4 + i % 24] = inst.get("validator_count", 4)

            # Build per-agent observations (m × 7)
            agent_obs = np.zeros((m, 7), dtype=np.float32)
            for i, inst in enumerate(instances):
                if i >= m:
                    break
                features = inst.get("trust_features", [])
                if features:
                    agent_obs[i, 0] = max(f.get("timeout_rate", 0) for f in features)
                    agent_obs[i, 1] = max(f.get("equivocation_rate", 0) for f in features)
                    agent_obs[i, 2] = max(f.get("view_change_rate", 0) for f in features)
                    agent_obs[i, 3] = np.mean([f.get("mean_latency", 0) for f in features])
                    agent_obs[i, 4] = np.mean([f.get("std_latency", 0) for f in features])
                agent_obs[i, 5] = inst.get("validator_count", 4) / 10.0
                agent_obs[i, 6] = float(i) / max(m - 1, 1)

            # Pad agent_obs if m < n_agents
            padded_obs = np.zeros((self.config.n_agents, 7), dtype=np.float32)
            padded_obs[:m] = agent_obs[:m]

            # Select actions (GPU inference)
            explore = self.train_mode and self.decisions < 5000
            noise_scale = max(0.05, 0.3 * (1.0 - self.decisions / 10000))
            actions = self.trainer.select_actions(
                padded_obs, explore=explore, noise_scale=noise_scale
            )

            # Cold-start conservative prior (Algorithm 2 Line 16):
            # When policy is not yet converged (warm=False), bias actions
            # toward conservative defaults: suppress detection signal,
            # shrink parameter deltas toward 0.
            if not getattr(self.trainer, 'converged', True):
                lambda_cs = 0.5  # cold-start bias strength
                # Conservative prior: detection → 0, deltas → 0
                conservative = np.zeros_like(actions)
                actions = actions + lambda_cs * (conservative - actions)

            # Apply safety guardian modulation (soft constraint, before hard mask)
            if self.safety_guardian is not None:
                modulation = self.safety_guardian.select_modulation(
                    state, explore=explore
                )
                actions = apply_safety_modulation(actions, modulation)
                self._last_modulation = modulation
            else:
                self._last_modulation = None

            # Pre-argmax safety mask (Eq.3, Lemma CSM C1-C3):
            # Clamp detection_signal to 0 for instances where eviction would
            # violate |Ω|-1 ≥ 3f+1+δ_s. This is the continuous-action
            # equivalent of setting unsafe logits to -∞ before argmax.
            actions = pre_argmax_safety_mask(actions, instances)

            # Store for transition construction on feedback
            self._last_state = state.copy()
            self._last_agent_obs = padded_obs.copy()
            self._last_actions = actions.copy()

            self.epoch += 1
            self.decisions += 1

            # Background training step
            if (self.train_mode
                    and self.decisions % self.train_every == 0
                    and self.trainer.can_train()):
                self.trainer.train_step()

            # Convert to SFACResponse format
            response_actions = []
            for k in range(m):
                # actions[k] = [committee_delta, timeout_delta, batch_delta, interval_delta]
                a = actions[k]
                # Detection threshold: use safety guardian's reconfig_threshold if available
                detect_threshold = 0.3
                if self._last_modulation is not None:
                    detect_threshold = self._last_modulation.reconfig_threshold
                detected = float(a[0]) > detect_threshold
                reconfig = []
                features_list = instances[k].get("trust_features", []) if k < len(instances) else []
                if detected and features_list:
                    # Score agents by adversary indicators
                    scores = [
                        f.get("timeout_rate", 0) + f.get("equivocation_rate", 0)
                        for f in features_list
                    ]
                    target_idx = int(np.argmax(scores))
                    reconfig = [0] * len(scores)
                    reconfig[target_idx] = -1
                elif detected:
                    reconfig = [-1]
                else:
                    reconfig = [0]

                action_dict = {
                    "instance_id": k,
                    "reconfig": reconfig,
                    "rotate": bool(detected),
                    "params": [float(a[1]), float(a[2]), float(a[3])],
                }
                response_actions.append(action_dict)

            result = {
                "actions": response_actions,
                "value": 0.0,
                "device": self.config.device,
                "train_steps": self.trainer.train_steps,
            }
            if self._last_modulation is not None:
                result["safety_modulation"] = {
                    "risk_tolerance": self._last_modulation.risk_tolerance,
                    "action_scale": self._last_modulation.action_scale,
                    "reconfig_threshold": self._last_modulation.reconfig_threshold,
                }
            return result

    def feedback(self, reward_data: dict) -> dict:
        """Store transition and optionally train."""
        with self.lock:
            if not self.train_mode:
                return {"status": "ignored", "reason": "eval-mode"}

            per_inst_rewards = reward_data.get("per_instance_rewards", [])
            done = reward_data.get("done", False)

            if self._last_state is None:
                return {"status": "skipped", "reason": "no_prior_decision"}

            # Team reward = mean of per-instance rewards
            team_reward = float(np.mean(per_inst_rewards)) if per_inst_rewards else 0.0

            # Role rewards: from Go reward.go RoleRewards map
            role_rewards_dict = reward_data.get("role_rewards", {})
            role_rewards_arr: np.ndarray | None = None
            if role_rewards_dict:
                role_rewards_arr = np.array([
                    float(role_rewards_dict.get("lane_tuner", 0.0)),
                    float(role_rewards_dict.get("recovery_tuner", 0.0)),
                    float(role_rewards_dict.get("membership_tuner", 0.0)),
                    float(role_rewards_dict.get("safety_guardian", 0.0)),
                ], dtype=np.float32)

            # Build next state (simplified: same as last with reward signal)
            next_state = self._last_state.copy()
            next_agent_obs = self._last_agent_obs.copy()

            transition = Transition(
                state=self._last_state,
                agent_obs=self._last_agent_obs,
                actions=self._last_actions,
                reward=team_reward,
                next_state=next_state,
                next_agent_obs=next_agent_obs,
                done=done,
                n_agents=self.m,
                role_rewards=role_rewards_arr,
            )
            self.trainer.store_transition(transition)

            # Train safety guardian (separate from main trainer)
            if self.safety_guardian is not None and self._last_modulation is not None:
                # Safety reward from Go's safety_guardian role
                safety_reward = 0.0
                if role_rewards_arr is not None:
                    safety_reward = float(role_rewards_arr[3])  # safety_guardian index
                else:
                    # Fallback: no violation = small positive
                    safety_reward = 0.1

                self.safety_guardian.store_experience(
                    state=self._last_state,
                    modulation=self._last_modulation.to_array(),
                    safety_reward=safety_reward,
                    next_state=next_state,
                    done=done,
                )
                # Train safety guardian every 8 feedback calls
                if self.decisions % 8 == 0:
                    self.safety_guardian.train_step()

            result = {
                "status": "stored",
                "buffer_size": len(self.trainer.buffer),
                "train_steps": self.trainer.train_steps,
            }

            return result

    def reset(self) -> dict:
        """Reset episode state."""
        with self.lock:
            self._last_state = None
            self._last_agent_obs = None
            self._last_actions = None
            self._last_modulation = None
            self.epoch = 0
            return {"status": "reset", "train_steps": self.trainer.train_steps}

    def health(self) -> dict:
        """Return bridge health with GPU status and safety constraint info."""
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_allocated_mb": round(torch.cuda.memory_allocated(0) / 1024**2, 1),
                "gpu_memory_total_mb": round(torch.cuda.get_device_properties(0).total_mem / 1024**2, 1),
            }

        result = {
            "status": "ok",
            "sfac_available": True,
            "backend": "pytorch",
            "device": self.config.device,
            "decisions": self.decisions,
            "epoch": self.epoch,
            "train_mode": self.train_mode,
            "train_steps": self.trainer.train_steps,
            "buffer_size": len(self.trainer.buffer),
            "can_train": self.trainer.can_train(),
            "m_instances": self.m,
            "safe_training": self.safe_training,
            **gpu_info,
        }

        # Add safety constraint stats if using SafeRoleFACMACTrainer
        if self.safe_training and isinstance(self.trainer, SafeRoleFACMACTrainer):
            result["lambda"] = self.trainer.lambda_value
            result["constraint_satisfied"] = self.trainer.is_constraint_satisfied(window=100)
            result.update(self.trainer.get_constraint_stats())

        # Safety guardian active policy stats
        if self.safety_guardian is not None:
            result["safety_guardian_params"] = self.safety_guardian.param_count()
            result["safety_guardian_train_steps"] = self.safety_guardian.train_steps

        return result

    def save(self, path: str) -> dict:
        """Save PyTorch checkpoint."""
        with self.lock:
            self.trainer.save(path)
            return {"status": "saved", "path": path, "train_steps": self.trainer.train_steps}

    def load(self, path: str) -> dict:
        """Load PyTorch checkpoint."""
        with self.lock:
            self.trainer.load(path)
            return {"status": "loaded", "path": path, "train_steps": self.trainer.train_steps}

    def export_numpy(self) -> dict:
        """Export current PyTorch weights as numpy arrays for lightweight serving.

        Returns dict of numpy arrays that can be used by the existing numpy policy
        for production deployment without torch dependency.
        """
        with self.lock:
            state_dict = self.trainer.actor.state_dict()
            numpy_weights = {}
            for name, tensor in state_dict.items():
                numpy_weights[f"actor.{name}"] = tensor.cpu().numpy().tolist()

            critic_dict = self.trainer.critic.state_dict()
            for name, tensor in critic_dict.items():
                numpy_weights[f"critic.{name}"] = tensor.cpu().numpy().tolist()

            mixer_dict = self.trainer.mixer.state_dict()
            for name, tensor in mixer_dict.items():
                numpy_weights[f"mixer.{name}"] = tensor.cpu().numpy().tolist()

            return {
                "status": "exported",
                "n_params": sum(p.numel() for p in self.trainer.actor.parameters())
                    + sum(p.numel() for p in self.trainer.critic.parameters())
                    + sum(p.numel() for p in self.trainer.mixer.parameters()),
                "weights": numpy_weights,
            }

    def metrics(self) -> dict:
        """Return training metrics summary including safety constraints."""
        m = self.trainer.metrics
        n = min(100, len(m.critic_loss) if hasattr(m, 'critic_loss') else len(m.total_critic_loss))

        result = {
            "train_steps": self.trainer.train_steps,
            "buffer_size": len(self.trainer.buffer),
        }

        if hasattr(m, 'critic_loss') and m.critic_loss:
            # Base FACMACTrainer metrics
            result["recent_critic_loss"] = float(np.mean(m.critic_loss[-n:]))
            result["recent_actor_loss"] = float(np.mean(m.actor_loss[-n:])) if m.actor_loss else 0.0
            result["recent_mean_q"] = float(np.mean(m.mean_q[-n:])) if m.mean_q else 0.0
            result["recent_td_error"] = float(np.mean(m.td_error_mean[-n:])) if m.td_error_mean else 0.0
        elif hasattr(m, 'total_critic_loss') and m.total_critic_loss:
            # RoleFACMACTrainer / SafeRoleFACMACTrainer metrics
            n = min(100, len(m.total_critic_loss))
            result["recent_critic_loss"] = float(np.mean(m.total_critic_loss[-n:]))
            result["recent_actor_loss"] = float(np.mean(m.actor_loss[-n:])) if m.actor_loss else 0.0
            result["recent_mean_q"] = float(np.mean(m.mean_q[-n:])) if m.mean_q else 0.0
            result["recent_td_error"] = float(np.mean(m.td_error_mean[-n:])) if m.td_error_mean else 0.0
            # Per-role losses
            for role, losses in m.role_critic_losses.items():
                if losses:
                    result[f"recent_{role}_loss"] = float(np.mean(losses[-n:]))

        # Safety constraint metrics
        if self.safe_training and isinstance(self.trainer, SafeRoleFACMACTrainer):
            sm = self.trainer.safe_metrics
            n_safe = min(100, len(sm.lambda_value))
            if n_safe > 0:
                result["lambda"] = self.trainer.lambda_value
                result["recent_cost"] = float(np.mean(sm.cost_estimate[-n_safe:]))
                result["recent_violation"] = float(np.mean(sm.constraint_violation[-n_safe:]))
                result["constraint_satisfied"] = self.trainer.is_constraint_satisfied(window=100)

        return result
