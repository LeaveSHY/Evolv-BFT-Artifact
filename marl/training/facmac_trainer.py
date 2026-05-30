"""PyTorch FACMAC Trainer with QMIX mixing and prioritized replay.

Paper mapping:
  - Algorithm 1 (SafeMARL) from §III
  - CTDE: Centralized Training (critic sees global state), Decentralized Execution (actors use local obs)
  - QMIX mixing (Eq.8): Q_tot = g_ψ(s, Q_1, ..., Q_m) with ∂Q_tot/∂Q_i ≥ 0
  - Soft target updates (Polyak averaging, τ=0.005)
  - Prioritized Experience Replay (Schaul et al. 2016)

Training loop:
  1. Sample minibatch from PER buffer
  2. Compute target Q: y = r + γ · Q_tot_target(s', μ_target(o'))
  3. Minimize critic loss: L_critic = Σ w_i · (y_i - Q_tot(s, a))²
  4. Maximize actor: ∇_θ J = Σ ∇_a Q_tot(s, a)|_{a=μ(o)} · ∇_θ μ(o)
  5. Update PER priorities with |TD-error|
  6. Soft-update target networks
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from marl.networks import (
    ActorNetwork,
    AgentActorNetwork,
    CriticNetwork,
    QMIXMixer,
    soft_update,
    hard_update,
)
from marl.networks.safety_guardian import pre_argmax_safety_mask_torch
from marl.training.replay_buffer import PrioritizedReplayBuffer, Transition


@dataclass
class TrainingConfig:
    """Hyperparameters for FACMAC training."""
    # Architecture
    state_dim: int = 28
    agent_obs_dim: int = 7
    agent_action_dim: int = 4
    action_dim: int = 7
    n_agents: int = 10
    mixing_hidden: int = 32

    # Training
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    mixer_lr: float = 1e-3
    gamma: float = 0.95
    tau: float = 0.005
    batch_size: int = 64
    max_grad_norm: float = 10.0

    # Replay buffer
    buffer_capacity: int = 100_000
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_epsilon: float = 1e-5

    # Training schedule
    warmup_steps: int = 1000
    total_steps: int = 500_000
    update_every: int = 4
    target_update_every: int = 1  # soft update every optimization step

    # Convergence gate (Algorithm 2, Line 18: warm_{t+1} ← ‖∇J‖ < ξ)
    convergence_xi: float = 0.01
    convergence_window: int = 50

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingMetrics:
    """Accumulated training metrics."""
    critic_loss: list[float] = field(default_factory=list)
    actor_loss: list[float] = field(default_factory=list)
    mean_q: list[float] = field(default_factory=list)
    mean_reward: list[float] = field(default_factory=list)
    td_error_mean: list[float] = field(default_factory=list)


class FACMACTrainer:
    """Factored Actor-Critic with Monotonic Mixing (FACMAC) PyTorch trainer.

    Implements centralized training with decentralized execution (CTDE):
    - Each agent has a local actor π_i(o_i) producing per-agent actions
    - A centralized critic Q(s, a_1, ..., a_m) evaluates joint actions
    - QMIX mixer ensures Q_tot monotonically depends on per-agent Q_i
    - Target networks provide stable TD targets
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device)
        self._build_networks()
        self._build_optimizers()
        self._build_buffer()
        self.metrics = TrainingMetrics()
        self.train_steps = 0
        # Convergence gate state (Algorithm 2, Line 18)
        self._grad_norm_history: list[float] = []
        self._converged = False

    def _build_networks(self):
        cfg = self.config

        # Online networks
        self.actor = AgentActorNetwork(
            agent_obs_dim=cfg.agent_obs_dim,
            agent_action_dim=cfg.agent_action_dim,
        ).to(self.device)

        self.critic = CriticNetwork(
            state_dim=cfg.state_dim,
            agent_action_dim=cfg.agent_action_dim,
        ).to(self.device)

        self.mixer = QMIXMixer(
            n_agents=cfg.n_agents,
            state_dim=cfg.state_dim,
            mixing_hidden=cfg.mixing_hidden,
        ).to(self.device)

        # Target networks (frozen copies)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.mixer_target = copy.deepcopy(self.mixer)

        # Freeze target parameters
        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic_target.parameters():
            p.requires_grad = False
        for p in self.mixer_target.parameters():
            p.requires_grad = False

    def _build_optimizers(self):
        cfg = self.config
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=cfg.actor_lr
        )
        self.critic_optimizer = optim.Adam(
            list(self.critic.parameters()) + list(self.mixer.parameters()),
            lr=cfg.critic_lr,
        )

    def _build_buffer(self):
        cfg = self.config
        self.buffer = PrioritizedReplayBuffer(
            capacity=cfg.buffer_capacity,
            alpha=cfg.per_alpha,
            beta_start=cfg.per_beta_start,
            epsilon=cfg.per_epsilon,
        )

    def store_transition(self, transition: Transition):
        """Store a transition in the replay buffer."""
        self.buffer.add(transition)

    def can_train(self) -> bool:
        """Check if enough samples are available for training."""
        return len(self.buffer) >= self.config.warmup_steps

    @property
    def converged(self) -> bool:
        """Algorithm 2 Line 18: warm_{t+1} ← (‖∇J‖ < ξ over window)."""
        return self._converged

    def train_step(self) -> dict[str, float]:
        """Execute one training step (critic + actor + target update).

        Returns:
            Dictionary of step metrics.
        """
        if not self.can_train():
            return {}

        cfg = self.config
        self.train_steps += 1

        # Anneal PER beta
        fraction = min(1.0, self.train_steps / cfg.total_steps)
        self.buffer.anneal_beta(fraction)

        # Sample from PER buffer
        transitions, indices, is_weights = self.buffer.sample(cfg.batch_size)

        # Collate batch (handle variable n_agents by padding)
        batch = self._collate(transitions)
        is_weights_t = torch.from_numpy(is_weights).to(self.device)

        # === Critic Update ===
        with torch.no_grad():
            # Target actions from target actor
            next_actions = self.actor_target(batch["next_agent_obs"])  # (B, n_agents, action_dim)
            # Pre-argmax mask on target actions (consistent with actor update)
            next_val_counts = (batch["next_agent_obs"][:, :, 5] * 10.0).clamp(min=1)
            next_faults = torch.ones_like(next_val_counts)
            next_actions = pre_argmax_safety_mask_torch(
                next_actions, next_val_counts, next_faults
            )
            # Target per-agent Q values
            next_q_per_agent = self.critic_target.forward_all_agents(
                batch["next_state"], next_actions
            )  # (B, n_agents)
            # Target Q_tot via target mixer
            next_q_tot = self.mixer_target(next_q_per_agent, batch["next_state"])  # (B, 1)
            # TD target
            targets = batch["reward"] + cfg.gamma * (1.0 - batch["done"]) * next_q_tot  # (B, 1)

        # Current per-agent Q values
        q_per_agent = self.critic.forward_all_agents(
            batch["state"], batch["actions"]
        )  # (B, n_agents)
        # Current Q_tot
        q_tot = self.mixer(q_per_agent, batch["state"])  # (B, 1)

        # Weighted MSE loss (PER importance sampling)
        td_errors = targets - q_tot  # (B, 1)
        critic_loss = (is_weights_t.unsqueeze(1) * td_errors.pow(2)).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic.parameters()) + list(self.mixer.parameters()),
            cfg.max_grad_norm,
        )
        self.critic_optimizer.step()

        # === Actor Update ===
        # Deterministic policy gradient: maximize Q_tot w.r.t. actor parameters
        current_actions = self.actor(batch["agent_obs"])  # (B, n_agents, action_dim)

        # Pre-argmax safety mask (Eq.3, Lemma CSM C1-C3):
        # Clamp detection_signal to 0 for instances where eviction would violate
        # |Ω|-1 ≥ 3f+1+δ_s. Applied BEFORE critic evaluation so gradients
        # respect the safety constraint (no Ω(f/n) projection bias).
        # validator_count is encoded in agent_obs[:, :, 5] as val_count / 10.0
        validator_counts = (batch["agent_obs"][:, :, 5] * 10.0).clamp(min=1)
        # Conservative faults_estimate = 1 (Go SafetyFilter default)
        faults_estimates = torch.ones_like(validator_counts)
        current_actions = pre_argmax_safety_mask_torch(
            current_actions, validator_counts, faults_estimates
        )

        q_per_agent_actor = self.critic.forward_all_agents(
            batch["state"], current_actions
        )
        q_tot_actor = self.mixer(q_per_agent_actor, batch["state"])
        actor_loss = -q_tot_actor.mean()  # maximize Q → minimize -Q

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
        self.actor_optimizer.step()

        # === Convergence Gate (Algorithm 2 Line 18) ===
        # Track ‖∇_θ J‖ to determine warm_{t+1}
        actor_grad_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in self.actor.parameters()
            if p.grad is not None
        ) ** 0.5
        self._grad_norm_history.append(actor_grad_norm)
        if len(self._grad_norm_history) > cfg.convergence_window:
            self._grad_norm_history = self._grad_norm_history[-cfg.convergence_window:]
        if (
            len(self._grad_norm_history) >= cfg.convergence_window
            and np.mean(self._grad_norm_history) < cfg.convergence_xi
        ):
            self._converged = True

        # === Target Network Update ===
        soft_update(self.actor_target, self.actor, cfg.tau)
        soft_update(self.critic_target, self.critic, cfg.tau)
        soft_update(self.mixer_target, self.mixer, cfg.tau)

        # === Update PER Priorities ===
        td_abs = td_errors.detach().abs().squeeze(1).cpu().numpy()
        self.buffer.update_priorities(indices, td_abs)

        # Record metrics
        metrics = {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "mean_q": q_tot.detach().mean().item(),
            "td_error_mean": td_abs.mean().item(),
            "actor_grad_norm": actor_grad_norm,
            "converged": self._converged,
        }
        self.metrics.critic_loss.append(metrics["critic_loss"])
        self.metrics.actor_loss.append(metrics["actor_loss"])
        self.metrics.mean_q.append(metrics["mean_q"])
        self.metrics.td_error_mean.append(metrics["td_error_mean"])

        return metrics

    def _collate(self, transitions: list[Transition]) -> dict[str, torch.Tensor]:
        """Collate transitions into padded batch tensors.

        Handles variable n_agents by padding to config.n_agents.
        """
        cfg = self.config
        batch_size = len(transitions)

        states = np.zeros((batch_size, cfg.state_dim), dtype=np.float32)
        next_states = np.zeros((batch_size, cfg.state_dim), dtype=np.float32)
        agent_obs = np.zeros((batch_size, cfg.n_agents, cfg.agent_obs_dim), dtype=np.float32)
        next_agent_obs = np.zeros((batch_size, cfg.n_agents, cfg.agent_obs_dim), dtype=np.float32)
        actions = np.zeros((batch_size, cfg.n_agents, cfg.agent_action_dim), dtype=np.float32)
        rewards = np.zeros((batch_size, 1), dtype=np.float32)
        dones = np.zeros((batch_size, 1), dtype=np.float32)

        for i, t in enumerate(transitions):
            states[i] = t.state
            next_states[i] = t.next_state
            n = min(t.n_agents, cfg.n_agents)
            agent_obs[i, :n] = t.agent_obs[:n]
            next_agent_obs[i, :n] = t.next_agent_obs[:n]
            actions[i, :n] = t.actions[:n]
            rewards[i, 0] = t.reward
            dones[i, 0] = float(t.done)

        return {
            "state": torch.from_numpy(states).to(self.device),
            "next_state": torch.from_numpy(next_states).to(self.device),
            "agent_obs": torch.from_numpy(agent_obs).to(self.device),
            "next_agent_obs": torch.from_numpy(next_agent_obs).to(self.device),
            "actions": torch.from_numpy(actions).to(self.device),
            "reward": torch.from_numpy(rewards).to(self.device),
            "done": torch.from_numpy(dones).to(self.device),
        }

    def select_actions(self, agent_obs: np.ndarray, explore: bool = False, noise_scale: float = 0.1) -> np.ndarray:
        """Select actions for all agents (decentralized execution).

        Args:
            agent_obs: (n_agents, agent_obs_dim)
            explore: if True, add Gaussian exploration noise
            noise_scale: std of exploration noise

        Returns:
            (n_agents, agent_action_dim) actions clipped to [-1, 1]
        """
        with torch.no_grad():
            obs_t = torch.from_numpy(agent_obs).float().unsqueeze(0).to(self.device)
            actions = self.actor(obs_t).squeeze(0).cpu().numpy()

        if explore:
            noise = np.random.normal(0, noise_scale, size=actions.shape)
            actions = np.clip(actions + noise, -1.0, 1.0)

        return actions

    def save(self, path: str):
        """Save all model state dicts."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "mixer": self.mixer.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "mixer_target": self.mixer_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "train_steps": self.train_steps,
            "config": self.config.__dict__,
        }, path)

    def load(self, path: str):
        """Load model state dicts."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.mixer_target.load_state_dict(checkpoint["mixer_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.train_steps = checkpoint.get("train_steps", 0)
