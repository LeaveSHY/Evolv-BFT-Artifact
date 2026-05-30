"""Role-Decomposed FACMAC Trainer — multi-objective per-role training.

Paper mapping:
  - §III-C (Self-Evolving Adaptation): each role has a dedicated reward signal
    (lane_tuner, recovery_tuner, membership_tuner, safety_guardian) from Go reward.go
  - Training decomposes the single-critic TD loss into per-role TD losses:
      L = Σ_role α_role · L_TD(Q_role, r_role + γ Q_role')
  - Actor gradient is the weighted sum of per-role policy gradients:
      ∇_θ J = Σ_role α_role · ∇_a Q_role(s, a)|_{a=μ(o)} · ∇_θ μ(o)
  - The QMIX mixer operates on the weighted per-agent Q-values to produce Q_tot.
  - Backward-compatible: when role_rewards are absent, falls back to team reward.

Architecture:
  Actor: AgentActorNetwork (shared, same as base FACMAC)
  Critic: RoleCritic (multi-head: 4 role heads + 1 team head, shared trunk)
  Mixer: QMIXMixer (operates on weighted Q from RoleCritic)
  Buffer: PrioritizedReplayBuffer (Transition now carries optional role_rewards)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from marl.networks import AgentActorNetwork, QMIXMixer, soft_update
from marl.networks.role_critic import NUM_ROLES, ROLE_NAMES, RoleCritic
from marl.training.replay_buffer import PrioritizedReplayBuffer, Transition


@dataclass
class RoleTrainingConfig:
    """Hyperparameters for role-decomposed FACMAC training."""
    # Architecture
    state_dim: int = 28
    agent_obs_dim: int = 7
    agent_action_dim: int = 4
    n_agents: int = 10
    mixing_hidden: int = 32
    trunk_dims: tuple[int, ...] = (256, 128)
    head_hidden: int = 64

    # Role weights: relative importance of each role's loss
    # Default: equal weight; can be tuned for safety emphasis
    role_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.5)  # safety_guardian gets 1.5x

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

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class RoleTrainingMetrics:
    """Training metrics with per-role breakdown."""
    total_critic_loss: list[float] = field(default_factory=list)
    actor_loss: list[float] = field(default_factory=list)
    mean_q: list[float] = field(default_factory=list)
    td_error_mean: list[float] = field(default_factory=list)
    # Per-role critic losses
    role_critic_losses: dict[str, list[float]] = field(default_factory=lambda: {r: [] for r in ROLE_NAMES})


class RoleFACMACTrainer:
    """FACMAC with role-decomposed multi-objective training.

    Key differences from standard FACMACTrainer:
    - Uses RoleCritic (multi-head) instead of single CriticNetwork
    - Computes per-role TD errors using role-specific rewards
    - Actor gradient is weighted sum of per-role policy gradients
    - Falls back to team reward when role_rewards are absent (backward-compatible)
    """

    def __init__(self, config: Optional[RoleTrainingConfig] = None):
        self.config = config or RoleTrainingConfig()
        self.device = torch.device(self.config.device)
        self._build_networks()
        self._build_optimizers()
        self._build_buffer()
        self.metrics = RoleTrainingMetrics()
        self.train_steps = 0

    def _build_networks(self):
        cfg = self.config

        # Actor (same as base FACMAC)
        self.actor = AgentActorNetwork(
            agent_obs_dim=cfg.agent_obs_dim,
            agent_action_dim=cfg.agent_action_dim,
        ).to(self.device)

        # Multi-head RoleCritic
        self.critic = RoleCritic(
            state_dim=cfg.state_dim,
            agent_action_dim=cfg.agent_action_dim,
            trunk_dims=cfg.trunk_dims,
            head_hidden=cfg.head_hidden,
            role_weights=list(cfg.role_weights),
        ).to(self.device)

        # QMIX mixer (operates on weighted Q per-agent)
        self.mixer = QMIXMixer(
            n_agents=cfg.n_agents,
            state_dim=cfg.state_dim,
            mixing_hidden=cfg.mixing_hidden,
        ).to(self.device)

        # Target networks
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.mixer_target = copy.deepcopy(self.mixer)
        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic_target.parameters():
            p.requires_grad = False
        for p in self.mixer_target.parameters():
            p.requires_grad = False

    def _build_optimizers(self):
        cfg = self.config
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
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
        """Check if enough samples for training."""
        return len(self.buffer) >= self.config.warmup_steps

    def train_step(self) -> dict[str, float]:
        """Execute one training step with role-decomposed losses.

        Critic update:
          For each role r: L_r = MSE(Q_r(s,a), r_r + γ Q_r'(s',a'))
          L_critic = Σ α_r · L_r   (+ team loss as regularization)

        Actor update:
          ∇_θ J = ∇_θ Q_weighted_tot(s, μ(o))
          where Q_weighted = Σ α_r · Q_r (mixed through QMIX for Q_tot)

        Returns dict of training metrics.
        """
        if not self.can_train():
            return {}

        cfg = self.config
        self.train_steps += 1

        # Anneal PER beta
        fraction = min(1.0, self.train_steps / cfg.total_steps)
        self.buffer.anneal_beta(fraction)

        # Sample batch
        transitions, indices, is_weights = self.buffer.sample(cfg.batch_size)
        batch = self._collate(transitions)
        is_weights_t = torch.from_numpy(is_weights).to(self.device)

        # Role weights as tensor
        role_w = torch.tensor(cfg.role_weights, dtype=torch.float32, device=self.device)
        role_w = role_w / role_w.sum()  # normalize

        # === Critic Update (multi-role TD) ===
        with torch.no_grad():
            # Target actions
            next_actions = self.actor_target(batch["next_agent_obs"])  # (B, n_agents, action_dim)
            # Target Q-values per role for all agents
            next_q_roles = self.critic_target.forward_all_agents_all_roles(
                batch["next_state"], next_actions
            )  # dict: role → (B, n_agents)

        # Current Q-values per role for all agents
        current_actions = batch["actions"]
        q_roles = self.critic.forward_all_agents_all_roles(
            batch["state"], current_actions
        )  # dict: role → (B, n_agents)

        # Compute per-role TD loss
        total_critic_loss = torch.tensor(0.0, device=self.device)
        role_losses = {}

        for role_idx, role in enumerate(ROLE_NAMES):
            # Per-role rewards: (B, 1)
            role_reward = batch["role_rewards"][:, role_idx:role_idx + 1]

            # Compute Q_tot via mixer for this role's per-agent Q-values
            q_per_agent = q_roles[role]  # (B, n_agents)
            q_tot = self.mixer(q_per_agent, batch["state"])  # (B, 1)

            with torch.no_grad():
                next_q_per_agent = next_q_roles[role]  # (B, n_agents)
                next_q_tot = self.mixer_target(next_q_per_agent, batch["next_state"])  # (B, 1)
                target = role_reward + cfg.gamma * (1.0 - batch["done"]) * next_q_tot

            td_error = target - q_tot
            role_loss = (is_weights_t.unsqueeze(1) * td_error.pow(2)).mean()
            role_losses[role] = role_loss.item()
            total_critic_loss = total_critic_loss + role_w[role_idx] * role_loss

        # Also train team head (regularization: team reward = sum of role rewards)
        q_team_per_agent = q_roles["team"]
        q_team_tot = self.mixer(q_team_per_agent, batch["state"])
        with torch.no_grad():
            next_q_team = next_q_roles["team"]
            next_q_team_tot = self.mixer_target(next_q_team, batch["next_state"])
            team_target = batch["reward"] + cfg.gamma * (1.0 - batch["done"]) * next_q_team_tot
        team_td = team_target - q_team_tot
        team_loss = (is_weights_t.unsqueeze(1) * team_td.pow(2)).mean()
        # Team loss contributes 0.5x weight (regularizer, not primary objective)
        total_critic_loss = total_critic_loss + 0.5 * team_loss

        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic.parameters()) + list(self.mixer.parameters()),
            cfg.max_grad_norm,
        )
        self.critic_optimizer.step()

        # === Actor Update (weighted multi-role policy gradient) ===
        # Use weighted Q across roles for actor gradient
        current_actions_actor = self.actor(batch["agent_obs"])
        q_weighted = self.critic.weighted_q_all_agents(
            batch["state"], current_actions_actor
        )  # (B, n_agents)
        q_tot_actor = self.mixer(q_weighted, batch["state"])  # (B, 1)
        actor_loss = -q_tot_actor.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
        self.actor_optimizer.step()

        # === Target Update ===
        soft_update(self.actor_target, self.actor, cfg.tau)
        soft_update(self.critic_target, self.critic, cfg.tau)
        soft_update(self.mixer_target, self.mixer, cfg.tau)

        # === PER Priority Update (use team TD error) ===
        td_abs = team_td.detach().abs().squeeze(1).cpu().numpy()
        self.buffer.update_priorities(indices, td_abs)

        # Record metrics
        metrics = {
            "total_critic_loss": total_critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "mean_q": q_team_tot.detach().mean().item(),
            "td_error_mean": td_abs.mean().item(),
            "team_critic_loss": team_loss.item(),
        }
        for role in ROLE_NAMES:
            metrics[f"critic_loss_{role}"] = role_losses[role]

        self.metrics.total_critic_loss.append(metrics["total_critic_loss"])
        self.metrics.actor_loss.append(metrics["actor_loss"])
        self.metrics.mean_q.append(metrics["mean_q"])
        self.metrics.td_error_mean.append(metrics["td_error_mean"])
        for role in ROLE_NAMES:
            self.metrics.role_critic_losses[role].append(role_losses[role])

        return metrics

    def _collate(self, transitions: list[Transition]) -> dict[str, torch.Tensor]:
        """Collate transitions into padded batch tensors with role rewards."""
        cfg = self.config
        batch_size = len(transitions)

        states = np.zeros((batch_size, cfg.state_dim), dtype=np.float32)
        next_states = np.zeros((batch_size, cfg.state_dim), dtype=np.float32)
        agent_obs = np.zeros((batch_size, cfg.n_agents, cfg.agent_obs_dim), dtype=np.float32)
        next_agent_obs = np.zeros((batch_size, cfg.n_agents, cfg.agent_obs_dim), dtype=np.float32)
        actions = np.zeros((batch_size, cfg.n_agents, cfg.agent_action_dim), dtype=np.float32)
        rewards = np.zeros((batch_size, 1), dtype=np.float32)
        role_rewards = np.zeros((batch_size, NUM_ROLES), dtype=np.float32)
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
            # Role rewards: use provided or fall back to splitting team reward equally
            if t.role_rewards is not None:
                role_rewards[i] = t.role_rewards[:NUM_ROLES]
            else:
                # Fallback: distribute team reward equally across roles
                role_rewards[i] = t.reward / NUM_ROLES

        return {
            "state": torch.from_numpy(states).to(self.device),
            "next_state": torch.from_numpy(next_states).to(self.device),
            "agent_obs": torch.from_numpy(agent_obs).to(self.device),
            "next_agent_obs": torch.from_numpy(next_agent_obs).to(self.device),
            "actions": torch.from_numpy(actions).to(self.device),
            "reward": torch.from_numpy(rewards).to(self.device),
            "role_rewards": torch.from_numpy(role_rewards).to(self.device),
            "done": torch.from_numpy(dones).to(self.device),
        }

    def select_actions(self, agent_obs: np.ndarray, explore: bool = False, noise_scale: float = 0.1) -> np.ndarray:
        """Select actions (decentralized execution)."""
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

    def param_count(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.actor.parameters() if p.requires_grad) + \
               sum(p.numel() for p in self.critic.parameters() if p.requires_grad) + \
               sum(p.numel() for p in self.mixer.parameters() if p.requires_grad)
