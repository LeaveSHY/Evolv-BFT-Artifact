"""Lagrangian Safety-Constrained Role-Decomposed FACMAC Trainer.

Paper mapping (§III-C Self-Evolving Adaptation):
  The safety constraint ensures that SFAC exploration never proposes consensus
  parameters violating BFT safety invariants. Formally:

    Primal:   min_θ  -J(π_θ) + λ · J_c(π_θ)
    Dual:     max_{λ≥0}  λ · (J_c(π) - d)

  where:
    J(π)  = E[Σ γ^t r_t]          (role-weighted reward)
    J_c(π) = E[Σ γ^t c_t]         (cumulative safety cost)
    c_t   = -reward_safety_guardian  (positive when safety violated)
    d     = cost_limit              (maximum acceptable cost per step)
    λ     = Lagrange multiplier     (auto-tuned via dual gradient ascent)

  The safety_guardian head of RoleCritic already estimates Q_cost.
  No additional network is needed.

Architecture:
  Inherits RoleFACMACTrainer fully.
  Adds:
    - log_lambda (learnable dual variable, log-space for positivity)
    - Modified actor loss: -Q_weighted_tot + λ · Q_cost_tot
    - Dual update: λ ← clip(λ + η·(Q_cost - d), 0, λ_max)
    - Cost tracking: rolling window for constraint satisfaction monitoring
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from marl.networks.role_critic import NUM_ROLES, ROLE_NAMES
from marl.training.role_trainer import RoleFACMACTrainer, RoleTrainingConfig


# Index of safety_guardian in ROLE_NAMES
SAFETY_ROLE_IDX = ROLE_NAMES.index("safety_guardian")


@dataclass
class SafeTrainingConfig(RoleTrainingConfig):
    """Extends RoleTrainingConfig with safety constraint parameters."""

    # Safety constraint: expected cost per step must be ≤ cost_limit
    cost_limit: float = 0.1

    # Dual variable (Lagrange multiplier)
    lambda_init: float = 0.1       # initial λ value
    lambda_lr: float = 5e-3        # dual learning rate
    lambda_max: float = 10.0       # cap to prevent unbounded penalty

    # Cost tracking
    cost_window_size: int = 1000   # rolling window for constraint monitoring

    # Whether to use log-parameterization (ensures λ > 0)
    log_lambda: bool = True


@dataclass
class SafeTrainingMetrics:
    """Extended metrics including safety constraint tracking."""
    lambda_value: list[float] = field(default_factory=list)
    cost_estimate: list[float] = field(default_factory=list)
    constraint_violation: list[float] = field(default_factory=list)
    safe_actor_loss: list[float] = field(default_factory=list)


class SafeRoleFACMACTrainer(RoleFACMACTrainer):
    """FACMAC with Lagrangian safety constraints on top of role-decomposed training.

    Key additions over RoleFACMACTrainer:
    1. Learns a Lagrange multiplier λ that penalizes safety violations in actor loss
    2. Actor loss = -Q_reward_tot + λ · Q_cost_tot (balances reward vs safety)
    3. Dual update: increases λ when constraint violated, decreases when satisfied
    4. Monitors rolling constraint satisfaction for early stopping / logging

    The safety cost is derived from the safety_guardian role:
      cost = -reward_safety_guardian (positive when safety is violated)
      Q_cost ≈ Q_safety_guardian with sign flip
    """

    def __init__(self, config: Optional[SafeTrainingConfig] = None):
        self.safe_config = config or SafeTrainingConfig()
        # Initialize parent with the same config (it's a RoleTrainingConfig subclass)
        super().__init__(self.safe_config)
        self._build_lagrangian()
        self.safe_metrics = SafeTrainingMetrics()

    def _build_lagrangian(self):
        """Initialize dual variable and its optimizer."""
        cfg = self.safe_config

        if cfg.log_lambda:
            # log-parameterization: λ = exp(log_lambda), ensures λ > 0
            init_val = float(np.log(max(cfg.lambda_init, 1e-8)))
            self._log_lambda = nn.Parameter(
                torch.tensor(init_val, dtype=torch.float32, device=self.device)
            )
        else:
            # Direct parameterization (clamped manually)
            self._log_lambda = nn.Parameter(
                torch.tensor(cfg.lambda_init, dtype=torch.float32, device=self.device)
            )

        self.lambda_optimizer = optim.Adam([self._log_lambda], lr=cfg.lambda_lr)

        # Rolling cost window for monitoring
        self._cost_window: deque[float] = deque(maxlen=cfg.cost_window_size)

    @property
    def lambda_value(self) -> float:
        """Current Lagrange multiplier value."""
        if self.safe_config.log_lambda:
            return self._log_lambda.exp().item()
        else:
            return max(0.0, self._log_lambda.item())

    def train_step(self) -> dict[str, float]:
        """Execute one training step with Lagrangian safety constraint.

        Extends parent's train_step:
        1. Critic update: same as RoleFACMACTrainer (multi-role TD)
        2. Actor update: MODIFIED — adds λ · Q_cost penalty
        3. Dual update: NEW — adjusts λ based on constraint satisfaction
        """
        if not self.can_train():
            return {}

        cfg = self.safe_config
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
        role_w = role_w / role_w.sum()

        # === CRITIC UPDATE (same as parent — multi-role TD) ===
        with torch.no_grad():
            next_actions = self.actor_target(batch["next_agent_obs"])
            next_q_roles = self.critic_target.forward_all_agents_all_roles(
                batch["next_state"], next_actions
            )

        current_actions = batch["actions"]
        q_roles = self.critic.forward_all_agents_all_roles(
            batch["state"], current_actions
        )

        total_critic_loss = torch.tensor(0.0, device=self.device)
        role_losses = {}

        for role_idx, role in enumerate(ROLE_NAMES):
            role_reward = batch["role_rewards"][:, role_idx:role_idx + 1]
            q_per_agent = q_roles[role]
            q_tot = self.mixer(q_per_agent, batch["state"])

            with torch.no_grad():
                next_q_per_agent = next_q_roles[role]
                next_q_tot = self.mixer_target(next_q_per_agent, batch["next_state"])
                target = role_reward + cfg.gamma * (1.0 - batch["done"]) * next_q_tot

            td_error = target - q_tot
            role_loss = (is_weights_t.unsqueeze(1) * td_error.pow(2)).mean()
            role_losses[role] = role_loss.item()
            total_critic_loss = total_critic_loss + role_w[role_idx] * role_loss

        # Team head
        q_team_per_agent = q_roles["team"]
        q_team_tot = self.mixer(q_team_per_agent, batch["state"])
        with torch.no_grad():
            next_q_team = next_q_roles["team"]
            next_q_team_tot = self.mixer_target(next_q_team, batch["next_state"])
            team_target = batch["reward"] + cfg.gamma * (1.0 - batch["done"]) * next_q_team_tot
        team_td = team_target - q_team_tot
        team_loss = (is_weights_t.unsqueeze(1) * team_td.pow(2)).mean()
        total_critic_loss = total_critic_loss + 0.5 * team_loss

        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic.parameters()) + list(self.mixer.parameters()),
            cfg.max_grad_norm,
        )
        self.critic_optimizer.step()

        # === ACTOR UPDATE (LAGRANGIAN-CONSTRAINED) ===
        # Reward gradient: maximize weighted Q
        current_actions_actor = self.actor(batch["agent_obs"])
        q_weighted = self.critic.weighted_q_all_agents(
            batch["state"], current_actions_actor
        )  # (B, n_agents)
        q_reward_tot = self.mixer(q_weighted, batch["state"])  # (B, 1)

        # Cost gradient: safety_guardian Q estimates expected cost
        # Cost = -safety_reward, so Q_cost = -Q_safety
        q_safety = self.critic.forward_all_agents_all_roles(
            batch["state"], current_actions_actor
        )["safety_guardian"]  # (B, n_agents)
        # Negate: higher Q_safety means more positive safety reward (good)
        # We want to penalize low Q_safety, so cost = -Q_safety_tot
        q_cost_tot = -self.mixer(q_safety, batch["state"])  # (B, 1), positive = bad

        # Current lambda
        if cfg.log_lambda:
            lam = self._log_lambda.exp()
        else:
            lam = torch.clamp(self._log_lambda, min=0.0)

        # Actor loss: minimize  -Q_reward + λ · Q_cost
        actor_loss = -q_reward_tot.mean() + lam.detach() * q_cost_tot.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
        self.actor_optimizer.step()

        # === DUAL UPDATE (Lagrange multiplier) ===
        # λ should increase when cost exceeds limit, decrease otherwise
        # Loss for λ: -λ · (J_cost - d)  (maximize)
        # Gradient ascent on λ: ∂/∂λ [λ(J_cost - d)] = J_cost - d
        cost_estimate = q_cost_tot.detach().mean().item()
        self._cost_window.append(cost_estimate)

        if cfg.log_lambda:
            # Gradient for log_lambda: λ · (cost - d) (chain rule through exp)
            dual_loss = -self._log_lambda.exp() * (
                q_cost_tot.detach().mean() - cfg.cost_limit
            )
        else:
            dual_loss = -self._log_lambda * (
                q_cost_tot.detach().mean() - cfg.cost_limit
            )

        self.lambda_optimizer.zero_grad()
        dual_loss.backward()
        self.lambda_optimizer.step()

        # Clamp lambda to [0, lambda_max]
        with torch.no_grad():
            if cfg.log_lambda:
                max_log = float(np.log(cfg.lambda_max))
                self._log_lambda.clamp_(max=max_log)
            else:
                self._log_lambda.clamp_(min=0.0, max=cfg.lambda_max)

        # === TARGET UPDATE ===
        from marl.networks import soft_update
        soft_update(self.actor_target, self.actor, cfg.tau)
        soft_update(self.critic_target, self.critic, cfg.tau)
        soft_update(self.mixer_target, self.mixer, cfg.tau)

        # === PER Priority Update ===
        td_abs = team_td.detach().abs().squeeze(1).cpu().numpy()
        self.buffer.update_priorities(indices, td_abs)

        # === Metrics ===
        lam_val = self.lambda_value
        constraint_violation = max(0.0, cost_estimate - cfg.cost_limit)

        metrics = {
            "total_critic_loss": total_critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "mean_q": q_reward_tot.detach().mean().item(),
            "td_error_mean": td_abs.mean().item(),
            "team_critic_loss": team_loss.item(),
            "lambda": lam_val,
            "cost_estimate": cost_estimate,
            "constraint_violation": constraint_violation,
            "cost_limit": cfg.cost_limit,
        }
        for role in ROLE_NAMES:
            metrics[f"critic_loss_{role}"] = role_losses[role]

        # Track safe metrics
        self.safe_metrics.lambda_value.append(lam_val)
        self.safe_metrics.cost_estimate.append(cost_estimate)
        self.safe_metrics.constraint_violation.append(constraint_violation)
        self.safe_metrics.safe_actor_loss.append(actor_loss.item())

        # Parent metrics
        self.metrics.total_critic_loss.append(total_critic_loss.item())
        self.metrics.actor_loss.append(actor_loss.item())
        self.metrics.mean_q.append(metrics["mean_q"])
        self.metrics.td_error_mean.append(td_abs.mean().item())
        for role in ROLE_NAMES:
            self.metrics.role_critic_losses[role].append(role_losses[role])

        return metrics

    def is_constraint_satisfied(self, window: int = 100) -> bool:
        """Check if safety constraint is satisfied over recent window."""
        if len(self._cost_window) < window:
            return False
        recent = list(self._cost_window)[-window:]
        avg_cost = sum(recent) / len(recent)
        return avg_cost <= self.safe_config.cost_limit

    def get_constraint_stats(self) -> dict[str, float]:
        """Get constraint satisfaction statistics."""
        if not self._cost_window:
            return {"avg_cost": 0.0, "max_cost": 0.0, "violation_rate": 0.0}
        costs = list(self._cost_window)
        limit = self.safe_config.cost_limit
        return {
            "avg_cost": sum(costs) / len(costs),
            "max_cost": max(costs),
            "violation_rate": sum(1.0 for c in costs if c > limit) / len(costs),
            "lambda": self.lambda_value,
        }

    def save(self, path: str):
        """Save all state including Lagrange multiplier."""
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "mixer": self.mixer.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "mixer_target": self.mixer_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "lambda_optimizer": self.lambda_optimizer.state_dict(),
            "log_lambda": self._log_lambda.data.clone(),
            "train_steps": self.train_steps,
            "config": self.safe_config.__dict__,
            "cost_window": list(self._cost_window),
        }
        torch.save(checkpoint, path)

    def load(self, path: str):
        """Load all state including Lagrange multiplier."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.mixer_target.load_state_dict(checkpoint["mixer_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.lambda_optimizer.load_state_dict(checkpoint["lambda_optimizer"])
        self._log_lambda.data.copy_(checkpoint["log_lambda"])
        self.train_steps = checkpoint.get("train_steps", 0)
        # Restore cost window
        window_data = checkpoint.get("cost_window", [])
        self._cost_window = deque(window_data, maxlen=self.safe_config.cost_window_size)
