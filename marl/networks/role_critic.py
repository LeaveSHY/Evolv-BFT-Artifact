"""Role-Decomposed Critic for multi-objective SFAC training.

Paper mapping:
  - §III-C: Self-Evolving Adaptation — role-specific reward signals drive
    specialized critic heads, each estimating the expected return for its role.
  - Go reward.go: RoleRewards = {lane_tuner, recovery_tuner, membership_tuner, safety_guardian}
  - Training: each head minimizes its own TD error; actor gradient is the weighted
    sum of per-role policy gradients (multi-objective factored DPG).

Architecture:
  Shared trunk:   [state(28), action_i(4)] → Linear(256) → ReLU → Linear(128) → ReLU
  Per-role heads:  Linear(128) → ReLU → Linear(1)   ×4
  Team head:       Linear(128) → ReLU → Linear(1)   ×1
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

# Role names must match Go reward.go JSON keys and Python schemas.
#
# Cross-language role mapping (Go roles.go → reward.go → Python):
#   MOISE+ org role     →  Reward head (wire name)  →  Responsibility
#   sentinel            →  recovery_tuner           →  detection quality
#   commander           →  membership_tuner         →  membership decisions
#   tuner               →  lane_tuner               →  throughput/latency params
#   guardian            →  safety_guardian           →  safety constraints
#
# See Go adaptive/roles.go RoleToRewardHead for canonical mapping.
ROLE_NAMES: tuple[str, ...] = (
    "lane_tuner",
    "recovery_tuner",
    "membership_tuner",
    "safety_guardian",
)
NUM_ROLES = len(ROLE_NAMES)


class RoleCritic(nn.Module):
    """Multi-head critic with shared feature extraction and per-role Q-heads.

    Shared trunk extracts joint state-action features.
    Each role head independently estimates Q_role(s, a_i).
    An additional team head estimates Q_team(s, a_i) for backward compatibility.

    This enables multi-objective training:
      L_critic = Σ_role α_role · MSE(Q_role, r_role + γ Q_role')
      ∇_actor  = Σ_role α_role · ∇_a Q_role(s, a)
    """

    def __init__(
        self,
        state_dim: int = 28,
        agent_action_dim: int = 4,
        trunk_dims: tuple[int, ...] = (256, 128),
        head_hidden: int = 64,
        role_weights: Sequence[float] | None = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.agent_action_dim = agent_action_dim

        # Default role weights for actor gradient aggregation
        if role_weights is None:
            # Equal weight per role + team head
            self._role_weights = [1.0] * NUM_ROLES
        else:
            assert len(role_weights) == NUM_ROLES
            self._role_weights = list(role_weights)

        # Shared trunk
        in_dim = state_dim + agent_action_dim
        trunk_layers: list[nn.Module] = []
        for h in trunk_dims:
            trunk_layers.append(nn.Linear(in_dim, h))
            trunk_layers.append(nn.ReLU())
            in_dim = h
        self.trunk = nn.Sequential(*trunk_layers)
        self.trunk_out_dim = in_dim  # Last hidden dimension

        # Per-role Q-heads
        self.role_heads = nn.ModuleDict()
        for role in ROLE_NAMES:
            self.role_heads[role] = nn.Sequential(
                nn.Linear(self.trunk_out_dim, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, 1),
            )

        # Team Q-head (for compatibility and weighted aggregation)
        self.team_head = nn.Sequential(
            nn.Linear(self.trunk_out_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    @property
    def role_weights_tensor(self) -> torch.Tensor:
        """Normalized role weights as tensor."""
        w = torch.tensor(self._role_weights, dtype=torch.float32)
        return w / w.sum()

    def forward_trunk(self, state: torch.Tensor, agent_action: torch.Tensor) -> torch.Tensor:
        """Shared feature extraction.

        Args:
            state: (batch, state_dim)
            agent_action: (batch, agent_action_dim)

        Returns:
            (batch, trunk_out_dim) shared features
        """
        x = torch.cat([state, agent_action], dim=-1)
        return self.trunk(x)

    def forward_role(self, features: torch.Tensor, role: str) -> torch.Tensor:
        """Single role Q-value from pre-computed features.

        Args:
            features: (batch, trunk_out_dim)
            role: one of ROLE_NAMES

        Returns:
            (batch, 1)
        """
        return self.role_heads[role](features)

    def forward_team(self, features: torch.Tensor) -> torch.Tensor:
        """Team Q-value from pre-computed features.

        Args:
            features: (batch, trunk_out_dim)

        Returns:
            (batch, 1)
        """
        return self.team_head(features)

    def forward_all_roles(self, state: torch.Tensor, agent_action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute all role Q-values + team Q-value.

        Args:
            state: (batch, state_dim)
            agent_action: (batch, agent_action_dim)

        Returns:
            Dict mapping role_name → (batch, 1), plus "team" → (batch, 1)
        """
        features = self.forward_trunk(state, agent_action)
        result = {}
        for role in ROLE_NAMES:
            result[role] = self.role_heads[role](features)
        result["team"] = self.team_head(features)
        return result

    def forward_all_agents(self, state: torch.Tensor, agent_actions: torch.Tensor) -> torch.Tensor:
        """Compute team Q-values for all agents (backward-compatible with CriticNetwork).

        Args:
            state: (batch, state_dim)
            agent_actions: (batch, n_agents, agent_action_dim)

        Returns:
            (batch, n_agents) per-agent team Q-values
        """
        batch_size, n_agents, action_dim = agent_actions.shape
        state_expanded = state.unsqueeze(1).expand(-1, n_agents, -1)
        x = torch.cat([state_expanded, agent_actions], dim=-1)
        x_flat = x.reshape(batch_size * n_agents, -1)
        features = self.trunk(x_flat)
        q_team = self.team_head(features)
        return q_team.reshape(batch_size, n_agents)

    def forward_all_agents_all_roles(
        self, state: torch.Tensor, agent_actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute per-role Q-values for all agents.

        Args:
            state: (batch, state_dim)
            agent_actions: (batch, n_agents, agent_action_dim)

        Returns:
            Dict mapping role_name → (batch, n_agents), plus "team" → (batch, n_agents)
        """
        batch_size, n_agents, action_dim = agent_actions.shape
        state_expanded = state.unsqueeze(1).expand(-1, n_agents, -1)
        x = torch.cat([state_expanded, agent_actions], dim=-1)
        x_flat = x.reshape(batch_size * n_agents, -1)
        features = self.trunk(x_flat)

        result = {}
        for role in ROLE_NAMES:
            q = self.role_heads[role](features)
            result[role] = q.reshape(batch_size, n_agents)
        q_team = self.team_head(features)
        result["team"] = q_team.reshape(batch_size, n_agents)
        return result

    def weighted_q(self, state: torch.Tensor, agent_action: torch.Tensor) -> torch.Tensor:
        """Compute weighted aggregate Q-value across all roles.

        Used for actor gradient: ∇_a Σ α_role · Q_role(s, a).

        Args:
            state: (batch, state_dim)
            agent_action: (batch, agent_action_dim)

        Returns:
            (batch, 1) weighted Q-value
        """
        features = self.forward_trunk(state, agent_action)
        w = self.role_weights_tensor.to(features.device)  # (NUM_ROLES,)
        q_vals = torch.cat(
            [self.role_heads[role](features) for role in ROLE_NAMES], dim=-1
        )  # (batch, NUM_ROLES)
        return (q_vals * w.unsqueeze(0)).sum(dim=-1, keepdim=True)  # (batch, 1)

    def weighted_q_all_agents(self, state: torch.Tensor, agent_actions: torch.Tensor) -> torch.Tensor:
        """Weighted Q-value for all agents (for actor gradient + QMIX mixing).

        Args:
            state: (batch, state_dim)
            agent_actions: (batch, n_agents, agent_action_dim)

        Returns:
            (batch, n_agents) weighted per-agent Q-values
        """
        batch_size, n_agents, action_dim = agent_actions.shape
        state_expanded = state.unsqueeze(1).expand(-1, n_agents, -1)
        x = torch.cat([state_expanded, agent_actions], dim=-1)
        x_flat = x.reshape(batch_size * n_agents, -1)
        features = self.trunk(x_flat)

        w = self.role_weights_tensor.to(features.device)
        q_vals = torch.cat(
            [self.role_heads[role](features) for role in ROLE_NAMES], dim=-1
        )  # (B*n_agents, NUM_ROLES)
        weighted = (q_vals * w.unsqueeze(0)).sum(dim=-1, keepdim=True)  # (B*n_agents, 1)
        return weighted.reshape(batch_size, n_agents)
