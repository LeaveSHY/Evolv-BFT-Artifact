"""Centralized Critic for SFAC (Safe Factored Actor-Critic).

Paper mapping:
  - §III-B: Centralized training with decentralized execution (CTDE)
  - Eq.14: Reward decomposition R = λ₁·tp + λ₂·lat + λ₃·vc + λ₄·safety
  - The critic estimates Q(s, a₁, ..., aₘ) for the joint state-action pair

Architecture:
  - Input: global state (28) + joint action (7 + m*4)
  - Output: scalar Q-value per agent (for QMIX mixing)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CriticNetwork(nn.Module):
    """Centralized critic for FACMAC-style training.

    Computes per-agent Q-values given global state and joint actions.
    These Q-values feed into the QMIX mixer for Q_tot computation.

    Architecture:
        [state(28), action_i(4)] → Linear(256) → ReLU → Linear(128) → ReLU → Linear(1)

    Each agent has its own Q-value computed from:
        - Global state features (shared across agents)
        - Its own local action (agent-specific)
    """

    def __init__(
        self,
        state_dim: int = 28,
        agent_action_dim: int = 4,
        hidden_dims: tuple[int, ...] = (256, 128),
    ):
        super().__init__()
        in_dim = state_dim + agent_action_dim
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor, agent_action: torch.Tensor) -> torch.Tensor:
        """Compute Q-value for one agent given global state and its action.

        Args:
            state: (batch, state_dim) global observation
            agent_action: (batch, agent_action_dim) single agent's action

        Returns:
            (batch, 1) Q-value
        """
        x = torch.cat([state, agent_action], dim=-1)
        return self.net(x)

    def forward_all_agents(self, state: torch.Tensor, agent_actions: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all agents simultaneously.

        Args:
            state: (batch, state_dim) global observation
            agent_actions: (batch, n_agents, agent_action_dim) all agents' actions

        Returns:
            (batch, n_agents) per-agent Q-values
        """
        batch_size, n_agents, action_dim = agent_actions.shape
        # Expand state: (batch, state_dim) → (batch, n_agents, state_dim)
        state_expanded = state.unsqueeze(1).expand(-1, n_agents, -1)
        # Concatenate: (batch, n_agents, state_dim + action_dim)
        x = torch.cat([state_expanded, agent_actions], dim=-1)
        # Reshape for batch processing: (batch * n_agents, state_dim + action_dim)
        x_flat = x.reshape(batch_size * n_agents, -1)
        # Forward: (batch * n_agents, 1)
        q_flat = self.net(x_flat)
        # Reshape back: (batch, n_agents)
        return q_flat.reshape(batch_size, n_agents)
