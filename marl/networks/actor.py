"""Actor networks for SFAC (Safe Factored Actor-Critic).

Paper mapping:
  - Global Actor: obs(28) → action(7) [committee, timeout, batch, interval, join, leave, discovery]
  - Agent Actor: agent_obs(7) → agent_action(4) [per-instance tuning]
  - Role Heads: 4 MOISE+ roles each controlling a field-family subset
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    """Global actor mapping full observation to continuous action vector.

    Architecture:
        obs(28) → Linear(128) → ReLU → Linear(64) → ReLU → Linear(7) → Tanh

    Output range: [-1, 1] (scaled to action bounds in policy layer).
    """

    def __init__(self, obs_dim: int = 28, action_dim: int = 7, hidden_dims: tuple[int, ...] = (128, 64)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, action_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            obs: (batch, obs_dim) observation tensor

        Returns:
            (batch, action_dim) continuous action in [-1, 1]
        """
        x = self.backbone(obs)
        return torch.tanh(self.head(x))


class AgentActorNetwork(nn.Module):
    """Per-agent actor for instance-level parameter tuning.

    Architecture:
        agent_obs(7) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(4) → Tanh

    Each BFT instance has an agent controlling:
        [committee_size, pacemaker_timeout_ms, mempool_max_batch_txs, mempool_proposal_interval_ms]
    """

    def __init__(self, agent_obs_dim: int = 7, agent_action_dim: int = 4, hidden_dims: tuple[int, ...] = (64, 32)):
        super().__init__()
        layers = []
        in_dim = agent_obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, agent_action_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, agent_obs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            agent_obs: (batch, n_agents, agent_obs_dim) or (batch, agent_obs_dim)

        Returns:
            (batch, n_agents, agent_action_dim) or (batch, agent_action_dim)
        """
        x = self.backbone(agent_obs)
        return torch.tanh(self.head(x))


class RoleHead(nn.Module):
    """MOISE+ role-specific action head.

    Each role controls a subset of action fields (field-family decomposition):
        - lane_tuner: (committee_size, mempool_max_batch_txs, mempool_proposal_interval_ms)
        - recovery_tuner: (pacemaker_timeout_ms,)
        - membership_tuner: (submit_join, submit_leave, hydra_discovery_target)
        - safety_guardian: () — no direct action, only vetoes via safety mask

    Architecture:
        obs(28) → Linear(64) → ReLU → Linear(role_action_dim)
    """

    ROLE_DIMS = {
        "lane_tuner": 3,
        "recovery_tuner": 1,
        "membership_tuner": 3,
        "safety_guardian": 0,
    }

    def __init__(self, role_name: str, obs_dim: int = 28, hidden_dim: int = 64):
        super().__init__()
        self.role_name = role_name
        self.action_dim = self.ROLE_DIMS.get(role_name, 0)
        if self.action_dim == 0:
            self.net = None
            return
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        if self.net is None:
            return
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor | None:
        """Forward pass. Returns None for safety_guardian (no direct action)."""
        if self.net is None:
            return None
        return self.net(obs)
