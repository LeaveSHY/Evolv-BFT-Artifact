"""QMIX Monotonic Mixing Network for SFAC.

Paper mapping:
  - Eq.8: Q_tot = g_ψ(s, Q_1, ..., Q_m) with IGM guarantee (ε_IGM = 0)
  - §III-B: Factored Q-value decomposition across m BFT instances
  - Monotonicity: ∂Q_tot/∂Q_i ≥ 0 via absolute-value hypernetwork weights

Architecture (QMIX, Rashid et al. 2018):
  Hypernetwork 1: state → |w1| (n_agents × hidden)    [positive via abs()]
  Bias network 1:  state → b1  (hidden)
  Hidden:         ELU(Q · |w1| + b1)
  Hypernetwork 2: state → |w2| (hidden × 1)           [positive via abs()]
  Bias network 2:  state → b2  (scalar, via 2-layer net)
  Output:         Q_tot = h · |w2| + b2
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QMIXMixer(nn.Module):
    """QMIX monotonic mixing network guaranteeing IGM (Individual-Global-Max).

    Given per-agent Q-values Q_i and global state s, produces Q_tot such that:
        argmax_a Q_tot(s, a) = (argmax_{a_1} Q_1(s, a_1), ..., argmax_{a_m} Q_m(s, a_m))

    This is achieved by enforcing ∂Q_tot/∂Q_i ≥ 0 through absolute-value weights.
    """

    def __init__(self, n_agents: int, state_dim: int = 28, mixing_hidden: int = 32):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.mixing_hidden = mixing_hidden

        # Hypernetwork 1: state → weights for layer 1 (n_agents × hidden)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden),
            nn.ReLU(),
            nn.Linear(mixing_hidden, n_agents * mixing_hidden),
        )
        # Bias network 1: state → bias for layer 1 (hidden)
        self.hyper_b1 = nn.Linear(state_dim, mixing_hidden)

        # Hypernetwork 2: state → weights for layer 2 (hidden × 1)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden),
            nn.ReLU(),
            nn.Linear(mixing_hidden, mixing_hidden),
        )
        # Bias network 2: state → scalar (via 2-layer net for expressiveness)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden),
            nn.ReLU(),
            nn.Linear(mixing_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Mix per-agent Q-values into Q_tot with monotonicity guarantee.

        Args:
            q_values: (batch, n_agents) per-agent Q-values
            state: (batch, state_dim) global state

        Returns:
            (batch, 1) Q_tot
        """
        batch_size = q_values.shape[0]

        # Layer 1: Q · |w1| + b1
        w1 = torch.abs(self.hyper_w1(state))  # (batch, n_agents * hidden)
        w1 = w1.view(batch_size, self.n_agents, self.mixing_hidden)  # (batch, n_agents, hidden)
        b1 = self.hyper_b1(state).unsqueeze(1)  # (batch, 1, hidden)

        # q_values: (batch, 1, n_agents) × w1: (batch, n_agents, hidden) → (batch, 1, hidden)
        q_reshaped = q_values.unsqueeze(1)  # (batch, 1, n_agents)
        h = F.elu(torch.bmm(q_reshaped, w1) + b1)  # (batch, 1, hidden)

        # Layer 2: h · |w2| + b2
        w2 = torch.abs(self.hyper_w2(state))  # (batch, hidden)
        w2 = w2.unsqueeze(2)  # (batch, hidden, 1)
        b2 = self.hyper_b2(state)  # (batch, 1)

        # h: (batch, 1, hidden) × w2: (batch, hidden, 1) → (batch, 1, 1)
        q_tot = torch.bmm(h, w2).squeeze(2) + b2  # (batch, 1)

        return q_tot

    def monotonicity_check(self, q_values: torch.Tensor, state: torch.Tensor) -> bool:
        """Verify monotonicity property holds (for debugging/testing).

        Returns True if ∂Q_tot/∂Q_i ≥ 0 for all i.
        """
        q_values.requires_grad_(True)
        q_tot = self.forward(q_values, state)
        grads = torch.autograd.grad(
            q_tot.sum(), q_values, create_graph=False
        )[0]
        return bool((grads >= -1e-6).all().item())
