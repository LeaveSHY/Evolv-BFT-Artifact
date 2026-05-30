"""Target network utilities for stable training.

Implements Polyak-averaged target networks (soft update) for actor-critic stability:
    θ_target ← τ · θ_online + (1 - τ) · θ_target

Paper mapping:
  - Standard MADDPG/FACMAC practice for off-policy training
  - τ = 0.005 (default, matches FACMAC paper)
"""
from __future__ import annotations

import torch.nn as nn


def soft_update(target: nn.Module, source: nn.Module, tau: float = 0.005) -> None:
    """Polyak-average update: θ_target ← τ·θ_source + (1-τ)·θ_target.

    Args:
        target: target network to update
        source: source (online) network
        tau: interpolation coefficient (0 = no update, 1 = hard copy)
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )


def hard_update(target: nn.Module, source: nn.Module) -> None:
    """Hard copy: θ_target ← θ_source.

    Used for initialization of target networks.
    """
    target.load_state_dict(source.state_dict())
