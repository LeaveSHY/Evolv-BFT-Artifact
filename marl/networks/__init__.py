"""PyTorch networks for Evolv-BFT SFAC (Safe Factored Actor-Critic).

Architecture (paper §III-B, Appendix B):
  - Actor: per-agent MLP mapping local observation to continuous action
  - Critic: centralized Q-function with global state + joint actions
  - Mixer: QMIX monotonic mixing network (Eq.8) ensuring IGM
  - Role heads: 4 MOISE+ roles with field-family action subspaces
"""
from marl.networks.actor import ActorNetwork, AgentActorNetwork, RoleHead
from marl.networks.critic import CriticNetwork
from marl.networks.mixer import QMIXMixer
from marl.networks.role_critic import RoleCritic, ROLE_NAMES, NUM_ROLES
from marl.networks.safety_guardian import (
    SafetyGuardianConfig,
    SafetyGuardianNetwork,
    SafetyGuardianPolicy,
    SafetyModulation,
    apply_safety_modulation,
)
from marl.networks.target import soft_update, hard_update

__all__ = [
    "ActorNetwork",
    "AgentActorNetwork",
    "RoleHead",
    "CriticNetwork",
    "RoleCritic",
    "ROLE_NAMES",
    "NUM_ROLES",
    "QMIXMixer",
    "SafetyGuardianConfig",
    "SafetyGuardianNetwork",
    "SafetyGuardianPolicy",
    "SafetyModulation",
    "apply_safety_modulation",
    "soft_update",
    "hard_update",
]
