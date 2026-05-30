"""SFAC Training Package — PyTorch GPU-accelerated FACMAC trainer."""
from marl.training.replay_buffer import PrioritizedReplayBuffer, Transition
from marl.training.facmac_trainer import FACMACTrainer, TrainingConfig
from marl.training.role_trainer import RoleFACMACTrainer, RoleTrainingConfig
from marl.training.safe_trainer import SafeRoleFACMACTrainer, SafeTrainingConfig

__all__ = [
    "PrioritizedReplayBuffer",
    "Transition",
    "FACMACTrainer",
    "TrainingConfig",
    "RoleFACMACTrainer",
    "RoleTrainingConfig",
    "SafeRoleFACMACTrainer",
    "SafeTrainingConfig",
]
