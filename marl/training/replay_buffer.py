"""Prioritized Experience Replay Buffer for SFAC training.

Paper mapping:
  - Proportional prioritization (Schaul et al. 2016)
  - Priority = |TD-error| + ε, importance-sampling bias correction
  - Supports variable n_agents (instances join/leave dynamically)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class Transition:
    """Single multi-agent transition."""
    state: np.ndarray          # (state_dim,)
    agent_obs: np.ndarray      # (n_agents, agent_obs_dim)
    actions: np.ndarray        # (n_agents, agent_action_dim)
    reward: float              # team reward scalar
    next_state: np.ndarray     # (state_dim,)
    next_agent_obs: np.ndarray # (n_agents, agent_obs_dim)
    done: bool
    n_agents: int
    role_rewards: Optional[np.ndarray] = None  # (NUM_ROLES,) per-role rewards


class SumTree:
    """Binary sum tree for O(log N) proportional sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0
        self.size = 0

    @property
    def total(self) -> float:
        return self.tree[0]

    def update(self, idx: int, priority: float):
        tree_idx = idx + self.capacity - 1
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += delta

    def add(self, priority: float) -> int:
        idx = self.data_pointer
        self.update(idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return idx

    def sample(self, value: float) -> int:
        """Sample index proportional to priority."""
        tree_idx = 0
        while tree_idx < self.capacity - 1:
            left = 2 * tree_idx + 1
            right = left + 1
            if value <= self.tree[left]:
                tree_idx = left
            else:
                value -= self.tree[left]
                tree_idx = right
        data_idx = tree_idx - (self.capacity - 1)
        return data_idx


class PrioritizedReplayBuffer:
    """Proportional prioritized replay with importance-sampling correction.

    Args:
        capacity: maximum buffer size
        alpha: prioritization exponent (0 = uniform, 1 = full priority)
        beta_start: initial IS exponent (annealed to 1.0 over training)
        epsilon: small constant to prevent zero priority
    """

    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        epsilon: float = 1e-5,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.epsilon = epsilon
        self.tree = SumTree(capacity)
        self.data: list[Optional[Transition]] = [None] * capacity
        self.max_priority = 1.0

    def __len__(self) -> int:
        return self.tree.size

    def add(self, transition: Transition, td_error: Optional[float] = None):
        """Add transition with priority = |td_error|^alpha or max_priority."""
        if td_error is not None:
            priority = (abs(td_error) + self.epsilon) ** self.alpha
        else:
            priority = self.max_priority
        idx = self.tree.add(priority)
        self.data[idx] = transition

    def sample(self, batch_size: int) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        """Sample batch proportional to priorities.

        Returns:
            transitions: list of sampled Transition objects
            indices: array of buffer indices (for priority update)
            weights: IS weights (batch_size,) normalized to max=1
        """
        indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        segment = self.tree.total / batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)
            idx = self.tree.sample(value)
            # Handle empty slots (shouldn't happen if len >= batch_size)
            while self.data[idx] is None:
                value = np.random.uniform(0, self.tree.total)
                idx = self.tree.sample(value)
            indices[i] = idx
            priorities[i] = self.tree.tree[idx + self.tree.capacity - 1]

        # Importance-sampling weights
        total = self.tree.total
        min_prob = priorities.min() / total
        max_weight = (self.tree.size * min_prob) ** (-self.beta)
        probs = priorities / total
        weights = (self.tree.size * probs) ** (-self.beta)
        weights /= max_weight  # normalize so max weight = 1

        transitions = [self.data[idx] for idx in indices]
        return transitions, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities after computing new TD errors."""
        for idx, td in zip(indices, td_errors):
            priority = (abs(td) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(int(idx), priority)

    def anneal_beta(self, fraction: float):
        """Anneal IS correction: beta → 1.0 over training."""
        self.beta = self.beta_start + fraction * (1.0 - self.beta_start)
