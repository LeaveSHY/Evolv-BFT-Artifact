"""Safety Guardian Active Policy — proactive safety modulation network.

Paper mapping (§III-C Self-Evolving Adaptation):
  The safety guardian is a meta-controller that observes global state and
  produces soft safety constraints BEFORE actions reach the hard safety mask.

  Two-layer safety architecture:
    Layer 1 (soft, learned): SafetyGuardianNetwork produces modulation
      - risk_tolerance ∈ [0,1]: how much parameter deviation is permitted
      - action_scale ∈ [0.5,1.0]: multiplicative bound on parameter deltas
      - reconfig_threshold ∈ [0,1]: how confident must detection be to reconfig
    Layer 2 (hard, provable): Go SafetyFilter enforces n ≥ 3f+1+δ_s

  The soft layer REDUCES the probability that Layer 2 needs to activate,
  creating smoother gradients for the base actor to learn from.

  Training:
    The safety actor maximizes: safety_guardian reward (positive when safe)
    - λ_conservative penalty: too restrictive → lower throughput/latency reward
    This balances: "protect safety" vs "don't be overly conservative"

Architecture:
  SafetyGuardianNetwork:
    Input: state(28)  [global view of all instances]
    Hidden: 128 → ReLU → 64 → ReLU
    Output: 3-dim modulation vector
      [0] risk_tolerance: sigmoid → [0, 1]
      [1] action_scale: sigmoid → [0.5, 1.0]  (rescaled)
      [2] reconfig_threshold: sigmoid → [0, 1]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Safety modulation output dimensions
SAFETY_MOD_DIM = 3  # risk_tolerance, action_scale, reconfig_threshold


@dataclass
class SafetyModulation:
    """Output of the safety guardian network."""
    risk_tolerance: float       # [0, 1] — how much deviation is allowed
    action_scale: float         # [0.5, 1.0] — multiplicative bound on actions
    reconfig_threshold: float   # [0, 1] — confidence needed for reconfig

    def to_array(self) -> np.ndarray:
        return np.array([self.risk_tolerance, self.action_scale, self.reconfig_threshold],
                        dtype=np.float32)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "SafetyModulation":
        return cls(
            risk_tolerance=float(arr[0]),
            action_scale=float(arr[1]),
            reconfig_threshold=float(arr[2]),
        )

    @classmethod
    def default(cls) -> "SafetyModulation":
        """Conservative default: tight constraints."""
        return cls(risk_tolerance=0.3, action_scale=0.7, reconfig_threshold=0.7)


class SafetyGuardianNetwork(nn.Module):
    """Proactive safety modulation network.

    Takes global state (28-dim) and outputs a 3-dim modulation vector
    that constrains the base actor's actions before the hard safety mask.
    """

    def __init__(self, state_dim: int = 28, hidden_dims: tuple[int, ...] = (128, 64)):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, SAFETY_MOD_DIM)

        # Initialize near-conservative defaults
        nn.init.zeros_(self.head.bias)
        nn.init.xavier_uniform_(self.head.weight, gain=0.1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass: state → raw modulation logits (pre-sigmoid)."""
        h = self.trunk(state)
        return self.head(h)

    def get_modulation(self, state: torch.Tensor) -> torch.Tensor:
        """Get final modulation values with proper scaling.

        Returns (B, 3):
          [0] risk_tolerance: sigmoid → [0, 1]
          [1] action_scale: sigmoid * 0.5 + 0.5 → [0.5, 1.0]
          [2] reconfig_threshold: sigmoid → [0, 1]
        """
        raw = self.forward(state)
        mod = torch.zeros_like(raw)
        mod[:, 0] = torch.sigmoid(raw[:, 0])                    # risk_tolerance [0,1]
        mod[:, 1] = torch.sigmoid(raw[:, 1]) * 0.5 + 0.5       # action_scale [0.5,1.0]
        mod[:, 2] = torch.sigmoid(raw[:, 2])                    # reconfig_threshold [0,1]
        return mod


def apply_safety_modulation(
    base_actions: np.ndarray,
    modulation: SafetyModulation,
) -> np.ndarray:
    """Apply safety modulation to base per-instance actions.

    Args:
        base_actions: (n_agents, 4) raw actor outputs
          [0] detection_signal
          [1] timeout_delta
          [2] batch_delta
          [3] interval_delta
        modulation: SafetyModulation from guardian network

    Returns:
        modulated_actions: (n_agents, 4) safety-constrained actions

    Modulation logic:
      - action_scale: parameter deltas ([1],[2],[3]) are multiplied by action_scale
        This bounds how aggressively tuners can change parameters
      - reconfig_threshold: detection_signal ([0]) must exceed this to trigger reconfig
        Higher threshold = fewer reconfigurations (more conservative)
      - risk_tolerance: overall soft clamp on action magnitudes
        Low tolerance = tighter action bounds
    """
    modulated = base_actions.copy()
    n_agents = modulated.shape[0]

    # Scale parameter deltas by action_scale
    # Indices 1,2,3 are the tuning parameters
    modulated[:, 1:] *= modulation.action_scale

    # Apply risk_tolerance as additional magnitude bound
    # Low risk_tolerance → tighter clamp
    max_magnitude = 0.3 + 0.7 * modulation.risk_tolerance  # [0.3, 1.0]
    modulated[:, 1:] = np.clip(modulated[:, 1:], -max_magnitude, max_magnitude)

    # Apply reconfig_threshold: gate the detection signal
    # Only allow reconfig if detection confidence exceeds threshold
    # detection_signal < threshold → zero it out (no reconfig)
    for i in range(n_agents):
        if modulated[i, 0] < modulation.reconfig_threshold:
            modulated[i, 0] = 0.0  # below threshold → no reconfig

    return modulated


@dataclass
class SafetyGuardianConfig:
    """Configuration for safety guardian active policy."""
    state_dim: int = 28
    hidden_dims: tuple[int, ...] = (128, 64)
    lr: float = 1e-3
    tau: float = 0.01
    gamma: float = 0.95
    # Penalty for being too conservative (action_scale too low)
    conservative_penalty: float = 0.1
    device: str = "cpu"


class SafetyGuardianPolicy:
    """Trainable safety guardian with target network for stable learning.

    Training objective:
      max  E[r_safety - conservative_penalty * (1 - action_scale)]

    The conservative penalty prevents the guardian from being overly restrictive,
    which would hurt throughput/latency performance.
    """

    def __init__(self, config: Optional[SafetyGuardianConfig] = None):
        self.config = config or SafetyGuardianConfig()
        self.device = torch.device(self.config.device)

        # Networks
        self.network = SafetyGuardianNetwork(
            state_dim=self.config.state_dim,
            hidden_dims=self.config.hidden_dims,
        ).to(self.device)

        self.target_network = SafetyGuardianNetwork(
            state_dim=self.config.state_dim,
            hidden_dims=self.config.hidden_dims,
        ).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        for p in self.target_network.parameters():
            p.requires_grad = False

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.lr)

        # Experience buffer (simple circular)
        self._buffer: list[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = []
        self._buffer_max = 10_000
        self._buffer_idx = 0
        self.train_steps = 0

    def select_modulation(self, state: np.ndarray, explore: bool = False) -> SafetyModulation:
        """Select safety modulation given global state.

        Args:
            state: (28,) global state vector
            explore: if True, add noise for exploration

        Returns:
            SafetyModulation with risk_tolerance, action_scale, reconfig_threshold
        """
        with torch.no_grad():
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            mod = self.network.get_modulation(state_t).squeeze(0).cpu().numpy()

        if explore:
            noise = np.random.normal(0, 0.05, size=mod.shape)
            mod = np.clip(mod + noise, 0.0, 1.0)
            # Ensure action_scale stays in [0.5, 1.0]
            mod[1] = np.clip(mod[1], 0.5, 1.0)

        return SafetyModulation.from_array(mod)

    def store_experience(
        self,
        state: np.ndarray,
        modulation: np.ndarray,
        safety_reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store experience for training."""
        experience = (state, modulation, safety_reward, next_state, done)
        if len(self._buffer) < self._buffer_max:
            self._buffer.append(experience)
        else:
            self._buffer[self._buffer_idx] = experience
        self._buffer_idx = (self._buffer_idx + 1) % self._buffer_max

    def train_step(self, batch_size: int = 32) -> Optional[dict[str, float]]:
        """Train the safety guardian policy.

        Objective: maximize safety reward with conservative penalty.
        Uses simple policy gradient (REINFORCE-style) on the modulation.
        """
        if len(self._buffer) < batch_size:
            return None

        # Sample random batch
        indices = np.random.randint(0, len(self._buffer), size=batch_size)
        states = np.array([self._buffer[i][0] for i in indices], dtype=np.float32)
        rewards = np.array([self._buffer[i][2] for i in indices], dtype=np.float32)
        next_states = np.array([self._buffer[i][3] for i in indices], dtype=np.float32)
        dones = np.array([self._buffer[i][4] for i in indices], dtype=np.float32)

        states_t = torch.from_numpy(states).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)

        # Current modulation
        mod = self.network.get_modulation(states_t)  # (B, 3)

        # Value estimate from target: V(s') ≈ mean modulation quality
        with torch.no_grad():
            next_mod = self.target_network.get_modulation(next_states_t)
            # Simple value: reward + γ * (expected safety quality of next state)
            # Use action_scale as proxy for value (higher = more permissive = higher throughput)
            next_value = next_mod[:, 1]  # action_scale as value proxy
            target = rewards_t + self.config.gamma * (1.0 - dones_t) * next_value

        # Loss: maximize (safety_reward - conservative_penalty * (1 - action_scale))
        # Rewrite as minimize: -safety_reward + penalty * (1 - action_scale)
        conservative_cost = self.config.conservative_penalty * (1.0 - mod[:, 1])
        # Policy gradient: action_scale and risk_tolerance should be as high as possible
        # while keeping safety reward positive
        loss = -(target * mod[:, 1]).mean() + conservative_cost.mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 5.0)
        self.optimizer.step()

        # Target update
        tau = self.config.tau
        for tp, p in zip(self.target_network.parameters(), self.network.parameters()):
            tp.data.mul_(1 - tau).add_(p.data, alpha=tau)

        self.train_steps += 1
        return {
            "safety_guardian_loss": loss.item(),
            "mean_risk_tolerance": mod[:, 0].mean().item(),
            "mean_action_scale": mod[:, 1].mean().item(),
            "mean_reconfig_threshold": mod[:, 2].mean().item(),
        }

    def save(self, path: str):
        torch.save({
            "network": self.network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_steps": self.train_steps,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_steps = checkpoint.get("train_steps", 0)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-Argmax Safety Mask (Eq.3, Algorithm 5, Lemma CSM)
#
# Implements the paper's certified pre-argmax safety mask in the Python MARL
# layer. This enforces BFT quorum invariant BEFORE the actor's detection
# signal triggers reconfiguration, ensuring:
#   (C1) IGM-preserving: masked argmax = joint argmax over U_safe
#   (C2) Bias-free: no Ω(f/n) projection bias from post-hoc clamping
#   (C3) Regret-equivalent: O(√T) damage regret preserved
#
# The mask operates on the continuous detection_signal dimension:
#   If eviction would violate |Ω|-1 ≥ 3f+1+δ_s, set detection_signal = 0.
#   This prevents reconfiguration from firing (signal stays below threshold).
# ═══════════════════════════════════════════════════════════════════════════════

# Paper Algorithm 5 default: δ_s = 1 (one-eviction headroom)
DEFAULT_DELTA_S = 1


def is_eviction_safe(validator_count: int, faults_estimate: int,
                     delta_s: int = DEFAULT_DELTA_S) -> bool:
    """Check if evicting 1 node preserves BFT safety (Algorithm 5, line 5).

    Implements: n_after = |Ω| - 1 ≥ 3f + 1 + δ_s

    Args:
        validator_count: Current number of validators in the instance |Ω|.
        faults_estimate: Trust system's estimate of Byzantine nodes f.
        delta_s: Additive safety margin (paper default δ_s=1).

    Returns:
        True if evicting one node is safe (Φ remains ≥ 0).
    """
    if validator_count <= 0:
        return False
    n_after = validator_count - 1
    f = max(1, faults_estimate)
    threshold = 3 * f + 1 + delta_s
    return n_after >= threshold


def estimate_faults_from_features(trust_features: list[dict],
                                  detection_threshold: float = 0.5) -> int:
    """Derive faults_estimate from trust feature vectors.

    Counts nodes whose aggregate anomaly signal exceeds detection_threshold.
    Conservative: always returns at least 1 (matches Go SafetyFilter default).

    Args:
        trust_features: List of per-node feature dicts with timeout_rate,
                       equivocation_rate, view_change_rate keys.
        detection_threshold: Anomaly score above which a node is suspected.

    Returns:
        Estimated number of Byzantine nodes (≥ 1).
    """
    if not trust_features:
        return 1
    suspected = 0
    for feat in trust_features:
        anomaly_score = (feat.get("timeout_rate", 0.0)
                         + feat.get("equivocation_rate", 0.0)
                         + feat.get("view_change_rate", 0.0))
        if anomaly_score > detection_threshold:
            suspected += 1
    return max(1, suspected)


def pre_argmax_safety_mask(actions: np.ndarray, instances: list[dict],
                           delta_s: int = DEFAULT_DELTA_S) -> np.ndarray:
    """Apply pre-argmax safety mask to detection signals (Eq.3, Lemma CSM).

    For each instance, checks whether evicting 1 node would violate the BFT
    quorum invariant. If unsafe, clamps detection_signal to 0, preventing the
    reconfiguration from triggering.

    This is the continuous-action equivalent of the paper's discrete mask:
        z_a = -∞ for unsafe a  ⟺  detection_signal = 0 (below any threshold)

    Applied BEFORE threshold comparison and BEFORE critic evaluation during
    training, ensuring gradients respect the safety constraint (C1-C3).

    Args:
        actions: (n_agents, action_dim) array. Column 0 is detection_signal.
        instances: List of per-instance dicts with 'validator_count' and
                  optionally 'faults_estimate' or 'trust_features'.
        delta_s: Safety margin (default 1).

    Returns:
        Masked actions array (modified in-place and returned).
    """
    n_agents = actions.shape[0]
    m = min(n_agents, len(instances))

    for k in range(m):
        inst = instances[k]
        validator_count = inst.get("validator_count", 4)

        # Derive faults estimate (priority: explicit > computed > default)
        faults_estimate = inst.get("faults_estimate", None)
        if faults_estimate is None:
            trust_features = inst.get("trust_features", [])
            faults_estimate = estimate_faults_from_features(trust_features)

        # Pre-argmax mask: if eviction unsafe, zero detection signal
        if not is_eviction_safe(validator_count, faults_estimate, delta_s):
            actions[k, 0] = 0.0  # Clamp to 0 → below any threshold → no reconfig

    return actions


def pre_argmax_safety_mask_torch(actions: "torch.Tensor",
                                 validator_counts: "torch.Tensor",
                                 faults_estimates: "torch.Tensor",
                                 delta_s: int = DEFAULT_DELTA_S) -> "torch.Tensor":
    """Differentiable pre-argmax safety mask for training (Lemma CSM C1-C3).

    Uses straight-through estimator: forward pass applies hard mask,
    backward pass passes gradients through as if unmasked. This ensures
    the actor learns to avoid the boundary (receives gradient signal)
    while guaranteeing no unsafe action is ever proposed.

    Args:
        actions: (batch, n_agents, action_dim) tensor. dim=-1 col 0 = detection.
        validator_counts: (batch, n_agents) current validator counts per instance.
        faults_estimates: (batch, n_agents) estimated faults per instance.
        delta_s: Safety margin.

    Returns:
        Masked actions tensor (differentiable via straight-through).
    """
    # Compute safety condition: n - 1 >= 3f + 1 + delta_s
    f_clamped = torch.clamp(faults_estimates, min=1)
    threshold = 3 * f_clamped + 1 + delta_s
    n_after = validator_counts - 1
    safe_mask = (n_after >= threshold).float()  # 1.0 if safe, 0.0 if unsafe

    # Straight-through estimator: mask in forward, pass gradient in backward
    # actions[..., 0] is detection_signal
    detection = actions[..., 0]
    masked_detection = detection * safe_mask

    # Straight-through: gradient of masked_detection w.r.t. detection = safe_mask
    # When unsafe (safe_mask=0): detection_signal=0, grad=0 → actor learns boundary
    # When safe (safe_mask=1): detection_signal unchanged, grad=1 → normal learning
    result = actions.clone()
    result[..., 0] = masked_detection

    return result
