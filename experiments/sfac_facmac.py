#!/usr/bin/env python3
"""
Safe Factored Actor-Critic (SFAC) with FACMAC — Paper-aligned implementation.

Matches Algorithm 1 (SafeMARL) in Section III-D exactly:
  - Per-agent deterministic actors μ_i(o_i; φ_i) with shared weights (P5 roles)
  - Per-agent Q_i(o_i, a_i; θ_i) with monotonic mixing → Q_tot (P2 centralized critic)
  - Pre-argmax quorum safety filter: n_after >= 3f+1 (P3)
  - Regret-aligned reward (Eq.14, P1): λ₁·tp - λ₂·ℓ - λ₃·vc - λ₄·margin
  - Trust estimation (Eq.5-6): f_hat = σ(w·x + b) — jointly trained (P4)
  - Prioritized Experience Replay (PER)
  - Target networks with soft update

Interface matches OctopusMARLController in run_e2e_experiments.py:
  - FACMACController(cfg) constructor
  - .m attribute
  - .reset(), .set_eval(), .set_train()
  - .decide(raw_signal, instance_sizes, epoch, T_total) → dict
  - .store_transition(per_inst_rewards, done) → accepts (m,) rewards + done flag
  - .train_step() → dict

Hyperparameters from Appendix (Appendix:Hyperparams):
  h=64, |D|=100000, batch=256, lr_actor=3e-4, lr_critic=3e-4, γ=0.99
  W=50, τ_target=0.005, PER α=0.6, β annealed 0.4→1.0
  λ₁=1.0, λ₂=0.1, λ₃=0.5, λ₄=100.0, δ_s=1
"""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataclasses import dataclass, field
from collections import deque
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FACMACConfig:
    """SFAC-FACMAC configuration matching paper Appendix:Hyperparams."""
    m_instances: int = 4
    obs_dim: int = 7          # per-instance observation dimension
    action_dim: int = 3       # (reconfig, rotate, param)
    hidden_dim: int = 64
    buffer_size: int = 100_000
    batch_size: int = 256
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_trust: float = 3e-4    # trust estimator (same as actor/critic per Appendix:Hyperparams)
    gamma: float = 0.99
    tau_target: float = 0.005
    window_W: int = 50
    # Reward weights (Eq.14)
    lambda_1: float = 1.0     # throughput
    lambda_2: float = 0.1     # latency
    lambda_3: float = 0.5     # view-changes
    lambda_4: float = 100.0   # safety margin penalty [Paper Appendix:Hyperparams]
    delta_s: int = 1          # safety margin
    # PER
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_anneal_steps: int = 50_000
    # Safety filter
    use_safety_filter: bool = True
    # Peak tracker
    use_peak_tracker: bool = True
    peak_decay: float = 0.95            # per-epoch decay for peak tracker
    peak_fallback_alpha: float = 0.29   # fast EWMA alpha when peak tracker disabled
    # Cross-instance coordination
    use_cross_instance: bool = True
    cross_sharpening: float = 0.20  # sharpening factor for cross-instance deviation
    # Detection threshold: slightly above BFT fault tolerance bound (1/3)
    # to reduce false-positive detections from estimation noise
    detection_threshold: float = 0.356
    # Fast-path detection: raw signal spike threshold for instant detection
    # (bypasses EWMA smoothing delay for first-epoch attack response)
    fast_detect_threshold: float = 0.08
    # EWMA warm-start: initialize to honest noise floor to reduce convergence time
    ewma_init: float = 0.02
    # Reconfiguration
    use_reconfig: bool = True
    # Centralized critic (False = independent Q per agent)
    use_centralized_critic: bool = True
    # Monotonic mixer (True = QMIX-style, False = VDN sum)
    use_monotonic_mixer: bool = True
    # EWMA smoothing
    ewma_alpha: float = 0.10
    # Exploration noise
    noise_start: float = 0.2
    noise_end: float = 0.02
    noise_anneal_steps: int = 20_000
    # Organizational reward (Eq. reward-org): r_t^org = r_t + λ5·grg - λ6·rrg
    lambda_5: float = 0.5     # grg (goal achievement bonus) weight  [Paper Appendix:Hyperparams]
    lambda_6: float = 1.0     # rrg (role violation penalty) weight  [Paper Appendix:Hyperparams]
    # rrg thresholds (Appendix Eq.rrg/grg — values documented here for reproducibility)
    theta_high: float = 0.7   # fault-prob threshold for Sentinel rrg [Eq.rrg: θ_high]
    r_miss: float = 1.0       # Sentinel missed detection penalty
    r_false: float = 0.8      # Commander false eviction penalty
    r_churn: float = 0.5      # Tuner excessive param change penalty
    eta_stab: float = 0.3     # Tuner stability threshold
    # grg parameters
    r_b: float = 1.0          # grg base reward
    kappa_det: int = 10       # detection deadline (epochs)
    tau_evict: int = 20       # eviction deadline (epochs)
    # Cosine-annealing LR schedule (paper Appendix:Hyperparams: α_k = c/√K)
    lr_anneal_steps: int = 50_000
    # Role Action Guide (RAG) constraint hardness (Algorithm 1, Lines 9-12)
    # ch_ς ∈ [0,1]: 0 = fully soft (reward only), 1 = deterministic constraint
    ch_sentinel: float = 0.5   # Sentinel: force rotation when f̂>θ_high
    ch_commander: float = 0.5  # Commander: block eviction of likely-honest
    ch_tuner: float = 0.3      # Tuner: clamp param change to ±η_stab
    ch_guardian: float = 1.0   # Guardian: hard safety filter (always active)
    # Device: 'cuda' if available, else 'cpu'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


# ═══════════════════════════════════════════════════════════════════════════════
# Networks (matching paper Section III-D architecture)
# ═══════════════════════════════════════════════════════════════════════════════

class AgentActor(nn.Module):
    """Per-agent deterministic actor: μ_i(o_i; φ_i) → actions.

    Three action heads per agent (P5 role decomposition):
      - reconfig (sigmoid): Commander role — per-instance binary eviction probability
        Note: Paper §III-D says "softmax for reconfig" because the paper formulation
        is per-agent over |Ω_t^i| members. In simulation, we simplify to per-instance
        binary eviction (sigmoid), which is functionally equivalent for single-eviction.
      - rotate (sigmoid): Sentinel role — leader rotation probability
      - param (tanh): Tuner role — consensus parameter adjustment
    Guardian role is enforced by the safety filter (external).
    """

    def __init__(self, obs_dim: int = 7, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.reconfig_head = nn.Linear(hidden, 1)
        self.rotate_head = nn.Linear(hidden, 1)
        self.param_head = nn.Linear(hidden, 1)
        # Initialize conservatively: default = no action
        with torch.no_grad():
            self.reconfig_head.bias.fill_(-2.0)
            self.rotate_head.bias.fill_(-1.0)
            self.param_head.bias.fill_(0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Input: (..., obs_dim), Output: (..., 3)."""
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        reconfig = torch.sigmoid(self.reconfig_head(x))
        rotate = torch.sigmoid(self.rotate_head(x))
        param = torch.tanh(self.param_head(x))
        return torch.cat([reconfig, rotate, param], dim=-1)


class AgentQNetwork(nn.Module):
    """Per-agent utility: Q_i(o_i, a_i; θ_i)."""

    def __init__(self, obs_dim: int = 7, action_dim: int = 3, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class MonotonicMixer(nn.Module):
    """Monotonic mixing network: Q_tot = g_ψ(s, Q_1,...,Q_m).

    Ensures IGM consistency (Proposition: IGM-structural) via abs weights:
      ∂Q_tot/∂Q_i ≥ 0  →  ε_IGM = 0
    """

    def __init__(self, m_instances: int, state_dim: int, hidden: int = 64):
        super().__init__()
        self.m = m_instances
        self.hidden = hidden
        self.hyper_w1 = nn.Linear(state_dim, m_instances * hidden)
        self.hyper_b1 = nn.Linear(state_dim, hidden)
        self.hyper_w2 = nn.Linear(state_dim, hidden)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """q_values: (batch, m), state: (batch, state_dim) → (batch,)"""
        batch = q_values.shape[0]
        w1 = torch.abs(self.hyper_w1(state)).view(batch, self.m, self.hidden)
        b1 = self.hyper_b1(state).view(batch, 1, self.hidden)
        h = F.elu(torch.bmm(q_values.unsqueeze(1), w1) + b1)
        w2 = torch.abs(self.hyper_w2(state)).view(batch, self.hidden, 1)
        b2 = self.hyper_b2(state).view(batch, 1, 1)
        q_tot = torch.bmm(h, w2) + b2
        return q_tot.squeeze(-1).squeeze(-1)


class VDNMixer(nn.Module):
    """Value Decomposition Network (VDN) mixer: Q_tot = sum(Q_i).

    Used as ablation baseline when use_monotonic_mixer=False.
    No learned parameters — simple summation.
    """

    def __init__(self, m_instances: int, state_dim: int, hidden: int = 64):
        super().__init__()
        # Accept same args as MonotonicMixer for API compatibility
        self.m = m_instances

    def forward(self, q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """q_values: (batch, m), state: (batch, state_dim) → (batch,)"""
        return q_values.sum(dim=-1)


class TrustEstimator(nn.Module):
    """Linear trust estimator (Eq.6): f_hat_k = σ(w^T x_k + b).

    Jointly trained with actor/critic via binary cross-entropy on fault labels.
    Input: 5-dim feature vector x_k (Eq.5).

    Initialized with informed weights: signal features positively weighted so
    higher trust feature values → higher fault probability. Bias set negative
    so honest instances (low signals) default to low fault probability.
    """

    def __init__(self, feature_dim: int = 5):
        super().__init__()
        self.linear = nn.Linear(feature_dim, 1)
        # Informed init: signal dims (d/W, e/W, v/W) → positive weight
        # Latency dims (τ̄, σ_τ) → moderate positive weight
        with torch.no_grad():
            self.linear.weight.fill_(0.0)
            # d/W (timeout fraction): strong positive
            self.linear.weight[0, 0] = 3.0
            # e/W (equivocation rate): strong positive
            self.linear.weight[0, 1] = 2.5
            # v/W (view-change rate): moderate positive
            self.linear.weight[0, 2] = 2.0
            # τ̄ (mean latency): moderate positive
            self.linear.weight[0, 3] = 1.5
            # σ_τ (latency std): mild positive
            self.linear.weight[0, 4] = 1.0
            # Bias: negative so honest instances (all ~0) → low fault prob
            self.linear.bias.fill_(-1.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))


# ═══════════════════════════════════════════════════════════════════════════════
# Prioritized Experience Replay (Algorithm 1, Line 18)
# ═══════════════════════════════════════════════════════════════════════════════

class PrioritizedReplayBuffer:
    """PER with proportional prioritization (Schaul et al., 2016)."""

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: list = []
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

    def push(self, transition: dict):
        if self.size < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = self.max_priority ** self.alpha
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4):
        if self.size == 0:
            return None, None, None
        probs = self.priorities[:self.size]
        prob_sum = probs.sum()
        if prob_sum < 1e-12:
            probs = np.ones(self.size) / self.size
        else:
            probs = probs / prob_sum
        n_sample = min(batch_size, self.size)
        indices = np.random.choice(self.size, size=n_sample, p=probs, replace=True)
        samples = [self.buffer[i] for i in indices]
        weights = (self.size * probs[indices]) ** (-beta)
        weights = weights / (weights.max() + 1e-10)
        return samples, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        for idx, td in zip(indices, td_errors):
            self.priorities[idx] = (abs(td) + 1e-6) ** self.alpha
            self.max_priority = max(self.max_priority, self.priorities[idx])

    def __len__(self):
        return self.size


# ═══════════════════════════════════════════════════════════════════════════════
# SFAC-FACMAC Controller
# ═══════════════════════════════════════════════════════════════════════════════

class FACMACController:
    """Safe Factored Actor-Critic controller with FACMAC (paper-aligned).

    Implements Algorithm 1 (SafeMARL) with four phases:
      Phase 1 (Estimate): Build 5-dim features → trust estimation (Eq.5-6)
      Phase 2 (Select): Per-agent actors → role-decomposed actions
      Phase 3 (Filter): Pre-argmax safety mask (n >= 3f+1)
      Phase 4 (Update): PER → FACMAC training (Q_tot, actors, trust weights)

    Key properties (P1-P5):
      P1: Regret-aligned reward (Eq.14)
      P2: Cross-instance centralized critic (monotonic mixer)
      P3: Pre-argmax safety masking
      P4: Metadata-bounded trust estimation (5-dim features)
      P5: Byzantine-aware role decomposition (4 roles)
    """

    def __init__(self, cfg: FACMACConfig):
        self.cfg = cfg
        self.m = cfg.m_instances
        self.device = torch.device(cfg.device)

        # State dimension for mixer/critic
        # obs_dim * m (all agents' observations) + m (cross-instance features)
        self.state_dim = cfg.obs_dim * cfg.m_instances + cfg.m_instances

        # ── Networks ──
        # Seed PyTorch for deterministic network initialization.
        # This ensures the informed-weight actor behaves consistently
        # regardless of external RNG state.
        _rng_state = torch.random.get_rng_state()
        torch.manual_seed(2024)
        self.actor = AgentActor(cfg.obs_dim, cfg.hidden_dim).to(self.device)
        self.actor_target = AgentActor(cfg.obs_dim, cfg.hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.q_networks = nn.ModuleList([
            AgentQNetwork(cfg.obs_dim, cfg.action_dim, cfg.hidden_dim)
            for _ in range(cfg.m_instances)
        ]).to(self.device)
        self.q_targets = nn.ModuleList([
            AgentQNetwork(cfg.obs_dim, cfg.action_dim, cfg.hidden_dim)
            for _ in range(cfg.m_instances)
        ]).to(self.device)
        for i in range(cfg.m_instances):
            self.q_targets[i].load_state_dict(self.q_networks[i].state_dict())

        MixerClass = MonotonicMixer if cfg.use_monotonic_mixer else VDNMixer
        self.mixer = MixerClass(cfg.m_instances, self.state_dim, cfg.hidden_dim).to(self.device)
        self.mixer_target = MixerClass(
            cfg.m_instances, self.state_dim, cfg.hidden_dim).to(self.device)
        self.mixer_target.load_state_dict(self.mixer.state_dict())

        # Trust estimator (Eq.6): uses 5-dim subset of obs
        self.trust_estimator = TrustEstimator(feature_dim=5).to(self.device)
        torch.random.set_rng_state(_rng_state)  # restore external RNG

        # ── Optimizers ──
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.critic_optim = optim.Adam(
            list(self.q_networks.parameters()) + list(self.mixer.parameters()),
            lr=cfg.lr_critic)
        self.trust_optim = optim.Adam(
            self.trust_estimator.parameters(), lr=cfg.lr_trust)

        # ── Cosine-Annealing LR Schedulers (paper Appendix:Hyperparams) ──
        self.actor_scheduler = CosineAnnealingLR(
            self.actor_optim, T_max=cfg.lr_anneal_steps)
        self.critic_scheduler = CosineAnnealingLR(
            self.critic_optim, T_max=cfg.lr_anneal_steps)
        self.trust_scheduler = CosineAnnealingLR(
            self.trust_optim, T_max=cfg.lr_anneal_steps)

        # ── PER Buffer ──
        self.buffer = PrioritizedReplayBuffer(cfg.buffer_size, cfg.per_alpha)
        self.beta = cfg.per_beta_start
        self.train_step_count = 0

        # ── Trust estimation state (EWMA) ──
        # Warm-start to honest noise floor for faster convergence on attack onset
        self.ewma = np.full(cfg.m_instances, cfg.ewma_init)
        self.ewma_equiv = np.zeros(cfg.m_instances)  # separate equivocation EWMA
        self.variance = np.full(cfg.m_instances, 0.25)
        self.n_obs = np.zeros(cfg.m_instances)
        self.peak = np.zeros(cfg.m_instances)
        self.peak_fallback = np.zeros(cfg.m_instances)  # fast EWMA for no-peak mode
        self.prev_signal = np.zeros(cfg.m_instances)

        # ── Delayed transition storage ──
        self._prev_obs: Optional[np.ndarray] = None
        self._prev_actions: Optional[np.ndarray] = None
        self._prev_state: Optional[np.ndarray] = None
        self._prev_fault_labels: Optional[np.ndarray] = None
        self._prev_instance_sizes: Optional[np.ndarray] = None
        self._pending_reward: Optional[float] = None

        self._curr_obs: Optional[np.ndarray] = None
        self._curr_actions: Optional[np.ndarray] = None
        self._curr_state: Optional[np.ndarray] = None
        self._curr_fault_labels: Optional[np.ndarray] = None
        self._curr_instance_sizes: Optional[np.ndarray] = None

        # ── Organizational reward (grg/rrg) tracking ──
        self._prev_epoch_actions: Optional[np.ndarray] = None  # for Tuner stability
        self._detection_epoch = np.full(cfg.m_instances, -1, dtype=int)  # when fault first detected
        self._fault_onset_epoch = np.full(cfg.m_instances, -1, dtype=int)  # when fault started
        self._eviction_epoch = np.full(cfg.m_instances, -1, dtype=int)  # when eviction happened
        self._stable_count = np.zeros(cfg.m_instances, dtype=int)  # consecutive stable epochs
        self._epoch_counter: int = 0

        self.train_mode = True

    # ─── Phase 1: Estimate (Eq.5-6) ──────────────────────────────────────────

    def _update_trust_ewma(self, raw_signal: np.ndarray,
                          equiv_signal: np.ndarray | None = None):
        """EWMA trust estimation with adaptive alpha (sliding window smoothing).

        Uses asymmetric alpha: standard alpha for rising signals (attack onset),
        but 2× faster alpha for falling signals (attack departure / burst dormancy).
        This reduces false positives during dormant phases while maintaining
        fast attack detection.

        Maintains separate EWMA for equivocation signals (Eq.5: e/W channel).
        """
        base_alpha = self.cfg.ewma_alpha
        # Asymmetric alpha: faster decay when signal drops
        alpha = np.where(raw_signal >= self.ewma, base_alpha, min(base_alpha * 2.5, 0.35))
        self.ewma = (1 - alpha) * self.ewma + alpha * raw_signal
        self.n_obs += 1
        delta = raw_signal - self.ewma
        self.variance = (1 - base_alpha) * self.variance + base_alpha * delta ** 2
        # Update equivocation EWMA (independent channel)
        if equiv_signal is not None:
            alpha_e = np.where(equiv_signal >= self.ewma_equiv, base_alpha,
                               min(base_alpha * 2.5, 0.35))
            self.ewma_equiv = (1 - alpha_e) * self.ewma_equiv + alpha_e * equiv_signal

    def _get_uncertainty(self) -> np.ndarray:
        return np.sqrt(self.variance / np.maximum(self.n_obs, 1))

    def _build_obs(self, raw_signal: np.ndarray, epoch: int,
                   T_total: int) -> np.ndarray:
        """Build per-instance observations. Returns (m, obs_dim) array.

        Paper §III-D: "The actor network processes a 7-dimensional observation:
        the 5 trust features augmented with a cross-instance coordination signal
        and an epoch progress indicator."

        7-dim features per instance:
          0: d/W  — timeout fraction (Eq.5)
          1: e/W  — equivocation rate (Eq.5)
          2: v/W  — view-change rate (Eq.5)
          3: τ̄    — mean latency (Eq.5)
          4: σ_τ  — latency std dev (Eq.5)
          5: cross-instance coordination signal
          6: epoch progress (t/T)
        """
        unc = self._get_uncertainty()

        if self.cfg.use_peak_tracker:
            self.peak = np.maximum(self.peak * self.cfg.peak_decay, raw_signal)
            peak_feat = self.peak.copy()
        else:
            # Fast-tracking EWMA for v/W: responds quickly during active
            # phases for timely detection, but decays during dormant.
            # Peak tracking retains dormant persistence — the main benefit
            # this ablation removes.
            fast_alpha = self.cfg.peak_fallback_alpha
            self.peak_fallback = ((1 - fast_alpha) * self.peak_fallback
                                  + fast_alpha * raw_signal)
            peak_feat = self.peak_fallback.copy()

        # Eq.5 trust features (simulation mapping):
        #   Features use EWMA-smoothed signals to model the sliding
        #   window statistics in the Go implementation. This creates natural
        #   detection lag after adversary roaming switches.
        #   ewma(timeout)     → d/W (timeout fraction — smoothed)
        #   ewma(equivocation)→ e/W (equivocation rate — independent channel)
        #   peak              → v/W (view-change rate proxy via peak detection)
        #   0.1 + 0.5*ewma   → τ̄ (synthetic latency — smoothed)
        #   uncertainty       → σ_τ (latency variance proxy)
        d_W = np.clip(self.ewma, 0, 1)
        e_W = np.clip(self.ewma_equiv, 0, 1)
        v_W = np.clip(peak_feat, 0, 1)
        tau_mean = np.clip(0.1 + 0.5 * self.ewma, 0, 1)
        sigma_tau = np.clip(unc, 0, 1)

        if self.cfg.use_cross_instance:
            cross_mean = np.full(self.m, np.mean(raw_signal))
        else:
            cross_mean = np.zeros(self.m)

        epoch_progress = epoch / max(T_total, 1)

        obs = np.stack([
            d_W, e_W, v_W, tau_mean, sigma_tau,
            cross_mean, np.full(self.m, epoch_progress),
        ], axis=-1)  # (m, 7)

        return obs

    def _build_trust_features(self, obs: np.ndarray) -> np.ndarray:
        """Extract 5-dim trust features from 7-dim obs (Eq.5).

        Paper §III-D: actor input = [5 trust features, cross_signal, epoch_progress].
        Trust features are dims 0-4: (d/W, e/W, v/W, τ̄, σ_τ).
        """
        return np.clip(obs[:, :5], 0.0, 1.0)

    def _build_global_state(self, obs: np.ndarray,
                            raw_signal: np.ndarray) -> np.ndarray:
        """Build global state for mixer. Shape: (m*obs_dim + m,)."""
        cross_features = np.abs(raw_signal - np.mean(raw_signal))
        if not self.cfg.use_centralized_critic:
            cross_features = np.zeros(self.m)
        return np.concatenate([obs.ravel(), cross_features])

    def _get_fault_probs(self, obs: np.ndarray) -> np.ndarray:
        """Trust estimation (Eq.6): f_hat = σ(w·x + b)."""
        features = self._build_trust_features(obs)
        features_t = torch.tensor(features, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            probs = self.trust_estimator(features_t).squeeze(-1).cpu().numpy()
        return probs

    # ─── Phase 2: Select (per-agent actors) ──────────────────────────────────

    @torch.no_grad()
    def decide(self, raw_signal: np.ndarray, instance_sizes: np.ndarray,
               epoch: int, T_total: int,
               equiv_signal: np.ndarray | None = None) -> dict:
        """Phase 1-3: Estimate → Select → Filter.

        Args:
            raw_signal: per-instance timeout/aggregate fault signal (m,)
            instance_sizes: per-instance replica count (m,)
            epoch: current epoch index
            T_total: total evaluation epochs
            equiv_signal: optional per-instance equivocation signal (m,)
                If None, e/W channel uses a decorrelated lag of raw_signal.

        Returns dict compatible with OctopusMARLController expectations.
        """
        # ── Phase 1: Estimate ──
        self._update_trust_ewma(raw_signal, equiv_signal)
        obs = self._build_obs(raw_signal, epoch, T_total)
        global_state = self._build_global_state(obs, raw_signal)
        fault_probs = self._get_fault_probs(obs)

        # Cross-instance sharpening: when enabled, the global view helps
        # distinguish truly attacked instances from noisy ones by boosting
        # instances that deviate from the global mean (Proposition iii).
        #
        # Coordinated attack adaptation (§III-D): when the adversary attacks
        # ALL instances simultaneously, deviation-based sharpening becomes
        # ineffective (all deviations ≈ 0). We detect this by checking if
        # the variance of fault_probs across instances is low while the mean
        # is elevated — indicating uniform attack. In this case, we bypass
        # sharpening and instead apply a uniform boost to all instances,
        # improving detection under coordinated adversaries.
        if self.cfg.use_cross_instance:
            global_mean = np.mean(fault_probs)
            global_std = np.std(fault_probs)
            # Coordinated attack detector: low cross-instance variance + elevated mean
            coordinated_pattern = (global_std < 0.08) and (global_mean > 0.25)
            if coordinated_pattern:
                # Uniform boost: all instances are under attack, sharpen uniformly
                boost = self.cfg.cross_sharpening * global_mean
                fault_probs = np.clip(fault_probs + boost, 0, 1)
            else:
                # Normal mode: sharpen instances deviating above the mean
                deviation = fault_probs - global_mean
                fault_probs = np.clip(
                    fault_probs + self.cfg.cross_sharpening * np.clip(deviation, 0, None), 0, 1)

        # Safety filter: operates at the action level (_safety_filter method)
        # to prevent unsafe evictions. At the detection level, the safety
        # filter gates false positive detections during high-uncertainty
        # dormant phases — but only for the reconfig decision, not detection.
        # This is handled below in the detection logic.

        # ── Phase 2: Select (deterministic actor + exploration noise) ──
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        actions = self.actor(obs_t).cpu().numpy()  # (m, 3)
        # Cache raw actor output BEFORE noise/RAG/safety mask so reward-shaping
        # ablations (which affect actor weights only) can be evaluated against
        # a non-masked signal. Used by experiments/run_role_ablation.py.
        self._curr_pre_mask_actions = actions.copy()

        if self.train_mode:
            frac = min(1.0, self.train_step_count / self.cfg.noise_anneal_steps)
            noise_scale = self.cfg.noise_start + frac * (
                self.cfg.noise_end - self.cfg.noise_start)
            noise = np.random.normal(0, noise_scale, actions.shape)
            actions = np.clip(actions + noise, -1, 1)
            actions[:, :2] = np.clip(actions[:, :2], 0, 1)

        # Snapshot post-noise, pre-mask actions for replay buffer storage
        # so critic Q(s, a) reflects actor's real exploration policy rather
        # than mask-overridden values. Without this, RAG mask induces a
        # degenerate actor solution (raw=0, mask=0.8) and reward-shaping
        # ablations (grg/rrg/per) cannot differentiate.
        actions_pre_mask = actions.copy()

        # ── Phase 2b: Role Action Guides (Algorithm 1, Lines 9-12) ──
        actions = self._apply_rag(actions, fault_probs)

        # ── Phase 3: Filter (pre-argmax safety mask) ──
        actions = self._safety_filter(actions, fault_probs, instance_sizes)

        # Detection decision: trust estimator threshold near BFT fault tolerance
        # bound (f/n = 1/3 ≈ 0.333). Slightly above 1/3 to reduce false-positive
        # detections from estimation noise in the sigmoid trust output.
        detected = fault_probs > self.cfg.detection_threshold
        # Also consider actor's reconfig signal as detection
        for k in range(self.m):
            if actions[k, 0] > 0.5:
                detected[k] = True

        # Fast detection via raw equivocation channel (§III-B).
        # Equivocation (double-voting) is cryptographically verifiable and
        # never produced by honest nodes. When per-instance equivocation
        # exceeds the dormant noise floor (>= 2 equivocating nodes out of
        # ~25 per instance), this is strong Byzantine evidence that enables
        # detection before the EWMA trust estimate fully converges.
        if equiv_signal is not None:
            equivocation_fast = equiv_signal > 0.075
            detected = detected | equivocation_fast

        # Fast-path spike detection (§III-D): raw signal exceeding the fast
        # threshold triggers immediate detection without waiting for EWMA
        # convergence. This provides CUSUM-like first-epoch responsiveness
        # while the EWMA-based trust estimation handles long-term management.
        # The threshold is set below BFT fault tolerance (1/3) to catch
        # attacks with high recall, since the safety filter gates any
        # false-positive reconfiguration actions downstream.
        spike_detected = raw_signal > self.cfg.fast_detect_threshold
        detected = detected | spike_detected

        # Reconfig decision with safety gating
        reconfig_target = -1
        if self.cfg.use_reconfig and np.any(detected):
            if self.cfg.use_safety_filter:
                # Safety filter: require higher confidence for reconfig to avoid
                # wasting throughput on false positive reconfigs (Theorem 5)
                unc = self._get_uncertainty()
                confident = detected & (unc < 0.10)
                if np.any(confident):
                    reconfig_target = int(np.argmax(fault_probs * confident))
            else:
                # Without safety: reconfig on any detection (may cause FP reconfigs)
                reconfig_target = int(np.argmax(fault_probs))

        # ── Cache for delayed transition storage ──
        # Shift: prev ← curr, curr ← new
        self._prev_obs = self._curr_obs
        self._prev_actions = self._curr_actions
        self._prev_state = self._curr_state
        self._prev_fault_labels = self._curr_fault_labels
        self._prev_instance_sizes = self._curr_instance_sizes

        self._curr_obs = obs.copy()
        self._curr_actions = actions_pre_mask.copy()
        self._curr_post_mask_actions = actions.copy()  # for eval metric only
        self._curr_state = global_state.copy()
        # Fault labels: 1 if signal > threshold (for trust estimator training)
        self._curr_fault_labels = (raw_signal > 0.3).astype(np.float32)
        self._curr_instance_sizes = instance_sizes.copy()

        self.prev_signal = raw_signal.copy()
        self._epoch_counter += 1

        return {
            "detected_instances": detected,
            "reconfig_target": reconfig_target,
            "leader_rotation": detected.tolist(),
            "detection_probs": fault_probs,
            "safety_masked": False,
            "combined_scores": fault_probs,
        }

    # ─── Role Action Guides (Algorithm 1, Lines 9-12) ──────────────────────

    def _apply_rag(self, actions: np.ndarray,
                   fault_probs: np.ndarray) -> np.ndarray:
        """Apply role action guides with stochastic constraint hardness.

        For each role ς ∈ {Sentinel, Commander, Tuner, Guardian}:
          with probability ch_ς, constrain a_t^{i,ς} via rag_ς.
        Guardian (ch=1.0) is handled separately by _safety_filter().
        """
        cfg = self.cfg
        for k in range(self.m):
            # Sentinel (→ rotate): if f̂ > θ_high, guide toward rotation
            if np.random.random() < cfg.ch_sentinel:
                if fault_probs[k] > cfg.theta_high:
                    actions[k, 1] = max(actions[k, 1], 0.8)  # push toward rotate

            # Commander (→ reconfig): block eviction of likely-honest
            if np.random.random() < cfg.ch_commander:
                if actions[k, 0] > 0.5 and fault_probs[k] < 0.3:
                    actions[k, 0] = 0.0  # block eviction of low-risk node

            # Tuner (→ param): clamp to ±η_stab if guide fires
            if np.random.random() < cfg.ch_tuner:
                if self._prev_epoch_actions is not None:
                    prev_param = self._prev_epoch_actions[k, 2]
                    actions[k, 2] = np.clip(
                        actions[k, 2],
                        prev_param - cfg.eta_stab,
                        prev_param + cfg.eta_stab)

        return actions

    # ─── Phase 3: Filter ─────────────────────────────────────────────────────

    def _safety_filter(self, actions: np.ndarray, fault_probs: np.ndarray,
                       instance_sizes: np.ndarray) -> np.ndarray:
        """Pre-argmax quorum safety filter (Algorithm 1, Lines 13-17).

        Paper: block eviction if n_after < 3*f_v^i + 1 + δ_s.
        When use_protocol_f_bound is True, f_v^i = floor((n_i - 1) / 3) as the
        protocol-level maximum tolerable faults (matching the paper's deterministic
        bound). Otherwise, f_v^i is estimated from trust scores as ceil(f_hat * n).
        """
        if not self.cfg.use_safety_filter:
            return actions

        delta_s = self.cfg.delta_s
        for k in range(self.m):
            if actions[k, 0] > 0.5:  # eviction proposed
                n_after = instance_sizes[k] - 1
                # Protocol-level fault bound: f_v = ⌊(n-1)/3⌋
                f_v = int((instance_sizes[k] - 1) // 3)
                if n_after < 3 * f_v + 1 + delta_s:
                    actions[k, 0] = 0.0  # mask unsafe eviction

        return actions

    # ─── Reward Computation (Eq.14, P1) ───────────────────────────────────────

    def _compute_eq14_reward(self, obs: np.ndarray, actions: np.ndarray,
                             instance_sizes: np.ndarray,
                             fault_probs: np.ndarray) -> float:
        """Compute regret-aligned reward (Eq.14).

        r_t = λ₁·Σ tp_i - λ₂·Σ ℓ_i - λ₃·Σ vc_i - λ₄·Σ max(0, 3f+1+δ-n_i)

        Mapping from observations:
          tp_i: throughput proxy = normalized instance size (larger → higher tp)
          ℓ_i: latency proxy = obs[k,0] (raw signal, higher faults → higher latency)
          vc_i: view-change proxy = actions[k,1] (rotation action taken)
          margin: quorum shortfall from fault_probs and instance_sizes
        """
        cfg = self.cfg
        tp_sum = 0.0
        latency_sum = 0.0
        vc_sum = 0.0
        margin_penalty = 0.0

        for k in range(self.m):
            # Throughput: base throughput decreases with fault level
            tp_sum += max(0.0, 1.0 - obs[k, 0] * 0.3) * (
                instance_sizes[k] / 25.0)  # normalized by instance size

            # Latency: increases with fault signal
            latency_sum += obs[k, 0] * 0.5

            # View-changes: triggered by rotation action
            vc_sum += float(actions[k, 1] > 0.5)

            # Safety margin penalty: max(0, 3f+1+δ-n)
            f_count = fault_probs[k] * instance_sizes[k]
            shortfall = 3 * f_count + 1 + cfg.delta_s - instance_sizes[k]
            margin_penalty += max(0.0, shortfall)

        reward = (cfg.lambda_1 * tp_sum
                  - cfg.lambda_2 * latency_sum
                  - cfg.lambda_3 * vc_sum
                  - cfg.lambda_4 * margin_penalty)

        return reward

    def _compute_org_reward(self, obs: np.ndarray, actions: np.ndarray,
                            instance_sizes: np.ndarray,
                            fault_probs: np.ndarray,
                            fault_labels: np.ndarray) -> float:
        """Compute organizational reward (Eq. reward-org).

        r_t^org = r_t + λ₅·Σ grg - λ₆·Σ rrg

        rrg (role violation penalties):
          Sentinel: high f_hat but no eviction → -r_miss
          Commander: eviction of non-Byzantine → -r_false
          Tuner: excessive param change → -r_churn

        grg (goal achievement bonuses):
          g_det: fast detection bonus
          g_evict: fast eviction bonus
          g_stab: stability window bonus
        """
        cfg = self.cfg

        # Base reward
        base_r = self._compute_eq14_reward(obs, actions, instance_sizes,
                                           fault_probs)

        # ── rrg: role violation penalties ──
        rrg_sum = 0.0
        for k in range(self.m):
            # Sentinel rrg: high fault probability but no eviction proposed
            if fault_probs[k] > cfg.theta_high and actions[k, 0] <= 0.5:
                rrg_sum += cfg.r_miss

            # Commander rrg: eviction proposed for honest node
            if actions[k, 0] > 0.5 and fault_labels[k] < 0.5:
                rrg_sum += cfg.r_false

            # Tuner rrg: excessive parameter change
            if self._prev_epoch_actions is not None:
                param_change = abs(actions[k, 2] - self._prev_epoch_actions[k, 2])
                if param_change > cfg.eta_stab:
                    rrg_sum += cfg.r_churn

        # ── grg: goal achievement bonuses ──
        grg_sum = 0.0
        for k in range(self.m):
            is_faulty = fault_labels[k] > 0.5

            # Track fault onset
            if is_faulty and self._fault_onset_epoch[k] < 0:
                self._fault_onset_epoch[k] = self._epoch_counter

            # Track detection
            if is_faulty and fault_probs[k] > 0.5 and self._detection_epoch[k] < 0:
                self._detection_epoch[k] = self._epoch_counter

            # Track eviction
            if actions[k, 0] > 0.5 and self._eviction_epoch[k] < 0:
                self._eviction_epoch[k] = self._epoch_counter

            # g_det: detection speed bonus
            if (self._detection_epoch[k] >= 0 and
                    self._fault_onset_epoch[k] >= 0):
                dt_det = self._detection_epoch[k] - self._fault_onset_epoch[k]
                if dt_det <= cfg.kappa_det:
                    grg_sum += cfg.r_b * (cfg.kappa_det - dt_det) / cfg.kappa_det

            # g_evict: eviction speed bonus
            if (self._eviction_epoch[k] >= 0 and
                    self._fault_onset_epoch[k] >= 0):
                dt_evict = self._eviction_epoch[k] - self._fault_onset_epoch[k]
                if dt_evict <= cfg.tau_evict:
                    grg_sum += cfg.r_b * (cfg.tau_evict - dt_evict) / cfg.tau_evict

            # g_stab: stability bonus (consecutive low-change epochs)
            if self._prev_epoch_actions is not None:
                param_change = abs(actions[k, 2] - self._prev_epoch_actions[k, 2])
                if param_change <= cfg.eta_stab:
                    self._stable_count[k] += 1
                else:
                    self._stable_count[k] = 0
                grg_sum += cfg.r_b * self._stable_count[k] / cfg.window_W

        # Update tracking
        self._prev_epoch_actions = actions.copy()

        # Eq. reward-org
        r_org = base_r + cfg.lambda_5 * grg_sum - cfg.lambda_6 * rrg_sum
        return r_org

    # ─── Transition Storage (delayed by 1 step) ──────────────────────────────

    def store_transition(self, per_inst_rewards: np.ndarray, done: bool = False):
        """Store transition using delayed pattern.

        Called after update() in OctopusMARLController. At this point we have:
          - _prev_obs/actions/state from decide(t-1)
          - _curr_obs from decide(t) as next_obs
          - per_inst_rewards from environment (used as auxiliary signal)

        Internal reward is computed via Eq.(reward-org) from cached observations.
        """
        if self._prev_obs is None or self._curr_obs is None:
            return

        # Compute Eq.(reward-org) = r_t + λ₅·grg - λ₆·rrg
        prev_fault_probs = self._get_fault_probs(self._prev_obs)
        prev_fault_labels = (self._prev_fault_labels
                             if self._prev_fault_labels is not None
                             else np.zeros(self.m))
        org_reward = self._compute_org_reward(
            self._prev_obs, self._prev_actions,
            self._prev_instance_sizes, prev_fault_probs, prev_fault_labels)

        # Blend: primarily use Eq.(reward-org), with small external signal
        external_reward = float(np.mean(per_inst_rewards))
        reward = 0.3 * org_reward + 0.7 * external_reward

        self.buffer.push({
            "obs": self._prev_obs.copy(),
            "actions": self._prev_actions.copy(),
            "reward": reward,
            "next_obs": self._curr_obs.copy(),
            "state": self._prev_state.copy(),
            "next_state": self._curr_state.copy(),
            "done": done,
            "fault_labels": self._prev_fault_labels.copy()
            if self._prev_fault_labels is not None else None,
        })

        if done:
            # Also store current step with zero next (terminal)
            curr_fault_probs = self._get_fault_probs(self._curr_obs)
            curr_fault_labels = (self._curr_fault_labels
                                 if self._curr_fault_labels is not None
                                 else np.zeros(self.m))
            curr_reward = self._compute_org_reward(
                self._curr_obs, self._curr_actions,
                self._curr_instance_sizes, curr_fault_probs, curr_fault_labels)
            self.buffer.push({
                "obs": self._curr_obs.copy(),
                "actions": self._curr_actions.copy(),
                "reward": 0.3 * curr_reward + 0.7 * external_reward,
                "next_obs": np.zeros_like(self._curr_obs),
                "state": self._curr_state.copy(),
                "next_state": np.zeros_like(self._curr_state),
                "done": True,
                "fault_labels": self._curr_fault_labels.copy()
                if self._curr_fault_labels is not None else None,
            })

    # ─── Phase 4: Update (FACMAC + PER) ──────────────────────────────────────

    def train_step(self) -> dict:
        """Algorithm 1, Line 18: update Q_tot, {μ_i}, (w,b) via PER."""
        cfg = self.cfg
        if len(self.buffer) < cfg.batch_size:
            return {}

        self.train_step_count += 1
        # Anneal PER β
        frac = min(1.0, self.train_step_count / cfg.per_beta_anneal_steps)
        self.beta = cfg.per_beta_start + frac * (
            cfg.per_beta_end - cfg.per_beta_start)

        samples, indices, is_weights = self.buffer.sample(
            cfg.batch_size, self.beta)
        if samples is None:
            return {}

        # Parse batch — move to device (GPU if available)
        dev = self.device
        obs_batch = torch.tensor(
            np.array([s["obs"] for s in samples]), dtype=torch.float32, device=dev)
        act_batch = torch.tensor(
            np.array([s["actions"] for s in samples]), dtype=torch.float32, device=dev)
        rew_batch = torch.tensor(
            np.array([s["reward"] for s in samples]), dtype=torch.float32, device=dev)
        next_obs_batch = torch.tensor(
            np.array([s["next_obs"] for s in samples]), dtype=torch.float32, device=dev)
        state_batch = torch.tensor(
            np.array([s["state"] for s in samples]), dtype=torch.float32, device=dev)
        next_state_batch = torch.tensor(
            np.array([s["next_state"] for s in samples]), dtype=torch.float32, device=dev)
        done_batch = torch.tensor(
            np.array([s["done"] for s in samples]), dtype=torch.float32, device=dev)
        is_weights = is_weights.to(dev)

        # ── Critic update: Q_tot via monotonic mixer ──
        with torch.no_grad():
            next_actions = torch.stack([
                self.actor_target(next_obs_batch[:, i, :])
                for i in range(self.m)
            ], dim=1)  # (batch, m, action_dim)
            target_qs = torch.stack([
                self.q_targets[i](
                    next_obs_batch[:, i, :], next_actions[:, i, :]
                ).squeeze(-1)
                for i in range(self.m)
            ], dim=1)  # (batch, m)
            q_tot_target = self.mixer_target(target_qs, next_state_batch)
            y = rew_batch + cfg.gamma * (1 - done_batch) * q_tot_target

        current_qs = torch.stack([
            self.q_networks[i](
                obs_batch[:, i, :], act_batch[:, i, :]
            ).squeeze(-1)
            for i in range(self.m)
        ], dim=1)  # (batch, m)
        q_tot = self.mixer(current_qs, state_batch)

        # PER priority update
        td_errors = (y - q_tot).detach().cpu().numpy()
        self.buffer.update_priorities(indices, td_errors)

        # Weighted critic loss
        critic_loss = (is_weights * (y - q_tot) ** 2).mean()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.q_networks.parameters()) + list(self.mixer.parameters()),
            0.5)
        self.critic_optim.step()

        # ── Actor update: deterministic policy gradient (Eq.9) ──
        actions_pred = torch.stack([
            self.actor(obs_batch[:, i, :]) for i in range(self.m)
        ], dim=1)
        q_values = torch.stack([
            self.q_networks[i](
                obs_batch[:, i, :], actions_pred[:, i, :]
            ).squeeze(-1)
            for i in range(self.m)
        ], dim=1)
        q_tot_actor = self.mixer(q_values, state_batch)
        actor_loss = -q_tot_actor.mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optim.step()

        # ── Trust estimator update (Eq.6 joint training) ──
        trust_loss_val = 0.0
        fault_labels_list = [s.get("fault_labels") for s in samples]
        if fault_labels_list[0] is not None:
            labels = torch.tensor(
                np.array([fl for fl in fault_labels_list]),
                dtype=torch.float32, device=dev).view(-1)  # flatten (batch, m) → (batch*m,)
            # Extract 5-dim features for trust estimation
            trust_features = self._batch_trust_features(obs_batch)
            pred_faults = self.trust_estimator(trust_features).squeeze(-1)
            trust_loss = F.binary_cross_entropy(pred_faults, labels)

            self.trust_optim.zero_grad()
            trust_loss.backward()
            nn.utils.clip_grad_norm_(self.trust_estimator.parameters(), 1.0)
            self.trust_optim.step()
            trust_loss_val = trust_loss.item()

        # ── Soft update target networks ──
        self._soft_update(self.actor, self.actor_target, cfg.tau_target)
        for i in range(self.m):
            self._soft_update(self.q_networks[i], self.q_targets[i], cfg.tau_target)
        self._soft_update(self.mixer, self.mixer_target, cfg.tau_target)

        # ── Step LR schedulers (cosine-annealing, Appendix:Hyperparams) ──
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        self.trust_scheduler.step()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "trust_loss": trust_loss_val,
        }

    def _batch_trust_features(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """Extract 5-dim trust features from 7-dim obs batch.

        obs_batch: (batch, m, 7) → output: (batch*m, 5)
        Trust features are dims 0-4: (d/W, e/W, v/W, τ̄, σ_τ).
        """
        features = obs_batch[:, :, :5]  # (batch, m, 5)
        return features.reshape(-1, 5)

    @staticmethod
    def _soft_update(source: nn.Module, target: nn.Module, tau: float):
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)

    # ─── Interface methods ────────────────────────────────────────────────────

    def reset(self):
        """Reset per-episode state (keeps network weights)."""
        self.ewma[:] = self.cfg.ewma_init
        self.variance[:] = 0.25
        self.n_obs[:] = 0
        self.peak[:] = 0
        self.peak_fallback[:] = 0
        self.prev_signal[:] = 0
        self._prev_obs = None
        self._prev_actions = None
        self._prev_state = None
        self._prev_fault_labels = None
        self._prev_instance_sizes = None
        self._curr_obs = None
        self._curr_actions = None
        self._curr_state = None
        self._curr_fault_labels = None
        self._curr_instance_sizes = None
        self._pending_reward = None
        # Reset grg/rrg tracking
        self._prev_epoch_actions = None
        self._detection_epoch[:] = -1
        self._fault_onset_epoch[:] = -1
        self._eviction_epoch[:] = -1
        self._stable_count[:] = 0
        self._epoch_counter = 0

    def set_eval(self):
        self.train_mode = False
        self.actor.eval()

    def set_train(self):
        self.train_mode = True
        self.actor.train()

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "q_networks": self.q_networks.state_dict(),
            "mixer": self.mixer.state_dict(),
            "trust_estimator": self.trust_estimator.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "q_targets": self.q_targets.state_dict(),
            "mixer_target": self.mixer_target.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, weights_only=True, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.q_networks.load_state_dict(ckpt["q_networks"])
        self.mixer.load_state_dict(ckpt["mixer"])
        self.trust_estimator.load_state_dict(ckpt["trust_estimator"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.q_targets.load_state_dict(ckpt["q_targets"])
        self.mixer_target.load_state_dict(ckpt["mixer_target"])
