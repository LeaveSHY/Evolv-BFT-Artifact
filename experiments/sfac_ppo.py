"""Safe Factored Actor-Critic (SFAC) with MAPPO for Evolv-BFT.

Implements the CTDE (Centralized Training, Decentralized Execution) architecture
described in Section III-D of the paper. Uses MAPPO (Multi-Agent PPO) where each
BFT instance is treated as a separate agent with shared actor parameters:
  - Per-instance actors (shared weights): local observation -> detection decision
  - Centralized critic: global state -> per-agent value estimate
  - Safety filter: deterministic mask enforcing n_j >= 3f_j+1
  - Peak tracker: persistent memory of attack signal history
  - Cross-instance module: global mean + deviation features

The key design: each (instance, timestep) pair is an independent training sample
with its own per-instance reward and advantage, solving the credit assignment
problem that causes single-reward PPO to fail on multi-instance detection.

Ablation variants disable exactly one module each.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataclasses import dataclass
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Network Architecture
# ═══════════════════════════════════════════════════════════════════════════════

class ActorNetwork(nn.Module):
    """Per-instance stochastic actor with 3 action heads (P5 role decomposition).

    Three action heads per agent:
      - reconfig (sigmoid → Bernoulli): Commander role — eviction probability
        Note: Paper §III-D says "softmax for reconfig" over |Ω_t^i| members.
        In simulation, simplified to per-instance binary eviction (sigmoid).
      - rotate (sigmoid → Bernoulli): Sentinel role — leader rotation probability
      - param (tanh → Gaussian): Tuner role — consensus parameter adjustment
    Guardian role is enforced by the safety filter (external).

    Input (obs_dim=7, per paper §III-D):
      [d/W, e/W, v/W, τ̄, σ_τ, cross_instance_signal, epoch_progress]
      i.e. 5 trust features (Eq.5) + cross-instance coordination + epoch progress.
    Output: (reconfig_logit, rotate_logit, param_mu) — 3 values
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
        """Input: (..., obs_dim), Output: (..., 3) — [reconfig, rotate, param]."""
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        reconfig = torch.sigmoid(self.reconfig_head(x))
        rotate = torch.sigmoid(self.rotate_head(x))
        param = torch.tanh(self.param_head(x))
        return torch.cat([reconfig, rotate, param], dim=-1)


class MonotonicMixer(nn.Module):
    """FACMAC monotonic mixing network (paper Eq.eq:critic, Prop.prop:igm-structural).

    Implements Q_tot = g_psi(s, [Q_1, ..., Q_m]) with structural guarantee
    d Q_tot / d Q_i >= 0 (monotonicity), which is sufficient for the IGM
    condition (Proposition 7) and ensures consistent decentralized argmax.

    Monotonicity is enforced via |W| on hypernet-generated mixer weights,
    following QMIX (Rashid+ 2020) and FACMAC (Peng+ 2021).
    """

    def __init__(self, state_dim: int, m: int,
                 mixer_hidden: int = 32, hypernet_hidden: int = 64):
        super().__init__()
        self.m = m
        self.mixer_hidden = mixer_hidden
        # Hypernets generate mixer weights conditioned on global state s
        self.hyper_w1 = nn.Linear(state_dim, m * mixer_hidden)
        self.hyper_b1 = nn.Linear(state_dim, mixer_hidden)
        self.hyper_w2 = nn.Linear(state_dim, mixer_hidden)
        # Output bias from a 2-layer hypernet (per QMIX paper)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden),
            nn.ReLU(),
            nn.Linear(hypernet_hidden, 1),
        )

    def forward(self, agent_qs: torch.Tensor,
                state: torch.Tensor) -> torch.Tensor:
        """Mix per-agent Q values into Q_tot.

        Args:
            agent_qs: (B, m) per-agent Q (or V) values.
            state:    (B, state_dim) global state for hypernet conditioning.
        Returns:
            (B,) scalar Q_tot per batch element.
        """
        B = state.size(0)
        # First mixer layer (monotonic via abs)
        w1 = torch.abs(self.hyper_w1(state)).view(B, self.m, self.mixer_hidden)
        b1 = self.hyper_b1(state).view(B, 1, self.mixer_hidden)
        agent_qs_exp = agent_qs.view(B, 1, self.m)
        hidden = F.elu(torch.bmm(agent_qs_exp, w1) + b1)  # (B,1,mixer_hidden)
        # Second mixer layer (monotonic via abs)
        w2 = torch.abs(self.hyper_w2(state)).view(B, self.mixer_hidden, 1)
        b2 = self.hyper_b2(state).view(B, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2  # (B,1,1)
        return q_tot.view(B)


class CriticNetwork(nn.Module):
    """FACMAC factored critic with monotonic mixer (paper §III-D, Eq.eq:critic).

    Architecture (matches paper Prop.prop:igm-structural):
      1. Decompose global state into m per-agent observation slices
         (state layout: [m*obs_dim per-agent obs, m cross-features]).
      2. Per-agent Q heads: Q_i = q_head_i([o_i, cross_i])  for i in 1..m
      3. Monotonic mixer: Q_tot = g_psi(s, [Q_1, ..., Q_m]) with d/dQ_i >= 0.

    This satisfies the IGM condition by structure (epsilon_IGM = 0 in Prop.7).

    Stage G1 note: per-agent heads currently consume only o_i (no a_i) so the
    interface stays V(s), keeping the existing MAPPO training loop intact.
    Stage G2 will extend heads to Q_i(o_i, a_i) and switch the loss to DDPG.
    """

    def __init__(self, state_dim: int, hidden: int = 64,
                 m: int = 4, obs_dim: int = 7):
        super().__init__()
        self.m = m
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.head_in_dim = obs_dim + 1  # per-agent obs + own cross-feature
        # Per-agent Q heads (parameter-shared optional; kept independent here
        # to allow per-instance specialisation under coordinated attacks)
        self.agent_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.head_in_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1),
            ) for _ in range(m)
        ])
        # Monotonic mixer (paper Prop.7)
        self.mixer = MonotonicMixer(
            state_dim=state_dim, m=m,
            mixer_hidden=32, hypernet_hidden=hidden,
        )

    def _split_state(self, state_flat: torch.Tensor) -> torch.Tensor:
        """Build per-agent head inputs of shape (B, m, obs_dim+1)."""
        B = state_flat.size(0)
        per_agent_obs = state_flat[:, : self.m * self.obs_dim].view(
            B, self.m, self.obs_dim)
        cross = state_flat[:, self.m * self.obs_dim:].view(B, self.m, 1)
        return torch.cat([per_agent_obs, cross], dim=-1)  # (B,m,obs_dim+1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns Q_tot scalar per batch element (preserves MAPPO V(s) API)."""
        original_shape = state.shape[:-1]
        state_flat = state.reshape(-1, state.size(-1))
        B = state_flat.size(0)
        head_inputs = self._split_state(state_flat)  # (B,m,obs_dim+1)
        # Per-agent Q (V) heads
        per_agent_qs = torch.cat([
            self.agent_heads[i](head_inputs[:, i]) for i in range(self.m)
        ], dim=-1)  # (B,m)
        # Monotonic mixer (Q_tot)
        q_tot = self.mixer(per_agent_qs, state_flat)  # (B,)
        return q_tot.view(*original_shape)

    def per_agent_qs(self, state: torch.Tensor) -> torch.Tensor:
        """Expose per-agent Q vector for diagnostics / G2 DDPG loss."""
        state_flat = state.reshape(-1, state.size(-1))
        head_inputs = self._split_state(state_flat)
        return torch.cat([
            self.agent_heads[i](head_inputs[:, i]) for i in range(self.m)
        ], dim=-1)


class TrustEstimator(nn.Module):
    """Linear trust estimator (Eq.6): f_hat_k = σ(w^T x_k + b).

    Jointly trained with actor/critic via binary cross-entropy on fault labels.
    Input: 5-dim feature vector x_k (Eq.5).
    """

    def __init__(self, feature_dim: int = 5):
        super().__init__()
        self.linear = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))


# ═══════════════════════════════════════════════════════════════════════════════
# SFAC Controller with MAPPO
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SFACConfig:
    m_instances: int = 4
    obs_dim: int = 7          # per-instance observation dimension
    action_dim: int = 3       # (reconfig, rotate, param) — P5 role decomposition
    hidden_actor: int = 64    # paper Appendix:Hyperparams h=64
    hidden_critic: int = 64
    lr_actor: float = 3e-4    # paper Appendix:Hyperparams
    lr_critic: float = 3e-4   # paper Appendix:Hyperparams
    lr_trust: float = 3e-4    # trust estimator (same LR per Appendix)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 3
    batch_size: int = 256
    ewma_alpha: float = 0.10
    window_W: int = 50        # trust feature sliding window (Eq.5)
    # Reward weights (Eq.14)
    lambda_1: float = 1.0     # throughput
    lambda_2: float = 0.1     # latency
    lambda_3: float = 0.5     # view-changes
    lambda_4: float = 100.0   # safety margin penalty
    delta_s: int = 1          # safety margin
    # Organizational reward (Eq.reward-org): r^org = r + λ₅·grg - λ₆·rrg
    lambda_5: float = 0.5     # grg weight  [Paper Appendix:Hyperparams]
    lambda_6: float = 1.0     # rrg weight  [Paper Appendix:Hyperparams]
    # rrg thresholds
    theta_high: float = 0.7   # fault-prob threshold for Sentinel rrg
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
    ch_sentinel: float = 0.5   # Sentinel: force rotation when f̂>θ_high
    ch_commander: float = 0.5  # Commander: block eviction of likely-honest
    ch_tuner: float = 0.3      # Tuner: clamp param change to ±η_stab
    ch_guardian: float = 1.0   # Guardian: hard safety filter
    # Ablation flags
    use_safety_filter: bool = True
    use_peak_tracker: bool = True
    use_cross_instance: bool = True
    use_reconfig: bool = True
    use_centralized_critic: bool = True  # False = independent


class SFACController:
    """Safe Factored Actor-Critic controller with MAPPO training.

    MAPPO design: each (instance, timestep) pair produces a separate training
    sample with its own per-instance reward and advantage. The actor has shared
    weights across instances (parameter sharing). The critic is centralized
    (sees global state) during training.

    This solves the credit assignment problem: the gradient for instance j's
    action depends only on instance j's reward, not the aggregate.
    """

    def __init__(self, cfg: SFACConfig):
        self.cfg = cfg
        self.m = cfg.m_instances
        self.obs_dim = cfg.obs_dim

        # Critic state dim: always full size (m*obs_dim + m)
        self.state_dim = self.m * self.obs_dim + self.m

        # Networks
        self.actor = ActorNetwork(self.obs_dim, cfg.hidden_actor)
        # FACMAC factored critic with monotonic mixer (paper Eq.eq:critic + Prop.7)
        self.critic = CriticNetwork(
            state_dim=self.state_dim,
            hidden=cfg.hidden_critic,
            m=self.m,
            obs_dim=self.obs_dim,
        )
        self.trust_estimator = TrustEstimator(feature_dim=5)

        # Optimizers
        actor_params = list(self.actor.parameters())
        critic_params = list(self.critic.parameters())
        trust_params = list(self.trust_estimator.parameters())
        self.actor_optim = torch.optim.Adam(actor_params, lr=cfg.lr_actor)
        self.critic_optim = torch.optim.Adam(critic_params, lr=cfg.lr_critic)
        self.trust_optim = torch.optim.Adam(trust_params, lr=cfg.lr_trust)

        # ── Cosine-Annealing LR Schedulers (paper Appendix:Hyperparams) ──
        self.actor_scheduler = CosineAnnealingLR(
            self.actor_optim, T_max=cfg.lr_anneal_steps)
        self.critic_scheduler = CosineAnnealingLR(
            self.critic_optim, T_max=cfg.lr_anneal_steps)
        self.trust_scheduler = CosineAnnealingLR(
            self.trust_optim, T_max=cfg.lr_anneal_steps)

        # Trust estimation state
        self.ewma = np.zeros(self.m)
        self.variance = np.full(self.m, 0.25)
        self.n_obs = np.zeros(self.m)
        self.peak = np.zeros(self.m)
        self.prev_signal = np.zeros(self.m)

        # Organizational reward tracking (grg/rrg)
        self._fault_onset_epoch = np.full(self.m, -1, dtype=int)
        self._detection_epoch = np.full(self.m, -1, dtype=int)
        self._eviction_epoch = np.full(self.m, -1, dtype=int)
        self._stable_count = np.zeros(self.m)
        self._prev_epoch_actions: Optional[np.ndarray] = None
        self._epoch_counter = 0

        # MAPPO rollout buffer: stores per-instance transitions
        self.buffer = MAPPORolloutBuffer()

        # Training state
        self.train_mode = True

    def reset(self):
        """Reset per-episode state (keeps network weights)."""
        self.ewma[:] = 0
        self.variance[:] = 0.25
        self.n_obs[:] = 0
        self.peak[:] = 0
        self.prev_signal[:] = 0
        # Reset org-reward tracking
        self._fault_onset_epoch[:] = -1
        self._detection_epoch[:] = -1
        self._eviction_epoch[:] = -1
        self._stable_count[:] = 0
        self._prev_epoch_actions = None
        self._epoch_counter = 0

    def _update_trust(self, raw_signal: np.ndarray):
        """EWMA trust estimation with asymmetric alpha.

        Uses faster decay when signal drops to reduce false positives
        during dormant phases (burst/coordinated attacks).
        """
        base_alpha = self.cfg.ewma_alpha
        alpha = np.where(raw_signal >= self.ewma, base_alpha, min(base_alpha * 2.5, 0.35))
        self.ewma = (1 - alpha) * self.ewma + alpha * raw_signal
        self.n_obs += 1
        delta = raw_signal - self.ewma
        self.variance = (1 - base_alpha) * self.variance + base_alpha * delta ** 2

    def _get_uncertainty(self) -> np.ndarray:
        return np.sqrt(self.variance / np.maximum(self.n_obs, 1))

    def _build_obs(self, raw_signal: np.ndarray, epoch: int,
                   T_total: int) -> np.ndarray:
        """Build per-instance observations. Returns (m, obs_dim) array.

        Paper §III-D: "The actor network processes a 7-dimensional observation:
        the 5 trust features augmented with a cross-instance coordination signal
        and an epoch progress indicator."

        Features per instance:
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
            self.peak = np.maximum(self.peak * 0.95, raw_signal)
            peak_feat = self.peak.copy()
        else:
            peak_feat = np.zeros(self.m)

        # Eq.5 trust features (simulation mapping):
        d_W = np.clip(raw_signal, 0, 1)
        e_W = np.clip(self.ewma, 0, 1)
        v_W = np.clip(peak_feat, 0, 1)
        tau_mean = np.clip(0.1 + 0.5 * raw_signal, 0, 1)
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
        """Build 5-dim trust features (Eq.5) per instance.

        Paper §III-D: actor input = [5 trust features, cross_signal, epoch_progress].
        Trust features are dims 0-4: (d/W, e/W, v/W, τ̄, σ_τ).
        """
        return np.clip(obs[:, :5], 0.0, 1.0)

    def _get_fault_probs(self, obs: np.ndarray) -> np.ndarray:
        """Trust estimation (Eq.6): f_hat = σ(w·x + b)."""
        features = self._build_trust_features(obs)
        features_t = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            probs = self.trust_estimator(features_t).squeeze(-1).numpy()
        return probs

    def _build_global_state(self, obs: np.ndarray,
                            raw_signal: np.ndarray) -> np.ndarray:
        """Build global state for critic. Always returns (m*obs_dim + m)."""
        cross_features = np.abs(raw_signal - np.mean(raw_signal))
        if not self.cfg.use_centralized_critic:
            # Independent mode: zero out cross-instance info
            # Critic can only see per-instance info (no global coordination)
            cross_features = np.zeros(self.m)
        return np.concatenate([obs.ravel(), cross_features])

    @torch.no_grad()
    def decide(self, raw_signal: np.ndarray, instance_sizes: np.ndarray,
               epoch: int, T_total: int) -> dict:
        """Phase 1-3: Estimate → Select → Filter (Algorithm 1).

        Phase 1 (Estimate): EWMA trust + Eq.5/6 trust features → fault probs
        Phase 2 (Select): 3-head actor → (reconfig, rotate, param) actions
        Phase 3 (Filter): quorum safety filter blocks unsafe evictions
        """
        # ── Phase 1: Estimate ──
        self._update_trust(raw_signal)
        obs = self._build_obs(raw_signal, epoch, T_total)
        global_state = self._build_global_state(obs, raw_signal)
        fault_probs = self._get_fault_probs(obs)

        # ── Phase 2: Select (3-head stochastic actor) ──
        obs_t = torch.tensor(obs, dtype=torch.float32)  # (m, obs_dim)
        action_out = self.actor(obs_t)  # (m, 3) — [reconfig, rotate, param]
        actions = action_out.numpy()

        if self.train_mode:
            # Stochastic: Bernoulli for binary heads, Gaussian noise for param
            reconfig_draw = np.random.random(self.m) < actions[:, 0]
            rotate_draw = np.random.random(self.m) < actions[:, 1]
            param_noise = actions[:, 2] + np.random.normal(0, 0.1, self.m)
            param_draw = np.clip(param_noise, -1, 1)
            sampled_actions = np.stack(
                [reconfig_draw.astype(float), rotate_draw.astype(float), param_draw],
                axis=-1)
        else:
            sampled_actions = np.stack(
                [(actions[:, 0] > 0.5).astype(float),
                 (actions[:, 1] > 0.5).astype(float),
                 actions[:, 2]], axis=-1)

        # ── Phase 2b: Role Action Guides (Algorithm 1, Lines 9-12) ──
        sampled_actions = self._apply_rag(sampled_actions, fault_probs)

        # ── Phase 3: Filter (pre-argmax quorum safety, Alg.1 L13-17) ──
        sampled_actions = self._safety_filter(
            sampled_actions, fault_probs, instance_sizes)

        # Detection: combine trust estimator + actor reconfig signal
        detected = fault_probs > 0.35
        for k in range(self.m):
            if sampled_actions[k, 0] > 0.5:
                detected[k] = True

        # Coordinated attack detection: when cross-instance variance is low
        # but mean fault probability is elevated, all instances are likely
        # under simultaneous attack. Lower detection threshold in this case.
        if self.cfg.use_cross_instance:
            fp_std = np.std(fault_probs)
            fp_mean = np.mean(fault_probs)
            if fp_std < 0.06 and fp_mean > 0.20:
                # Coordinated pattern: use lower threshold for all instances
                coordinated_detected = fault_probs > 0.22
                detected = detected | coordinated_detected

        # Reconfig target: instance with highest fault prob
        reconfig_target = -1
        if self.cfg.use_reconfig and np.any(detected):
            reconfig_target = int(np.argmax(fault_probs))

        self.prev_signal = raw_signal.copy()
        self._epoch_counter += 1

        # Cache for store_transition
        self._last_obs = obs
        self._last_global_state = global_state
        self._last_actions = sampled_actions.copy()   # (m, 3)
        self._last_action_probs = actions.copy()      # (m, 3) — network outputs
        self._last_detected = detected.copy()
        self._last_fault_probs = fault_probs.copy()
        self._last_fault_labels = (raw_signal > 0.3).astype(np.float32)
        self._last_instance_sizes = instance_sizes.copy()

        return {
            "detected_instances": detected,
            "reconfig_target": reconfig_target,
            "leader_rotation": detected.tolist(),
            "detection_probs": fault_probs,
            "safety_masked": False,
        }

    def _apply_rag(self, actions: np.ndarray,
                   fault_probs: np.ndarray) -> np.ndarray:
        """Apply role action guides with stochastic constraint hardness.

        Algorithm 1, Lines 9-12: for each role ς, with prob ch_ς,
        constrain a_t^{i,ς} via rag_ς. Guardian handled by _safety_filter().
        """
        cfg = self.cfg
        for k in range(self.m):
            # Sentinel (→ rotate): if f̂ > θ_high, guide toward rotation
            if np.random.random() < cfg.ch_sentinel:
                if fault_probs[k] > cfg.theta_high:
                    actions[k, 1] = max(actions[k, 1], 0.8)

            # Commander (→ reconfig): block eviction of likely-honest
            if np.random.random() < cfg.ch_commander:
                if actions[k, 0] > 0.5 and fault_probs[k] < 0.3:
                    actions[k, 0] = 0.0

            # Tuner (→ param): clamp to ±η_stab if guide fires
            if np.random.random() < cfg.ch_tuner:
                if self._prev_epoch_actions is not None:
                    prev_param = self._prev_epoch_actions[k, 2]
                    actions[k, 2] = np.clip(
                        actions[k, 2],
                        prev_param - cfg.eta_stab,
                        prev_param + cfg.eta_stab)

        return actions

    def _safety_filter(self, actions: np.ndarray, fault_probs: np.ndarray,
                       instance_sizes: np.ndarray) -> np.ndarray:
        """Pre-argmax quorum safety filter (Algorithm 1, Lines 13-17).

        Paper: block eviction if n_after < 3*f_v^i + 1 + δ_s.
        Simulation: f_v^i estimated as ceil(fault_prob * n) since the protocol
        fault-tolerance parameter is not directly available in the simulation.
        This preserves IGM (ε_IGM = 0) while enforcing coupled BFT constraints.
        """
        if not self.cfg.use_safety_filter:
            return actions

        delta_s = self.cfg.delta_s
        for k in range(self.m):
            if actions[k, 0] > 0.5:  # eviction proposed
                n_after = instance_sizes[k] - 1
                # f_v estimate: ceil of estimated fault count (conservative)
                f_v = int(np.ceil(fault_probs[k] * instance_sizes[k]))
                if n_after < 3 * f_v + 1 + delta_s:
                    actions[k, 0] = 0.0  # mask unsafe eviction

        return actions

    def store_transition(self, per_inst_rewards: np.ndarray,
                         done: bool = False):
        """Store per-instance transitions in MAPPO buffer.

        Computes Eq.14 + Eq.(reward-org) to augment raw per-instance rewards.

        Args:
            per_inst_rewards: (m,) array of per-instance rewards
            done: whether the episode ended
        """
        if not self.train_mode:
            return

        # Compute Eq.(reward-org): r_t^org = r_t + λ₅·grg - λ₆·rrg
        org_r = self._compute_org_reward(
            self._last_obs, self._last_actions,
            self._last_instance_sizes, self._last_fault_probs,
            self._last_fault_labels)

        # Blend: per-instance reward + shared organizational reward
        augmented_rewards = per_inst_rewards + org_r / self.m

        self.buffer.add(
            obs=self._last_obs.copy(),                # (m, obs_dim)
            global_state=self._last_global_state.copy(),
            actions=self._last_actions.copy(),         # (m, 3)
            action_probs=self._last_action_probs.copy(),  # (m, 3)
            rewards=augmented_rewards.copy(),           # (m,)
            done=done,
            fault_labels=self._last_fault_labels.copy(),  # (m,)
        )

    # ─── Reward Computation (Eq.14, P1) ───────────────────────────────────────

    def _compute_eq14_reward(self, obs: np.ndarray, actions: np.ndarray,
                             instance_sizes: np.ndarray,
                             fault_probs: np.ndarray) -> float:
        """Compute regret-aligned reward (Eq.14).

        r_t = λ₁·Σ tp_i - λ₂·Σ ℓ_i - λ₃·Σ vc_i - λ₄·Σ max(0, 3f+1+δ-n_i)
        """
        cfg = self.cfg
        tp_sum = 0.0
        latency_sum = 0.0
        vc_sum = 0.0
        margin_penalty = 0.0

        for k in range(self.m):
            tp_sum += max(0.0, 1.0 - obs[k, 0] * 0.3) * (
                instance_sizes[k] / 25.0)
            latency_sum += obs[k, 0] * 0.5
            vc_sum += float(actions[k, 1] > 0.5)
            f_count = fault_probs[k] * instance_sizes[k]
            shortfall = 3 * f_count + 1 + cfg.delta_s - instance_sizes[k]
            margin_penalty += max(0.0, shortfall)

        return (cfg.lambda_1 * tp_sum
                - cfg.lambda_2 * latency_sum
                - cfg.lambda_3 * vc_sum
                - cfg.lambda_4 * margin_penalty)

    def _compute_org_reward(self, obs: np.ndarray, actions: np.ndarray,
                            instance_sizes: np.ndarray,
                            fault_probs: np.ndarray,
                            fault_labels: np.ndarray) -> float:
        """Compute organizational reward (Eq. reward-org).

        r_t^org = r_t + λ₅·Σ grg - λ₆·Σ rrg  (includes base reward r_t)
        """
        cfg = self.cfg
        base_r = self._compute_eq14_reward(obs, actions, instance_sizes,
                                           fault_probs)

        # ── rrg: role violation penalties ──
        rrg_sum = 0.0
        for k in range(self.m):
            if fault_probs[k] > cfg.theta_high and actions[k, 0] <= 0.5:
                rrg_sum += cfg.r_miss
            if actions[k, 0] > 0.5 and fault_labels[k] < 0.5:
                rrg_sum += cfg.r_false
            if self._prev_epoch_actions is not None:
                param_change = abs(actions[k, 2] - self._prev_epoch_actions[k, 2])
                if param_change > cfg.eta_stab:
                    rrg_sum += cfg.r_churn

        # ── grg: goal achievement bonuses ──
        grg_sum = 0.0
        for k in range(self.m):
            is_faulty = fault_labels[k] > 0.5
            if is_faulty and self._fault_onset_epoch[k] < 0:
                self._fault_onset_epoch[k] = self._epoch_counter
            if is_faulty and fault_probs[k] > 0.5 and self._detection_epoch[k] < 0:
                self._detection_epoch[k] = self._epoch_counter
            if actions[k, 0] > 0.5 and self._eviction_epoch[k] < 0:
                self._eviction_epoch[k] = self._epoch_counter

            if (self._detection_epoch[k] >= 0 and
                    self._fault_onset_epoch[k] >= 0):
                dt_det = self._detection_epoch[k] - self._fault_onset_epoch[k]
                if dt_det <= cfg.kappa_det:
                    grg_sum += cfg.r_b * (cfg.kappa_det - dt_det) / cfg.kappa_det

            if (self._eviction_epoch[k] >= 0 and
                    self._fault_onset_epoch[k] >= 0):
                dt_evict = self._eviction_epoch[k] - self._fault_onset_epoch[k]
                if dt_evict <= cfg.tau_evict:
                    grg_sum += cfg.r_b * (cfg.tau_evict - dt_evict) / cfg.tau_evict

            if self._prev_epoch_actions is not None:
                param_change = abs(actions[k, 2] - self._prev_epoch_actions[k, 2])
                if param_change <= cfg.eta_stab:
                    self._stable_count[k] += 1
                else:
                    self._stable_count[k] = 0
                grg_sum += cfg.r_b * self._stable_count[k] / cfg.window_W

        self._prev_epoch_actions = actions.copy()
        return base_r + cfg.lambda_5 * grg_sum - cfg.lambda_6 * rrg_sum

    def _batch_trust_features(self, obs: np.ndarray) -> np.ndarray:
        """Extract 5-dim trust features from 7-dim obs (N, obs_dim) → (N, 5).

        Trust features are dims 0-4: (d/W, e/W, v/W, τ̄, σ_τ).
        """
        return np.clip(obs[:, :5], 0.0, 1.0)

    def train_step(self) -> dict:
        """Run MAPPO PPO update with 3-head actor. Returns loss dict."""
        if self.buffer.n_steps < 32:
            return {}

        data = self.buffer.get_data()
        returns, advantages = self._compute_gae(data)

        # Flatten: each (timestep, instance) pair becomes a training sample
        n_steps = data["obs"].shape[0]
        m = self.m
        N = n_steps * m

        flat_obs = data["obs"].reshape(N, self.obs_dim)           # (N, obs_dim)
        flat_actions = data["actions"].reshape(N, 3)               # (N, 3)
        flat_action_probs = data["action_probs"].reshape(N, 3)     # (N, 3)
        flat_returns = returns.reshape(N)                          # (N,)
        flat_advantages = advantages.reshape(N)                    # (N,)
        flat_states = np.repeat(data["global_states"], m, axis=0)  # (N, state_dim)

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        n_updates = 0

        for _ in range(self.cfg.ppo_epochs):
            indices = np.random.permutation(N)
            for start in range(0, N, self.cfg.batch_size):
                batch_idx = indices[start:start + self.cfg.batch_size]
                if len(batch_idx) < 8:
                    continue

                b_obs = torch.tensor(flat_obs[batch_idx], dtype=torch.float32)
                b_states = torch.tensor(flat_states[batch_idx], dtype=torch.float32)
                b_actions = torch.tensor(flat_actions[batch_idx], dtype=torch.float32)
                b_old_probs = torch.tensor(flat_action_probs[batch_idx], dtype=torch.float32)
                b_ret = torch.tensor(flat_returns[batch_idx], dtype=torch.float32)
                b_adv = torch.tensor(flat_advantages[batch_idx], dtype=torch.float32)

                # Actor forward: (batch, 3) — [reconfig_prob, rotate_prob, param_mu]
                new_out = self.actor(b_obs)
                new_reconfig_p = new_out[:, 0]
                new_rotate_p = new_out[:, 1]

                old_reconfig_p = b_old_probs[:, 0]
                old_rotate_p = b_old_probs[:, 1]

                # Log-prob for Bernoulli heads (reconfig + rotate)
                a_reconfig = b_actions[:, 0]
                a_rotate = b_actions[:, 1]

                new_log_prob = (
                    a_reconfig * torch.log(new_reconfig_p + 1e-8)
                    + (1 - a_reconfig) * torch.log(1 - new_reconfig_p + 1e-8)
                    + a_rotate * torch.log(new_rotate_p + 1e-8)
                    + (1 - a_rotate) * torch.log(1 - new_rotate_p + 1e-8)
                )
                old_log_prob = (
                    a_reconfig * torch.log(old_reconfig_p + 1e-8)
                    + (1 - a_reconfig) * torch.log(1 - old_reconfig_p + 1e-8)
                    + a_rotate * torch.log(old_rotate_p + 1e-8)
                    + (1 - a_rotate) * torch.log(1 - old_rotate_p + 1e-8)
                )
                # Param head: treat as Gaussian log-prob (fixed σ=0.1)
                param_sigma = 0.1
                a_param = b_actions[:, 2]
                new_param_mu = new_out[:, 2]
                old_param_mu = b_old_probs[:, 2]
                new_log_prob = new_log_prob - 0.5 * ((a_param - new_param_mu) / param_sigma) ** 2
                old_log_prob = old_log_prob - 0.5 * ((a_param - old_param_mu) / param_sigma) ** 2

                ratio = torch.exp(new_log_prob - old_log_prob)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(
                    ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps
                ) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # Entropy: Bernoulli entropy for reconfig + rotate
                entropy = -(
                    new_reconfig_p * torch.log(new_reconfig_p + 1e-8)
                    + (1 - new_reconfig_p) * torch.log(1 - new_reconfig_p + 1e-8)
                    + new_rotate_p * torch.log(new_rotate_p + 1e-8)
                    + (1 - new_rotate_p) * torch.log(1 - new_rotate_p + 1e-8)
                ).mean()

                values = self.critic(b_states)
                critic_loss = F.mse_loss(values, b_ret)

                loss = (
                    actor_loss
                    + self.cfg.value_coef * critic_loss
                    - self.cfg.entropy_coef * entropy
                )

                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.cfg.max_grad_norm)
                self.actor_optim.step()
                self.critic_optim.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        # ── Trust estimator training (Eq.6 joint training, Alg.1 L18) ──
        trust_loss_val = 0.0
        if "fault_labels" in data and n_updates > 0:
            flat_labels = data["fault_labels"].reshape(-1)  # (T*m,)
            flat_obs_t = data["obs"].reshape(-1, self.obs_dim)  # (T*m, obs_dim)
            trust_feat = self._batch_trust_features(flat_obs_t)
            trust_feat_t = torch.tensor(trust_feat, dtype=torch.float32)
            trust_label_t = torch.tensor(flat_labels, dtype=torch.float32)
            pred = self.trust_estimator(trust_feat_t).squeeze(-1)
            trust_loss = F.binary_cross_entropy(pred, trust_label_t)
            self.trust_optim.zero_grad()
            trust_loss.backward()
            nn.utils.clip_grad_norm_(self.trust_estimator.parameters(), 1.0)
            self.trust_optim.step()
            trust_loss_val = trust_loss.item()

        self.buffer.clear()
        if n_updates == 0:
            return {}

        # ── Step LR schedulers (cosine-annealing, Appendix:Hyperparams) ──
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        self.trust_scheduler.step()

        return {
            "actor_loss": total_actor_loss / n_updates,
            "critic_loss": total_critic_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "trust_loss": trust_loss_val,
        }

    def _compute_gae(self, data: dict) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-instance GAE returns and advantages.

        Returns arrays of shape (n_steps, m).
        """
        rewards = data["rewards"]       # (n_steps, m)
        dones = data["dones"]           # (n_steps,)
        states = data["global_states"]  # (n_steps, state_dim)
        T_steps = len(dones)
        m = self.m

        # Critic values (centralized, shared across instances at same step)
        with torch.no_grad():
            states_t = torch.tensor(states, dtype=torch.float32)
            values = self.critic(states_t).numpy()  # (T_steps,)

        returns = np.zeros((T_steps, m))
        advantages = np.zeros((T_steps, m))

        for j in range(m):
            last_gae = 0
            for t in reversed(range(T_steps)):
                if t == T_steps - 1 or dones[t]:
                    next_val = 0
                else:
                    next_val = values[t + 1]

                delta = rewards[t, j] + self.cfg.gamma * next_val - values[t]
                advantages[t, j] = last_gae = (
                    delta
                    + self.cfg.gamma * self.cfg.gae_lambda * (1 - dones[t]) * last_gae
                )
                returns[t, j] = advantages[t, j] + values[t]

        # Normalize advantages across ALL (step, instance) pairs
        flat_adv = advantages.ravel()
        if np.std(flat_adv) > 1e-8:
            advantages = (
                (advantages - np.mean(flat_adv)) / (np.std(flat_adv) + 1e-8)
            )

        return returns, advantages

    def set_eval(self):
        self.train_mode = False
        self.actor.eval()
        self.critic.eval()

    def set_train(self):
        self.train_mode = True
        self.actor.train()
        self.critic.train()

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "trust_estimator": self.trust_estimator.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if "trust_estimator" in ckpt:
            self.trust_estimator.load_state_dict(ckpt["trust_estimator"])


class MAPPORolloutBuffer:
    """MAPPO rollout buffer storing per-instance transitions with 3-dim actions."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.obs_list = []          # each: (m, obs_dim)
        self.state_list = []        # each: (state_dim,)
        self.actions_list = []      # each: (m, 3) — sampled actions
        self.action_probs_list = [] # each: (m, 3) — network output probs/mu
        self.rewards_list = []      # each: (m,)
        self.dones_list = []        # each: scalar
        self.fault_labels_list = [] # each: (m,) — for trust estimator training
        self.n_steps = 0

    def add(self, obs, global_state, actions, action_probs, rewards, done,
            fault_labels=None):
        self.obs_list.append(obs)
        self.state_list.append(global_state)
        self.actions_list.append(actions)
        self.action_probs_list.append(action_probs)
        self.rewards_list.append(rewards)
        self.dones_list.append(float(done))
        self.fault_labels_list.append(
            fault_labels if fault_labels is not None
            else np.zeros(obs.shape[0]))
        self.n_steps += 1

    def get_data(self) -> dict:
        return {
            "obs": np.array(self.obs_list),             # (T, m, obs_dim)
            "global_states": np.array(self.state_list), # (T, state_dim)
            "actions": np.array(self.actions_list),     # (T, m, 3)
            "action_probs": np.array(self.action_probs_list),  # (T, m, 3)
            "rewards": np.array(self.rewards_list),     # (T, m)
            "dones": np.array(self.dones_list),         # (T,)
            "fault_labels": np.array(self.fault_labels_list),  # (T, m)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Ablation Variant Factories
# ═══════════════════════════════════════════════════════════════════════════════

def make_sfac(m: int = 4, **overrides) -> SFACController:
    cfg = SFACConfig(m_instances=m, **overrides)
    return SFACController(cfg)


def make_sfac_no_safety(m: int = 4) -> SFACController:
    cfg = SFACConfig(m_instances=m, use_safety_filter=False)
    return SFACController(cfg)


def make_sfac_no_peak(m: int = 4) -> SFACController:
    cfg = SFACConfig(m_instances=m, use_peak_tracker=False)
    return SFACController(cfg)


def make_sfac_no_cross(m: int = 4) -> SFACController:
    cfg = SFACConfig(m_instances=m, use_cross_instance=False)
    return SFACController(cfg)


def make_sfac_no_reconfig(m: int = 4) -> SFACController:
    cfg = SFACConfig(m_instances=m, use_reconfig=False)
    return SFACController(cfg)


def make_sfac_independent(m: int = 4) -> SFACController:
    """Independent per-instance (no CTDE): own critic, no cross-instance, no reconfig."""
    cfg = SFACConfig(
        m_instances=m,
        use_cross_instance=False,
        use_centralized_critic=False,
        use_reconfig=False,
    )
    return SFACController(cfg)
