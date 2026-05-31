#!/usr/bin/env python3
"""
SFAC-FACMAC Controller — Paper-aligned implementation.

Matches 3framework.tex exactly:
- 5-dim observation (Eq.5): (d/W, e/W, v/W, τ_mean, τ_std)
- Linear trust estimator (Eq.6): σ(w·x + b)
- FACMAC: per-agent actors + monotonic mixing critic + deterministic PG
- Quorum safety filter (Alg lines 13-17): n_after >= 3*f_hat + 1
- Paper reward (Eq.reward): λ₁·tp - λ₂·latency - λ₃·vc - λ₄·margin_penalty
- PER (prioritized experience replay)

Hyperparameters from Appendix Table (Appendix:Hyperparams):
  h=64, |D|=100000, batch=256, lr=3e-4, γ=0.99, W=50, B_ctrl=100 (commits per control epoch)
  λ₁=1.0, λ₂=0.1, λ₃=0.5, λ₄=100.0, λ₅=0.5, λ₆=1.0
  PER α=0.6, β annealed 0.4→1.0
"""

import numpy as np
from dataclasses import dataclass
from collections import deque
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration (matches paper Appendix:Hyperparams)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FACMACConfig:
    m_instances: int = 4        # parallel BFT instances
    n_total: int = 100          # total replicas
    obs_dim: int = 5            # per-agent feature dim (Eq.5)
    hidden_dim: int = 64        # h=64
    buffer_size: int = 100_000  # |D|=10^5
    batch_size: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    tau_target: float = 0.005   # target network soft update
    window_W: int = 50          # sliding window epochs
    # Reward weights (Eq.reward)
    lambda_1: float = 1.0       # throughput
    lambda_2: float = 0.1       # latency
    lambda_3: float = 0.5       # view-changes
    lambda_4: float = 100.0     # safety margin penalty
    lambda_5: float = 0.5       # role goal achievement (grg)
    lambda_6: float = 1.0       # role restriction penalty (rrg)
    delta_s: int = 1            # safety margin
    # PER
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_anneal_steps: int = 50_000
    # Safety filter
    use_safety_filter: bool = True
    # Ablation flags
    use_cross_instance: bool = True
    use_peak_tracker: bool = True
    use_reconfig: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# Networks
# ═══════════════════════════════════════════════════════════════════════════════

class AgentActor(nn.Module):
    """Per-agent actor: deterministic policy μ_i(o_i) → continuous action.

    Per paper Section III-D: three action heads
      - reconfig (sigmoid): probability of proposing eviction
      - rotate (sigmoid): probability of leader rotation
      - param (tanh): consensus parameter adjustment
    """

    def __init__(self, obs_dim: int = 5, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.reconfig_head = nn.Linear(hidden, 1)  # Commander: eviction prob
        self.rotate_head = nn.Linear(hidden, 1)    # Sentinel: rotation prob
        self.param_head = nn.Linear(hidden, 1)     # Tuner: param adjustment

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns (3,) continuous actions."""
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        reconfig = torch.sigmoid(self.reconfig_head(x))
        rotate = torch.sigmoid(self.rotate_head(x))
        param = torch.tanh(self.param_head(x))
        return torch.cat([reconfig, rotate, param], dim=-1)


class AgentQNetwork(nn.Module):
    """Per-agent utility Q_i(o_i, a_i)."""

    def __init__(self, obs_dim: int = 5, action_dim: int = 3, hidden: int = 64):
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
    """Monotonic mixing network: Q_tot = g_ψ(s, Q_1, ..., Q_m).

    Ensures IGM consistency via abs weights (Eq.8).
    """

    def __init__(self, m_instances: int, state_dim: int, hidden: int = 64):
        super().__init__()
        self.m = m_instances
        self.hyper_w1 = nn.Linear(state_dim, m_instances * hidden)
        self.hyper_b1 = nn.Linear(state_dim, hidden)
        self.hyper_w2 = nn.Linear(state_dim, hidden)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, hidden),
                                      nn.ReLU(), nn.Linear(hidden, 1))
        self.hidden = hidden

    def forward(self, q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch = q_values.shape[0]
        w1 = torch.abs(self.hyper_w1(state)).view(batch, self.m, self.hidden)
        b1 = self.hyper_b1(state).view(batch, 1, self.hidden)
        h = F.elu(torch.bmm(q_values.unsqueeze(1), w1) + b1)
        w2 = torch.abs(self.hyper_w2(state)).view(batch, self.hidden, 1)
        b2 = self.hyper_b2(state).view(batch, 1, 1)
        q_tot = torch.bmm(h, w2) + b2
        return q_tot.squeeze(-1).squeeze(-1)


class TrustEstimator(nn.Module):
    """Linear trust estimator (Eq.6): f_hat = σ(w·x + b)."""

    def __init__(self, obs_dim: int = 5):
        super().__init__()
        self.linear = nn.Linear(obs_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))


# ═══════════════════════════════════════════════════════════════════════════════
# Prioritized Experience Replay
# ═══════════════════════════════════════════════════════════════════════════════

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
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
        probs = probs / probs.sum()
        indices = np.random.choice(self.size, size=min(batch_size, self.size),
                                   p=probs, replace=True)
        samples = [self.buffer[i] for i in indices]
        weights = (self.size * probs[indices]) ** (-beta)
        weights = weights / weights.max()
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

class SFACFACMACController:
    """Safe Factored Actor-Critic with FACMAC (paper-aligned).

    Key properties matching paper:
    1. 5-dim observations (Eq.5)
    2. Linear trust estimator (Eq.6)
    3. Monotonic mixing critic (Eq.8)
    4. Deterministic policy gradient (Eq.9)
    5. Pre-argmax quorum safety filter
    6. PER
    """

    def __init__(self, cfg: FACMACConfig):
        self.cfg = cfg
        self.m = cfg.m_instances

        self.actor = AgentActor(cfg.obs_dim, cfg.hidden_dim)
        self.actor_target = AgentActor(cfg.obs_dim, cfg.hidden_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.q_networks = nn.ModuleList([
            AgentQNetwork(cfg.obs_dim, 3, cfg.hidden_dim) for _ in range(cfg.m_instances)
        ])
        self.q_targets = nn.ModuleList([
            AgentQNetwork(cfg.obs_dim, 3, cfg.hidden_dim) for _ in range(cfg.m_instances)
        ])
        for i in range(cfg.m_instances):
            self.q_targets[i].load_state_dict(self.q_networks[i].state_dict())

        state_dim = cfg.m_instances * cfg.obs_dim + cfg.m_instances
        self.mixer = MonotonicMixer(cfg.m_instances, state_dim, cfg.hidden_dim)
        self.mixer_target = MonotonicMixer(cfg.m_instances, state_dim, cfg.hidden_dim)
        self.mixer_target.load_state_dict(self.mixer.state_dict())

        self.trust_estimator = TrustEstimator(cfg.obs_dim)

        actor_params = list(self.actor.parameters())
        critic_params = (list(self.q_networks.parameters()) +
                         list(self.mixer.parameters()))
        self.actor_optim = optim.Adam(actor_params, lr=cfg.lr)
        self.critic_optim = optim.Adam(critic_params, lr=cfg.lr)
        # Separate trust optimizer: higher lr for faster convergence
        self.trust_optim = optim.Adam(
            self.trust_estimator.parameters(), lr=cfg.lr * 5)

        self.buffer = PrioritizedReplayBuffer(cfg.buffer_size, cfg.per_alpha)
        self.beta = cfg.per_beta_start
        self.train_step_count = 0
        self._window: deque = deque(maxlen=cfg.window_W)
        self.train_mode = True

    def reset(self):
        self._window.clear()

    def set_train(self):
        self.train_mode = True
        self.actor.train()

    def set_eval(self):
        self.train_mode = False
        self.actor.eval()

    # ─── Observation (Eq.5) ──────────────────────────────────────────────────

    def _build_obs_from_signals(self, raw_signal: np.ndarray,
                                 instance_sizes: np.ndarray,
                                 epoch: int, T_total: int) -> np.ndarray:
        """Map simulation signals to paper's 5-dim features (Eq.5).

        Mapping (simulation → paper semantics):
          raw_signal[k] → d/W (timeout fraction proxy)
          raw_signal[k]*0.5 → e/W (equivocation proxy)
          max(0, raw_signal-0.3) → v/W (view-change under high fault)
          0.1 + 0.5*raw_signal → τ_mean (latency increases with faults)
          0.05 + 0.3*raw_signal → τ_std (variance increases too)
        """
        m = self.m
        obs = np.zeros((m, 5))
        for k in range(m):
            s = raw_signal[k]
            obs[k] = [
                s,
                s * 0.5,
                max(0, s - 0.3),
                min(1.0, 0.1 + 0.5 * s),
                min(1.0, 0.05 + 0.3 * s),
            ]
        return np.clip(obs, 0.0, 1.0)

    def _build_global_state(self, obs: np.ndarray) -> np.ndarray:
        cross = np.abs(obs[:, 0] - np.mean(obs[:, 0]))
        return np.concatenate([obs.ravel(), cross])

    # ─── Trust Estimation (Eq.6) ─────────────────────────────────────────────

    def get_fault_probs(self, obs: np.ndarray) -> np.ndarray:
        """Inference-time fault probability (no grad)."""
        obs_t = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            probs = self.trust_estimator(obs_t).squeeze(-1).numpy()
        return probs

    def get_fault_probs_differentiable(self, obs_t: torch.Tensor) -> torch.Tensor:
        """Training-time fault probability (with grad for end-to-end learning)."""
        return self.trust_estimator(obs_t).squeeze(-1)

    # ─── Safety Filter (Section III-D3) ──────────────────────────────────────

    def _safety_filter(self, actions: np.ndarray, obs: np.ndarray,
                       instance_sizes: np.ndarray) -> np.ndarray:
        """Pre-argmax quorum safety filter (Algorithm 1): n_after >= 3*f_v + 1 + δ_s.
        Uses protocol-level fault bound f_v = ⌊(n-1)/3⌋."""
        if not self.cfg.use_safety_filter:
            return actions

        delta_s = self.cfg.delta_s

        for k in range(self.m):
            if actions[k, 0] > 0.5:  # eviction proposed
                n_after = instance_sizes[k] - 1
                f_v = int((instance_sizes[k] - 1) // 3)
                if n_after < 3 * f_v + 1 + delta_s:
                    actions[k, 0] = 0.0  # mask

        return actions

    # ─── Decision ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def decide_from_obs(self, obs: np.ndarray, instance_sizes: np.ndarray,
                        epoch: int = 0) -> dict:
        """Decision path accepting real 5-dim observations directly (production).

        Use this when real per-instance 5-dim features (Eq.5) are available,
        bypassing the synthetic reconstruction in _build_obs_from_signals().
        """
        obs = np.clip(obs, 0.0, 1.0).astype(np.float32)
        return self._decide_impl(obs, instance_sizes)

    @torch.no_grad()
    def decide(self, raw_signal: np.ndarray, instance_sizes: np.ndarray,
               epoch: int, T_total: int) -> dict:
        """Decision path with synthetic observation reconstruction (simulation)."""
        obs = self._build_obs_from_signals(raw_signal, instance_sizes, epoch, T_total)
        return self._decide_impl(obs, instance_sizes)

    @torch.no_grad()
    def _decide_impl(self, obs: np.ndarray, instance_sizes: np.ndarray) -> dict:
        """Shared decision logic for both real and synthetic observation paths."""
        obs_t = torch.tensor(obs, dtype=torch.float32)

        actions = self.actor(obs_t).numpy()  # (m, 3)

        if self.train_mode:
            # Anneal exploration noise: start at 0.2, decay to 0.02
            noise_scale = max(0.02, 0.2 * (1.0 - self.train_step_count / 20000))
            noise = np.random.normal(0, noise_scale, actions.shape)
            actions = np.clip(actions + noise, -1, 1)
            actions[:, :2] = np.clip(actions[:, :2], 0, 1)

        actions = self._safety_filter(actions, obs, instance_sizes)

        # Detection: use trust estimator (Eq.6) with learned threshold
        fault_probs = self.get_fault_probs(obs)
        detected = fault_probs > 0.35  # ρ threshold (calibrated)
        reconfig_target = int(np.argmax(fault_probs)) if np.any(detected) else -1

        return {
            "detected_instances": detected,
            "reconfig_target": reconfig_target,
            "leader_rotation": actions[:, 1] > 0.5,
            "actions_raw": actions,
            "obs": obs,
            "fault_probs": fault_probs,
        }

    # ─── Reward (Eq.reward) ──────────────────────────────────────────────────

    def compute_reward(self, throughputs: np.ndarray, latencies: np.ndarray,
                       view_changes: np.ndarray, instance_sizes: np.ndarray,
                       fault_probs: np.ndarray,
                       decision: dict = None,
                       fault_labels: np.ndarray = None,
                       prev_params: np.ndarray = None) -> float:
        """r_t^org = r_t + λ₅·Σgrg - λ₆·Σrrg (Eq. reward-org)

        Args:
            decision: output of decide() with detected_instances, reconfig_target, actions_raw
            fault_labels: per-instance binary ground truth (1=Byzantine active)
            prev_params: previous epoch's tuner parameters for churn penalty
        """
        cfg = self.cfg
        margin_penalty = 0.0
        for k in range(self.m):
            f_count = fault_probs[k] * instance_sizes[k]
            shortfall = 3 * f_count + 1 + cfg.delta_s - instance_sizes[k]
            margin_penalty += max(0.0, shortfall)

        r_base = (cfg.lambda_1 * np.sum(throughputs)
                  - cfg.lambda_2 * np.sum(latencies)
                  - cfg.lambda_3 * np.sum(view_changes)
                  - cfg.lambda_4 * margin_penalty)

        # Organizational reward (Eq. reward-org, Appendix OrgSpecs)
        grg_total = 0.0
        rrg_total = 0.0

        if decision is not None and fault_labels is not None:
            detected = decision.get("detected_instances", np.zeros(self.m, dtype=bool))
            reconfig_target = decision.get("reconfig_target", -1)
            actions_raw = decision.get("actions_raw", np.zeros((self.m, 3)))

            # --- rrg: role violation penalties (Eq. rrg) ---
            r_miss = 1.0   # sentinel miss penalty
            r_false = 1.0  # commander false eviction penalty
            r_churn = 0.5  # tuner parameter churn penalty
            eta_stab = 0.1 # stability threshold

            for k in range(self.m):
                # Sentinel: penalize if fault detected (f_hat > θ_high) but no reconfig action
                if fault_probs[k] > 0.7 and fault_labels[k] > 0.5:
                    if reconfig_target != k:
                        rrg_total += r_miss  # missed detection

                # Commander: penalize false eviction (reconfig honest node)
                if reconfig_target == k and fault_labels[k] < 0.5:
                    rrg_total += r_false

                # Tuner: penalize excessive parameter churn
                if prev_params is not None:
                    param_change = np.linalg.norm(actions_raw[k, 2:] -
                                                  (prev_params[k, 2:] if prev_params.ndim > 1
                                                   else prev_params[2:]))
                    if param_change > eta_stab:
                        rrg_total += r_churn

            # --- grg: goal achievement bonuses (Eq. grg) ---
            r_b = 2.0              # base bonus
            kappa_det = 10         # detection deadline (epochs)
            tau_evict = 20         # eviction deadline (epochs)

            for k in range(self.m):
                if fault_labels[k] > 0.5 and detected[k]:
                    # Detection speed bonus: r_b * (kappa_det - Δt_det) / kappa_det
                    det_delay = getattr(self, '_det_delays', {}).get(k, kappa_det)
                    if det_delay <= kappa_det:
                        grg_total += r_b * (kappa_det - det_delay) / kappa_det

                    # Eviction speed bonus (if reconfig targets this instance)
                    if reconfig_target == k:
                        evict_delay = getattr(self, '_evict_delays', {}).get(k, tau_evict)
                        if evict_delay <= tau_evict:
                            grg_total += r_b * (tau_evict - evict_delay) / tau_evict

            # Stability bonus: fraction of window with stable operation
            W = cfg.window_W
            stable_frac = 1.0 - np.mean(view_changes) / max(1.0, np.max(view_changes) + 1)
            grg_total += r_b * stable_frac

        return r_base + cfg.lambda_5 * grg_total - cfg.lambda_6 * rrg_total

    # ─── Training (FACMAC + PER) ─────────────────────────────────────────────

    def store_transition(self, obs: np.ndarray, actions: np.ndarray,
                         reward: float, next_obs: np.ndarray,
                         global_state: np.ndarray, next_global_state: np.ndarray,
                         done: bool, fault_labels: np.ndarray = None):
        """Store transition. fault_labels: per-instance binary (1=Byzantine active)."""
        self.buffer.push({
            "obs": obs.copy(), "actions": actions.copy(),
            "reward": reward, "next_obs": next_obs.copy(),
            "state": global_state.copy(), "next_state": next_global_state.copy(),
            "done": done,
            "fault_labels": fault_labels.copy() if fault_labels is not None else None,
        })

    def train_step(self) -> dict:
        cfg = self.cfg
        if len(self.buffer) < cfg.batch_size:
            return {}

        self.train_step_count += 1
        frac = min(1.0, self.train_step_count / cfg.per_beta_anneal_steps)
        self.beta = cfg.per_beta_start + frac * (cfg.per_beta_end - cfg.per_beta_start)

        samples, indices, is_weights = self.buffer.sample(cfg.batch_size, self.beta)
        if samples is None:
            return {}

        obs_batch = torch.tensor(np.array([s["obs"] for s in samples]), dtype=torch.float32)
        act_batch = torch.tensor(np.array([s["actions"] for s in samples]), dtype=torch.float32)
        rew_batch = torch.tensor(np.array([s["reward"] for s in samples]), dtype=torch.float32)
        next_obs_batch = torch.tensor(np.array([s["next_obs"] for s in samples]), dtype=torch.float32)
        state_batch = torch.tensor(np.array([s["state"] for s in samples]), dtype=torch.float32)
        next_state_batch = torch.tensor(np.array([s["next_state"] for s in samples]), dtype=torch.float32)
        done_batch = torch.tensor(np.array([s["done"] for s in samples]), dtype=torch.float32)

        # Critic update
        with torch.no_grad():
            next_actions = torch.stack([
                self.actor_target(next_obs_batch[:, i, :])
                for i in range(self.m)
            ], dim=1)
            target_qs = torch.stack([
                self.q_targets[i](next_obs_batch[:, i, :], next_actions[:, i, :]).squeeze(-1)
                for i in range(self.m)
            ], dim=1)
            q_tot_target = self.mixer_target(target_qs, next_state_batch)
            y = rew_batch + cfg.gamma * (1 - done_batch) * q_tot_target

        current_qs = torch.stack([
            self.q_networks[i](obs_batch[:, i, :], act_batch[:, i, :]).squeeze(-1)
            for i in range(self.m)
        ], dim=1)
        q_tot = self.mixer(current_qs, state_batch)

        td_errors = (y - q_tot).detach().numpy()
        self.buffer.update_priorities(indices, td_errors)

        critic_loss = (is_weights * (y - q_tot) ** 2).mean()

        # Trust estimator loss (separate optimizer for stability)
        trust_loss = torch.tensor(0.0)
        fault_labels_list = [s.get("fault_labels") for s in samples]
        if fault_labels_list[0] is not None:
            labels = torch.tensor(
                np.array([fl for fl in fault_labels_list]), dtype=torch.float32)
            pred_faults = self.get_fault_probs_differentiable(obs_batch)
            trust_loss = F.binary_cross_entropy(pred_faults, labels)
            self.trust_optim.zero_grad()
            trust_loss.backward()
            nn.utils.clip_grad_norm_(self.trust_estimator.parameters(), 1.0)
            self.trust_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.q_networks.parameters()) + list(self.mixer.parameters()), 0.5)
        self.critic_optim.step()

        # Actor update (deterministic PG, Eq.9)
        actions_pred = torch.stack([
            self.actor(obs_batch[:, i, :]) for i in range(self.m)
        ], dim=1)
        q_values = torch.stack([
            self.q_networks[i](obs_batch[:, i, :], actions_pred[:, i, :]).squeeze(-1)
            for i in range(self.m)
        ], dim=1)
        q_tot_actor = self.mixer(q_values, state_batch)
        actor_loss = -q_tot_actor.mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optim.step()

        # Soft update targets
        self._soft_update(self.actor, self.actor_target, cfg.tau_target)
        for i in range(self.m):
            self._soft_update(self.q_networks[i], self.q_targets[i], cfg.tau_target)
        self._soft_update(self.mixer, self.mixer_target, cfg.tau_target)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}

    @staticmethod
    def _soft_update(source: nn.Module, target: nn.Module, tau: float):
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)
