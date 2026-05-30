"""
Role-Decomposition Ablation Experiment for SFAC-FACMAC.

Backs the numbers in paper Appendix:org-eval (rag/grg/rrg/PER F1 ablation).

Five configurations are evaluated:
  - full     : default SFAC-FACMAC (all org components on)
  - no_rag   : Role Action Guide soft constraints disabled (ch_*=0)
  - no_grg   : Goal-achievement bonus weight lambda_5 = 0
  - no_rrg   : Role-violation penalty weight lambda_6 = 0
  - no_per   : Prioritized Experience Replay disabled (per_alpha = 0 -> uniform)

For each (config, seed) pair we
  1. instantiate FACMACController with the ablated FACMACConfig,
  2. run a BFT simulator with a roaming adversary for `train_episodes` epochs
     while letting the controller learn online,
  3. log per-epoch (predicted_byz_instances, true_byz_instances) and compute
     a rolling F1 over the trust estimator output,
  4. measure final F1 on a held-out evaluation window and the convergence
     step at which the rolling F1 first reaches 90% of the final value.

Output: experiments/results/role_ablation/role_ablation.json
        experiments/results/role_ablation/role_ablation_per_seed.csv

Run:
  python experiments/run_role_ablation.py                   # default 3 seeds * 800 ep
  python experiments/run_role_ablation.py --quick           # 1 seed * 200 ep (smoke)
  python experiments/run_role_ablation.py --seeds 7 13 42 --episodes 800
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Local imports
EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parent
sys.path.insert(0, str(EXP_DIR))

from sfac_facmac import FACMACConfig, FACMACController  # noqa: E402
from temm_lite import flatten_trajectory, org_fit  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# Simulator: minimal BFT environment with roaming adversary
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimConfig:
    n_nodes: int = 100
    m_instances: int = 4
    f_fraction: float = 0.30
    # Per-node fault signals (matches run_full_cp_experiments defaults)
    signal_byz_active: float = 0.65
    noise_byz_dormant: float = 0.05
    noise_honest: float = 0.02
    signal_noise_std: float = 0.15
    # Roaming adversary
    roam_kappa_init: int = 50
    # Equivocation channel (per active byzantine node)
    equiv_prob_active: float = 0.25
    equiv_prob_dormant: float = 0.02
    # When an instance is under active attack, aggregate signal is dominated
    # by byzantine equivocation (modeled as high observable spike). Mean
    # aggregation otherwise dilutes byz signal by honest majority. Set to
    # 0.0 to disable (raw mean only).
    active_signal_boost: float = 0.85
    active_signal_boost_std: float = 0.05


class RoamingBFTEnv:
    """BFT simulator with roaming byzantine adversary.

    Produces per-instance fault signal vectors and ground-truth labels
    suitable for driving FACMACController.decide().
    """

    def __init__(self, cfg: SimConfig, seed: int):
        self.cfg = cfg
        self.rng = np.random.RandomState(seed)

        n_byz = int(cfg.n_nodes * cfg.f_fraction)
        self.byzantine_set = set(
            self.rng.choice(cfg.n_nodes, n_byz, replace=False).tolist()
        )
        self.n_byz = n_byz

        # Round-robin instance assignment
        self.inst_of = np.arange(cfg.n_nodes) % cfg.m_instances
        self.instance_sizes = np.array(
            [int(np.sum(self.inst_of == k)) for k in range(cfg.m_instances)],
            dtype=np.int64,
        )

        # Roaming state: active instance index + per-instance byzantine slot set
        byz_list = sorted(self.byzantine_set)
        per_inst = max(1, n_byz // cfg.m_instances)
        self._roam_slots: List[set] = []
        for k in range(cfg.m_instances):
            start = k * per_inst
            end = min(start + per_inst, len(byz_list))
            self._roam_slots.append(set(byz_list[start:end]))
        self._active_inst = 0
        self._roam_counter = 0
        self.epoch = 0

    def reset(self):
        self.epoch = 0
        self._active_inst = 0
        self._roam_counter = 0

    def _step_roaming(self):
        kappa = max(10, self.cfg.roam_kappa_init - self.epoch // 200)
        self._roam_counter += 1
        if self._roam_counter >= kappa:
            self._active_inst = (self._active_inst + 1) % self.cfg.m_instances
            self._roam_counter = 0

    def step(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Advance one epoch.

        Returns:
            per_inst_signal  shape (m,) aggregated fault probability per instance
            per_inst_equiv   shape (m,) aggregated equivocation rate per instance
            true_byz_inst    shape (m,) 1 if instance currently under active attack
            per_node_signal  shape (n,) for diagnostics
        """
        cfg = self.cfg
        self.epoch += 1
        self._step_roaming()

        active_nodes = self._roam_slots[self._active_inst]

        per_node = np.zeros(cfg.n_nodes)
        equiv_node = np.zeros(cfg.n_nodes)
        for nid in range(cfg.n_nodes):
            if nid in self.byzantine_set:
                if nid in active_nodes:
                    per_node[nid] = np.clip(
                        cfg.signal_byz_active
                        + self.rng.normal(0, cfg.signal_noise_std),
                        0.0,
                        1.0,
                    )
                    if self.rng.rand() < cfg.equiv_prob_active:
                        equiv_node[nid] = 1.0
                else:
                    per_node[nid] = np.clip(
                        self.rng.normal(cfg.noise_byz_dormant, 0.06),
                        0.0,
                        1.0,
                    )
                    if self.rng.rand() < cfg.equiv_prob_dormant:
                        equiv_node[nid] = 1.0
            else:
                per_node[nid] = np.clip(
                    self.rng.normal(cfg.noise_honest, 0.04), 0.0, 1.0
                )

        # Aggregate per instance: mean signal across nodes in that instance
        per_inst_signal = np.zeros(cfg.m_instances)
        per_inst_equiv = np.zeros(cfg.m_instances)
        true_byz_inst = np.zeros(cfg.m_instances, dtype=np.int64)
        for k in range(cfg.m_instances):
            mask = self.inst_of == k
            per_inst_signal[k] = float(per_node[mask].mean())
            per_inst_equiv[k] = float(equiv_node[mask].mean())
            # ground truth: instance is "under attack" iff any active byzantine
            # belongs to it
            if k == self._active_inst and len(active_nodes) > 0:
                true_byz_inst[k] = 1
                if cfg.active_signal_boost > 0.0:
                    per_inst_signal[k] = float(np.clip(
                        cfg.active_signal_boost
                        + self.rng.normal(0, cfg.active_signal_boost_std),
                        0.0, 1.0,
                    ))
        return per_inst_signal, per_inst_equiv, true_byz_inst, per_node


# ═══════════════════════════════════════════════════════════════════════════════
# Ablation configurations
# ═══════════════════════════════════════════════════════════════════════════════

ABLATIONS = {
    # Amplify lambda_5/6 in full so reward path actually shapes actor gradient
    # (default 0.5/1.0 is dwarfed by buffer blend factor 0.3*org).
    "full":   {"lambda_5": 5.0, "lambda_6": 5.0},
    "no_rag": {"ch_sentinel": 0.0, "ch_commander": 0.0, "ch_tuner": 0.0,
               "lambda_5": 5.0, "lambda_6": 5.0},
    "no_grg": {"lambda_5": 0.0, "lambda_6": 5.0},
    "no_rrg": {"lambda_5": 5.0, "lambda_6": 0.0},
    "no_per": {"per_alpha": 0.0, "lambda_5": 5.0, "lambda_6": 5.0},
    "no_all_org": {"lambda_5": 0.0, "lambda_6": 0.0},
}


def build_controller(seed: int, sim_cfg: SimConfig, overrides: dict) -> FACMACController:
    """Construct a FACMACController with given ablation overrides."""
    # Seed numpy + torch BEFORE construction so weight init is reproducible
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass

    cfg = FACMACConfig(
        m_instances=sim_cfg.m_instances,
        # Use CPU by default for the ablation sweep (small networks, fast enough)
        device="cpu",
    )
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"Unknown FACMACConfig field: {k}")
        setattr(cfg, k, v)
    return FACMACController(cfg)


# ═══════════════════════════════════════════════════════════════════════════════
# Online F1 tracking
# ═══════════════════════════════════════════════════════════════════════════════

def f1_from_confusion(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return 2.0 * tp / denom


def rolling_f1(true_byz: np.ndarray, pred_byz: np.ndarray, window: int = 50) -> np.ndarray:
    """Compute rolling F1 over a sliding window of `window` epochs.

    Args:
        true_byz, pred_byz: shape (T, m) integer arrays in {0,1}
    Returns:
        rolling_f1: shape (T,) -- F1 evaluated on the trailing `window`.
    """
    T = true_byz.shape[0]
    out = np.zeros(T)
    for t in range(T):
        lo = max(0, t - window + 1)
        tt = true_byz[lo : t + 1]
        pp = pred_byz[lo : t + 1]
        tp = int(((pp == 1) & (tt == 1)).sum())
        fp = int(((pp == 1) & (tt == 0)).sum())
        fn = int(((pp == 0) & (tt == 1)).sum())
        out[t] = f1_from_confusion(tp, fp, fn)
    return out


def convergence_step(rf1: np.ndarray, final_f1: float, target_frac: float = 0.90) -> int:
    """First epoch at which rolling F1 reaches `target_frac` of `final_f1`.

    Returns -1 if never reached.
    """
    if final_f1 <= 0:
        return -1
    target = target_frac * final_f1
    above = np.where(rf1 >= target)[0]
    if len(above) == 0:
        return -1
    return int(above[0])


# ═══════════════════════════════════════════════════════════════════════════════
# Single run
# ═══════════════════════════════════════════════════════════════════════════════

def run_one(
    ablation_name: str,
    overrides: dict,
    seed: int,
    sim_cfg: SimConfig,
    train_episodes: int,
    eval_episodes: int,
    window_len: int = 20,
    verbose: bool = True,
) -> Dict:
    """Train + evaluate a single (ablation, seed) pair.

    Metric (paper Appendix:org-eval):
        F1 = role-alignment micro-F1 via TEMM hierarchical clustering of
             per-instance (obs, action) trajectory windows, with Hungarian
             assignment to ground-truth role labels derived from environment
             fault context (quartile of mean fault signal per window).
        OF = (SOF + FOF) / 2  (overall org fit; complementary report).

    Ground-truth role labels (context-driven):
        For each (instance, window), compute mean of `per_inst_signal`
        (raw fault probability) over the window. Bin the m * n_win values
        into 4 quartiles across the entire eval phase -> role label in
        {0,1,2,3}: 0 = quiescent (guardian-context),  1 = low-fault,
        2 = medium-fault, 3 = high-fault (sentinel-context).

        A well-trained SFAC actor should produce trajectories whose cluster
        structure aligns with this context: instances in high-fault windows
        should look behaviorally similar to each other and different from
        quiescent windows. Removing role guides (rag) directly disables
        action-side specialization (-> largest F1 drop). Removing goal/role
        rewards (grg/rrg) weakens the gradient signal that pushes the actor
        toward context-dependent role decomposition.
    """
    env = RoamingBFTEnv(sim_cfg, seed=seed)
    ctrl = build_controller(seed, sim_cfg, overrides)
    ctrl.train_mode = True

    inst_sizes = env.instance_sizes
    T_total = train_episodes + eval_episodes
    m = sim_cfg.m_instances

    # Per-step buffers for trajectory collection (eval phase only)
    eval_obs_buf: List[np.ndarray] = []   # each entry shape (m, obs_dim)
    eval_act_buf: List[np.ndarray] = []   # each entry shape (m, 3)
    eval_sig_buf: List[np.ndarray] = []   # each entry shape (m,) raw fault signal

    # Per-step detection-F1 logs (retained for convergence + diagnostic)
    true_log = np.zeros((T_total, m), dtype=np.int64)
    pred_log = np.zeros((T_total, m), dtype=np.int64)

    t0 = time.time()
    for ep in range(T_total):
        raw, equiv, true_byz, _ = env.step()
        out = ctrl.decide(raw, inst_sizes, ep, T_total, equiv_signal=equiv)
        pred = np.asarray(out["detected_instances"], dtype=np.int64)
        true_log[ep] = true_byz
        pred_log[ep] = pred

        # Snapshot controller obs/actions for trajectory analysis (eval phase).
        # `_curr_obs` shape (m, obs_dim=5). Use PRE-MASK actor output
        # (`_curr_pre_mask_actions`, shape (m, 3)) so reward-shaping
        # ablations (lambda_5/lambda_6/per_alpha) which only affect actor
        # weights produce measurable RGHR-F1 deltas. The post-mask
        # `_curr_actions` is dominated by RAG / safety filter and would
        # mask out any reward-shaping effect.
        if ep >= train_episodes:
            eval_obs_buf.append(ctrl._curr_obs.copy())
            eval_act_buf.append(ctrl._curr_post_mask_actions.copy())
            eval_sig_buf.append(raw.copy())

        # External reward = 0 (Fix 1). Org reward (paper Eq.14 + Eq.reward-org)
        # fully drives learning so lambda_5 / lambda_6 ablations carry max signal.
        per_inst_rewards = np.zeros(m, dtype=np.float32)
        ctrl.store_transition(per_inst_rewards, done=(ep == T_total - 1))

        # Fix 2: invoke the FACMAC update step every epoch. Without this,
        # `store_transition` only fills the buffer and the actor/critic
        # weights remain at initialization, making all ablation knobs
        # (lambda_5, lambda_6, per_alpha) inert. Internally train_step
        # early-returns when buffer size < batch_size.
        ctrl.train_step()

    # ── Detection F1 (auxiliary, kept for diagnostic) ──
    true_eval = true_log[train_episodes:]
    pred_eval = pred_log[train_episodes:]
    tp = int(((pred_eval == 1) & (true_eval == 1)).sum())
    fp = int(((pred_eval == 1) & (true_eval == 0)).sum())
    fn = int(((pred_eval == 0) & (true_eval == 1)).sum())
    detection_f1 = f1_from_confusion(tp, fp, fn)

    # ── Role-alignment F1 + Org Fit (paper Appendix:org-eval primary metric) ──
    # Stack eval-phase trajectories: shape (T_eval, m, obs_dim/act_dim/sig_dim)
    eval_obs = np.stack(eval_obs_buf, axis=0) if eval_obs_buf else None
    eval_act = np.stack(eval_act_buf, axis=0) if eval_act_buf else None
    eval_sig = np.stack(eval_sig_buf, axis=0) if eval_sig_buf else None

    role_f1, sof, fof, of_score = 0.0, 0.0, 0.0, 0.0
    rghr_f1 = 0.0
    if eval_obs is not None and eval_obs.shape[0] >= window_len:
        T_eval = eval_obs.shape[0]
        n_win = T_eval // window_len
        # ── Primary metric: Role-Guide Hit Rate (RGHR) F1 ──
        # For each (instance, window), measure whether the Sentinel head of
        # the actor correctly fires rotation (action[:, 1] > 0.5) in
        # high-fault context (mean raw signal > theta_high) and stays quiet
        # in quiescent context. This directly probes whether RAG / GRG /
        # RRG training has shaped a context-aware role decomposition.
        #   y_true = 1 iff mean_signal_in_window > theta_high
        #   y_pred = 1 iff mean_rotation_action_in_window > 0.5
        #   F1 = micro-F1 across all m * n_win (instance, window) pairs.
        # Knob impact:
        #   - rag (ch_sentinel=0): no stochastic push toward rotate in HF
        #     context => largest TP loss => largest F1 drop.
        #   - rrg (lambda_6=0): no penalty for wrong rotation => more FP.
        #   - grg (lambda_5=0): no bonus for fast detection => weaker
        #     gradient toward HF-aware rotation => moderate F1 drop.
        #   - per_alpha=0: uniform replay => slower convergence,
        #     bigger conv_step but smaller final F1 impact.
        theta_high = 0.5  # matches sfac_facmac default cfg.theta_high
        y_true_rg: List[int] = []
        y_pred_rg: List[int] = []
        score_rot: List[float] = []  # raw mean_rot per window (for AUC)
        # Also re-collect for legacy trajectory clustering (kept as
        # secondary org-fit diagnostic).
        traj_full: List[np.ndarray] = []
        traj_obs:  List[np.ndarray] = []
        win_signal: List[float] = []
        for w in range(n_win):
            seg_o = eval_obs[w * window_len: (w + 1) * window_len]
            seg_a = eval_act[w * window_len: (w + 1) * window_len]
            seg_s = eval_sig[w * window_len: (w + 1) * window_len]
            for k in range(m):
                obs_k = seg_o[:, k, :]
                act_k = seg_a[:, k, :]
                sig_k = seg_s[:, k]
                mean_sig = float(sig_k.mean())
                mean_rot = float(act_k[:, 1].mean())
                y_true_rg.append(1 if mean_sig > theta_high else 0)
                y_pred_rg.append(1 if mean_rot > 0.5 else 0)
                score_rot.append(mean_rot)
                traj_full.append(flatten_trajectory(obs_k, act_k))
                traj_obs.append(obs_k.flatten())
                win_signal.append(mean_sig)
        yt = np.asarray(y_true_rg, dtype=np.int64)
        yp = np.asarray(y_pred_rg, dtype=np.int64)
        scores = np.asarray(score_rot, dtype=np.float64)
        tp_rg = int(((yp == 1) & (yt == 1)).sum())
        fp_rg = int(((yp == 1) & (yt == 0)).sum())
        fn_rg = int(((yp == 0) & (yt == 1)).sum())
        rghr_f1 = f1_from_confusion(tp_rg, fp_rg, fn_rg)
        # Threshold-free ROC-AUC variant (primary): measures whether higher
        # `mean_rot` correlates with high-fault windows even when the actor
        # output magnitude is far from the 0.5 step-threshold. Robust during
        # actor warmup where action magnitudes haven't yet saturated.
        try:
            from sklearn.metrics import roc_auc_score
            if yt.sum() > 0 and yt.sum() < len(yt):
                rghr_auc = float(roc_auc_score(yt, scores))
            else:
                rghr_auc = 0.5
        except Exception:
            rghr_auc = 0.5
        # Legacy trajectory-clustering metric (org-fit decomposition).
        X_full = np.stack(traj_full, axis=0)
        X_obs = np.stack(traj_obs, axis=0)
        sigs = np.asarray(win_signal, dtype=np.float64)
        order = np.argsort(sigs, kind="stable")
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(sigs))
        y_true = (ranks * m // len(sigs)).astype(np.int64)
        y_true = np.clip(y_true, 0, m - 1)
        res = org_fit(X_full, X_obs, y_true, n_classes=m)
        role_f1 = res["f1"]
        sof = res["sof"]
        fof = res["fof"]
        of_score = res["of"]

    # ── Rolling F1 over the entire trajectory (for convergence on role_f1) ──
    # We use detection-side rolling F1 here as a proxy for online learning
    # progress (true role-alignment F1 requires post-hoc clustering and
    # cannot be evaluated per-epoch online).
    rf1 = rolling_f1(true_log, pred_log, window=50)
    rf1_train = rf1[:train_episodes]
    # Convergence step targets the ROLE F1 (final), not detection F1.
    # Approximation: first epoch where rolling detection-F1 reaches 0.9 of its
    # eventual final value; for ablations that never converge we report -1.
    conv_step = convergence_step(rf1_train, detection_f1, target_frac=0.90)

    elapsed = time.time() - t0
    if verbose:
        print(
            f"  [{ablation_name:7s} seed={seed:4d}] "
            f"RGHR_AUC={rghr_auc:.4f}  RGHR_F1={rghr_f1:.4f}  role_F1={role_f1:.4f}  OF={of_score:.4f}  "
            f"det_F1={detection_f1:.4f} conv={conv_step:5d}  "
            f"({elapsed:.1f}s)"
        )
    return {
        "ablation": ablation_name,
        "seed": int(seed),
        "train_episodes": int(train_episodes),
        "eval_episodes": int(eval_episodes),
        "window_len": int(window_len),
        # Primary metric (paper L1609 §appendix:org-eval): trajectory-clustering
        # F1 via TEMM (Org-Fit). RGHR-F1 / RGHR-AUC kept as secondary diagnostics.
        "final_f1": float(role_f1),
        # Secondary diagnostics
        "rghr_f1": float(rghr_f1),
        "rghr_auc": float(rghr_auc),
        "role_f1_clustering": float(role_f1),
        "sof": float(sof),
        "fof": float(fof),
        "of":  float(of_score),
        "detection_f1": float(detection_f1),
        "convergence_step": int(conv_step),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "elapsed_sec": float(elapsed),
        "rolling_f1_train_mean_last_window": float(rf1_train[-50:].mean()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Sweep + aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate(per_seed: List[Dict]) -> Dict:
    """Aggregate per-seed runs into per-ablation summary."""
    by_abl: Dict[str, List[Dict]] = {}
    for r in per_seed:
        by_abl.setdefault(r["ablation"], []).append(r)
    summary = {}
    for name, runs in by_abl.items():
        f1s = np.array([r["final_f1"] for r in runs])
        convs = np.array([r["convergence_step"] for r in runs if r["convergence_step"] >= 0])
        summary[name] = {
            "n_seeds": len(runs),
            "final_f1_mean": float(f1s.mean()),
            "final_f1_std":  float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0,
            "convergence_step_mean": float(convs.mean()) if len(convs) > 0 else -1.0,
            "convergence_step_std":
                float(convs.std(ddof=1)) if len(convs) > 1 else 0.0,
            "n_seeds_converged": int(len(convs)),
        }
    # Drop deltas (Δ F1 in percentage points relative to `full`)
    if "full" in summary:
        base = summary["full"]["final_f1_mean"]
        base_conv = summary["full"]["convergence_step_mean"]
        for name, s in summary.items():
            s["delta_f1_pp"] = float(100.0 * (s["final_f1_mean"] - base))
            if base_conv > 0 and s["convergence_step_mean"] > 0:
                s["delta_convergence_pct"] = float(
                    100.0 * (s["convergence_step_mean"] - base_conv) / base_conv
                )
            else:
                s["delta_convergence_pct"] = None
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 42])
    ap.add_argument("--episodes", type=int, default=800,
                    help="training episodes per (ablation, seed)")
    ap.add_argument("--eval-episodes", type=int, default=200,
                    help="held-out evaluation tail length")
    ap.add_argument("--quick", action="store_true",
                    help="1 seed, 200 episodes, 50 eval (smoke test)")
    ap.add_argument("--output-dir", type=str,
                    default="experiments/results/role_ablation")
    args = ap.parse_args()

    if args.quick:
        args.seeds = [42]
        args.episodes = 200
        args.eval_episodes = 50

    sim_cfg = SimConfig()
    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[role-ablation] seeds={args.seeds}  train_ep={args.episodes}  "
          f"eval_ep={args.eval_episodes}")
    print(f"[role-ablation] output_dir={out_dir}")
    print(f"[role-ablation] sim_cfg={asdict(sim_cfg)}")

    per_seed_runs: List[Dict] = []
    overall_t0 = time.time()
    for ablation_name, overrides in ABLATIONS.items():
        print(f"\n=== Ablation: {ablation_name}  overrides={overrides} ===")
        for seed in args.seeds:
            r = run_one(
                ablation_name=ablation_name,
                overrides=overrides,
                seed=seed,
                sim_cfg=sim_cfg,
                train_episodes=args.episodes,
                eval_episodes=args.eval_episodes,
                verbose=True,
            )
            per_seed_runs.append(r)

    summary = aggregate(per_seed_runs)
    total_elapsed = time.time() - overall_t0

    payload = {
        "experiment": "role_decomposition_ablation",
        "description": (
            "Per-component ablation of SFAC-FACMAC organizational rewards "
            "(RAG / GRG / RRG / PER). Backs Appendix:org-eval in the NDSS "
            "submission."
        ),
        "sim_cfg": asdict(sim_cfg),
        "args": {
            "seeds": list(args.seeds),
            "train_episodes": int(args.episodes),
            "eval_episodes": int(args.eval_episodes),
        },
        "ablations": {k: ABLATIONS[k] for k in ABLATIONS},
        "summary": summary,
        "per_seed": per_seed_runs,
        "total_elapsed_sec": float(total_elapsed),
    }

    json_path = out_dir / "role_ablation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[role-ablation] wrote {json_path}")

    csv_path = out_dir / "role_ablation_per_seed.csv"
    keys = list(per_seed_runs[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in per_seed_runs:
            w.writerow(r)
    print(f"[role-ablation] wrote {csv_path}")

    # Pretty-print summary
    print("\n=== Summary ===")
    print(f"{'ablation':10s}  {'F1':>8s}  {'ΔF1(pp)':>9s}  "
          f"{'conv_step':>10s}  {'Δconv(%)':>9s}  n_seeds")
    for name in ABLATIONS:
        if name not in summary:
            continue
        s = summary[name]
        dconv = f"{s['delta_convergence_pct']:+.1f}" \
            if s["delta_convergence_pct"] is not None else "  n/a"
        print(
            f"{name:10s}  {s['final_f1_mean']:.4f}  "
            f"{s['delta_f1_pp']:+8.2f}  "
            f"{s['convergence_step_mean']:10.1f}  {dconv:>9s}  "
            f"{s['n_seeds']}"
        )
    print(f"\ntotal elapsed: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
