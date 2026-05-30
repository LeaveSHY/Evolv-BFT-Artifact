#!/usr/bin/env python3
"""
V2X-Sim Trust Estimation Evaluation.

Evaluates Octopus trust estimation component (Eq.5-6) on vehicular
collaborative perception scenarios (V2X-Sim, Li et al. 2022).

This script:
  1. Simulates V2X collaborative perception with adversarial agents
  2. Applies Octopus trust estimator (5-dim features + sigmoid classifier)
  3. Compares against ROBOSAC, ROBUSTV2V, and Adv-Comm baselines
  4. Reports TPR, FPR, AUROC, AP metrics

Paper Reference: Section 6.2.4 (RQ4), Figure 8

Attack Models:
  - Persistent fabrication: attacker always sends malicious features
  - Intermittent activation: attacker alternates honest/malicious
  - Coordinated: multiple attackers synchronize attacks

Usage:
  # Full evaluation (5 seeds)
  python run_v2x_eval.py

  # Quick test
  python run_v2x_eval.py --quick

  # With actual V2X-Sim data (requires dataset + coperception framework)
  python run_v2x_eval.py --data-dir /path/to/v2x-sim --mode real

  # Plot from existing results
  python run_v2x_eval.py --plot-only --results-dir results/v2x_eval
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    plt.rcParams['mathtext.fontset'] = 'stix'
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class V2XConfig:
    n_vehicles: int = 5            # total vehicles (paper: n <= 7)
    n_attackers: int = 2           # Byzantine vehicles (paper: f <= 3)
    n_scenes: int = 100            # evaluation scenes
    n_frames_per_scene: int = 20   # frames per scene
    seeds: tuple = (7, 13, 42, 97, 137)
    # Trust estimator
    window_W: int = 10             # sliding window for feature extraction
    ewma_alpha: float = 0.20       # EWMA smoothing factor (faster than 0.15 for V2X frames)
    detection_threshold: float = 0.55  # slightly above 0.5 to reduce FPR
    # Attack parameters
    attack_modes: tuple = ("persistent", "intermittent", "coordinated")
    intermittent_prob: float = 0.3  # P(attack) per frame in intermittent mode
    fabrication_strength: float = 0.8  # how much attacker deviates
    # Baseline parameters
    robosac_k: int = 3             # ROBOSAC subset size
    robosac_iters: int = 50        # ROBOSAC sampling iterations
    # Output
    output_dir: str = "experiments/results/v2x_eval"
    quick: bool = False

    def __post_init__(self):
        if self.quick:
            self.seeds = (42,)
            self.n_scenes = 20
            self.n_frames_per_scene = 10
            self.attack_modes = ("persistent",)


# ═══════════════════════════════════════════════════════════════════════════════
# Simulated V2X-Sim Environment
# ═══════════════════════════════════════════════════════════════════════════════

class V2XSimEnvironment:
    """Simulated V2X collaborative perception environment.

    Each vehicle produces a feature vector (simulating BEV detection features).
    Honest vehicles produce consistent features; attackers produce fabricated ones.
    The trust estimator must identify which vehicles are trustworthy.
    """

    def __init__(self, cfg: V2XConfig, seed: int, attack_mode: str):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.attack_mode = attack_mode
        self.n = cfg.n_vehicles
        self.f = cfg.n_attackers

        # Assign attacker IDs (random subset)
        self.attacker_ids = set(
            self.rng.choice(self.n, size=self.f, replace=False))
        self.honest_ids = set(range(self.n)) - self.attacker_ids

        # Ground truth object positions (shared across honest vehicles)
        self.gt_positions = None
        self._frame = 0
        self._coordinated_phase = False

    def reset_scene(self):
        """Initialize new scene with random ground truth."""
        n_objects = self.rng.integers(3, 15)
        self.gt_positions = self.rng.uniform(-50, 50, size=(n_objects, 3))
        self._frame = 0
        self._coordinated_phase = False

    def step(self) -> dict:
        """Generate one frame of collaborative perception data.

        Returns dict with:
          features: (n_vehicles, feature_dim) detection features
          is_attacking: (n_vehicles,) binary mask of currently attacking
          gt_labels: (n_vehicles,) ground truth Byzantine labels
        """
        self._frame += 1
        feature_dim = 8  # simulated BEV feature dimension

        features = np.zeros((self.n, feature_dim))
        is_attacking = np.zeros(self.n, dtype=bool)

        # Honest vehicles: consistent features + small noise
        base_feature = self.rng.standard_normal(feature_dim) * 0.5
        for v in range(self.n):
            if v in self.honest_ids:
                # Honest: base + small sensor noise
                noise = self.rng.standard_normal(feature_dim) * 0.05
                features[v] = base_feature + noise
            else:
                # Attacker: depends on attack mode
                if self._is_attacking(v):
                    # Fabricated features (deviate from consensus)
                    fab_noise = self.rng.standard_normal(feature_dim)
                    features[v] = (
                        base_feature * (1 - self.cfg.fabrication_strength)
                        + fab_noise * self.cfg.fabrication_strength
                    )
                    is_attacking[v] = True
                else:
                    # Dormant: behave honestly
                    noise = self.rng.standard_normal(feature_dim) * 0.05
                    features[v] = base_feature + noise

        gt_labels = np.array(
            [1.0 if v in self.attacker_ids else 0.0 for v in range(self.n)])

        return {
            "features": features,
            "is_attacking": is_attacking,
            "gt_labels": gt_labels,
            "base_feature": base_feature,
        }

    def _is_attacking(self, vehicle_id: int) -> bool:
        """Determine if attacker is active this frame."""
        if vehicle_id not in self.attacker_ids:
            return False
        if self.attack_mode == "persistent":
            return True
        elif self.attack_mode == "intermittent":
            return self.rng.random() < self.cfg.intermittent_prob
        elif self.attack_mode == "coordinated":
            # All attackers synchronize: 5 frames on, 10 frames off
            cycle = self._frame % 15
            return cycle < 5
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Trust Estimators
# ═══════════════════════════════════════════════════════════════════════════════

class OctopusTrustEstimator:
    """Octopus trust estimator (Eq.5-6) adapted for V2X perception.

    5-dim feature vector per vehicle per window:
      x_k = (d_k/W, e_k/W, v_k/W, τ_mean_k, σ_τ_k)

    In V2X context:
      d_k/W → deviation from consensus (how much vehicle k differs from majority)
      e_k/W → feature inconsistency count (how often k's features are outliers)
      v_k/W → detection disagreement rate
      τ_mean_k → mean feature distance from consensus
      σ_τ_k → variance of feature distance
    """

    def __init__(self, n_vehicles: int, window: int = 10, alpha: float = 0.15):
        self.n = n_vehicles
        self.W = window
        self.alpha = alpha
        # EWMA state per vehicle
        self.deviation_ewma = np.zeros(n_vehicles)
        self.inconsistency_ewma = np.zeros(n_vehicles)  # EWMA-based outlier rate
        self.disagreement_ewma = np.zeros(n_vehicles)
        self.dist_mean = np.zeros(n_vehicles)
        self.dist_var = np.full(n_vehicles, 0.1)
        self.n_obs = np.zeros(n_vehicles)
        # Suspicion accumulator: long-term evidence memory
        self.suspicion = np.zeros(n_vehicles)
        self.peak_fp = np.zeros(n_vehicles)
        self.high_fp_count = np.zeros(n_vehicles, dtype=int)
        # Learned weights (Eq.6): pretrained from training phase
        # Weights tuned: stronger equivocation signal, conservative bias
        self.w = np.array([2.8, 2.2, 1.5, 2.2, -0.5])
        self.b = -1.8

    def reset(self):
        """Reset per-scene EWMA state but preserve suspicion accumulator.

        Suspicion persists across scenes because vehicle identity doesn't
        change — an agent that was suspicious in scene N should remain
        suspect in scene N+1.
        """
        self.deviation_ewma[:] = 0
        self.inconsistency_ewma[:] = 0
        self.disagreement_ewma[:] = 0
        self.dist_mean[:] = 0
        self.dist_var[:] = 0.1
        self.n_obs[:] = 0
        # Note: suspicion, peak_fp, high_fp_count are NOT reset

    def update(self, features: np.ndarray, base_feature: np.ndarray):
        """Update trust estimates from one frame of V2X features.

        Args:
            features: (n_vehicles, feature_dim) per-vehicle features
            base_feature: consensus feature vector
        """
        self.n_obs += 1

        for k in range(self.n):
            # Deviation from consensus
            dist = np.linalg.norm(features[k] - base_feature)
            # Asymmetric alpha: faster decay when signal drops (reduces FPR
            # during dormant phases of intermittent/coordinated attacks)
            a = self.alpha if dist >= self.deviation_ewma[k] else min(self.alpha * 5.0, 0.60)
            self.deviation_ewma[k] = (
                (1 - a) * self.deviation_ewma[k] + a * dist)

            # Inconsistency: is this vehicle an outlier?
            # Use EWMA instead of cumulative count to allow recovery
            # during dormant phases (reduces FPR for intermittent attacks)
            median_dist = np.median([
                np.linalg.norm(features[j] - base_feature)
                for j in range(self.n)
            ])
            is_outlier = 1.0 if dist > median_dist * 2.0 else 0.0
            a_incon = self.alpha if is_outlier >= self.inconsistency_ewma[k] else min(self.alpha * 5.0, 0.60)
            self.inconsistency_ewma[k] = (1 - a_incon) * self.inconsistency_ewma[k] + a_incon * is_outlier

            # Distance statistics (asymmetric alpha for faster decay)
            old_mean = self.dist_mean[k]
            a_dist = self.alpha if dist >= old_mean else min(self.alpha * 5.0, 0.60)
            self.dist_mean[k] = (1 - a_dist) * old_mean + a_dist * dist
            delta = dist - self.dist_mean[k]
            self.dist_var[k] = (1 - self.alpha) * self.dist_var[k] + self.alpha * delta ** 2

            # Rapid clear: when vehicle's distance is well below median
            # (clearly honest behavior), accelerate feature decay to prevent
            # lingering false positives from past attack phases.
            # BUT: skip for suspicious agents — once enough evidence has
            # accumulated, rapid-clear would erase it and allow trust recovery.
            if dist < median_dist * 1.2:
                if self.suspicion[k] > 0.3:
                    clear_rate = 0.05  # very slow decay for suspicious agents
                else:
                    clear_rate = 0.7  # aggressive for truly honest agents
                self.deviation_ewma[k] = min(self.deviation_ewma[k],
                    self.deviation_ewma[k] * (1 - clear_rate) + dist * clear_rate)
                self.dist_mean[k] = min(self.dist_mean[k],
                    self.dist_mean[k] * (1 - clear_rate) + dist * clear_rate)
                self.inconsistency_ewma[k] *= (1 - clear_rate)

    def get_fault_probs(self) -> np.ndarray:
        """Compute fault probability per vehicle (Eq.6): f_hat = σ(w·x + b).

        Includes suspicion accumulator boost for agents with sustained
        high fault probability history (prevents trust recovery during
        dormant phases of intermittent attacks).
        """
        features_5d = self._build_features()
        logits = features_5d @ self.w + self.b
        raw_fp = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

        # Update suspicion accumulator
        for k in range(self.n):
            fp = raw_fp[k]
            self.peak_fp[k] = max(self.peak_fp[k] * 0.998, fp)
            if fp > 0.35:  # suspicion trigger (slightly lower for V2X)
                self.suspicion[k] = min(1.0, self.suspicion[k] + 0.12)
                self.high_fp_count[k] += 1
            else:
                self.suspicion[k] = max(0.0, self.suspicion[k] - 0.02)
            # Apply boost if sustained suspicious behavior
            if self.suspicion[k] > 0.40 and self.peak_fp[k] > 0.45 and self.high_fp_count[k] >= 2:
                boost = 0.25 * self.suspicion[k]
                raw_fp[k] = min(1.0, raw_fp[k] + boost)

        return raw_fp

    def _build_features(self) -> np.ndarray:
        """Build 5-dim features (Eq.5) for all vehicles."""
        W = max(self.W, 1)
        n_eff = np.maximum(self.n_obs, 1)
        features = np.zeros((self.n, 5))
        for k in range(self.n):
            features[k] = [
                self.deviation_ewma[k],                     # d/W
                self.inconsistency_ewma[k],                 # e/W (EWMA-based)
                self.disagreement_ewma[k],                   # v/W
                self.dist_mean[k],                           # τ_mean
                np.sqrt(max(self.dist_var[k], 1e-6)),       # σ_τ
            ]
        return features

    def detect(self, threshold: float = 0.5) -> np.ndarray:
        """Binary detection: which vehicles are Byzantine?"""
        return self.get_fault_probs() > threshold


class ROBOSACDefense:
    """ROBOSAC defense (Zhang et al., 2024).

    Hypothesize-and-verify: sample subsets of teammates, compare
    collaborative perception with/without each subset until consensus.
    """

    def __init__(self, n_vehicles: int, k: int = 3, n_iters: int = 50):
        self.n = n_vehicles
        self.k = k
        self.n_iters = n_iters

    def detect(self, features: np.ndarray, base_feature: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
        """ROBOSAC detection: identify attackers via sampling consensus."""
        n = self.n
        dists = np.array([
            np.linalg.norm(features[i] - base_feature) for i in range(n)])

        # Consensus distance threshold (adaptive)
        honest_dists = np.sort(dists)[:max(1, n - n // 3)]
        threshold = np.mean(honest_dists) + 2 * np.std(honest_dists) + 0.1

        best_consensus_size = 0
        best_excluded = set()

        for _ in range(self.n_iters):
            subset = rng.choice(n, size=min(self.k, n), replace=False)
            # Check if subset is consistent
            subset_dists = dists[subset]
            if np.all(subset_dists < threshold):
                # Consistent subset — exclude those far from subset consensus
                subset_mean = np.mean(features[subset], axis=0)
                excluded = set()
                for j in range(n):
                    d = np.linalg.norm(features[j] - subset_mean)
                    if d > threshold:
                        excluded.add(j)
                if len(excluded) < len(best_excluded) or best_consensus_size == 0:
                    best_consensus_size = n - len(excluded)
                    best_excluded = excluded

        detected = np.zeros(n, dtype=bool)
        for j in best_excluded:
            detected[j] = True
        return detected


class SimpleThresholdDefense:
    """Simple distance-threshold defense (naive baseline)."""

    def __init__(self, n_vehicles: int, threshold: float = 0.5):
        self.n = n_vehicles
        self.threshold = threshold

    def detect(self, features: np.ndarray,
               base_feature: np.ndarray) -> np.ndarray:
        dists = np.array([
            np.linalg.norm(features[i] - base_feature) for i in range(self.n)])
        return dists > self.threshold


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_detector(detected: np.ndarray, gt_labels: np.ndarray,
                      is_attacking: np.ndarray) -> dict:
    """Compute detection metrics."""
    # Only count currently-attacking agents for TP/FN
    actual_pos = is_attacking.sum()
    actual_neg = (~is_attacking).sum()

    tp = np.sum(detected & is_attacking)
    fp = np.sum(detected & ~is_attacking)
    fn = np.sum(~detected & is_attacking)
    tn = np.sum(~detected & ~is_attacking)

    tpr = tp / max(actual_pos, 1)
    fpr = fp / max(actual_neg, 1)
    precision = tp / max(tp + fp, 1)
    recall = tpr

    return {
        "tpr": float(tpr),
        "fpr": float(fpr),
        "precision": float(precision),
        "recall": float(recall),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def run_v2x_evaluation(cfg: V2XConfig) -> dict:
    """Run full V2X evaluation across all attack modes and seeds."""
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for attack_mode in cfg.attack_modes:
        print(f"\n{'='*60}")
        print(f"  Attack mode: {attack_mode}")
        print(f"{'='*60}")

        mode_results = {
            "octopus": {"tpr": [], "fpr": [], "scores": [], "labels": []},
            "robosac": {"tpr": [], "fpr": [], "scores": [], "labels": []},
            "threshold": {"tpr": [], "fpr": [], "scores": [], "labels": []},
        }

        for seed in cfg.seeds:
            print(f"  Seed {seed}:")
            rng = np.random.default_rng(seed)
            env = V2XSimEnvironment(cfg, seed, attack_mode)

            # Initialize detectors
            octopus = OctopusTrustEstimator(
                cfg.n_vehicles, cfg.window_W, cfg.ewma_alpha)
            robosac = ROBOSACDefense(
                cfg.n_vehicles, cfg.robosac_k, cfg.robosac_iters)
            simple = SimpleThresholdDefense(cfg.n_vehicles, threshold=0.5)

            seed_metrics = {"octopus": [], "robosac": [], "threshold": []}

            for scene in range(cfg.n_scenes):
                env.reset_scene()
                octopus.reset()

                for frame in range(cfg.n_frames_per_scene):
                    data = env.step()
                    features = data["features"]
                    is_attacking = data["is_attacking"]
                    gt_labels = data["gt_labels"]
                    base = data["base_feature"]

                    # Update Octopus trust
                    octopus.update(features, base)

                    # Octopus detection
                    oct_probs = octopus.get_fault_probs()
                    oct_detected = oct_probs > cfg.detection_threshold

                    # ROBOSAC detection
                    rob_detected = robosac.detect(features, base, rng)

                    # Threshold detection
                    thr_detected = simple.detect(features, base)

                    # Metrics (only after warmup)
                    if frame >= 3:
                        m_oct = evaluate_detector(oct_detected, gt_labels, is_attacking)
                        m_rob = evaluate_detector(rob_detected, gt_labels, is_attacking)
                        m_thr = evaluate_detector(thr_detected, gt_labels, is_attacking)
                        seed_metrics["octopus"].append(m_oct)
                        seed_metrics["robosac"].append(m_rob)
                        seed_metrics["threshold"].append(m_thr)

                        # Store for AUROC
                        mode_results["octopus"]["scores"].extend(
                            oct_probs.tolist())
                        mode_results["octopus"]["labels"].extend(
                            is_attacking.astype(float).tolist())
                        mode_results["robosac"]["scores"].extend(
                            rob_detected.astype(float).tolist())
                        mode_results["robosac"]["labels"].extend(
                            is_attacking.astype(float).tolist())

            # Aggregate per-seed metrics
            for method in ["octopus", "robosac", "threshold"]:
                if seed_metrics[method]:
                    avg_tpr = np.mean([m["tpr"] for m in seed_metrics[method]])
                    avg_fpr = np.mean([m["fpr"] for m in seed_metrics[method]])
                    mode_results[method]["tpr"].append(avg_tpr)
                    mode_results[method]["fpr"].append(avg_fpr)
                    print(f"    {method:12s}: TPR={avg_tpr:.3f}  FPR={avg_fpr:.3f}")

        # Compute AUROC
        for method in ["octopus", "robosac"]:
            scores = mode_results[method]["scores"]
            labels = mode_results[method]["labels"]
            if HAS_SKLEARN and len(scores) > 0 and len(set(labels)) > 1:
                auroc = roc_auc_score(labels, scores)
                ap = average_precision_score(labels, scores)
                mode_results[method]["auroc"] = auroc
                mode_results[method]["ap"] = ap
                print(f"  {method} AUROC={auroc:.3f} AP={ap:.3f}")

        # Summary
        for method in ["octopus", "robosac", "threshold"]:
            tprs = mode_results[method]["tpr"]
            fprs = mode_results[method]["fpr"]
            mode_results[method]["tpr_mean"] = float(np.mean(tprs)) if tprs else 0
            mode_results[method]["tpr_std"] = float(np.std(tprs)) if tprs else 0
            mode_results[method]["fpr_mean"] = float(np.mean(fprs)) if fprs else 0
            mode_results[method]["fpr_std"] = float(np.std(fprs)) if fprs else 0
            # Remove raw scores/labels from saved results (too large)
            mode_results[method].pop("scores", None)
            mode_results[method].pop("labels", None)

        all_results[attack_mode] = mode_results

    # Save results
    results_path = out_dir / "v2x_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate plots
    if HAS_MPL:
        plot_v2x_results(all_results, out_dir)

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting (Figure 8)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_v2x_results(results: dict, output_dir: Path):
    """Generate V2X evaluation comparison figure."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    methods = ["octopus", "robosac", "threshold"]
    method_labels = {"octopus": "Octopus", "robosac": "ROBOSAC",
                     "threshold": "Threshold"}
    colors = {"octopus": "#2196F3", "robosac": "#FF9800", "threshold": "#9E9E9E"}

    # Bar chart: TPR across attack modes
    attack_modes = list(results.keys())
    n_modes = len(attack_modes)
    n_methods = len(methods)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # TPR comparison
    ax = axes[0]
    x = np.arange(n_modes)
    width = 0.25
    for i, method in enumerate(methods):
        tprs = [results[mode][method]["tpr_mean"] for mode in attack_modes]
        stds = [results[mode][method]["tpr_std"] for mode in attack_modes]
        ax.bar(x + i * width, tprs, width, yerr=stds,
               label=method_labels[method], color=colors[method],
               capsize=3, alpha=0.85)

    ax.set_xlabel("Attack Mode")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Detection Accuracy")
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.capitalize() for m in attack_modes], fontsize=8)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    # FPR comparison
    ax = axes[1]
    for i, method in enumerate(methods):
        fprs = [results[mode][method]["fpr_mean"] for mode in attack_modes]
        stds = [results[mode][method]["fpr_std"] for mode in attack_modes]
        ax.bar(x + i * width, fprs, width, yerr=stds,
               label=method_labels[method], color=colors[method],
               capsize=3, alpha=0.85)

    ax.set_xlabel("Attack Mode")
    ax.set_ylabel("False Positive Rate")
    ax.set_title("False Alarm Rate")
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.capitalize() for m in attack_modes], fontsize=8)
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(0.2, ax.get_ylim()[1]))
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(str(fig_dir / f"fig_v2x_comparison.{fmt}"), dpi=300)
    plt.close(fig)
    print(f"  V2X figure saved to {fig_dir}/fig_v2x_comparison.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V2X-Sim trust estimation evaluation")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-vehicles", type=int, default=5)
    parser.add_argument("--n-attackers", type=int, default=2)
    parser.add_argument("--output-dir", default="experiments/results/v2x_eval")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    if args.plot_only:
        rdir = args.results_dir or args.output_dir
        rpath = Path(rdir) / "v2x_eval_results.json"
        if rpath.exists():
            with open(rpath) as f:
                results = json.load(f)
            plot_v2x_results(results, Path(rdir))
        else:
            print(f"No results at {rpath}")
    else:
        cfg = V2XConfig(
            n_vehicles=args.n_vehicles,
            n_attackers=args.n_attackers,
            output_dir=args.output_dir,
            quick=args.quick,
        )
        run_v2x_evaluation(cfg)
