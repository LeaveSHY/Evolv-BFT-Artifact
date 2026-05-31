#!/usr/bin/env python3
"""Generate figures for NDSS 2027 experiment improvements (P0-P4).

Produces publication-quality figures (Times New Roman, 600 DPI, Nature-style
palette) for:
  P0: V2X ε sweep — defense robustness across attack budgets
  P1: PPO baseline — MARL vs single-agent RL comparison
  P3: V2X extended — multi-scene/multi-seed statistical analysis
  P4: HotStuff baseline — single-instance vs multi-instance throughput

Usage:
  python gen_improvement_figures.py --results-dir experiments/results/improvements_XXXXXXXX
  python gen_improvement_figures.py --p1-dir ... --p0-dir ... (individual dirs)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

# Times New Roman + Nature-style palette
try:
    import register_fonts; register_fonts.register()
except ImportError:
    pass
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8

# Nature-inspired color palette
COLORS = {
    'evolvbft': '#1f77b4',       # blue
    'ppo': '#9467bd',           # purple
    'cusum': '#d62728',         # red
    'gossip_thresh': '#ff7f0e', # orange
    'exp3': '#2ca02c',          # green
    'ucb': '#8c564b',           # brown
    'robosac': '#e377c2',       # pink
    'mate': '#7f7f7f',          # gray
    'advcp': '#bcbd22',         # olive
    'single_bft': '#17becf',    # cyan
}

LABELS = {
    'evolvbft': 'Evolv-BFT (MARL-CTDE)',
    'ppo': 'Single-Agent PPO',
    'cusum': 'CUSUM',
    'gossip_thresh': 'Gossip+Threshold',
    'exp3': 'EXP3+Safety',
    'ucb': 'Centralized UCB',
    'robosac': 'ROBOSAC',
    'mate': 'MATE',
    'advcp': 'AdvCP',
    'single_bft': 'Single-Instance BFT',
}

DPI = 600


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# P1: PPO Baseline Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def gen_p1_figure(p1_dir: str, output_dir: str):
    """Generate PPO vs Evolv-BFT comparison bar chart."""
    results_path = Path(p1_dir) / "e2e_summary.json"
    if not results_path.exists():
        # Try e2e_results.json
        results_path = Path(p1_dir) / "e2e_results.json"
    if not results_path.exists():
        print(f"[P1] No results found in {p1_dir}")
        return

    data = load_json(str(results_path))

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))

    controllers = []
    d_t_means = []
    d_t_stds = []
    latencies = []
    safety_viols = []

    for name in ['cusum', 'gossip_thresh', 'exp3', 'ucb', 'ppo', 'evolvbft']:
        if name not in data:
            continue
        runs = data[name]
        if isinstance(runs, dict):
            # Summary format
            d_t_means.append(runs.get('D_T_mean', 0))
            d_t_stds.append(runs.get('D_T_std', 0))
            latencies.append(runs.get('detection_latency_mean', 0))
            safety_viols.append(runs.get('safety_violations_mean', 0))
        else:
            # Raw results format
            dts = [r['D_T'] for r in runs]
            d_t_means.append(np.mean(dts))
            d_t_stds.append(np.std(dts))
            latencies.append(np.mean([r['detection_latency'] for r in runs]))
            safety_viols.append(np.mean([r['safety_violations'] for r in runs]))
        controllers.append(name)

    x = np.arange(len(controllers))
    colors = [COLORS.get(c, '#333333') for c in controllers]
    labels = [LABELS.get(c, c) for c in controllers]

    # (a) Cumulative damage
    ax = axes[0]
    bars = ax.bar(x, d_t_means, yerr=d_t_stds, color=colors, capsize=3, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Cumulative Damage $D(T)$', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_title('(a) Cumulative Damage', fontsize=10)
    ax.set_yscale('log')

    # (b) Detection latency
    ax = axes[1]
    ax.bar(x, latencies, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Detection Latency (epochs)', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_title('(b) Detection Latency', fontsize=10)

    # (c) Safety violations
    ax = axes[2]
    ax.bar(x, safety_viols, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Safety Violations', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_title('(c) Safety Violations', fontsize=10)

    plt.tight_layout()
    out_path = Path(output_dir) / "fig_p1_ppo_comparison.pdf"
    fig.savefig(str(out_path), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"[P1] Saved {out_path}")

    # Also save PNG preview
    fig2, axes2 = plt.subplots(1, 3, figsize=(10, 3.2))
    for i in range(3):
        axes2[i].bar(x, [d_t_means, latencies, safety_viols][i],
                     color=colors, edgecolor='black', linewidth=0.5)
        axes2[i].set_xticks(x)
        axes2[i].set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    plt.tight_layout()
    fig2.savefig(str(out_path.with_suffix('.png')), dpi=150, bbox_inches='tight')
    plt.close(fig2)


# ═══════════════════════════════════════════════════════════════════════════════
# P0: V2X ε Sweep
# ═══════════════════════════════════════════════════════════════════════════════

def gen_p0_figure(p0_dir: str, output_dir: str):
    """Generate ε sweep line plot showing defense robustness."""
    eps_values = []
    defense_data = {}  # defense -> {eps -> mAP}

    p0_path = Path(p0_dir)
    for eps_dir in sorted(p0_path.glob("eps_*")):
        eps = float(eps_dir.name.replace("eps_", ""))
        eps_values.append(eps)

        # Load results from each eps directory
        for json_file in eps_dir.glob("*.json"):
            data = load_json(str(json_file))
            if isinstance(data, list):
                for entry in data:
                    defense = entry.get('defense', entry.get('scenario', ''))
                    mAP = entry.get('mAP_05_mean', entry.get('mAP_05', 0))
                    if defense and mAP > 0:
                        if defense not in defense_data:
                            defense_data[defense] = {}
                        defense_data[defense][eps] = mAP

    if not defense_data:
        print(f"[P0] No ε sweep results found in {p0_dir}")
        return

    eps_values = sorted(set(eps_values))

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for defense, eps_map in defense_data.items():
        color = COLORS.get(defense, '#333333')
        label = LABELS.get(defense, defense)
        xs = sorted(eps_map.keys())
        ys = [eps_map[x] for x in xs]
        ax.plot(xs, ys, 'o-', color=color, label=label, markersize=4, linewidth=1.5)

    ax.set_xlabel('Attack Budget $\\varepsilon$', fontsize=10)
    ax.set_ylabel('mAP@0.5', fontsize=10)
    ax.set_title('Defense Robustness vs. Attack Strength', fontsize=11)
    ax.legend(fontsize=7, loc='lower left')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(output_dir) / "fig_p0_eps_sweep.pdf"
    fig.savefig(str(out_path), dpi=DPI, bbox_inches='tight')
    fig.savefig(str(out_path.with_suffix('.png')), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[P0] Saved {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# P3: V2X Extended Results
# ═══════════════════════════════════════════════════════════════════════════════

def gen_p3_figure(p3_dir: str, output_dir: str):
    """Generate multi-scene/multi-seed statistical comparison."""
    p3_path = Path(p3_dir)

    # Collect data across scenes and attack types
    scene_data = {}  # (scene, attack, defense) -> [mAP values across seeds]

    for scene_dir in sorted(p3_path.glob("scene*_*")):
        parts = scene_dir.name.split("_", 1)
        scene = parts[0].replace("scene", "")
        attack = parts[1] if len(parts) > 1 else "persistent"

        for json_file in scene_dir.glob("*.json"):
            data = load_json(str(json_file))
            if isinstance(data, list):
                for entry in data:
                    defense = entry.get('defense', '')
                    mAP = entry.get('mAP_05_mean', 0)
                    if defense and mAP > 0:
                        key = (scene, attack, defense)
                        if key not in scene_data:
                            scene_data[key] = []
                        scene_data[key].append(mAP)

    if not scene_data:
        print(f"[P3] No extended results found in {p3_dir}")
        return

    # Create grouped bar chart
    defenses = sorted(set(k[2] for k in scene_data.keys()))
    scenes = sorted(set(k[0] for k in scene_data.keys()))

    fig, axes = plt.subplots(1, len(scenes), figsize=(4 * len(scenes), 3.5), sharey=True)
    if len(scenes) == 1:
        axes = [axes]

    for idx, scene in enumerate(scenes):
        ax = axes[idx]
        x = np.arange(len(defenses))
        means = []
        stds = []
        colors = []
        for d in defenses:
            key = (scene, 'persistent', d)
            vals = scene_data.get(key, [0])
            means.append(np.mean(vals))
            stds.append(np.std(vals))
            colors.append(COLORS.get(d, '#333333'))

        ax.bar(x, means, yerr=stds, color=colors, capsize=3,
               edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS.get(d, d) for d in defenses],
                           rotation=45, ha='right', fontsize=7)
        ax.set_title(f'Scene {scene}', fontsize=10)
        if idx == 0:
            ax.set_ylabel('mAP@0.5', fontsize=10)

    plt.tight_layout()
    out_path = Path(output_dir) / "fig_p3_multiscene.pdf"
    fig.savefig(str(out_path), dpi=DPI, bbox_inches='tight')
    fig.savefig(str(out_path.with_suffix('.png')), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[P3] Saved {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Root results directory from run_all_improvements.sh")
    parser.add_argument("--p0-dir", type=str, default=None)
    parser.add_argument("--p1-dir", type=str, default=None)
    parser.add_argument("--p3-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for figures (default: NDSS figures/improvements/)")
    args = parser.parse_args()

    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "NDSS 2027_SUBMISSION", "figures", "improvements")
    os.makedirs(out_dir, exist_ok=True)

    root = args.results_dir

    p1_dir = args.p1_dir or (os.path.join(root, "p1_ppo_baseline") if root else None)
    p0_dir = args.p0_dir or (os.path.join(root, "p0_eps_sweep") if root else None)
    p3_dir = args.p3_dir or (os.path.join(root, "p3_v2x_extended") if root else None)

    if p1_dir and os.path.isdir(p1_dir):
        gen_p1_figure(p1_dir, out_dir)
    if p0_dir and os.path.isdir(p0_dir):
        gen_p0_figure(p0_dir, out_dir)
    if p3_dir and os.path.isdir(p3_dir):
        gen_p3_figure(p3_dir, out_dir)

    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()
