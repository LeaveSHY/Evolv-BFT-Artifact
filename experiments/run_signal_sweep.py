#!/usr/bin/env python3
"""Signal sensitivity sweep: vary Byzantine signal strength.

Tests Octopus + UCB (best baseline) across signal_byz_active in {0.15, 0.20, 0.30, 0.40, 0.50, 0.60}.
Uses 3 seeds for speed. Reports D(T), detection_latency, safety_violations.
"""

import sys
import os
import json
import time
import numpy as np
from dataclasses import dataclass, replace

# Add parent path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_e2e_experiments import (
    E2EConfig, run_e2e,
    CUSUMController, GossipThresholdController,
    CentralizedUCBController, OctopusMARLController,
)

SIGNAL_LEVELS = [0.15, 0.20, 0.30, 0.40, 0.50, 0.60]
SEEDS = (7, 42, 97)
CONTROLLERS = ["cusum", "ucb", "octopus"]


def make_controller(name, cfg):
    """Instantiate controller by name."""
    if name == "cusum":
        return CUSUMController()
    elif name == "ucb":
        return CentralizedUCBController()
    elif name == "octopus":
        return OctopusMARLController(lr=cfg.lr)
    else:
        raise ValueError(f"Unknown controller: {name}")


def main():
    output_dir = "experiments/results/signal_sweep"
    os.makedirs(output_dir, exist_ok=True)

    results = {}  # signal_level -> {controller -> {metric: value}}

    print("=" * 60)
    print("  Signal Sensitivity Sweep")
    print("  Controllers:", CONTROLLERS)
    print("  Signal levels:", SIGNAL_LEVELS)
    print("  Seeds:", SEEDS)
    print("=" * 60)

    t0 = time.time()

    for sig in SIGNAL_LEVELS:
        print(f"\n{'=' * 60}")
        print(f"  Signal: {sig:.2f}")
        print(f"{'=' * 60}")

        cfg = E2EConfig(
            signal_byz_active=sig,
            seeds=SEEDS,
            T_epochs=500,
            train_episodes=500,
            episode_length=200,
            output_dir=output_dir,
        )

        results[str(sig)] = {}

        for ctrl_name in CONTROLLERS:
            print(f"\n  Controller: {ctrl_name}")
            seed_results = []
            train_first = ctrl_name == "octopus"

            for s in SEEDS:
                ctrl = make_controller(ctrl_name, cfg)
                result = run_e2e(cfg, ctrl, s, train_first=train_first)
                seed_results.append(result)
                print(f"    seed={s}: D(T)={result['D_T']:,}  "
                      f"lat={result['detection_latency']:.1f}  "
                      f"viol={result['safety_violations']}")

            # Compute mean/std
            d_ts = [r['D_T'] for r in seed_results]
            lats = [r['detection_latency'] for r in seed_results]
            viols = [r['safety_violations'] for r in seed_results]

            results[str(sig)][ctrl_name] = {
                "D_T_mean": float(np.mean(d_ts)),
                "D_T_std": float(np.std(d_ts)),
                "D_T_seeds": d_ts,
                "latency_mean": float(np.mean(lats)),
                "latency_std": float(np.std(lats)),
                "violations_mean": float(np.mean(viols)),
                "violations_std": float(np.std(viols)),
            }

    elapsed = time.time() - t0

    # Save results
    out_path = os.path.join(output_dir, "signal_sweep_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {out_path}")

    # Print summary table
    print(f"\n{'=' * 70}")
    print(f"  Signal Sensitivity Summary (T=500, {len(SEEDS)} seeds)")
    print(f"{'=' * 70}")
    print(f"{'Signal':>8} | {'Controller':>12} | {'D(T)':>12} | {'Latency':>10} | {'Violations':>10}")
    print("-" * 70)
    for sig in SIGNAL_LEVELS:
        for ctrl_name in CONTROLLERS:
            r = results[str(sig)][ctrl_name]
            d = f"{r['D_T_mean']:.0f}±{r['D_T_std']:.0f}"
            l = f"{r['latency_mean']:.1f}±{r['latency_std']:.1f}"
            v = f"{r['violations_mean']:.0f}±{r['violations_std']:.0f}"
            print(f"{sig:>8.2f} | {ctrl_name:>12} | {d:>12} | {l:>10} | {v:>10}")
        print("-" * 70)

    print(f"\n  Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
