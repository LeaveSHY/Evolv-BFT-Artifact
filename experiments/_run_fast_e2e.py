"""Run full E2E evaluation with spike detection, minimal training.

This uses a reduced training budget (500 episodes) to verify that
the spike detection dominates D(T) improvement regardless of
MARL convergence quality.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Monkey-patch: reduce training to 500 episodes for fast verification
import run_e2e_experiments as e2e
e2e.E2EConfig.train_episodes = 500

if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--output-dir", "experiments/results/e2e_post_opt",
        "--seeds", "7", "13", "42", "97", "137",
    ]
    e2e.main()
