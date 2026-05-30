#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Unified Experiment Runner for NDSS 2027 Paper Improvements (P0-P4)
#
# Usage (on WSL):
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate robosac-gpu
#   cd /mnt/d/Alex/Papers/Experiment/Octopus
#   bash experiments/run_all_improvements.sh [--skip-p0] [--skip-p1] [--skip-p3] [--skip-p4]
#
# Prerequisites:
#   - conda env `robosac-gpu` with PyTorch, numpy, matplotlib
#   - V2X-Sim data at Collaboration-Benchmark/V2X-Sim-det/V2X-Sim-det
#   - Pre-trained checkpoint at Collaboration-Benchmark/epoch_49.pth
#   - Go toolchain for P4 (HotStuff benchmark)
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${ROOT_DIR}/experiments/results/improvements_${TIMESTAMP}"

# V2X paths
V2X_DIR="${ROOT_DIR}/Collaboration-Benchmark"
V2X_DATA="${V2X_DIR}/V2X-Sim-det/V2X-Sim-det"
V2X_CKPT="${V2X_DIR}/epoch_49.pth"
V2X_SCRIPT="${V2X_DIR}/coperception/tools/det/octopus_v2x_experiment.py"

# Parse flags
SKIP_P0=false
SKIP_P1=false
SKIP_P3=false
SKIP_P4=false
for arg in "$@"; do
    case $arg in
        --skip-p0) SKIP_P0=true ;;
        --skip-p1) SKIP_P1=true ;;
        --skip-p3) SKIP_P3=true ;;
        --skip-p4) SKIP_P4=true ;;
    esac
done

mkdir -p "$RESULTS_DIR"
echo "═══════════════════════════════════════════════════════════════"
echo "  NDSS 2027 Experiment Improvements (P0-P4)"
echo "  Timestamp: $TIMESTAMP"
echo "  Results:   $RESULTS_DIR"
echo "═══════════════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════════════════════
# P1: Single-Agent PPO Baseline (RQ3 E2E experiment)
# Purpose: validates multi-agent coordination necessity
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_P1" = false ]; then
    echo ""
    echo "━━━ P1: Single-Agent PPO Baseline ━━━"
    P1_DIR="${RESULTS_DIR}/p1_ppo_baseline"
    mkdir -p "$P1_DIR"

    cd "$ROOT_DIR"
    python experiments/run_e2e_experiments.py \
        --output-dir "$P1_DIR" \
        --controllers ppo 2>&1 | tee "${P1_DIR}/run.log"

    echo "[P1] PPO baseline complete. Results in $P1_DIR"
else
    echo "[SKIP] P1: PPO baseline"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# P0: V2X Baseline ε Sweep
# Purpose: show baselines degrade at high ε, Octopus is robust
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_P0" = false ]; then
    echo ""
    echo "━━━ P0: V2X ε Sweep ━━━"
    P0_DIR="${RESULTS_DIR}/p0_eps_sweep"
    mkdir -p "$P0_DIR"

    cd "$V2X_DIR"

    for EPS in 0.1 0.2 0.3 0.5 0.8 1.0; do
        echo ""
        echo "--- Running ε=$EPS ---"
        EPS_OUT="${P0_DIR}/eps_${EPS}"
        mkdir -p "$EPS_OUT"

        python "$V2X_SCRIPT" \
            --data "$V2X_DATA" \
            --resume "$V2X_CKPT" \
            --defense all \
            --all-scenarios \
            --eps "$EPS" \
            --seeds 7 42 137 \
            --scene-id 1 \
            --max-frames 30 \
            --output-dir "$EPS_OUT" 2>&1 | tee "${EPS_OUT}/run.log"

        echo "[P0] ε=$EPS complete"
    done

    echo "[P0] ε sweep complete. Results in $P0_DIR"
else
    echo "[SKIP] P0: V2X ε sweep"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# P3: V2X Extended Seeds & Scenes
# Purpose: increase statistical power (5 seeds × 3 scenes)
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_P3" = false ]; then
    echo ""
    echo "━━━ P3: V2X Extended Seeds & Scenes ━━━"
    P3_DIR="${RESULTS_DIR}/p3_v2x_extended"
    mkdir -p "$P3_DIR"

    cd "$V2X_DIR"

    # Multi-scene with 5 seeds, all attack types
    for SCENE in 1 8 78; do
        for ATTACK in persistent intermittent coordinated; do
            echo ""
            echo "--- Scene=$SCENE, Attack=$ATTACK ---"
            SCENE_OUT="${P3_DIR}/scene${SCENE}_${ATTACK}"
            mkdir -p "$SCENE_OUT"

            python "$V2X_SCRIPT" \
                --data "$V2X_DATA" \
                --resume "$V2X_CKPT" \
                --defense all \
                --all-scenarios \
                --eps 0.5 \
                --attack-type "$ATTACK" \
                --seeds 7 42 137 256 512 \
                --scene-id "$SCENE" \
                --max-frames 30 \
                --output-dir "$SCENE_OUT" 2>&1 | tee "${SCENE_OUT}/run.log"

            echo "[P3] Scene=$SCENE, Attack=$ATTACK complete"
        done
    done

    echo "[P3] Extended V2X experiments complete. Results in $P3_DIR"
else
    echo "[SKIP] P3: V2X extended seeds/scenes"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# P4: HotStuff Single-Instance Benchmark
# Purpose: shows multi-instance pipelining advantage over single-instance
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_P4" = false ]; then
    echo ""
    echo "━━━ P4: HotStuff Single-Instance Benchmark ━━━"
    P4_DIR="${RESULTS_DIR}/p4_hotstuff"
    mkdir -p "$P4_DIR"

    cd "$ROOT_DIR/src"
    if command -v go &> /dev/null; then
        # Run the existing Go benchmark with single-bft output
        go test -v -run TestConsensusBenchmark_WANScaling ./octopus/benchmark/ \
            2>&1 | tee "${P4_DIR}/wan_scaling.log"
        go test -v -run TestConsensusBenchmark_LANScaling ./octopus/benchmark/ \
            2>&1 | tee "${P4_DIR}/lan_scaling.log"
        go test -v -run TestReconfigBenchmark ./octopus/benchmark/ \
            2>&1 | tee "${P4_DIR}/reconfig.log"

        # Copy generated JSON results
        cp -f octopus/benchmark/testdata/*.json "$P4_DIR/" 2>/dev/null || true
        echo "[P4] HotStuff benchmark complete. Results in $P4_DIR"
    else
        echo "[P4] Go not found, skipping HotStuff benchmark"
    fi
else
    echo "[SKIP] P4: HotStuff benchmark"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  All experiments complete!"
echo "  Results directory: $RESULTS_DIR"
echo ""
echo "  Contents:"
ls -la "$RESULTS_DIR/" 2>/dev/null || true
echo ""
echo "  Next steps:"
echo "  1. Run: python experiments/gen_improvement_figures.py --results-dir $RESULTS_DIR"
echo "  2. Update appendix.tex with new tables and figures"
echo "═══════════════════════════════════════════════════════════════"
