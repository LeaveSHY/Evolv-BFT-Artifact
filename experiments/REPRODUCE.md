# Experiment Reproduction Guide

This document provides a complete, reproducible path from source code to all paper claims.
Three tiers are provided depending on time budget and hardware availability.

---

## Tier 1: Smoke Test (5 minutes, any machine)

Verifies the codebase builds, all unit tests pass, and TLA+ specifications are syntactically valid.

```bash
# Build and test Go consensus layer
cd src/
make build && make test
# Expected: ALL PASS (66 source files, 60 test files, ~250 test functions)

# Verify Go vet (no static analysis issues)
make vet
# Expected: no output (clean)

# Quick Python import check
cd ../experiments/
python -c "from sfac_facmac_aligned import SFACFACMACController; print('OK')"
```

**Pass criteria**: `go test ./... -short` exits 0, Python import succeeds.

```bash
# (Optional) Race detection — requires Linux + gcc (CGO_ENABLED=1)
CGO_ENABLED=1 go test -race -short -count=1 ./...
# Expected: PASS (119 sync primitives across source ensure race freedom)
```

---

## Tier 2: Figure Reproduction (30 minutes, 8-core CPU)

Regenerates all paper figures from pre-computed results or fast analytical models.

```bash
cd experiments/

# Consensus layer figures (Figs 3-6): analytical pipeline model
python gen_consensus_figures.py --output-dir ../results/figures/

# Regenerate all figures with unified styling
python regen_all_figures.py

# Verify font compliance (no Type 3 fonts)
for pdf in ../../NDSS\ 2027_SUBMISSION/figures/Octopus/*.pdf; do
  pdffonts "$pdf" 2>/dev/null | grep -q "Type 3" && echo "FAIL $(basename $pdf)" || echo "OK $(basename $pdf)"
done
```

**Pass criteria**: All figures generated, zero Type 3 font violations.

---

## Tier 3: Full E2E Reproduction (~6 hours, 8+ core CPU)

Reproduces all experimental claims from scratch with deterministic seeds.

### Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU      | 4 cores | 8+ cores    |
| RAM      | 4 GB    | 16 GB       |
| GPU      | None    | None        |
| Disk     | 500 MB  | 2 GB        |
| OS       | Linux / WSL2 | Ubuntu 22.04 |

### Environment Setup

```bash
# Create conda environment
conda create -n robosac-gpu python=3.10 -y
conda activate robosac-gpu

# Install Python dependencies
pip install -r requirements.txt

# Navigate to experiment directory
cd experiments/
```

### Reproduction Steps

```bash
# Step 1: E2E MARL training + evaluation (Table IV, V + scaling)
# ~2 hours on 8-core CPU (5 seeds × 3000 episodes)
python run_e2e_experiments.py --output-dir results/e2e_final --ablation --multi-T
# → results/e2e_final/{e2e_summary.json, ablation_summary.json, figures/}
# Expected: Octopus D(T) ≈ 36, UCB ≈ 264, EXP3 ≈ 459, CUSUM ≈ 770

# Step 2: Theorem 4 verification (Regret Bound: D(T) = O(√T))
# ~1 hour (5 seeds, log-log regression + CI)
python verify_regret_bound.py
# → results/regret_verification/regret_verification_report.json
# Expected: slope α ≤ 0.55, PASS verdict

# Step 3: Appendix experiments A-G (sensitivity, scaling, ablation)
# ~3 hours
python run_ndss_appendix_full.py
# → results/ndss_appendix/exp_{a..g}_*/

# Step 4: Additional appendix figures
# ~30 minutes
python run_appendix_experiments.py

# Step 5: Consensus figures + unified style + copy to paper
python gen_consensus_figures.py --output-dir ../results/figures/
python regen_all_figures.py
```

### Seeds & Parameters

All experiments use seeds `[7, 13, 42, 97, 137]`, n=100, m=4, f=30, T=500 epochs, 3000 MARL training episodes. Full determinism via `seed_everything()` (numpy, torch, Python random).

### Expected Outputs

| Claim | Script | Metric | Expected |
|-------|--------|--------|----------|
| Table IV (E2E damage) | `run_e2e_experiments.py` | D(T=500) | Octopus ≈ 36 |
| Theorem 4 (regret) | `verify_regret_bound.py` | log-log slope | α ≤ 0.55 |
| Figs 3-5 (throughput) | `gen_consensus_figures.py` | ktx/s at n=1000 | ≥250 |
| Fig 6 (reconfiguration) | `gen_consensus_figures.py` | zero-stall | 0 view-changes |

---

## Distributed WAN Benchmarks (Consensus Layer)

For reproducing the EC2 WAN results (Figures 3-5, Table tab:e2e):

```bash
cd baselines/
# Requires AWS CLI configured with appropriate IAM permissions
# Cost estimate: ~$200 for full 3-trial sweep (~6h runtime)
./deploy_ec2.sh --all --scale 1000
```

Deploys 100 c5.xlarge instances across 5 AWS regions (us-east-1, us-west-2, eu-west-1, ap-northeast-1, ap-southeast-1). Runs Octopus, Bullshark, and Ladon with identical configurations (512 KB batch, 64B payload, Ed25519).

---

## V2X Evaluation (RQ4)

The V2X evaluation can run in two modes:

### Simulated (no external data needed)

```bash
cd experiments/
python run_v2x_eval.py --n-vehicles 6 --output-dir results/v2x/
# Reproduces trust estimation + detection metrics (TPR, FPR, detection latency)
# Uses simulated perception features calibrated from V2X-Sim statistics
```

### Full perception pipeline (requires V2X-Sim dataset)

1. Download V2X-Sim dataset from https://ai4ce.github.io/V2X-Sim/
2. Clone the Collaboration-Benchmark repo (see README.md "External Baselines")
3. Follow its setup instructions for FaFNet + MeanFusion
4. Run `octopus_v2x_eval.py` with paths to the trained perception model

---

## TLA+ Model Checking

```bash
cd tla/
bash run_tlc.sh
# Verifies 6 specifications: Safety, Byzantine, MultiLeader, GBC, Reconfiguration, Composed
# Expected: all PASS (no invariant violations, no deadlocks)
# Runtime: ~5-15 minutes depending on hardware
```
