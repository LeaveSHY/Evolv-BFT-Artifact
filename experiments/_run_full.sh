#!/bin/bash
cd /mnt/d/Alex/Papers/Experiment/Evolv-BFT
rm -rf experiments/results/e2e_opt_quick
mkdir -p experiments/results/e2e_post_opt
nohup python3 -u experiments/run_e2e_experiments.py \
    --output-dir experiments/results/e2e_post_opt \
    > experiments/results/e2e_post_opt_log.txt 2>&1 &
echo "PID=$!"
echo "Experiment launched in background. Check experiments/results/e2e_post_opt_log.txt for progress."
