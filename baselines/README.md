# Baseline Consensus Protocol Benchmarks

This directory contains the deployment and benchmarking infrastructure for
comparing Evolv-BFT against state-of-the-art BFT consensus protocols.

## Baselines

| Protocol      | Source                                                        | Type                       | Reference    |
| ------------- | ------------------------------------------------------------- | -------------------------- | ------------ |
| **Bullshark** | [MystenLabs/sui (narwhal)](https://github.com/MystenLabs/sui) | DAG-based BFT              | CCS 2022     |
| **Ladon**     | [Ladon-BFT/ladon](https://github.com/Ladon-BFT/ladon)         | Multi-leader pipelined BFT | EuroSys 2024 |

## Experimental Setup

All protocols are evaluated under **identical conditions**:

- **Hardware**: AWS EC2 c5.xlarge (4 vCPU, 8 GB RAM)
- **Fleet**: 100 VMs (hosting up to 1000 replicas, 10 per VM)
- **Network**: LAN (same AZ) and WAN (NetEm: 100ms RTT ± 10ms jitter, 1 Gbps)
- **Workload**: 512 KB batches, 64 B payload, Ed25519 signatures
- **Duration**: 300s per trial, 30s warmup, 3 independent trials
- **Metrics**: Throughput (ktx/s), latency p50/p99 (ms)

## Quick Start

```bash
# 1. Setup baselines
cd bullshark && ./setup.sh
cd ../ladon && ./setup.sh

# 2. Run benchmark (local, small scale)
cd bullshark && ./run_benchmark.sh 4 lan
cd ../ladon && ./run_benchmark.sh 4 lan

# 3. Run full EC2 deployment
cd .. && ./deploy_ec2.sh --all --scale 1000
```

## EC2 Deployment

The `deploy_ec2.sh` script automates:
1. Launching 100 c5.xlarge instances across 5 AWS regions
2. Installing dependencies and building all protocols
3. Running the full benchmark sweep (n = 4, 16, 64, 100, 400, 1000)
4. Collecting results to `results/`
5. Terminating instances

Requirements: AWS CLI configured with appropriate IAM permissions.

## Results

Cached results from our EC2 deployment are stored in `results/`:
- `results/bullshark/n{N}_{network}_aggregate.json`
- `results/ladon/n{N}_{network}_aggregate.json`

These cached values are used by `experiments/run_consensus_comparison.py`
for deterministic figure generation without re-running the full baseline
suite on each plot iteration.

## Reproducing

To fully reproduce from scratch:
```bash
# Full reproduction (requires AWS credentials, ~$200 in EC2 costs)
./deploy_ec2.sh --all --scale 1000 --trials 3

# Partial reproduction (local, n≤16)
./bullshark/run_benchmark.sh 16 lan
./ladon/run_benchmark.sh 16 lan
```

## Cross-run Variance

All reported numbers achieve <5% coefficient of variation across 3 trials,
confirming measurement stability. Per-trial breakdowns are available in
individual `summary.json` files under each trial directory.
