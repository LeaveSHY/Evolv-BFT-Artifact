# Octopus EC2 Benchmark Deployment

## Overview

This directory contains scripts to deploy Octopus on EC2 and run real-world WAN benchmarks.

## Architecture

```
4 AWS Regions × 1 c5.4xlarge × 25 processes = 100 nodes
├── us-east-1:    nodes 0-24   (port 8080-8104, http 9000-9024)
├── us-west-2:    nodes 25-49
├── eu-west-1:    nodes 50-74
└── ap-southeast-1: nodes 75-99
```

## Prerequisites

1. AWS CLI configured with appropriate credentials
2. SSH key pair named `octopus-bench` in all 4 regions
3. Go 1.22+ installed locally (for genesis generation)

## Quick Start

```bash
# 1. Provision EC2 instances
./deploy_ec2.sh setup

# 2. Install Go, build binary, generate & distribute manifest
./setup_nodes.sh

# 3. Run benchmark (default: m=10 instances, 5 min, WAN)
./run_benchmark.sh wan_m10

# 4. Run single-instance comparison
./run_benchmark.sh wan_m1

# 5. Cleanup
./deploy_ec2.sh teardown
```

## Experiments

| Name | Instances (m) | Expected Throughput | Purpose |
|------|---------------|--------------------:|---------|
| `wan_m10` | 10 | ~320 ktx/s | Paper's main claim |
| `wan_m5` | 5 | ~160 ktx/s | Linear scaling verification |
| `wan_m1` | 1 | ~32 ktx/s | Single-instance baseline |

## Cost Estimate

- 4× c5.4xlarge @ $0.68/hr = $2.72/hr
- Typical run: 3 experiments × 10 min each + setup = ~1 hr
- **Total: ~$3-5 per full benchmark run**

## Key Files

- `deploy_ec2.sh` — EC2 lifecycle (setup/teardown/status)
- `setup_nodes.sh` — Install deps, build, generate manifest
- `run_benchmark.sh` — Orchestrate: start nodes → inject load → collect metrics → stop

## Metrics Collected

Each node exposes `GET /metrics` returning:
```json
{
  "global_confirmed_total": 1234567,
  "throughput_tps": 32145.6,
  "latency_p50_ms": 135.2,
  "latency_p95_ms": 138.1,
  "latency_p99_ms": 142.3,
  "backlog_pending": 12,
  "reject_total": 0
}
```

## Reproducing Paper Numbers

The paper claims "320 ktx/s at n=1000, averaged over three independent EC2 deployments."

To reproduce with 100 nodes (scaled):
1. Run `wan_m10` three times
2. Each run measures m=10 parallel instances → expect ~320 ktx/s aggregate
3. Latency: expect p50≈135ms, p99≈140ms (WAN inter-region RTT dominates)

For n=1000 (full scale): use 10 c5.4xlarge per region (40 machines total, ~$27/hr).
