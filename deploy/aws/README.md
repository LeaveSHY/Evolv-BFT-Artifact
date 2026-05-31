# Evolv-BFT — AWS EC2 1000-Node Deployment & Benchmark

## Overview

This directory contains the Terraform infrastructure and orchestration scripts used to deploy and benchmark 1000 Evolv-BFT replicas across 100 EC2 c5.xlarge instances (4 vCPU, 8 GiB each). This is the production deployment that produces the throughput and latency results reported in the paper (§VI, RQ1).

## Architecture

```
100 × EC2 c5.xlarge (4 vCPU, 8 GiB)
  └── 10 Evolv-BFT replicas per VM (Docker containers, --network host)
      └── Total: 1000 replicas
          └── VRF committee size k=25 per view
          └── m=10 parallel consensus lanes
          └── NetEm: 40ms one-way delay (80ms RTT)
```

## Quick Start

```bash
# 1. Configure
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your SSH key name

# 2. Run end-to-end benchmark (provisions, benchmarks, destroys)
./run_ec2_benchmark.sh --key-name your-key --region us-east-1

# 3. Results are in ../../experiments/results/ec2_benchmark/
```

## Prerequisites

| Tool       | Version | Purpose                    |
|------------|---------|----------------------------|
| Terraform  | ≥ 1.5   | Infrastructure provisioning|
| AWS CLI    | v2      | ECR push, S3, EC2 access   |
| Docker     | 20+     | Image build                |
| Go         | ≥ 1.22  | Build evolvbft-genesis      |
| Python     | ≥ 3.10  | Config generation          |

### AWS Permissions Required

- EC2: RunInstances, DescribeInstances, TerminateInstances
- VPC: CreateVpc, CreateSubnet, CreateSecurityGroup, etc.
- ECR: CreateRepository, PutImage, GetAuthorizationToken
- S3: CreateBucket, PutObject, DeleteBucket
- IAM: GetCallerIdentity

## Cost Estimate

| Resource           | Qty  | Rate         | Duration | Cost    |
|--------------------|------|--------------|----------|---------|
| c5.xlarge (on-demand)| 100 | $0.17/hr    | 0.5 hr   | ~$8.50  |
| gp3 EBS (20 GiB)  | 100  | $0.08/GiB-mo| 0.5 hr   | ~$0.01  |
| Data transfer      | —    | —            | —        | ~$1.00  |
| **Total**          |      |              |          | **~$10** |

A complete benchmark run (provision → warmup → 2min bench → teardown) takes ~30 minutes and costs approximately $10.

## Files

| File                      | Description                              |
|---------------------------|------------------------------------------|
| `main.tf`                 | Terraform: VPC, SG, EC2 instances        |
| `user_data.sh.tpl`        | Cloud-init script for each VM            |
| `run_ec2_benchmark.sh`    | End-to-end orchestration script          |
| `terraform.tfvars.example`| Example configuration                   |

## Manual Steps (Alternative)

If you prefer to run steps individually:

```bash
# Generate genesis manifest
cd ../../src && go build -o cmd/evolvbft-genesis/evolvbft-genesis ./cmd/evolvbft-genesis
./cmd/evolvbft-genesis/evolvbft-genesis -nodes 1000 -out genesis.json -seed ec2-bench

# Provision infrastructure
cd ../../deploy/aws
terraform init && terraform apply -var="key_name=mykey"

# Upload manifests
aws s3 sync ./manifests/ s3://your-bucket/manifests/

# Wait for nodes, then run benchmark
cd ../
./run_benchmark.sh --nodes 1000 --k8s --endpoints endpoints.txt --tps-target 200000

# Destroy
cd aws && terraform destroy -var="key_name=mykey"
```

## Validation Criteria (Paper Claims)

The benchmark validates (all met in our EC2 deployment):
- **Throughput**: ≥ 100,000 tx/s (measured ~210 ktx/s at n=1000, m=10)
- **Latency**: p50 ≤ 100ms under WAN (80ms RTT via NetEm)
- **Liveness**: All rounds committed (no stalls)

## Deployment Evidence

The in-process integration test (`TestConsensusBenchmarkVRF1000`) validates the protocol's crypto correctness at n=1000 with real Ed25519 key generation, signing, and verification. The EC2 deployment adds production-grade networking:
- Real libp2p GossipSub over TCP across 100 physical machines
- WAN latency shaping via NetEm/TBF (40ms one-way = 80ms RTT)
- OS-level scheduling and Docker container isolation
- Per-VM resource constraints (4 vCPU, 8 GiB per 10 replicas)

Results are archived in `experiments/results/ec2_benchmark/`.
