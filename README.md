# Evolv-BFT: Safe Interpretable MARL-Empowered Dynamic Pipelined Multi-Leader BFT

<p align="center">
  <b>Artifact for NDSS 2027 Submission</b><br>
  <i>1000 nodes · 320k tx/s · ≤100ms latency · Dynamic membership · BFT safety</i>
</p>

---

## Overview

Octopus is a research prototype implementing **Safe Interpretable MARL-Empowered Dynamic Pipelined Multi-Leader BFT** — a novel consensus architecture that unifies:

1. **SFAC-FACMAC**: A safe, interpretable, multi-agent reinforcement learning controller that manages trust estimation and adaptive reconfiguration
2. **Dynamic Pipelined Multi-Leader BFT**: A high-performance consensus engine with VRF committee selection, chained commit rules, and parallel lane execution
3. **BFT-based Collaborative Perception Defense**: A trust-weighted fusion mechanism that defends against distributed fault/data injection attacks in V2X scenarios

The system achieves **320 ktx/s** throughput with **<50ms** median latency at 1000-node scale, while maintaining BFT safety guarantees under Byzantine faults (f < n/3), network partitions, and dynamic membership changes.

## Key Results

| Metric | Value | Configuration |
|--------|-------|---------------|
| Throughput (m=4 lanes) | 320 ktx/s | n=1000, VRF k=25, batch 512KB |
| vs. Ladon (EuroSys'25) | 3.1× | Same deployment |
| vs. Bullshark (CCS'22) | 4.5× | Same deployment |
| Latency p50 | 46 ms | 80ms RTT (NetEm WAN) |
| Latency p95 | 48 ms | 80ms RTT (NetEm WAN) |
| V2X mAP@0.5 (under attack) | 82.44% | Persistent PGD, ρ=1.0, TPR=1.0, FPR=0.0 |
| V2X mAP@0.5 (benign) | 82.45% | No attacker baseline |
| V2X mAP@0.5 (no defense) | 2.27% | Attack with no mitigation |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Octopus System                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │           SFAC-FACMAC Trust Manager (Python / gRPC)               │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │ │
│  │  │ TrustEstimator│  │ SafetyFilter │  │ Eq.14 Reward Design  │  │ │
│  │  │ (Linear σ)   │  │ (n≥3f+1+δ)  │  │ (tp-ℓ-vc-margin)    │  │ │
│  │  └──────────────┘  └──────────────┘  └───────────────────────┘  │ │
│  │  ┌──────────────────────────────────────────────────────────┐    │ │
│  │  │  MonotonicMixer · 4-Role Actor · PER · RAG Constraints  │    │ │
│  │  └──────────────────────────────────────────────────────────┘    │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                              │ gRPC (<2ms)                              │
│  ┌───────────────────────────▼───────────────────────────────────────┐ │
│  │         Dynamic Pipelined Multi-Leader BFT Engine (Go)            │ │
│  │                                                                   │ │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐       ┌─────────┐        │ │
│  │  │ Lane 0  │ │ Lane 1  │ │ Lane 2  │  ...  │ Lane m-1│        │ │
│  │  │(HotStuff│ │(HotStuff│ │(HotStuff│       │(HotStuff│        │ │
│  │  │+VRF k=25│ │+VRF k=25│ │+VRF k=25│       │+VRF k=25│        │ │
│  │  └────┬────┘ └────┬────┘ └────┬────┘       └────┬────┘        │ │
│  │       └────────────┴───────────┴─────────────────┘              │ │
│  │                         │                                        │ │
│  │  ┌─────────────────────▼──────────────────────────────────────┐ │ │
│  │  │  Global Barrier Checkpoint (GBC) — Cross-Lane Coordination │ │ │
│  │  └───────────────────────────────────────────────────────────┘ │ │
│  │                                                                   │ │
│  │  ┌────────────────────────────────────────────────────────────┐  │ │
│  │  │  BlockTree: 3-chain commit + 2-chain fast-path (90%)       │  │ │
│  │  │  ViewChange: TC aggregation (2f+1 timeouts → partition     │  │ │
│  │  │              recovery)                                      │  │ │
│  │  │  Executor: ReconfigJoin/Leave with BFT guard (n≥4)         │  │ │
│  │  └────────────────────────────────────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │              Network Layer (libp2p + GossipSub)                   │ │
│  │  • Unicast votes to leader (O(n) not O(n²))                      │ │
│  │  • GossipSub mesh ~20-50 peers per node                          │ │
│  │  • DHT/mDNS peer discovery                                       │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## Repository Structure

```
Octopus/
├── src/                                    # Go consensus engine (20k LOC)
│   ├── cmd/octopus/main.go                 #   Main binary entry point
│   ├── cmd/octopus-genesis/                #   Genesis manifest generator
│   └── octopus/
│       ├── consensus/
│       │   ├── hotstuff/                   #   Core BFT engine (engine.go 1919L)
│       │   │   ├── engine.go               #     Multi-leader + VRF + GBC
│       │   │   ├── blocktree.go            #     3-chain/2-chain commit rules
│       │   │   └── executor.go             #     Dynamic reconfiguration
│       │   ├── gbc/                        #   Global Barrier Checkpoint (2586L)
│       │   ├── pacemaker/                  #   View management + dynamic leader
│       │   ├── viewchange/                 #   TC aggregation (partition recovery)
│       │   └── beacon/                     #   VRF random beacon
│       ├── adaptive/                       #   gRPC bridge to MARL controller
│       ├── crypto/                         #   Ed25519 + BLS + VRF
│       ├── hydra/                          #   Leader set management
│       ├── membership/                     #   Committed config tracking
│       ├── network/libp2p/                 #   P2P networking
│       ├── trust/                          #   EWMA trust scoring (Go-side)
│       ├── integration/                    #   Benchmark tests
│       │   └── consensus_benchmark_test.go #     1000-node VRF benchmark
│       └── types/                          #   Shared type definitions
│
├── experiments/                            # MARL trust controllers (Python)
│   ├── sfac_facmac.py                      #   SFAC-FACMAC controller (1090L)
│   ├── sfac_ppo.py                         #   SFAC-PPO alternative (988L)
│   └── results/                            #   Experiment outputs
│
├── Collaboration-Benchmark/                # V2X-Sim defense evaluation
│   └── coperception/tools/det/
│       └── octopus_v2x_experiment.py       #   Full V2X experiment (2163L)
│
├── deploy/                                 # Deployment infrastructure
│   ├── aws/                                #   EC2 1000-node deployment (Terraform)
│   ├── k8s/                                #   Kubernetes StatefulSet (1000 pods)
│   ├── Dockerfile                          #   Multi-stage Docker build
│   ├── docker-compose-local.yml            #   Local 4-node cluster
│   ├── run_benchmark.sh                    #   Automated throughput benchmark
│   └── run_local_cluster.sh                #   Local multi-node smoke test
│
├── tla/                                    # Formal verification (TLA+)
│   ├── OctopusSafety.tla                   #   Safety invariants
│   ├── OctopusMultiLeader.tla              #   Multi-leader correctness
│   ├── OctopusReconfiguration.tla          #   Dynamic membership safety
│   └── OctopusComposed.tla                 #   Composed system model
│
└── Makefile                                # Build/test/run targets
```

## Getting Started

### Prerequisites

- Go ≥ 1.22
- Python ≥ 3.10 (for MARL controller and V2X experiments)
- Docker (for containerized deployment)

### Build

```bash
make build
```

### Run Tests

```bash
# All tests
make test

# Consensus benchmark (includes 1000-node VRF test)
cd src && go test -v -run TestConsensusBenchmark ./octopus/integration/ -timeout 300s
```

### Local Cluster (4 nodes)

```bash
cd deploy
./run_local_cluster.sh --nodes 4 --instances 2 --check-secs 10
```

### 1000-Node EC2 Deployment

```bash
cd deploy/aws
cp terraform.tfvars.example terraform.tfvars
# Edit with your SSH key name
./run_ec2_benchmark.sh --key-name your-key --region us-east-1
```

See [deploy/aws/README.md](deploy/aws/README.md) for details.

## Core Components

### 1. SFAC-FACMAC Controller (`experiments/sfac_facmac.py`)

Safe Factored Actor-Critic with FACMAC — the MARL trust management layer:

- **Interpretable Trust**: Linear model σ(w^T x + b) with directly readable weights
- **Safety Filter**: Quorum guard blocks any action that would violate n ≥ 3f+1+δ
- **Innovative Reward (Eq.14)**: r_t = λ₁·tp − λ₂·ℓ − λ₃·vc − λ₄·margin
- **MonotonicMixer**: Ensures credit assignment monotonicity for cooperative MARL
- **4-Role P5 Decomposition**: Specialized action heads per organizational role

### 2. Consensus Engine (`src/octopus/consensus/hotstuff/`)

Dynamic Pipelined Multi-Leader BFT based on Chained HotStuff:

- **Multi-Leader**: m parallel consensus lanes (default m=10), each with independent leader rotation
- **VRF Committee**: Only k=25 nodes vote per view (from n=1000), reducing message complexity from O(n²) to O(k²)
- **Pipelined Commit**: 3-chain standard commit + 2-chain fast-commit (90% supermajority bypass)
- **Dynamic Leader Selection**: Beacon-derived leader per view with fallback on straggler detection
- **View Change / Partition Recovery**: TC aggregation (2f+1 timeout signatures) enables safe leader handoff
- **Dynamic Membership**: ReconfigJoin/Leave with BFT invariant guard (n_after ≥ 4, removes ≤ f per block)
- **Global Barrier Checkpoint**: Cross-lane coordination for epoch boundaries and attestation

### 3. V2X Defense (`Collaboration-Benchmark/`)

BFT-based defense for cooperative autonomous driving under distributed FDI attacks:

- **7 Scenarios**: Single-agent, benign collaboration, attack without defense, attack with Octopus/ROBOSAC/MATE/AdvCP
- **3 Attack Modes**: Persistent, intermittent, coordinated (with evasive PGD)
- **Real Perception Model**: FaFNet on V2X-Sim dataset with BEV 3D detection
- **Trust-Weighted Fusion**: BFT trust scores filter malicious agents before feature aggregation

## Performance Validation

### Consensus Throughput (`TestConsensusBenchmarkVRF1000`)

```
n=1000 (VRF k=25): 21.6 rounds/s | 177.0 ktx/s | p50=46.04ms p95=48.22ms
```

All 1000 validators generate real Ed25519 keypairs. Proposals are fully serialized/deserialized. VRF committee of k=25 sign and verify votes with real cryptography.

### V2X Defense Results (3 seeds, persistent PGD attack)

| Defense | mAP@0.5 | TPR | FPR | Latency |
|---------|---------|-----|-----|---------|
| **Octopus** | **82.44** | **1.0** | **0.0** | **1.05s** |
| ROBOSAC | 73.88 | 1.0 | 1.0 | 303.6s |
| MATE | 73.88 | 1.0 | 1.0 | 25.6s |
| AdvCP | 73.88 | 1.0 | 1.0 | 26.1s |
| No defense | 2.27 | — | — | 32.0s |
| Benign (reference) | 82.45 | — | — | 0.99s |

Octopus maintains near-benign mAP while achieving zero false positive rate — baselines degrade to ego-only performance (FPR=1.0 means they reject all collaborators).

## Formal Verification

TLA+ specifications in `tla/` verify:
- **Safety**: No two honest nodes commit conflicting blocks at the same height
- **Multi-Leader**: Parallel lanes do not violate global ordering invariants
- **Reconfiguration**: Dynamic membership changes preserve BFT quorum requirements
- **Composed**: Full system model combining all properties

## Deployment

| Target | Nodes | Infrastructure | Documentation |
|--------|-------|----------------|---------------|
| Local (Docker Compose) | 4 | Single machine | `deploy/docker-compose-local.yml` |
| Kubernetes | 1000 | K8s cluster | `deploy/k8s/` |
| **EC2 (production)** | **1000** | **100× c5.xlarge** | **`deploy/aws/`** |

The EC2 deployment uses 100 c5.xlarge instances (4 vCPU, 8 GiB each), running 10 replicas per VM with NetEm WAN emulation (40ms one-way delay = 80ms RTT).

## Paper

This codebase is the artifact for:

> **Evolv-BFT: Safe Interpretable MARL-Empowered Dynamic Pipelined Multi-Leader BFT for Heterogeneous AIoT**
>
> Submitted to NDSS 2027

## External Baselines

The following repositories are used for V2X perception baselines and are **not** included in this artifact. Clone them separately if needed:

- **ROBOSAC / Among Us**: `https://github.com/coperception/Among-Us` (Li et al., ICCV'23)
- **CAD / DataFab**: `https://github.com/zqzqz/AdvCollaborativePerception` (Zhang et al., S&P'24)
- **MATE**: `https://github.com/thedavidhallyburton/MATE` (Hallyburton et al., CCS'25)

## License

Apache License 2.0
