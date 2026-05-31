# Evolv-BFT Architecture: Paper → Code Mapping

This document maps each component described in the NDSS 2027 submission to the
corresponding Go/Python implementation, with honest claim boundaries.

---

## Three-Layer Architecture (Paper §III)

| Paper Component                   | Paper Section | Code Location                                      | Status                               |
| --------------------------------- | ------------- | -------------------------------------------------- | ------------------------------------ |
| Dynamic Pipelined Consensus       | §III-B        | `evolvbft/consensus/hotstuff/`                      | ✅ Production                         |
| Global Beacon Chain (GBC)         | §III-C        | `evolvbft/consensus/gbc/`                           | ⚠️ Local log with signed attestations |
| Safe Factored Actor-Critic (SFAC) | §III-D        | `marl/` (Python) + `evolvbft/adaptive/` (Go bridge) | ✅ Split-platform                     |

---

## Consensus Layer (`evolvbft/consensus/hotstuff/`)

**Paper claim**: m concurrent pipelined BFT instances with certified chain-internal reconfiguration.

| File            | Lines | Purpose                                                                                                                                           | Paper Reference                                     |
| --------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| `engine.go`     | ~1650 | Core per-instance consensus engine: proposals, votes, TC aggregation, equivocation detection, VRF committee selection, mempool, leader reputation | Algorithm 6 (DynaPipe), Algorithms I-III (Appendix) |
| `orderer.go`    | ~180  | GlobalOrderer: collects InstanceOutputs from m parallel instances, emits in strict rank order with barrier timeouts and nil-fill                  | §III-E, global ordering                             |
| `executor.go`   | ~610  | Block execution with G4 epoch safety barrier: reconfigs queued, applied at CommitReconfigs()                                                      | Theorem 3 (cross-view safety)                       |
| `blocktree.go`  | ~225  | BlockTree with 3-chain commit + optimistic 2-chain fast-commit, safety rules                                                                      | §III-B pipelined phases                             |
| `reputation.go` | ~450  | Per-leader performance tracking with Eq. 5 trust features (d/W, e/W, v/W, τ̄, σ_τ)                                                                 | §III-D Eq. 5 (Trust Estimation)                     |
| `pacemaker.go`  | ~250  | Timer-driven view-change with exponential backoff, VRF-based leader selection                                                                     | §III-B View-Change                                  |

**Tests**: `engine_test.go`, `orderer_test.go`, `executor_test.go`, `blocktree_test.go` (~5000+ lines)

---

## Global Beacon Chain (`evolvbft/consensus/gbc/`)

**Paper claim**: BFT log among m primaries recording QCs, membership changes, checkpoints, policy updates. Properties G1-G4.

| File            | Lines | Purpose                                                                                | Paper Reference                                   |
| --------------- | ----- | -------------------------------------------------------------------------------------- | ------------------------------------------------- |
| `gbc.go`        | ~170  | Thread-safe append-only log with attestation collection from m primaries               | §III-C, G1 (append-only), G4 (quorum attestation) |
| `types.go`      | ~50   | Entry types (QC, Checkpoint, Membership, PolicyUpdate), Attestation struct, QuorumSize | §III-C four metadata categories                   |
| `checkpoint.go` | ~90   | Checkpoint record decode/validate against committed head                               | §III-C epoch-anchored checkpoints                 |

**Claim boundary**: The GBC log currently runs within a single process. Cross-process BFT replication among m primaries is not implemented. The `Attest()` method collects 2f+1 signatures (G4) but signature verification is caller's responsibility. G2 (honest-primary agreement) relies on the deterministic rank-based ordering in GlobalOrderer rather than a separate BFT protocol.

---

## Adaptive Trust Management

### Go Bridge (`evolvbft/adaptive/`)

**Paper claim**: The consensus layer observes metrics and the SFAC produces trust-driven reconfiguration actions.

| File            | Lines | Purpose                                                                             | Paper Reference                        |
| --------------- | ----- | ----------------------------------------------------------------------------------- | -------------------------------------- |
| `controller.go` | ~300  | Observer→Policy→Governance→Guardrail→Actuator pipeline                              | Algorithm 5 (SafeMARL) control loop    |
| `policy.go`     | ~250  | SafeBaselinePolicy (rule-based), HTTPPolicy (bridge to Python SFAC), ScriptedPolicy | §III-D, `-adaptive-policy=facmac-http` |
| `types.go`      | ~200  | Observation/Action schemas, per-instance agent actions                              | §III-D Dec-POMDP observation/action    |
| `guardrails.go` | ~150  | Pre-argmax safety filter: n_v >= 3f_v + 1 enforcement                               | §III-D P3, Algorithm 5 lines 13-17     |
| `reward.go`     | ~200  | WeightedRewardModel with 20 reward terms, λ1-λ4 weights                             | §III-D Eq. 4 (reward)                  |

### Python SFAC (`marl/`)

**Paper claim**: Safe Factored Actor-Critic with centralized critic, per-agent actors, role decomposition, end-to-end training.

| File              | Purpose                                                                   | Paper Reference                                |
| ----------------- | ------------------------------------------------------------------------- | ---------------------------------------------- |
| `policy.py`       | SafeFACMACPolicy: factored actor-critic with per-role outputs             | §III-D Eq. 7 (critic), Eq. 8 (policy gradient) |
| `trainer.py`      | SafeFACMACTrainer: training loop with prioritized experience replay       | §III-D P2 (centralized critic)                 |
| `organization.py` | MOISEOrganizationModel: four roles (Sentinel, Commander, Tuner, Guardian) | §III-D P5 (role decomposition)                 |
| `app.py`          | FastAPI service: `/infer`, `/train/online`, `/trace/ingest` endpoints     | Narrow interface (d=5 features)                |
| `service.py`      | PolicyService: orchestrates training, inference, checkpointing            | §III-E Algorithm 7 (Evolv-BFT loop)              |

**Integration**: Go sends observations to Python via HTTP POST to `/infer`, receives actions. Python trains from JSONL traces via `/train/offline` or online via `/trace/ingest` + `/train/online`.

---

## Supporting Infrastructure

| Package           | Purpose                                                                        | Paper Reference                       |
| ----------------- | ------------------------------------------------------------------------------ | ------------------------------------- |
| `hydra/`          | L-set manager with quorum-based auto-transition, temporary config manager      | Dynamic membership, Theorem 3         |
| `membership/`     | Join/leave request processing, config history                                  | §III-B chain-internal reconfiguration |
| `trust/`          | Minimal sliding-window trust scaffold (superseded by `reputation.go` features) | Legacy, see `reputation.go`           |
| `crypto/`         | Kyber VRF, Ed25519 signatures, Threshold BLS                                   | §III-B VRF committee selection        |
| `network/libp2p/` | Production P2P with GossipSub pub/sub                                          | Network layer                         |
| `bootstrap/`      | Genesis manifest generation, config parsing                                    | Deployment                            |
| `storage/`        | In-memory block/QC store                                                       | Data persistence                      |

---

## Deployment

| File                              | Purpose                                        |
| --------------------------------- | ---------------------------------------------- |
| `deploy/run_integrated.sh`        | One-command Go+Python cluster launch (Linux)   |
| `deploy/run_integrated.ps1`       | One-command Go+Python cluster launch (Windows) |
| `deploy/run_local_cluster.ps1`    | Consensus-only local cluster (Windows)         |
| `deploy/docker-compose-local.yml` | Docker Compose for 4-node cluster              |
| `deploy/Dockerfile`               | Multi-stage Go build                           |
| `deploy/k8s/`                     | Kubernetes manifests for 1000-node deployment  |

---

## Experiments

| Directory                            | Purpose                                | Paper Reference   |
| ------------------------------------ | -------------------------------------- | ----------------- |
| `experiments/run_e2e_experiments.py` | E2E simulation: 5 seeds × T=500 epochs | §6 RQ3 (Table IV) |
| `experiments/`                       | Benchmark scripts, result analysis     | §6 RQ1-RQ4        |

---

## Claim Boundaries (Honest Disclosure)

1. **Split-platform design**: Consensus (Go/EC2) and trust manager (Python/PyTorch) run on separate platforms connected by HTTP. This is disclosed in §6 Limitations.
2. **GBC ordering**: The GBC piggybacks on the GlobalOrderer's deterministic rank scheme rather than running a separate BFT protocol among primaries. Signed attestation collection (G4) is implemented.
3. **SFAC training**: The neural network (factored critic, per-agent actors) runs in Python. The Go runtime executes safe actions via the HTTPPolicy bridge.
4. **In-memory storage**: No persistent disk storage. Suitable for benchmarks (minutes-long runs) but not production deployment.
