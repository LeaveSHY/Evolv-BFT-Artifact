# Evolv-BFT — Deployment & Evaluation Guide

Complete deployment automation for Evolv-BFT: a dynamic, pipelined, multi-leader Byzantine fault-tolerant consensus protocol supporting 1000-node clusters at ≥100k tx/s.

---

## Table of Contents

1. [Quick Start: Integrated Go + Python](#quick-start-integrated-go--python)
2. [Prerequisites](#prerequisites)
3. [Building the Docker Image](#building-the-docker-image)
4. [Local Testing with Docker Compose (4 nodes)](#local-testing-with-docker-compose)
5. [Kubernetes Deployment (1000 nodes)](#kubernetes-deployment)
6. [Configuration Generator](#configuration-generator)
7. [Running Benchmarks](#running-benchmarks)
8. [Collecting and Analyzing Metrics](#collecting-and-analyzing-metrics)
9. [HTTP API Reference](#http-api-reference)
10. [Scaling Considerations](#scaling-considerations)
11. [Expected Performance Characteristics](#expected-performance-characteristics)
12. [Troubleshooting](#troubleshooting)

---

## Quick Start: Integrated Go + Python

The full Evolv-BFT pipeline connects two components through a narrow HTTP interface:
- **Go consensus layer** (`src/cmd/evolvbft`): m parallel pipelined BFT instances with certified chain-internal reconfiguration
- **Python SFAC trust manager** (`marl/app.py`): Safe Factored Actor-Critic with d=5 consensus features per agent per epoch

### One-command launch

```bash
# Linux/macOS
./deploy/run_integrated.sh --nodes 4 --instances 2

# Windows PowerShell
.\deploy\run_integrated.ps1 -Nodes 4 -Instances 2
```

This will:
1. Build the Go binary (`evolvbft`, `evolvbft-genesis`)
2. Start the Python SFAC service on port 18080 (FastAPI/uvicorn)
3. Generate a genesis manifest for N nodes
4. Start N consensus nodes with `-adaptive-policy=facmac-http` pointing to the SFAC service
5. Monitor health of both components

### Manual setup

```bash
# Terminal 1: Start SFAC service
cd /path/to/evolvbft
python -m uvicorn marl.app:app --host 0.0.0.0 --port 18080

# Terminal 2: Start consensus (after SFAC is ready)
./build/evolvbft -id=0 -port=8080 -http=9000 \
  -total-nodes=4 -initial-validators=4 -instances=2 \
  -adaptive-enabled -adaptive-policy=facmac-http \
  -adaptive-policy-url=http://127.0.0.1:18080/infer \
  -adaptive-interval-ms=5000
```

### Verifying the integration

```bash
# Check SFAC service health
curl http://127.0.0.1:18080/health
# Expected: {"ok":true,"model_ready":false}

# Check consensus node metrics
curl http://127.0.0.1:9000/metrics
# Expected: JSON with throughput_tps, global_confirmed_total, etc.

# Check adaptive controller state
curl http://127.0.0.1:9000/adaptive
# Expected: JSON showing policy=facmac-http, observations, actions
```

---

## Prerequisites

| Tool               | Version | Purpose                         |
| ------------------ | ------- | ------------------------------- |
| Docker             | 24+     | Container build & local testing |
| Docker Compose     | v2+     | Local multi-node clusters       |
| kubectl            | 1.28+   | Kubernetes cluster management   |
| Python             | 3.10+   | Configuration generator         |
| curl               | 7.x+    | HTTP endpoint testing           |
| jq                 | 1.6+    | JSON processing in scripts      |
| bc                 | any     | Arithmetic in benchmark scripts |
| Kubernetes cluster | 1.28+   | Production 1000-node deployment |

### Cluster sizing for 1000 nodes

Each Evolv-BFT pod currently requests 500m CPU / 512Mi RAM and is limited to 1000m CPU / 1Gi RAM. For a 1000-node deployment:

- **Minimum**: 500 CPU cores, 512 GiB RAM across all worker nodes
- **Recommended**: 200+ worker nodes (5 pods per node for anti-affinity spread)
- **Storage**: 1 GiB PVC per pod = 1 TiB total persistent storage
- **Extra manifest storage**: one read-only PVC named `evolvbft-manifests` containing `/config/manifests/node-<id>-manifest.json` for every pod

---

## Building the Docker Image

From the **Evolv-BFT project root** (parent of `deploy/`):

```bash
# Build the image
docker build -t evolvbft:latest -f deploy/Dockerfile .

# Verify
docker run --rm evolvbft:latest --help

# Tag for a registry (optional)
docker tag evolvbft:latest your-registry.io/evolvbft:latest
docker push your-registry.io/evolvbft:latest
```

The Dockerfile uses a **multi-stage build**:
1. **Builder stage** (`golang:1.22-alpine`): compiles `cmd/evolvbft/main.go` with static linking
2. **Runtime stage** (`alpine:3.19`): minimal image with `curl` and `jq` for health checks

---

## Local Testing with Docker Compose

### Quick start (4 nodes)

```bash
cd deploy/

# Build and start 4 nodes
docker compose -f docker-compose-local.yml up -d --build

# Check health
docker compose -f docker-compose-local.yml ps

# View logs
docker compose -f docker-compose-local.yml logs -f

# Check metrics from node 0
curl -s http://localhost:9000/metrics | jq .

# Check all nodes
for i in 0 1 2 3; do
  echo "Node $i:"
  curl -s "http://localhost:$((9000 + i))/metrics" | jq '{tps: .throughput_tps, confirmed: .global_confirmed_total}'
done

# Stop and clean up
docker compose -f docker-compose-local.yml down -v
```

### Scaling to more nodes

Use `generate_config.py` to create a Compose file for N nodes:

```bash
python3 generate_config.py --nodes 10 --compose --output-dir ./configs
# Then use the generated docker-compose-generated.yml
```

### Port mapping (local)

| Node | HTTP Port      | P2P Port       |
| ---- | -------------- | -------------- |
| 0    | localhost:9000 | localhost:8080 |
| 1    | localhost:9001 | localhost:8081 |
| 2    | localhost:9002 | localhost:8082 |
| 3    | localhost:9003 | localhost:8083 |

---

## Kubernetes Deployment

### Step 1: Push the image

```bash
# For cloud registries
docker tag evolvbft:latest your-registry.io/evolvbft:latest
docker push your-registry.io/evolvbft:latest

# Update the StatefulSet image reference
sed -i 's|evolvbft:latest|your-registry.io/evolvbft:latest|' k8s/evolvbft-statefulset.yaml
```

### Step 2: Prepare runtime manifests

`evolvbft-statefulset.yaml` now expects a read-only PVC named `evolvbft-manifests` mounted at `/config/manifests`.
That PVC must contain the files generated by `generate_config.py`:

- `/config/manifests/node-0-manifest.json`
- ...
- `/config/manifests/node-999-manifest.json`

A practical workflow is:

1. Generate the genesis manifest and per-node runtime manifests locally.
2. Copy `configs/manifests/` into a shared volume.
3. Create or bind that shared volume to a PVC named `evolvbft-manifests` in the target namespace.

Without this step, multi-node Evolv-BFT startup will fail because the binary rejects ephemeral bootstrap for clusters larger than one node.

### Step 3: Deploy

```bash
# Apply in order: ConfigMap → Service → StatefulSet
kubectl apply -f k8s/evolvbft-configmap.yaml
kubectl apply -f k8s/evolvbft-service.yaml
kubectl apply -f k8s/evolvbft-statefulset.yaml

# Watch rollout (pods start in parallel)
kubectl rollout status statefulset/evolvbft --timeout=600s

# Check pod status
kubectl get pods -l app=evolvbft --no-headers | wc -l    # should be 1000
kubectl get pods -l app=evolvbft | grep -c Running        # healthy count
```

### Step 4: Verify cluster health

```bash
# Port-forward to a single node for inspection
kubectl port-forward pod/evolvbft-0 9000:9000 &

# Check metrics
curl -s http://localhost:9000/metrics | jq .

# Check network connectivity
curl -s http://localhost:9000/network | jq '.connection'

# Check Hydra membership
curl -s http://localhost:9000/hydra | jq .
```

### Scaling the StatefulSet

```bash
# Scale to different sizes
kubectl scale statefulset evolvbft --replicas=100   # small test
kubectl scale statefulset evolvbft --replicas=500   # medium
kubectl scale statefulset evolvbft --replicas=1000  # full scale

# Regenerate cluster config for the new size and re-apply the generated patch
python3 generate_config.py \
  --nodes 500 \
  --k8s \
  --manifest ./configs/genesis.json \
  --instances 5 \
  --batch-txs 2048 \
  --timeout-ms 1000 \
  --output-dir ./configs
kubectl apply -f ./configs/configmap-patch.yaml

# Refresh the mounted runtime manifests if the validator set changed, then restart
kubectl rollout restart statefulset/evolvbft
```

### Tear down

```bash
kubectl delete -f k8s/evolvbft-statefulset.yaml
kubectl delete -f k8s/evolvbft-service.yaml
kubectl delete -f k8s/evolvbft-configmap.yaml

# Delete persistent volumes
kubectl delete pvc -l app=evolvbft
```

---

## Configuration Generator

The `generate_config.py` script generates per-node CLI arguments, peer lists, and deployment-specific configs.

### Stable identity bootstrap

For reproducible multi-node runs, first generate a genesis manifest with stable Ed25519 identities and libp2p `PeerID`s:

```bash
cd ../src
go run ./cmd/evolvbft-genesis -nodes 10 -seed test-cluster -out ../deploy/configs/genesis.json
```

Then feed that manifest into the deployment generator:

```bash
python3 generate_config.py \
  --nodes 10 \
  --compose \
  --manifest ./configs/genesis.json \
  --output-dir ./configs
```

This produces:

- `cluster-manifest.json`: public cluster identity data with peer multiaddrs
- `manifests/node-<id>-manifest.json`: runtime manifest for each node, containing that node's private key and all validators' public keys
- per-node args with `-manifest=...`
- `peers.json` with valid `/p2p/<PeerID>` multiaddrs

Multi-node Evolv-BFT now refuses ephemeral bootstrap at runtime. Generate and pass a manifest for every compose, Kubernetes, or bare-metal cluster with more than one node.

When a manifest is supplied, Docker Compose output mounts the generated config directory at `/config` and each node starts with its own runtime manifest.

For Kubernetes, `generate_config.py --k8s` generates the cluster-wide values and the per-node runtime manifests locally, but you must still publish `configs/manifests/` to the `evolvbft-manifests` PVC consumed by the StatefulSet.

### Kubernetes mode

```bash
python3 generate_config.py \
  --nodes 1000 \
  --k8s \
  --manifest ./configs/genesis.json \
  --instances 5 \
  --batch-txs 2048 \
  --timeout-ms 1000 \
  --output-dir ./configs
```

### Docker Compose mode

```bash
python3 generate_config.py \
  --nodes 10 \
  --compose \
  --manifest ./configs/genesis.json \
  --output-dir ./configs
```

### Bare-metal mode

Create a `hosts.txt` file with one IP per line:

```
192.168.1.10
192.168.1.11
192.168.1.12
```

```bash
python3 generate_config.py \
  --nodes 30 \
  --hosts hosts.txt \
  --manifest ./configs/genesis.json \
  --output-dir ./configs
```

### Output files

| File                                   | Description                             |
| -------------------------------------- | --------------------------------------- |
| `configs/nodes/node-{i}.args`          | CLI arguments for each node             |
| `configs/peers.json`                   | Full peer list with hostnames and ports |
| `configs/peers.txt`                    | Flat peer list (space-separated)        |
| `configs/http_endpoints.txt`           | HTTP endpoint URLs for benchmarking     |
| `configs/configmap-patch.yaml`         | K8s ConfigMap patch (K8s mode)          |
| `configs/docker-compose-generated.yml` | Compose file (Compose mode)             |

---

## Running Benchmarks

### Local (Docker Compose, 4 nodes)

```bash
chmod +x run_benchmark.sh

./run_benchmark.sh \
  --nodes 4 \
  --tps-target 1000 \
  --duration 30 \
  --tx-size 256 \
  --concurrency 8 \
  --output benchmark_local.json
```

### Kubernetes (1000 nodes)

```bash
./run_benchmark.sh \
  --nodes 1000 \
  --k8s \
  --tps-target 100000 \
  --duration 120 \
  --tx-size 256 \
  --warmup 15 \
  --cooldown 10 \
  --concurrency 64 \
  --output benchmark_k8s.json
```

### Using a pre-built endpoints file

```bash
# Generate endpoints
python3 generate_config.py --nodes 1000 --k8s --output-dir ./configs

# Run benchmark against those endpoints
./run_benchmark.sh \
  --endpoints ./configs/http_endpoints.txt \
  --tps-target 100000 \
  --duration 60 \
  --output benchmark_results.json
```

### Benchmark output

Results are written as JSON to the `--output` file:

```json
{
  "benchmark": {
    "timestamp": "2026-03-06T12:00:00Z",
    "nodes": 1000,
    "tps_target": 100000,
    "duration_seconds": 120,
    "tx_size_bytes": 256,
    "concurrency": 64
  },
  "results": {
    "total_sent": 12000000,
    "total_success": 11985000,
    "achieved_tps": 100000.00,
    "consensus_tps": 105234.50,
    "latency_ms": {
      "avg": 45.2,
      "p50": 38.5,
      "p95": 82.1,
      "p99": 98.7
    }
  }
}
```

---

## Collecting and Analyzing Metrics

### Continuous monitoring (local)

```bash
chmod +x collect_metrics.sh

./collect_metrics.sh \
  --nodes 4 \
  --http-base-port 9000 \
  --interval 5 \
  --output-dir ./metrics_output
```

### Kubernetes monitoring

```bash
./collect_metrics.sh \
  --nodes 1000 \
  --k8s \
  --sample-nodes 20 \
  --interval 10 \
  --duration 300 \
  --output-dir ./metrics_k8s
```

### Output

| File          | Description                                            |
| ------------- | ------------------------------------------------------ |
| `metrics.csv` | Time-series data: TPS, latency, backlog, network stats |
| `summary.txt` | Human-readable summary with averages and peaks         |
| `raw/iter_N/` | Per-iteration raw JSON from each polled node           |

### CSV columns

```
timestamp, nodes_polled, avg_tps, avg_latency_p50_ms, avg_latency_p95_ms,
avg_latency_p99_ms, total_confirmed, total_nil, total_pending, total_missing,
avg_peers, total_bytes_sent, total_bytes_recv, avg_propagation_p50_ms,
avg_propagation_p95_ms, avg_propagation_p99_ms
```

---

## HTTP API Reference

Each Evolv-BFT node exposes the following HTTP endpoints on its admin port (default 9000):

| Endpoint            | Method   | Description                                             |
| ------------------- | -------- | ------------------------------------------------------- |
| `/metrics`          | GET      | Consensus metrics: TPS, latency, backlog, orderer stats |
| `/network`          | GET      | Network stats: peers, bytes, propagation latency        |
| `/config`           | GET      | Current epoch, validator count, quorum size             |
| `/hydra`            | GET      | Hydra dynamic membership state                          |
| `/adaptive`         | GET      | Adaptive control-plane state and last applied decision  |
| `/adaptive/context` | GET/POST | Read or inject AIoT scenario context for MARL policies  |
| `/join`             | POST     | Submit a join request for this node                     |
| `/leave`            | POST     | Submit a leave request for this node                    |

### Example: `/metrics` response

```json
{
  "global_confirmed_total": 15234,
  "global_confirmed_nil": 12,
  "throughput_tps": 105234.50,
  "latency_p50_ms": 38.5,
  "latency_p95_ms": 82.1,
  "latency_p99_ms": 98.7,
  "backlog_pending": 42,
  "backlog_missing": 0,
  "reject_total": 0
}
```

### Example: `/adaptive` response

```json
{
  "enabled": true,
  "last_decision": {
    "policy_name": "safe-baseline",
    "applied_action": {
      "committee_size": 0,
      "pacemaker_timeout_ms": 1250,
      "mempool_max_batch_txs": 1024,
      "mempool_proposal_interval_ms": 125,
      "reason": "degraded-path"
    }
  }
}
```

## Adaptive Control Plane

Evolv-BFT now includes a separate adaptive control plane that is designed to host future MARL policies such as safety-constrained FACMAC without placing RL logic directly on the consensus hot path.

### Current modes

- `off`: disable adaptive tuning entirely
- `safe-baseline`: built-in deterministic fallback policy
- `scripted`: load actions from a JSON file, useful as a bridge for external MARL inference services
- `http` / `facmac-http`: POST observations to an external MARL/FACMAC inference endpoint and decode the returned action

### CLI flags

```bash
-adaptive-enabled=true
-adaptive-policy=safe-baseline
-adaptive-interval-ms=1000
-adaptive-script=/path/to/policy.json
-adaptive-policy-url=http://127.0.0.1:18080/infer
-adaptive-trace-path=/tmp/evolvbft-trace.jsonl
```

### External MARL integration path

The intended FACMAC-style integration is:

1. Evolv-BFT exports runtime observations from consensus, orderer, Hydra, and network subsystems.
2. An external MARL trainer/inference service consumes those observations.
3. The service emits an action in the same schema exposed by the `scripted` and `http` policy bridges.
4. Evolv-BFT computes a built-in reward signal and optionally writes JSONL trajectories for offline training.
5. Evolv-BFT guardrails sanitize the action before it reaches any live engine.

This keeps safety-critical invariants in Go while allowing policy learning to evolve outside the protocol core.

### FACMAC-compatible service quick start

From the Evolv-BFT project root:

```bash
python -m unittest discover -s marl/tests -v
uvicorn marl.app:app --host 127.0.0.1 --port 18080
```

Then start Evolv-BFT with:

```bash
-adaptive-enabled=true
-adaptive-policy=facmac-http
-adaptive-policy-url=http://127.0.0.1:18080/infer
-adaptive-trace-path=/tmp/evolvbft-trace.jsonl
```

Offline training can be triggered with:

```bash
curl -X POST http://127.0.0.1:18080/train/offline \
  -H "Content-Type: application/json" \
  -d '{"trace_path":"/tmp/evolvbft-trace.jsonl"}'
```

### AIoT scenario context injection

External scenario generators can inject heterogeneity, churn, adversarial pressure, jitter, and AI workload signals:

```bash
curl -X POST http://localhost:9000/adaptive/context \
  -H "Content-Type: application/json" \
  -d '{
    "heterogeneity_score": 0.8,
    "churn_rate": 0.3,
    "adversary_score": 0.6,
    "network_jitter_ms": 45,
    "ai_load_score": 0.7
  }'
```

This context is merged into the observation that the external FACMAC service receives.

### Multi-agent observations

Evolv-BFT now includes per-instance agent observations in the payload sent to external MARL services.
Each agent corresponds to one consensus instance and currently exports:

- `instance_id`
- `epoch`
- `validator_count`
- `committee_size`
- `pacemaker_timeout_ms`
- `mempool_max_batch_txs`
- `mempool_proposal_interval_ms`

This allows FACMAC-style trainers to consume decentralized agent observations while keeping a centralized service-side critic.

### Per-agent actions

External FACMAC-style services may now return `agent_actions` in addition to the legacy top-level action fields.
Evolv-BFT routes each `agent_action` to the matching consensus instance by `instance_id`, then applies guardrails before tuning that instance.

### External MARL service lifecycle

The bundled Python service now supports:

- offline training from trace JSONL
- online replay ingestion and mini-batch updates
- checkpoint save/load
- AIoT scenario rollout driving
- curriculum-based experiment orchestration

This is enough to run a full closed loop:

1. Evolv-BFT emits traces via `-adaptive-trace-path`
2. The Python service trains from those traces
3. Evolv-BFT queries `/infer`
4. A scenario runner keeps injecting heterogeneous/dynamic/adversarial context

---

## Scaling Considerations

### Network topology

- **P2P**: Evolv-BFT uses libp2p with GossipSub for message propagation. At 1000 nodes, each node maintains connections to ~20-50 peers (GossipSub mesh), not all 999.
- **Discovery**: Nodes discover each other via libp2p's DHT and mDNS. In Kubernetes, the headless service provides DNS-based discovery.

### Resource tuning for large clusters

| Parameter           | Small (4-10) | Medium (100) | Large (1000) |
| ------------------- | ------------ | ------------ | ------------ |
| `instances`         | 5            | 5            | 5            |
| `batch-txs`         | 2048         | 2048         | 2048         |
| `timeout-ms`        | 1000         | 1000         | 1000         |
| `inbound-msg-queue` | 4096         | 4096         | 8192         |
| `inbound-tx-queue`  | 32768        | 32768        | 65536        |
| CPU limit           | 1000m        | 500m         | 500m         |
| Memory limit        | 512Mi        | 512Mi        | 512Mi        |

### StatefulSet considerations

- **`podManagementPolicy: Parallel`**: All 1000 pods start simultaneously, dramatically reducing rollout time.
- **`maxUnavailable: 50`**: Rolling updates replace 50 pods at a time.
- **Anti-affinity**: `preferredDuringScheduling` spreads pods across nodes without blocking scheduling.
- **Topology spread**: Ensures even distribution across hosts and zones.

---

## Expected Performance Characteristics

Based on the Evolv-BFT protocol design (dynamic pipelined multi-leader):

| Metric                   | Target        | Notes                                      |
| ------------------------ | ------------- | ------------------------------------------ |
| Throughput               | ≥100,000 tx/s | With 5 parallel instances × 2048 batch     |
| End-to-end latency (p50) | ≤50 ms        | Single-round HotStuff within each instance |
| End-to-end latency (p99) | ≤100 ms       | Under stable conditions                    |
| Node count               | 1000          | With f ≤ 333 Byzantine faults (n ≥ 3f+1)   |
| Recovery time            | ≤2s           | After leader failure (pacemaker timeout)   |
| Dynamic membership       | Online        | Join/leave via Hydra TCM without halting   |

### Key protocol features

1. **Multi-leader**: Multiple instances run in parallel, each with independent leader rotation
2. **Pipelined**: Proposals are pipelined across views for minimal latency
3. **Global ordering**: The `GlobalOrderer` merges outputs from all instances into a total order
4. **Dynamic membership**: Hydra manager handles join/leave without consensus interruption
5. **GossipSub**: Efficient O(log N) message propagation via libp2p

---

## Troubleshooting

### Pods stuck in Pending

```bash
kubectl describe pod evolvbft-0  # Check events
kubectl get events --sort-by='.lastTimestamp'
```

Common causes:
- Insufficient cluster resources → scale up worker nodes
- PVC provisioning failure → check StorageClass

### Nodes not discovering peers

```bash
kubectl exec -it evolvbft-0 -- curl -s http://localhost:9000/network | jq '.connection'
```

- Check DNS resolution: `kubectl exec evolvbft-0 -- nslookup evolvbft-headless`
- Verify headless service: `kubectl get svc evolvbft-headless`
- Ensure `publishNotReadyAddresses: true` is set in the Service

### Low throughput

1. Check backlog: `curl -s http://localhost:9000/metrics | jq '.backlog_pending'`
2. Check reject stats: `curl -s http://localhost:9000/metrics | jq '.reject_by_reason'`
3. Check network propagation: `curl -s http://localhost:9000/network | jq '.propagation_p95_ms'`
4. Increase `batch-txs` or `instances` in the ConfigMap

### High latency

1. Check if the pacemaker is timing out frequently (high `global_confirmed_nil` count)
2. Verify network latency between nodes
3. Consider reducing `timeout-ms` for faster view changes
4. Check CPU throttling: `kubectl top pods -l app=evolvbft --sort-by=cpu`
