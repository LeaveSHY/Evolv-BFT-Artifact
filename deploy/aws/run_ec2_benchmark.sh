#!/usr/bin/env bash
# ============================================================================
# Evolv-BFT — EC2 1000-Node Deployment & Benchmark
# ============================================================================
# Automated end-to-end script for the 1000-replica WAN experiment (§VI RQ1).
# Deploys Evolv-BFT across 100 EC2 c5.xlarge VMs with NetEm WAN emulation
# and measures consensus throughput and latency under production conditions.
#
# Steps:
#   1. Generates a 1000-node genesis manifest (stable keys)
#   2. Builds + pushes the Docker image to ECR
#   3. Provisions 100 EC2 VMs via Terraform
#   4. Uploads manifests to S3 for node bootstrap
#   5. Waits for all nodes to come online
#   6. Runs the throughput/latency benchmark
#   7. Collects results and validates paper claims
#   8. (Optional) Tears down infrastructure
#
# Prerequisites:
#   - AWS CLI configured with sufficient permissions
#   - Terraform >= 1.5 installed
#   - Go >= 1.22 installed (for building evolvbft-genesis)
#   - Docker installed (for image build)
#   - SSH key pair registered in EC2
#
# Usage:
#   ./run_ec2_benchmark.sh --key-name my-key [--region us-east-1] [--no-destroy]
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOY_DIR="$SCRIPT_DIR"
TF_DIR="$SCRIPT_DIR"
RESULTS_DIR="$ROOT_DIR/experiments/results/ec2_benchmark"

# ── Defaults ────────────────────────────────────────────────────────────────
AWS_REGION="us-east-1"
KEY_NAME=""
NUM_VMS=100
REPLICAS_PER_VM=10
INSTANCES=10
BATCH_TXS=8192
WAN_DELAY_MS=40
BENCHMARK_DURATION=120
BENCHMARK_TPS_TARGET=200000
NO_DESTROY=false
SKIP_BUILD=false

# ── Parse Arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --key-name)       KEY_NAME="$2"; shift 2 ;;
        --region)         AWS_REGION="$2"; shift 2 ;;
        --num-vms)        NUM_VMS="$2"; shift 2 ;;
        --replicas)       REPLICAS_PER_VM="$2"; shift 2 ;;
        --instances)      INSTANCES="$2"; shift 2 ;;
        --wan-delay)      WAN_DELAY_MS="$2"; shift 2 ;;
        --duration)       BENCHMARK_DURATION="$2"; shift 2 ;;
        --tps-target)     BENCHMARK_TPS_TARGET="$2"; shift 2 ;;
        --no-destroy)     NO_DESTROY=true; shift ;;
        --skip-build)     SKIP_BUILD=true; shift ;;
        -h|--help)
            head -n 27 "$0" | tail -n +2 | sed 's/^# //' | sed 's/^#//'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$KEY_NAME" ]]; then
    echo "ERROR: --key-name is required"
    exit 1
fi

TOTAL_REPLICAS=$((NUM_VMS * REPLICAS_PER_VM))
echo "============================================================"
echo "Evolv-BFT EC2 Benchmark"
echo "  Region:          $AWS_REGION"
echo "  VMs:             $NUM_VMS × $REPLICAS_PER_VM = $TOTAL_REPLICAS replicas"
echo "  Instances (m):   $INSTANCES"
echo "  WAN delay:       ${WAN_DELAY_MS}ms one-way ($((WAN_DELAY_MS * 2))ms RTT)"
echo "  Benchmark:       ${BENCHMARK_DURATION}s @ ${BENCHMARK_TPS_TARGET} tps target"
echo "============================================================"

mkdir -p "$RESULTS_DIR"

# ── Step 1: Generate Genesis Manifest ───────────────────────────────────────
echo ""
echo "=== Step 1: Generate $TOTAL_REPLICAS-node genesis manifest ==="

GENESIS_BIN="$ROOT_DIR/src/cmd/evolvbft-genesis/evolvbft-genesis"
if [[ ! -f "$GENESIS_BIN" ]]; then
    echo "Building evolvbft-genesis..."
    cd "$ROOT_DIR/src"
    go build -o "$GENESIS_BIN" ./cmd/evolvbft-genesis
    cd "$SCRIPT_DIR"
fi

MANIFEST_DIR="$RESULTS_DIR/manifests"
mkdir -p "$MANIFEST_DIR"

# Generate cluster genesis (deterministic seed for reproducibility)
GENESIS_MANIFEST="$MANIFEST_DIR/genesis.json"
"$GENESIS_BIN" \
    -nodes "$TOTAL_REPLICAS" \
    -out "$GENESIS_MANIFEST" \
    -seed "ec2-benchmark-$(date +%Y%m%d)" \
    -power 1

echo "Genesis manifest: $GENESIS_MANIFEST ($TOTAL_REPLICAS nodes)"

# Generate per-node manifests.
# NOTE: At this point we don't have EC2 private IPs yet (those come from Terraform in Step 4).
# The genesis manifest contains p2p_multiaddr entries that will be updated post-provision.
# For now, copy the full genesis as each node's manifest (each node finds itself by -id flag).
cd "$ROOT_DIR/deploy"
echo "  (Copying full genesis manifest for each node; IPs resolved post-provision)"
for i in $(seq 0 $((TOTAL_REPLICAS - 1))); do
    cp "$GENESIS_MANIFEST" "$MANIFEST_DIR/node-$i-manifest.json"
done
cd "$SCRIPT_DIR"

echo "Per-node manifests generated in $MANIFEST_DIR/"

# ── Step 2: Build + Push Docker Image ──────────────────────────────────────
echo ""
echo "=== Step 2: Build and push Docker image ==="

ECR_REPO_NAME="evolvbft"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME"

if [[ "$SKIP_BUILD" == "false" ]]; then
    # Create ECR repo if not exists
    aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$AWS_REGION" 2>/dev/null || \
        aws ecr create-repository --repository-name "$ECR_REPO_NAME" --region "$AWS_REGION" > /dev/null

    # Build image
    echo "Building Docker image..."
    docker build -t evolvbft:latest -f "$ROOT_DIR/deploy/Dockerfile" "$ROOT_DIR"

    # Push to ECR
    echo "Pushing to ECR: $ECR_URI"
    aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_URI"
    docker tag evolvbft:latest "$ECR_URI:latest"
    docker push "$ECR_URI:latest"
fi

# ── Step 3: Upload Manifests to S3 ─────────────────────────────────────────
echo ""
echo "=== Step 3: Upload manifests to S3 ==="

S3_BUCKET="evolvbft-benchmark-$ACCOUNT_ID-$AWS_REGION"
aws s3 mb "s3://$S3_BUCKET" --region "$AWS_REGION" 2>/dev/null || true
aws s3 sync "$MANIFEST_DIR/" "s3://$S3_BUCKET/manifests/" --quiet
echo "Manifests uploaded to s3://$S3_BUCKET/manifests/"

# Write config files that user_data.sh reads
mkdir -p "$RESULTS_DIR/vm-config"
echo "$ECR_URI" > "$RESULTS_DIR/vm-config/ecr_repo"
echo "$S3_BUCKET" > "$RESULTS_DIR/vm-config/s3_bucket"

# ── Step 4: Terraform Apply ────────────────────────────────────────────────
echo ""
echo "=== Step 4: Provision $NUM_VMS EC2 instances ==="

cd "$TF_DIR"
terraform init -input=false

terraform apply -auto-approve \
    -var="key_name=$KEY_NAME" \
    -var="aws_region=$AWS_REGION" \
    -var="num_vms=$NUM_VMS" \
    -var="replicas_per_vm=$REPLICAS_PER_VM" \
    -var="instance_type=c5.xlarge" \
    -var="wan_delay_ms=$WAN_DELAY_MS" \
    -var="consensus_instances=$INSTANCES" \
    -var="batch_txs=$BATCH_TXS" \
    -var="ecr_repo_uri=$ECR_URI" \
    -var="s3_manifest_bucket=$S3_BUCKET"

# Capture outputs
INSTANCE_IPS=$(terraform output -json instance_ips | jq -r '.[]')
PRIVATE_IPS=$(terraform output -json private_ips | jq -r '.[]')

echo "$INSTANCE_IPS" > "$RESULTS_DIR/instance_ips.txt"
echo "$PRIVATE_IPS" > "$RESULTS_DIR/private_ips.txt"

echo "Instances launched."

# ── Step 4b: Regenerate genesis with real private IPs and re-upload ─────────
echo ""
echo "=== Step 4b: Regenerate genesis with EC2 private IPs ==="

# Build hosts file for generate_config.py (bare-metal mode)
HOSTS_FILE="$RESULTS_DIR/ec2_hosts.txt"
echo "$PRIVATE_IPS" > "$HOSTS_FILE"

# Regenerate genesis manifest with the actual IPs
# Each VM hosts $REPLICAS_PER_VM replicas; evolvbft-genesis assigns sequential ports
"$GENESIS_BIN" \
    -nodes "$TOTAL_REPLICAS" \
    -out "$GENESIS_MANIFEST" \
    -seed "ec2-benchmark-$(date +%Y%m%d)" \
    -power 1

# Use generate_config.py with bare-metal mode to create per-node manifests
# that contain correct multiaddrs pointing to the actual EC2 private IPs
cd "$ROOT_DIR/deploy"
python3 generate_config.py \
    --nodes "$TOTAL_REPLICAS" \
    --manifest "$GENESIS_MANIFEST" \
    --output-dir "$MANIFEST_DIR" \
    --hosts "$HOSTS_FILE" \
    --base-port 8080 \
    --http-base-port 9000 \
    --instances "$INSTANCES" \
    --batch-txs "$BATCH_TXS" 2>/dev/null || {
    echo "  (generate_config.py failed; using full genesis for all nodes)"
    for i in $(seq 0 $((TOTAL_REPLICAS - 1))); do
        cp "$GENESIS_MANIFEST" "$MANIFEST_DIR/node-$i-manifest.json"
    done
}
cd "$SCRIPT_DIR"

# Re-upload updated manifests to S3
aws s3 sync "$MANIFEST_DIR/" "s3://$S3_BUCKET/manifests/" --quiet --delete
echo "Updated manifests uploaded to S3"

echo "Waiting for bootstrap..."

# ── Step 5: Wait for All Nodes Online ──────────────────────────────────────
echo ""
echo "=== Step 5: Wait for all nodes to come online ==="

MAX_WAIT=600  # 10 minutes
INTERVAL=15
ELAPSED=0

while [[ $ELAPSED -lt $MAX_WAIT ]]; do
    READY_COUNT=0
    while IFS= read -r ip; do
        # Check first replica's HTTP on each VM
        if curl -sf --connect-timeout 3 "http://$ip:9000/metrics" > /dev/null 2>&1; then
            READY_COUNT=$((READY_COUNT + 1))
        fi
    done <<< "$INSTANCE_IPS"

    echo "  [$ELAPSED s] $READY_COUNT / $NUM_VMS VMs responsive"
    if [[ $READY_COUNT -ge $NUM_VMS ]]; then
        echo "All VMs online!"
        break
    fi
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

if [[ $READY_COUNT -lt $NUM_VMS ]]; then
    echo "WARNING: Only $READY_COUNT/$NUM_VMS VMs online after ${MAX_WAIT}s. Proceeding anyway."
fi

# Additional wait for consensus stabilization
echo "Waiting 30s for consensus to stabilize..."
sleep 30

# ── Step 6: Run Benchmark ──────────────────────────────────────────────────
echo ""
echo "=== Step 6: Run benchmark (${BENCHMARK_DURATION}s, target ${BENCHMARK_TPS_TARGET} tps) ==="

# Build endpoints file
ENDPOINTS_FILE="$RESULTS_DIR/http_endpoints.txt"
> "$ENDPOINTS_FILE"
while IFS= read -r ip; do
    for offset in $(seq 0 $((REPLICAS_PER_VM - 1))); do
        echo "http://$ip:$((9000 + offset))" >> "$ENDPOINTS_FILE"
    done
done <<< "$INSTANCE_IPS"

# Run the existing benchmark script
cd "$ROOT_DIR/deploy"
./run_benchmark.sh \
    --nodes "$TOTAL_REPLICAS" \
    --endpoints "$ENDPOINTS_FILE" \
    --tps-target "$BENCHMARK_TPS_TARGET" \
    --duration "$BENCHMARK_DURATION" \
    --tx-size 64 \
    --warmup 15 \
    --cooldown 10 \
    --concurrency 128 \
    --output "$RESULTS_DIR/ec2_benchmark_results.json"

# ── Step 7: Collect Per-Node Metrics ───────────────────────────────────────
echo ""
echo "=== Step 7: Collect metrics ==="

METRICS_DIR="$RESULTS_DIR/metrics"
mkdir -p "$METRICS_DIR"

VM_IDX=0
while IFS= read -r ip; do
    for offset in $(seq 0 $((REPLICAS_PER_VM - 1))); do
        RID=$((VM_IDX * REPLICAS_PER_VM + offset))
        curl -sf "http://$ip:$((9000 + offset))/metrics" > "$METRICS_DIR/node-$RID.json" 2>/dev/null || true
    done
    VM_IDX=$((VM_IDX + 1))
done <<< "$INSTANCE_IPS"

# ── Step 8: Summarize Results ──────────────────────────────────────────────
echo ""
echo "=== Step 8: Summarize ==="

python3 - "$RESULTS_DIR" <<'PYEOF'
import json
import sys
import os
from pathlib import Path

results_dir = Path(sys.argv[1])
bench_file = results_dir / "ec2_benchmark_results.json"

if not bench_file.exists():
    print("ERROR: benchmark results not found")
    sys.exit(1)

bench = json.loads(bench_file.read_text())

# Collect node metrics
metrics_dir = results_dir / "metrics"
committed_heights = []
if metrics_dir.exists():
    for f in sorted(metrics_dir.glob("node-*.json")):
        try:
            m = json.loads(f.read_text())
            if "committed_height" in m:
                committed_heights.append(m["committed_height"])
        except Exception:
            pass

summary = {
    "experiment": "ec2_1000_node_benchmark",
    "total_replicas": bench.get("benchmark", {}).get("nodes", 0),
    "throughput_tps": bench.get("results", {}).get("achieved_tps", 0),
    "throughput_ktps": bench.get("results", {}).get("achieved_tps", 0) / 1000,
    "latency_p50_ms": bench.get("results", {}).get("latency_ms", {}).get("p50", 0),
    "latency_p95_ms": bench.get("results", {}).get("latency_ms", {}).get("p95", 0),
    "latency_p99_ms": bench.get("results", {}).get("latency_ms", {}).get("p99", 0),
    "duration_seconds": bench.get("benchmark", {}).get("duration_seconds", 0),
    "wan_rtt_ms": bench.get("benchmark", {}).get("wan_rtt_ms", 0),
    "committed_heights": {
        "min": min(committed_heights) if committed_heights else 0,
        "max": max(committed_heights) if committed_heights else 0,
        "mean": sum(committed_heights) / len(committed_heights) if committed_heights else 0,
    },
    "paper_claims_met": {
        "throughput_ge_100k": bench.get("results", {}).get("achieved_tps", 0) >= 100000,
        "latency_le_100ms": bench.get("results", {}).get("latency_ms", {}).get("p50", 999) <= 100,
    },
}

summary_file = results_dir / "ec2_benchmark_summary.json"
summary_file.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PYEOF

echo ""
echo "Results saved to: $RESULTS_DIR/"
echo "  - ec2_benchmark_results.json  (raw benchmark output)"
echo "  - ec2_benchmark_summary.json  (summary with paper claim checks)"
echo "  - metrics/                    (per-node metrics snapshots)"

# ── Step 9: Cleanup ────────────────────────────────────────────────────────
if [[ "$NO_DESTROY" == "true" ]]; then
    echo ""
    echo "=== --no-destroy specified. Infrastructure left running. ==="
    echo "    To destroy: cd $TF_DIR && terraform destroy -var='key_name=$KEY_NAME'"
else
    echo ""
    echo "=== Step 9: Destroying infrastructure ==="
    cd "$TF_DIR"
    terraform destroy -auto-approve \
        -var="key_name=$KEY_NAME" \
        -var="aws_region=$AWS_REGION" \
        -var="num_vms=$NUM_VMS" \
        -var="replicas_per_vm=$REPLICAS_PER_VM" \
        -var="wan_delay_ms=$WAN_DELAY_MS" \
        -var="consensus_instances=$INSTANCES" \
        -var="batch_txs=$BATCH_TXS" \
        -var="ecr_repo_uri=$ECR_URI" \
        -var="s3_manifest_bucket=$S3_BUCKET"

    # Cleanup S3
    aws s3 rb "s3://$S3_BUCKET" --force 2>/dev/null || true
    echo "Infrastructure destroyed."
fi

echo ""
echo "============================================================"
echo "EC2 1000-Node Benchmark Complete"
echo "============================================================"
