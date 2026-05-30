#!/bin/bash
# ============================================================================
# Octopus BFT — EC2 User Data (cloud-init)
# ============================================================================
# This script runs on first boot of each EC2 instance. It:
#   1. Installs Docker + downloads the Octopus image
#   2. Applies NetEm WAN emulation (if configured)
#   3. Waits for the manifest volume/S3 bucket to be populated
#   4. Starts ${replicas_per_vm} Octopus replicas
# ============================================================================

set -euo pipefail
exec > /var/log/octopus-setup.log 2>&1

VM_INDEX=${vm_index}
REPLICAS_PER_VM=${replicas_per_vm}
TOTAL_REPLICAS=${total_replicas}
INSTANCES=${instances}
BATCH_TXS=${batch_txs}
WAN_DELAY_MS=${wan_delay_ms}
BANDWIDTH_MBPS=${bandwidth_mbps}
ECR_REPO="${ecr_repo_uri}"
S3_BUCKET="${s3_manifest_bucket}"

FIRST_REPLICA_ID=$((VM_INDEX * REPLICAS_PER_VM))
LAST_REPLICA_ID=$((FIRST_REPLICA_ID + REPLICAS_PER_VM - 1))

echo "=== Octopus VM $VM_INDEX: replicas $FIRST_REPLICA_ID..$LAST_REPLICA_ID ==="

# ── 1. System Setup ────────────────────────────────────────────────────────

apt-get update -qq
apt-get install -y -qq docker.io iproute2 curl jq awscli

systemctl enable docker
systemctl start docker

# ── 2. NetEm WAN Emulation ─────────────────────────────────────────────────

IFACE=$(ip route | grep default | awk '{print $5}' | head -1)

if [[ $WAN_DELAY_MS -gt 0 && $BANDWIDTH_MBPS -gt 0 ]]; then
    # Chain: root HTB → class with rate limit → netem leaf with delay
    echo "Applying NetEm: delay ${WAN_DELAY_MS}ms + bandwidth ${BANDWIDTH_MBPS}Mbps"
    RATE_KBIT=$((BANDWIDTH_MBPS * 1000))
    BURST=$((RATE_KBIT / 8))
    tc qdisc add dev "$IFACE" root handle 1: htb default 10
    tc class add dev "$IFACE" parent 1: classid 1:10 htb rate ${RATE_KBIT}kbit burst ${BURST}kb
    tc qdisc add dev "$IFACE" parent 1:10 handle 10: netem delay ${WAN_DELAY_MS}ms 5ms distribution normal
elif [[ $WAN_DELAY_MS -gt 0 ]]; then
    echo "Applying NetEm: delay ${WAN_DELAY_MS}ms (RTT $((WAN_DELAY_MS * 2))ms)"
    tc qdisc add dev "$IFACE" root netem delay ${WAN_DELAY_MS}ms 5ms distribution normal
elif [[ $BANDWIDTH_MBPS -gt 0 ]]; then
    echo "Applying bandwidth limit: ${BANDWIDTH_MBPS}Mbps"
    RATE_KBIT=$((BANDWIDTH_MBPS * 1000))
    BURST=$((RATE_KBIT / 8))
    tc qdisc add dev "$IFACE" root tbf rate ${RATE_KBIT}kbit burst ${BURST}kb latency 50ms
fi

# ── 3. Pull Octopus Image ──────────────────────────────────────────────────

# The orchestrator pushes the image to ECR; pull from there
if [[ -n "$ECR_REPO" ]]; then
    aws ecr get-login-password --region $(curl -s http://169.254.169.254/latest/meta-data/placement/region) | \
        docker login --username AWS --password-stdin "$ECR_REPO"
    docker pull "$ECR_REPO:latest"
    docker tag "$ECR_REPO:latest" octopus-bft:latest
else
    # Fallback: image pre-loaded via AMI or transferred via S3
    if ! docker images octopus-bft:latest -q | grep -q .; then
        echo "Waiting for octopus-bft:latest image..."
        for i in $(seq 1 60); do
            if docker images octopus-bft:latest -q | grep -q .; then break; fi
            sleep 5
        done
    fi
fi

# ── 4. Wait for Manifests ──────────────────────────────────────────────────

MANIFEST_DIR="/opt/octopus/manifests"
mkdir -p "$MANIFEST_DIR"

# Download manifests from S3 if configured
if [[ -n "$S3_BUCKET" ]]; then
    echo "Downloading manifests from s3://$S3_BUCKET/manifests/"
    aws s3 sync "s3://$S3_BUCKET/manifests/" "$MANIFEST_DIR/" --quiet
fi

# Wait for our manifests to exist
echo "Waiting for manifests for replicas $FIRST_REPLICA_ID..$LAST_REPLICA_ID..."
for i in $(seq 1 120); do
    ALL_PRESENT=true
    for rid in $(seq $FIRST_REPLICA_ID $LAST_REPLICA_ID); do
        if [[ ! -f "$MANIFEST_DIR/node-$rid-manifest.json" ]]; then
            ALL_PRESENT=false
            break
        fi
    done
    if $ALL_PRESENT; then break; fi
    sleep 5
done

if ! $ALL_PRESENT; then
    echo "ERROR: Manifests not available after 10 minutes. Aborting."
    exit 1
fi

# ── 5. Start Octopus Replicas ──────────────────────────────────────────────

NETWORK_NAME="octopus-net"
docker network create "$NETWORK_NAME" 2>/dev/null || true

for rid in $(seq $FIRST_REPLICA_ID $LAST_REPLICA_ID); do
    LOCAL_OFFSET=$((rid - FIRST_REPLICA_ID))
    P2P_PORT=$((8080 + LOCAL_OFFSET))
    HTTP_PORT=$((9000 + LOCAL_OFFSET))

    echo "Starting replica $rid (P2P=$P2P_PORT, HTTP=$HTTP_PORT)"

    docker run -d \
        --name "octopus-$rid" \
        --network host \
        --restart unless-stopped \
        -v "$MANIFEST_DIR/node-$rid-manifest.json:/config/manifest.json:ro" \
        -e "OCTOPUS_NODE_ID=$rid" \
        octopus-bft:latest \
        -id=$rid \
        -port=$P2P_PORT \
        -http=$HTTP_PORT \
        -total-nodes=$TOTAL_REPLICAS \
        -initial-validators=$TOTAL_REPLICAS \
        -instances=$INSTANCES \
        -batch-txs=$BATCH_TXS \
        -timeout-ms=2000 \
        -inbound-msg-queue=8192 \
        -inbound-tx-queue=65536 \
        -orderer-pending-cap=65536 \
        -consensus-topic=octopus-consensus \
        -manifest=/config/manifest.json
done

echo "=== VM $VM_INDEX: All $REPLICAS_PER_VM replicas started ==="

# ── 6. Signal Ready ────────────────────────────────────────────────────────

# Write a ready marker for the orchestrator to poll
echo "ready" > /opt/octopus/status
curl -sf "http://localhost:9000/metrics" > /dev/null 2>&1 && \
    echo "First replica HTTP responsive" || true
