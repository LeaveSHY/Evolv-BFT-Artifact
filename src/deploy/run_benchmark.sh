#!/bin/bash
# run_benchmark.sh — Start Octopus nodes, inject load, collect metrics.
# Orchestrates the full benchmark lifecycle.
#
# Usage:
#   ./run_benchmark.sh [experiment_name]
#   ./run_benchmark.sh wan_m10    # Run with m=10 instances
#   ./run_benchmark.sh wan_m1     # Run with m=1 instance (single BFT comparison)

set -euo pipefail

EXPERIMENT="${1:-wan_m10}"
INSTANCE_FILE="ec2_instances.json"
MAPPING_FILE="node_mapping.json"
SSH_KEY="~/.ssh/octopus-bench.pem"
SSH_USER="ec2-user"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -i $SSH_KEY"
RESULTS_DIR="results/${EXPERIMENT}_$(date +%Y%m%d_%H%M%S)"

# Experiment parameters (override per experiment name)
case "$EXPERIMENT" in
  wan_m10)
    INSTANCES=10
    BATCH_TXS=4352
    TIMEOUT_MS=500
    LOAD_RATE=400000    # 400k tx/s injection (saturate at ~320k consensus)
    DURATION=300        # 5 minutes
    WARMUP=60           # 1 minute warmup
    ;;
  wan_m1)
    INSTANCES=1
    BATCH_TXS=4352
    TIMEOUT_MS=500
    LOAD_RATE=50000
    DURATION=300
    WARMUP=60
    ;;
  wan_m5)
    INSTANCES=5
    BATCH_TXS=4352
    TIMEOUT_MS=500
    LOAD_RATE=200000
    DURATION=300
    WARMUP=60
    ;;
  *)
    echo "Unknown experiment: $EXPERIMENT"
    echo "Available: wan_m10, wan_m1, wan_m5"
    exit 1
    ;;
esac

mkdir -p "$RESULTS_DIR"

echo "=== Benchmark: $EXPERIMENT ==="
echo "  Instances: $INSTANCES"
echo "  Batch: $BATCH_TXS txs"
echo "  Load rate: $LOAD_RATE tx/s"
echo "  Duration: ${DURATION}s (+${WARMUP}s warmup)"
echo "  Results: $RESULTS_DIR/"
echo ""

# Parse instance info
mapfile -t PUBLIC_IPS < <(python3 -c "
import json
with open('$INSTANCE_FILE') as f:
    for inst in json.load(f):
        print(inst['public_ip'])
")

NODES_PER_MACHINE=25
TOTAL_NODES=$((${#PUBLIC_IPS[@]} * NODES_PER_MACHINE))

# --- Step 1: Stop any existing nodes ---
echo "--- Step 1: Stopping existing nodes ---"
for ip in "${PUBLIC_IPS[@]}"; do
  ssh $SSH_OPTS $SSH_USER@"$ip" "pkill -f 'octopus -id' || true" &
done
wait
sleep 2

# --- Step 2: Start Octopus nodes ---
echo "--- Step 2: Starting $TOTAL_NODES nodes (m=$INSTANCES) ---"

for idx in "${!PUBLIC_IPS[@]}"; do
  ip="${PUBLIC_IPS[$idx]}"
  (
    for j in $(seq 0 $((NODES_PER_MACHINE - 1))); do
      node_id=$((idx * NODES_PER_MACHINE + j))
      consensus_port=$((8080 + j))
      http_port=$((9000 + j))

      ssh $SSH_OPTS $SSH_USER@"$ip" << REMOTE_START
        cd ~/octopus
        nohup ./octopus \
          -id=$node_id \
          -port=$consensus_port \
          -http=$http_port \
          -http-listen-addr=0.0.0.0 \
          -total-nodes=$TOTAL_NODES \
          -initial-validators=$TOTAL_NODES \
          -instances=$INSTANCES \
          -batch-txs=$BATCH_TXS \
          -timeout-ms=$TIMEOUT_MS \
          -manifest=genesis.json \
          > /tmp/octopus_node_${node_id}.log 2>&1 &
REMOTE_START
    done
    echo "  Started $NODES_PER_MACHINE nodes on $ip"
  ) &
done
wait

echo "  Waiting 15s for peer connections to establish..."
sleep 15

# --- Step 3: Verify connectivity ---
echo "--- Step 3: Checking node connectivity ---"
HEALTHY=0
for ip in "${PUBLIC_IPS[@]}"; do
  STATUS=$(curl -s --connect-timeout 3 "http://$ip:9000/config" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('validators',0))" 2>/dev/null || echo 0)
  if [ "$STATUS" -gt 0 ]; then
    echo "  $ip: healthy (validators=$STATUS)"
    HEALTHY=$((HEALTHY + 1))
  else
    echo "  $ip: NOT READY"
  fi
done

if [ "$HEALTHY" -lt "${#PUBLIC_IPS[@]}" ]; then
  echo "WARNING: Not all machines healthy. Continuing anyway..."
fi

# --- Step 4: Build target list for loadgen and collector ---
TARGETS=""
for ip in "${PUBLIC_IPS[@]}"; do
  for j in $(seq 0 $((NODES_PER_MACHINE - 1))); do
    http_port=$((9000 + j))
    if [ -n "$TARGETS" ]; then TARGETS+=","; fi
    TARGETS+="http://$ip:$http_port"
  done
done

# Use a subset of nodes for load injection (4 nodes, one per region)
LOAD_TARGETS=""
for ip in "${PUBLIC_IPS[@]}"; do
  if [ -n "$LOAD_TARGETS" ]; then LOAD_TARGETS+=","; fi
  LOAD_TARGETS+="http://$ip:9000"
done

# Use first node of each machine for metrics (representative sample)
METRIC_TARGETS=""
for ip in "${PUBLIC_IPS[@]}"; do
  if [ -n "$METRIC_TARGETS" ]; then METRIC_TARGETS+=","; fi
  METRIC_TARGETS+="http://$ip:9000"
done

# --- Step 5: Start metrics collector ---
echo "--- Step 5: Starting metrics collector ---"
COLLECT_DURATION=$((DURATION + WARMUP + 30))
ssh $SSH_OPTS $SSH_USER@"${PUBLIC_IPS[0]}" << REMOTE_COLLECT &
  cd ~/octopus
  ./collect-metrics \
    -targets="$METRIC_TARGETS" \
    -duration="${COLLECT_DURATION}s" \
    -interval=5s \
    -warmup="${WARMUP}s" \
    -out=/tmp/benchmark_results.json \
    > /tmp/collect_metrics.log 2>&1
REMOTE_COLLECT
COLLECTOR_PID=$!

# --- Step 6: Start load generator ---
echo "--- Step 6: Starting load generator (rate=$LOAD_RATE tx/s) ---"
LOAD_DURATION=$((DURATION + WARMUP))
ssh $SSH_OPTS $SSH_USER@"${PUBLIC_IPS[0]}" << REMOTE_LOAD &
  cd ~/octopus
  ./loadgen \
    -targets="$LOAD_TARGETS" \
    -rate=$LOAD_RATE \
    -duration="${LOAD_DURATION}s" \
    -payload=64 \
    -workers=256 \
    -ramp=10s \
    > /tmp/loadgen.log 2>&1
REMOTE_LOAD
LOADGEN_PID=$!

echo "  Load injection running for ${LOAD_DURATION}s..."
echo "  Metrics collection running for ${COLLECT_DURATION}s..."
echo ""

# --- Step 7: Wait for completion ---
echo "--- Step 7: Waiting for benchmark to complete ---"
wait $LOADGEN_PID 2>/dev/null || true
echo "  Load generator finished."
wait $COLLECTOR_PID 2>/dev/null || true
echo "  Metrics collector finished."

# --- Step 8: Collect results ---
echo "--- Step 8: Collecting results ---"
scp $SSH_OPTS $SSH_USER@"${PUBLIC_IPS[0]}":/tmp/benchmark_results.json "$RESULTS_DIR/"
scp $SSH_OPTS $SSH_USER@"${PUBLIC_IPS[0]}":/tmp/loadgen.log "$RESULTS_DIR/"
scp $SSH_OPTS $SSH_USER@"${PUBLIC_IPS[0]}":/tmp/collect_metrics.log "$RESULTS_DIR/"

# Collect node logs (first node of each machine for debugging)
for idx in "${!PUBLIC_IPS[@]}"; do
  ip="${PUBLIC_IPS[$idx]}"
  scp $SSH_OPTS $SSH_USER@"$ip":/tmp/octopus_node_$((idx * NODES_PER_MACHINE)).log \
    "$RESULTS_DIR/node_${idx}_region.log" 2>/dev/null || true
done

# --- Step 9: Stop all nodes ---
echo "--- Step 9: Stopping nodes ---"
for ip in "${PUBLIC_IPS[@]}"; do
  ssh $SSH_OPTS $SSH_USER@"$ip" "pkill -f 'octopus -id' || true" &
done
wait

# --- Step 10: Print summary ---
echo ""
echo "=== Benchmark Complete ==="
if [ -f "$RESULTS_DIR/benchmark_results.json" ]; then
  python3 -c "
import json
with open('$RESULTS_DIR/benchmark_results.json') as f:
    r = json.load(f)
print(f\"  Throughput (avg): {r['avg_throughput_tps']:.0f} tx/s\")
print(f\"  Throughput (max): {r['max_throughput_tps']:.0f} tx/s\")
print(f\"  Latency p50:     {r['p50_latency_ms']:.1f} ms\")
print(f\"  Latency p95:     {r['p95_latency_ms']:.1f} ms\")
print(f\"  Latency p99:     {r['p99_latency_ms']:.1f} ms\")
print(f\"  Total confirmed: {r['total_confirmed']}\")
print(f\"  Nodes:           {r['num_nodes']}\")
print(f\"  Samples:         {r['num_samples']}\")
"
fi
echo "  Results saved: $RESULTS_DIR/"
echo ""
echo "To reproduce: ./run_benchmark.sh $EXPERIMENT"
