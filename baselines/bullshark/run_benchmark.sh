#!/bin/bash
# Bullshark Benchmark Runner
# Runs Narwhal-Bullshark under identical workload to Evolv-BFT.
#
# Usage: ./run_benchmark.sh [n_replicas] [network] [output_dir]
#   n_replicas: number of nodes (default: 4)
#   network:    wan|lan (default: wan)
#   output_dir: where to store results (default: ../results/bullshark/)
#
# Workload parameters (identical to Evolv-BFT for fair comparison):
#   - Batch size: 512 KB
#   - Payload: 64 B per transaction
#   - Duration: 300s (5 min) per trial
#   - Warmup: 30s
#   - Trials: 3 (three independent runs)
#   - Signatures: Ed25519

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NARWHAL_DIR="${SCRIPT_DIR}/narwhal/narwhal"
BINARY_DIR="${NARWHAL_DIR}/target/release"
OUTPUT_DIR="${3:-${SCRIPT_DIR}/../results/bullshark}"

N_REPLICAS="${1:-4}"
NETWORK="${2:-wan}"
N_TRIALS=3
BATCH_SIZE=524288    # 512 KB
TX_SIZE=64           # 64 B
DURATION=300         # 5 min
WARMUP=30

# Network emulation parameters
if [ "${NETWORK}" = "wan" ]; then
    LATENCY_MS=100
    BANDWIDTH_MBPS=1000
    echo "=== WAN mode: ${LATENCY_MS}ms RTT, ${BANDWIDTH_MBPS} Mbps ==="
else
    LATENCY_MS=1
    BANDWIDTH_MBPS=10000
    echo "=== LAN mode: ${LATENCY_MS}ms RTT, ${BANDWIDTH_MBPS} Mbps ==="
fi

echo "=== Bullshark Benchmark: n=${N_REPLICAS}, ${NETWORK}, ${N_TRIALS} trials ==="
echo "    Batch: ${BATCH_SIZE}B, Payload: ${TX_SIZE}B, Duration: ${DURATION}s"

mkdir -p "${OUTPUT_DIR}"

# Generate committee configuration
COMMITTEE_FILE="${OUTPUT_DIR}/committee_n${N_REPLICAS}.json"
python3 "${SCRIPT_DIR}/gen_committee.py" \
    --nodes "${N_REPLICAS}" \
    --output "${COMMITTEE_FILE}"

# Run trials
for trial in $(seq 1 ${N_TRIALS}); do
    echo ""
    echo "--- Trial ${trial}/${N_TRIALS} ---"
    TRIAL_DIR="${OUTPUT_DIR}/n${N_REPLICAS}_${NETWORK}_trial${trial}"
    mkdir -p "${TRIAL_DIR}"

    # Apply network emulation (tc/netem)
    if [ "${NETWORK}" = "wan" ]; then
        sudo tc qdisc add dev eth0 root netem delay ${LATENCY_MS}ms 10ms \
            rate ${BANDWIDTH_MBPS}mbit 2>/dev/null || true
    fi

    # Start Narwhal nodes
    echo "  Starting ${N_REPLICAS} Narwhal nodes..."
    PIDS=()
    for i in $(seq 0 $((N_REPLICAS - 1))); do
        PORT_BASE=$((9000 + i * 10))
        "${BINARY_DIR}/narwhal-node" \
            --committee "${COMMITTEE_FILE}" \
            --id "${i}" \
            --store "${TRIAL_DIR}/node_${i}" \
            --parameters "${SCRIPT_DIR}/parameters.json" \
            > "${TRIAL_DIR}/node_${i}.log" 2>&1 &
        PIDS+=($!)
    done

    # Wait for nodes to start
    sleep 5

    # Run benchmark client
    echo "  Running benchmark client (${DURATION}s)..."
    "${BINARY_DIR}/narwhal-benchmark-client" \
        --committee "${COMMITTEE_FILE}" \
        --size "${TX_SIZE}" \
        --rate 0 \
        --duration "${DURATION}" \
        > "${TRIAL_DIR}/benchmark.json" 2>&1

    # Collect results
    echo "  Stopping nodes..."
    for pid in "${PIDS[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null

    # Remove network emulation
    if [ "${NETWORK}" = "wan" ]; then
        sudo tc qdisc del dev eth0 root 2>/dev/null || true
    fi

    # Parse results
    if [ -f "${TRIAL_DIR}/benchmark.json" ]; then
        python3 "${SCRIPT_DIR}/parse_results.py" \
            --input "${TRIAL_DIR}/benchmark.json" \
            --output "${TRIAL_DIR}/summary.json" \
            --protocol "Bullshark" \
            --replicas "${N_REPLICAS}" \
            --network "${NETWORK}" \
            --trial "${trial}"
        echo "  Trial ${trial} complete: $(cat ${TRIAL_DIR}/summary.json | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"{d[\"throughput_ktxs\"]:.0f} ktx/s, p50={d[\"latency_p50_ms\"]:.1f}ms")')"
    else
        echo "  WARNING: Trial ${trial} produced no output"
    fi
done

# Aggregate across trials
echo ""
echo "=== Aggregating ${N_TRIALS} trials ==="
python3 "${SCRIPT_DIR}/aggregate_trials.py" \
    --input-dir "${OUTPUT_DIR}" \
    --pattern "n${N_REPLICAS}_${NETWORK}_trial*/summary.json" \
    --output "${OUTPUT_DIR}/n${N_REPLICAS}_${NETWORK}_aggregate.json"

echo "=== Done. Results: ${OUTPUT_DIR}/n${N_REPLICAS}_${NETWORK}_aggregate.json ==="
