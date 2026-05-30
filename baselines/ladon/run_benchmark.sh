#!/bin/bash
# Ladon Benchmark Runner
# Runs Ladon multi-leader BFT under identical workload to Octopus.
#
# Usage: ./run_benchmark.sh [n_replicas] [network] [output_dir]
#
# Workload parameters (identical to Octopus):
#   - Batch size: 512 KB
#   - Payload: 64 B per transaction
#   - Duration: 300s per trial
#   - Trials: 3
#   - Signatures: Ed25519
#   - Ladon-specific: 4 parallel instances (as in EuroSys'24 eval)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LADON_DIR="${SCRIPT_DIR}/ladon"
OUTPUT_DIR="${3:-${SCRIPT_DIR}/../results/ladon}"

N_REPLICAS="${1:-4}"
NETWORK="${2:-wan}"
N_TRIALS=3
BATCH_SIZE=524288
TX_SIZE=64
DURATION=300
N_INSTANCES=4  # Ladon multi-leader instances

# Network emulation
if [ "${NETWORK}" = "wan" ]; then
    LATENCY_MS=100
    echo "=== WAN mode: ${LATENCY_MS}ms RTT ==="
else
    LATENCY_MS=1
    echo "=== LAN mode: ${LATENCY_MS}ms RTT ==="
fi

echo "=== Ladon Benchmark: n=${N_REPLICAS}, ${NETWORK}, ${N_TRIALS} trials ==="
echo "    Instances: ${N_INSTANCES}, Batch: ${BATCH_SIZE}B, Payload: ${TX_SIZE}B"

mkdir -p "${OUTPUT_DIR}"

# Generate configuration
CONFIG_DIR="${OUTPUT_DIR}/config_n${N_REPLICAS}"
mkdir -p "${CONFIG_DIR}"
python3 "${SCRIPT_DIR}/gen_config.py" \
    --nodes "${N_REPLICAS}" \
    --instances "${N_INSTANCES}" \
    --output-dir "${CONFIG_DIR}"

for trial in $(seq 1 ${N_TRIALS}); do
    echo ""
    echo "--- Trial ${trial}/${N_TRIALS} ---"
    TRIAL_DIR="${OUTPUT_DIR}/n${N_REPLICAS}_${NETWORK}_trial${trial}"
    mkdir -p "${TRIAL_DIR}"

    # Apply network emulation
    if [ "${NETWORK}" = "wan" ]; then
        sudo tc qdisc add dev eth0 root netem delay ${LATENCY_MS}ms 10ms \
            rate 1000mbit 2>/dev/null || true
    fi

    # Start Ladon servers
    echo "  Starting ${N_REPLICAS} Ladon servers (${N_INSTANCES} instances each)..."
    PIDS=()
    for i in $(seq 0 $((N_REPLICAS - 1))); do
        "${LADON_DIR}/bin/ladon-server" \
            --config "${CONFIG_DIR}/node_${i}.toml" \
            --id "${i}" \
            > "${TRIAL_DIR}/server_${i}.log" 2>&1 &
        PIDS+=($!)
    done

    sleep 5

    # Run benchmark client
    echo "  Running benchmark client (${DURATION}s)..."
    "${LADON_DIR}/bin/ladon-client" \
        --config "${CONFIG_DIR}/client.toml" \
        --size "${TX_SIZE}" \
        --duration "${DURATION}" \
        --json \
        > "${TRIAL_DIR}/benchmark.json" 2>&1

    # Cleanup
    for pid in "${PIDS[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null

    if [ "${NETWORK}" = "wan" ]; then
        sudo tc qdisc del dev eth0 root 2>/dev/null || true
    fi

    # Parse
    if [ -f "${TRIAL_DIR}/benchmark.json" ]; then
        python3 "${SCRIPT_DIR}/parse_results.py" \
            --input "${TRIAL_DIR}/benchmark.json" \
            --output "${TRIAL_DIR}/summary.json" \
            --protocol "Ladon" \
            --replicas "${N_REPLICAS}" \
            --network "${NETWORK}" \
            --trial "${trial}"
    fi
done

# Aggregate
echo ""
echo "=== Aggregating ==="
python3 "${SCRIPT_DIR}/../bullshark/aggregate_trials.py" \
    --input-dir "${OUTPUT_DIR}" \
    --pattern "n${N_REPLICAS}_${NETWORK}_trial*/summary.json" \
    --output "${OUTPUT_DIR}/n${N_REPLICAS}_${NETWORK}_aggregate.json"

echo "=== Done ==="
