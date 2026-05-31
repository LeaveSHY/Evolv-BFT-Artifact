#!/usr/bin/env bash
# ============================================================================
# Evolv-BFT — Automated Benchmark Runner
# ============================================================================
# Sends synthetic transactions to Evolv-BFT nodes and measures throughput
# and latency. Works with both Docker Compose (local) and Kubernetes.
#
# Usage:
#   ./run_benchmark.sh [OPTIONS]
#
# Options:
#   --nodes N             Number of nodes (default: 4)
#   --tps-target N        Target transactions per second (default: 10000)
#   --duration N          Benchmark duration in seconds (default: 60)
#   --tx-size N           Transaction payload size in bytes (default: 256)
#   --warmup N            Warmup period in seconds (default: 10)
#   --cooldown N          Cooldown period in seconds (default: 5)
#   --http-base-port N    Base HTTP port for local mode (default: 9000)
#   --k8s                 Use Kubernetes mode (kubectl port-forward)
#   --endpoints FILE      Read HTTP endpoints from file
#   --concurrency N       Number of parallel workers (default: 32)
#   --output FILE         Output file for results (default: benchmark_results.json)
#   --quiet               Suppress progress output
# ============================================================================

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
NODES=4
TPS_TARGET=10000
DURATION_SECONDS=60
TX_SIZE_BYTES=256
WARMUP_SECONDS=10
COOLDOWN_SECONDS=5
HTTP_BASE_PORT=9000
K8S_MODE=false
ENDPOINTS_FILE=""
CONCURRENCY=32
OUTPUT_FILE="benchmark_results.json"
QUIET=false

# ── Parse arguments ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nodes)         NODES="$2"; shift 2 ;;
        --tps-target)    TPS_TARGET="$2"; shift 2 ;;
        --duration)      DURATION_SECONDS="$2"; shift 2 ;;
        --tx-size)       TX_SIZE_BYTES="$2"; shift 2 ;;
        --warmup)        WARMUP_SECONDS="$2"; shift 2 ;;
        --cooldown)      COOLDOWN_SECONDS="$2"; shift 2 ;;
        --http-base-port) HTTP_BASE_PORT="$2"; shift 2 ;;
        --k8s)           K8S_MODE=true; shift ;;
        --endpoints)     ENDPOINTS_FILE="$2"; shift 2 ;;
        --concurrency)   CONCURRENCY="$2"; shift 2 ;;
        --output)        OUTPUT_FILE="$2"; shift 2 ;;
        --quiet)         QUIET=true; shift ;;
        -h|--help)
            head -n 22 "$0" | tail -n +2 | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# ── Helpers ─────────────────────────────────────────────────────────────────
log() {
    if [[ "$QUIET" == "false" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    fi
}

error() {
    echo "[ERROR] $*" >&2
    exit 1
}

check_deps() {
    for cmd in curl jq bc; do
        if ! command -v "$cmd" &>/dev/null; then
            error "Required command not found: $cmd"
        fi
    done
    if [[ "$K8S_MODE" == "true" ]] && ! command -v kubectl &>/dev/null; then
        error "kubectl not found (required for --k8s mode)"
    fi
}

generate_tx_payload() {
    # Generate a random binary payload file of TX_SIZE_BYTES
    local size=$1
    local payload_file="${TMPDIR:-/tmp}/evolvbft_tx_payload.bin"
    head -c "$size" /dev/urandom > "$payload_file" 2>/dev/null || \
        python3 -c "import os,sys; sys.stdout.buffer.write(os.urandom($size))" > "$payload_file"
    echo "$payload_file"
}

# ── Build endpoint list ────────────────────────────────────────────────────
build_endpoints() {
    local endpoints=()

    if [[ -n "$ENDPOINTS_FILE" ]]; then
        if [[ ! -f "$ENDPOINTS_FILE" ]]; then
            error "Endpoints file not found: $ENDPOINTS_FILE"
        fi
        while IFS= read -r line; do
            [[ -z "$line" || "$line" == \#* ]] && continue
            endpoints+=("$line")
        done < "$ENDPOINTS_FILE"
    elif [[ "$K8S_MODE" == "true" ]]; then
        # Use kubectl port-forward in background for a subset of nodes
        local forward_count=$((NODES < 20 ? NODES : 20))
        log "Setting up kubectl port-forward for ${forward_count} nodes..."

        for i in $(seq 0 $((forward_count - 1))); do
            local local_port=$((30000 + i))
            kubectl port-forward "pod/evolvbft-${i}" "${local_port}:9000" \
                --namespace=default &>/dev/null &
            endpoints+=("http://localhost:${local_port}")
        done
        # Wait for port-forwards to establish
        sleep 3
    else
        for i in $(seq 0 $((NODES - 1))); do
            endpoints+=("http://localhost:$((HTTP_BASE_PORT + i))")
        done
    fi

    if [[ ${#endpoints[@]} -eq 0 ]]; then
        error "No endpoints available"
    fi

    printf '%s\n' "${endpoints[@]}"
}

# ── Wait for cluster readiness ──────────────────────────────────────────────
wait_for_cluster() {
    local endpoints=("$@")
    log "Waiting for cluster readiness..."

    local ready=0
    local attempts=0
    local max_attempts=60

    while [[ $ready -lt 1 && $attempts -lt $max_attempts ]]; do
        for ep in "${endpoints[@]}"; do
            if curl -sf "${ep}/metrics" --connect-timeout 2 --max-time 5 &>/dev/null; then
                ready=$((ready + 1))
                break
            fi
        done
        if [[ $ready -lt 1 ]]; then
            attempts=$((attempts + 1))
            sleep 2
        fi
    done

    if [[ $ready -lt 1 ]]; then
        error "Cluster not ready after $((max_attempts * 2)) seconds"
    fi

    # Count how many nodes are actually reachable
    local reachable=0
    for ep in "${endpoints[@]}"; do
        if curl -sf "${ep}/metrics" --connect-timeout 2 --max-time 5 &>/dev/null; then
            reachable=$((reachable + 1))
        fi
    done
    log "Cluster ready: ${reachable}/${#endpoints[@]} endpoints reachable"
}

# ── Snapshot metrics from a node ────────────────────────────────────────────
snapshot_metrics() {
    local endpoint="$1"
    curl -sf "${endpoint}/metrics" --connect-timeout 5 --max-time 10 2>/dev/null || echo "{}"
}

# ── Single TX sender worker ────────────────────────────────────────────────
# Sends transactions in a tight loop, recording latencies to a temp file
send_tx_worker() {
    local worker_id="$1"
    local endpoints_str="$2"
    local tx_count="$3"
    local tx_payload_file="$4"
    local latency_file="$5"

    IFS='|' read -ra eps <<< "$endpoints_str"
    local num_eps=${#eps[@]}

    for i in $(seq 1 "$tx_count"); do
        local ep_idx=$((RANDOM % num_eps))
        local endpoint="${eps[$ep_idx]}"

        local start_ns
        start_ns=$(date +%s%N 2>/dev/null || python3 -c "import time; print(int(time.time()*1e9))")

        local http_code
        http_code=$(curl -sf -o /dev/null -w '%{http_code}' \
            -X POST "${endpoint}/tx" \
            -H "Content-Type: application/octet-stream" \
            --data-binary "@${tx_payload_file}" \
            --connect-timeout 2 --max-time 5 2>/dev/null || echo "000")

        local end_ns
        end_ns=$(date +%s%N 2>/dev/null || python3 -c "import time; print(int(time.time()*1e9))")

        local latency_us=$(( (end_ns - start_ns) / 1000 ))
        echo "${latency_us} ${http_code}" >> "$latency_file"
    done
}

# ── Main benchmark loop ────────────────────────────────────────────────────
run_benchmark() {
    local endpoints=("$@")
    local num_endpoints=${#endpoints[@]}

    log "============================================================"
    log "  Evolv-BFT Benchmark"
    log "============================================================"
    log "  Nodes:          ${NODES}"
    log "  Endpoints:      ${num_endpoints}"
    log "  TPS target:     ${TPS_TARGET}"
    log "  Duration:       ${DURATION_SECONDS}s"
    log "  TX size:        ${TX_SIZE_BYTES} bytes"
    log "  Warmup:         ${WARMUP_SECONDS}s"
    log "  Cooldown:       ${COOLDOWN_SECONDS}s"
    log "  Concurrency:    ${CONCURRENCY}"
    log "============================================================"

    # Generate TX payload once (binary file)
    local tx_payload_file
    tx_payload_file=$(generate_tx_payload "$TX_SIZE_BYTES")

    # Calculate TXs per worker
    local total_txs=$((TPS_TARGET * DURATION_SECONDS))
    local txs_per_worker=$(( (total_txs + CONCURRENCY - 1) / CONCURRENCY ))
    log "Total TXs to send: ${total_txs} (${txs_per_worker}/worker × ${CONCURRENCY} workers)"

    # Capture pre-benchmark metrics
    log "Capturing pre-benchmark metrics..."
    local pre_metrics
    pre_metrics=$(snapshot_metrics "${endpoints[0]}")

    # Warmup phase
    log "Warmup phase (${WARMUP_SECONDS}s)..."
    local warmup_txs=$((TPS_TARGET * WARMUP_SECONDS / CONCURRENCY))
    local warmup_dir
    warmup_dir=$(mktemp -d)
    local eps_joined
    eps_joined=$(IFS='|'; echo "${endpoints[*]}")

    for w in $(seq 1 "$CONCURRENCY"); do
        send_tx_worker "$w" "$eps_joined" "$warmup_txs" "$tx_payload_file" \
            "${warmup_dir}/worker_${w}.lat" &
    done
    wait
    rm -rf "$warmup_dir"
    log "Warmup complete"

    # Main benchmark phase
    log "Starting benchmark (${DURATION_SECONDS}s)..."
    local bench_dir
    bench_dir=$(mktemp -d)
    local start_time
    start_time=$(date +%s)

    for w in $(seq 1 "$CONCURRENCY"); do
        send_tx_worker "$w" "$eps_joined" "$txs_per_worker" "$tx_payload_file" \
            "${bench_dir}/worker_${w}.lat" &
    done
    wait

    local end_time
    end_time=$(date +%s)
    local actual_duration=$((end_time - start_time))
    log "Benchmark phase complete (actual duration: ${actual_duration}s)"

    # Cooldown
    log "Cooldown phase (${COOLDOWN_SECONDS}s)..."
    sleep "$COOLDOWN_SECONDS"

    # Capture post-benchmark metrics
    log "Capturing post-benchmark metrics..."
    local post_metrics
    post_metrics=$(snapshot_metrics "${endpoints[0]}")

    # ── Analyze results ──────────────────────────────────────────────────
    log "Analyzing results..."

    # Merge all latency files
    local all_latencies="${bench_dir}/all_latencies.txt"
    cat "${bench_dir}"/worker_*.lat > "$all_latencies" 2>/dev/null || touch "$all_latencies"

    local total_sent
    total_sent=$(wc -l < "$all_latencies" 2>/dev/null || echo "0")
    total_sent=$(echo "$total_sent" | tr -d ' ')

    local total_success
    total_success=$(awk '$2 == 200 { count++ } END { print count+0 }' "$all_latencies" 2>/dev/null || echo "0")

    local total_failed
    total_failed=$((total_sent - total_success))

    # Calculate achieved TPS
    local achieved_tps="0"
    if [[ $actual_duration -gt 0 ]]; then
        achieved_tps=$(echo "scale=2; $total_sent / $actual_duration" | bc 2>/dev/null || echo "0")
    fi

    # Calculate latency percentiles (in microseconds)
    local p50=0 p95=0 p99=0 avg=0 min_lat=0 max_lat=0
    if [[ $total_sent -gt 0 ]]; then
        local sorted_lats="${bench_dir}/sorted_lats.txt"
        awk '{print $1}' "$all_latencies" | sort -n > "$sorted_lats"

        local lines
        lines=$(wc -l < "$sorted_lats" | tr -d ' ')
        local p50_idx=$(( (lines * 50) / 100 + 1 ))
        local p95_idx=$(( (lines * 95) / 100 + 1 ))
        local p99_idx=$(( (lines * 99) / 100 + 1 ))

        p50=$(sed -n "${p50_idx}p" "$sorted_lats" 2>/dev/null || echo "0")
        p95=$(sed -n "${p95_idx}p" "$sorted_lats" 2>/dev/null || echo "0")
        p99=$(sed -n "${p99_idx}p" "$sorted_lats" 2>/dev/null || echo "0")
        min_lat=$(head -1 "$sorted_lats" 2>/dev/null || echo "0")
        max_lat=$(tail -1 "$sorted_lats" 2>/dev/null || echo "0")
        avg=$(awk '{ sum += $1; n++ } END { if(n>0) printf "%.0f", sum/n; else print 0 }' "$sorted_lats")
    fi

    # Convert microseconds to milliseconds for display
    local p50_ms p95_ms p99_ms avg_ms min_ms max_ms
    p50_ms=$(echo "scale=2; ${p50:-0} / 1000" | bc 2>/dev/null || echo "0")
    p95_ms=$(echo "scale=2; ${p95:-0} / 1000" | bc 2>/dev/null || echo "0")
    p99_ms=$(echo "scale=2; ${p99:-0} / 1000" | bc 2>/dev/null || echo "0")
    avg_ms=$(echo "scale=2; ${avg:-0} / 1000" | bc 2>/dev/null || echo "0")
    min_ms=$(echo "scale=2; ${min_lat:-0} / 1000" | bc 2>/dev/null || echo "0")
    max_ms=$(echo "scale=2; ${max_lat:-0} / 1000" | bc 2>/dev/null || echo "0")

    # Extract consensus-level TPS from node metrics
    local consensus_tps
    consensus_tps=$(echo "$post_metrics" | jq -r '.throughput_tps // 0' 2>/dev/null || echo "0")

    # ── Print results ──────────────────────────────────────────────────
    log ""
    log "============================================================"
    log "  BENCHMARK RESULTS"
    log "============================================================"
    log "  Duration:            ${actual_duration}s"
    log "  TXs sent:            ${total_sent}"
    log "  TXs succeeded:       ${total_success}"
    log "  TXs failed:          ${total_failed}"
    log "  Success rate:        $(echo "scale=1; ${total_success} * 100 / (${total_sent} + 1)" | bc 2>/dev/null || echo "N/A")%"
    log "  Achieved TPS:        ${achieved_tps} tx/s (client-side)"
    log "  Consensus TPS:       ${consensus_tps} tx/s (node-reported)"
    log "  ─────────────────────────────────────────────────────────"
    log "  Latency (client-side round-trip):"
    log "    avg:               ${avg_ms} ms"
    log "    p50:               ${p50_ms} ms"
    log "    p95:               ${p95_ms} ms"
    log "    p99:               ${p99_ms} ms"
    log "    min:               ${min_ms} ms"
    log "    max:               ${max_ms} ms"
    log "  ─────────────────────────────────────────────────────────"
    log "  Consensus metrics (from node):"
    log "    confirmed total:   $(echo "$post_metrics" | jq -r '.global_confirmed_total // "N/A"' 2>/dev/null)"
    log "    confirmed nil:     $(echo "$post_metrics" | jq -r '.global_confirmed_nil // "N/A"' 2>/dev/null)"
    log "    latency p50:       $(echo "$post_metrics" | jq -r '.latency_p50_ms // "N/A"' 2>/dev/null) ms"
    log "    latency p95:       $(echo "$post_metrics" | jq -r '.latency_p95_ms // "N/A"' 2>/dev/null) ms"
    log "    latency p99:       $(echo "$post_metrics" | jq -r '.latency_p99_ms // "N/A"' 2>/dev/null) ms"
    log "    backlog pending:   $(echo "$post_metrics" | jq -r '.backlog_pending // "N/A"' 2>/dev/null)"
    log "============================================================"

    # ── Write JSON results ──────────────────────────────────────────────
    cat > "$OUTPUT_FILE" <<ENDJSON
{
  "benchmark": {
    "timestamp": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
    "nodes": ${NODES},
    "tps_target": ${TPS_TARGET},
    "duration_seconds": ${actual_duration},
    "tx_size_bytes": ${TX_SIZE_BYTES},
    "concurrency": ${CONCURRENCY}
  },
  "results": {
    "total_sent": ${total_sent},
    "total_success": ${total_success},
    "total_failed": ${total_failed},
    "achieved_tps": ${achieved_tps},
    "consensus_tps": ${consensus_tps},
    "latency_ms": {
      "avg": ${avg_ms},
      "p50": ${p50_ms},
      "p95": ${p95_ms},
      "p99": ${p99_ms},
      "min": ${min_ms},
      "max": ${max_ms}
    }
  },
  "node_metrics_pre": ${pre_metrics:-"{}"},
  "node_metrics_post": ${post_metrics:-"{}"}
}
ENDJSON
    log "Results written to: ${OUTPUT_FILE}"

    # Cleanup
    rm -rf "$bench_dir"
    rm -f "$tx_payload_file"

    # Kill any background port-forwards
    if [[ "$K8S_MODE" == "true" ]]; then
        jobs -p | xargs kill 2>/dev/null || true
    fi
}

# ── Entry point ─────────────────────────────────────────────────────────────
main() {
    check_deps

    log "Building endpoint list..."
    local endpoints_list
    endpoints_list=$(build_endpoints)

    local endpoints=()
    while IFS= read -r line; do
        [[ -n "$line" ]] && endpoints+=("$line")
    done <<< "$endpoints_list"

    log "Endpoints: ${#endpoints[@]}"

    wait_for_cluster "${endpoints[@]}"
    run_benchmark "${endpoints[@]}"
}

main "$@"
