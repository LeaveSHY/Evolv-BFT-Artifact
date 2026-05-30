#!/usr/bin/env bash
# ============================================================================
# Octopus BFT — Metrics Collection & Analysis Script
# ============================================================================
# Polls the /metrics and /network endpoints of N Octopus nodes at a
# configurable interval and aggregates:
#   - TPS (throughput)
#   - Latency percentiles (p50/p95/p99)
#   - Backlog (pending/missing)
#   - Network stats (peers, bytes, propagation)
#   - Hydra membership state
#
# Outputs CSV for analysis and a human-readable summary.
#
# Usage:
#   ./collect_metrics.sh [OPTIONS]
#
# Options:
#   --nodes N             Number of nodes to poll (default: 4)
#   --http-base-port N    Base HTTP port (default: 9000)
#   --interval N          Polling interval in seconds (default: 5)
#   --duration N          Collection duration in seconds (0 = indefinite, default: 0)
#   --k8s                 Use Kubernetes mode
#   --endpoints FILE      Read HTTP endpoints from file
#   --output-dir DIR      Output directory (default: ./metrics_output)
#   --sample-nodes N      Only poll N random nodes (for large clusters, default: all)
# ============================================================================

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
NODES=4
HTTP_BASE_PORT=9000
INTERVAL=5
DURATION=0
K8S_MODE=false
ENDPOINTS_FILE=""
OUTPUT_DIR="./metrics_output"
SAMPLE_NODES=0

# ── Parse arguments ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nodes)         NODES="$2"; shift 2 ;;
        --http-base-port) HTTP_BASE_PORT="$2"; shift 2 ;;
        --interval)      INTERVAL="$2"; shift 2 ;;
        --duration)      DURATION="$2"; shift 2 ;;
        --k8s)           K8S_MODE=true; shift ;;
        --endpoints)     ENDPOINTS_FILE="$2"; shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2"; shift 2 ;;
        --sample-nodes)  SAMPLE_NODES="$2"; shift 2 ;;
        -h|--help)
            head -n 24 "$0" | tail -n +2 | sed 's/^# //' | sed 's/^#//'
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
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
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
        # Set up port-forwards for sampling
        local poll_count=$NODES
        if [[ $SAMPLE_NODES -gt 0 && $SAMPLE_NODES -lt $NODES ]]; then
            poll_count=$SAMPLE_NODES
        fi
        # Cap at 50 for practical port-forward limits
        if [[ $poll_count -gt 50 ]]; then
            poll_count=50
        fi

        log "Setting up kubectl port-forward for ${poll_count} nodes..."
        for i in $(seq 0 $((poll_count - 1))); do
            local node_idx=$i
            # If sampling, pick evenly spaced nodes
            if [[ $SAMPLE_NODES -gt 0 && $SAMPLE_NODES -lt $NODES ]]; then
                node_idx=$(( (i * NODES) / poll_count ))
            fi
            local local_port=$((31000 + i))
            kubectl port-forward "pod/octopus-${node_idx}" "${local_port}:9000" \
                --namespace=default &>/dev/null &
            endpoints+=("http://localhost:${local_port}")
        done
        sleep 3
    else
        local poll_count=$NODES
        if [[ $SAMPLE_NODES -gt 0 && $SAMPLE_NODES -lt $NODES ]]; then
            poll_count=$SAMPLE_NODES
        fi
        for i in $(seq 0 $((poll_count - 1))); do
            local node_idx=$i
            if [[ $SAMPLE_NODES -gt 0 && $SAMPLE_NODES -lt $NODES ]]; then
                node_idx=$(( (i * NODES) / poll_count ))
            fi
            endpoints+=("http://localhost:$((HTTP_BASE_PORT + node_idx))")
        done
    fi

    printf '%s\n' "${endpoints[@]}"
}

# ── Poll a single node ─────────────────────────────────────────────────────
poll_node() {
    local endpoint="$1"
    local metrics_json
    local network_json
    local hydra_json

    metrics_json=$(curl -sf "${endpoint}/metrics" --connect-timeout 3 --max-time 5 2>/dev/null || echo '{}')
    network_json=$(curl -sf "${endpoint}/network" --connect-timeout 3 --max-time 5 2>/dev/null || echo '{}')
    hydra_json=$(curl -sf "${endpoint}/hydra" --connect-timeout 3 --max-time 5 2>/dev/null || echo '{}')

    # Merge into one JSON object
    echo "{\"metrics\":${metrics_json},\"network\":${network_json},\"hydra\":${hydra_json}}"
}

# ── Aggregate across nodes ──────────────────────────────────────────────────
aggregate_metrics() {
    local data_dir="$1"
    local timestamp="$2"
    local node_count=0
    local total_tps=0
    local total_confirmed=0
    local total_nil=0
    local total_pending=0
    local total_missing=0
    local total_peers=0
    local total_bytes_sent=0
    local total_bytes_recv=0
    local sum_lat_p50=0
    local sum_lat_p95=0
    local sum_lat_p99=0
    local sum_prop_p50=0
    local sum_prop_p95=0
    local sum_prop_p99=0

    for f in "${data_dir}"/node_*.json; do
        [[ -f "$f" ]] || continue
        local data
        data=$(cat "$f")

        # Skip empty/invalid
        if [[ -z "$data" || "$data" == "{}" ]]; then
            continue
        fi

        node_count=$((node_count + 1))

        # Extract metrics
        local tps confirmed nil pending missing
        tps=$(echo "$data" | jq -r '.metrics.throughput_tps // 0' 2>/dev/null || echo "0")
        confirmed=$(echo "$data" | jq -r '.metrics.global_confirmed_total // 0' 2>/dev/null || echo "0")
        nil=$(echo "$data" | jq -r '.metrics.global_confirmed_nil // 0' 2>/dev/null || echo "0")
        pending=$(echo "$data" | jq -r '.metrics.backlog_pending // 0' 2>/dev/null || echo "0")
        missing=$(echo "$data" | jq -r '.metrics.backlog_missing // 0' 2>/dev/null || echo "0")

        local lat_p50 lat_p95 lat_p99
        lat_p50=$(echo "$data" | jq -r '.metrics.latency_p50_ms // 0' 2>/dev/null || echo "0")
        lat_p95=$(echo "$data" | jq -r '.metrics.latency_p95_ms // 0' 2>/dev/null || echo "0")
        lat_p99=$(echo "$data" | jq -r '.metrics.latency_p99_ms // 0' 2>/dev/null || echo "0")

        local peers bytes_sent bytes_recv
        peers=$(echo "$data" | jq -r '.network.connection.connected_peers // 0' 2>/dev/null || echo "0")
        bytes_sent=$(echo "$data" | jq -r '.network.network.total_bytes_sent // 0' 2>/dev/null || echo "0")
        bytes_recv=$(echo "$data" | jq -r '.network.network.total_bytes_recv // 0' 2>/dev/null || echo "0")

        local prop_p50 prop_p95 prop_p99
        prop_p50=$(echo "$data" | jq -r '.network.propagation_p50_ms // 0' 2>/dev/null || echo "0")
        prop_p95=$(echo "$data" | jq -r '.network.propagation_p95_ms // 0' 2>/dev/null || echo "0")
        prop_p99=$(echo "$data" | jq -r '.network.propagation_p99_ms // 0' 2>/dev/null || echo "0")

        total_tps=$(echo "$total_tps + $tps" | bc 2>/dev/null || echo "$total_tps")
        total_confirmed=$((total_confirmed + ${confirmed%.*}))
        total_nil=$((total_nil + ${nil%.*}))
        total_pending=$((total_pending + ${pending%.*}))
        total_missing=$((total_missing + ${missing%.*}))
        total_peers=$((total_peers + ${peers%.*}))
        total_bytes_sent=$((total_bytes_sent + ${bytes_sent%.*}))
        total_bytes_recv=$((total_bytes_recv + ${bytes_recv%.*}))

        sum_lat_p50=$(echo "$sum_lat_p50 + $lat_p50" | bc 2>/dev/null || echo "$sum_lat_p50")
        sum_lat_p95=$(echo "$sum_lat_p95 + $lat_p95" | bc 2>/dev/null || echo "$sum_lat_p95")
        sum_lat_p99=$(echo "$sum_lat_p99 + $lat_p99" | bc 2>/dev/null || echo "$sum_lat_p99")
        sum_prop_p50=$(echo "$sum_prop_p50 + $prop_p50" | bc 2>/dev/null || echo "$sum_prop_p50")
        sum_prop_p95=$(echo "$sum_prop_p95 + $prop_p95" | bc 2>/dev/null || echo "$sum_prop_p95")
        sum_prop_p99=$(echo "$sum_prop_p99 + $prop_p99" | bc 2>/dev/null || echo "$sum_prop_p99")
    done

    # Compute averages
    local avg_tps=0 avg_lat_p50=0 avg_lat_p95=0 avg_lat_p99=0
    local avg_prop_p50=0 avg_prop_p95=0 avg_prop_p99=0 avg_peers=0
    if [[ $node_count -gt 0 ]]; then
        avg_tps=$(echo "scale=2; $total_tps / $node_count" | bc 2>/dev/null || echo "0")
        avg_lat_p50=$(echo "scale=2; $sum_lat_p50 / $node_count" | bc 2>/dev/null || echo "0")
        avg_lat_p95=$(echo "scale=2; $sum_lat_p95 / $node_count" | bc 2>/dev/null || echo "0")
        avg_lat_p99=$(echo "scale=2; $sum_lat_p99 / $node_count" | bc 2>/dev/null || echo "0")
        avg_prop_p50=$(echo "scale=2; $sum_prop_p50 / $node_count" | bc 2>/dev/null || echo "0")
        avg_prop_p95=$(echo "scale=2; $sum_prop_p95 / $node_count" | bc 2>/dev/null || echo "0")
        avg_prop_p99=$(echo "scale=2; $sum_prop_p99 / $node_count" | bc 2>/dev/null || echo "0")
        avg_peers=$(echo "scale=0; $total_peers / $node_count" | bc 2>/dev/null || echo "0")
    fi

    # Return as CSV row
    echo "${timestamp},${node_count},${avg_tps},${avg_lat_p50},${avg_lat_p95},${avg_lat_p99},${total_confirmed},${total_nil},${total_pending},${total_missing},${avg_peers},${total_bytes_sent},${total_bytes_recv},${avg_prop_p50},${avg_prop_p95},${avg_prop_p99}"
}

# ── Print live summary ──────────────────────────────────────────────────────
print_summary() {
    local csv_row="$1"
    IFS=',' read -ra fields <<< "$csv_row"

    local ts="${fields[0]}"
    local nodes="${fields[1]}"
    local tps="${fields[2]}"
    local lat_p50="${fields[3]}"
    local lat_p95="${fields[4]}"
    local lat_p99="${fields[5]}"
    local confirmed="${fields[6]}"
    local nil="${fields[7]}"
    local pending="${fields[8]}"
    local missing="${fields[9]}"
    local peers="${fields[10]}"
    local sent="${fields[11]}"
    local recv="${fields[12]}"
    local prop_p50="${fields[13]}"
    local prop_p95="${fields[14]}"
    local prop_p99="${fields[15]}"

    log "nodes=${nodes} tps=${tps} lat_ms(p50/p95/p99)=${lat_p50}/${lat_p95}/${lat_p99} confirmed=${confirmed} nil=${nil} backlog=${pending}/${missing} peers=${peers} prop_ms=${prop_p50}/${prop_p95}/${prop_p99} bytes(s/r)=${sent}/${recv}"
}

# ── Main collection loop ───────────────────────────────────────────────────
main() {
    check_deps

    log "Building endpoint list..."
    local endpoints_list
    endpoints_list=$(build_endpoints)

    local endpoints=()
    while IFS= read -r line; do
        [[ -n "$line" ]] && endpoints+=("$line")
    done <<< "$endpoints_list"

    local num_endpoints=${#endpoints[@]}
    if [[ $num_endpoints -eq 0 ]]; then
        error "No endpoints available"
    fi

    log "Polling ${num_endpoints} endpoints every ${INTERVAL}s"

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # CSV header
    local csv_file="${OUTPUT_DIR}/metrics.csv"
    echo "timestamp,nodes_polled,avg_tps,avg_latency_p50_ms,avg_latency_p95_ms,avg_latency_p99_ms,total_confirmed,total_nil,total_pending,total_missing,avg_peers,total_bytes_sent,total_bytes_recv,avg_propagation_p50_ms,avg_propagation_p95_ms,avg_propagation_p99_ms" > "$csv_file"

    # Raw data directory
    local raw_dir="${OUTPUT_DIR}/raw"
    mkdir -p "$raw_dir"

    local start_time
    start_time=$(date +%s)
    local iteration=0

    log "Starting metrics collection → ${csv_file}"
    log "Press Ctrl+C to stop (or wait for --duration ${DURATION}s)"

    # Trap for cleanup
    trap 'cleanup' INT TERM

    while true; do
        # Check duration
        if [[ $DURATION -gt 0 ]]; then
            local elapsed=$(( $(date +%s) - start_time ))
            if [[ $elapsed -ge $DURATION ]]; then
                log "Duration reached (${DURATION}s). Stopping."
                break
            fi
        fi

        local timestamp
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        local iter_dir="${raw_dir}/iter_${iteration}"
        mkdir -p "$iter_dir"

        # Poll all endpoints in parallel
        local idx=0
        for ep in "${endpoints[@]}"; do
            (
                local data
                data=$(poll_node "$ep")
                echo "$data" > "${iter_dir}/node_${idx}.json"
            ) &
            idx=$((idx + 1))
        done
        wait

        # Aggregate
        local csv_row
        csv_row=$(aggregate_metrics "$iter_dir" "$timestamp")
        echo "$csv_row" >> "$csv_file"

        # Print live summary
        print_summary "$csv_row"

        iteration=$((iteration + 1))
        sleep "$INTERVAL"
    done

    # Final summary
    generate_final_summary "$csv_file"
}

# ── Generate final summary ──────────────────────────────────────────────────
generate_final_summary() {
    local csv_file="$1"
    local summary_file="${OUTPUT_DIR}/summary.txt"

    log ""
    log "============================================================"
    log "  METRICS COLLECTION SUMMARY"
    log "============================================================"

    local total_rows
    total_rows=$(tail -n +2 "$csv_file" | wc -l | tr -d ' ')

    if [[ $total_rows -eq 0 ]]; then
        log "  No data collected"
        return
    fi

    # Compute aggregates from CSV
    local avg_tps max_tps avg_lat_p50 avg_lat_p99
    avg_tps=$(tail -n +2 "$csv_file" | awk -F',' '{ sum += $3; n++ } END { if(n>0) printf "%.2f", sum/n; else print 0 }')
    max_tps=$(tail -n +2 "$csv_file" | awk -F',' 'BEGIN{m=0} { if($3+0>m) m=$3+0 } END { printf "%.2f", m }')
    avg_lat_p50=$(tail -n +2 "$csv_file" | awk -F',' '{ sum += $4; n++ } END { if(n>0) printf "%.2f", sum/n; else print 0 }')
    avg_lat_p99=$(tail -n +2 "$csv_file" | awk -F',' '{ sum += $6; n++ } END { if(n>0) printf "%.2f", sum/n; else print 0 }')

    local first_ts last_ts
    first_ts=$(tail -n +2 "$csv_file" | head -1 | cut -d',' -f1)
    last_ts=$(tail -n +2 "$csv_file" | tail -1 | cut -d',' -f1)

    {
        echo "============================================================"
        echo "  Octopus BFT — Metrics Collection Summary"
        echo "============================================================"
        echo "  Collection period:   ${first_ts} → ${last_ts}"
        echo "  Samples:             ${total_rows}"
        echo "  Avg TPS:             ${avg_tps}"
        echo "  Peak TPS:            ${max_tps}"
        echo "  Avg latency p50:     ${avg_lat_p50} ms"
        echo "  Avg latency p99:     ${avg_lat_p99} ms"
        echo "============================================================"
    } | tee "$summary_file"

    log "Summary written to: ${summary_file}"
    log "CSV data: ${csv_file}"
    log "Raw snapshots: ${OUTPUT_DIR}/raw/"
}

# ── Cleanup ─────────────────────────────────────────────────────────────────
cleanup() {
    log "Interrupted. Generating final summary..."
    local csv_file="${OUTPUT_DIR}/metrics.csv"
    if [[ -f "$csv_file" ]]; then
        generate_final_summary "$csv_file"
    fi
    # Kill any background port-forwards
    if [[ "$K8S_MODE" == "true" ]]; then
        jobs -p | xargs kill 2>/dev/null || true
    fi
    exit 0
}

main "$@"
