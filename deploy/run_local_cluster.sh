#!/usr/bin/env bash
# ============================================================================
# Octopus BFT — Local Multi-Node Cluster Runner
# ============================================================================
# Generates a genesis manifest and starts N local Octopus nodes.
#
# Usage:
#   ./run_local_cluster.sh [OPTIONS]
#
# Options:
#   --nodes N         Number of nodes (default: 4)
#   --instances N     Number of parallel consensus lanes (default: 2)
#   --base-port N     Base P2P port (default: 8080)
#   --http-base N     Base HTTP admin port (default: 9000)
#   --seed S          Deterministic cluster seed (default: "localtest")
#   --timeout N       Pacemaker timeout in ms (default: 1000)
#   --check-secs N    Seconds to wait before checking committed blocks (default: 10)
#   --keep            Keep running after check (default: stop after check)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="${OCTOPUS_LOCAL_CLUSTER_OUTPUT_DIR:-$ROOT_DIR/experiments/results/local_cluster_smoke}"
SUMMARY_PATH="$RESULTS_DIR/local_cluster_smoke_summary.json"
STDOUT_PATH="$RESULTS_DIR/local_cluster_stdout.txt"
STDERR_PATH="$RESULTS_DIR/local_cluster_stderr.txt"
mkdir -p "$RESULTS_DIR"
exec > >(tee "$STDOUT_PATH") 2> >(tee "$STDERR_PATH" >&2)

write_summary() {
    local status="$1"
    local exit_code="$2"
    local error_detail="$3"
    local command_string="$4"
    local node_logs_json="${5:-[]}"
    local node_results_json="${6:-[]}"
    local node_logs_file node_results_file python_status
    node_logs_file="$(mktemp)"
    node_results_file="$(mktemp)"
    printf '%s' "$node_logs_json" >"$node_logs_file"
    printf '%s' "$node_results_json" >"$node_results_file"
    python_status=0
    python3 - <<'PY' "$SUMMARY_PATH" "$status" "$exit_code" "$error_detail" "$STDOUT_PATH" "$STDERR_PATH" "$command_string" "$node_logs_file" "$node_results_file" || python_status=$?
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

summary_path = Path(sys.argv[1])
status = sys.argv[2]
exit_code = int(sys.argv[3])
error_detail = sys.argv[4]
stdout_path = Path(sys.argv[5])
stderr_path = Path(sys.argv[6])
command_string = sys.argv[7]
node_logs = json.loads(Path(sys.argv[8]).read_text(encoding="utf-8"))
node_results = json.loads(Path(sys.argv[9]).read_text(encoding="utf-8"))
reachable_nodes = sum(1 for item in node_results if item.get("reachable"))
committed_nodes = sum(1 for item in node_results if item.get("committed_blocks_observed"))
if reachable_nodes == 0:
    truthful_outcome = "unreachable_nodes"
elif committed_nodes == len(node_results):
    truthful_outcome = "minimal_success"
elif committed_nodes == 0:
    truthful_outcome = "no_commits_observed"
else:
    truthful_outcome = "partial_commit"
verification = {
    "nodes_requested": len(node_results),
    "nodes_checked": len(node_results),
    "reachable_nodes": reachable_nodes,
    "committed_nodes": committed_nodes,
    "truthful_outcome": truthful_outcome,
    "node_results": node_results,
}
degradation = None
if status in {"failed", "partial"}:
    if error_detail.startswith("missing dependency: "):
        dependency = error_detail.split(": ", 1)[1]
        reason = f"missing_dependency:{dependency}"
    elif error_detail == "unknown option":
        reason = "invalid_usage:unknown_option"
    elif error_detail == "committed blocks not observed on all nodes within check window":
        reason = "verification_window:committed_blocks_not_observed"
    elif error_detail:
        reason = error_detail.lower().replace(" ", "_")
    else:
        reason = "unspecified_degradation"
    degradation = {
        "status": status,
        "reason": reason,
        "missing_fields": [],
    }
payload = {
    "suite": "octopus-local-cluster",
    "purpose": "minimal local multi-node smoke entrypoint (not paper-grade benchmark evidence)",
    "command": command_string,
    "status": status,
    "exit_code": exit_code,
    "timestamp": datetime.now(UTC).isoformat(),
    "stdout": stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else "",
    "stderr": stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else "",
    "error_detail": error_detail,
    "evidence_manifest": {
        "producer": "deploy/run_local_cluster.sh",
        "schema_version": "octopus-evidence-v1",
        "truth_level": "minimal_truthful_evidence",
        "claim_boundary": "local-cluster smoke evidence only; not benchmark closure or paper-grade evidence",
        "evidence_kinds": ["local_cluster_smoke"],
        "excludes": [
            "benchmark_closure",
            "attack_study_closure",
            "paper_grade_closure",
        ],
    },
    "verification": verification,
    "artifacts": {
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "node_logs": node_logs,
    },
}
if degradation is not None:
    payload["degradation"] = degradation
summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
    rm -f "$node_logs_file" "$node_results_file"
    return "$python_status"
}

# ── Defaults ────────────────────────────────────────────────────────────────
NODES=4
INSTANCES=2
BASE_PORT=8080
HTTP_BASE=9000
SEED="localtest"
TIMEOUT_MS=1000
CHECK_SECS=10
KEEP=false

# ── Parse arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nodes)       NODES="$2"; shift 2 ;;
        --instances)   INSTANCES="$2"; shift 2 ;;
        --base-port)   BASE_PORT="$2"; shift 2 ;;
        --http-base)   HTTP_BASE="$2"; shift 2 ;;
        --seed)        SEED="$2"; shift 2 ;;
        --timeout)     TIMEOUT_MS="$2"; shift 2 ;;
        --check-secs)  CHECK_SECS="$2"; shift 2 ;;
        --keep)        KEEP=true; shift ;;
        -h|--help)
            head -n 18 "$0" | tail -n +2 | sed 's/^# //' | sed 's/^#//' | tee "$STDOUT_PATH"
            : >"$STDERR_PATH"
            write_summary "usage" 0 "" "bash deploy/run_local_cluster.sh --help"
            printf 'Wrote local-cluster smoke summary to %s\n' "$SUMMARY_PATH"
            exit 0
            ;;
        *) echo "Unknown option: $1" | tee "$STDERR_PATH"; : >"$STDOUT_PATH"; write_summary "failed" 1 "unknown option" "bash deploy/run_local_cluster.sh $1"; exit 1 ;;
    esac
done

if ! command -v go >/dev/null 2>&1; then
    : >"$STDOUT_PATH"
    printf '%s\n' "missing dependency: go" >"$STDERR_PATH"
    write_summary "failed" 127 "missing dependency: go" "bash deploy/run_local_cluster.sh"
    printf 'Wrote local-cluster smoke summary to %s\n' "$SUMMARY_PATH" >&2
    exit 127
fi

SRC_DIR="$ROOT_DIR/src"
BUILD_DIR="$ROOT_DIR/build"
GENESIS_FILE="$SCRIPT_DIR/tmp-genesis-local.json"

PIDS=()

cleanup() {
    echo ""
    echo "Stopping all nodes..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    echo "All nodes stopped."
}
trap cleanup EXIT

# ── Build ───────────────────────────────────────────────────────────────────
echo "=== Building Octopus ==="
mkdir -p "$BUILD_DIR"
(cd "$SRC_DIR" && go build -o "$BUILD_DIR/octopus" ./cmd/octopus)
(cd "$SRC_DIR" && go build -o "$BUILD_DIR/octopus-genesis" ./cmd/octopus-genesis)

# ── Generate genesis manifest ───────────────────────────────────────────────
echo "=== Generating genesis manifest (nodes=$NODES, seed=$SEED) ==="
"$BUILD_DIR/octopus-genesis" \
    -nodes="$NODES" \
    -seed="$SEED" \
    -base-host="127.0.0.1" \
    -base-port="$BASE_PORT" \
    -verbose \
    -out="$GENESIS_FILE"

echo ""

# ── Start nodes ─────────────────────────────────────────────────────────────
echo "=== Starting $NODES nodes ==="
for i in $(seq 0 $((NODES - 1))); do
    P2P_PORT=$((BASE_PORT + i))
    HTTP_PORT=$((HTTP_BASE + i))
    LOG_FILE="$SCRIPT_DIR/tmp-node-${i}.log"

    "$BUILD_DIR/octopus" \
        -id="$i" \
        -port="$P2P_PORT" \
        -http="$HTTP_PORT" \
        -manifest="$GENESIS_FILE" \
        -total-nodes="$NODES" \
        -initial-validators="$NODES" \
        -instances="$INSTANCES" \
        -timeout-ms="$TIMEOUT_MS" \
        > "$LOG_FILE" 2>&1 &
    PIDS+=($!)
    echo "  Node $i: PID=${PIDS[-1]}, P2P=:$P2P_PORT, HTTP=:$HTTP_PORT, log=$LOG_FILE"
done

echo ""
echo "=== Waiting ${CHECK_SECS}s for consensus to start ==="
sleep "$CHECK_SECS"

# ── Check health ────────────────────────────────────────────────────────────
echo "=== Checking node health ==="
ALL_OK=true
NODE_RESULTS_JSON="$(python3 - <<'PY' "$NODES"
import json
import sys
print(json.dumps([
    {
        "node_id": idx,
        "reachable": False,
        "committed_blocks": None,
        "committed_blocks_observed": False,
        "metrics_excerpt": None,
    }
    for idx in range(int(sys.argv[1]))
]))
PY
)"
for i in $(seq 0 $((NODES - 1))); do
    HTTP_PORT=$((HTTP_BASE + i))
    METRICS=$(curl -s --max-time 3 "http://127.0.0.1:${HTTP_PORT}/metrics" 2>/dev/null || echo "UNREACHABLE")
    if echo "$METRICS" | grep -q "global_confirmed_total"; then
        COMMITTED=$(echo "$METRICS" | grep -o '"global_confirmed_total":[0-9]*' | grep -o '[0-9]*$' || echo "0")
        echo "  Node $i: committed_blocks=$COMMITTED"
        NODE_RESULTS_JSON="$(python3 - <<'PY' "$NODE_RESULTS_JSON" "$i" "$COMMITTED" "$METRICS"
import json
import sys

results = json.loads(sys.argv[1])
node_id = int(sys.argv[2])
committed = int(sys.argv[3])
metrics = sys.argv[4]
for item in results:
    if item["node_id"] == node_id:
        item["reachable"] = True
        item["committed_blocks"] = committed
        item["committed_blocks_observed"] = committed > 0
        item["metrics_excerpt"] = metrics[:200]
        break
print(json.dumps(results))
PY
)"
        if [[ "$COMMITTED" -eq 0 ]]; then
            ALL_OK=false
        fi
    else
        echo "  Node $i: $METRICS"
        NODE_RESULTS_JSON="$(python3 - <<'PY' "$NODE_RESULTS_JSON" "$i" "$METRICS"
import json
import sys

results = json.loads(sys.argv[1])
node_id = int(sys.argv[2])
metrics = sys.argv[3]
reachable = metrics != "UNREACHABLE"
for item in results:
    if item["node_id"] == node_id:
        item["reachable"] = reachable
        item["committed_blocks"] = None
        item["committed_blocks_observed"] = False
        item["metrics_excerpt"] = metrics[:200]
        break
print(json.dumps(results))
PY
)"
        ALL_OK=false
    fi
done

echo ""
if $ALL_OK; then
    echo "✅ All nodes have committed blocks — consensus is working!"
    FINAL_STATUS="passed"
    FINAL_EXIT_CODE=0
    FINAL_ERROR_DETAIL=""
else
    echo "⚠️  Some nodes have not committed blocks yet."
    echo "   Check logs in $SCRIPT_DIR/tmp-node-*.log"
    FINAL_STATUS="partial"
    FINAL_EXIT_CODE=0
    FINAL_ERROR_DETAIL="committed blocks not observed on all nodes within check window"
fi

NODE_LOGS_JSON="$(python3 - <<'PY' "$SCRIPT_DIR" "$NODES"
import json
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
nodes = int(sys.argv[2])
print(json.dumps([str(script_dir / f"tmp-node-{idx}.log") for idx in range(nodes)]))
PY
)"
write_summary "$FINAL_STATUS" "$FINAL_EXIT_CODE" "$FINAL_ERROR_DETAIL" "bash deploy/run_local_cluster.sh" "$NODE_LOGS_JSON" "$NODE_RESULTS_JSON"
printf 'Wrote local-cluster smoke summary to %s\n' "$SUMMARY_PATH"

if $KEEP; then
    echo ""
    echo "=== Running (Ctrl+C to stop) ==="
    wait
fi
