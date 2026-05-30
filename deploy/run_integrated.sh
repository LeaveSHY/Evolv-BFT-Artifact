#!/usr/bin/env bash
# ============================================================================
# Octopus BFT — Integrated Go + Python MARL Cluster Runner (Linux/macOS)
# ============================================================================
# Launches the Python SFAC service, then starts N Go consensus nodes connected
# to it via the narrow HTTP interface (d=5 features per agent per epoch).
#
# Usage:
#   ./run_integrated.sh [--nodes 4] [--instances 2] [--base-port 8080]
#                       [--http-base 9000] [--marl-port 18080] [--seed localtest]
#                       [--timeout-ms 1000] [--check-secs 10]
# ============================================================================
set -euo pipefail

NODES=4
INSTANCES=2
BASE_PORT=8080
HTTP_BASE=9000
MARL_PORT=18080
SEED="localtest"
TIMEOUT_MS=1000
CHECK_SECS=10

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nodes)       NODES="$2"; shift 2;;
    --instances)   INSTANCES="$2"; shift 2;;
    --base-port)   BASE_PORT="$2"; shift 2;;
    --http-base)   HTTP_BASE="$2"; shift 2;;
    --marl-port)   MARL_PORT="$2"; shift 2;;
    --seed)        SEED="$2"; shift 2;;
    --timeout-ms)  TIMEOUT_MS="$2"; shift 2;;
    --check-secs)  CHECK_SECS="$2"; shift 2;;
    *) echo "Unknown option: $1"; exit 1;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$ROOT_DIR/src"
BUILD_DIR="$ROOT_DIR/build"
MARL_DIR="$ROOT_DIR/marl"
GENESIS_FILE="$SCRIPT_DIR/tmp-genesis-local.json"

PIDS=()

cleanup() {
  echo ""
  echo "Stopping all processes..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  echo "All processes stopped."
}
trap cleanup EXIT INT TERM

# ── Build Go binaries ──────────────────────────────────────────────────────
echo "=== Building Octopus Go binaries ==="
mkdir -p "$BUILD_DIR"
(cd "$SRC_DIR" && go build -o "$BUILD_DIR/octopus" ./cmd/octopus)
(cd "$SRC_DIR" && go build -o "$BUILD_DIR/octopus-genesis" ./cmd/octopus-genesis)

# ── Check Python environment ───────────────────────────────────────────────
echo "=== Checking Python MARL environment ==="
VENV_DIR="$ROOT_DIR/.venv_marl"
if [ -f "$VENV_DIR/bin/python" ]; then
  PYTHON_EXE="$VENV_DIR/bin/python"
  echo "Using existing venv at $VENV_DIR"
else
  PYTHON_EXE="python3"
  echo "No venv found, using system Python. Run: python3 -m venv .venv_marl && .venv_marl/bin/pip install -r marl/requirements.txt"
fi

# ── Start Python MARL service ──────────────────────────────────────────────
echo "=== Starting SFAC MARL service on port $MARL_PORT ==="
cd "$ROOT_DIR"
$PYTHON_EXE -m uvicorn marl.app:app --host 0.0.0.0 --port "$MARL_PORT" --log-level info \
  > "$SCRIPT_DIR/tmp-marl-service.log" 2>&1 &
MARL_PID=$!
PIDS+=("$MARL_PID")

# Wait for MARL service to be ready
echo "Waiting for MARL service to become ready..."
MARL_URL="http://127.0.0.1:$MARL_PORT/health"
for i in $(seq 1 30); do
  sleep 1
  if curl -sf "$MARL_URL" >/dev/null 2>&1; then
    echo "MARL service ready (attempt $i)"
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo "WARNING: MARL service did not respond after 30 seconds. Continuing anyway..."
  fi
done

# ── Generate genesis manifest ──────────────────────────────────────────────
echo "=== Generating genesis manifest (nodes=$NODES, seed=$SEED) ==="
"$BUILD_DIR/octopus-genesis" \
  -nodes="$NODES" \
  -seed="$SEED" \
  -base-host="127.0.0.1" \
  -base-port="$BASE_PORT" \
  -verbose \
  -out="$GENESIS_FILE"

echo ""

# ── Start Go consensus nodes ──────────────────────────────────────────────
MARL_INFER_URL="http://127.0.0.1:$MARL_PORT/infer"
echo "=== Starting $NODES consensus nodes (adaptive-policy=facmac-http) ==="
mkdir -p "$ROOT_DIR/traces"
for i in $(seq 0 $((NODES - 1))); do
  P2P_PORT=$((BASE_PORT + i))
  HTTP_PORT=$((HTTP_BASE + i))
  LOG_FILE="$SCRIPT_DIR/tmp-node-$i.log"

  "$BUILD_DIR/octopus" \
    -id="$i" -port="$P2P_PORT" -http="$HTTP_PORT" \
    -manifest="$GENESIS_FILE" -total-nodes="$NODES" \
    -initial-validators="$NODES" -instances="$INSTANCES" \
    -timeout-ms="$TIMEOUT_MS" \
    -adaptive-enabled \
    -adaptive-policy=facmac-http \
    -adaptive-policy-url="$MARL_INFER_URL" \
    -adaptive-interval-ms=5000 \
    -adaptive-trace-path="traces/node-$i-trace.jsonl" \
    -consensus-topic=octopus-consensus \
    > "$LOG_FILE" 2>&1 &
  NODE_PID=$!
  PIDS+=("$NODE_PID")
  echo "  Node $i: PID=$NODE_PID P2P=:$P2P_PORT HTTP=:$HTTP_PORT -> MARL=$MARL_INFER_URL"
done

# ── Health monitoring ──────────────────────────────────────────────────────
echo ""
echo "=== Integrated cluster running ==="
echo "  MARL service: http://127.0.0.1:$MARL_PORT  (PID=$MARL_PID)"
echo "  Consensus nodes: $NODES nodes on HTTP ports $HTTP_BASE-$((HTTP_BASE + NODES - 1))"
echo "  Press Ctrl+C to stop all."
echo ""

if [ "$CHECK_SECS" -gt 0 ]; then
  echo "Checking health every $CHECK_SECS seconds..."
  while true; do
    sleep "$CHECK_SECS"
    MARL_ALIVE="false"
    if curl -sf "http://127.0.0.1:$MARL_PORT/health" >/dev/null 2>&1; then
      MARL_ALIVE="true"
    fi
    NODES_ALIVE=0
    for i in $(seq 0 $((NODES - 1))); do
      if curl -sf "http://127.0.0.1:$((HTTP_BASE + i))/metrics" >/dev/null 2>&1; then
        NODES_ALIVE=$((NODES_ALIVE + 1))
      fi
    done
    echo "[$(date +%H:%M:%S)] MARL=$MARL_ALIVE  Nodes=$NODES_ALIVE/$NODES alive"

    if ! kill -0 "$MARL_PID" 2>/dev/null; then
      echo "MARL service exited. Stopping cluster."
      break
    fi
  done
else
  wait
fi
