#!/bin/bash
# Ladon Baseline Setup
# Clones and builds Ladon multi-leader pipelined BFT.
#
# Prerequisites: Go 1.21+, protoc, make
# Tested on: Ubuntu 22.04 (EC2 c5.xlarge)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LADON_DIR="${SCRIPT_DIR}/ladon"
COMMIT="main"  # Pin to specific commit for reproducibility

echo "=== Ladon Setup ==="

# Clone if not present
if [ ! -d "${LADON_DIR}" ]; then
    echo "[1/3] Cloning Ladon repository..."
    git clone https://github.com/Ladon-BFT/ladon.git "${LADON_DIR}"
    cd "${LADON_DIR}"
    git checkout "${COMMIT}"
else
    echo "[1/3] Ladon directory exists, skipping clone"
    cd "${LADON_DIR}"
fi

# Build
echo "[2/3] Building Ladon..."
make build

echo "[3/3] Verifying binaries..."
ls -la bin/ladon-server
ls -la bin/ladon-client

echo ""
echo "=== Setup Complete ==="
echo "Binaries at: ${LADON_DIR}/bin/"
echo "Run benchmark with: ./run_benchmark.sh"
