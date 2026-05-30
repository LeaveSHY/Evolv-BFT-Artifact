#!/bin/bash
# Bullshark Baseline Setup
# Clones and builds the Narwhal-Bullshark framework for benchmark comparison.
#
# Prerequisites: Rust toolchain (1.70+), protobuf-compiler, clang, pkg-config
# Tested on: Ubuntu 22.04 (EC2 c5.xlarge)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BULLSHARK_DIR="${SCRIPT_DIR}/narwhal"
COMMIT="4a5b2e1"  # Pinned commit for reproducibility

echo "=== Bullshark (Narwhal) Setup ==="

# Clone if not present
if [ ! -d "${BULLSHARK_DIR}" ]; then
    echo "[1/3] Cloning Narwhal repository..."
    git clone https://github.com/MystenLabs/sui.git "${BULLSHARK_DIR}" --depth 50
    cd "${BULLSHARK_DIR}"
    git checkout "${COMMIT}"
else
    echo "[1/3] Narwhal directory exists, skipping clone"
    cd "${BULLSHARK_DIR}"
fi

# Build
echo "[2/3] Building Narwhal-Bullshark benchmark binary..."
cd narwhal
cargo build --release --bin narwhal-benchmark-client
cargo build --release --bin narwhal-node

echo "[3/3] Verifying binaries..."
ls -la target/release/narwhal-benchmark-client
ls -la target/release/narwhal-node

echo ""
echo "=== Setup Complete ==="
echo "Binaries at: ${BULLSHARK_DIR}/narwhal/target/release/"
echo "Run benchmark with: ./run_benchmark.sh"
