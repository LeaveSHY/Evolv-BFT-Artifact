#!/bin/bash
# setup_nodes.sh — Install Go, build Octopus, generate manifest, distribute to nodes.
# Reads ec2_instances.json from deploy_ec2.sh output.
#
# Usage:
#   ./setup_nodes.sh

set -euo pipefail

INSTANCE_FILE="ec2_instances.json"
GO_VERSION="1.22.4"
SSH_KEY="~/.ssh/octopus-bench.pem"
SSH_USER="ec2-user"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -i $SSH_KEY"
NODES_PER_REGION=25
BASE_CONSENSUS_PORT=8080
BASE_HTTP_PORT=9000

if [ ! -f "$INSTANCE_FILE" ]; then
  echo "ERROR: $INSTANCE_FILE not found. Run deploy_ec2.sh setup first."
  exit 1
fi

# Parse instance IPs
mapfile -t PUBLIC_IPS < <(python3 -c "
import json
with open('$INSTANCE_FILE') as f:
    instances = json.load(f)
for inst in instances:
    print(inst['public_ip'])
")

mapfile -t PRIVATE_IPS < <(python3 -c "
import json
with open('$INSTANCE_FILE') as f:
    instances = json.load(f)
for inst in instances:
    print(inst['private_ip'])
")

mapfile -t REGIONS < <(python3 -c "
import json
with open('$INSTANCE_FILE') as f:
    instances = json.load(f)
for inst in instances:
    print(inst['region'])
")

NUM_MACHINES=${#PUBLIC_IPS[@]}
TOTAL_NODES=$((NUM_MACHINES * NODES_PER_REGION))

echo "=== Setup: $NUM_MACHINES machines, $TOTAL_NODES total nodes ==="

# Step 1: Install Go on all machines (parallel)
echo ""
echo "--- Step 1: Installing Go $GO_VERSION ---"
for ip in "${PUBLIC_IPS[@]}"; do
  (
    echo "  Installing Go on $ip..."
    ssh $SSH_OPTS $SSH_USER@"$ip" << 'REMOTE_INSTALL'
      if ! command -v go &>/dev/null || [[ "$(go version)" != *"go1.22"* ]]; then
        sudo yum install -y git wget tar gzip > /dev/null 2>&1
        wget -q "https://go.dev/dl/go1.22.4.linux-amd64.tar.gz" -O /tmp/go.tar.gz
        sudo rm -rf /usr/local/go
        sudo tar -C /usr/local -xzf /tmp/go.tar.gz
        echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
        echo 'export GOPATH=$HOME/go' >> ~/.bashrc
        echo 'export PATH=$PATH:$HOME/go/bin' >> ~/.bashrc
      fi
      source ~/.bashrc
      go version
REMOTE_INSTALL
  ) &
done
wait
echo "  Go installed on all machines."

# Step 2: Upload source and build
echo ""
echo "--- Step 2: Building Octopus binary ---"

# Create tarball of source (excluding test files to speed up)
TARBALL="/tmp/octopus-src.tar.gz"
cd "$(dirname "$0")/.."
tar czf "$TARBALL" --exclude='*.exe' --exclude='.git' --exclude='$tmp' \
  go.mod go.sum octopus/ cmd/octopus/ cmd/octopus-genesis/ cmd/loadgen/ cmd/collect-metrics/

for ip in "${PUBLIC_IPS[@]}"; do
  (
    echo "  Building on $ip..."
    scp $SSH_OPTS "$TARBALL" $SSH_USER@"$ip":/tmp/octopus-src.tar.gz
    ssh $SSH_OPTS $SSH_USER@"$ip" << 'REMOTE_BUILD'
      export PATH=$PATH:/usr/local/go/bin:$HOME/go/bin
      rm -rf ~/octopus && mkdir -p ~/octopus
      cd ~/octopus
      tar xzf /tmp/octopus-src.tar.gz
      go build -o octopus ./cmd/octopus/
      go build -o octopus-genesis ./cmd/octopus-genesis/
      go build -o loadgen ./cmd/loadgen/
      go build -o collect-metrics ./cmd/collect-metrics/
      echo "Build complete: $(ls -la octopus octopus-genesis loadgen collect-metrics)"
REMOTE_BUILD
  ) &
done
wait
echo "  Binary built on all machines."

# Step 3: Generate genesis manifest
echo ""
echo "--- Step 3: Generating $TOTAL_NODES-node manifest ---"

# Generate manifest locally (needs the genesis tool)
# Build genesis tool locally
cd "$(dirname "$0")/.."
go build -o /tmp/octopus-genesis ./cmd/octopus-genesis/

# We need to create a manifest that assigns nodes to machines with correct multiaddrs.
# Each machine runs NODES_PER_REGION processes with consecutive IDs and ports.
python3 << MANIFEST_SCRIPT
import json, subprocess, sys

instances = json.load(open("deploy/$INSTANCE_FILE"))
nodes_per_region = $NODES_PER_REGION
total = len(instances) * nodes_per_region
base_port = $BASE_CONSENSUS_PORT

# Generate base manifest with octopus-genesis
result = subprocess.run(
    ["/tmp/octopus-genesis", f"-nodes={total}", "-out=/tmp/genesis_base.json"],
    capture_output=True, text=True
)
if result.returncode != 0:
    print(f"genesis failed: {result.stderr}", file=sys.stderr)
    sys.exit(1)

# Patch multiaddrs with actual private IPs
with open("/tmp/genesis_base.json") as f:
    manifest = json.load(f)

node_idx = 0
for i, inst in enumerate(instances):
    private_ip = inst["private_ip"]
    for j in range(nodes_per_region):
        node = manifest["nodes"][node_idx]
        port = base_port + j
        # Update the p2p_multiaddr to use the actual private IP
        peer_id = node.get("peer_id", "")
        node["p2p_multiaddr"] = f"/ip4/{private_ip}/tcp/{port}/p2p/{peer_id}"
        node_idx += 1

with open("/tmp/genesis_patched.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Manifest generated: {total} nodes across {len(instances)} machines")

# Also generate a node-to-machine mapping
mapping = []
node_idx = 0
for i, inst in enumerate(instances):
    for j in range(nodes_per_region):
        mapping.append({
            "node_id": node_idx,
            "machine_idx": i,
            "public_ip": inst["public_ip"],
            "private_ip": inst["private_ip"],
            "consensus_port": base_port + j,
            "http_port": $BASE_HTTP_PORT + j,
            "region": inst["region"]
        })
        node_idx += 1

with open("/tmp/node_mapping.json", "w") as f:
    json.dump(mapping, f, indent=2)
print(f"Node mapping saved: /tmp/node_mapping.json")
MANIFEST_SCRIPT

# Step 4: Distribute manifest to all machines
echo ""
echo "--- Step 4: Distributing manifest ---"
for ip in "${PUBLIC_IPS[@]}"; do
  scp $SSH_OPTS /tmp/genesis_patched.json $SSH_USER@"$ip":~/octopus/genesis.json &
done
wait
echo "  Manifest distributed."

# Save mapping locally for the run script
cp /tmp/node_mapping.json deploy/node_mapping.json
cp /tmp/genesis_patched.json deploy/genesis.json

echo ""
echo "=== Setup complete ==="
echo "  Machines: ${PUBLIC_IPS[*]}"
echo "  Nodes per machine: $NODES_PER_REGION"
echo "  Total nodes: $TOTAL_NODES"
echo "  Manifest: deploy/genesis.json"
echo "  Mapping: deploy/node_mapping.json"
echo ""
echo "Next: run './run_benchmark.sh' to start nodes and inject load"
