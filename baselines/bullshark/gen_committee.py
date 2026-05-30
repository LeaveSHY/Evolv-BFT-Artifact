#!/usr/bin/env python3
"""Generate Narwhal committee configuration for N nodes."""

import argparse
import json
import os
from pathlib import Path


def generate_committee(n_nodes: int, base_port: int = 9000) -> dict:
    """Generate a committee configuration for Narwhal-Bullshark.

    Each node gets:
    - Primary address (for consensus messages)
    - Worker address (for transaction submission)
    - Ed25519 keypair placeholder (generated at runtime)
    """
    authorities = {}
    for i in range(n_nodes):
        port_base = base_port + i * 10
        node_id = f"node_{i}"
        authorities[node_id] = {
            "stake": 1,
            "primary": {
                "primary_to_primary": f"127.0.0.1:{port_base}",
                "worker_to_primary": f"127.0.0.1:{port_base + 1}",
            },
            "workers": {
                "0": {
                    "transactions": f"127.0.0.1:{port_base + 2}",
                    "worker_to_worker": f"127.0.0.1:{port_base + 3}",
                }
            },
        }

    return {
        "authorities": authorities,
        "epoch": 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate Narwhal committee config")
    parser.add_argument("--nodes", type=int, required=True, help="Number of nodes")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--base-port", type=int, default=9000, help="Base port")
    args = parser.parse_args()

    committee = generate_committee(args.nodes, args.base_port)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(committee, f, indent=2)

    print(f"Generated committee for {args.nodes} nodes: {args.output}")


if __name__ == "__main__":
    main()
