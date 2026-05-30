#!/usr/bin/env python3
"""Generate Ladon configuration files for N nodes with M instances."""

import argparse
import json
from pathlib import Path


def generate_config(n_nodes: int, n_instances: int, output_dir: str,
                    base_port: int = 7000):
    """Generate Ladon TOML configs for each node."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Node configs
    for i in range(n_nodes):
        port = base_port + i * 100
        config = f"""# Ladon node {i} configuration
[node]
id = {i}
n_replicas = {n_nodes}
n_instances = {n_instances}

[network]
listen_addr = "0.0.0.0:{port}"
batch_size = 524288
tx_size = 64

[consensus]
protocol = "bullshark-pipelined"
pipeline_depth = 3

[crypto]
scheme = "ed25519"

[peers]
"""
        for j in range(n_nodes):
            if j != i:
                peer_port = base_port + j * 100
                config += f'peer_{j} = "127.0.0.1:{peer_port}"\n'

        with open(output_path / f"node_{i}.toml", "w") as f:
            f.write(config)

    # Client config
    client_config = f"""# Ladon client configuration
[client]
n_replicas = {n_nodes}

[targets]
"""
    for i in range(n_nodes):
        port = base_port + i * 100 + 1
        client_config += f'node_{i} = "127.0.0.1:{port}"\n'

    with open(output_path / "client.toml", "w") as f:
        f.write(client_config)

    print(f"Generated configs for {n_nodes} nodes, {n_instances} instances")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--instances", type=int, default=4)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    generate_config(args.nodes, args.instances, args.output_dir)


if __name__ == "__main__":
    main()
