#!/usr/bin/env python3
"""Parse Ladon benchmark output into standardized result format."""

import argparse
import json
import sys
from pathlib import Path


def parse_ladon_output(input_path: str) -> dict:
    """Parse Ladon benchmark-client JSON output."""
    with open(input_path) as f:
        raw = json.load(f)

    duration_s = raw.get("duration_sec", 300)
    total_tx = raw.get("committed_transactions", 0)
    throughput_tps = total_tx / duration_s if duration_s > 0 else 0
    throughput_ktxs = throughput_tps / 1000.0

    latency = raw.get("latency_ms", {})
    lat_p50 = latency.get("p50", 0)
    lat_p99 = latency.get("p99", 0)

    return {
        "throughput_tps": throughput_tps,
        "throughput_ktxs": throughput_ktxs,
        "latency_p50_ms": lat_p50,
        "latency_p99_ms": lat_p99,
        "duration_s": duration_s,
        "total_transactions": total_tx,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", default="Ladon")
    parser.add_argument("--replicas", type=int, required=True)
    parser.add_argument("--network", required=True)
    parser.add_argument("--trial", type=int, required=True)
    args = parser.parse_args()

    try:
        parsed = parse_ladon_output(args.input)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    result = {
        "protocol": args.protocol,
        "replicas": args.replicas,
        "network": args.network,
        "trial": args.trial,
        **parsed,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
