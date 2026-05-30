#!/usr/bin/env python3
"""Parse Narwhal benchmark output into standardized result format."""

import argparse
import json
import sys
from pathlib import Path


def parse_narwhal_output(input_path: str) -> dict:
    """Parse Narwhal benchmark-client JSON output.

    Expected format from narwhal-benchmark-client:
    {
        "duration_ms": ...,
        "total_transactions": ...,
        "latency": {"p50": ..., "p99": ..., "average": ...}
    }
    """
    with open(input_path) as f:
        raw = json.load(f)

    duration_s = raw.get("duration_ms", 300000) / 1000.0
    total_tx = raw.get("total_transactions", 0)
    throughput_tps = total_tx / duration_s if duration_s > 0 else 0
    throughput_ktxs = throughput_tps / 1000.0

    latency = raw.get("latency", {})
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
    parser = argparse.ArgumentParser(description="Parse Narwhal benchmark results")
    parser.add_argument("--input", required=True, help="Input JSON from benchmark")
    parser.add_argument("--output", required=True, help="Output summary JSON")
    parser.add_argument("--protocol", default="Bullshark", help="Protocol name")
    parser.add_argument("--replicas", type=int, required=True)
    parser.add_argument("--network", required=True, choices=["wan", "lan"])
    parser.add_argument("--trial", type=int, required=True)
    args = parser.parse_args()

    try:
        parsed = parse_narwhal_output(args.input)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"ERROR: Cannot parse {args.input}: {e}", file=sys.stderr)
        sys.exit(1)

    result = {
        "protocol": args.protocol,
        "replicas": args.replicas,
        "network": args.network,
        "trial": args.trial,
        **parsed,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Parsed: {result['throughput_ktxs']:.1f} ktx/s, "
          f"p50={result['latency_p50_ms']:.1f}ms")


if __name__ == "__main__":
    main()
