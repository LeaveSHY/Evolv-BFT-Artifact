#!/usr/bin/env python3
"""Aggregate multiple trial results into a single summary."""

import argparse
import glob
import json
import statistics
from pathlib import Path


def aggregate(input_dir: str, pattern: str) -> dict:
    """Aggregate trial summaries matching the glob pattern."""
    search_path = str(Path(input_dir) / pattern)
    files = sorted(glob.glob(search_path))

    if not files:
        raise FileNotFoundError(f"No files matching {search_path}")

    results = []
    for f in files:
        with open(f) as fh:
            results.append(json.load(fh))

    throughputs = [r["throughput_ktxs"] for r in results]
    lat_p50s = [r["latency_p50_ms"] for r in results]
    lat_p99s = [r["latency_p99_ms"] for r in results]

    mean_tput = statistics.mean(throughputs)
    std_tput = statistics.stdev(throughputs) if len(throughputs) > 1 else 0
    cv = std_tput / mean_tput if mean_tput > 0 else 0

    return {
        "protocol": results[0]["protocol"],
        "replicas": results[0]["replicas"],
        "network": results[0]["network"],
        "n_trials": len(results),
        "throughput_ktxs": round(mean_tput, 1),
        "throughput_std": round(std_tput, 1),
        "throughput_cv": round(cv, 4),
        "latency_p50_ms": round(statistics.mean(lat_p50s), 1),
        "latency_p99_ms": round(statistics.mean(lat_p99s), 1),
        "per_trial_throughputs": throughputs,
        "cached": True,
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate trial results")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = aggregate(args.input_dir, args.pattern)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Aggregated {result['n_trials']} trials: "
          f"{result['throughput_ktxs']} ± {result['throughput_std']} ktx/s "
          f"(CV={result['throughput_cv']:.2%})")


if __name__ == "__main__":
    main()
