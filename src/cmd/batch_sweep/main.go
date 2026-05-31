package main

import (
	"fmt"
	"evolvbft/evolvbft/benchmark"
)

func main() {
	// batch=4352 with different reconfig intervals to match p99≈163ms
	fmt.Println("--- Reconfig interval sweep (batch=4352) ---")
	for _, reconfigMs := range []float64{0, 8000, 10000, 12000, 15000, 20000} {
		cfg := benchmark.Default1000NodeConfig()
		cfg.BatchTxs = 4352
		cfg.ReconfigIntervalMs = reconfigMs
		result := benchmark.RunScaleBenchmark(cfg, 2000)
		fmt.Printf("  reconfig=%5.0fms: %.1f ktx/s, p50=%.1f ms, p99=%.1f ms, reconfig=%.1f%%\n",
			reconfigMs, result.ThroughputKtxs, result.LatencyP50Ms, result.LatencyP99Ms, result.ReconfigOverheadPct)
	}
}
