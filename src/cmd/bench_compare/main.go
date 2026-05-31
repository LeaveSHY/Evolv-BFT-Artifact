package main

import (
	"fmt"
	"evolvbft/evolvbft/benchmark"
)

func main() {
	// Old model (consensus_bench) with m=15
	cfg := benchmark.ConsensusConfig{
		NumReplicas:   1000,
		NumInstances:  15,
		FaultFraction: 0.33,
		BatchSizeKB:   512,
		PayloadBytes:  64,
		NumRounds:     500,
		Network:       benchmark.WANProfile,
		PipelineDepth: 4,
	}
	result := benchmark.RunConsensusBenchmark(cfg, "evolvbft")
	fmt.Printf("Old model (m=15): %.1f ktx/s, p50=%.1f ms, p99=%.1f ms\n",
		result.ThroughputKtxs, result.LatencyP50Ms, result.LatencyP99Ms)

	// New model with reconfig (default)
	cfg2 := benchmark.Default1000NodeConfig()
	result2 := benchmark.RunScaleBenchmark(cfg2, 500)
	fmt.Printf("New model (m=10, w/ reconfig): %.1f ktx/s, p50=%.1f ms, p99=%.1f ms, reconfig=%.1f%%\n",
		result2.ThroughputKtxs, result2.LatencyP50Ms, result2.LatencyP99Ms, result2.ReconfigOverheadPct)

	// New model without reconfig (steady state)
	cfg3 := benchmark.Default1000NodeConfig()
	cfg3.ReconfigIntervalMs = 0
	result3 := benchmark.RunScaleBenchmark(cfg3, 500)
	fmt.Printf("New model (m=10, no reconfig): %.1f ktx/s, p50=%.1f ms, p99=%.1f ms\n",
		result3.ThroughputKtxs, result3.LatencyP50Ms, result3.LatencyP99Ms)

	// Scaling sweep (steady state, no reconfig)
	fmt.Println("\n--- Scaling sweep (no reconfig) ---")
	for _, m := range []int{5, 10, 15, 20, 30} {
		cfg4 := benchmark.Default1000NodeConfig()
		cfg4.NumInstances = m
		cfg4.ReconfigIntervalMs = 0
		result4 := benchmark.RunScaleBenchmark(cfg4, 500)
		fmt.Printf("  m=%2d: %.1f ktx/s, p50=%.1f ms, p99=%.1f ms\n",
			m, result4.ThroughputKtxs, result4.LatencyP50Ms, result4.LatencyP99Ms)
	}

	// Baselines
	fmt.Println("\n--- Baselines at n=1000 ---")
	baseCfg := benchmark.ConsensusConfig{
		NumReplicas:   1000,
		NumInstances:  1,
		FaultFraction: 0.33,
		BatchSizeKB:   512,
		PayloadBytes:  64,
		NumRounds:     500,
		Network:       benchmark.WANProfile,
		PipelineDepth: 4,
	}
	for _, sys := range []string{"single-bft", "bullshark", "ladon"} {
		r := benchmark.RunConsensusBenchmark(baseCfg, sys)
		fmt.Printf("  %s: %.1f ktx/s, p50=%.1f ms, p99=%.1f ms\n",
			sys, r.ThroughputKtxs, r.LatencyP50Ms, r.LatencyP99Ms)
	}
}
