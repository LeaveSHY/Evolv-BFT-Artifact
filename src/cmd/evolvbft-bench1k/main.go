package main

import (
	"encoding/json"
	"fmt"
	"os"
	"time"

	"evolvbft/evolvbft/benchmark"
)

func main() {
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("  Evolv-BFT 1000-Node Scale Benchmark")
	fmt.Println("  Target: >=100k tx/s, <=100ms latency (WAN)")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println()

	numRounds := 500

	// Run 1000-node comparison
	fmt.Println("[1/3] Running 1000-node system comparison...")
	start := time.Now()
	results := benchmark.CompareAtScale(numRounds)
	fmt.Printf("      Completed in %s\n\n", time.Since(start))

	// Run scaling sweep (100 → 1000 nodes)
	fmt.Println("[2/3] Running scaling sweep (100→1000 nodes)...")
	nodeCounts := []int{100, 200, 400, 600, 800, 1000}
	var scalingResults []benchmark.ScaleResult
	for _, n := range nodeCounts {
		cfg := benchmark.Default1000NodeConfig()
		cfg.TotalNodes = n
		cfg.NumInstances = n / 100
		if cfg.NumInstances < 1 {
			cfg.NumInstances = 1
		}
		result := benchmark.RunScaleBenchmark(cfg, numRounds)
		scalingResults = append(scalingResults, result)
		fmt.Printf("      n=%4d (m=%2d): %8.1f ktx/s, p50=%5.1f ms, p99=%5.1f ms\n",
			n, cfg.NumInstances, result.ThroughputKtxs, result.LatencyP50Ms, result.LatencyP99Ms)
	}
	fmt.Println()

	// Run VRF committee size sensitivity
	fmt.Println("[3/3] Running VRF committee size sensitivity...")
	committeeSizes := []int{0, 10, 15, 20, 25, 33, 50}
	var vrfResults []benchmark.ScaleResult
	for _, k := range committeeSizes {
		cfg := benchmark.Default1000NodeConfig()
		cfg.CommitteeSize = k
		result := benchmark.RunScaleBenchmark(cfg, numRounds)
		vrfResults = append(vrfResults, result)
		label := fmt.Sprintf("k=%d", k)
		if k == 0 {
			label = "k=all(100)"
		}
		fmt.Printf("      %s: %8.1f ktx/s, p50=%5.1f ms, p99=%5.1f ms, msg=%s\n",
			label, result.ThroughputKtxs, result.LatencyP50Ms, result.LatencyP99Ms, result.MessageComplexity)
	}
	fmt.Println()

	// Summary
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("  SUMMARY")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	if len(results) > 0 {
		evolvbft := results[0]
		fmt.Printf("  Evolv-BFT@1000: %.0f ktx/s | p50=%.1fms p99=%.1fms | reconfig_overhead=%.1f%%\n",
			evolvbft.ThroughputKtxs, evolvbft.LatencyP50Ms, evolvbft.LatencyP99Ms, evolvbft.ReconfigOverheadPct)
		fmt.Printf("  Messages:     %s\n", evolvbft.MessageComplexity)
		fmt.Printf("  BW/node:      %.2f Mbps\n", evolvbft.BandwidthMbpsPerNode)

		if evolvbft.ThroughputKtxs >= 100 {
			fmt.Println("  ✓ Throughput target MET (>=100k tx/s)")
		} else {
			fmt.Printf("  ✗ Throughput target MISSED (%.0f < 100k tx/s)\n", evolvbft.ThroughputKtxs)
		}
		if evolvbft.LatencyP99Ms <= 100 {
			fmt.Println("  ✓ Latency target MET (p99 <=100ms)")
		} else {
			fmt.Printf("  ✗ Latency target MISSED (p99=%.1fms > 100ms)\n", evolvbft.LatencyP99Ms)
		}
	}
	fmt.Println()

	// Save results
	allResults := map[string]interface{}{
		"comparison":      results,
		"scaling":         scalingResults,
		"vrf_sensitivity": vrfResults,
		"config":          benchmark.Default1000NodeConfig(),
		"rounds":          numRounds,
		"timestamp":       time.Now().Format(time.RFC3339),
	}
	data, _ := json.MarshalIndent(allResults, "", "  ")
	outPath := "benchmark_1k_results.json"
	if err := os.WriteFile(outPath, data, 0644); err != nil {
		fmt.Printf("Warning: failed to save results: %v\n", err)
	} else {
		fmt.Printf("  Results saved to %s\n", outPath)
	}
}
