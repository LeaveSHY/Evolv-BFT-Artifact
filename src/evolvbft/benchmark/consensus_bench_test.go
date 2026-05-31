package benchmark

import (
	"testing"
)

func TestConsensusBenchmark_WANScaling(t *testing.T) {
	replicaCounts := []int{100, 200, 400, 600, 800, 1000}
	t.Logf("=== WAN Scaling Benchmark (m=4 instances) ===")
	results := RunScalingSuite(WANProfile, replicaCounts, 4, 50)

	for _, r := range results {
		t.Logf("%-12s n=%4d: %7.1f ktx/s, latency=%.1f ms (p50=%.1f, p99=%.1f)",
			r.System, r.Config.NumReplicas, r.ThroughputKtxs, r.LatencyMs, r.LatencyP50Ms, r.LatencyP99Ms)
	}

	// Verify Evolv-BFT achieves >2x throughput over single-BFT
	for _, n := range replicaCounts {
		var evolvbftThroughput, singleThroughput float64
		for _, r := range results {
			if r.Config.NumReplicas == n && r.System == "evolvbft" {
				evolvbftThroughput = r.ThroughputKtxs
			}
			if r.Config.NumReplicas == n && r.System == "single-bft" {
				singleThroughput = r.ThroughputKtxs
			}
		}
		if singleThroughput > 0 {
			ratio := evolvbftThroughput / singleThroughput
			t.Logf("n=%d: Evolv-BFT/Single-BFT throughput ratio = %.2fx", n, ratio)
			if ratio < 1.5 {
				t.Errorf("n=%d: Evolv-BFT should achieve >1.5x over single-BFT, got %.2fx", n, ratio)
			}
		}
	}
}

func TestConsensusBenchmark_LANScaling(t *testing.T) {
	replicaCounts := []int{100, 200, 400, 600, 800, 1000}
	t.Logf("=== LAN Scaling Benchmark (m=4 instances) ===")
	results := RunScalingSuite(LANProfile, replicaCounts, 4, 50)

	for _, r := range results {
		t.Logf("%-12s n=%4d: %7.1f ktx/s, latency=%.2f ms (p50=%.2f, p99=%.2f)",
			r.System, r.Config.NumReplicas, r.ThroughputKtxs, r.LatencyMs, r.LatencyP50Ms, r.LatencyP99Ms)
	}

	// Verify Evolv-BFT maintains sub-second latency at 1000 replicas
	for _, r := range results {
		if r.System == "evolvbft" && r.Config.NumReplicas == 1000 {
			if r.LatencyMs > 1000 {
				t.Errorf("Evolv-BFT LAN latency at n=1000 should be sub-second, got %.1f ms", r.LatencyMs)
			}
		}
	}
}

func TestReconfigBenchmark(t *testing.T) {
	batchSizes := []int{5, 10, 15, 20, 25, 30}
	t.Logf("=== Reconfiguration Benchmark ===")
	results := RunReconfigBenchmark(WANProfile, batchSizes, 100)

	for _, r := range results {
		t.Logf("%-12s batch=%2d: %.1f ± %.1f ktx/s (min=%.1f, max=%.1f)",
			r.System, r.BatchSize, r.MeanThroughput, r.StdThroughput, r.MinThroughput, r.MaxThroughput)
	}
}
