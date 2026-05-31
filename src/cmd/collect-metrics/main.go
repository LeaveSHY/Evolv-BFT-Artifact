// Copyright 2024 Evolv-BFT Project
// Licensed under Apache License 2.0

// collect-metrics polls /metrics from multiple Evolv-BFT nodes and produces
// aggregated benchmark results suitable for paper figures.
//
// Usage:
//
//	collect-metrics -targets=http://10.0.1.10:9000,http://10.0.2.10:9000 -duration=300s -interval=5s -out=results.json
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"os/signal"
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"
)

// MetricsSnapshot mirrors the JSON from Evolv-BFT /metrics endpoint
type MetricsSnapshot struct {
	GlobalConfirmedTotal int64   `json:"global_confirmed_total"`
	GlobalConfirmedNil   int64   `json:"global_confirmed_nil"`
	ThroughputTPS        float64 `json:"throughput_tps"`
	LatencyP50Ms         float64 `json:"latency_p50_ms"`
	LatencyP95Ms         float64 `json:"latency_p95_ms"`
	LatencyP99Ms         float64 `json:"latency_p99_ms"`
	RecoveryP50Ms        float64 `json:"recovery_p50_ms"`
	RecoveryP95Ms        float64 `json:"recovery_p95_ms"`
	RecoveryP99Ms        float64 `json:"recovery_p99_ms"`
	BacklogPending       int64   `json:"backlog_pending"`
	BacklogMissing       int64   `json:"backlog_missing"`
	RejectTotal          int64   `json:"reject_total"`
}

type TimestampedSnapshot struct {
	Timestamp time.Time       `json:"timestamp"`
	NodeURL   string          `json:"node_url"`
	Metrics   MetricsSnapshot `json:"metrics"`
}

type AggregatedResult struct {
	StartTime        time.Time `json:"start_time"`
	EndTime          time.Time `json:"end_time"`
	DurationSec      float64   `json:"duration_sec"`
	NumNodes         int       `json:"num_nodes"`
	NumSamples       int       `json:"num_samples"`
	AvgThroughputTPS float64   `json:"avg_throughput_tps"`
	MaxThroughputTPS float64   `json:"max_throughput_tps"`
	MinThroughputTPS float64   `json:"min_throughput_tps"`
	P50LatencyMs     float64   `json:"p50_latency_ms"`
	P95LatencyMs     float64   `json:"p95_latency_ms"`
	P99LatencyMs     float64   `json:"p99_latency_ms"`
	AvgBacklog       float64   `json:"avg_backlog"`
	TotalConfirmed   int64     `json:"total_confirmed"`
	TotalRejected    int64     `json:"total_rejected"`
	// Per-node breakdown
	PerNode map[string]*NodeSummary `json:"per_node"`
}

type NodeSummary struct {
	Samples       int     `json:"samples"`
	AvgTPS        float64 `json:"avg_tps"`
	MaxTPS        float64 `json:"max_tps"`
	AvgLatP50     float64 `json:"avg_latency_p50_ms"`
	AvgLatP99     float64 `json:"avg_latency_p99_ms"`
	FinalConfirms int64   `json:"final_confirmed"`
}

type config struct {
	Targets  string
	Duration time.Duration
	Interval time.Duration
	Warmup   time.Duration
	OutFile  string
}

func main() {
	cfg := config{}
	flag.StringVar(&cfg.Targets, "targets", "http://127.0.0.1:9000", "Comma-separated node endpoints")
	flag.DurationVar(&cfg.Duration, "duration", 5*time.Minute, "Collection duration")
	flag.DurationVar(&cfg.Interval, "interval", 5*time.Second, "Polling interval")
	flag.DurationVar(&cfg.Warmup, "warmup", 30*time.Second, "Warmup period to discard")
	flag.StringVar(&cfg.OutFile, "out", "benchmark_results.json", "Output file path")
	flag.Parse()

	targets := strings.Split(cfg.Targets, ",")
	for i := range targets {
		targets[i] = strings.TrimSpace(targets[i])
	}

	log.Printf("collect-metrics: targets=%v duration=%s interval=%s warmup=%s out=%s",
		targets, cfg.Duration, cfg.Interval, cfg.Warmup, cfg.OutFile)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		cancel()
	}()

	client := &http.Client{Timeout: 3 * time.Second}

	var allSnapshots []TimestampedSnapshot
	var mu sync.Mutex

	start := time.Now()
	deadline := start.Add(cfg.Duration)
	ticker := time.NewTicker(cfg.Interval)
	defer ticker.Stop()

	log.Printf("collect-metrics: starting collection (warmup until %s)", start.Add(cfg.Warmup).Format("15:04:05"))

	for {
		select {
		case <-ctx.Done():
			goto aggregate
		case t := <-ticker.C:
			if t.After(deadline) {
				goto aggregate
			}
			// Poll all targets in parallel
			var wg sync.WaitGroup
			for _, target := range targets {
				wg.Add(1)
				go func(url string) {
					defer wg.Done()
					snap, err := fetchMetrics(ctx, client, url)
					if err != nil {
						log.Printf("collect-metrics: error from %s: %v", url, err)
						return
					}
					ts := TimestampedSnapshot{
						Timestamp: t,
						NodeURL:   url,
						Metrics:   *snap,
					}
					mu.Lock()
					allSnapshots = append(allSnapshots, ts)
					mu.Unlock()
				}(target)
			}
			wg.Wait()

			// Log progress
			elapsed := time.Since(start)
			log.Printf("collect-metrics: %.0fs/%s samples=%d",
				elapsed.Seconds(), cfg.Duration, len(allSnapshots))
		}
	}

aggregate:
	log.Printf("collect-metrics: collection done, total samples=%d", len(allSnapshots))

	// Filter out warmup period
	warmupEnd := start.Add(cfg.Warmup)
	var steadyState []TimestampedSnapshot
	for _, s := range allSnapshots {
		if s.Timestamp.After(warmupEnd) {
			steadyState = append(steadyState, s)
		}
	}
	log.Printf("collect-metrics: steady-state samples=%d (discarded %d warmup)",
		len(steadyState), len(allSnapshots)-len(steadyState))

	result := aggregate(steadyState, targets, start, time.Now())

	// Write output
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		log.Fatalf("marshal results: %v", err)
	}
	if err := os.WriteFile(cfg.OutFile, data, 0644); err != nil {
		log.Fatalf("write output: %v", err)
	}

	// Print summary
	fmt.Println("\n=== Benchmark Results ===")
	fmt.Printf("Duration:        %.0fs\n", result.DurationSec)
	fmt.Printf("Nodes:           %d\n", result.NumNodes)
	fmt.Printf("Samples:         %d\n", result.NumSamples)
	fmt.Printf("Throughput (avg): %.0f tx/s\n", result.AvgThroughputTPS)
	fmt.Printf("Throughput (max): %.0f tx/s\n", result.MaxThroughputTPS)
	fmt.Printf("Latency p50:     %.1f ms\n", result.P50LatencyMs)
	fmt.Printf("Latency p95:     %.1f ms\n", result.P95LatencyMs)
	fmt.Printf("Latency p99:     %.1f ms\n", result.P99LatencyMs)
	fmt.Printf("Total Confirmed: %d\n", result.TotalConfirmed)
	fmt.Printf("Total Rejected:  %d\n", result.TotalRejected)
	fmt.Printf("Output:          %s\n", cfg.OutFile)
}

func fetchMetrics(ctx context.Context, client *http.Client, baseURL string) (*MetricsSnapshot, error) {
	url := baseURL + "/metrics"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}
	var snap MetricsSnapshot
	if err := json.Unmarshal(body, &snap); err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	return &snap, nil
}

func aggregate(snapshots []TimestampedSnapshot, targets []string, start, end time.Time) *AggregatedResult {
	result := &AggregatedResult{
		StartTime:   start,
		EndTime:     end,
		DurationSec: end.Sub(start).Seconds(),
		NumNodes:    len(targets),
		NumSamples:  len(snapshots),
		PerNode:     make(map[string]*NodeSummary),
	}

	if len(snapshots) == 0 {
		return result
	}

	// Per-node aggregation
	for _, s := range snapshots {
		ns, ok := result.PerNode[s.NodeURL]
		if !ok {
			ns = &NodeSummary{}
			result.PerNode[s.NodeURL] = ns
		}
		ns.Samples++
		ns.AvgTPS += s.Metrics.ThroughputTPS
		ns.AvgLatP50 += s.Metrics.LatencyP50Ms
		ns.AvgLatP99 += s.Metrics.LatencyP99Ms
		if s.Metrics.ThroughputTPS > ns.MaxTPS {
			ns.MaxTPS = s.Metrics.ThroughputTPS
		}
		if s.Metrics.GlobalConfirmedTotal > ns.FinalConfirms {
			ns.FinalConfirms = s.Metrics.GlobalConfirmedTotal
		}
	}
	for _, ns := range result.PerNode {
		if ns.Samples > 0 {
			ns.AvgTPS /= float64(ns.Samples)
			ns.AvgLatP50 /= float64(ns.Samples)
			ns.AvgLatP99 /= float64(ns.Samples)
		}
	}

	// Global aggregation: use median of per-node TPS (more robust than mean)
	var allTPS []float64
	var allP50 []float64
	var allP95 []float64
	var allP99 []float64
	var totalBacklog float64
	var maxConfirmed int64
	var maxReject int64

	for _, s := range snapshots {
		allTPS = append(allTPS, s.Metrics.ThroughputTPS)
		allP50 = append(allP50, s.Metrics.LatencyP50Ms)
		allP95 = append(allP95, s.Metrics.LatencyP95Ms)
		allP99 = append(allP99, s.Metrics.LatencyP99Ms)
		totalBacklog += float64(s.Metrics.BacklogPending)
		if s.Metrics.GlobalConfirmedTotal > maxConfirmed {
			maxConfirmed = s.Metrics.GlobalConfirmedTotal
		}
		if s.Metrics.RejectTotal > maxReject {
			maxReject = s.Metrics.RejectTotal
		}
	}

	sort.Float64s(allTPS)
	sort.Float64s(allP50)
	sort.Float64s(allP95)
	sort.Float64s(allP99)

	result.AvgThroughputTPS = mean(allTPS)
	result.MaxThroughputTPS = allTPS[len(allTPS)-1]
	result.MinThroughputTPS = allTPS[0]
	result.P50LatencyMs = percentile(allP50, 0.50)
	result.P95LatencyMs = percentile(allP95, 0.95)
	result.P99LatencyMs = percentile(allP99, 0.99)
	result.AvgBacklog = totalBacklog / float64(len(snapshots))
	result.TotalConfirmed = maxConfirmed
	result.TotalRejected = maxReject

	return result
}

func mean(sorted []float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range sorted {
		sum += v
	}
	return sum / float64(len(sorted))
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(math.Ceil(p*float64(len(sorted)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}
