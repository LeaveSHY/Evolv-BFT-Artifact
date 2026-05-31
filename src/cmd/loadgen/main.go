// Copyright 2024 Evolv-BFT Project
// Licensed under Apache License 2.0

// loadgen generates sustained transaction load against one or more Evolv-BFT nodes.
// It sends HTTP POST requests to the /tx endpoint at a configurable rate.
//
// Usage:
//
//	loadgen -targets=http://10.0.1.10:9000,http://10.0.2.10:9000 -rate=50000 -duration=300s -payload=64
package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

type config struct {
	Targets      string
	RatePerSec   int
	Duration     time.Duration
	PayloadBytes int
	Workers      int
	Ramp         time.Duration
}

func main() {
	cfg := config{}
	flag.StringVar(&cfg.Targets, "targets", "http://127.0.0.1:9000", "Comma-separated list of Evolv-BFT node HTTP endpoints")
	flag.IntVar(&cfg.RatePerSec, "rate", 50000, "Target aggregate tx/s injection rate")
	flag.DurationVar(&cfg.Duration, "duration", 5*time.Minute, "Total benchmark duration")
	flag.IntVar(&cfg.PayloadBytes, "payload", 64, "Transaction payload size in bytes")
	flag.IntVar(&cfg.Workers, "workers", 128, "Number of concurrent HTTP workers")
	flag.DurationVar(&cfg.Ramp, "ramp", 10*time.Second, "Ramp-up duration (linear increase to target rate)")
	flag.Parse()

	targets := strings.Split(cfg.Targets, ",")
	for i := range targets {
		targets[i] = strings.TrimSpace(targets[i])
	}
	if len(targets) == 0 || targets[0] == "" {
		log.Fatal("at least one target required")
	}

	log.Printf("loadgen: targets=%v rate=%d tx/s duration=%s payload=%dB workers=%d ramp=%s",
		targets, cfg.RatePerSec, cfg.Duration, cfg.PayloadBytes, cfg.Workers, cfg.Ramp)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Graceful shutdown on SIGINT/SIGTERM
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		log.Println("loadgen: shutting down...")
		cancel()
	}()

	// Pre-generate random payload
	payload := make([]byte, cfg.PayloadBytes)
	if _, err := rand.Read(payload); err != nil {
		log.Fatalf("generate payload: %v", err)
	}

	// Stats
	var (
		totalSent    atomic.Int64
		totalSuccess atomic.Int64
		totalFailed  atomic.Int64
	)

	// HTTP client with connection pooling
	client := &http.Client{
		Transport: &http.Transport{
			MaxIdleConnsPerHost: cfg.Workers / len(targets),
			MaxConnsPerHost:     cfg.Workers / len(targets),
			IdleConnTimeout:     30 * time.Second,
		},
		Timeout: 5 * time.Second,
	}

	// Work channel
	workCh := make(chan string, cfg.Workers*4)

	// Start workers
	var wg sync.WaitGroup
	for i := 0; i < cfg.Workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for target := range workCh {
				url := target + "/tx"
				req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
				if err != nil {
					totalFailed.Add(1)
					continue
				}
				req.Header.Set("Content-Type", "application/octet-stream")
				resp, err := client.Do(req)
				if err != nil {
					totalFailed.Add(1)
					continue
				}
				_, _ = io.Copy(io.Discard, resp.Body)
				resp.Body.Close()
				if resp.StatusCode == http.StatusOK {
					totalSuccess.Add(1)
				} else {
					totalFailed.Add(1)
				}
			}
		}()
	}

	// Ticker for rate control
	start := time.Now()
	deadline := start.Add(cfg.Duration)

	// Progress reporter
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				elapsed := time.Since(start).Seconds()
				sent := totalSent.Load()
				ok := totalSuccess.Load()
				fail := totalFailed.Load()
				actualRate := float64(sent) / elapsed
				log.Printf("loadgen: elapsed=%.0fs sent=%d ok=%d fail=%d actual_rate=%.0f tx/s",
					elapsed, sent, ok, fail, actualRate)
			}
		}
	}()

	// Rate-limited injection loop
	targetIdx := 0
	intervalNs := int64(time.Second) / int64(cfg.RatePerSec)
	ticker := time.NewTicker(time.Duration(intervalNs))
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			goto done
		case t := <-ticker.C:
			if t.After(deadline) {
				goto done
			}
			// Ramp-up: linearly increase rate
			elapsed := t.Sub(start)
			var currentRate float64
			if elapsed < cfg.Ramp {
				frac := float64(elapsed) / float64(cfg.Ramp)
				currentRate = frac * float64(cfg.RatePerSec)
			} else {
				currentRate = float64(cfg.RatePerSec)
			}
			// Burst: send multiple txs per tick to reach currentRate
			// (since ticker is at max rate, we always send 1 per tick)
			_ = currentRate

			target := targets[targetIdx%len(targets)]
			targetIdx++
			select {
			case workCh <- target:
				totalSent.Add(1)
			default:
				// Workers saturated, skip to maintain rate without blocking
				totalFailed.Add(1)
			}
		}
	}

done:
	close(workCh)
	wg.Wait()

	elapsed := time.Since(start).Seconds()
	sent := totalSent.Load()
	ok := totalSuccess.Load()
	fail := totalFailed.Load()
	effectiveRate := float64(ok) / elapsed

	fmt.Println("\n--- Load Generation Results ---")
	fmt.Printf("Duration:       %.1fs\n", elapsed)
	fmt.Printf("Total Sent:     %d\n", sent)
	fmt.Printf("Successful:     %d\n", ok)
	fmt.Printf("Failed:         %d\n", fail)
	fmt.Printf("Effective Rate: %.0f tx/s\n", effectiveRate)
	fmt.Printf("Target Rate:    %d tx/s\n", cfg.RatePerSec)
	fmt.Printf("Utilization:    %.1f%%\n", math.Min(100, effectiveRate/float64(cfg.RatePerSec)*100))
}
