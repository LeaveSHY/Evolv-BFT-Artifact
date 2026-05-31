// Copyright 2024 Evolv-BFT Project
// Licensed under Apache License 2.0

package hotstuff

import (
	"math"
	"sync"
	"time"
)

// LeaderReputation tracks per-leader performance metrics for straggler detection.
// At 1000 nodes, a single slow/crashed leader can stall an entire instance.
// This module:
//  1. Records success/timeout/nil-block events per leader
//  2. Provides adaptive timeout: good leaders get tight timeouts, bad leaders get exponential backoff
//  3. Detects stragglers via a failure-rate threshold
//  4. Supports fallback leader selection: skip known-bad leaders in the rotation
type LeaderReputation struct {
	mu sync.RWMutex

	// Per-leader stats indexed by node ID
	records map[uint64]*leaderRecord

	// Configuration
	config ReputationConfig
}

// leaderRecord holds performance data for a single leader
type leaderRecord struct {
	// Lifetime counters (for straggler detection and fallback leader selection)
	successes uint64
	timeouts  uint64
	nilBlocks uint64

	// Sliding window event ring buffers for trust features (Eq. 5: d/W, e/W, v/W).
	// Each ring stores per-epoch event counts bounded to window W.
	eventWindow    []eventSample // ring buffer of size W
	eventWindowIdx int
	eventWindowLen int // actual fill count, up to W

	// Equivocations: detected equivocation events (double-voting, conflicting proposals)
	equivocations uint64
	// ViewChangeInitiations: number of view-changes initiated by this node
	viewChangeInits uint64
	// Consecutive timeouts (resets on success)
	consecutiveTimeouts uint64
	// Last successful proposal time
	lastSuccess time.Time
	// Last timeout time
	lastTimeout time.Time
	// Latency samples (bounded ring buffer for the sliding window)
	latencySamples []float64
	latencyIdx     int
	latencyFull    bool
}

// eventSample records event counts for one observation epoch within the sliding window.
type eventSample struct {
	timeouts        uint64
	equivocations   uint64
	viewChangeInits uint64
}

// ReputationConfig controls straggler detection behavior
type ReputationConfig struct {
	// BaseTimeoutMs is the default view timeout in milliseconds
	BaseTimeoutMs int64
	// MaxTimeoutMs is the maximum timeout after exponential backoff
	MaxTimeoutMs int64
	// TimeoutMultiplier is the exponential backoff factor per consecutive timeout
	TimeoutMultiplier float64
	// StragglerThreshold: if failure_rate > this, leader is marked as straggler (0.0-1.0)
	StragglerThreshold float64
	// MinSamples: minimum events before reputation is considered reliable
	MinSamples uint64
	// MaxConsecutiveTimeouts: after this many, leader is considered crashed
	MaxConsecutiveTimeouts uint64
	// DecayInterval: successes older than this are worth less (0 = no decay)
	DecayInterval time.Duration
	// LatencyWindowSize: ring buffer size for latency samples (Eq. 5: τ̄ and σ_τ)
	LatencyWindowSize int
}

// DefaultReputationConfig returns sensible defaults for 1000-node operation
func DefaultReputationConfig() ReputationConfig {
	return ReputationConfig{
		// Tighter timeouts for 1000-node scale: healthy leaders should respond within 1s
		BaseTimeoutMs:          1000,
		MaxTimeoutMs:           15000,
		TimeoutMultiplier:      1.5,
		StragglerThreshold:     0.5,
		MinSamples:             3,
		MaxConsecutiveTimeouts: 5,
		DecayInterval:          0,  // No decay for now
		LatencyWindowSize:      64, // Sliding window W for trust features
	}
}

// NewLeaderReputation creates a new reputation tracker
func NewLeaderReputation(config ReputationConfig) *LeaderReputation {
	if config.BaseTimeoutMs <= 0 {
		config.BaseTimeoutMs = 2000
	}
	if config.MaxTimeoutMs <= 0 {
		config.MaxTimeoutMs = 30000
	}
	if config.TimeoutMultiplier <= 1.0 {
		config.TimeoutMultiplier = 1.5
	}
	if config.StragglerThreshold <= 0 || config.StragglerThreshold > 1.0 {
		config.StragglerThreshold = 0.5
	}
	if config.MinSamples == 0 {
		config.MinSamples = 3
	}
	if config.MaxConsecutiveTimeouts == 0 {
		config.MaxConsecutiveTimeouts = 5
	}
	return &LeaderReputation{
		records: make(map[uint64]*leaderRecord),
		config:  config,
	}
}

// RecordSuccess records a successful block proposal+commit by the leader
func (lr *LeaderReputation) RecordSuccess(leaderID uint64) {
	lr.mu.Lock()
	defer lr.mu.Unlock()
	r := lr.getOrCreate(leaderID)
	r.successes++
	r.consecutiveTimeouts = 0
	r.lastSuccess = time.Now()
}

// RecordTimeout records a view timeout while leaderID was the expected leader
func (lr *LeaderReputation) RecordTimeout(leaderID uint64) {
	lr.mu.Lock()
	defer lr.mu.Unlock()
	r := lr.getOrCreate(leaderID)
	r.timeouts++
	r.consecutiveTimeouts++
	r.lastTimeout = time.Now()
}

// RecordNilBlock records a nil/empty block proposed by the leader (weak straggler signal)
func (lr *LeaderReputation) RecordNilBlock(leaderID uint64) {
	lr.mu.Lock()
	defer lr.mu.Unlock()
	r := lr.getOrCreate(leaderID)
	r.nilBlocks++
}

// AdaptiveTimeout returns the timeout to use for a view where leaderID is leader.
// Good leaders: BaseTimeoutMs
// Bad leaders: BaseTimeoutMs * multiplier^consecutiveTimeouts (capped at MaxTimeoutMs)
func (lr *LeaderReputation) AdaptiveTimeout(leaderID uint64) time.Duration {
	lr.mu.RLock()
	defer lr.mu.RUnlock()

	r, exists := lr.records[leaderID]
	if !exists {
		return time.Duration(lr.config.BaseTimeoutMs) * time.Millisecond
	}

	if r.consecutiveTimeouts == 0 {
		return time.Duration(lr.config.BaseTimeoutMs) * time.Millisecond
	}

	// Exponential backoff: base * multiplier^consecutive
	timeoutMs := float64(lr.config.BaseTimeoutMs)
	for i := uint64(0); i < r.consecutiveTimeouts; i++ {
		timeoutMs *= lr.config.TimeoutMultiplier
		if timeoutMs >= float64(lr.config.MaxTimeoutMs) {
			timeoutMs = float64(lr.config.MaxTimeoutMs)
			break
		}
	}
	return time.Duration(int64(timeoutMs)) * time.Millisecond
}

// IsStraggler returns true if the leader has a failure rate above the threshold
func (lr *LeaderReputation) IsStraggler(leaderID uint64) bool {
	lr.mu.RLock()
	defer lr.mu.RUnlock()
	return lr.isStraggler(leaderID)
}

func (lr *LeaderReputation) isStraggler(leaderID uint64) bool {
	r, exists := lr.records[leaderID]
	if !exists {
		return false
	}

	total := r.successes + r.timeouts + r.nilBlocks
	if total < lr.config.MinSamples {
		return false
	}

	// Failure rate = (timeouts + nilBlocks) / total
	failures := r.timeouts + r.nilBlocks
	failRate := float64(failures) / float64(total)
	return failRate > lr.config.StragglerThreshold
}

// IsCrashed returns true if the leader has exceeded MaxConsecutiveTimeouts
func (lr *LeaderReputation) IsCrashed(leaderID uint64) bool {
	lr.mu.RLock()
	defer lr.mu.RUnlock()
	r, exists := lr.records[leaderID]
	if !exists {
		return false
	}
	return r.consecutiveTimeouts >= lr.config.MaxConsecutiveTimeouts
}

// SelectFallbackLeader picks the next non-straggler leader from the sorted validator list.
// If all leaders are stragglers, returns the one with the lowest failure rate.
// validatorIDs must be sorted.
func (lr *LeaderReputation) SelectFallbackLeader(primaryLeader uint64, validatorIDs []uint64, view uint64) uint64 {
	lr.mu.RLock()
	defer lr.mu.RUnlock()

	if len(validatorIDs) == 0 {
		return primaryLeader
	}

	// Find primary's index
	primaryIdx := -1
	for i, id := range validatorIDs {
		if id == primaryLeader {
			primaryIdx = i
			break
		}
	}
	if primaryIdx < 0 {
		return validatorIDs[0]
	}

	// Try subsequent validators as fallback (round-robin from primary+1)
	n := len(validatorIDs)
	bestCandidate := primaryLeader
	bestFailRate := float64(2.0) // > 1.0 sentinel

	for offset := 1; offset < n; offset++ {
		candidateID := validatorIDs[(primaryIdx+offset)%n]
		if !lr.isStraggler(candidateID) {
			return candidateID
		}
		// Track best-of-bad
		r, exists := lr.records[candidateID]
		if !exists {
			return candidateID // No record = no failures = good candidate
		}
		total := r.successes + r.timeouts + r.nilBlocks
		if total == 0 {
			return candidateID
		}
		failRate := float64(r.timeouts+r.nilBlocks) / float64(total)
		if failRate < bestFailRate {
			bestFailRate = failRate
			bestCandidate = candidateID
		}
	}

	return bestCandidate
}

// GetStats returns a snapshot of a leader's reputation data
func (lr *LeaderReputation) GetStats(leaderID uint64) LeaderReputationStats {
	lr.mu.RLock()
	defer lr.mu.RUnlock()

	r, exists := lr.records[leaderID]
	if !exists {
		return LeaderReputationStats{LeaderID: leaderID}
	}

	total := r.successes + r.timeouts + r.nilBlocks
	failRate := 0.0
	if total > 0 {
		failRate = float64(r.timeouts+r.nilBlocks) / float64(total)
	}

	return LeaderReputationStats{
		LeaderID:            leaderID,
		Successes:           r.successes,
		Timeouts:            r.timeouts,
		NilBlocks:           r.nilBlocks,
		ConsecutiveTimeouts: r.consecutiveTimeouts,
		FailureRate:         failRate,
		IsStraggler:         lr.isStraggler(leaderID),
		IsCrashed:           r.consecutiveTimeouts >= lr.config.MaxConsecutiveTimeouts,
	}
}

// GetAllStats returns reputation stats for all tracked leaders
func (lr *LeaderReputation) GetAllStats() []LeaderReputationStats {
	lr.mu.RLock()
	defer lr.mu.RUnlock()

	stats := make([]LeaderReputationStats, 0, len(lr.records))
	for id := range lr.records {
		r := lr.records[id]
		t := r.successes + r.timeouts + r.nilBlocks
		var failRate float64
		if t > 0 {
			failRate = float64(r.timeouts+r.nilBlocks) / float64(t)
		}

		// Compute latency stats (Eq. 5: τ̄ and σ_τ)
		var avgLat, stdLat float64
		n := lr.latencyCount(r)
		if n > 0 {
			samples := lr.latencySlice(r)
			avgLat, stdLat = latencyStats(samples)
		}

		stats = append(stats, LeaderReputationStats{
			LeaderID:            id,
			Successes:           r.successes,
			Timeouts:            r.timeouts,
			NilBlocks:           r.nilBlocks,
			Equivocations:       r.equivocations,
			ViewChangeInits:     r.viewChangeInits,
			ConsecutiveTimeouts: r.consecutiveTimeouts,
			FailureRate:         failRate,
			AvgLatencyMs:        avgLat,
			StdLatencyMs:        stdLat,
			IsStraggler:         lr.isStraggler(id),
			IsCrashed:           r.consecutiveTimeouts >= lr.config.MaxConsecutiveTimeouts,
		})
	}
	return stats
}

func (lr *LeaderReputation) SetBaseTimeoutMs(timeoutMs int64) {
	lr.mu.Lock()
	defer lr.mu.Unlock()
	if timeoutMs > 0 {
		lr.config.BaseTimeoutMs = timeoutMs
	}
}

// LeaderReputationStats is a snapshot of a leader's reputation
type LeaderReputationStats struct {
	LeaderID            uint64  `json:"leader_id"`
	Successes           uint64  `json:"successes"`
	Timeouts            uint64  `json:"timeouts"`
	NilBlocks           uint64  `json:"nil_blocks"`
	Equivocations       uint64  `json:"equivocations"`     // Eq. 5: e (equivocation events)
	ViewChangeInits     uint64  `json:"view_change_inits"` // Eq. 5: v (view-change initiations)
	ConsecutiveTimeouts uint64  `json:"consecutive_timeouts"`
	FailureRate         float64 `json:"failure_rate"`
	AvgLatencyMs        float64 `json:"avg_latency_ms"` // Eq. 5: τ̄ (mean latency)
	StdLatencyMs        float64 `json:"std_latency_ms"` // Eq. 5: σ_τ (latency std dev)
	IsStraggler         bool    `json:"is_straggler"`
	IsCrashed           bool    `json:"is_crashed"`
}

func (lr *LeaderReputation) getOrCreate(leaderID uint64) *leaderRecord {
	r, exists := lr.records[leaderID]
	if !exists {
		w := lr.config.LatencyWindowSize
		if w <= 0 {
			w = 64
		}
		r = &leaderRecord{
			latencySamples: make([]float64, w),
			eventWindow:    make([]eventSample, w),
		}
		lr.records[leaderID] = r
	}
	return r
}

// RecordEquivocation records a detected equivocation event for this node (e_t^k in Eq. 5).
func (lr *LeaderReputation) RecordEquivocation(nodeID uint64) {
	lr.mu.Lock()
	defer lr.mu.Unlock()
	r := lr.getOrCreate(nodeID)
	r.equivocations++
}

// RecordViewChangeInit records that this node initiated a view-change (v_t^k in Eq. 5).
func (lr *LeaderReputation) RecordViewChangeInit(nodeID uint64) {
	lr.mu.Lock()
	defer lr.mu.Unlock()
	r := lr.getOrCreate(nodeID)
	r.viewChangeInits++
}

// RecordEventEpoch snapshots the current epoch's event counts into the sliding window
// and resets per-epoch accumulators. Call once per consensus epoch per node.
// This ensures TrustFeatureVector computes d/W, e/W, v/W over the last W epochs
// as specified by the paper (Eq. 5).
func (lr *LeaderReputation) RecordEventEpoch(nodeID uint64) {
	lr.mu.Lock()
	defer lr.mu.Unlock()
	r := lr.getOrCreate(nodeID)
	w := len(r.eventWindow)
	if w == 0 {
		return
	}
	r.eventWindow[r.eventWindowIdx] = eventSample{
		timeouts:        r.timeouts,
		equivocations:   r.equivocations,
		viewChangeInits: r.viewChangeInits,
	}
	r.eventWindowIdx = (r.eventWindowIdx + 1) % w
	if r.eventWindowLen < w {
		r.eventWindowLen++
	}
}

// windowEventSums returns the sum of event counts over the sliding window.
func (lr *LeaderReputation) windowEventSums(r *leaderRecord) (d, e, v uint64) {
	n := r.eventWindowLen
	for i := 0; i < n; i++ {
		s := r.eventWindow[i]
		d += s.timeouts
		e += s.equivocations
		v += s.viewChangeInits
	}
	return
}

// RecordLatency records a commit latency sample in milliseconds (for τ̄ and σ_τ in Eq. 5).
func (lr *LeaderReputation) RecordLatency(leaderID uint64, latencyMs float64) {
	lr.mu.Lock()
	defer lr.mu.Unlock()
	r := lr.getOrCreate(leaderID)

	windowSize := lr.config.LatencyWindowSize
	if windowSize <= 0 {
		return
	}
	if len(r.latencySamples) == 0 {
		r.latencySamples = make([]float64, windowSize)
	}

	r.latencySamples[r.latencyIdx] = latencyMs
	r.latencyIdx = (r.latencyIdx + 1) % windowSize
	if r.latencyIdx == 0 {
		r.latencyFull = true
	}
}

// TrustFeatureVector computes the paper's Eq. 5 trust feature vector for a node:
//
//	x_t^k = (d_t^k/W, e_t^k/W, v_t^k/W, τ̄_t^k, σ_τ,t^k)
//
// where W is the window size, d=timeouts, e=equivocations, v=view-change initiations,
// τ̄=mean latency, σ_τ=latency standard deviation.
// All values are bounded in [0,1]^5 as required by the paper.
type TrustFeatures struct {
	TimeoutRate      float64 `json:"timeout_rate"`      // d_t^k / W
	EquivocationRate float64 `json:"equivocation_rate"` // e_t^k / W
	ViewChangeRate   float64 `json:"view_change_rate"`  // v_t^k / W
	MeanLatency      float64 `json:"mean_latency"`      // τ̄_t^k (normalized)
	StdLatency       float64 `json:"std_latency"`       // σ_τ,t^k (normalized)
}

// TrustFeatureVector computes the 5-dimensional trust feature vector (Eq. 5) for nodeID.
// Returns the features and true if enough samples exist, or zero features and false otherwise.
func (lr *LeaderReputation) TrustFeatureVector(nodeID uint64) (TrustFeatures, bool) {
	lr.mu.RLock()
	defer lr.mu.RUnlock()

	r, exists := lr.records[nodeID]
	if !exists {
		return TrustFeatures{}, false
	}

	total := r.successes + r.timeouts + r.nilBlocks
	if total < lr.config.MinSamples {
		return TrustFeatures{}, false
	}

	// Use sliding window W for event-rate features (Eq. 5: d/W, e/W, v/W)
	w := float64(lr.config.LatencyWindowSize)
	if w <= 0 {
		w = 64
	}
	d, e, v := lr.windowEventSums(r)
	// If the event window hasn't been populated yet, fall back to lifetime totals
	// capped by W to avoid unbounded growth.
	if r.eventWindowLen == 0 {
		d = r.timeouts
		e = r.equivocations
		v = r.viewChangeInits
	}

	features := TrustFeatures{
		TimeoutRate:      clampFloat(float64(d) / w),
		EquivocationRate: clampFloat(float64(e) / w),
		ViewChangeRate:   clampFloat(float64(v) / w),
	}

	// Compute latency statistics from the ring buffer
	n := lr.latencyCount(r)
	if n > 0 {
		samples := lr.latencySlice(r)
		mean, stddev := latencyStats(samples)
		// Normalize to [0,1] using max timeout as reference
		maxMs := float64(lr.config.MaxTimeoutMs)
		if maxMs <= 0 {
			maxMs = 30000
		}
		features.MeanLatency = clampFloat(mean / maxMs)
		features.StdLatency = clampFloat(stddev / maxMs)
	}

	return features, true
}

// GetAllTrustFeatures returns trust feature vectors for all tracked nodes.
func (lr *LeaderReputation) GetAllTrustFeatures() map[uint64]TrustFeatures {
	lr.mu.RLock()
	defer lr.mu.RUnlock()

	wf := float64(lr.config.LatencyWindowSize)
	if wf <= 0 {
		wf = 64
	}

	result := make(map[uint64]TrustFeatures, len(lr.records))
	for id, r := range lr.records {
		total := r.successes + r.timeouts + r.nilBlocks
		if total < lr.config.MinSamples {
			continue
		}
		d, e, v := lr.windowEventSums(r)
		if r.eventWindowLen == 0 {
			d = r.timeouts
			e = r.equivocations
			v = r.viewChangeInits
		}
		f := TrustFeatures{
			TimeoutRate:      clampFloat(float64(d) / wf),
			EquivocationRate: clampFloat(float64(e) / wf),
			ViewChangeRate:   clampFloat(float64(v) / wf),
		}
		n := lr.latencyCount(r)
		if n > 0 {
			samples := lr.latencySlice(r)
			mean, stddev := latencyStats(samples)
			maxMs := float64(lr.config.MaxTimeoutMs)
			if maxMs <= 0 {
				maxMs = 30000
			}
			f.MeanLatency = clampFloat(mean / maxMs)
			f.StdLatency = clampFloat(stddev / maxMs)
		}
		result[id] = f
	}
	return result
}

func (lr *LeaderReputation) latencyCount(r *leaderRecord) int {
	windowSize := lr.config.LatencyWindowSize
	if windowSize <= 0 || len(r.latencySamples) == 0 {
		return 0
	}
	if r.latencyFull {
		return windowSize
	}
	return r.latencyIdx
}

func (lr *LeaderReputation) latencySlice(r *leaderRecord) []float64 {
	n := lr.latencyCount(r)
	if n == 0 {
		return nil
	}
	if r.latencyFull {
		return r.latencySamples
	}
	return r.latencySamples[:r.latencyIdx]
}

func latencyStats(samples []float64) (mean, stddev float64) {
	n := len(samples)
	if n == 0 {
		return 0, 0
	}
	sum := 0.0
	for _, v := range samples {
		sum += v
	}
	mean = sum / float64(n)
	if n < 2 {
		return mean, 0
	}
	variance := 0.0
	for _, v := range samples {
		d := v - mean
		variance += d * d
	}
	variance /= float64(n - 1)
	// math.Sqrt imported at top
	stddev = 0
	if variance > 0 {
		stddev = math.Sqrt(variance)
	}
	return mean, stddev
}

func clampFloat(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}
