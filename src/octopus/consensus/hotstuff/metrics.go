package hotstuff

import (
	"sort"
	"sync"
	"time"
)

type GlobalConfirmedMetrics struct {
	mu sync.RWMutex

	startedAt time.Time
	lastAt    time.Time

	barTimeout time.Duration

	globalConfirmedTotal uint64
	globalConfirmedNil   uint64

	latenciesMs []float64
	recoveryMs  []float64
}

type GlobalConfirmedSnapshot struct {
	GlobalConfirmedTotal uint64             `json:"global_confirmed_total"`
	GlobalConfirmedNil   uint64             `json:"global_confirmed_nil"`
	ThroughputTPS        float64            `json:"throughput_tps"`
	LatencyP50Ms         float64            `json:"latency_p50_ms"`
	LatencyP95Ms         float64            `json:"latency_p95_ms"`
	LatencyP99Ms         float64            `json:"latency_p99_ms"`
	RecoveryP50Ms        float64            `json:"recovery_p50_ms"`
	RecoveryP95Ms        float64            `json:"recovery_p95_ms"`
	RecoveryP99Ms        float64            `json:"recovery_p99_ms"`
	BacklogPending       uint64             `json:"backlog_pending"`
	BacklogMissing       uint64             `json:"backlog_missing"`
	OrdererLateTotal     uint64             `json:"orderer_late_total"`
	OrdererNilledTotal   uint64             `json:"orderer_nilled_total"`
	RejectTotal          uint64             `json:"reject_total"`
	RejectByReason       map[string]uint64  `json:"reject_by_reason"`
	WindowStartUnixMs    int64              `json:"window_start_unix_ms"`
	LastUpdateUnixMs     int64              `json:"last_update_unix_ms"`
}

func NewGlobalConfirmedMetrics(barTimeout time.Duration) *GlobalConfirmedMetrics {
	if barTimeout <= 0 {
		barTimeout = 2 * time.Second
	}
	now := time.Now()
	return &GlobalConfirmedMetrics{
		startedAt:   now,
		lastAt:      now,
		barTimeout:  barTimeout,
		latenciesMs: make([]float64, 0, 1024),
		recoveryMs:  make([]float64, 0, 1024),
	}
}

func (m *GlobalConfirmedMetrics) ObserveGlobalConfirmed(out InstanceOutput, now time.Time) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.lastAt.IsZero() {
		gap := now.Sub(m.lastAt)
		if gap > m.barTimeout {
			m.recoveryMs = append(m.recoveryMs, float64(gap.Milliseconds()))
		}
	}
	m.lastAt = now
	m.globalConfirmedTotal++
	if out.IsNil || out.Block == nil {
		m.globalConfirmedNil++
		return
	}
	if out.Block.Timestamp > 0 {
		blockTs := time.Unix(0, out.Block.Timestamp)
		if !blockTs.After(now) {
			m.latenciesMs = append(m.latenciesMs, float64(now.Sub(blockTs).Milliseconds()))
		}
	}
}

func (m *GlobalConfirmedMetrics) Snapshot(orderer *GlobalOrderer, rejectByReason map[string]uint64) GlobalConfirmedSnapshot {
	m.mu.RLock()
	startedAt := m.startedAt
	lastAt := m.lastAt
	confirmedTotal := m.globalConfirmedTotal
	confirmedNil := m.globalConfirmedNil
	latencySamples := append([]float64(nil), m.latenciesMs...)
	recoverySamples := append([]float64(nil), m.recoveryMs...)
	m.mu.RUnlock()

	_, nilled, late := orderer.Stats()
	pending, missing := orderer.BacklogStats()
	rejectTotal := uint64(0)
	rejectCopy := make(map[string]uint64, len(rejectByReason))
	for k, v := range rejectByReason {
		rejectCopy[k] = v
		rejectTotal += v
	}

	windowSeconds := time.Since(startedAt).Seconds()
	throughput := 0.0
	if windowSeconds > 0 {
		throughput = float64(confirmedTotal-confirmedNil) / windowSeconds
	}

	return GlobalConfirmedSnapshot{
		GlobalConfirmedTotal: confirmedTotal,
		GlobalConfirmedNil:   confirmedNil,
		ThroughputTPS:        throughput,
		LatencyP50Ms:         quantile(latencySamples, 0.50),
		LatencyP95Ms:         quantile(latencySamples, 0.95),
		LatencyP99Ms:         quantile(latencySamples, 0.99),
		RecoveryP50Ms:        quantile(recoverySamples, 0.50),
		RecoveryP95Ms:        quantile(recoverySamples, 0.95),
		RecoveryP99Ms:        quantile(recoverySamples, 0.99),
		BacklogPending:       pending,
		BacklogMissing:       missing,
		OrdererLateTotal:     late,
		OrdererNilledTotal:   nilled,
		RejectTotal:          rejectTotal,
		RejectByReason:       rejectCopy,
		WindowStartUnixMs:    startedAt.UnixMilli(),
		LastUpdateUnixMs:     lastAt.UnixMilli(),
	}
}

func quantile(samples []float64, q float64) float64 {
	if len(samples) == 0 {
		return 0
	}
	sort.Float64s(samples)
	if q <= 0 {
		return samples[0]
	}
	if q >= 1 {
		return samples[len(samples)-1]
	}
	idx := int(float64(len(samples)-1) * q)
	return samples[idx]
}
