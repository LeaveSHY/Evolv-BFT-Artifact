package hotstuff

import (
	"math"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// Construction & Defaults
// ---------------------------------------------------------------------------

func TestNewLeaderReputation_AppliesDefaults(t *testing.T) {
	lr := NewLeaderReputation(ReputationConfig{})
	if lr.config.BaseTimeoutMs <= 0 {
		t.Error("BaseTimeoutMs should default to positive value")
	}
	if lr.config.MaxTimeoutMs <= 0 {
		t.Error("MaxTimeoutMs should default to positive value")
	}
	if lr.config.TimeoutMultiplier <= 1.0 {
		t.Error("TimeoutMultiplier should default > 1.0")
	}
	if lr.config.StragglerThreshold <= 0 || lr.config.StragglerThreshold > 1.0 {
		t.Error("StragglerThreshold should default to (0,1]")
	}
}

func TestDefaultReputationConfig_SensibleValues(t *testing.T) {
	cfg := DefaultReputationConfig()
	if cfg.BaseTimeoutMs != 1000 {
		t.Errorf("expected BaseTimeoutMs=1000, got %d", cfg.BaseTimeoutMs)
	}
	if cfg.LatencyWindowSize != 64 {
		t.Errorf("expected LatencyWindowSize=64, got %d", cfg.LatencyWindowSize)
	}
}

// ---------------------------------------------------------------------------
// RecordSuccess / RecordTimeout / RecordNilBlock
// ---------------------------------------------------------------------------

func TestRecordSuccess_ResetsConsecutiveTimeouts(t *testing.T) {
	lr := NewLeaderReputation(DefaultReputationConfig())
	lr.RecordTimeout(1)
	lr.RecordTimeout(1)
	lr.RecordSuccess(1)

	stats := lr.GetStats(1)
	if stats.ConsecutiveTimeouts != 0 {
		t.Errorf("expected consecutive timeouts reset to 0, got %d", stats.ConsecutiveTimeouts)
	}
	if stats.Successes != 1 {
		t.Errorf("expected 1 success, got %d", stats.Successes)
	}
	if stats.Timeouts != 2 {
		t.Errorf("expected 2 timeouts, got %d", stats.Timeouts)
	}
}

func TestRecordNilBlock_IncreasesFailureRate(t *testing.T) {
	lr := NewLeaderReputation(DefaultReputationConfig())
	lr.RecordSuccess(1)
	lr.RecordSuccess(1)
	lr.RecordNilBlock(1)

	stats := lr.GetStats(1)
	if stats.NilBlocks != 1 {
		t.Errorf("expected 1 nil block, got %d", stats.NilBlocks)
	}
	// failure rate = (0 + 1) / 3 = 0.333
	expected := 1.0 / 3.0
	if math.Abs(stats.FailureRate-expected) > 0.01 {
		t.Errorf("expected failure rate ~%.3f, got %.3f", expected, stats.FailureRate)
	}
}

// ---------------------------------------------------------------------------
// AdaptiveTimeout
// ---------------------------------------------------------------------------

func TestAdaptiveTimeout_BaseForUnknownLeader(t *testing.T) {
	lr := NewLeaderReputation(DefaultReputationConfig())
	d := lr.AdaptiveTimeout(999)
	if d != time.Duration(lr.config.BaseTimeoutMs)*time.Millisecond {
		t.Errorf("unknown leader should get base timeout, got %v", d)
	}
}

func TestAdaptiveTimeout_BaseAfterSuccess(t *testing.T) {
	lr := NewLeaderReputation(DefaultReputationConfig())
	lr.RecordSuccess(1)
	d := lr.AdaptiveTimeout(1)
	if d != time.Duration(lr.config.BaseTimeoutMs)*time.Millisecond {
		t.Errorf("leader with success should get base timeout, got %v", d)
	}
}

func TestAdaptiveTimeout_ExponentialBackoff(t *testing.T) {
	cfg := DefaultReputationConfig()
	lr := NewLeaderReputation(cfg)
	lr.RecordTimeout(1)
	lr.RecordTimeout(1)

	d := lr.AdaptiveTimeout(1)
	expected := float64(cfg.BaseTimeoutMs) * cfg.TimeoutMultiplier * cfg.TimeoutMultiplier
	if math.Abs(float64(d.Milliseconds())-expected) > 1 {
		t.Errorf("expected ~%.0fms after 2 timeouts, got %dms", expected, d.Milliseconds())
	}
}

func TestAdaptiveTimeout_CapsAtMax(t *testing.T) {
	cfg := DefaultReputationConfig()
	lr := NewLeaderReputation(cfg)
	for i := 0; i < 100; i++ {
		lr.RecordTimeout(1)
	}
	d := lr.AdaptiveTimeout(1)
	if d > time.Duration(cfg.MaxTimeoutMs)*time.Millisecond {
		t.Errorf("timeout should cap at max, got %v", d)
	}
}

// ---------------------------------------------------------------------------
// IsStraggler / IsCrashed
// ---------------------------------------------------------------------------

func TestIsStraggler_BelowMinSamples(t *testing.T) {
	cfg := DefaultReputationConfig()
	cfg.MinSamples = 5
	lr := NewLeaderReputation(cfg)
	lr.RecordTimeout(1)
	lr.RecordTimeout(1)

	if lr.IsStraggler(1) {
		t.Error("should not be straggler below MinSamples")
	}
}

func TestIsStraggler_AboveThreshold(t *testing.T) {
	cfg := DefaultReputationConfig()
	cfg.MinSamples = 3
	cfg.StragglerThreshold = 0.5
	lr := NewLeaderReputation(cfg)
	lr.RecordSuccess(1)
	lr.RecordTimeout(1)
	lr.RecordTimeout(1)
	lr.RecordTimeout(1)

	// failure = 3/4 = 0.75 > 0.5
	if !lr.IsStraggler(1) {
		t.Error("should be straggler at 75% failure rate")
	}
}

func TestIsStraggler_UnknownLeader(t *testing.T) {
	lr := NewLeaderReputation(DefaultReputationConfig())
	if lr.IsStraggler(999) {
		t.Error("unknown leader should not be straggler")
	}
}

func TestIsCrashed_AfterMaxConsecutiveTimeouts(t *testing.T) {
	cfg := DefaultReputationConfig()
	cfg.MaxConsecutiveTimeouts = 3
	lr := NewLeaderReputation(cfg)
	for i := 0; i < 3; i++ {
		lr.RecordTimeout(1)
	}
	if !lr.IsCrashed(1) {
		t.Error("should be crashed after max consecutive timeouts")
	}
}

func TestIsCrashed_ResetsAfterSuccess(t *testing.T) {
	cfg := DefaultReputationConfig()
	cfg.MaxConsecutiveTimeouts = 3
	lr := NewLeaderReputation(cfg)
	lr.RecordTimeout(1)
	lr.RecordTimeout(1)
	lr.RecordSuccess(1)
	lr.RecordTimeout(1)

	if lr.IsCrashed(1) {
		t.Error("should not be crashed after success reset")
	}
}

// ---------------------------------------------------------------------------
// SelectFallbackLeader
// ---------------------------------------------------------------------------

func TestSelectFallbackLeader_SkipsStragglers(t *testing.T) {
	cfg := DefaultReputationConfig()
	cfg.MinSamples = 3
	cfg.StragglerThreshold = 0.5
	lr := NewLeaderReputation(cfg)

	// Make leader 10 a straggler
	lr.RecordTimeout(10)
	lr.RecordTimeout(10)
	lr.RecordTimeout(10)

	validators := []uint64{10, 20, 30}
	fallback := lr.SelectFallbackLeader(10, validators, 1)
	if fallback == 10 {
		t.Error("fallback should skip straggler leader 10")
	}
	if fallback != 20 {
		t.Errorf("expected fallback 20, got %d", fallback)
	}
}

func TestSelectFallbackLeader_AllStragglers_PicksBest(t *testing.T) {
	cfg := DefaultReputationConfig()
	cfg.MinSamples = 3
	cfg.StragglerThreshold = 0.3
	lr := NewLeaderReputation(cfg)

	// All are stragglers but 30 is slightly better
	for _, id := range []uint64{10, 20, 30} {
		lr.RecordTimeout(id)
		lr.RecordTimeout(id)
		lr.RecordTimeout(id)
	}
	lr.RecordSuccess(30) // 30 has 3/4=75% failure, better than 100%

	validators := []uint64{10, 20, 30}
	fallback := lr.SelectFallbackLeader(10, validators, 1)
	if fallback != 30 {
		t.Errorf("expected best-of-bad leader 30, got %d", fallback)
	}
}

func TestSelectFallbackLeader_EmptyValidators(t *testing.T) {
	lr := NewLeaderReputation(DefaultReputationConfig())
	result := lr.SelectFallbackLeader(10, []uint64{}, 1)
	if result != 10 {
		t.Errorf("empty validators should return primary, got %d", result)
	}
}

// ---------------------------------------------------------------------------
// RecordEquivocation / RecordViewChangeInit / RecordLatency
// ---------------------------------------------------------------------------

func TestRecordEquivocation(t *testing.T) {
	cfg := DefaultReputationConfig()
	cfg.LatencyWindowSize = 4 // small window for test
	lr := NewLeaderReputation(cfg)
	lr.RecordSuccess(1)
	lr.RecordSuccess(1)
	lr.RecordSuccess(1)
	lr.RecordEquivocation(1)
	// Snapshot epoch into sliding window so TrustFeatureVector uses window-based rates
	lr.RecordEventEpoch(1)

	features, ok := lr.TrustFeatureVector(1)
	if !ok {
		t.Fatal("expected features available")
	}
	// With W=4, 1 equivocation / 4 = 0.25
	expected := 1.0 / 4.0
	if math.Abs(features.EquivocationRate-expected) > 0.01 {
		t.Errorf("expected equivocation rate %f, got %f", expected, features.EquivocationRate)
	}
}

func TestRecordViewChangeInit(t *testing.T) {
	cfg := DefaultReputationConfig()
	cfg.LatencyWindowSize = 4 // small window for test
	lr := NewLeaderReputation(cfg)
	lr.RecordSuccess(1)
	lr.RecordSuccess(1)
	lr.RecordSuccess(1)
	lr.RecordViewChangeInit(1)
	lr.RecordViewChangeInit(1)
	// Snapshot epoch into sliding window
	lr.RecordEventEpoch(1)

	features, ok := lr.TrustFeatureVector(1)
	if !ok {
		t.Fatal("expected features available")
	}
	// With W=4, 2 view-change inits / 4 = 0.50
	expected := 2.0 / 4.0
	if math.Abs(features.ViewChangeRate-expected) > 0.01 {
		t.Errorf("expected view change rate %f, got %f", expected, features.ViewChangeRate)
	}
}

func TestRecordLatency_MeanAndStd(t *testing.T) {
	cfg := DefaultReputationConfig()
	cfg.MinSamples = 3
	lr := NewLeaderReputation(cfg)

	lr.RecordSuccess(1)
	lr.RecordSuccess(1)
	lr.RecordSuccess(1)

	// Record latencies: 100, 200, 300 → mean=200, std=100
	lr.RecordLatency(1, 100)
	lr.RecordLatency(1, 200)
	lr.RecordLatency(1, 300)

	features, ok := lr.TrustFeatureVector(1)
	if !ok {
		t.Fatal("expected features available")
	}
	// Mean = 200/15000 ≈ 0.0133 (normalized by MaxTimeoutMs=15000)
	expectedMean := 200.0 / float64(cfg.MaxTimeoutMs)
	if math.Abs(features.MeanLatency-expectedMean) > 0.001 {
		t.Errorf("expected normalized mean ~%.4f, got %.4f", expectedMean, features.MeanLatency)
	}
	if features.StdLatency <= 0 {
		t.Error("expected positive std latency")
	}
}

// ---------------------------------------------------------------------------
// TrustFeatureVector
// ---------------------------------------------------------------------------

func TestTrustFeatureVector_BelowMinSamples(t *testing.T) {
	lr := NewLeaderReputation(DefaultReputationConfig())
	lr.RecordSuccess(1)

	_, ok := lr.TrustFeatureVector(1)
	if ok {
		t.Error("should return false below MinSamples")
	}
}

func TestTrustFeatureVector_UnknownNode(t *testing.T) {
	lr := NewLeaderReputation(DefaultReputationConfig())
	_, ok := lr.TrustFeatureVector(999)
	if ok {
		t.Error("should return false for unknown node")
	}
}

func TestTrustFeatureVector_AllValuesClamped(t *testing.T) {
	cfg := DefaultReputationConfig()
	cfg.MinSamples = 1
	lr := NewLeaderReputation(cfg)
	lr.RecordTimeout(1) // 100% timeout rate

	features, ok := lr.TrustFeatureVector(1)
	if !ok {
		t.Fatal("expected features available")
	}
	if features.TimeoutRate < 0 || features.TimeoutRate > 1 {
		t.Errorf("timeout rate out of [0,1]: %f", features.TimeoutRate)
	}
	if features.EquivocationRate < 0 || features.EquivocationRate > 1 {
		t.Errorf("equivocation rate out of [0,1]: %f", features.EquivocationRate)
	}
}

// ---------------------------------------------------------------------------
// GetAllStats / GetAllTrustFeatures
// ---------------------------------------------------------------------------

func TestGetAllStats(t *testing.T) {
	lr := NewLeaderReputation(DefaultReputationConfig())
	lr.RecordSuccess(1)
	lr.RecordTimeout(2)

	stats := lr.GetAllStats()
	if len(stats) != 2 {
		t.Fatalf("expected 2 stats entries, got %d", len(stats))
	}
}

func TestGetAllTrustFeatures_FiltersMinSamples(t *testing.T) {
	cfg := DefaultReputationConfig()
	cfg.MinSamples = 3
	lr := NewLeaderReputation(cfg)

	lr.RecordSuccess(1) // only 1 sample → excluded
	for i := 0; i < 5; i++ {
		lr.RecordSuccess(2) // 5 samples → included
	}

	features := lr.GetAllTrustFeatures()
	if len(features) != 1 {
		t.Fatalf("expected 1 node with enough samples, got %d", len(features))
	}
	if _, ok := features[2]; !ok {
		t.Error("expected node 2 in features")
	}
}

// ---------------------------------------------------------------------------
// SetBaseTimeoutMs
// ---------------------------------------------------------------------------

func TestSetBaseTimeoutMs(t *testing.T) {
	lr := NewLeaderReputation(DefaultReputationConfig())
	lr.SetBaseTimeoutMs(500)
	d := lr.AdaptiveTimeout(999)
	if d != 500*time.Millisecond {
		t.Errorf("expected 500ms, got %v", d)
	}
}

func TestSetBaseTimeoutMs_IgnoresZero(t *testing.T) {
	lr := NewLeaderReputation(DefaultReputationConfig())
	original := lr.config.BaseTimeoutMs
	lr.SetBaseTimeoutMs(0)
	if lr.config.BaseTimeoutMs != original {
		t.Error("SetBaseTimeoutMs(0) should not change config")
	}
}
