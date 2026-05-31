package adaptive

import (
	"math"
	"sync"
	"testing"
)

func TestAdvantage_DuringWarmup_ReturnsRawReward(t *testing.T) {
	cfg := DefaultAdvantageConfig()
	cfg.WarmupSteps = 5
	vb := NewValueBaseline(cfg)

	for i := 0; i < 5; i++ {
		reward := float64(i + 1) // 1,2,3,4,5
		adv := vb.Advantage(reward)
		if adv != reward {
			t.Errorf("step %d: warmup should return raw reward %.2f, got %.2f", i, reward, adv)
		}
	}
	if vb.Steps() != 5 {
		t.Errorf("expected 5 steps, got %d", vb.Steps())
	}
}

func TestAdvantage_AfterWarmup_SubtractsBaseline(t *testing.T) {
	cfg := DefaultAdvantageConfig()
	cfg.WarmupSteps = 3
	cfg.NormalizeAdvantage = false // disable normalization for clarity
	vb := NewValueBaseline(cfg)

	// Warmup with constant 1.0
	for i := 0; i < 3; i++ {
		vb.Advantage(1.0)
	}
	// Baseline should be ~1.0 after warmup
	baseline := vb.Baseline()
	if math.Abs(baseline-1.0) > 1e-6 {
		t.Fatalf("expected baseline ~1.0 after warmup, got %f", baseline)
	}

	// Now a reward of 2.0 should give advantage ~1.0 (2.0 - 1.0)
	adv := vb.Advantage(2.0)
	if math.Abs(adv-1.0) > 0.1 {
		t.Errorf("expected advantage ~1.0, got %f", adv)
	}
}

func TestAdvantage_BaselineAdaptsToReward(t *testing.T) {
	cfg := DefaultAdvantageConfig()
	cfg.WarmupSteps = 0
	cfg.Alpha = 0.5 // fast adaptation
	cfg.NormalizeAdvantage = false
	vb := NewValueBaseline(cfg)

	// First observation sets baseline
	vb.Advantage(10.0)

	// After many observations at 20.0, baseline should approach 20.0
	for i := 0; i < 50; i++ {
		vb.Advantage(20.0)
	}
	baseline := vb.Baseline()
	if math.Abs(baseline-20.0) > 0.1 {
		t.Errorf("expected baseline ~20.0 after convergence, got %f", baseline)
	}
}

func TestAdvantage_NormalizationReducesMagnitude(t *testing.T) {
	cfg := DefaultAdvantageConfig()
	cfg.WarmupSteps = 0
	cfg.Alpha = 0.1
	cfg.NormalizeAdvantage = false
	vbRaw := NewValueBaseline(cfg)

	cfg.NormalizeAdvantage = true
	vbNorm := NewValueBaseline(cfg)

	// Feed same sequence
	rewards := []float64{1, 5, 2, 8, 3, 7, 4, 6}
	for _, r := range rewards {
		vbRaw.Advantage(r)
		vbNorm.Advantage(r)
	}

	// Now a big reward: unnormalized should have large magnitude
	rawAdv := vbRaw.Advantage(100.0)
	normAdv := vbNorm.Advantage(100.0)

	if math.Abs(normAdv) >= math.Abs(rawAdv) {
		t.Errorf("normalized advantage (%.2f) should be smaller than raw (%.2f)", normAdv, rawAdv)
	}
}

func TestAdvantage_ZeroRewardZeroAdvantage(t *testing.T) {
	cfg := DefaultAdvantageConfig()
	cfg.WarmupSteps = 0
	cfg.NormalizeAdvantage = false
	vb := NewValueBaseline(cfg)

	// All zeros → baseline stays 0 → advantage always 0
	for i := 0; i < 10; i++ {
		adv := vb.Advantage(0.0)
		if adv != 0.0 {
			t.Errorf("step %d: expected 0 advantage for constant 0 reward, got %f", i, adv)
		}
	}
}

func TestAdvantage_Stats(t *testing.T) {
	cfg := DefaultAdvantageConfig()
	cfg.WarmupSteps = 2
	cfg.NormalizeAdvantage = false
	vb := NewValueBaseline(cfg)

	vb.Advantage(1.0) // warmup
	vb.Advantage(1.0) // warmup
	vb.Advantage(2.0) // first real: adv = 2.0-1.0 = 1.0
	vb.Advantage(0.0) // adv = 0.0 - baseline (shifted)

	stats := vb.Stats()
	if stats.Steps != 4 {
		t.Errorf("expected 4 steps, got %d", stats.Steps)
	}
	if !stats.WarmedUp {
		t.Error("should be warmed up after 4 steps with warmup=2")
	}
	if stats.MaxAdv < stats.MinAdv {
		t.Error("max should be >= min")
	}
}

func TestAdvantage_IsWarmedUp(t *testing.T) {
	cfg := DefaultAdvantageConfig()
	cfg.WarmupSteps = 3
	vb := NewValueBaseline(cfg)

	if vb.IsWarmedUp() {
		t.Error("should not be warmed up initially")
	}
	vb.Advantage(1.0)
	vb.Advantage(1.0)
	if vb.IsWarmedUp() {
		t.Error("should not be warmed up after 2 steps")
	}
	vb.Advantage(1.0)
	if !vb.IsWarmedUp() {
		t.Error("should be warmed up after 3 steps")
	}
}

func TestAdvantage_ConcurrentSafe(t *testing.T) {
	cfg := DefaultAdvantageConfig()
	cfg.WarmupSteps = 0
	vb := NewValueBaseline(cfg)

	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(seed int) {
			defer wg.Done()
			for i := 0; i < 200; i++ {
				vb.Advantage(float64(seed*100 + i))
			}
		}(g)
	}
	wg.Wait()

	if vb.Steps() != 1600 {
		t.Errorf("expected 1600 steps, got %d", vb.Steps())
	}
	stats := vb.Stats()
	if math.IsNaN(stats.Baseline) || math.IsInf(stats.Baseline, 0) {
		t.Error("baseline should not be NaN or Inf after concurrent access")
	}
}
