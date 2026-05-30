package adaptive

import (
	"math"
	"sync"
	"testing"
)

func TestExplorationBonus_UnvisitedReturnsMax(t *testing.T) {
	eb := NewExplorationBonus(DefaultExplorationConfig())
	action := Action{CommitteeSize: 10, PacemakerTimeoutMs: 500, MempoolMaxBatchTxs: 1024}
	bonus := eb.Bonus(action)
	if bonus != 1.0 {
		t.Errorf("unvisited action should get MaxBonus=1.0, got %f", bonus)
	}
}

func TestExplorationBonus_DecaysWithVisits(t *testing.T) {
	eb := NewExplorationBonus(ExplorationConfig{
		Beta:                1.0,
		CommitteeBucketSize: 5,
		TimeoutBucketMs:     100,
		BatchBucketSize:     512,
		MaxBonus:            10.0, // high cap to avoid clamping
	})
	action := Action{CommitteeSize: 10, PacemakerTimeoutMs: 500, MempoolMaxBatchTxs: 1024}

	// Observe the same action repeatedly; bonus should decay as 1/√N
	var lastBonus float64 = math.MaxFloat64
	for i := 0; i < 100; i++ {
		bonus := eb.ObserveAndBonus(action)
		if bonus >= lastBonus {
			t.Errorf("iteration %d: bonus %f should be strictly less than previous %f", i+1, bonus, lastBonus)
		}
		lastBonus = bonus
	}
	// After 100 visits: β/√100 = 1/10 = 0.1
	expected := 1.0 / math.Sqrt(100)
	if math.Abs(lastBonus-expected) > 1e-9 {
		t.Errorf("after 100 visits expected bonus ~%f, got %f", expected, lastBonus)
	}
}

func TestExplorationBonus_DifferentBucketsIndependent(t *testing.T) {
	eb := NewExplorationBonus(DefaultExplorationConfig())
	action1 := Action{CommitteeSize: 5, PacemakerTimeoutMs: 200, MempoolMaxBatchTxs: 512}
	action2 := Action{CommitteeSize: 20, PacemakerTimeoutMs: 800, MempoolMaxBatchTxs: 4096}

	// Visit action1 many times
	for i := 0; i < 50; i++ {
		eb.ObserveAndBonus(action1)
	}

	// action2 should still have max bonus (unvisited)
	bonus2 := eb.Bonus(action2)
	if bonus2 != 1.0 {
		t.Errorf("unvisited action2 should have MaxBonus=1.0, got %f", bonus2)
	}
}

func TestExplorationBonus_SameBucketMerges(t *testing.T) {
	cfg := ExplorationConfig{
		Beta:                1.0,
		CommitteeBucketSize: 10,
		TimeoutBucketMs:     200,
		BatchBucketSize:     1000,
		MaxBonus:            5.0,
	}
	eb := NewExplorationBonus(cfg)

	// These two actions fall into the same bucket (committee: 10/10=1, timeout: 300/200=1, batch: 500/1000=0)
	a1 := Action{CommitteeSize: 11, PacemakerTimeoutMs: 350, MempoolMaxBatchTxs: 500}
	a2 := Action{CommitteeSize: 15, PacemakerTimeoutMs: 380, MempoolMaxBatchTxs: 900}

	eb.ObserveAndBonus(a1)
	// a2 should share the same bucket and thus have count=1
	bonus := eb.Bonus(a2)
	expected := 1.0 / math.Sqrt(1) // β/√1 = 1.0 (clamped by MaxBonus=5.0)
	if math.Abs(bonus-expected) > 1e-9 {
		t.Errorf("same-bucket action expected bonus %f, got %f", expected, bonus)
	}

	if eb.UniqueBuckets() != 1 {
		t.Errorf("expected 1 unique bucket, got %d", eb.UniqueBuckets())
	}
}

func TestExplorationBonus_MaxBonusCap(t *testing.T) {
	cfg := ExplorationConfig{
		Beta:                100.0, // very high beta
		CommitteeBucketSize: 5,
		TimeoutBucketMs:     100,
		BatchBucketSize:     512,
		MaxBonus:            2.0, // low cap
	}
	eb := NewExplorationBonus(cfg)
	action := Action{CommitteeSize: 10, PacemakerTimeoutMs: 500, MempoolMaxBatchTxs: 1024}

	// Even with beta=100, bonus should be capped at MaxBonus
	bonus := eb.ObserveAndBonus(action) // β/√1 = 100, capped to 2.0
	if bonus != 2.0 {
		t.Errorf("expected capped bonus=2.0, got %f", bonus)
	}
}

func TestExplorationBonus_MinBonusFloor(t *testing.T) {
	cfg := ExplorationConfig{
		Beta:                0.01, // very low beta
		CommitteeBucketSize: 5,
		TimeoutBucketMs:     100,
		BatchBucketSize:     512,
		MinBonus:            0.05, // floor
		MaxBonus:            1.0,
	}
	eb := NewExplorationBonus(cfg)
	action := Action{CommitteeSize: 10, PacemakerTimeoutMs: 500, MempoolMaxBatchTxs: 1024}

	// Visit 100 times: β/√100 = 0.01/10 = 0.001, below MinBonus=0.05
	for i := 0; i < 100; i++ {
		eb.ObserveAndBonus(action)
	}
	bonus := eb.Bonus(action)
	if bonus != 0.05 {
		t.Errorf("expected floor bonus=0.05, got %f", bonus)
	}
}

func TestExplorationBonus_Stats(t *testing.T) {
	eb := NewExplorationBonus(DefaultExplorationConfig())
	a1 := Action{CommitteeSize: 5, PacemakerTimeoutMs: 200, MempoolMaxBatchTxs: 512}
	a2 := Action{CommitteeSize: 50, PacemakerTimeoutMs: 1000, MempoolMaxBatchTxs: 8192}

	for i := 0; i < 10; i++ {
		eb.ObserveAndBonus(a1)
	}
	for i := 0; i < 5; i++ {
		eb.ObserveAndBonus(a2)
	}

	stats := eb.Stats()
	if stats.TotalVisits != 15 {
		t.Errorf("expected TotalVisits=15, got %d", stats.TotalVisits)
	}
	if stats.UniqueBuckets != 2 {
		t.Errorf("expected UniqueBuckets=2, got %d", stats.UniqueBuckets)
	}
	if stats.Beta != 1.0 {
		t.Errorf("expected Beta=1.0, got %f", stats.Beta)
	}
	if stats.AvgBonus <= 0 {
		t.Error("expected positive AvgBonus")
	}
}

func TestExplorationBonus_ConcurrentSafe(t *testing.T) {
	eb := NewExplorationBonus(DefaultExplorationConfig())
	actions := []Action{
		{CommitteeSize: 5, PacemakerTimeoutMs: 200, MempoolMaxBatchTxs: 512},
		{CommitteeSize: 15, PacemakerTimeoutMs: 600, MempoolMaxBatchTxs: 2048},
		{CommitteeSize: 25, PacemakerTimeoutMs: 1000, MempoolMaxBatchTxs: 4096},
	}

	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			a := actions[idx%len(actions)]
			eb.ObserveAndBonus(a)
			eb.Bonus(a)
			eb.Stats()
		}(i)
	}
	wg.Wait()

	if eb.TotalVisits() != 100 {
		t.Errorf("expected 100 total visits after concurrent access, got %d", eb.TotalVisits())
	}
}

func TestExplorationBonus_SqrtBetaDecay(t *testing.T) {
	cfg := ExplorationConfig{
		Beta:                2.0,
		BetaDecay:           "sqrt",
		CommitteeBucketSize: 1,
		TimeoutBucketMs:     100,
		BatchBucketSize:     512,
		MaxBonus:            100.0, // high to avoid clamping
	}
	eb := NewExplorationBonus(cfg)
	action := Action{CommitteeSize: 1, PacemakerTimeoutMs: 100, MempoolMaxBatchTxs: 512}

	// First observation: β(1)=2.0, bonus for N=1 = 2.0/√1 = 2.0
	b1 := eb.ObserveAndBonus(action)

	// After many observations, effective beta should shrink
	for i := 0; i < 99; i++ {
		eb.ObserveAndBonus(action)
	}
	// totalN=100, effective beta = 2.0/√100 = 0.2, N(action)=100, bonus = 0.2/√100 = 0.02
	b100 := eb.Bonus(action)
	if b100 >= b1 {
		t.Errorf("bonus should decrease with sqrt decay: b1=%f, b100=%f", b1, b100)
	}
	// With sqrt decay: β(100) = 2/√100 = 0.2, bonus = 0.2/√100 = 0.02
	expected := 2.0 / math.Sqrt(100) / math.Sqrt(100)
	if math.Abs(b100-expected) > 0.01 {
		t.Errorf("expected bonus ~%.4f, got %.4f", expected, b100)
	}
}

func TestExplorationBonus_LogBetaDecay(t *testing.T) {
	cfg := ExplorationConfig{
		Beta:                2.0,
		BetaDecay:           "log",
		CommitteeBucketSize: 1,
		TimeoutBucketMs:     100,
		BatchBucketSize:     512,
		MaxBonus:            100.0,
	}
	eb := NewExplorationBonus(cfg)
	action := Action{CommitteeSize: 1, PacemakerTimeoutMs: 100, MempoolMaxBatchTxs: 512}

	b1 := eb.ObserveAndBonus(action)
	for i := 0; i < 99; i++ {
		eb.ObserveAndBonus(action)
	}
	b100 := eb.Bonus(action)
	if b100 >= b1 {
		t.Errorf("bonus should decrease with log decay: b1=%f, b100=%f", b1, b100)
	}
	// Log decay should be slower than sqrt decay
	cfgSqrt := cfg
	cfgSqrt.BetaDecay = "sqrt"
	ebSqrt := NewExplorationBonus(cfgSqrt)
	for i := 0; i < 100; i++ {
		ebSqrt.ObserveAndBonus(action)
	}
	bSqrt := ebSqrt.Bonus(action)
	if b100 <= bSqrt {
		t.Errorf("log decay should be slower (higher bonus) than sqrt: log=%f, sqrt=%f", b100, bSqrt)
	}
}

func TestExplorationBonus_NoBetaDecayConstant(t *testing.T) {
	cfg := ExplorationConfig{
		Beta:                2.0,
		BetaDecay:           "none",
		CommitteeBucketSize: 1,
		TimeoutBucketMs:     100,
		BatchBucketSize:     512,
		MaxBonus:            100.0,
	}
	eb := NewExplorationBonus(cfg)
	action := Action{CommitteeSize: 1, PacemakerTimeoutMs: 100, MempoolMaxBatchTxs: 512}

	for i := 0; i < 100; i++ {
		eb.ObserveAndBonus(action)
	}
	// With no decay, bonus = 2.0/√100 = 0.2 (constant β=2.0)
	b := eb.Bonus(action)
	expected := 2.0 / math.Sqrt(100)
	if math.Abs(b-expected) > 0.01 {
		t.Errorf("no decay: expected bonus ~%.4f, got %.4f", expected, b)
	}
}

func TestExplorationBonus_StatsShowsEffectiveBeta(t *testing.T) {
	cfg := DefaultExplorationConfig()
	cfg.BetaDecay = "sqrt"
	eb := NewExplorationBonus(cfg)
	action := Action{CommitteeSize: 10, PacemakerTimeoutMs: 500, MempoolMaxBatchTxs: 1024}

	for i := 0; i < 100; i++ {
		eb.ObserveAndBonus(action)
	}
	stats := eb.Stats()
	if stats.EffectiveBeta >= stats.Beta {
		t.Errorf("effective beta should be less than initial: effective=%f, initial=%f",
			stats.EffectiveBeta, stats.Beta)
	}
	if stats.BetaDecay != "sqrt" {
		t.Errorf("expected BetaDecay=sqrt, got %s", stats.BetaDecay)
	}
}
