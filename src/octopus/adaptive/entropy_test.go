package adaptive

import (
	"math"
	"sync"
	"testing"
)

func TestEntropy_SingleAction_ZeroEntropy(t *testing.T) {
	em := NewEntropyMonitor(EntropyConfig{WindowSize: 50, LowEntropyThreshold: 0.5, HighEntropyThreshold: 2.0})
	action := Action{CommitteeSize: 10, PacemakerTimeoutMs: 500, MempoolMaxBatchTxs: 1024}

	// Same action repeated → entropy should be 0
	for i := 0; i < 50; i++ {
		em.Observe(action)
	}
	entropy := em.Entropy()
	if entropy != 0 {
		t.Errorf("single action should have 0 entropy, got %f", entropy)
	}
}

func TestEntropy_UniformDistribution_MaxEntropy(t *testing.T) {
	em := NewEntropyMonitor(EntropyConfig{WindowSize: 100, LowEntropyThreshold: 0.5, HighEntropyThreshold: 2.0})
	// 4 distinct actions in round-robin → H = log₂(4) = 2 bits
	actions := []Action{
		{CommitteeSize: 5, PacemakerTimeoutMs: 100, MempoolMaxBatchTxs: 512},
		{CommitteeSize: 10, PacemakerTimeoutMs: 200, MempoolMaxBatchTxs: 1024},
		{CommitteeSize: 15, PacemakerTimeoutMs: 300, MempoolMaxBatchTxs: 1536},
		{CommitteeSize: 20, PacemakerTimeoutMs: 400, MempoolMaxBatchTxs: 2048},
	}
	for i := 0; i < 100; i++ {
		em.Observe(actions[i%4])
	}
	entropy := em.Entropy()
	expected := math.Log2(4) // 2.0 bits
	if math.Abs(entropy-expected) > 0.01 {
		t.Errorf("uniform 4-action entropy should be ~%.2f bits, got %.4f", expected, entropy)
	}
}

func TestEntropy_LowEntropyAlarm(t *testing.T) {
	em := NewEntropyMonitor(EntropyConfig{WindowSize: 50, LowEntropyThreshold: 0.5, HighEntropyThreshold: 2.0})
	action := Action{CommitteeSize: 10, PacemakerTimeoutMs: 500, MempoolMaxBatchTxs: 1024}

	for i := 0; i < 50; i++ {
		em.Observe(action)
	}
	if !em.IsLowEntropy() {
		t.Error("should flag low entropy for single-action policy")
	}
	stats := em.Stats()
	if stats.LowAlarmCount == 0 {
		t.Error("should have accumulated low alarm count")
	}
}

func TestEntropy_HighEntropy_NoAlarm(t *testing.T) {
	em := NewEntropyMonitor(EntropyConfig{WindowSize: 80, LowEntropyThreshold: 0.5, HighEntropyThreshold: 2.0})
	actions := []Action{
		{CommitteeSize: 5, PacemakerTimeoutMs: 100, MempoolMaxBatchTxs: 512},
		{CommitteeSize: 10, PacemakerTimeoutMs: 200, MempoolMaxBatchTxs: 1024},
		{CommitteeSize: 15, PacemakerTimeoutMs: 300, MempoolMaxBatchTxs: 1536},
		{CommitteeSize: 20, PacemakerTimeoutMs: 400, MempoolMaxBatchTxs: 2048},
	}
	for i := 0; i < 80; i++ {
		em.Observe(actions[i%4])
	}
	if em.IsLowEntropy() {
		t.Error("should NOT flag low entropy for diverse actions")
	}
}

func TestEntropy_WindowRingBuffer(t *testing.T) {
	em := NewEntropyMonitor(EntropyConfig{WindowSize: 10, LowEntropyThreshold: 0.1, HighEntropyThreshold: 2.0})
	a1 := Action{CommitteeSize: 5, PacemakerTimeoutMs: 100, MempoolMaxBatchTxs: 512}
	a2 := Action{CommitteeSize: 50, PacemakerTimeoutMs: 1000, MempoolMaxBatchTxs: 4096}

	// Fill window with diverse actions
	for i := 0; i < 10; i++ {
		if i%2 == 0 {
			em.Observe(a1)
		} else {
			em.Observe(a2)
		}
	}
	eDiverse := em.Entropy()

	// Overwrite with single action → entropy should drop to 0
	for i := 0; i < 10; i++ {
		em.Observe(a1)
	}
	eMono := em.Entropy()
	if eMono >= eDiverse {
		t.Errorf("overwritten window should have lower entropy: mono=%f, diverse=%f", eMono, eDiverse)
	}
	if eMono != 0 {
		t.Errorf("all-same window should have 0 entropy, got %f", eMono)
	}
}

func TestEntropy_Stats(t *testing.T) {
	em := NewEntropyMonitor(DefaultEntropyConfig())
	a := Action{CommitteeSize: 10, PacemakerTimeoutMs: 500, MempoolMaxBatchTxs: 1024}
	for i := 0; i < 20; i++ {
		em.Observe(a)
	}
	stats := em.Stats()
	if stats.TotalObservations != 20 {
		t.Errorf("expected 20 observations, got %d", stats.TotalObservations)
	}
	if stats.WindowSize != 100 {
		t.Errorf("expected window size 100, got %d", stats.WindowSize)
	}
	if stats.WindowFill != 20 {
		t.Errorf("expected window fill 20, got %d", stats.WindowFill)
	}
	if stats.UniqueBuckets != 1 {
		t.Errorf("expected 1 unique bucket, got %d", stats.UniqueBuckets)
	}
}

func TestEntropy_NotTriggeredDuringWarmup(t *testing.T) {
	// Low entropy alarm should not fire until window is at least half full
	em := NewEntropyMonitor(EntropyConfig{WindowSize: 100, LowEntropyThreshold: 0.5, HighEntropyThreshold: 2.0})
	a := Action{CommitteeSize: 10, PacemakerTimeoutMs: 500, MempoolMaxBatchTxs: 1024}

	// Only 10 observations (< 50 = half window)
	for i := 0; i < 10; i++ {
		em.Observe(a)
	}
	if em.IsLowEntropy() {
		t.Error("should not flag low entropy before window is half-full")
	}
}

func TestEntropy_ConcurrentSafe(t *testing.T) {
	em := NewEntropyMonitor(DefaultEntropyConfig())
	actions := []Action{
		{CommitteeSize: 5, PacemakerTimeoutMs: 100, MempoolMaxBatchTxs: 512},
		{CommitteeSize: 10, PacemakerTimeoutMs: 200, MempoolMaxBatchTxs: 1024},
		{CommitteeSize: 15, PacemakerTimeoutMs: 300, MempoolMaxBatchTxs: 1536},
	}
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(seed int) {
			defer wg.Done()
			for i := 0; i < 100; i++ {
				em.Observe(actions[(seed+i)%len(actions)])
			}
		}(g)
	}
	wg.Wait()

	stats := em.Stats()
	if stats.TotalObservations != 800 {
		t.Errorf("expected 800 observations, got %d", stats.TotalObservations)
	}
	if math.IsNaN(stats.Entropy) || stats.Entropy < 0 {
		t.Errorf("entropy should be non-negative and not NaN, got %f", stats.Entropy)
	}
}
