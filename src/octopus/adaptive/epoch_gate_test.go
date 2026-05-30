package adaptive

import (
	"sync"
	"testing"
	"time"
)

func TestEpochGate_FirstActionAppliedImmediately(t *testing.T) {
	eg := NewEpochGate(DefaultEpochGateConfig())
	now := time.Now()

	result := eg.Submit(Action{CommitteeSize: 10}, false, now)
	if result != GateApplied {
		t.Errorf("first action should be GateApplied, got %d", result)
	}
}

func TestEpochGate_CooldownDefersAction(t *testing.T) {
	cfg := DefaultEpochGateConfig()
	cfg.MinCooldownEpochs = 3
	eg := NewEpochGate(cfg)
	now := time.Now()

	// First action: applied immediately at epoch 0
	result := eg.Submit(Action{CommitteeSize: 10}, false, now)
	if result != GateApplied {
		t.Fatalf("first action should apply, got %d", result)
	}

	// Advance to epoch 1 (cooldown = 3, so next apply at epoch 3)
	eg.AdvanceEpoch(1, now)

	// Second action at epoch 1: should be deferred
	result = eg.Submit(Action{CommitteeSize: 20}, false, now)
	if result != GateDeferred {
		t.Errorf("action at epoch 1 should be deferred (cooldown 3), got %d", result)
	}
	if !eg.HasPending() {
		t.Error("should have pending action")
	}
}

func TestEpochGate_ReleaseAtEpochBoundary(t *testing.T) {
	cfg := DefaultEpochGateConfig()
	cfg.MinCooldownEpochs = 2
	eg := NewEpochGate(cfg)
	now := time.Now()

	// Apply first action at epoch 0
	eg.Submit(Action{CommitteeSize: 10}, false, now)

	// Advance to epoch 1, submit another
	eg.AdvanceEpoch(1, now)
	eg.Submit(Action{CommitteeSize: 20}, false, now)

	// Advance to epoch 2 (cooldown satisfied: 2 >= 0 + 2)
	released := eg.AdvanceEpoch(2, now)
	if released == nil {
		t.Fatal("expected pending action to be released at epoch 2")
	}
	if released.CommitteeSize != 20 {
		t.Errorf("expected committee size 20, got %d", released.CommitteeSize)
	}
	if eg.HasPending() {
		t.Error("pending should be cleared after release")
	}
}

func TestEpochGate_NewerActionReplacesOlder(t *testing.T) {
	cfg := DefaultEpochGateConfig()
	cfg.MinCooldownEpochs = 5
	eg := NewEpochGate(cfg)
	now := time.Now()

	// Apply first, then defer two more
	eg.Submit(Action{CommitteeSize: 10}, false, now)
	eg.AdvanceEpoch(1, now)

	result1 := eg.Submit(Action{CommitteeSize: 20}, false, now)
	if result1 != GateDeferred {
		t.Fatalf("expected deferred, got %d", result1)
	}

	result2 := eg.Submit(Action{CommitteeSize: 30}, false, now)
	if result2 != GateDropped {
		t.Errorf("second deferred should report GateDropped (replaced), got %d", result2)
	}

	// Release at epoch 5
	released := eg.AdvanceEpoch(5, now)
	if released == nil {
		t.Fatal("expected release at epoch 5")
	}
	if released.CommitteeSize != 30 {
		t.Errorf("should release latest action (30), got %d", released.CommitteeSize)
	}
}

func TestEpochGate_EmergencyBypassesCooldown(t *testing.T) {
	cfg := DefaultEpochGateConfig()
	cfg.MinCooldownEpochs = 10
	cfg.AllowEmergencyBypass = true
	eg := NewEpochGate(cfg)
	now := time.Now()

	// Apply first action
	eg.Submit(Action{CommitteeSize: 10}, false, now)

	// Emergency at epoch 0 (within cooldown): should bypass
	result := eg.Submit(Action{CommitteeSize: 99}, true, now)
	if result != GateEmergency {
		t.Errorf("emergency should bypass, got %d", result)
	}
}

func TestEpochGate_EmergencyDisabled(t *testing.T) {
	cfg := DefaultEpochGateConfig()
	cfg.MinCooldownEpochs = 10
	cfg.AllowEmergencyBypass = false
	eg := NewEpochGate(cfg)
	now := time.Now()

	// Apply first
	eg.Submit(Action{CommitteeSize: 10}, false, now)

	// "Emergency" but bypass disabled: should defer
	result := eg.Submit(Action{CommitteeSize: 99}, true, now)
	if result != GateDeferred {
		t.Errorf("with bypass disabled, emergency should defer, got %d", result)
	}
}

func TestEpochGate_StaleActionDropped(t *testing.T) {
	cfg := DefaultEpochGateConfig()
	cfg.MinCooldownEpochs = 5
	cfg.MaxPendingAge = 10 * time.Second
	eg := NewEpochGate(cfg)
	now := time.Now()

	// Apply first, then defer
	eg.Submit(Action{CommitteeSize: 10}, false, now)
	eg.AdvanceEpoch(1, now)
	eg.Submit(Action{CommitteeSize: 20}, false, now)

	// Advance epoch 5, but 30 seconds later (action is stale)
	future := now.Add(30 * time.Second)
	released := eg.AdvanceEpoch(5, future)
	if released != nil {
		t.Error("stale action should be dropped, not released")
	}
	if eg.HasPending() {
		t.Error("stale action should be cleared from pending")
	}

	stats := eg.Stats()
	if stats.TotalStale != 1 {
		t.Errorf("expected 1 stale drop, got %d", stats.TotalStale)
	}
}

func TestEpochGate_NoAdvanceNoRelease(t *testing.T) {
	eg := NewEpochGate(DefaultEpochGateConfig())
	now := time.Now()

	// Try to "advance" to same or lower epoch
	released := eg.AdvanceEpoch(0, now)
	if released != nil {
		t.Error("no advance should return nil")
	}
}

func TestEpochGate_Stats(t *testing.T) {
	cfg := DefaultEpochGateConfig()
	cfg.MinCooldownEpochs = 2
	eg := NewEpochGate(cfg)
	now := time.Now()

	eg.Submit(Action{}, false, now)     // applied
	eg.AdvanceEpoch(1, now)
	eg.Submit(Action{}, false, now)     // deferred
	eg.Submit(Action{}, true, now)      // emergency

	stats := eg.Stats()
	if stats.TotalSubmitted != 3 {
		t.Errorf("expected 3 submitted, got %d", stats.TotalSubmitted)
	}
	if stats.TotalApplied != 1 {
		t.Errorf("expected 1 applied, got %d", stats.TotalApplied)
	}
	if stats.TotalEmergency != 1 {
		t.Errorf("expected 1 emergency, got %d", stats.TotalEmergency)
	}
}

func TestEpochGate_ConcurrentSafe(t *testing.T) {
	eg := NewEpochGate(DefaultEpochGateConfig())
	now := time.Now()

	var wg sync.WaitGroup
	for g := 0; g < 4; g++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := 0; i < 100; i++ {
				eg.Submit(Action{CommitteeSize: id*100 + i}, i%10 == 0, now)
			}
		}(g)
	}
	for g := 0; g < 4; g++ {
		wg.Add(1)
		go func(start int) {
			defer wg.Done()
			for i := 0; i < 100; i++ {
				eg.AdvanceEpoch(uint64(start*100+i), now)
			}
		}(g)
	}
	wg.Wait()

	stats := eg.Stats()
	if stats.TotalSubmitted != 400 {
		t.Errorf("expected 400 submitted, got %d", stats.TotalSubmitted)
	}
}
