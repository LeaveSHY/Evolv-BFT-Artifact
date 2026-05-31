package pacemaker

import (
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// GetCurrentView
// ---------------------------------------------------------------------------

func TestGetCurrentView_InitialIsOne(t *testing.T) {
	pm := NewPacemaker([]uint64{1, 2, 3}, 1000)
	if v := pm.GetCurrentView(); v != 1 {
		t.Errorf("initial view should be 1, got %d", v)
	}
}

// ---------------------------------------------------------------------------
// AdvanceView
// ---------------------------------------------------------------------------

func TestAdvanceView_MovesForward(t *testing.T) {
	pm := NewPacemaker([]uint64{1, 2, 3}, 1000)
	ok := pm.AdvanceView(1) // qcView=1 >= currentView=1 → advance to 2
	if !ok {
		t.Error("AdvanceView should return true when qcView >= currentView")
	}
	if v := pm.GetCurrentView(); v != 2 {
		t.Errorf("expected view 2 after AdvanceView(1), got %d", v)
	}
}

func TestAdvanceView_SkipsViews(t *testing.T) {
	pm := NewPacemaker([]uint64{1, 2, 3}, 1000)
	ok := pm.AdvanceView(5)
	if !ok {
		t.Error("AdvanceView should return true for large qcView")
	}
	if v := pm.GetCurrentView(); v != 6 {
		t.Errorf("expected view 6 after AdvanceView(5), got %d", v)
	}
}

func TestAdvanceView_RejectsOldQC(t *testing.T) {
	pm := NewPacemaker([]uint64{1, 2, 3}, 1000)
	pm.AdvanceView(5) // now at view 6

	ok := pm.AdvanceView(3) // qcView=3 < currentView=6
	if ok {
		t.Error("AdvanceView should return false when qcView < currentView")
	}
	if v := pm.GetCurrentView(); v != 6 {
		t.Errorf("view should remain 6, got %d", v)
	}
}

func TestAdvanceView_EqualQCAdvances(t *testing.T) {
	pm := NewPacemaker([]uint64{1, 2, 3}, 1000)
	pm.AdvanceView(1) // now at view 2

	ok := pm.AdvanceView(2) // qcView=2 >= currentView=2
	if !ok {
		t.Error("AdvanceView should return true when qcView == currentView")
	}
	if v := pm.GetCurrentView(); v != 3 {
		t.Errorf("expected view 3, got %d", v)
	}
}

func TestAdvanceView_Sequential(t *testing.T) {
	pm := NewPacemaker([]uint64{1, 2}, 1000)
	for i := uint64(1); i <= 10; i++ {
		pm.AdvanceView(i)
	}
	if v := pm.GetCurrentView(); v != 11 {
		t.Errorf("expected view 11 after 10 sequential advances, got %d", v)
	}
}

// ---------------------------------------------------------------------------
// TimeoutChan + Start
// ---------------------------------------------------------------------------

func TestTimeoutChan_FiresAfterTimeout(t *testing.T) {
	pm := NewPacemaker([]uint64{1, 2, 3}, 50) // 50ms timeout
	pm.Start()

	select {
	case v := <-pm.TimeoutChan():
		if v != 1 {
			t.Errorf("timeout should report view 1, got %d", v)
		}
	case <-time.After(500 * time.Millisecond):
		t.Error("timeout should have fired within 500ms")
	}
}

func TestAdvanceView_ResetsTimer(t *testing.T) {
	pm := NewPacemaker([]uint64{1, 2, 3}, 80) // 80ms timeout
	pm.Start()

	// Advance before timeout fires → old timer for view 1 should be cancelled
	time.Sleep(30 * time.Millisecond)
	pm.AdvanceView(1) // now at view 2, timer reset

	// The timeout we receive should be for view 2, not view 1
	select {
	case v := <-pm.TimeoutChan():
		if v != 2 {
			t.Errorf("timeout should be for view 2 after advance, got view %d", v)
		}
	case <-time.After(500 * time.Millisecond):
		t.Error("timeout should have fired for view 2")
	}
}

// ---------------------------------------------------------------------------
// SetTimeout
// ---------------------------------------------------------------------------

func TestSetTimeout_ChangesDelay(t *testing.T) {
	pm := NewPacemaker([]uint64{1, 2, 3}, 5000) // initially 5s
	pm.SetTimeout(50 * time.Millisecond)        // change to 50ms
	pm.Start()

	select {
	case <-pm.TimeoutChan():
		// good — fired quickly
	case <-time.After(500 * time.Millisecond):
		t.Error("timeout should fire at 50ms, not 5s")
	}
}

func TestSetTimeout_IgnoresZeroOrNegative(t *testing.T) {
	pm := NewPacemaker([]uint64{1, 2, 3}, 100)
	original := pm.viewTimeout

	pm.SetTimeout(0)
	if pm.viewTimeout != original {
		t.Error("SetTimeout(0) should not change timeout")
	}

	pm.SetTimeout(-1 * time.Millisecond)
	if pm.viewTimeout != original {
		t.Error("SetTimeout(negative) should not change timeout")
	}
}

// ---------------------------------------------------------------------------
// UpdateValidators
// ---------------------------------------------------------------------------

func TestUpdateValidators_ChangesLeaderRotation(t *testing.T) {
	pm := NewPacemaker([]uint64{1, 2, 3}, 1000)

	// With validators [1,2,3], view 1 lane 0 → validators[(1-1+0)%3] = validators[0] = 1
	l1 := pm.GetLeader(1)
	if l1 != 1 {
		t.Errorf("expected leader 1 for view 1 with [1,2,3], got %d", l1)
	}

	pm.UpdateValidators([]uint64{10, 20, 30})
	l2 := pm.GetLeader(1)
	if l2 != 10 {
		t.Errorf("expected leader 10 for view 1 with [10,20,30], got %d", l2)
	}
}

func TestUpdateValidators_EmptyList(t *testing.T) {
	pm := NewPacemaker([]uint64{1, 2, 3}, 1000)
	pm.UpdateValidators([]uint64{})
	if l := pm.GetLeader(1); l != 0 {
		t.Errorf("empty validators should return leader 0, got %d", l)
	}
}

// ---------------------------------------------------------------------------
// GetLeader edge cases
// ---------------------------------------------------------------------------

func TestGetLeader_ViewZeroTreatedAsOne(t *testing.T) {
	pm := NewPacemaker([]uint64{10, 20, 30}, 1000)
	l0 := pm.GetLeader(0)
	l1 := pm.GetLeader(1)
	if l0 != l1 {
		t.Errorf("GetLeader(0) should equal GetLeader(1), got %d vs %d", l0, l1)
	}
}

func TestGetLeader_EmptyValidators(t *testing.T) {
	pm := NewPacemaker([]uint64{}, 1000)
	if l := pm.GetLeader(1); l != 0 {
		t.Errorf("empty validators should return 0, got %d", l)
	}
}

func TestSetLeaderSelector_NilRevertsToRoundRobin(t *testing.T) {
	validators := []uint64{10, 20, 30}
	pm := NewPacemaker(validators, 1000)

	// Set custom selector
	pm.SetLeaderSelector(func(view uint64) uint64 { return 99 })
	if l := pm.GetLeader(1); l != 99 {
		t.Errorf("custom selector should return 99, got %d", l)
	}

	// Revert to round-robin
	pm.SetLeaderSelector(nil)
	l := pm.GetLeader(1)
	expected := validators[0] // (1-1+0) % 3 = 0 → validators[0] = 10
	if l != expected {
		t.Errorf("after nil selector, expected round-robin leader %d, got %d", expected, l)
	}
}

// ---------------------------------------------------------------------------
// Concurrency
// ---------------------------------------------------------------------------

func TestPacemaker_ConcurrentAccess(t *testing.T) {
	pm := NewPacemaker([]uint64{1, 2, 3, 4}, 1000)

	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := uint64(1); i <= 50; i++ {
			pm.AdvanceView(i)
		}
	}()

	// Concurrent reads while advancing
	for i := 0; i < 100; i++ {
		pm.GetCurrentView()
		pm.GetLeader(uint64(i + 1))
	}

	<-done

	v := pm.GetCurrentView()
	if v != 51 {
		t.Errorf("expected final view 51, got %d", v)
	}
}
