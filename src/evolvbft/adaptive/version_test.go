package adaptive

import "testing"

func sampleParams(validators, faults, instances int) ConfigParams {
	return ConfigParams{
		ValidatorCount: validators,
		FaultBound:     faults,
		InstanceCount:  instances,
	}
}

func TestNewVersionChain_SeedsCommittedGenesis(t *testing.T) {
	vc := NewVersionChain(sampleParams(7, 2, 1), 1, 0)
	if vc.Len() != 1 {
		t.Fatalf("expected 1 genesis version, got %d", vc.Len())
	}
	g := vc.Latest()
	if g.VersionID != 0 || g.ParentID != 0 {
		t.Fatalf("genesis should be v0 with parent 0, got v%d parent %d", g.VersionID, g.ParentID)
	}
	if g.Status != StatusCommitted {
		t.Fatalf("genesis must be COMMITTED, got %s", g.Status)
	}
	if !g.IsSafe(1) {
		t.Fatalf("genesis with phi=1 must be safe at delta_s=1")
	}
}

func TestAppendCommitted_LinksParent(t *testing.T) {
	vc := NewVersionChain(sampleParams(7, 2, 1), 1, 0)
	v1 := vc.AppendCommitted(sampleParams(10, 3, 1), 2, 5)
	if v1.VersionID != 1 || v1.ParentID != 0 {
		t.Fatalf("v1 should link to genesis, got v%d parent %d", v1.VersionID, v1.ParentID)
	}
	if v1.Status != StatusCommitted {
		t.Fatalf("appended version must be COMMITTED, got %s", v1.Status)
	}
	if vc.Latest().VersionID != 1 {
		t.Fatalf("Latest must follow the newest append")
	}
}

func TestSafeAncestor_SkipsUnsafeVersions(t *testing.T) {
	vc := NewVersionChain(sampleParams(7, 2, 1), 1, 0) // v0 safe
	vc.AppendCommitted(sampleParams(7, 2, 1), 1, 1)    // v1 safe
	vc.AppendCommitted(sampleParams(7, 2, 1), -1, 2)   // v2 UNSAFE (phi<delta_s)

	anc, ok := vc.SafeAncestor(1)
	if !ok {
		t.Fatalf("expected a safe ancestor to exist")
	}
	if anc.VersionID != 1 {
		t.Fatalf("nearest safe ancestor should be v1, got v%d", anc.VersionID)
	}
}

func TestSafeAncestor_FallsBackToGenesis(t *testing.T) {
	vc := NewVersionChain(sampleParams(7, 2, 1), 5, 0) // v0 strongly safe
	vc.AppendCommitted(sampleParams(7, 2, 1), 0, 1)    // v1 unsafe at delta_s=1
	anc, ok := vc.SafeAncestor(1)
	if !ok || anc.VersionID != 0 {
		t.Fatalf("should fall back to genesis v0, got v%d ok=%v", anc.VersionID, ok)
	}
}

func TestAppendRollback_RestoresAncestorParams(t *testing.T) {
	vc := NewVersionChain(sampleParams(7, 2, 1), 3, 0)
	safe := vc.AppendCommitted(sampleParams(7, 2, 1), 2, 1) // safe target
	vc.AppendCommitted(sampleParams(13, 4, 1), -1, 2)       // unsafe drift

	rb := vc.AppendRollback(safe, 9)
	if rb.Status != StatusRolledBack {
		t.Fatalf("rollback version must be ROLLEDBACK, got %s", rb.Status)
	}
	if rb.Params.ValidatorCount != safe.Params.ValidatorCount {
		t.Fatalf("rollback must restore safe params: got %d want %d",
			rb.Params.ValidatorCount, safe.Params.ValidatorCount)
	}
	if rb.PhiAtCommit != safe.PhiAtCommit {
		t.Fatalf("rollback must carry the safe phi: got %d want %d", rb.PhiAtCommit, safe.PhiAtCommit)
	}
	if vc.Latest().VersionID != rb.VersionID {
		t.Fatalf("rollback must become the effective config")
	}
}

func TestRollbackTriggered_AllConditions(t *testing.T) {
	cases := []struct {
		name string
		obs  ObservedSafety
		want bool
	}{
		{"healthy", ObservedSafety{Phi: 2, ViewStalled: false, ThroughputTPS: 100, ThroughputFloor: 50}, false},
		{"phi-breach", ObservedSafety{Phi: -1, ThroughputTPS: 100, ThroughputFloor: 50}, true},
		{"view-stall", ObservedSafety{Phi: 2, ViewStalled: true, ThroughputTPS: 100, ThroughputFloor: 50}, true},
		{"throughput-collapse", ObservedSafety{Phi: 2, ThroughputTPS: 10, ThroughputFloor: 50}, true},
		{"no-floor-no-trigger", ObservedSafety{Phi: 2, ThroughputTPS: 0, ThroughputFloor: 0}, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := c.obs.RollbackTriggered(); got != c.want {
				t.Fatalf("RollbackTriggered()=%v want %v", got, c.want)
			}
		})
	}
}

func TestEvaluateRollback_GuardsAndDecides(t *testing.T) {
	vc := NewVersionChain(sampleParams(7, 2, 1), 3, 0)
	vc.AppendCommitted(sampleParams(7, 2, 1), 2, 1)
	vc.AppendCommitted(sampleParams(13, 4, 1), -1, 2) // unsafe drift

	// No trigger -> no rollback.
	if _, ok := EvaluateRollback(vc, ObservedSafety{Phi: 2, ThroughputFloor: 0}, 1); ok {
		t.Fatalf("healthy observation must not trigger rollback")
	}
	// Trigger -> nearest safe ancestor.
	target, ok := EvaluateRollback(vc, ObservedSafety{Phi: -1}, 1)
	if !ok {
		t.Fatalf("phi breach must trigger rollback")
	}
	if !target.IsSafe(1) {
		t.Fatalf("rollback target must be safe, got phi=%d", target.PhiAtCommit)
	}
	// Nil chain -> no rollback, no panic.
	if _, ok := EvaluateRollback(nil, ObservedSafety{Phi: -1}, 1); ok {
		t.Fatalf("nil chain must not trigger rollback")
	}
}
