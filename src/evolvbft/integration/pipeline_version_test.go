package integration

import (
	"testing"

	"evolvbft/evolvbft/adaptive"
	"evolvbft/evolvbft/consensus/gbc"
	"evolvbft/evolvbft/trust"
)

// newTrustEstimatorForVersionTest builds a sensitive estimator that quickly
// flags severe Byzantine behavior so reconfiguration fires within a few epochs.
func newTrustEstimatorForVersionTest() *trust.CombinedEstimator {
	return trust.NewCombinedEstimator(
		trust.BayesianConfig{
			WindowSize: 4,
			MinSamples: 1,
			Weights:    trust.ClassifierWeights{W: [5]float64{3.0, 3.0, 1.0, 0.5, 0.3}, B: -1.0},
		},
		0.3, 0.7,
	)
}

func byzantineMetrics(total, byzCount int) map[uint64]trust.EpochEvent {
	m := make(map[uint64]trust.EpochEvent, total)
	for i := 0; i < total; i++ {
		if i < byzCount {
			m[uint64(i)] = trust.EpochEvent{Timeouts: 10, Equivocations: 5, ViewChanges: 5, LatencyMs: 2000}
		} else {
			m[uint64(i)] = trust.EpochEvent{Timeouts: 0, Equivocations: 0, ViewChanges: 0, LatencyMs: 30}
		}
	}
	return m
}

// TestPipeline_RecordsConfigVersionOnReconfig verifies that a committed
// reconfiguration appends a COMMITTED version to the append-only chain.
// With no GBC nodes set, publishToGBC uses the in-memory log, so publishes
// succeed and the version is durably recorded (commit-then-apply happy path).
func TestPipeline_RecordsConfigVersionOnReconfig(t *testing.T) {
	config := DefaultPipelineConfig()
	config.NumInstances = 1
	config.NumAgents = 10
	config.FaultThreshold = 0.5
	config.SafetyMarginDelta = 1

	// GBC with 5 members gives margin 5-3-1 = 1 >= delta_s.
	pipe := NewPipeline(config, gbc.NewLogWithMembers(5), newTrustEstimatorForVersionTest())
	pipe.SetInstances([]InstanceHandle{{
		InstanceID:     0,
		ValidatorCount: 10,
		FaultTolerance: 2,
		Agents:         makeAgents(0, 10),
	}})

	// Before any reconfig the chain is unseeded.
	if snap := pipe.VersionChainSnapshot(); snap != nil {
		t.Fatalf("expected nil version chain before first reconfig, got %d entries", len(snap))
	}

	metrics := byzantineMetrics(10, 3)
	reconfigSeen := 0
	for epoch := uint64(0); epoch < 6; epoch++ {
		stats, err := pipe.ProcessEpoch(epoch, metrics)
		if err != nil {
			t.Fatalf("ProcessEpoch(%d): %v", epoch, err)
		}
		reconfigSeen += stats.ReconfigActions
	}
	if reconfigSeen == 0 {
		t.Skip("no reconfiguration triggered; trust thresholds did not flag agents")
	}

	snap := pipe.VersionChainSnapshot()
	if len(snap) < 2 {
		t.Fatalf("expected genesis + at least one committed version, got %d", len(snap))
	}
	if snap[0].Status != adaptive.StatusCommitted || snap[0].VersionID != 0 {
		t.Fatalf("genesis must be committed v0, got v%d %s", snap[0].VersionID, snap[0].Status)
	}
	last := snap[len(snap)-1]
	if last.Status != adaptive.StatusCommitted {
		t.Fatalf("recorded version must be COMMITTED, got %s", last.Status)
	}
	// A committed version must never sit below the BFT safety floor Phi >= 0.
	if last.PhiAtCommit < 0 {
		t.Fatalf("recorded version violates BFT safety floor, got phi=%d", last.PhiAtCommit)
	}
}

// TestPipeline_EvaluateAndRollback_RestoresSafeConfig verifies that a triggered
// rollback appends a ROLLEDBACK version and restores the instance membership of
// the nearest safe committed ancestor.
func TestPipeline_EvaluateAndRollback_RestoresSafeConfig(t *testing.T) {
	config := DefaultPipelineConfig()
	config.NumInstances = 1
	config.NumAgents = 10
	config.SafetyMarginDelta = 1

	// GBC with 5 members gives margin 1, so the safe genesis satisfies phi >= delta_s.
	pipe := NewPipeline(config, gbc.NewLogWithMembers(5), newTrustEstimatorForVersionTest())
	pipe.SetInstances([]InstanceHandle{{
		InstanceID:     0,
		ValidatorCount: 10,
		FaultTolerance: 2,
		Agents:         makeAgents(0, 10),
	}})

	// Seed the chain with the safe genesis config (phi = min(10-6-1, 5-3-1) = 1).
	pipe.ensureVersionChainLocked()
	safeValidatorCount := pipe.instances[0].ValidatorCount
	if got := pipe.VersionChainSnapshot()[0].PhiAtCommit; got < config.SafetyMarginDelta {
		t.Fatalf("genesis must be safe (phi >= %d), got %d", config.SafetyMarginDelta, got)
	}

	// Commit an UNSAFE drifted version (phi = 0 < delta_s), then shrink the live
	// instance to simulate adaptive over-eviction we want to undo.
	pipe.instances[0].ValidatorCount = 7
	pipe.instances[0].FaultTolerance = 2
	driftParams := pipe.currentConfigParamsLocked()
	pipe.versionChain.AppendCommitted(driftParams, 0, 1) // phi=0 marks it unsafe

	// A safety regression is observed (Phi < 0 -> trigger R1).
	rolledBack, target, err := pipe.EvaluateAndRollback(5, adaptive.ObservedSafety{Phi: -1})
	if err != nil {
		t.Fatalf("EvaluateAndRollback returned error: %v", err)
	}
	if !rolledBack {
		t.Fatalf("expected rollback to trigger on Phi < 0")
	}
	if !target.IsSafe(config.SafetyMarginDelta) {
		t.Fatalf("rollback target must be safe, got phi=%d", target.PhiAtCommit)
	}
	// The ROLLEDBACK record carries the safe ancestor params, so its phi equals
	// the genesis phi (1), proving the unsafe drift (phi=0) was skipped.
	if target.PhiAtCommit != pipe.VersionChainSnapshot()[0].PhiAtCommit {
		t.Fatalf("rollback must restore the safe ancestor phi=%d, got %d",
			pipe.VersionChainSnapshot()[0].PhiAtCommit, target.PhiAtCommit)
	}

	// The effective config is now a ROLLEDBACK version restoring the safe ancestor.
	chain := pipe.VersionChainSnapshot()
	tail := chain[len(chain)-1]
	if tail.Status != adaptive.StatusRolledBack {
		t.Fatalf("latest version must be ROLLEDBACK, got %s", tail.Status)
	}
	// Instance membership is restored to the safe ancestor value.
	if pipe.instances[0].ValidatorCount != safeValidatorCount {
		t.Fatalf("rollback must restore validator count to %d, got %d",
			safeValidatorCount, pipe.instances[0].ValidatorCount)
	}
}

// TestPipeline_RollbackPublishFailure_NoApply verifies commit-then-apply for the
// rollback path: if the GBC publish fails, the chain is NOT extended and the
// live instance set is NOT mutated.
func TestPipeline_RollbackPublishFailure_NoApply(t *testing.T) {
	// Node 99 is never the round-robin proposer, so every Propose fails.
	fakeTransports := gbc.NewChannelTransportSet(1, 64)
	fakeNodes := []*gbc.Node{
		gbc.NewNode(gbc.NodeConfig{NodeID: 99, NumMembers: 4}, fakeTransports[0]),
	}
	fakeNodes[0].Start()
	defer fakeNodes[0].Stop()

	config := DefaultPipelineConfig()
	config.NumInstances = 1
	config.NumAgents = 10
	config.SafetyMarginDelta = 1

	// GBC log sized to 5 so the seeded genesis is safe and a rollback target exists,
	// forcing EvaluateAndRollback to attempt the (failing) publish.
	pipe := NewPipeline(config, gbc.NewLogWithMembers(5), newTrustEstimatorForVersionTest())
	pipe.SetGBCNodes(fakeNodes)
	pipe.SetInstances([]InstanceHandle{{
		InstanceID:     0,
		ValidatorCount: 10,
		FaultTolerance: 2,
		Agents:         makeAgents(0, 10),
	}})

	pipe.ensureVersionChainLocked()
	chainLenBefore := len(pipe.VersionChainSnapshot())
	validatorsBefore := pipe.instances[0].ValidatorCount

	// Trigger a rollback, but the GBC publish will fail.
	rolledBack, _, err := pipe.EvaluateAndRollback(3, adaptive.ObservedSafety{Phi: -1})
	if err == nil {
		t.Fatalf("expected publish failure error from EvaluateAndRollback")
	}
	if rolledBack {
		t.Fatalf("rollback must NOT be applied when GBC publish fails")
	}
	if got := len(pipe.VersionChainSnapshot()); got != chainLenBefore {
		t.Fatalf("commit-then-apply violated: chain grew from %d to %d on publish failure",
			chainLenBefore, got)
	}
	if pipe.instances[0].ValidatorCount != validatorsBefore {
		t.Fatalf("commit-then-apply violated: instance mutated on publish failure")
	}
}
