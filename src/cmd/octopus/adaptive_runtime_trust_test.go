package main

import (
	"testing"

	"octopus-bft/octopus/adaptive"
	"octopus-bft/octopus/consensus/hotstuff"
	"octopus-bft/octopus/trust"
	"octopus-bft/octopus/types"
)

func TestOctopusAdaptiveRuntimeObserveIncludesTrustSnapshots(t *testing.T) {
	engine := buildAdaptiveEngineForTest(t)
	engine.GetLeaderReputation().RecordSuccess(1)
	engine.GetLeaderReputation().RecordTimeout(1)
	engine.GetLeaderReputation().RecordTimeout(2)

	rt := octopusAdaptiveRuntime{
		nodeID:  0,
		engines: []*hotstuff.Engine{engine},
	}

	obs := rt.Observe()
	if len(obs.Agents) != 1 {
		t.Fatalf("expected one agent observation, got %d", len(obs.Agents))
	}
	if len(obs.TrustSnapshots) != 2 {
		t.Fatalf("expected two trust snapshots, got %d", len(obs.TrustSnapshots))
	}

	byNode := make(map[uint64]struct {
		SampleCount        uint64
		SuccessRate        float64
		FailureProbability float64
		ClaimBoundary      string
	}, len(obs.TrustSnapshots))
	for _, snapshot := range obs.TrustSnapshots {
		byNode[snapshot.NodeID] = struct {
			SampleCount        uint64
			SuccessRate        float64
			FailureProbability float64
			ClaimBoundary      string
		}{
			SampleCount:        snapshot.SampleCount,
			SuccessRate:        snapshot.SuccessRate,
			FailureProbability: snapshot.FailureProbability,
			ClaimBoundary:      snapshot.ClaimBoundary,
		}
	}
	if got := byNode[1]; got.SampleCount != 2 || got.SuccessRate != 0.5 || got.FailureProbability != 0.5 {
		t.Fatalf("unexpected node 1 trust snapshot: %+v", got)
	}
	if got := byNode[1].ClaimBoundary; got == "" || got == adaptive.AdminClaimBoundary {
		t.Fatalf("expected trust snapshot specific claim boundary, got %q", got)
	}
	if got := byNode[2]; got.SampleCount != 1 || got.SuccessRate != 0.0 || got.FailureProbability != 1.0 {
		t.Fatalf("unexpected node 2 trust snapshot: %+v", got)
	}
}

func TestOctopusAdaptiveRuntimeObserveTrustSnapshotsStayEmptyWithoutSignals(t *testing.T) {
	engine := buildAdaptiveEngineForTest(t)
	rt := octopusAdaptiveRuntime{
		nodeID:  0,
		engines: []*hotstuff.Engine{engine},
	}

	obs := rt.Observe()
	if len(obs.TrustSnapshots) != 0 {
		t.Fatalf("expected no trust snapshots, got %+v", obs.TrustSnapshots)
	}
}

func TestOctopusAdaptiveRuntimeObserveTrustSnapshotsTreatNilBlocksAsFailureSignals(t *testing.T) {
	engine := buildAdaptiveEngineForTest(t)
	engine.GetLeaderReputation().RecordSuccess(4)
	engine.GetLeaderReputation().RecordNilBlock(4)

	rt := octopusAdaptiveRuntime{
		nodeID:  0,
		engines: []*hotstuff.Engine{engine},
	}

	obs := rt.Observe()
	byNode := make(map[uint64]struct {
		SampleCount        uint64
		SuccessRate        float64
		FailureProbability float64
		ClaimBoundary      string
	}, len(obs.TrustSnapshots))
	for _, snapshot := range obs.TrustSnapshots {
		byNode[snapshot.NodeID] = struct {
			SampleCount        uint64
			SuccessRate        float64
			FailureProbability float64
			ClaimBoundary      string
		}{
			SampleCount:        snapshot.SampleCount,
			SuccessRate:        snapshot.SuccessRate,
			FailureProbability: snapshot.FailureProbability,
			ClaimBoundary:      snapshot.ClaimBoundary,
		}
	}
	if got := byNode[4].SampleCount; got != 2 {
		t.Fatalf("unexpected node 4 sample count: %d", got)
	}
	if got := byNode[4].SuccessRate; got != 0.5 {
		t.Fatalf("unexpected node 4 success rate: %f", got)
	}
	if got := byNode[4].FailureProbability; got != 0.5 {
		t.Fatalf("unexpected node 4 failure probability: %f", got)
	}
}

func TestOctopusAdaptiveRuntimeObserveTrustSnapshotsDoNotReplaceTimingFields(t *testing.T) {
	engine := buildAdaptiveEngineForTest(t)
	engine.GetLeaderReputation().RecordTimeout(3)

	rt := octopusAdaptiveRuntime{
		nodeID:  0,
		engines: []*hotstuff.Engine{engine},
	}

	obs := rt.Observe()
	if obs.Timestamp.IsZero() {
		t.Fatalf("expected timestamp to remain populated")
	}
	if len(obs.TrustSnapshots) != 1 {
		t.Fatalf("expected one trust snapshot, got %d", len(obs.TrustSnapshots))
	}
	if snapshot := obs.TrustSnapshots[0]; snapshot.SampleCount != 1 || snapshot.SuccessRate != 0.0 || snapshot.FailureProbability != 1.0 {
		t.Fatalf("unexpected trust snapshot: %+v", snapshot)
	}
}

func TestOctopusAdaptiveRuntimeObserveTrustSnapshotsAreSortedByNodeID(t *testing.T) {
	engine := buildAdaptiveEngineForTest(t)
	engine.GetLeaderReputation().RecordTimeout(9)
	engine.GetLeaderReputation().RecordTimeout(2)
	engine.GetLeaderReputation().RecordTimeout(5)

	rt := octopusAdaptiveRuntime{
		nodeID:  0,
		engines: []*hotstuff.Engine{engine},
	}

	obs := rt.Observe()
	if len(obs.TrustSnapshots) != 3 {
		t.Fatalf("expected three trust snapshots, got %d", len(obs.TrustSnapshots))
	}
	want := []uint64{2, 5, 9}
	for idx, nodeID := range want {
		if got := obs.TrustSnapshots[idx].NodeID; got != nodeID {
			t.Fatalf("unexpected trust snapshot order at %d: got %d want %d", idx, got, nodeID)
		}
	}
}

func TestEstimateFaults_NilEstimatorReturnsZero(t *testing.T) {
	rt := &octopusAdaptiveRuntime{trustEst: nil}
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		1: {ID: 1, Power: 1, IsActive: true},
		2: {ID: 2, Power: 1, IsActive: true},
	})
	if got := rt.estimateFaults(valSet); got != 0 {
		t.Fatalf("expected 0 with nil trustEst, got %d", got)
	}
}

func TestEstimateFaults_CountsHighProbabilityNodes(t *testing.T) {
	// 4 validators: nodes 1,2,3,4. Nodes 2,3 are Byzantine (all timeouts).
	est := trust.NewBayesianEstimator(trust.BayesianConfig{
		WindowSize: 4,
		MinSamples: 1,
		Weights: trust.ClassifierWeights{
			W: [5]float64{10.0, 5.0, 3.0, 0.01, 0.01},
			B: -2.0,
		},
	})
	// Node 1: all successes → low fault prob
	est.ObserveEpoch(1, trust.EpochEvent{Timeouts: 0, Equivocations: 0, LatencyMs: 50})
	est.ObserveEpoch(1, trust.EpochEvent{Timeouts: 0, Equivocations: 0, LatencyMs: 50})
	// Node 2: all timeouts → high fault prob
	est.ObserveEpoch(2, trust.EpochEvent{Timeouts: 1, Equivocations: 1, LatencyMs: 500})
	est.ObserveEpoch(2, trust.EpochEvent{Timeouts: 1, Equivocations: 1, LatencyMs: 500})
	// Node 3: all timeouts → high fault prob
	est.ObserveEpoch(3, trust.EpochEvent{Timeouts: 1, Equivocations: 1, LatencyMs: 500})
	est.ObserveEpoch(3, trust.EpochEvent{Timeouts: 1, Equivocations: 1, LatencyMs: 500})
	// Node 4: all successes → low fault prob
	est.ObserveEpoch(4, trust.EpochEvent{Timeouts: 0, Equivocations: 0, LatencyMs: 50})
	est.ObserveEpoch(4, trust.EpochEvent{Timeouts: 0, Equivocations: 0, LatencyMs: 50})

	rt := &octopusAdaptiveRuntime{trustEst: est}
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		1: {ID: 1, Power: 1, IsActive: true},
		2: {ID: 2, Power: 1, IsActive: true},
		3: {ID: 3, Power: 1, IsActive: true},
		4: {ID: 4, Power: 1, IsActive: true},
	})

	f := rt.estimateFaults(valSet)
	// max f for n=4 is (4-1)/3 = 1, so capped at 1 even though 2 nodes are faulty
	if f != 1 {
		t.Fatalf("expected FaultsEstimate=1 (capped at maxF), got %d", f)
	}
}

func TestEstimateFaults_RespectsMaxFCap(t *testing.T) {
	// 7 validators: nodes 1-7. Nodes 5,6,7 are Byzantine.
	// max f for n=7 is (7-1)/3 = 2
	est := trust.NewBayesianEstimator(trust.BayesianConfig{
		WindowSize: 4,
		MinSamples: 1,
		Weights: trust.ClassifierWeights{
			W: [5]float64{10.0, 5.0, 3.0, 0.01, 0.01},
			B: -2.0,
		},
	})
	for id := uint64(1); id <= 4; id++ {
		est.ObserveEpoch(id, trust.EpochEvent{Timeouts: 0, Equivocations: 0, LatencyMs: 50})
	}
	for id := uint64(5); id <= 7; id++ {
		est.ObserveEpoch(id, trust.EpochEvent{Timeouts: 1, Equivocations: 1, LatencyMs: 500})
	}

	rt := &octopusAdaptiveRuntime{trustEst: est}
	validators := make(map[uint64]*types.Validator)
	for id := uint64(1); id <= 7; id++ {
		validators[id] = &types.Validator{ID: id, Power: 1, IsActive: true}
	}
	valSet := types.NewValidatorSet(1, validators)

	f := rt.estimateFaults(valSet)
	// 3 faulty, but max f = (7-1)/3 = 2
	if f != 2 {
		t.Fatalf("expected FaultsEstimate=2 (capped), got %d", f)
	}
}

func TestEstimateFaults_ZeroWhenAllHealthy(t *testing.T) {
	est := trust.NewBayesianEstimator(trust.BayesianConfig{
		WindowSize: 4,
		MinSamples: 1,
		Weights: trust.ClassifierWeights{
			W: [5]float64{10.0, 5.0, 3.0, 0.01, 0.01},
			B: -2.0,
		},
	})
	for id := uint64(1); id <= 4; id++ {
		est.ObserveEpoch(id, trust.EpochEvent{Timeouts: 0, Equivocations: 0, LatencyMs: 50})
	}

	rt := &octopusAdaptiveRuntime{trustEst: est}
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		1: {ID: 1, Power: 1, IsActive: true},
		2: {ID: 2, Power: 1, IsActive: true},
		3: {ID: 3, Power: 1, IsActive: true},
		4: {ID: 4, Power: 1, IsActive: true},
	})

	if f := rt.estimateFaults(valSet); f != 0 {
		t.Fatalf("expected FaultsEstimate=0 for all-healthy, got %d", f)
	}
}

func TestEstimateFaults_IntegrationWithObserve(t *testing.T) {
	// Verify FaultsEstimate flows through Observe() into AgentObservation.
	engine := buildAdaptiveEngineForTest(t)
	// Feed timeouts for nodes 2,3 (which are in the validator set)
	engine.GetLeaderReputation().RecordTimeout(2)
	engine.GetLeaderReputation().RecordTimeout(3)

	est := trust.NewBayesianEstimator(trust.BayesianConfig{
		WindowSize: 4,
		MinSamples: 1,
		Weights: trust.ClassifierWeights{
			W: [5]float64{10.0, 5.0, 3.0, 0.01, 0.01},
			B: -2.0,
		},
	})

	rt := octopusAdaptiveRuntime{
		nodeID:   0,
		engines:  []*hotstuff.Engine{engine},
		trustEst: est,
	}

	obs := rt.Observe()
	if len(obs.Agents) != 1 {
		t.Fatalf("expected 1 agent, got %d", len(obs.Agents))
	}
	// The trust estimator gets fed via buildTrustSnapshotsWithBayesian during Observe,
	// so after first Observe, nodes 2 and 3 should have high fault prob.
	// Call Observe again to pick up the filled estimator state.
	obs = rt.Observe()
	// With n=4, max f = 1. Nodes 2,3 are faulty → capped to 1.
	if obs.Agents[0].FaultsEstimate != 1 {
		t.Fatalf("expected FaultsEstimate=1 after second Observe, got %d", obs.Agents[0].FaultsEstimate)
	}
}

func TestTrustSnapshots_Eq5FeaturesPopulated_FallbackPath(t *testing.T) {
	// Fallback path (no trustEst): features derived from LeaderReputationStats.
	engine := buildAdaptiveEngineForTest(t)
	engine.GetLeaderReputation().RecordSuccess(1)
	engine.GetLeaderReputation().RecordTimeout(1)
	engine.GetLeaderReputation().RecordTimeout(2)

	rt := octopusAdaptiveRuntime{
		nodeID:  0,
		engines: []*hotstuff.Engine{engine},
		// trustEst deliberately nil → fallback path
	}

	obs := rt.Observe()
	if len(obs.TrustSnapshots) < 2 {
		t.Fatalf("expected ≥2 trust snapshots, got %d", len(obs.TrustSnapshots))
	}

	// Find node 1: 1 success + 1 timeout → total=2, TimeoutRate=0.5
	var node1 *adaptive.TrustSnapshot
	for i := range obs.TrustSnapshots {
		if obs.TrustSnapshots[i].NodeID == 1 {
			node1 = &obs.TrustSnapshots[i]
			break
		}
	}
	if node1 == nil {
		t.Fatal("node 1 snapshot missing")
	}
	if node1.TimeoutRate != 0.5 {
		t.Fatalf("expected TimeoutRate=0.5, got %f", node1.TimeoutRate)
	}

	// Node 2: 1 timeout → total=1, TimeoutRate=1.0
	var node2 *adaptive.TrustSnapshot
	for i := range obs.TrustSnapshots {
		if obs.TrustSnapshots[i].NodeID == 2 {
			node2 = &obs.TrustSnapshots[i]
			break
		}
	}
	if node2 == nil {
		t.Fatal("node 2 snapshot missing")
	}
	if node2.TimeoutRate != 1.0 {
		t.Fatalf("expected TimeoutRate=1.0 for node 2, got %f", node2.TimeoutRate)
	}
}

func TestTrustSnapshots_Eq5FeaturesPopulated_BayesianPath(t *testing.T) {
	// Bayesian path: features from trust.BayesianEstimator.Features().
	engine := buildAdaptiveEngineForTest(t)
	engine.GetLeaderReputation().RecordSuccess(1)
	engine.GetLeaderReputation().RecordTimeout(1)
	engine.GetLeaderReputation().RecordTimeout(2)

	est := trust.NewBayesianEstimator(trust.BayesianConfig{
		WindowSize:   8,
		MinSamples:   1,
		MaxLatencyMs: 1000,
		Weights: trust.ClassifierWeights{
			W: [5]float64{10.0, 5.0, 3.0, 0.01, 0.01},
			B: -2.0,
		},
	})

	rt := octopusAdaptiveRuntime{
		nodeID:   0,
		engines:  []*hotstuff.Engine{engine},
		trustEst: est,
	}

	obs := rt.Observe()
	if len(obs.TrustSnapshots) < 2 {
		t.Fatalf("expected ≥2 trust snapshots, got %d", len(obs.TrustSnapshots))
	}

	// Node 2: 1 timeout epoch → TimeoutRate=1.0, EquivocationRate=0.0
	var node2 *adaptive.TrustSnapshot
	for i := range obs.TrustSnapshots {
		if obs.TrustSnapshots[i].NodeID == 2 {
			node2 = &obs.TrustSnapshots[i]
			break
		}
	}
	if node2 == nil {
		t.Fatal("node 2 snapshot missing in Bayesian path")
	}
	if node2.TimeoutRate != 1.0 {
		t.Fatalf("expected TimeoutRate=1.0 for node 2 (Bayesian), got %f", node2.TimeoutRate)
	}
	// EquivocationRate should be 0 since RecordTimeout doesn't set equivocations
	if node2.EquivocationRate != 0.0 {
		t.Fatalf("expected EquivocationRate=0.0, got %f", node2.EquivocationRate)
	}
}

func TestEstimateFaults_FusesWithAggregator(t *testing.T) {
	// Verify that cross-instance trust aggregator elevates fault estimate.
	// Setup: node 5 is locally healthy (prob=0.1), but known-bad globally (prob=0.9).
	// FusedFaultProb uses max(local, global) → 0.9 > threshold → counted as faulty.
	est := trust.NewBayesianEstimator(trust.BayesianConfig{
		WindowSize: 4,
		MinSamples: 1,
		Weights: trust.ClassifierWeights{
			W: [5]float64{10.0, 5.0, 3.0, 0.01, 0.01},
			B: -2.0,
		},
	})
	// Feed healthy observations for all nodes
	for id := uint64(1); id <= 4; id++ {
		est.ObserveEpoch(id, trust.EpochEvent{Timeouts: 0, Equivocations: 0, LatencyMs: 50})
	}

	agg := trust.NewAggregator()
	// Simulate cross-instance report: node 3 was Byzantine in another instance
	agg.Ingest(trust.TrustReport{
		InstanceID: 99,
		Epoch:      1,
		FaultProbs: map[uint64]float64{3: 0.95},
	})

	rt := &octopusAdaptiveRuntime{trustEst: est, trustAgg: agg}
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		1: {ID: 1, Power: 1, IsActive: true},
		2: {ID: 2, Power: 1, IsActive: true},
		3: {ID: 3, Power: 1, IsActive: true},
		4: {ID: 4, Power: 1, IsActive: true},
	})

	f := rt.estimateFaults(valSet)
	// Node 3 is locally healthy but globally bad → fused prob > 0.5 → f=1
	if f != 1 {
		t.Fatalf("expected FaultsEstimate=1 (aggregator elevates node 3), got %d", f)
	}
}

func TestEstimateFaults_AggregatorNilFallsBackToLocal(t *testing.T) {
	// Without aggregator, behavior matches local-only estimation.
	est := trust.NewBayesianEstimator(trust.BayesianConfig{
		WindowSize: 4,
		MinSamples: 1,
		Weights: trust.ClassifierWeights{
			W: [5]float64{10.0, 5.0, 3.0, 0.01, 0.01},
			B: -2.0,
		},
	})
	for id := uint64(1); id <= 4; id++ {
		est.ObserveEpoch(id, trust.EpochEvent{Timeouts: 0, Equivocations: 0, LatencyMs: 50})
	}

	rt := &octopusAdaptiveRuntime{trustEst: est, trustAgg: nil}
	valSet := types.NewValidatorSet(1, map[uint64]*types.Validator{
		1: {ID: 1, Power: 1, IsActive: true},
		2: {ID: 2, Power: 1, IsActive: true},
		3: {ID: 3, Power: 1, IsActive: true},
		4: {ID: 4, Power: 1, IsActive: true},
	})

	f := rt.estimateFaults(valSet)
	if f != 0 {
		t.Fatalf("expected FaultsEstimate=0 (all locally healthy, no aggregator), got %d", f)
	}
}
