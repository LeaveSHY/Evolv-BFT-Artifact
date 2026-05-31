package integration

import (
	"testing"
	"time"

	"evolvbft/evolvbft/consensus/gbc"
	"evolvbft/evolvbft/trust"
)

func TestPipeline_BasicEpoch(t *testing.T) {
	// Setup GBC log
	gbcLog := gbc.NewLogWithMembers(4)

	// Setup combined trust estimator
	trustEst := trust.NewCombinedEstimator(
		trust.BayesianConfig{
			WindowSize: 8,
			MinSamples: 1,
			Weights: trust.ClassifierWeights{
				W: [5]float64{2.0, 2.0, 1.0, 0.5, 0.3},
				B: -1.5,
			},
		},
		0.2, // EWMA alpha
		0.7, // gamma: 70% bayesian, 30% EWMA
	)

	// Create pipeline
	config := DefaultPipelineConfig()
	config.NumInstances = 4
	config.NumAgents = 100
	config.FaultThreshold = 0.7
	pipe := NewPipeline(config, gbcLog, trustEst)

	// Setup instances
	instances := make([]InstanceHandle, 4)
	for i := 0; i < 4; i++ {
		agents := make([]uint64, 25)
		for j := 0; j < 25; j++ {
			agents[j] = uint64(i*25 + j)
		}
		instances[i] = InstanceHandle{
			InstanceID:     uint64(i),
			ValidatorCount: 25,
			FaultTolerance: 8,
			Agents:         agents,
		}
	}
	pipe.SetInstances(instances)

	// Track callbacks
	var reconfigCount int
	pipe.OnReconfigDecision(func(instanceID uint64, evict []uint64, admit []uint64) {
		reconfigCount += len(evict)
	})

	var trustUpdateCount int
	pipe.OnTrustUpdate(func(epoch uint64, scores map[uint64]float64) {
		trustUpdateCount++
	})

	// Epoch 1: All honest behavior — no reconfigs expected
	metrics := make(map[uint64]trust.EpochEvent)
	for i := uint64(0); i < 100; i++ {
		metrics[i] = trust.EpochEvent{
			Timeouts:      0,
			Equivocations: 0,
			ViewChanges:   0,
			LatencyMs:     50,
		}
	}
	stats, err := pipe.ProcessEpoch(0, metrics)
	if err != nil {
		t.Fatalf("ProcessEpoch(0): %v", err)
	}
	if stats.ReconfigActions != 0 {
		t.Fatalf("epoch 0: expected 0 reconfigs, got %d", stats.ReconfigActions)
	}
	if trustUpdateCount != 1 {
		t.Fatalf("expected 1 trust update callback, got %d", trustUpdateCount)
	}

	// Epochs 2-5: Inject Byzantine behavior for agents 10-15
	for epoch := uint64(1); epoch <= 4; epoch++ {
		for i := uint64(0); i < 100; i++ {
			if i >= 10 && i < 16 {
				metrics[i] = trust.EpochEvent{
					Timeouts:      3,
					Equivocations: 2,
					ViewChanges:   1,
					LatencyMs:     500,
				}
			} else {
				metrics[i] = trust.EpochEvent{
					Timeouts:      0,
					Equivocations: 0,
					ViewChanges:   0,
					LatencyMs:     50,
				}
			}
		}
		_, err := pipe.ProcessEpoch(epoch, metrics)
		if err != nil {
			t.Fatalf("ProcessEpoch(%d): %v", epoch, err)
		}
	}

	// After several epochs of Byzantine behavior, expect some reconfigs
	summary := pipe.Summary()
	if summary.EpochsProcessed != 5 {
		t.Fatalf("expected 5 epochs, got %d", summary.EpochsProcessed)
	}
	if summary.TotalTrustUpdates < 100 {
		t.Fatalf("expected many trust updates, got %d", summary.TotalTrustUpdates)
	}
	t.Logf("Summary: %s", summary)
	t.Logf("Reconfigs via callback: %d", reconfigCount)
}

func TestPipeline_SafetyMask(t *testing.T) {
	gbcLog := gbc.NewLogWithMembers(4)
	trustEst := trust.NewCombinedEstimator(
		trust.BayesianConfig{
			WindowSize: 2,
			MinSamples: 1,
			Weights: trust.ClassifierWeights{
				W: [5]float64{5.0, 5.0, 3.0, 1.0, 0.5},
				B: -2.0,
			},
		},
		0.5, 0.5,
	)

	config := DefaultPipelineConfig()
	config.FaultThreshold = 0.6
	pipe := NewPipeline(config, gbcLog, trustEst)

	// Create a small instance where evicting would violate safety
	pipe.SetInstances([]InstanceHandle{{
		InstanceID:     0,
		ValidatorCount: 4, // minimum safe size
		FaultTolerance: 1,
		Agents:         []uint64{0, 1, 2, 3},
	}})

	// Make ALL agents look Byzantine — safety filter should prevent evictions
	metrics := map[uint64]trust.EpochEvent{
		0: {Timeouts: 5, Equivocations: 3, LatencyMs: 1000},
		1: {Timeouts: 5, Equivocations: 3, LatencyMs: 1000},
		2: {Timeouts: 5, Equivocations: 3, LatencyMs: 1000},
		3: {Timeouts: 5, Equivocations: 3, LatencyMs: 1000},
	}

	// Run enough epochs for estimator to build confidence
	for epoch := uint64(0); epoch < 3; epoch++ {
		stats, err := pipe.ProcessEpoch(epoch, metrics)
		if err != nil {
			t.Fatalf("epoch %d: %v", epoch, err)
		}
		// Should not evict below safety threshold
		if stats.ReconfigActions > 0 && pipe.instances[0].ValidatorCount < 4 {
			t.Fatalf("safety violation: evicted below minimum 4 validators")
		}
	}

	// Verify safety was preserved
	if pipe.instances[0].ValidatorCount < 4 {
		t.Fatalf("safety violated: validator count %d < 4", pipe.instances[0].ValidatorCount)
	}
}

func TestPipeline_GBCEntries(t *testing.T) {
	gbcLog := gbc.NewLogWithMembers(4)
	trustEst := trust.NewCombinedEstimator(
		trust.BayesianConfig{
			WindowSize: 4,
			MinSamples: 1,
			Weights: trust.ClassifierWeights{
				W: [5]float64{2.0, 2.0, 1.0, 0.5, 0.3},
				B: -1.0,
			},
		},
		0.2, 0.7,
	)

	pipe := NewPipeline(DefaultPipelineConfig(), gbcLog, trustEst)
	pipe.SetInstances([]InstanceHandle{{
		InstanceID:     0,
		ValidatorCount: 25,
		FaultTolerance: 8,
		Agents:         makeAgents(0, 25),
	}})

	metrics := make(map[uint64]trust.EpochEvent, 25)
	for i := uint64(0); i < 25; i++ {
		metrics[i] = trust.EpochEvent{LatencyMs: 50}
	}

	pipe.ProcessEpoch(0, metrics)

	// Should have at least one EntryPolicyUpdate in the GBC log
	entry, ok := gbcLog.LatestByType(gbc.EntryPolicyUpdate)
	if !ok {
		t.Fatal("expected EntryPolicyUpdate in GBC log")
	}
	if entry.Height == 0 {
		t.Fatal("entry height should be > 0")
	}
}

func makeAgents(start, count int) []uint64 {
	agents := make([]uint64, count)
	for i := 0; i < count; i++ {
		agents[i] = uint64(start + i)
	}
	return agents
}

func TestPipeline_GBCNodeProtocolPath(t *testing.T) {
	// Verify that when GBC nodes are configured, the pipeline routes
	// membership entries through the distributed Propose-Attest-Commit
	// protocol (§III-C) instead of direct local log writes.
	numNodes := 4
	transports := gbc.NewChannelTransportSet(numNodes, 256)

	nodes := make([]*gbc.Node, numNodes)
	var committed []gbc.Entry
	for i := 0; i < numNodes; i++ {
		nodes[i] = gbc.NewNode(gbc.NodeConfig{
			NodeID:     uint64(i),
			NumMembers: numNodes,
		}, transports[i])
		nodes[i].OnCommit(func(e gbc.Entry) {
			committed = append(committed, e)
		})
		nodes[i].Start()
	}
	defer func() {
		for _, n := range nodes {
			n.Stop()
		}
	}()

	// Build pipeline with GBC nodes (no direct log fallback).
	trustEst := trust.NewCombinedEstimator(
		trust.BayesianConfig{
			WindowSize: 4,
			MinSamples: 1,
			Weights:    trust.ClassifierWeights{W: [5]float64{3.0, 3.0, 1.0, 0.5, 0.3}, B: -1.0},
		},
		0.3, 0.7,
	)

	config := DefaultPipelineConfig()
	config.NumInstances = 1
	config.NumAgents = 7
	config.FaultThreshold = 0.5
	config.SafetyMarginDelta = 0

	// Use nodes[0]'s log as the pipeline's gbcLog (unused in node mode, but needed for init).
	pipe := NewPipeline(config, gbc.NewLogWithMembers(numNodes), trustEst)
	pipe.SetGBCNodes(nodes)
	pipe.SetInstances([]InstanceHandle{{
		InstanceID:     0,
		ValidatorCount: 7,
		FaultTolerance: 2,
		Agents:         makeAgents(0, 7),
	}})

	// Inject extremely Byzantine behavior for agents 0-2 to force eviction
	metrics := make(map[uint64]trust.EpochEvent, 7)
	for i := uint64(0); i < 7; i++ {
		if i < 3 {
			metrics[i] = trust.EpochEvent{Timeouts: 10, Equivocations: 5, ViewChanges: 5, LatencyMs: 2000}
		} else {
			metrics[i] = trust.EpochEvent{Timeouts: 0, Equivocations: 0, ViewChanges: 0, LatencyMs: 30}
		}
	}

	// Run several epochs to build trust confidence then trigger eviction
	for epoch := uint64(0); epoch < 5; epoch++ {
		_, err := pipe.ProcessEpoch(epoch, metrics)
		if err != nil {
			t.Fatalf("epoch %d: %v", epoch, err)
		}
	}

	// The pipeline should have attempted at least one trust update via the GBC node protocol.
	// committed collects only entries that passed Propose-Attest-Commit quorum (2f+1=3).
	if len(committed) == 0 {
		t.Fatal("expected at least one GBC entry committed through distributed Node protocol")
	}

	// Verify entry types: should include EntryPolicyUpdate (trust updates go every epoch)
	var hasTrustEntry bool
	for _, e := range committed {
		if e.Type == gbc.EntryPolicyUpdate {
			hasTrustEntry = true
			break
		}
	}
	if !hasTrustEntry {
		t.Fatal("expected EntryPolicyUpdate committed through GBC Node protocol")
	}
	t.Logf("GBC Node protocol committed %d entries (trust + membership)", len(committed))
}

func TestPipeline_WaitsForGBCQuorumBeforeApplyingReconfig(t *testing.T) {
	// A proposer that cannot collect quorum must not cause local membership
	// mutation. This verifies the positive side of commit-then-apply: success
	// means quorum commit, not just local proposal acceptance.
	transports := gbc.NewChannelTransportSet(4, 64)
	proposer := gbc.NewNode(gbc.NodeConfig{
		NodeID:     1,
		NumMembers: 4,
	}, transports[1])
	proposer.Start()
	defer proposer.Stop()

	trustEst := trust.NewCombinedEstimator(
		trust.BayesianConfig{
			WindowSize: 4,
			MinSamples: 1,
			Weights:    trust.ClassifierWeights{W: [5]float64{3.0, 3.0, 1.0, 0.5, 0.3}, B: -1.0},
		},
		0.3, 0.7,
	)

	config := DefaultPipelineConfig()
	config.NumInstances = 1
	config.NumAgents = 7
	config.FaultThreshold = 0.5
	config.SafetyMarginDelta = 0
	config.GBCCommitTimeout = 50 * time.Millisecond

	pipe := NewPipeline(config, gbc.NewLogWithMembers(4), trustEst)
	pipe.SetGBCNodes([]*gbc.Node{proposer})
	pipe.SetInstances([]InstanceHandle{{
		InstanceID:     0,
		ValidatorCount: 7,
		FaultTolerance: 2,
		Agents:         makeAgents(0, 7),
	}})

	initialValidatorCount := pipe.instances[0].ValidatorCount
	metrics := make(map[uint64]trust.EpochEvent, 7)
	for i := uint64(0); i < 7; i++ {
		if i < 3 {
			metrics[i] = trust.EpochEvent{Timeouts: 10, Equivocations: 5, ViewChanges: 5, LatencyMs: 2000}
		} else {
			metrics[i] = trust.EpochEvent{Timeouts: 0, Equivocations: 0, ViewChanges: 0, LatencyMs: 30}
		}
	}

	for epoch := uint64(0); epoch < 3; epoch++ {
		if _, err := pipe.ProcessEpoch(epoch, metrics); err != nil {
			t.Fatalf("epoch %d: %v", epoch, err)
		}
	}

	if pipe.instances[0].ValidatorCount != initialValidatorCount {
		t.Fatalf("commit-then-apply violated: validator count changed from %d to %d without GBC quorum commit",
			initialValidatorCount, pipe.instances[0].ValidatorCount)
	}
}

func TestPipeline_PublishFailureRollback(t *testing.T) {
	// Verify that when GBC publish fails, the pipeline does NOT apply
	// local reconfiguration (commit-then-apply semantics from §III-E).
	// This is the defensive path that prevents split-brain membership.

	// Create a node with ID 99 in a 4-member committee. Round-robin proposer
	// selection will never choose ID 99, so all Propose calls return errNotProposer.
	fakeTransports := gbc.NewChannelTransportSet(1, 64)
	fakeNodes := []*gbc.Node{
		gbc.NewNode(gbc.NodeConfig{
			NodeID:     99, // never the proposer for heights 0,1,2,...
			NumMembers: 4,
		}, fakeTransports[0]),
	}
	fakeNodes[0].Start()
	defer fakeNodes[0].Stop()

	trustEst := trust.NewCombinedEstimator(
		trust.BayesianConfig{
			WindowSize: 4,
			MinSamples: 1,
			Weights:    trust.ClassifierWeights{W: [5]float64{3.0, 3.0, 1.0, 0.5, 0.3}, B: -1.0},
		},
		0.3, 0.7,
	)

	config := DefaultPipelineConfig()
	config.NumInstances = 1
	config.NumAgents = 7
	config.FaultThreshold = 0.5
	config.SafetyMarginDelta = 0

	pipe := NewPipeline(config, gbc.NewLogWithMembers(4), trustEst)
	pipe.SetGBCNodes(fakeNodes) // Node 99 will always return "not proposer"
	pipe.SetInstances([]InstanceHandle{{
		InstanceID:     0,
		ValidatorCount: 7,
		FaultTolerance: 2,
		Agents:         makeAgents(0, 7),
	}})

	// Record initial state
	initialValidatorCount := pipe.instances[0].ValidatorCount

	// Inject severe Byzantine behavior to trigger eviction decision
	metrics := make(map[uint64]trust.EpochEvent, 7)
	for i := uint64(0); i < 7; i++ {
		if i < 3 {
			metrics[i] = trust.EpochEvent{Timeouts: 10, Equivocations: 5, ViewChanges: 5, LatencyMs: 2000}
		} else {
			metrics[i] = trust.EpochEvent{Timeouts: 0, Equivocations: 0, ViewChanges: 0, LatencyMs: 30}
		}
	}

	// Run epochs — all GBC publishes will fail because node 99 is never proposer
	for epoch := uint64(0); epoch < 5; epoch++ {
		pipe.ProcessEpoch(epoch, metrics)
	}

	// Key assertion: despite the pipeline wanting to evict Byzantine agents,
	// the validator count must NOT have changed because GBC publish failed.
	// This verifies commit-then-apply semantics.
	if pipe.instances[0].ValidatorCount != initialValidatorCount {
		t.Fatalf("rollback violated: validator count changed from %d to %d despite GBC publish failure",
			initialValidatorCount, pipe.instances[0].ValidatorCount)
	}
	t.Log("commit-then-apply verified: local state unchanged after GBC publish failures")
}
