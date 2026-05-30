package main

import (
	"reflect"
	"testing"
	"time"
	"unsafe"

	"octopus-bft/octopus/adaptive"
	"octopus-bft/octopus/bootstrap"
	"octopus-bft/octopus/consensus/hotstuff"
	octcrypto "octopus-bft/octopus/crypto"
	"octopus-bft/octopus/hydra"
	"octopus-bft/octopus/membership"
	"octopus-bft/octopus/storage"
	"octopus-bft/octopus/types"
)

func buildAdaptiveEngineForTest(t *testing.T) *hotstuff.Engine {
	t.Helper()
	kp, err := octcrypto.GenerateKeyPair()
	if err != nil {
		t.Fatalf("generate keypair: %v", err)
	}
	validators := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp.PublicKey, Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: kp.PublicKey, Power: 1, IsActive: true},
		3: {ID: 3, PublicKey: kp.PublicKey, Power: 1, IsActive: true},
	}
	return hotstuff.NewEngineWithInstanceAndOptions(
		0,
		&types.Keypair{PublicKey: kp.PublicKey, PrivateKey: kp.PrivateKey},
		types.NewValidatorSet(1, validators),
		nil,
		storage.NewStorageManager(0),
		0,
		1,
		"octopus-consensus",
		nil,
		hotstuff.DefaultEngineOptions(),
	)
}

func buildAdaptiveEngineAndKeypairForTest(t *testing.T, nodeID uint64, validators map[uint64]*types.Validator) (*hotstuff.Engine, *types.Keypair) {
	t.Helper()
	kp, err := octcrypto.GenerateKeyPair()
	if err != nil {
		t.Fatalf("generate keypair: %v", err)
	}
	return hotstuff.NewEngineWithInstanceAndOptions(
		nodeID,
		&types.Keypair{PublicKey: kp.PublicKey, PrivateKey: kp.PrivateKey},
		types.NewValidatorSet(1, validators),
		nil,
		storage.NewStorageManager(nodeID),
		0,
		1,
		"octopus-consensus",
		nil,
		hotstuff.DefaultEngineOptions(),
	), &types.Keypair{PublicKey: kp.PublicKey, PrivateKey: kp.PrivateKey}
}

type adaptiveRuntimeRecordingNetwork struct {
	broadcasts []interface{}
}

func (n *adaptiveRuntimeRecordingNetwork) Broadcast(msg interface{}) {
	n.broadcasts = append(n.broadcasts, msg)
}

func (n *adaptiveRuntimeRecordingNetwork) Send(to uint64, msg interface{}) {}

func TestAdaptivePolicyFromConfig_SelectsScripted(t *testing.T) {
	cfg := &bootstrap.EngineConfig{
		AdaptiveEnabled: true,
		AdaptivePolicy:  "scripted",
		AdaptiveScript:  "C:/tmp/policy.json",
	}
	policy := adaptivePolicyFromConfig(cfg)
	if policy == nil {
		t.Fatalf("expected scripted policy")
	}
	if policy.Name() != "scripted" {
		t.Fatalf("unexpected policy name: %q", policy.Name())
	}
}

func TestOctopusAdaptiveRuntimeObserve(t *testing.T) {
	engine := buildAdaptiveEngineForTest(t)
	memberMgr := membership.NewMembershipManager(engine.GetCurrentValidatorSet().Validators)
	orderer := hotstuff.NewGlobalOrderer(1, 50*time.Millisecond)
	metrics := hotstuff.NewGlobalConfirmedMetrics(50 * time.Millisecond)
	block := types.NewBlock(7, nil, []byte("payload"), 3, 1, 0, 7, nil, nil)
	out := hotstuff.InstanceOutput{
		InstanceID:  0,
		LocalHeight: block.Height,
		Rank:        7,
		BlockHash:   block.Hash,
		Block:       block,
		EpochTransitions: []*types.EpochTransition{{
			OldEpoch:         1,
			NewEpoch:         2,
			ActivationHeight: block.Height,
			ActivationRank:   7,
		}},
	}
	metrics.ObserveGlobalConfirmed(out, time.Now())

	rt := octopusAdaptiveRuntime{
		nodeID:      0,
		engines:     []*hotstuff.Engine{engine},
		memberMgr:   memberMgr,
		orderer:     orderer,
		metrics:     metrics,
		rejectStats: func() map[string]uint64 { return map[string]uint64{"x": 2} },
	}
	rt.recordOrderedOutput(out)

	obs := rt.Observe()
	if obs.ValidatorCount != 4 {
		t.Fatalf("unexpected validator count: %d", obs.ValidatorCount)
	}
	if obs.PacemakerTimeoutMs != 500 {
		t.Fatalf("unexpected timeout: %d", obs.PacemakerTimeoutMs)
	}
	if obs.MempoolMaxBatchTxs != 2048 {
		t.Fatalf("unexpected batch size: %d", obs.MempoolMaxBatchTxs)
	}
	if obs.RejectTotal != 2 {
		t.Fatalf("unexpected reject total: %d", obs.RejectTotal)
	}
	if obs.CurrentConfigID != 1 || obs.HighestKnownConfigID != 1 || !obs.LocalValidator || !obs.CanParticipate {
		t.Fatalf("unexpected membership observation fields: %+v", obs)
	}
	if obs.GlobalConfirmedTotal != 1 || obs.GlobalConfirmedNil != 0 {
		t.Fatalf("unexpected global confirmation stats: %+v", obs)
	}
	if obs.LastOrderedRank != 7 || obs.LastOrderedHeight != block.Height || obs.LastOrderedLaneID != block.LaneID || obs.LastOrderedConfigID != block.ConfigID || obs.LastOrderedNil {
		t.Fatalf("unexpected ordered output state: %+v", obs)
	}
	if obs.LastOrderedTransitionCount != 1 || obs.LastReconfigEpoch != 2 {
		t.Fatalf("unexpected reconfiguration observation state: %+v", obs)
	}
}

func TestOctopusAdaptiveRuntimeRecordOrderedOutputPreservesMonotonicState(t *testing.T) {
	block := types.NewBlock(7, nil, []byte("payload"), 3, 1, 0, 7, nil, nil)
	rt := octopusAdaptiveRuntime{}
	rt.recordOrderedOutput(hotstuff.InstanceOutput{
		Rank:  7,
		Block: block,
		EpochTransitions: []*types.EpochTransition{{
			NewEpoch: 2,
		}},
	})
	rt.recordOrderedOutput(hotstuff.InstanceOutput{
		Rank:  8,
		IsNil: true,
	})

	obs := rt.Observe()
	if obs.LastOrderedRank != 8 || !obs.LastOrderedNil {
		t.Fatalf("expected latest rank/nil marker to advance, got %+v", obs)
	}
	if obs.LastOrderedHeight != block.Height {
		t.Fatalf("expected ordered height to remain monotonic, got %+v", obs)
	}
	if obs.LastReconfigEpoch != 2 {
		t.Fatalf("expected reconfig epoch to remain monotonic, got %+v", obs)
	}
	if obs.LastOrderedTransitionCount != 0 {
		t.Fatalf("expected last transition count to reflect latest output, got %+v", obs)
	}
}

func TestOctopusAdaptiveRuntimeApply(t *testing.T) {
	engine := buildAdaptiveEngineForTest(t)
	rt := octopusAdaptiveRuntime{
		engines: []*hotstuff.Engine{engine},
	}

	err := rt.Apply(adaptive.Action{
		CommitteeSize:             6,
		PacemakerTimeoutMs:        1500,
		MempoolMaxBatchTxs:        512,
		MempoolProposalIntervalMs: 80,
	})
	if err != nil {
		t.Fatalf("apply action failed: %v", err)
	}

	got := engine.GetAdaptiveTuning()
	if got.CommitteeSize != 6 || got.TimeoutMs != 1500 {
		t.Fatalf("unexpected engine tuning after apply: %+v", got)
	}
	mem := engine.GetMempoolAdaptiveTuning()
	if mem.MaxBatchTxs != 512 {
		t.Fatalf("unexpected mempool batch tuning: %d", mem.MaxBatchTxs)
	}
	if mem.ProposalInterval != 80*time.Millisecond {
		t.Fatalf("unexpected mempool proposal interval: %v", mem.ProposalInterval)
	}
}

func TestOctopusAdaptiveRuntimeObserveIncludesScenarioContext(t *testing.T) {
	engine := buildAdaptiveEngineForTest(t)
	rt := octopusAdaptiveRuntime{
		nodeID:  0,
		engines: []*hotstuff.Engine{engine},
	}
	rt.SetScenarioContext(adaptive.ScenarioContext{
		HeterogeneityScore: 0.8,
		ChurnRate:          0.3,
		AdversaryScore:     0.6,
		NetworkJitterMs:    45,
		AILoadScore:        0.7,
	})

	obs := rt.Observe()
	if obs.HeterogeneityScore != 0.8 || obs.AdversaryScore != 0.6 || obs.NetworkJitterMs != 45 {
		t.Fatalf("unexpected scenario context in observation: %+v", obs)
	}
}

func TestOctopusAdaptiveRuntimeObserveUsesCommittedConfigForCurrentAndDiscoveryForHighestKnown(t *testing.T) {
	engine := buildAdaptiveEngineForTest(t)
	memberMgr := membership.NewMembershipManager(engine.GetCurrentValidatorSet().Validators)
	validators := make(map[uint64]*hydra.Validator, len(engine.GetCurrentValidatorSet().Validators))
	for id, v := range engine.GetCurrentValidatorSet().Validators {
		copyVal := *v
		validators[id] = &hydra.Validator{ID: copyVal.ID, PublicKey: copyVal.PublicKey, Power: copyVal.Power, IsActive: copyVal.IsActive}
	}
	hydraMgr, err := hydra.NewHydraManager(0, validators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	if err := hydraMgr.InstallCommittedConfiguration(&types.Configuration{ID: 1, Validators: engine.GetCurrentValidatorSet().Validators, QuorumSize: 3}); err != nil {
		t.Fatalf("install committed config: %v", err)
	}
	discovered := &hydra.Configuration{ID: 2, Validators: validators, QuorumSize: 3}
	if err := hydraMgr.DiscoveryManager.AddConfiguration(discovered); err != nil {
		t.Fatalf("add discovered config: %v", err)
	}

	rt := octopusAdaptiveRuntime{
		nodeID:    0,
		engines:   []*hotstuff.Engine{engine},
		memberMgr: memberMgr,
		hydraMgr:  hydraMgr,
	}
	obs := rt.Observe()
	if obs.CurrentConfigID != 1 {
		t.Fatalf("expected committed current config id 1, got %d", obs.CurrentConfigID)
	}
	if obs.HighestKnownConfigID != 2 {
		t.Fatalf("expected discovered highest-known config id 2, got %d", obs.HighestKnownConfigID)
	}
}

func TestOctopusAdaptiveRuntimeObserveToleratesMissingHydraSubcomponents(t *testing.T) {
	engine := buildAdaptiveEngineForTest(t)
	memberMgr := membership.NewMembershipManager(engine.GetCurrentValidatorSet().Validators)
	validators := make(map[uint64]*hydra.Validator, len(engine.GetCurrentValidatorSet().Validators))
	for id, v := range engine.GetCurrentValidatorSet().Validators {
		copyVal := *v
		validators[id] = &hydra.Validator{ID: copyVal.ID, PublicKey: copyVal.PublicKey, Power: copyVal.Power, IsActive: copyVal.IsActive}
	}
	hydraMgr, err := hydra.NewHydraManager(0, validators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	hydraMgr.LSetManager = nil
	hydraMgr.TempConfigManager = nil

	rt := octopusAdaptiveRuntime{
		nodeID:    0,
		engines:   []*hotstuff.Engine{engine},
		memberMgr: memberMgr,
		hydraMgr:  hydraMgr,
	}
	obs := rt.Observe()
	if obs.LSetSize != 0 || obs.PendingJoins != 0 || obs.PendingLeaves != 0 {
		t.Fatalf("expected zeroed hydra subcomponent observation when managers are nil, got %+v", obs)
	}
	if !obs.CanParticipate {
		t.Fatalf("expected can_participate to still reflect hydra manager status")
	}
}

func TestOctopusAdaptiveRuntimeObserveExportsAgentObservations(t *testing.T) {
	engine0 := buildAdaptiveEngineForTest(t)
	engine1 := hotstuff.NewEngineWithInstanceAndOptions(
		0,
		&types.Keypair{PublicKey: engine0.GetCurrentValidatorSet().Validators[0].PublicKey, PrivateKey: nil},
		engine0.GetCurrentValidatorSet().Copy(),
		nil,
		storage.NewStorageManager(100),
		1,
		2,
		"octopus-consensus",
		nil,
		hotstuff.DefaultEngineOptions(),
	)
	engine0.SetAdaptiveTuning(hotstuff.AdaptiveTuning{CommitteeSize: 4, TimeoutMs: 1200})
	engine1.SetAdaptiveTuning(hotstuff.AdaptiveTuning{CommitteeSize: 6, TimeoutMs: 1500})

	rt := octopusAdaptiveRuntime{
		nodeID:  0,
		engines: []*hotstuff.Engine{engine0, engine1},
	}
	obs := rt.Observe()
	if len(obs.Agents) != 2 {
		t.Fatalf("expected 2 agent observations, got %d", len(obs.Agents))
	}
	if obs.Agents[0].InstanceID != 0 || obs.Agents[1].InstanceID != 1 {
		t.Fatalf("unexpected agent instance ids: %+v", obs.Agents)
	}
}

func TestOctopusAdaptiveRuntimeApplyRoutesPerAgentActions(t *testing.T) {
	engine0 := buildAdaptiveEngineForTest(t)
	engine1 := hotstuff.NewEngineWithInstanceAndOptions(
		0,
		&types.Keypair{PublicKey: engine0.GetCurrentValidatorSet().Validators[0].PublicKey, PrivateKey: nil},
		engine0.GetCurrentValidatorSet().Copy(),
		nil,
		storage.NewStorageManager(101),
		1,
		2,
		"octopus-consensus",
		nil,
		hotstuff.DefaultEngineOptions(),
	)
	rt := octopusAdaptiveRuntime{
		engines: []*hotstuff.Engine{engine0, engine1},
	}

	err := rt.Apply(adaptive.Action{
		AgentActions: []adaptive.AgentAction{
			{InstanceID: 0, CommitteeSize: 4, PacemakerTimeoutMs: 1200, MempoolMaxBatchTxs: 512, MempoolProposalIntervalMs: 80},
			{InstanceID: 1, CommitteeSize: 6, PacemakerTimeoutMs: 1500, MempoolMaxBatchTxs: 1024, MempoolProposalIntervalMs: 60},
		},
	})
	if err != nil {
		t.Fatalf("apply per-agent action failed: %v", err)
	}

	got0 := engine0.GetAdaptiveTuning()
	got1 := engine1.GetAdaptiveTuning()
	if got0.CommitteeSize != 4 || got1.CommitteeSize != 6 {
		t.Fatalf("unexpected per-agent committee sizes: engine0=%+v engine1=%+v", got0, got1)
	}
	mem0 := engine0.GetMempoolAdaptiveTuning()
	mem1 := engine1.GetMempoolAdaptiveTuning()
	if mem0.MaxBatchTxs != 512 || mem1.MaxBatchTxs != 1024 {
		t.Fatalf("unexpected per-agent mempool tuning: mem0=%+v mem1=%+v", mem0, mem1)
	}
}

func TestOctopusAdaptiveRuntimeApplySubmitsSFACRemoteEvictIntent(t *testing.T) {
	engine := buildAdaptiveEngineForTest(t)
	memberMgr := membership.NewMembershipManager(engine.GetCurrentValidatorSet().Validators)
	hydraValidators := make(map[uint64]*hydra.Validator, len(engine.GetCurrentValidatorSet().Validators))
	for id, v := range engine.GetCurrentValidatorSet().Validators {
		copyVal := *v
		hydraValidators[id] = &copyVal
	}
	hydraMgr, err := hydra.NewHydraManager(0, hydraValidators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	rt := octopusAdaptiveRuntime{
		nodeID:    0,
		engines:   []*hotstuff.Engine{engine},
		memberMgr: memberMgr,
		hydraMgr:  hydraMgr,
	}

	if err := rt.Apply(adaptive.Action{AgentActions: []adaptive.AgentAction{{
		InstanceID:           0,
		ReconfigEvictNodeIDs: []uint64{2},
	}}}); err != nil {
		t.Fatalf("apply SFAC evict intent: %v", err)
	}
	pending := hydraMgr.TempConfigManager.GetPendingLeaves()
	if len(pending) != 1 || pending[0].ID != 2 {
		t.Fatalf("expected pending Hydra leave for validator 2, got %+v", pending)
	}
}

func TestOctopusAdaptiveRuntimeApplyRollsBackHydraPendingJoinWhenTxEnqueueFails(t *testing.T) {
	validators := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: []byte("v0"), Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: []byte("v1"), Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: []byte("v2"), Power: 1, IsActive: true},
	}
	engine, kp := buildAdaptiveEngineAndKeypairForTest(t, 3, validators)
	mp := hydraInjectionMempool(engine)
	if mp == nil {
		t.Fatalf("expected mempool")
	}
	mpValue := reflect.ValueOf(mp).Elem().FieldByName("maxTxSize")
	reflect.NewAt(mpValue.Type(), unsafe.Pointer(mpValue.UnsafeAddr())).Elem().SetInt(1)
	memberMgr := membership.NewMembershipManager(validators)
	hydraValidators := make(map[uint64]*hydra.Validator, len(validators))
	for id, v := range validators {
		copyVal := *v
		hydraValidators[id] = &copyVal
	}
	hydraMgr, err := hydra.NewHydraManager(3, hydraValidators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	rt := octopusAdaptiveRuntime{
		nodeID:    3,
		keypair:   kp,
		engines:   []*hotstuff.Engine{engine},
		memberMgr: memberMgr,
		hydraMgr:  hydraMgr,
	}

	if err := rt.Apply(adaptive.Action{SubmitJoin: true}); err == nil {
		t.Fatalf("expected join enqueue failure")
	}
	if pending := hydraMgr.TempConfigManager.GetPendingJoins(); len(pending) != 0 {
		t.Fatalf("expected hydra pending join rollback after enqueue failure, got %+v", pending)
	}
}

func TestOctopusAdaptiveRuntimeApplyRollsBackHydraPendingJoinWhenEngineMissing(t *testing.T) {
	validators := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: []byte("v0"), Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: []byte("v1"), Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: []byte("v2"), Power: 1, IsActive: true},
	}
	_, kp := buildAdaptiveEngineAndKeypairForTest(t, 3, validators)
	memberMgr := membership.NewMembershipManager(validators)
	hydraValidators := make(map[uint64]*hydra.Validator, len(validators))
	for id, v := range validators {
		copyVal := *v
		hydraValidators[id] = &copyVal
	}
	hydraMgr, err := hydra.NewHydraManager(3, hydraValidators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	rt := octopusAdaptiveRuntime{
		nodeID:    3,
		keypair:   kp,
		memberMgr: memberMgr,
		hydraMgr:  hydraMgr,
	}

	if err := rt.Apply(adaptive.Action{SubmitJoin: true, HydraDiscoveryTarget: 2}); err == nil {
		t.Fatalf("expected join submission to fail when engine is missing")
	}
	if pending := hydraMgr.TempConfigManager.GetPendingJoins(); len(pending) != 0 {
		t.Fatalf("expected hydra pending join rollback when engine is missing, got %+v", pending)
	}
}

func TestOctopusAdaptiveRuntimeApplyRollsBackHydraPendingLeaveWhenEngineMissing(t *testing.T) {
	validators := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: []byte("v0"), Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: []byte("v1"), Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: []byte("v2"), Power: 1, IsActive: true},
		3: {ID: 3, PublicKey: []byte("v3"), Power: 1, IsActive: true},
	}
	_, kp := buildAdaptiveEngineAndKeypairForTest(t, 0, validators)
	memberMgr := membership.NewMembershipManager(validators)
	hydraValidators := make(map[uint64]*hydra.Validator, len(validators))
	for id, v := range validators {
		copyVal := *v
		hydraValidators[id] = &copyVal
	}
	hydraMgr, err := hydra.NewHydraManager(0, hydraValidators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	rt := octopusAdaptiveRuntime{
		nodeID:    0,
		keypair:   kp,
		memberMgr: memberMgr,
		hydraMgr:  hydraMgr,
	}

	if err := rt.Apply(adaptive.Action{SubmitLeave: true}); err == nil {
		t.Fatalf("expected leave submission to fail when engine is missing")
	}
	if pending := hydraMgr.TempConfigManager.GetPendingLeaves(); len(pending) != 0 {
		t.Fatalf("expected hydra pending leave rollback when engine is missing, got %+v", pending)
	}
}

func TestOctopusAdaptiveRuntimeApplyDoesNotRequestDiscoveryWhenJoinFails(t *testing.T) {
	validators := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: []byte("v0"), Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: []byte("v1"), Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: []byte("v2"), Power: 1, IsActive: true},
	}
	_, kp := buildAdaptiveEngineAndKeypairForTest(t, 3, validators)
	memberMgr := membership.NewMembershipManager(validators)
	hydraValidators := make(map[uint64]*hydra.Validator, len(validators))
	for id, v := range validators {
		copyVal := *v
		hydraValidators[id] = &copyVal
	}
	net := &adaptiveRuntimeRecordingNetwork{}
	hydraMgr, err := hydra.NewHydraManager(3, hydraValidators, net)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	rt := octopusAdaptiveRuntime{
		nodeID:    3,
		keypair:   kp,
		memberMgr: memberMgr,
		hydraMgr:  hydraMgr,
	}

	if err := rt.Apply(adaptive.Action{SubmitJoin: true, HydraDiscoveryTarget: 2}); err == nil {
		t.Fatalf("expected join submission to fail when engine is missing")
	}
	if got := len(net.broadcasts); got != 0 {
		t.Fatalf("expected no discovery broadcast when join fails, got %d", got)
	}
}
