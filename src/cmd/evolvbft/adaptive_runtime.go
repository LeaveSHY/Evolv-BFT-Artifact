package main

import (
	"encoding/json"
	"errors"
	"math"
	"sort"
	"sync"
	"time"

	"evolvbft/evolvbft/adaptive"
	"evolvbft/evolvbft/bootstrap"
	"evolvbft/evolvbft/consensus/hotstuff"
	"evolvbft/evolvbft/consensus/mempool"
	"evolvbft/evolvbft/hydra"
	"evolvbft/evolvbft/membership"
	"evolvbft/evolvbft/network/libp2p"
	"evolvbft/evolvbft/trust"
	"evolvbft/evolvbft/types"
)

type evolvbftAdaptiveRuntime struct {
	nodeID      uint64
	keypair     *types.Keypair
	engines     []*hotstuff.Engine
	memberMgr   *membership.MembershipManager
	orderer     *hotstuff.GlobalOrderer
	metrics     *hotstuff.GlobalConfirmedMetrics
	net         *libp2p.P2PNetwork
	hydraMgr    *hydra.HydraManager
	rejectStats func() map[string]uint64
	trustEst    *trust.BayesianEstimator // Paper-grade trust estimator (§III-D)
	trustAgg    *trust.Aggregator       // Cross-instance trust sharing (§III-C)
	trustDecay  *trust.TrustDecay       // Temporal forgetting (evolving adversaries)
	ctrl        *adaptive.Controller    // back-reference for epoch gate advance
	mu          sync.RWMutex
	scenario    adaptive.ScenarioContext
	lastOrdered orderedState
}

type orderedState struct {
	Rank            uint64
	Height          uint64
	LaneID          uint64
	ConfigID        uint64
	IsNil           bool
	TransitionCount int
	ReconfigEpoch   uint64
}

func (r *evolvbftAdaptiveRuntime) Observe() adaptive.Observation {
	obs := adaptive.Observation{
		Timestamp: time.Now(),
		NodeID:    r.nodeID,
	}

	var snapshot hotstuff.GlobalConfirmedSnapshot
	if r.metrics != nil && r.orderer != nil {
		rejects := map[string]uint64{}
		if r.rejectStats != nil {
			rejects = r.rejectStats()
		}
		snapshot = r.metrics.Snapshot(r.orderer, rejects)
		obs.ThroughputTPS = snapshot.ThroughputTPS
		obs.LatencyP50Ms = snapshot.LatencyP50Ms
		obs.LatencyP95Ms = snapshot.LatencyP95Ms
		obs.LatencyP99Ms = snapshot.LatencyP99Ms
		obs.RecoveryP95Ms = snapshot.RecoveryP95Ms
		obs.BacklogPending = snapshot.BacklogPending
		obs.BacklogMissing = snapshot.BacklogMissing
		obs.RejectTotal = snapshot.RejectTotal
		obs.GlobalConfirmedTotal = snapshot.GlobalConfirmedTotal
		obs.GlobalConfirmedNil = snapshot.GlobalConfirmedNil
	}

	if r.memberMgr != nil {
		cfg := r.memberMgr.GetCurrentConfig()
		if cfg != nil {
			obs.Epoch = cfg.ID
			obs.CurrentConfigID = cfg.ID
			obs.ValidatorCount = len(cfg.Validators)
			_, obs.LocalValidator = cfg.Validators[r.nodeID]
		}
	}

	if len(r.engines) > 0 && r.engines[0] != nil {
		obs.Agents = make([]adaptive.AgentObservation, 0, len(r.engines))
		for _, engine := range r.engines {
			if engine == nil {
				continue
			}
			tuning := engine.GetAdaptiveTuning()
			memTuning := engine.GetMempoolAdaptiveTuning()
			valSet := engine.GetCurrentValidatorSet()
			agentObs := adaptive.AgentObservation{
				InstanceID:                engine.GetInstanceID(),
				CommitteeSize:             tuning.CommitteeSize,
				PacemakerTimeoutMs:        int(tuning.TimeoutMs),
				MempoolMaxBatchTxs:        memTuning.MaxBatchTxs,
				MempoolProposalIntervalMs: int(memTuning.ProposalInterval / time.Millisecond),
			}
			if valSet != nil {
				agentObs.Epoch = valSet.Epoch
				agentObs.ValidatorCount = len(valSet.Validators)
				// Fill FaultsEstimate from trust system (§III-D):
				// Count validators whose Bayesian fault probability >= threshold.
				// This drives the SafetyFilter's f in "n_after >= 3f+1+δ_s".
				agentObs.FaultsEstimate = r.estimateFaults(valSet)
			}
			obs.Agents = append(obs.Agents, agentObs)
		}

		engine := r.engines[0]
		tuning := engine.GetAdaptiveTuning()
		memTuning := engine.GetMempoolAdaptiveTuning()
		if obs.Epoch == 0 && engine.GetCurrentValidatorSet() != nil {
			obs.Epoch = engine.GetCurrentValidatorSet().Epoch
			obs.ValidatorCount = len(engine.GetCurrentValidatorSet().Validators)
		}
		obs.CommitteeSize = tuning.CommitteeSize
		obs.PacemakerTimeoutMs = int(tuning.TimeoutMs)
		obs.MempoolMaxBatchTxs = memTuning.MaxBatchTxs
		obs.MempoolProposalIntervalMs = int(memTuning.ProposalInterval / time.Millisecond)
		obs.TrustSnapshots = r.buildTrustSnapshotsWithBayesian(engine.GetLeaderReputation())
	}

	if r.net != nil {
		stats := r.net.GetNetworkStats()
		obs.ConnectedPeers = stats.ConnectedPeers
		obs.KnownPeers = stats.KnownPeers
	}

	if r.hydraMgr != nil {
		obs.CanParticipate = r.hydraMgr.CanParticipate()
		if r.hydraMgr.LSetManager != nil {
			obs.LSetSize = len(r.hydraMgr.LSetManager.GetLSet())
		}
		if r.hydraMgr.TempConfigManager != nil {
			obs.PendingJoins = len(r.hydraMgr.TempConfigManager.GetPendingJoins())
			obs.PendingLeaves = len(r.hydraMgr.TempConfigManager.GetPendingLeaves())
		}
		if config := r.hydraMgr.GetHighestKnownConfiguration(); config != nil {
			obs.HighestKnownConfigID = config.ID
		}
	}
	if obs.HighestKnownConfigID == 0 {
		obs.HighestKnownConfigID = obs.CurrentConfigID
	}
	if r.hydraMgr == nil {
		obs.CanParticipate = true
	}

	r.mu.RLock()
	obs.HeterogeneityScore = r.scenario.HeterogeneityScore
	obs.ChurnRate = r.scenario.ChurnRate
	obs.AdversaryScore = r.scenario.AdversaryScore
	obs.NetworkJitterMs = r.scenario.NetworkJitterMs
	obs.AILoadScore = r.scenario.AILoadScore
	obs.LastOrderedRank = r.lastOrdered.Rank
	obs.LastOrderedHeight = r.lastOrdered.Height
	obs.LastOrderedLaneID = r.lastOrdered.LaneID
	obs.LastOrderedConfigID = r.lastOrdered.ConfigID
	obs.LastOrderedNil = r.lastOrdered.IsNil
	obs.LastOrderedTransitionCount = r.lastOrdered.TransitionCount
	obs.LastReconfigEpoch = r.lastOrdered.ReconfigEpoch
	r.mu.RUnlock()

	return obs
}

func (r *evolvbftAdaptiveRuntime) Apply(action adaptive.Action) error {
	if len(action.AgentActions) > 0 {
		engineByInstance := make(map[uint64]*hotstuff.Engine, len(r.engines))
		for _, engine := range r.engines {
			if engine == nil {
				continue
			}
			engineByInstance[engine.GetInstanceID()] = engine
		}
		for _, agentAction := range action.AgentActions {
			engine, ok := engineByInstance[agentAction.InstanceID]
			if ok && engine != nil && agentActionHasTuning(agentAction) {
				engine.SetAdaptiveTuning(hotstuff.AdaptiveTuning{
					CommitteeSize: agentAction.CommitteeSize,
					TimeoutMs:     int64(agentAction.PacemakerTimeoutMs),
				})
				engine.SetMempoolAdaptiveTuning(mempool.AdaptiveTuning{
					MaxBatchTxs:      agentAction.MempoolMaxBatchTxs,
					ProposalInterval: time.Duration(agentAction.MempoolProposalIntervalMs) * time.Millisecond,
				})
			}
			if err := r.applyAgentReconfiguration(agentAction); err != nil {
				return err
			}
		}
	} else if action.CommitteeSize > 0 || action.PacemakerTimeoutMs > 0 || action.MempoolMaxBatchTxs > 0 || action.MempoolProposalIntervalMs > 0 {
		for _, engine := range r.engines {
			if engine == nil {
				continue
			}
			engine.SetAdaptiveTuning(hotstuff.AdaptiveTuning{
				CommitteeSize: action.CommitteeSize,
				TimeoutMs:     int64(action.PacemakerTimeoutMs),
			})
			engine.SetMempoolAdaptiveTuning(mempool.AdaptiveTuning{
				MaxBatchTxs:      action.MempoolMaxBatchTxs,
				ProposalInterval: time.Duration(action.MempoolProposalIntervalMs) * time.Millisecond,
			})
		}
	}

	if action.SubmitJoin {
		if err := r.submitLocalJoinIntent(); err != nil {
			return err
		}
	}
	if action.SubmitLeave {
		if err := r.submitLocalLeaveIntent(); err != nil {
			return err
		}
	}
	if action.HydraDiscoveryTarget > 0 && r.hydraMgr != nil {
		if err := r.hydraMgr.RequestConfigDiscovery(uint64(action.HydraDiscoveryTarget)); err != nil {
			return err
		}
	}
	return nil
}

func agentActionHasTuning(action adaptive.AgentAction) bool {
	return action.CommitteeSize > 0 || action.PacemakerTimeoutMs > 0 || action.MempoolMaxBatchTxs > 0 || action.MempoolProposalIntervalMs > 0
}

func (r *evolvbftAdaptiveRuntime) applyAgentReconfiguration(action adaptive.AgentAction) error {
	for _, targetID := range action.ReconfigEvictNodeIDs {
		if targetID == r.nodeID {
			if err := r.submitLocalLeaveIntent(); err != nil {
				return err
			}
			continue
		}
		if r.hydraMgr == nil || hasPendingHydraRequest(r.hydraMgr, targetID, false) {
			continue
		}
		if err := r.hydraMgr.SubmitLeaveRequest(targetID); err != nil {
			return err
		}
	}
	for _, targetID := range action.ReconfigAdmitNodeIDs {
		if targetID == r.nodeID {
			if err := r.submitLocalJoinIntent(); err != nil {
				return err
			}
			continue
		}
		if r.hydraMgr == nil || hasPendingHydraRequest(r.hydraMgr, targetID, true) || r.isCurrentValidator(targetID) {
			continue
		}
		pubKey, power, ok := r.lookupKnownValidator(targetID)
		if !ok {
			continue
		}
		if err := r.hydraMgr.SubmitJoinRequest(targetID, pubKey, power); err != nil {
			return err
		}
	}
	return nil
}

func (r *evolvbftAdaptiveRuntime) isCurrentValidator(nodeID uint64) bool {
	if r.memberMgr == nil {
		return false
	}
	cfg := r.memberMgr.GetCurrentConfig()
	if cfg == nil || cfg.Validators == nil {
		return false
	}
	_, ok := cfg.Validators[nodeID]
	return ok
}

func (r *evolvbftAdaptiveRuntime) lookupKnownValidator(nodeID uint64) ([]byte, uint64, bool) {
	if r.memberMgr != nil {
		if cfg := r.memberMgr.GetCurrentConfig(); cfg != nil && cfg.Validators != nil {
			if validator, ok := cfg.Validators[nodeID]; ok && validator != nil {
				return append([]byte(nil), validator.PublicKey...), validator.Power, true
			}
		}
	}
	if r.hydraMgr != nil {
		if cfg := r.hydraMgr.GetHighestKnownConfiguration(); cfg != nil && cfg.Validators != nil {
			if validator, ok := cfg.Validators[nodeID]; ok && validator != nil {
				return append([]byte(nil), validator.PublicKey...), validator.Power, true
			}
		}
		if cfg := r.hydraMgr.GetCurrentConfiguration(); cfg != nil && cfg.Validators != nil {
			if validator, ok := cfg.Validators[nodeID]; ok && validator != nil {
				return append([]byte(nil), validator.PublicKey...), validator.Power, true
			}
		}
	}
	return nil, 0, false
}

func (r *evolvbftAdaptiveRuntime) submitLocalJoinIntent() error {
	return submitLocalJoinIntent(r.nodeID, r.keypair, firstEngine(r.engines), r.memberMgr, r.hydraMgr)
}

func (r *evolvbftAdaptiveRuntime) submitLocalLeaveIntent() error {
	return submitLocalLeaveIntent(r.nodeID, r.keypair, firstEngine(r.engines), r.memberMgr, r.hydraMgr)
}

func buildTrustSnapshots(reputation *hotstuff.LeaderReputation) []adaptive.TrustSnapshot {
	if reputation == nil {
		return nil
	}
	stats := reputation.GetAllStats()
	if len(stats) == 0 {
		return nil
	}
	sort.Slice(stats, func(i, j int) bool {
		return stats[i].LeaderID < stats[j].LeaderID
	})
	snapshots := make([]adaptive.TrustSnapshot, 0, len(stats))
	for _, stat := range stats {
		total := stat.Successes + stat.Timeouts + stat.NilBlocks
		if total == 0 {
			continue
		}
		successRate := clampProbability(float64(stat.Successes) / float64(total))
		failureProbability := clampProbability(float64(stat.Timeouts+stat.NilBlocks) / float64(total))
		snap := adaptive.TrustSnapshot{
			NodeID:             stat.LeaderID,
			SampleCount:        total,
			SuccessRate:        successRate,
			FailureProbability: failureProbability,
			ClaimBoundary:      adaptive.TrustSnapshotBoundary,
		}
		// Populate Eq. 5 features from reputation counters (fallback path).
		// Rates are normalized by total events (approximating W).
		w := float64(total)
		snap.TimeoutRate = clampProbability(float64(stat.Timeouts) / w)
		snap.EquivocationRate = clampProbability(float64(stat.Equivocations) / w)
		snap.ViewChangeRate = clampProbability(float64(stat.ViewChangeInits) / w)
		// Latency features: already normalized in stats (ms), normalize to [0,1] with 1000ms cap.
		snap.MeanLatency = clampProbability(stat.AvgLatencyMs / 1000.0)
		snap.StdLatency = clampProbability(stat.StdLatencyMs / 1000.0)
		snapshots = append(snapshots, snap)
	}
	if len(snapshots) == 0 {
		return nil
	}
	return snapshots
}

func clampProbability(v float64) float64 {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return 0
	}
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func submitLocalJoinIntent(nodeID uint64, keypair *types.Keypair, engine *hotstuff.Engine, memberMgr *membership.MembershipManager, hydraMgr *hydra.HydraManager) error {
	if memberMgr == nil || keypair == nil {
		return nil
	}
	cfg := memberMgr.GetCurrentConfig()
	if cfg == nil {
		return nil
	}
	if _, exists := cfg.Validators[nodeID]; exists {
		return nil
	}
	if hasPendingHydraRequest(hydraMgr, nodeID, true) {
		return nil
	}
	registeredHydraIntent := false
	if hydraMgr != nil {
		if err := hydraMgr.SubmitJoinRequest(nodeID, keypair.PublicKey, 1); err != nil {
			return err
		}
		registeredHydraIntent = true
	}
	if engine == nil {
		if registeredHydraIntent && hydraMgr != nil && hydraMgr.TempConfigManager != nil {
			hydraMgr.TempConfigManager.RemoveJoinRequest(nodeID)
		}
		return errors.New("join engine unavailable")
	}
	vrfPub := engine.GetVRFPubKey()
	var vrfPubRaw []byte
	if vrfPub != nil {
		raw, err := vrfPub.MarshalBinary()
		if err != nil {
			if registeredHydraIntent && hydraMgr != nil && hydraMgr.TempConfigManager != nil {
				hydraMgr.TempConfigManager.RemoveJoinRequest(nodeID)
			}
			return err
		}
		vrfPubRaw = raw
	}
	reconfigData := types.ReconfigData{
		Type:         types.ReconfigJoin,
		NodeID:       nodeID,
		PublicKey:    keypair.PublicKey,
		VRFPublicKey: vrfPubRaw,
		Power:        1,
		Epoch:        cfg.ID,
		TargetEpoch:  cfg.ID + 1,
	}
	reconfigData.Sign(keypair.PrivateKey)
	payload, err := json.Marshal(reconfigData)
	if err != nil {
		if registeredHydraIntent && hydraMgr != nil && hydraMgr.TempConfigManager != nil {
			hydraMgr.TempConfigManager.RemoveJoinRequest(nodeID)
		}
		return err
	}
	if err := engine.AddTransaction(&types.Transaction{
		Type:    types.TxTypeReconfig,
		Payload: payload,
	}); err != nil {
		if registeredHydraIntent && hydraMgr != nil && hydraMgr.TempConfigManager != nil {
			hydraMgr.TempConfigManager.RemoveJoinRequest(nodeID)
		}
		return err
	}
	return nil
}

func submitLocalLeaveIntent(nodeID uint64, keypair *types.Keypair, engine *hotstuff.Engine, memberMgr *membership.MembershipManager, hydraMgr *hydra.HydraManager) error {
	if memberMgr == nil || keypair == nil {
		return nil
	}
	cfg := memberMgr.GetCurrentConfig()
	if cfg == nil {
		return nil
	}
	if _, exists := cfg.Validators[nodeID]; !exists {
		return nil
	}
	if len(cfg.Validators) <= 3 {
		return nil
	}
	if hasPendingHydraRequest(hydraMgr, nodeID, false) {
		return nil
	}
	registeredHydraIntent := false
	if hydraMgr != nil {
		if err := hydraMgr.SubmitLeaveRequest(nodeID); err != nil {
			return err
		}
		registeredHydraIntent = true
	}
	if engine == nil {
		if registeredHydraIntent && hydraMgr != nil && hydraMgr.TempConfigManager != nil {
			hydraMgr.TempConfigManager.RemoveLeaveRequest(nodeID)
		}
		return errors.New("leave engine unavailable")
	}
	reconfigData := types.ReconfigData{
		Type:        types.ReconfigLeave,
		NodeID:      nodeID,
		PublicKey:   keypair.PublicKey,
		Epoch:       cfg.ID,
		TargetEpoch: cfg.ID + 1,
	}
	reconfigData.Sign(keypair.PrivateKey)
	payload, err := json.Marshal(reconfigData)
	if err != nil {
		if registeredHydraIntent && hydraMgr != nil && hydraMgr.TempConfigManager != nil {
			hydraMgr.TempConfigManager.RemoveLeaveRequest(nodeID)
		}
		return err
	}
	if err := engine.AddTransaction(&types.Transaction{
		Type:    types.TxTypeReconfig,
		Payload: payload,
	}); err != nil {
		if registeredHydraIntent && hydraMgr != nil && hydraMgr.TempConfigManager != nil {
			hydraMgr.TempConfigManager.RemoveLeaveRequest(nodeID)
		}
		return err
	}
	return nil
}

func hasPendingHydraRequest(hydraMgr *hydra.HydraManager, nodeID uint64, join bool) bool {
	if hydraMgr == nil || hydraMgr.TempConfigManager == nil {
		return false
	}
	var requests []*hydra.MemberRequest
	if join {
		requests = hydraMgr.TempConfigManager.GetPendingJoins()
	} else {
		requests = hydraMgr.TempConfigManager.GetPendingLeaves()
	}
	for _, req := range requests {
		if req != nil && req.ID == nodeID {
			return true
		}
	}
	return false
}

func firstEngine(engines []*hotstuff.Engine) *hotstuff.Engine {
	for _, engine := range engines {
		if engine != nil {
			return engine
		}
	}
	return nil
}

func adaptivePolicyFromConfig(cfg *bootstrap.EngineConfig) adaptive.Policy {
	if cfg == nil || !cfg.AdaptiveEnabled {
		return nil
	}
	return adaptive.PolicyByName(cfg.AdaptivePolicy, cfg.AdaptiveScript, cfg.AdaptivePolicyURL)
}

func buildAdaptiveController(cfg *bootstrap.EngineConfig, runtime *evolvbftAdaptiveRuntime) *adaptive.Controller {
	if cfg == nil || runtime == nil || !cfg.AdaptiveEnabled {
		return nil
	}
	policy := adaptivePolicyFromConfig(cfg)
	if policy == nil {
		return nil
	}
	controller := adaptive.NewController(
		adaptive.Config{
			Enabled:  true,
			Interval: time.Duration(cfg.AdaptiveIntervalMs) * time.Millisecond,
		},
		runtime,
		runtime,
		policy,
		adaptive.DefaultGuardrails(),
	)
	controller.SetRewardModel(adaptive.DefaultRewardModel())
	// §IV Theorem regret-bound: attach cumulative regret tracker for O(√T) verification
	controller.SetRegretTracker(adaptive.NewRegretTracker(adaptive.RegretConfig{
		RewardUpperBound: 0,   // 0 = adaptive (uses max-observed as hindsight optimum)
		WindowSize:       100, // sliding window for recent average
	}))
	// Def.1 evidence-sensitivity: UCB exploration bonus for action space coverage
	explCfg := adaptive.DefaultExplorationConfig()
	explCfg.BetaDecay = "sqrt" // β(t) = β₀/√t for theoretical convergence guarantee
	controller.SetExplorationBonus(adaptive.NewExplorationBonus(explCfg))
	// §IV safety invariant: Lyapunov stability certificate
	controller.SetLyapunovMonitor(adaptive.NewLyapunovMonitor(adaptive.DefaultLyapunovConfig()))
	// Actor-Critic variance reduction: EMA baseline for advantage estimation
	controller.SetValueBaseline(adaptive.NewValueBaseline(adaptive.DefaultAdvantageConfig()))
	// §III-C liveness: epoch-boundary gating prevents mid-round parameter changes
	controller.SetEpochGate(adaptive.NewEpochGate(adaptive.DefaultEpochGateConfig()))
	// Def.1 non-degeneracy (ρ_evol > 0): entropy monitor detects premature convergence
	controller.SetEntropyMonitor(adaptive.NewEntropyMonitor(adaptive.DefaultEntropyConfig()))
	// §III-D P3: pre-argmax safety filter enforcing n_v >= 3f+1 per instance
	sf := adaptive.DefaultSafetyFilter()
	controller.SetSafetyFilter(&sf)
	if cfg.AdaptiveTracePath != "" {
		if writer, err := adaptive.NewJSONLTraceWriter(cfg.AdaptiveTracePath); err == nil {
			controller.SetTraceWriter(writer)
		}
	}
	return controller
}

func (r *evolvbftAdaptiveRuntime) recordOrderedOutput(out hotstuff.InstanceOutput) {
	r.mu.Lock()
	defer r.mu.Unlock()

	state := r.lastOrdered
	state.Rank = out.Rank
	state.IsNil = out.IsNil
	state.TransitionCount = len(out.EpochTransitions)
	if out.Block != nil {
		if out.Block.Height > state.Height {
			state.Height = out.Block.Height
		}
		if out.Block.LaneID > 0 || !out.IsNil {
			state.LaneID = out.Block.LaneID
		}
		if out.Block.ConfigID > 0 || !out.IsNil {
			state.ConfigID = out.Block.ConfigID
		}
	}
	for _, transition := range out.EpochTransitions {
		if transition != nil && transition.NewEpoch > state.ReconfigEpoch {
			state.ReconfigEpoch = transition.NewEpoch
			// Notify epoch gate to release pending actions at boundary
			if r.ctrl != nil {
				_ = r.ctrl.AdvanceEpoch(transition.NewEpoch)
			}
		}
	}
	r.lastOrdered = state
}

func (r *evolvbftAdaptiveRuntime) SetScenarioContext(ctx adaptive.ScenarioContext) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.scenario = ctx
}

func (r *evolvbftAdaptiveRuntime) GetScenarioContext() adaptive.ScenarioContext {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.scenario
}

// buildTrustSnapshotsWithBayesian uses the BayesianEstimator (§III-D) when available,
// falling back to leader reputation stats otherwise.
func (r *evolvbftAdaptiveRuntime) buildTrustSnapshotsWithBayesian(reputation *hotstuff.LeaderReputation) []adaptive.TrustSnapshot {
	if r.trustEst != nil && reputation != nil {
		stats := reputation.GetAllStats()
		now := time.Now()
		// Feed epoch events into Bayesian estimator using real fields (Eq. 5)
		for _, stat := range stats {
			total := stat.Successes + stat.Timeouts + stat.NilBlocks
			if total == 0 {
				continue
			}
			r.trustEst.ObserveEpoch(stat.LeaderID, trust.EpochEvent{
				Timeouts:      int(stat.Timeouts),
				Equivocations: int(stat.Equivocations),
				ViewChanges:   int(stat.ViewChangeInits),
				LatencyMs:     stat.AvgLatencyMs,
			})
			// Mark fresh observation for temporal decay tracker
			if r.trustDecay != nil {
				r.trustDecay.Touch(stat.LeaderID, now)
			}
		}
		// Build snapshots from Bayesian fault probabilities
		snapshots := make([]adaptive.TrustSnapshot, 0, len(stats))
		for _, stat := range stats {
			prob, ok := r.trustEst.FaultProbability(stat.LeaderID)
			if !ok {
				continue
			}
			total := stat.Successes + stat.Timeouts + stat.NilBlocks
			snap := adaptive.TrustSnapshot{
				NodeID:             stat.LeaderID,
				SampleCount:        total,
				SuccessRate:        1.0 - prob,
				FailureProbability: prob,
				ClaimBoundary:      adaptive.TrustSnapshotBoundary,
			}
			// Populate Eq. 5 five-dimensional trust feature vector
			if fv, fvOK := r.trustEst.Features(stat.LeaderID); fvOK {
				snap.TimeoutRate = fv[0]
				snap.EquivocationRate = fv[1]
				snap.ViewChangeRate = fv[2]
				snap.MeanLatency = fv[3]
				snap.StdLatency = fv[4]
			}
			snapshots = append(snapshots, snap)
		}
		if len(snapshots) > 0 {
			sort.Slice(snapshots, func(i, j int) bool {
				return snapshots[i].NodeID < snapshots[j].NodeID
			})
			// Publish local trust to cross-instance aggregator (§III-C).
			// In production, this report travels via GBC EntryTrust messages.
			if r.trustAgg != nil && len(r.engines) > 0 {
				for _, engine := range r.engines {
					if engine == nil {
						continue
					}
					var epoch uint64
					if vs := engine.GetCurrentValidatorSet(); vs != nil {
						epoch = vs.Epoch
					}
					report := trust.TrustReport{
						InstanceID: engine.GetInstanceID(),
						Epoch:      epoch,
						FaultProbs: make(map[uint64]float64, len(snapshots)),
					}
					for _, snap := range snapshots {
						report.FaultProbs[snap.NodeID] = snap.FailureProbability
					}
					r.trustAgg.Ingest(report)
				}
			}
			return snapshots
		}
	}
	// Fallback to leader reputation based snapshots
	return buildTrustSnapshots(reputation)
}

// faultProbThreshold is the probability above which a validator is counted as
// suspected-Byzantine for the purpose of FaultsEstimate (consistent with
// integration.PipelineConfig.FaultThreshold = 0.5).
const faultProbThreshold = 0.5

// estimateFaults derives the per-instance FaultsEstimate f by counting
// validators in the given set whose Bayesian fault probability > threshold.
// A probability of exactly 0.5 (σ(0)) means "no evidence" and is NOT counted.
// When the trust estimator is unavailable, returns 0 (safety filter falls
// back to f=1 minimum).
func (r *evolvbftAdaptiveRuntime) estimateFaults(valSet *types.ValidatorSet) int {
	if r.trustEst == nil || valSet == nil {
		return 0
	}
	now := time.Now()
	f := 0
	for nodeID := range valSet.Validators {
		prob, ok := r.trustEst.FaultProbability(nodeID)
		if !ok {
			continue
		}
		// Fuse local estimate with global cross-instance view (§III-C).
		if r.trustAgg != nil {
			prob = r.trustAgg.FusedFaultProb(nodeID, prob)
		}
		// Temporal decay: stale trust drifts toward uncertain baseline (§III-D).
		if r.trustDecay != nil {
			prob = r.trustDecay.Decay(nodeID, prob, now)
		}
		if prob > faultProbThreshold {
			f++
		}
	}
	// Cap at BFT maximum: f cannot exceed ⌊(n-1)/3⌋
	n := len(valSet.Validators)
	maxF := (n - 1) / 3
	if f > maxF {
		f = maxF
	}
	return f
}
