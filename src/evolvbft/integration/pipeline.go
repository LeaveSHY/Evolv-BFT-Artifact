package integration

import (
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	"evolvbft/evolvbft/adaptive"
	"evolvbft/evolvbft/consensus/gbc"
	"evolvbft/evolvbft/trust"
)

// Pipeline implements the closed-loop feedback described in §III-E:
//
//	Engine commit → GBC publish (EntryQC)
//	                  ↓
//	GBC commit callback → Trust estimator update
//	                  ↓
//	Trust scores → Controller/SFAC decision
//	                  ↓
//	Reconfig action → GBC publish (EntryMembership)
//	                  ↓
//	GBC commit → Engine apply reconfiguration
//
// This is the glue layer that turns the independent modules into
// the full Evolv-BFT defense pipeline (Algorithm 3: Evolv-BFT).
type Pipeline struct {
	mu sync.RWMutex

	gbcLog    *gbc.Log
	gbcNodes  []*gbc.Node
	orderer   *gbc.Orderer
	trust     *trust.CombinedEstimator
	instances []InstanceHandle
	safety    adaptive.SafetyFilter
	logger    *log.Logger

	// Epoch tracking
	currentEpoch uint64
	epochStats   map[uint64]*EpochStats

	// Versioned adaptive config with reversible rollback (§III-D, App. B).
	// Lazily seeded from the instance set on first commit.
	versionChain *adaptive.VersionChain

	// Callbacks
	onReconfigDecision func(instanceID uint64, evict []uint64, admit []uint64)
	onTrustUpdate      func(epoch uint64, scores map[uint64]float64)

	// Config
	config PipelineConfig
}

// PipelineConfig configures the integration pipeline.
type PipelineConfig struct {
	NumInstances      int           // m: number of BFT instances
	NumAgents         int           // total agents across all instances
	FaultThreshold    float64       // threshold above which an agent is considered faulty
	ReconfigCooldown  time.Duration // minimum time between reconfigurations
	SafetyMarginDelta int           // δ_s: safety margin above 3f+1
	GBCCommitTimeout  time.Duration // max wait for GBC quorum commit before applying membership changes
}

// DefaultPipelineConfig returns standard pipeline configuration.
func DefaultPipelineConfig() PipelineConfig {
	return PipelineConfig{
		NumInstances:      4,
		NumAgents:         100,
		FaultThreshold:    0.5,
		ReconfigCooldown:  time.Second,
		SafetyMarginDelta: 1,
		GBCCommitTimeout:  2 * time.Second,
	}
}

// InstanceHandle represents a consensus instance in the pipeline.
type InstanceHandle struct {
	InstanceID     uint64
	ValidatorCount int
	FaultTolerance int // f_v for this instance
	Agents         []uint64
}

// EpochStats tracks per-epoch metrics for the pipeline.
type EpochStats struct {
	Epoch            uint64
	CommitsReceived  int
	TrustUpdates     int
	ReconfigActions  int
	SafetyMasked     int
	DetectionLatency float64
	StartTime        time.Time
	EndTime          time.Time
}

// TrustUpdatePayload is the JSON payload for EntryPolicyUpdate entries.
type TrustUpdatePayload struct {
	Epoch         uint64             `json:"epoch"`
	Scores        map[uint64]float64 `json:"scores"`
	Evictions     []uint64           `json:"evictions,omitempty"`
	Admissions    []uint64           `json:"admissions,omitempty"`
	PolicyWeights *[5]float64        `json:"policy_weights,omitempty"`
}

// ReconfigPayload is the JSON payload for EntryMembership entries.
type ReconfigPayload struct {
	Epoch      uint64   `json:"epoch"`
	InstanceID uint64   `json:"instance_id"`
	Evictions  []uint64 `json:"evictions"`
	Admissions []uint64 `json:"admissions"`
	NAfter     int      `json:"n_after"`
	FAfter     int      `json:"f_after"`
}

// ConfigVersionPayload is the JSON payload for EntryConfigVersion entries.
// It records one node in the append-only configuration chain (§III-D, App. B).
type ConfigVersionPayload struct {
	Epoch       uint64                 `json:"epoch"`
	VersionID   uint64                 `json:"version_id"`
	ParentID    uint64                 `json:"parent_id"`
	Params      adaptive.ConfigParams  `json:"params"`
	PhiAtCommit int                    `json:"phi_at_commit"`
	Status      adaptive.VersionStatus `json:"status"`
}

// NewPipeline creates the integration pipeline.
func NewPipeline(config PipelineConfig, gbcLog *gbc.Log, trustEstimator *trust.CombinedEstimator) *Pipeline {
	return &Pipeline{
		gbcLog:     gbcLog,
		trust:      trustEstimator,
		config:     config,
		epochStats: make(map[uint64]*EpochStats),
		logger:     log.Default(),
		safety: adaptive.SafetyFilter{
			DeltaS:          config.SafetyMarginDelta,
			GlobalBudgetMin: 4 + config.SafetyMarginDelta,
		},
	}
}

// SetInstances configures the BFT instances managed by the pipeline.
func (p *Pipeline) SetInstances(instances []InstanceHandle) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.instances = append([]InstanceHandle(nil), instances...)
}

// SetOrderer attaches the GBC orderer.
func (p *Pipeline) SetOrderer(orderer *gbc.Orderer) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.orderer = orderer
}

// SetGBCNodes attaches the GBC protocol nodes.
func (p *Pipeline) SetGBCNodes(nodes []*gbc.Node) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.gbcNodes = nodes
}

// OnReconfigDecision registers a callback for reconfiguration decisions.
func (p *Pipeline) OnReconfigDecision(fn func(instanceID uint64, evict []uint64, admit []uint64)) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.onReconfigDecision = fn
}

// OnTrustUpdate registers a callback for trust score updates.
func (p *Pipeline) OnTrustUpdate(fn func(epoch uint64, scores map[uint64]float64)) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.onTrustUpdate = fn
}

// ProcessEpoch runs one complete epoch of the Evolv-BFT defense pipeline
// (Algorithm 3: Evolv-BFT). This is the main entry point for the closed loop.
//
// Steps:
//  1. Collect consensus metrics from all instances (trust features)
//  2. Update trust estimator with new observations
//  3. Compute fault probabilities for all agents
//  4. Apply safety-filtered reconfiguration decisions
//  5. Publish trust update and reconfig to GBC
func (p *Pipeline) ProcessEpoch(epoch uint64, agentMetrics map[uint64]trust.EpochEvent) (*EpochStats, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	stats := &EpochStats{
		Epoch:     epoch,
		StartTime: time.Now(),
	}

	// Step 1: Feed observations to trust estimator
	for nodeID, event := range agentMetrics {
		p.trust.ObserveEpoch(nodeID, event)
		stats.TrustUpdates++
	}

	// Step 2: Compute fault probabilities for all agents
	scores := make(map[uint64]float64)
	for nodeID := range agentMetrics {
		if prob, ok := p.trust.FaultProbability(nodeID); ok {
			scores[nodeID] = prob
		}
	}

	// Step 3: Identify agents exceeding fault threshold
	var candidates []uint64
	for nodeID, prob := range scores {
		if prob >= p.config.FaultThreshold {
			candidates = append(candidates, nodeID)
		}
	}

	// Step 4: Safety-filtered reconfiguration per instance (Algorithm 5)
	for i := range p.instances {
		inst := &p.instances[i]
		var evictions []uint64

		for _, candidate := range candidates {
			if containsAgent(inst.Agents, candidate) {
				// Use SafetyFilter with δ_s margin (Algorithm 5 line 5)
				state := adaptive.InstanceState{
					InstanceID:     inst.InstanceID,
					ValidatorCount: inst.ValidatorCount,
					PendingEvicts:  len(evictions) + 1,
					PendingAdmits:  0,
				}
				if p.safety.CheckQuorumInvariant(state) {
					evictions = append(evictions, candidate)
				} else {
					stats.SafetyMasked++
				}
			}
		}

		// Cross-instance coupled constraint check
		if len(evictions) > 0 {
			allStates := make([]adaptive.InstanceState, 0, len(p.instances))
			for j := range p.instances {
				evict := 0
				if j == i {
					evict = len(evictions)
				}
				allStates = append(allStates, adaptive.InstanceState{
					InstanceID:     p.instances[j].InstanceID,
					ValidatorCount: p.instances[j].ValidatorCount,
					PendingEvicts:  evict,
				})
			}
			if !p.safety.CheckCoupledConstraint(allStates) {
				stats.SafetyMasked += len(evictions)
				evictions = nil
			}
		}

		if len(evictions) > 0 {
			stats.ReconfigActions += len(evictions)

			// Publish membership change to GBC
			payload, _ := json.Marshal(ReconfigPayload{
				Epoch:      epoch,
				InstanceID: inst.InstanceID,
				Evictions:  evictions,
				NAfter:     inst.ValidatorCount - len(evictions),
				FAfter:     (inst.ValidatorCount - len(evictions) - 1) / 3,
			})

			if err := p.publishToGBC(gbc.EntryMembership, payload); err != nil {
				// GBC publish failed — do NOT apply local reconfiguration.
				// This enforces commit-then-apply semantics: only mutate local
				// state after GBC has durably logged the reconfiguration.
				p.logger.Printf("WARN: GBC publish failed for instance %d reconfig, rolling back: %v",
					inst.InstanceID, err)
				continue
			}

			// Notify callback
			if p.onReconfigDecision != nil {
				p.onReconfigDecision(inst.InstanceID, evictions, nil)
			}

			// Update instance state
			inst.ValidatorCount -= len(evictions)
			inst.FaultTolerance = (inst.ValidatorCount - 1) / 3
			inst.Agents = removeAgents(inst.Agents, evictions)
		}
	}

	// Step 4b: Record the post-reconfig effective config as a committed version.
	// This anchors the configuration in the append-only version chain so a later
	// safety or liveness regression can revert to a proven-safe ancestor (App. B).
	// Commit-then-apply: append to the chain only after the GBC publish succeeds.
	if stats.ReconfigActions > 0 {
		p.ensureVersionChainLocked()
		params := p.currentConfigParamsLocked()
		phi := p.computePhiLocked()
		parent := p.versionChain.Latest()
		payload, _ := json.Marshal(ConfigVersionPayload{
			Epoch:       epoch,
			VersionID:   parent.VersionID + 1,
			ParentID:    parent.VersionID,
			Params:      params,
			PhiAtCommit: phi,
			Status:      adaptive.StatusCommitted,
		})
		if err := p.publishToGBC(gbc.EntryConfigVersion, payload); err != nil {
			// GBC publish failed — do NOT append the version locally.
			p.logger.Printf("WARN: GBC publish failed for config version epoch %d: %v", epoch, err)
		} else {
			p.versionChain.AppendCommitted(params, phi, int(p.gbcLog.Height()))
		}
	}

	// Step 5: Publish trust update to GBC
	trustPayload, _ := json.Marshal(TrustUpdatePayload{
		Epoch:  epoch,
		Scores: scores,
	})
	if err := p.publishToGBC(gbc.EntryPolicyUpdate, trustPayload); err != nil {
		// Trust update publish failed — log warning but proceed (non-critical).
		// Trust scores are still applied locally; GBC will catch up on retry.
		p.logger.Printf("WARN: GBC publish failed for trust update epoch %d: %v", epoch, err)
	}

	// Notify trust callback
	if p.onTrustUpdate != nil {
		p.onTrustUpdate(epoch, scores)
	}

	stats.EndTime = time.Now()
	p.epochStats[epoch] = stats
	p.currentEpoch = epoch + 1
	return stats, nil
}

// GetEpochStats returns the stats for a given epoch.
func (p *Pipeline) GetEpochStats(epoch uint64) (*EpochStats, bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	s, ok := p.epochStats[epoch]
	return s, ok
}

// ensureVersionChainLocked lazily seeds the version chain from the current
// instance set. The genesis records the live config and its safety margin Phi
// (Assumption A2: the bootstrap config is safe). Caller holds p.mu.
func (p *Pipeline) ensureVersionChainLocked() {
	if p.versionChain != nil {
		return
	}
	p.versionChain = adaptive.NewVersionChain(
		p.currentConfigParamsLocked(),
		p.computePhiLocked(),
		int(p.gbcLog.Height()),
	)
}

// currentConfigParamsLocked snapshots the effective configuration parameters.
// Caller holds p.mu. ValidatorCount/FaultBound summarize the instance set so a
// restored ancestor pins the quorum invariant n >= 3f+1+delta_s.
func (p *Pipeline) currentConfigParamsLocked() adaptive.ConfigParams {
	totalValidators := 0
	totalFaults := 0
	snapshot := make([]adaptive.InstanceConfig, 0, len(p.instances))
	for i := range p.instances {
		totalValidators += p.instances[i].ValidatorCount
		totalFaults += p.instances[i].FaultTolerance
		snapshot = append(snapshot, adaptive.InstanceConfig{
			InstanceID:     p.instances[i].InstanceID,
			ValidatorCount: p.instances[i].ValidatorCount,
			FaultBound:     p.instances[i].FaultTolerance,
		})
	}
	return adaptive.ConfigParams{
		ValidatorCount: totalValidators,
		FaultBound:     totalFaults,
		InstanceCount:  len(p.instances),
		Instances:      snapshot,
	}
}

// computePhiLocked evaluates the joint safety invariant Phi over the live
// instance set and GBC (Definition 3, via adaptive.ComputePhi). Caller holds
// p.mu. The GBC member count is read from the beacon log, which is the
// authoritative committee enforcing attestation quorum.
func (p *Pipeline) computePhiLocked() int {
	states := make([]adaptive.InstanceState, 0, len(p.instances))
	for i := range p.instances {
		states = append(states, adaptive.InstanceState{
			InstanceID:     p.instances[i].InstanceID,
			ValidatorCount: p.instances[i].ValidatorCount,
			FaultsEstimate: p.instances[i].FaultTolerance,
		})
	}
	gbcMembers := p.gbcLog.NumMembers()
	gbcFaults := 0
	if gbcMembers > 0 {
		gbcFaults = (gbcMembers - 1) / 3
	}
	return adaptive.ComputePhi(states, gbcMembers, gbcFaults)
}

// EvaluateAndRollback applies the third defense gate (§III-D, App. B).
// It reads objective, on-chain verifiable signals and, when a trigger fires,
// reverts the effective configuration to the nearest proven-safe ancestor.
//
// Commit-then-apply: the ROLLEDBACK version is published to GBC first; only on
// a durable quorum commit does the pipeline append it to the chain and restore
// the safe parameters. A failed publish leaves the effective config unchanged.
//
// Returns (true, target, nil) when a rollback is committed, (false, zero, nil)
// when no trigger fires or no safe ancestor exists, and a non-nil error only on
// a GBC publish failure during an attempted rollback.
func (p *Pipeline) EvaluateAndRollback(epoch uint64, observed adaptive.ObservedSafety) (bool, adaptive.ConfigVersion, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.ensureVersionChainLocked()
	target, ok := adaptive.EvaluateRollback(p.versionChain, observed, p.safety.DeltaS)
	if !ok {
		return false, adaptive.ConfigVersion{}, nil
	}

	parent := p.versionChain.Latest()
	payload, _ := json.Marshal(ConfigVersionPayload{
		Epoch:       epoch,
		VersionID:   parent.VersionID + 1,
		ParentID:    parent.VersionID,
		Params:      target.Params,
		PhiAtCommit: target.PhiAtCommit,
		Status:      adaptive.StatusRolledBack,
	})
	if err := p.publishToGBC(gbc.EntryConfigVersion, payload); err != nil {
		// Publish failed — do NOT apply rollback. Effective config is unchanged.
		p.logger.Printf("WARN: GBC publish failed for rollback epoch %d: %v", epoch, err)
		return false, adaptive.ConfigVersion{}, err
	}

	applied := p.versionChain.AppendRollback(target, int(p.gbcLog.Height()))
	p.restoreInstancesLocked(target.Params)
	p.logger.Printf("ROLLBACK epoch %d: reverted to safe ancestor v%d (phi=%d) as v%d",
		epoch, target.VersionID, target.PhiAtCommit, applied.VersionID)
	return true, applied, nil
}

// restoreInstancesLocked applies a version's per-instance snapshot back onto the
// live instance set, pinning each instance quorum invariant n_v >= 3f_v+1+delta_s.
// Caller holds p.mu. Instances absent from the snapshot are left unchanged.
func (p *Pipeline) restoreInstancesLocked(params adaptive.ConfigParams) {
	byID := make(map[uint64]adaptive.InstanceConfig, len(params.Instances))
	for _, ic := range params.Instances {
		byID[ic.InstanceID] = ic
	}
	for i := range p.instances {
		if ic, ok := byID[p.instances[i].InstanceID]; ok {
			p.instances[i].ValidatorCount = ic.ValidatorCount
			p.instances[i].FaultTolerance = ic.FaultBound
		}
	}
}

// VersionChainSnapshot returns a copy of the current configuration version
// lineage for inspection and testing. Returns nil if no version is recorded.
func (p *Pipeline) VersionChainSnapshot() []adaptive.ConfigVersion {
	p.mu.RLock()
	defer p.mu.RUnlock()
	if p.versionChain == nil {
		return nil
	}
	return p.versionChain.Versions()
}

// CurrentEpoch returns the current epoch.
func (p *Pipeline) CurrentEpoch() uint64 {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.currentEpoch
}

// Summary returns aggregate pipeline statistics.
func (p *Pipeline) Summary() PipelineSummary {
	p.mu.RLock()
	defer p.mu.RUnlock()

	var totalReconfigs, totalSafetyMasked, totalTrustUpdates int
	for _, s := range p.epochStats {
		totalReconfigs += s.ReconfigActions
		totalSafetyMasked += s.SafetyMasked
		totalTrustUpdates += s.TrustUpdates
	}
	return PipelineSummary{
		EpochsProcessed:   len(p.epochStats),
		TotalReconfigs:    totalReconfigs,
		TotalSafetyMasked: totalSafetyMasked,
		TotalTrustUpdates: totalTrustUpdates,
	}
}

// PipelineSummary aggregates pipeline statistics.
type PipelineSummary struct {
	EpochsProcessed   int
	TotalReconfigs    int
	TotalSafetyMasked int
	TotalTrustUpdates int
}

func (s PipelineSummary) String() string {
	return fmt.Sprintf("Pipeline: %d epochs, %d reconfigs, %d safety-masked, %d trust updates",
		s.EpochsProcessed, s.TotalReconfigs, s.TotalSafetyMasked, s.TotalTrustUpdates)
}

func containsAgent(agents []uint64, target uint64) bool {
	for _, a := range agents {
		if a == target {
			return true
		}
	}
	return false
}

func removeAgents(agents []uint64, toRemove []uint64) []uint64 {
	removeSet := make(map[uint64]bool, len(toRemove))
	for _, r := range toRemove {
		removeSet[r] = true
	}
	var result []uint64
	for _, a := range agents {
		if !removeSet[a] {
			result = append(result, a)
		}
	}
	return result
}

// publishToGBC publishes an entry via the GBC Node protocol (Propose-Attest-Commit)
// when distributed nodes are configured, or falls back to local log write.
// This ensures integration tests can run without full network setup, while
// production deployments go through the distributed consensus path.
func (p *Pipeline) publishToGBC(entryType gbc.EntryType, payload []byte) error {
	// If GBC nodes are configured, route through the distributed protocol.
	if len(p.gbcNodes) > 0 {
		// Try each node (round-robin proposer will accept from exactly one).
		for _, node := range p.gbcNodes {
			_, _, err := node.ProposeAndWait(entryType, payload, p.config.GBCCommitTimeout)
			if err == nil {
				return nil // success via distributed quorum commit
			}
			if gbc.IsNotProposer(err) {
				continue // try next node (wrong proposer for this height)
			}
			return err // real error
		}
		return fmt.Errorf("gbc: no node accepted proposal (all returned not-proposer)")
	}

	// Fallback: direct local log append (unit test / single-node mode).
	return p.gbcLog.Publish(gbc.Entry{
		Height:  p.gbcLog.Height(),
		Type:    entryType,
		Payload: payload,
	})
}
