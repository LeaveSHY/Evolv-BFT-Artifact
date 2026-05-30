package integration

import (
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	"octopus-bft/octopus/adaptive"
	"octopus-bft/octopus/consensus/gbc"
	"octopus-bft/octopus/trust"
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
// the full Octopus defense pipeline (Algorithm 3: Octopus).
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

// ProcessEpoch runs one complete epoch of the Octopus defense pipeline
// (Algorithm 3: Octopus). This is the main entry point for the closed loop.
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
