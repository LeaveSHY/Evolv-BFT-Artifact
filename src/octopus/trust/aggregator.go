package trust

import (
	"math"
	"sync"
)

// TrustReport is a cross-instance trust snapshot published to GBC (EntryTrust).
// Each instance primary publishes its local fault probabilities at epoch boundaries.
// The paper's Limitation (i) defense: instances share trust so attack migration
// is detected by the receiving instance before the Byzantine node participates.
type TrustReport struct {
	InstanceID uint64             `json:"instance_id"`
	Epoch      uint64             `json:"epoch"`
	FaultProbs map[uint64]float64 `json:"fault_probs"` // nodeID → P(Byzantine)
}

// Aggregator fuses TrustReports from multiple instances into a global view.
// Implements the paper's cross-instance trust sharing: when a node migrates from
// instance A to instance B, B queries the aggregator for A's trust assessment.
//
// Fusion policy (Eq. trust-fusion):
//
//	f_global(node) = max(f_local(node), max over all instances { f_instance(node) })
//
// The max-policy is conservative: any instance flagging a node as suspicious
// immediately propagates to all other instances. This prevents attack migration
// (a node cannot escape its reputation by moving to a naive instance).
type Aggregator struct {
	mu sync.RWMutex
	// reports[instanceID][nodeID] = latest fault probability from that instance
	reports map[uint64]map[uint64]float64
	// epochs[instanceID] = latest epoch received from that instance
	epochs map[uint64]uint64
}

// NewAggregator creates an empty cross-instance trust aggregator.
func NewAggregator() *Aggregator {
	return &Aggregator{
		reports: make(map[uint64]map[uint64]float64),
		epochs:  make(map[uint64]uint64),
	}
}

// Ingest processes a TrustReport from a GBC entry. Only monotonically newer
// epochs are accepted (stale reports from older epochs are discarded).
func (a *Aggregator) Ingest(report TrustReport) bool {
	a.mu.Lock()
	defer a.mu.Unlock()

	// Monotonic epoch check: reject stale reports
	if existingEpoch, ok := a.epochs[report.InstanceID]; ok && report.Epoch <= existingEpoch {
		return false
	}
	a.epochs[report.InstanceID] = report.Epoch

	// Store the latest fault probabilities from this instance
	instMap := make(map[uint64]float64, len(report.FaultProbs))
	for nodeID, prob := range report.FaultProbs {
		instMap[nodeID] = math.Min(1.0, math.Max(0.0, prob))
	}
	a.reports[report.InstanceID] = instMap
	return true
}

// GlobalFaultProb returns the maximum fault probability observed for a node
// across all instances (Eq. trust-fusion: max-policy).
// The second return value indicates whether any instance has reported on this node.
func (a *Aggregator) GlobalFaultProb(nodeID uint64) (float64, bool) {
	a.mu.RLock()
	defer a.mu.RUnlock()

	maxProb := 0.0
	found := false
	for _, instMap := range a.reports {
		if prob, ok := instMap[nodeID]; ok {
			found = true
			if prob > maxProb {
				maxProb = prob
			}
		}
	}
	return maxProb, found
}

// GlobalFaultProbs returns the aggregated global fault probability map
// (max over all instances for each node).
func (a *Aggregator) GlobalFaultProbs() map[uint64]float64 {
	a.mu.RLock()
	defer a.mu.RUnlock()

	global := make(map[uint64]float64)
	for _, instMap := range a.reports {
		for nodeID, prob := range instMap {
			if prob > global[nodeID] {
				global[nodeID] = prob
			}
		}
	}
	return global
}

// FusedFaultProb combines a local fault probability with the global aggregated
// probability using max-policy (conservative fusion).
func (a *Aggregator) FusedFaultProb(nodeID uint64, localProb float64) float64 {
	globalProb, found := a.GlobalFaultProb(nodeID)
	if !found {
		return localProb
	}
	if globalProb > localProb {
		return globalProb
	}
	return localProb
}

// InstanceCount returns the number of instances that have reported trust data.
func (a *Aggregator) InstanceCount() int {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return len(a.reports)
}

// ReportingInstances returns the set of instance IDs that have contributed reports.
func (a *Aggregator) ReportingInstances() []uint64 {
	a.mu.RLock()
	defer a.mu.RUnlock()
	ids := make([]uint64, 0, len(a.reports))
	for id := range a.reports {
		ids = append(ids, id)
	}
	return ids
}
