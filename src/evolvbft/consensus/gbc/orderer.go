// Copyright 2024 Evolv-BFT Project
// Licensed under Apache License 2.0

package gbc

import (
	"encoding/json"
	"fmt"
	"sort"
	"sync"
)

// Orderer implements cross-instance global ordering on top of the GBC log.
// It collects committed block metadata from m instances and produces a total
// order by (epoch, rank, instanceID), matching the paper's description of
// GBC as the global ordering backbone (§III-C).
//
// The ordering protocol:
//  1. Each instance primary publishes its committed block QC to the GBC.
//  2. The Orderer collects outputs within an epoch window.
//  3. When all m instances have contributed, the Orderer produces a globally
//     ordered sequence and publishes an epoch checkpoint to the GBC.
//
// This enforces G2 (common prefix agreement) at the cross-instance level.
type Orderer struct {
	mu            sync.Mutex
	log           *Log
	numInstances  int
	currentEpoch  uint64
	epochOutputs  map[uint64][]InstanceCommit // epoch -> collected commits
	onGlobalOrder func(epoch uint64, ordered []InstanceCommit)
}

// InstanceCommit represents a committed block from one BFT instance,
// received by the GBC for global ordering.
type InstanceCommit struct {
	InstanceID  uint64 `json:"instance_id"`
	LocalHeight uint64 `json:"local_height"`
	Rank        uint64 `json:"rank"`
	Epoch       uint64 `json:"epoch"`
	BlockHash   []byte `json:"block_hash"`
}

// NewOrderer creates a GBC orderer for m instances.
func NewOrderer(log *Log, numInstances int) *Orderer {
	return &Orderer{
		log:          log,
		numInstances: numInstances,
		epochOutputs: make(map[uint64][]InstanceCommit),
	}
}

// OnGlobalOrder registers a callback invoked when a complete epoch is globally ordered.
func (o *Orderer) OnGlobalOrder(fn func(epoch uint64, ordered []InstanceCommit)) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.onGlobalOrder = fn
}

// SubmitCommit receives a committed block from an instance and adds it to the
// current epoch's collection. If all m instances have contributed for this epoch,
// the orderer produces the global order and publishes an epoch checkpoint.
func (o *Orderer) SubmitCommit(commit InstanceCommit) error {
	o.mu.Lock()
	defer o.mu.Unlock()

	epoch := commit.Epoch
	outputs := o.epochOutputs[epoch]

	// Dedup: don't accept duplicate commits from the same instance in the same epoch
	for _, existing := range outputs {
		if existing.InstanceID == commit.InstanceID && existing.LocalHeight == commit.LocalHeight {
			return nil // already received
		}
	}

	outputs = append(outputs, commit)
	o.epochOutputs[epoch] = outputs

	// Check if we have at least one commit from each instance for this epoch
	instanceSet := make(map[uint64]bool, o.numInstances)
	for _, c := range outputs {
		instanceSet[c.InstanceID] = true
	}

	if len(instanceSet) < o.numInstances {
		return nil // still waiting for more instances
	}

	// All instances have contributed — produce global order
	ordered := o.produceGlobalOrder(outputs)

	// Publish ordered sequence as a GBC checkpoint entry
	if err := o.publishEpochCheckpoint(epoch, ordered); err != nil {
		return fmt.Errorf("gbc orderer: publish checkpoint: %w", err)
	}

	// Advance epoch
	if epoch >= o.currentEpoch {
		o.currentEpoch = epoch + 1
	}

	// Invoke callback
	if o.onGlobalOrder != nil {
		o.onGlobalOrder(epoch, ordered)
	}

	// Clean up old epoch data (keep only current and next)
	for e := range o.epochOutputs {
		if e+2 <= o.currentEpoch {
			delete(o.epochOutputs, e)
		}
	}

	return nil
}

// produceGlobalOrder sorts the epoch's commits into the canonical total order:
// primary sort by rank (ascending), secondary by instanceID (ascending).
// This matches the paper's interleaved ordering where blocks from different
// instances are globally ordered by their consensus rank.
func (o *Orderer) produceGlobalOrder(commits []InstanceCommit) []InstanceCommit {
	ordered := make([]InstanceCommit, len(commits))
	copy(ordered, commits)
	sort.Slice(ordered, func(i, j int) bool {
		if ordered[i].Rank != ordered[j].Rank {
			return ordered[i].Rank < ordered[j].Rank
		}
		return ordered[i].InstanceID < ordered[j].InstanceID
	})
	return ordered
}

// publishEpochCheckpoint publishes the globally ordered epoch as a GBC checkpoint entry.
func (o *Orderer) publishEpochCheckpoint(epoch uint64, ordered []InstanceCommit) error {
	payload, err := json.Marshal(struct {
		Epoch   uint64           `json:"epoch"`
		Ordered []InstanceCommit `json:"ordered"`
	}{
		Epoch:   epoch,
		Ordered: ordered,
	})
	if err != nil {
		return fmt.Errorf("marshal epoch checkpoint: %w", err)
	}

	return o.log.Publish(Entry{
		Height:  o.log.Height(),
		Type:    EntryCheckpoint,
		Payload: payload,
	})
}

// CurrentEpoch returns the current epoch being collected.
func (o *Orderer) CurrentEpoch() uint64 {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.currentEpoch
}

// PendingCount returns the number of instance commits received for the given epoch.
func (o *Orderer) PendingCount(epoch uint64) int {
	o.mu.Lock()
	defer o.mu.Unlock()
	return len(o.epochOutputs[epoch])
}
