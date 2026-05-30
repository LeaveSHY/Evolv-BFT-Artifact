// Copyright 2024 Octopus Project
// Licensed under Apache License 2.0

package viewchange

import (
	"fmt"
	"sync"
	"time"

	"octopus-bft/octopus/types"
)

var logger = struct {
	Info  func(format string, args ...interface{})
	Error func(format string, args ...interface{})
}{
	Info:  func(format string, args ...interface{}) { fmt.Printf(format+"\n", args...) },
	Error: func(format string, args ...interface{}) { fmt.Printf("ERROR: "+format+"\n", args...) },
}

// tcCollector aggregates timeout votes for a single view.
type tcCollector struct {
	view      uint64
	epoch     uint64
	highestQC *types.QuorumCertificate
	voters    map[uint64][]byte // voterID -> signature
	done      bool
}

// ViewChangeManager manages view changes and TC (Timeout Certificate) aggregation.
// When a node's view timer expires, it broadcasts a TimeoutVote. This manager
// collects timeout votes from 2f+1 validators and forms a TimeoutCertificate,
// which enables the next leader to propose safely without the previous view's QC.
type ViewChangeManager struct {
	mu sync.RWMutex

	nodeID   uint64
	quorumFn func() uint64 // Returns the current quorum size

	// TC collectors indexed by view
	collectors map[uint64]*tcCollector

	// Highest TC formed
	highestTC *types.TimeoutCertificate

	// Timer
	viewChangeTimeout time.Duration
	timer             *time.Timer

	// Callbacks
	onTCFormed func(tc *types.TimeoutCertificate) // Called when a TC is formed

	isRunning bool
}

// VCM is shorthand
type VCM = ViewChangeManager

// NewViewChangeManager creates a new view change manager.
// quorumFn should return the current quorum size (2f+1).
func NewViewChangeManager(nodeID uint64, timeout time.Duration, quorumFn func() uint64) *VCM {
	return &VCM{
		nodeID:            nodeID,
		quorumFn:          quorumFn,
		collectors:        make(map[uint64]*tcCollector),
		viewChangeTimeout: timeout,
		isRunning:         false,
	}
}

// Start starts the view change manager.
func (v *VCM) Start() {
	v.mu.Lock()
	v.isRunning = true
	v.mu.Unlock()
	logger.Info("ViewChangeManager started for node %d, timeout: %v", v.nodeID, v.viewChangeTimeout)
}

// Stop stops the view change manager.
func (v *VCM) Stop() {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.isRunning = false
	if v.timer != nil {
		v.timer.Stop()
	}
}

// OnTCFormed sets the callback invoked when a TimeoutCertificate is formed.
func (v *VCM) OnTCFormed(callback func(tc *types.TimeoutCertificate)) {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.onTCFormed = callback
}

// HandleTimeoutVote processes an incoming timeout vote and attempts to form a TC.
// Returns the formed TC if quorum is reached, nil otherwise.
func (v *VCM) HandleTimeoutVote(tv *types.TimeoutVote) *types.TimeoutCertificate {
	if tv == nil {
		return nil
	}

	v.mu.Lock()
	defer v.mu.Unlock()

	collector, exists := v.collectors[tv.View]
	if !exists {
		collector = &tcCollector{
			view:   tv.View,
			epoch:  tv.Epoch,
			voters: make(map[uint64][]byte),
		}
		v.collectors[tv.View] = collector
	}

	if collector.done {
		return nil
	}
	if collector.epoch != tv.Epoch {
		return nil
	}

	// Deduplicate
	if _, voted := collector.voters[tv.VoterID]; voted {
		return nil
	}

	collector.voters[tv.VoterID] = tv.Signature

	// Track the highest QC among all timeout voters
	if tv.HighestQC != nil {
		if collector.highestQC == nil || tv.HighestQC.View > collector.highestQC.View {
			collector.highestQC = tv.HighestQC
		}
	}

	// Check if we have quorum
	quorum := v.quorumFn()
	if uint64(len(collector.voters)) < quorum {
		return nil
	}

	// Form TC
	tc := &types.TimeoutCertificate{
		View:       tv.View,
		Epoch:      tv.Epoch,
		ConfigID:   tv.ConfigID,
		Lane:       tv.Lane,
		HighestQC:  collector.highestQC,
		Signatures: make(map[uint64][]byte, len(collector.voters)),
		NumVoters:  len(collector.voters),
	}
	for id, sig := range collector.voters {
		tc.Signatures[id] = sig
	}
	collector.done = true

	// Update highest TC
	if v.highestTC == nil || tc.View > v.highestTC.View {
		v.highestTC = tc
	}

	logger.Info("TC formed for View %d with %d voters", tc.View, tc.NumVoters)

	if v.onTCFormed != nil {
		v.onTCFormed(tc)
	}

	return tc
}

// GetHighestTC returns the highest timeout certificate formed.
func (v *VCM) GetHighestTC() *types.TimeoutCertificate {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.highestTC
}

// GCCollectors removes stale collectors for views older than threshold.
func (v *VCM) GCCollectors(threshold uint64) {
	v.mu.Lock()
	defer v.mu.Unlock()
	for view := range v.collectors {
		if view < threshold {
			delete(v.collectors, view)
		}
	}
}

// Legacy API note:
// The repository previously exposed OnNewView/UpdateHighestExecuted/
// ResetViewChangeTimer for a pre-HotStuff compatibility path in node.go.
// Those hooks have been retired from active use. The authoritative runtime now
// advances views via TC formation and the HotStuff pacemaker. Do not reintroduce
// silent compatibility shims here; stale call sites should be migrated to the
// TC-driven path instead.
