// Copyright 2024 Octopus Project
// Licensed under Apache License 2.0

package pacemaker

import (
	"fmt"
	"sync"
	"time"
)

var logger = struct {
	Info func(format string, args ...interface{})
}{
	Info: func(format string, args ...interface{}) { fmt.Printf(format+"\n", args...) },
}

// LeaderSelector is a function that selects a leader for a given view.
// When set, it replaces the default round-robin selection with VRF/beacon-
// derived leader selection. The function receives the view number and must
// return a deterministic leader ID (all honest nodes must agree).
type LeaderSelector func(view uint64) uint64

// Pacemaker manages views and leader election
type Pacemaker struct {
	mu sync.RWMutex

	currentView uint64
	validators  []uint64 // List of validator IDs
	laneID      uint64   // Lane offset for round-robin diversity

	// Dynamic leader selection: when set, GetLeader uses this function
	// instead of static round-robin. Nil = fallback to round-robin.
	leaderSelector LeaderSelector

	// Timeouts
	viewTimeout time.Duration
	timer       *time.Timer

	// Channels
	timeoutChan chan uint64
}

// NewPacemaker creates a new pacemaker with laneID=0 (default for tests).
func NewPacemaker(validators []uint64, timeoutMs int64) *Pacemaker {
	return NewPacemakerWithLane(validators, timeoutMs, 0)
}

// NewPacemakerWithLane creates a new pacemaker bound to a specific lane.
// The laneID is used as an offset in round-robin leader selection so that
// different lanes elect different leaders even without a random beacon.
func NewPacemakerWithLane(validators []uint64, timeoutMs int64, laneID uint64) *Pacemaker {
	pm := &Pacemaker{
		currentView: 1,
		validators:  validators,
		laneID:      laneID,
		viewTimeout: time.Duration(timeoutMs) * time.Millisecond,
		timeoutChan: make(chan uint64, 10),
	}

	return pm
}

// Start starts the pacemaker timer
func (pm *Pacemaker) Start() {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	pm.startTimer()
	logger.Info("Pacemaker started at View %d", pm.currentView)
}

func (pm *Pacemaker) startTimer() {
	if pm.timer != nil {
		pm.timer.Stop()
	}

	pm.timer = time.AfterFunc(pm.viewTimeout, func() {
		pm.mu.Lock()
		view := pm.currentView
		pm.mu.Unlock()
		pm.timeoutChan <- view
	})
}

// GetCurrentView returns the current view
func (pm *Pacemaker) GetCurrentView() uint64 {
	pm.mu.RLock()
	defer pm.mu.RUnlock()
	return pm.currentView
}

// GetLeader returns the leader for a given view.
// If a LeaderSelector is set (VRF/beacon-based), it is used for dynamic
// leader selection. Otherwise, falls back to static round-robin.
func (pm *Pacemaker) GetLeader(view uint64) uint64 {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	if len(pm.validators) == 0 {
		return 0
	}
	if view == 0 {
		view = 1
	}

	// Dynamic leader selection: use VRF/beacon-derived selector
	if pm.leaderSelector != nil {
		return pm.leaderSelector(view)
	}

	// Fallback: static round-robin with lane offset for multi-leader diversity.
	// Different lanes get different leaders even without beacon.
	index := (view - 1 + pm.laneID) % uint64(len(pm.validators))
	return pm.validators[index]
}

// AdvanceView moves to the next view
func (pm *Pacemaker) AdvanceView(qcView uint64) bool {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if qcView >= pm.currentView {
		pm.currentView = qcView + 1
		pm.startTimer()
		logger.Info("Advanced to View %d", pm.currentView)
		return true
	}
	return false
}

// TimeoutChan returns the timeout channel
func (pm *Pacemaker) TimeoutChan() <-chan uint64 {
	return pm.timeoutChan
}

// SetTimeout dynamically adjusts the view timeout duration.
// This is used by the engine to wire in adaptive timeouts from
// LeaderReputation: when a leader is known to be slow, the pacemaker
// should wait longer; when the network is healthy, shorter timeouts
// improve latency.
func (pm *Pacemaker) SetTimeout(d time.Duration) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	if d > 0 {
		pm.viewTimeout = d
	}
}

// UpdateValidators replaces the validator set used for leader selection.
// Called during epoch transitions or dynamic membership changes.
func (pm *Pacemaker) UpdateValidators(validators []uint64) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	pm.validators = validators
}

// SetLeaderSelector sets the dynamic leader selection function.
// When set, GetLeader() uses this function instead of static round-robin.
// Pass nil to revert to round-robin.
func (pm *Pacemaker) SetLeaderSelector(fn LeaderSelector) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	pm.leaderSelector = fn
}
