// Copyright 2024 Octopus Project
// Licensed under Apache License 2.0

package membership

import (
	"bytes"
	"fmt"
	"sort"
	"sync"

	"octopus-bft/octopus/types"
)

var logger = struct {
	Info func(format string, args ...interface{})
}{
	Info: func(format string, args ...interface{}) { fmt.Printf(format+"\n", args...) },
}

// MembershipManager handles dynamic membership changes
type MembershipManager struct {
	mu sync.RWMutex

	// Current configuration
	currentConfig *types.Configuration

	// Pending requests
	pendingJoins  map[uint64]*types.JoinRequest
	pendingLeaves map[uint64]*types.LeaveRequest

	// Configuration history
	configHistory []*types.Configuration

	latestEvent *ConfigChangeEvent
}

type ConfigChangeEvent struct {
	OldEpoch   uint64
	NewEpoch   uint64
	Added      []uint64
	Removed    []uint64
	QuorumSize uint64
	ConfigHash []byte
}

// NewMembershipManager creates a new membership manager
func NewMembershipManager(initialValidators map[uint64]*types.Validator) *MembershipManager {
	validators := make(map[uint64]*types.Validator, len(initialValidators))
	for id, v := range initialValidators {
		if v == nil {
			continue
		}
		copyVal := *v
		validators[id] = &copyVal
	}
	initialConfig := &types.Configuration{
		ID:         1,
		Validators: validators,
		QuorumSize: uint64((2*len(validators))/3 + 1),
	}
	mm := &MembershipManager{
		currentConfig: initialConfig,
		pendingJoins:  make(map[uint64]*types.JoinRequest),
		pendingLeaves: make(map[uint64]*types.LeaveRequest),
		configHistory: []*types.Configuration{initialConfig},
	}

	return mm
}

// GetCurrentConfig returns the current configuration
func (mm *MembershipManager) GetCurrentConfig() *types.Configuration {
	mm.mu.RLock()
	defer mm.mu.RUnlock()
	return mm.cloneConfig(mm.currentConfig)
}

func (mm *MembershipManager) GetConfigHistory() []*types.Configuration {
	mm.mu.RLock()
	defer mm.mu.RUnlock()
	result := make([]*types.Configuration, len(mm.configHistory))
	for i, config := range mm.configHistory {
		result[i] = mm.cloneConfig(config)
	}
	return result
}

// SubmitJoinRequest submits a join request
func (mm *MembershipManager) SubmitJoinRequest(req *types.JoinRequest) error {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	if _, exists := mm.currentConfig.Validators[req.ID]; exists {
		return nil
	}
	if _, exists := mm.pendingJoins[req.ID]; exists {
		return nil
	}

	mm.pendingJoins[req.ID] = req
	logger.Info("Join request submitted for node %d", req.ID)
	return nil
}

// SubmitLeaveRequest submits a leave request
func (mm *MembershipManager) SubmitLeaveRequest(req *types.LeaveRequest) error {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	if _, exists := mm.currentConfig.Validators[req.ID]; !exists {
		return nil
	}
	if _, exists := mm.pendingLeaves[req.ID]; exists {
		return nil
	}

	mm.pendingLeaves[req.ID] = req
	logger.Info("Leave request submitted for node %d", req.ID)
	return nil
}

// ApplyMembershipChanges applies confirmed membership changes
func (mm *MembershipManager) ApplyMembershipChanges(block *types.Block) error {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	current := mm.cloneConfig(mm.currentConfig)
	added := make([]uint64, 0)
	removed := make([]uint64, 0)

	for _, key := range block.GetJoinRequests() {
		for id, req := range mm.pendingJoins {
			if string(req.PublicKey) == string(key) {
				if _, exists := current.Validators[id]; exists {
					delete(mm.pendingJoins, id)
					continue
				}
				power := req.Power
				if power == 0 {
					power = 1
				}
				current.Validators[id] = &types.Validator{
					ID:        req.ID,
					PublicKey: req.PublicKey,
					Power:     power,
					IsActive:  true,
				}
				delete(mm.pendingJoins, id)
				added = append(added, id)
				logger.Info("Applied join for node %d", id)
			}
		}
	}

	for _, key := range block.GetLeaveRequests() {
		for id, req := range mm.pendingLeaves {
			if string(req.PublicKey) == string(key) {
				if _, exists := current.Validators[id]; exists {
					delete(current.Validators, id)
					removed = append(removed, id)
				}
				delete(mm.pendingLeaves, id)
				logger.Info("Applied leave for node %d", id)
			}
		}
	}

	if len(added) == 0 && len(removed) == 0 {
		return nil
	}
	current.QuorumSize = uint64((2*len(current.Validators))/3 + 1)
	current.ID = mm.currentConfig.ID + 1
	mm.currentConfig = current
	mm.configHistory = append(mm.configHistory, mm.cloneConfig(current))
	mm.latestEvent = &ConfigChangeEvent{
		OldEpoch:   current.ID - 1,
		NewEpoch:   current.ID,
		Added:      append([]uint64(nil), added...),
		Removed:    append([]uint64(nil), removed...),
		QuorumSize: current.QuorumSize,
		ConfigHash: current.Hash(),
	}
	return nil
}

// GetValidatorCount returns the number of validators
func (mm *MembershipManager) GetValidatorCount() uint64 {
	mm.mu.RLock()
	defer mm.mu.RUnlock()
	return uint64(len(mm.currentConfig.Validators))
}

func (mm *MembershipManager) ApplyReconfigData(data *types.ReconfigData) (*types.Configuration, *ConfigChangeEvent, bool, error) {
	if data == nil {
		return nil, nil, false, nil
	}
	mm.mu.Lock()
	defer mm.mu.Unlock()

	current := mm.cloneConfig(mm.currentConfig)
	oldEpoch := current.ID
	changed := false
	added := make([]uint64, 0)
	removed := make([]uint64, 0)

	switch data.Type {
	case types.ReconfigJoin:
		if _, exists := current.Validators[data.NodeID]; !exists {
			power := data.Power
			if power == 0 {
				power = 1
			}
			current.Validators[data.NodeID] = &types.Validator{
				ID:           data.NodeID,
				PublicKey:    data.PublicKey,
				VRFPublicKey: append([]byte(nil), data.VRFPublicKey...),
				Power:        power,
				IsActive:     true,
			}
			changed = true
			added = append(added, data.NodeID)
		}
	case types.ReconfigLeave, types.ReconfigAutoLeave:
		if _, exists := current.Validators[data.NodeID]; exists {
			delete(current.Validators, data.NodeID)
			changed = true
			removed = append(removed, data.NodeID)
		}
	}

	if !changed {
		return mm.cloneConfig(mm.currentConfig), mm.latestEvent, false, nil
	}
	current.QuorumSize = uint64((2*len(current.Validators))/3 + 1)
	current.ID = oldEpoch + 1
	mm.currentConfig = current
	mm.configHistory = append(mm.configHistory, mm.cloneConfig(current))
	event := &ConfigChangeEvent{
		OldEpoch:   oldEpoch,
		NewEpoch:   current.ID,
		Added:      added,
		Removed:    removed,
		QuorumSize: current.QuorumSize,
		ConfigHash: current.Hash(),
	}
	mm.latestEvent = event
	return mm.cloneConfig(current), event, true, nil
}

func (mm *MembershipManager) InstallValidatorSet(valSet *types.ValidatorSet) (*types.Configuration, *ConfigChangeEvent, bool, error) {
	return mm.InstallValidatorSetFromTransitions(valSet, nil)
}

func (mm *MembershipManager) InstallValidatorSetFromTransitions(valSet *types.ValidatorSet, transitions []*types.EpochTransition) (*types.Configuration, *ConfigChangeEvent, bool, error) {
	if valSet == nil {
		return nil, nil, false, fmt.Errorf("validator set is nil")
	}

	mm.mu.Lock()
	defer mm.mu.Unlock()

	next := &types.Configuration{
		ID:         valSet.Epoch,
		Validators: make(map[uint64]*types.Validator, len(valSet.Validators)),
		QuorumSize: valSet.QuorumSize,
	}
	for id, v := range valSet.Validators {
		if v == nil {
			continue
		}
		copyVal := *v
		next.Validators[id] = &copyVal
	}
	if next.QuorumSize == 0 {
		next.QuorumSize = uint64((2*len(next.Validators))/3 + 1)
	}

	if mm.currentConfig != nil {
		if next.ID < mm.currentConfig.ID {
			return mm.cloneConfig(mm.currentConfig), cloneConfigChangeEvent(mm.latestEvent), false, nil
		}
		if next.ID == mm.currentConfig.ID {
			if configsEqual(mm.currentConfig, next) {
				return mm.cloneConfig(mm.currentConfig), cloneConfigChangeEvent(mm.latestEvent), false, nil
			}
			return nil, nil, false, fmt.Errorf("conflicting committed config for epoch %d", next.ID)
		}
	}

	oldConfig := mm.cloneConfig(mm.currentConfig)
	added, removed := diffValidatorIDs(oldConfig, next)
	event := latestTransitionEvent(transitions)
	if event != nil {
		if err := validateTransitionEvent(event, oldConfig, next, added, removed); err != nil {
			return nil, nil, false, err
		}
	}

	mm.currentConfig = next
	mm.configHistory = append(mm.configHistory, mm.cloneConfig(next))
	if event == nil {
		event = &ConfigChangeEvent{
			OldEpoch:   oldConfig.ID,
			NewEpoch:   next.ID,
			Added:      added,
			Removed:    removed,
			QuorumSize: next.QuorumSize,
			ConfigHash: next.Hash(),
		}
	}
	mm.latestEvent = event
	return mm.cloneConfig(next), cloneConfigChangeEvent(mm.latestEvent), true, nil
}

func (mm *MembershipManager) GetLatestEvent() *ConfigChangeEvent {
	mm.mu.RLock()
	defer mm.mu.RUnlock()
	return cloneConfigChangeEvent(mm.latestEvent)
}

func cloneConfigChangeEvent(src *ConfigChangeEvent) *ConfigChangeEvent {
	if src == nil {
		return nil
	}
	ev := *src
	ev.Added = append([]uint64(nil), src.Added...)
	ev.Removed = append([]uint64(nil), src.Removed...)
	ev.ConfigHash = append([]byte(nil), src.ConfigHash...)
	return &ev
}

func latestTransitionEvent(transitions []*types.EpochTransition) *ConfigChangeEvent {
	if len(transitions) == 0 {
		return nil
	}
	transition := transitions[len(transitions)-1]
	if transition == nil {
		return nil
	}
	return &ConfigChangeEvent{
		OldEpoch:   transition.OldEpoch,
		NewEpoch:   transition.NewEpoch,
		Added:      append([]uint64(nil), transition.Added...),
		Removed:    append([]uint64(nil), transition.Removed...),
		QuorumSize: transition.QuorumSize,
		ConfigHash: append([]byte(nil), transition.ConfigHash...),
	}
}

func validateTransitionEvent(event *ConfigChangeEvent, oldConfig *types.Configuration, next *types.Configuration, added []uint64, removed []uint64) error {
	if event == nil {
		return nil
	}
	oldEpoch := uint64(0)
	if oldConfig != nil {
		oldEpoch = oldConfig.ID
	}
	if event.OldEpoch != oldEpoch || event.NewEpoch != next.ID || event.QuorumSize != next.QuorumSize || !bytes.Equal(event.ConfigHash, next.Hash()) {
		return fmt.Errorf("transition metadata does not match committed validator set for epoch %d", next.ID)
	}
	if !uint64SlicesEqual(event.Added, added) || !uint64SlicesEqual(event.Removed, removed) {
		return fmt.Errorf("transition validator diff does not match committed validator set for epoch %d", next.ID)
	}
	return nil
}

func uint64SlicesEqual(a []uint64, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func (mm *MembershipManager) cloneConfig(src *types.Configuration) *types.Configuration {
	if src == nil {
		return nil
	}
	out := &types.Configuration{
		ID:         src.ID,
		Validators: make(map[uint64]*types.Validator, len(src.Validators)),
		QuorumSize: src.QuorumSize,
	}
	for id, v := range src.Validators {
		if v == nil {
			continue
		}
		copyVal := *v
		out.Validators[id] = &copyVal
	}
	return out
}

func configsEqual(a *types.Configuration, b *types.Configuration) bool {
	if a == nil || b == nil {
		return a == b
	}
	if a.ID != b.ID || a.QuorumSize != b.QuorumSize || len(a.Validators) != len(b.Validators) {
		return false
	}
	for id, av := range a.Validators {
		bv, exists := b.Validators[id]
		if !exists || av == nil || bv == nil {
			return av == bv && exists
		}
		if av.ID != bv.ID || av.Power != bv.Power || av.IsActive != bv.IsActive || !bytes.Equal(av.PublicKey, bv.PublicKey) || !bytes.Equal(av.VRFPublicKey, bv.VRFPublicKey) {
			return false
		}
	}
	return true
}

func diffValidatorIDs(oldConfig *types.Configuration, newConfig *types.Configuration) ([]uint64, []uint64) {
	added := make([]uint64, 0)
	removed := make([]uint64, 0)

	if newConfig != nil {
		for id := range newConfig.Validators {
			if oldConfig == nil {
				added = append(added, id)
				continue
			}
			if _, exists := oldConfig.Validators[id]; !exists {
				added = append(added, id)
			}
		}
	}
	if oldConfig != nil {
		for id := range oldConfig.Validators {
			if newConfig == nil {
				removed = append(removed, id)
				continue
			}
			if _, exists := newConfig.Validators[id]; !exists {
				removed = append(removed, id)
			}
		}
	}
	sort.Slice(added, func(i, j int) bool { return added[i] < added[j] })
	sort.Slice(removed, func(i, j int) bool { return removed[i] < removed[j] })
	return added, removed
}

// GetQuorumSize returns the quorum size
func (mm *MembershipManager) GetQuorumSize() uint64 {
	mm.mu.RLock()
	defer mm.mu.RUnlock()
	return mm.currentConfig.QuorumSize
}

// IsValidator checks if a node is a validator
func (mm *MembershipManager) IsValidator(id uint64) bool {
	mm.mu.RLock()
	defer mm.mu.RUnlock()
	_, exists := mm.currentConfig.Validators[id]
	return exists
}
