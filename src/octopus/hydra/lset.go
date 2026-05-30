// Copyright 2024 Octopus Project
// Licensed under Apache License 2.0

package hydra

import (
	"fmt"
	"sort"
	"sync"
	"time"

	"octopus-bft/octopus/types"
)

type Validator = types.Validator

type Configuration struct {
	ID         uint64
	Validators map[uint64]*types.Validator
	QuorumSize int
	BlockHash  []byte
	Timestamp  time.Time
}

func cloneValidator(v *Validator) *Validator {
	if v == nil {
		return nil
	}
	return &Validator{
		ID:           v.ID,
		PublicKey:    append([]byte(nil), v.PublicKey...),
		VRFPublicKey: append([]byte(nil), v.VRFPublicKey...),
		Power:        v.Power,
		IsActive:     v.IsActive,
	}
}

func (c *Configuration) Copy() *Configuration {
	if c == nil {
		return nil
	}
	validators := make(map[uint64]*Validator, len(c.Validators))
	for k, v := range c.Validators {
		validators[k] = cloneValidator(v)
	}
	return &Configuration{
		ID:         c.ID,
		Validators: validators,
		QuorumSize: c.QuorumSize,
		BlockHash:  append([]byte(nil), c.BlockHash...),
		Timestamp:  c.Timestamp,
	}
}

type NetworkInterface interface {
	Broadcast(msg interface{})
	Send(to uint64, msg interface{})
}

type LSetManager struct {
	mu        sync.RWMutex
	LSet      map[uint64]*Validator
	mmtable   map[uint64]*MMTableEntry
	threshold int
}

type FaultClass uint8

const (
	FaultClassNone FaultClass = iota
	FaultClassDegraded
	FaultClassUnavailable
	FaultClassByzantine
)

func (fc FaultClass) String() string {
	switch fc {
	case FaultClassDegraded:
		return "degraded"
	case FaultClassUnavailable:
		return "unavailable"
	case FaultClassByzantine:
		return "byzantine"
	default:
		return "none"
	}
}

func (fc FaultClass) BlocksProposal() bool {
	switch fc {
	case FaultClassDegraded, FaultClassUnavailable, FaultClassByzantine:
		return true
	default:
		return false
	}
}

func (fc FaultClass) Evictable() bool {
	switch fc {
	case FaultClassUnavailable, FaultClassByzantine:
		return true
	default:
		return false
	}
}

type MMTableEntry struct {
	Class     FaultClass
	Timestamp time.Time
	Count     int
}

func NewLSetManager(initialValidators map[uint64]*Validator) (*LSetManager, error) {
	lm := &LSetManager{
		LSet:      make(map[uint64]*Validator),
		mmtable:   make(map[uint64]*MMTableEntry),
		threshold: calculateThreshold(len(initialValidators)),
	}

	lm.rebuildLSetLocked(initialValidators)

	logger.Info("L-set initialized: LSize %d threshold %d", len(lm.LSet), lm.threshold)
	return lm, nil
}

func (lm *LSetManager) IsInL(id uint64) bool {
	lm.mu.RLock()
	defer lm.mu.RUnlock()
	_, exists := lm.LSet[id]
	return exists
}

func (lm *LSetManager) GetLSet() map[uint64]*Validator {
	lm.mu.RLock()
	defer lm.mu.RUnlock()
	result := make(map[uint64]*Validator, len(lm.LSet))
	for k, v := range lm.LSet {
		result[k] = cloneValidator(v)
	}
	return result
}

func (lm *LSetManager) MarkFault(id uint64, class FaultClass) {
	if class == FaultClassNone {
		return
	}

	lm.mu.Lock()
	defer lm.mu.Unlock()
	entry, exists := lm.mmtable[id]
	if !exists {
		entry = &MMTableEntry{
			Class:     class,
			Timestamp: time.Now(),
			Count:     1,
		}
	} else {
		entry.Count++
		entry.Timestamp = time.Now()
		if class > entry.Class {
			entry.Class = class
		}
	}
	lm.mmtable[id] = entry
}

func (lm *LSetManager) GetFaultClass(id uint64) FaultClass {
	lm.mu.RLock()
	defer lm.mu.RUnlock()
	entry, exists := lm.mmtable[id]
	if !exists || entry == nil {
		return FaultClassNone
	}
	return entry.Class
}

func (lm *LSetManager) IsMarked(id uint64) bool {
	return lm.GetFaultClass(id) != FaultClassNone
}

func (lm *LSetManager) HasEvictableFault(id uint64) bool {
	return lm.GetFaultClass(id).Evictable()
}

func (lm *LSetManager) ClearMarks(duration time.Duration) {
	lm.mu.Lock()
	defer lm.mu.Unlock()
	now := time.Now()
	for id, entry := range lm.mmtable {
		if now.Sub(entry.Timestamp) > duration {
			delete(lm.mmtable, id)
		}
	}
}

func (lm *LSetManager) InstallConfiguration(config *Configuration) {
	if config == nil {
		return
	}

	lm.mu.Lock()
	defer lm.mu.Unlock()
	lm.threshold = calculateThreshold(len(config.Validators))
	lm.rebuildLSetLocked(config.Validators)
	lm.mmtable = make(map[uint64]*MMTableEntry)
}

func (lm *LSetManager) IsAllowedToPropose(id uint64) bool {
	return !lm.GetFaultClass(id).BlocksProposal()
}

func calculateThreshold(n int) int {
	return (n - 1) / 3
}

func (lm *LSetManager) rebuildLSetLocked(validators map[uint64]*Validator) {
	lm.LSet = make(map[uint64]*Validator)
	ids := make([]uint64, 0, len(validators))
	for id, v := range validators {
		if v == nil || !v.IsActive {
			continue
		}
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })

	lSize := len(ids)/2 + 1
	for idx, id := range ids {
		if idx >= lSize {
			break
		}
		lm.LSet[id] = cloneValidator(validators[id])
	}
}

type TemporaryConfigurationManager struct {
	mu               sync.RWMutex
	Mvalid           *Configuration
	Mhigh            *Configuration
	history          []*Configuration
	pendingJoins     map[uint64]*MemberRequest
	pendingLeaves    map[uint64]*MemberRequest
	currentConfigID  uint64
	reservedConfigID uint64
}

type MemberRequest struct {
	Type      RequestType
	ID        uint64
	PublicKey []byte
	Power     uint64
	Signature []byte
	View      uint64
	Timestamp time.Time
}

func cloneMemberRequest(req *MemberRequest) *MemberRequest {
	if req == nil {
		return nil
	}
	return &MemberRequest{
		Type:      req.Type,
		ID:        req.ID,
		PublicKey: append([]byte(nil), req.PublicKey...),
		Power:     req.Power,
		Signature: append([]byte(nil), req.Signature...),
		View:      req.View,
		Timestamp: req.Timestamp,
	}
}

type RequestType int

const (
	RequestJoin RequestType = iota
	RequestLeave
)

func NewTemporaryConfigurationManager(initialConfig *Configuration) *TCM {
	tcm := &TemporaryConfigurationManager{
		Mvalid:           initialConfig.Copy(),
		Mhigh:            initialConfig.Copy(),
		history:          []*Configuration{initialConfig.Copy()},
		pendingJoins:     make(map[uint64]*MemberRequest),
		pendingLeaves:    make(map[uint64]*MemberRequest),
		currentConfigID:  initialConfig.ID,
		reservedConfigID: initialConfig.ID,
	}
	return tcm
}

func (tcm *TemporaryConfigurationManager) ApplyMemberRequests() (*Configuration, error) {
	tcm.mu.Lock()
	defer tcm.mu.Unlock()
	mhigh := tcm.Mvalid.Copy()
	for _, req := range tcm.pendingJoins {
		if req.Type == RequestJoin {
			mhigh.Validators[req.ID] = &Validator{
				ID:        req.ID,
				PublicKey: append([]byte(nil), req.PublicKey...),
				Power:     req.Power,
				IsActive:  true,
			}
		}
	}
	for _, req := range tcm.pendingLeaves {
		if req.Type == RequestLeave {
			delete(mhigh.Validators, req.ID)
		}
	}
	mhigh.QuorumSize = (2*len(mhigh.Validators))/3 + 1
	tcm.Mhigh = mhigh
	return mhigh, nil
}

func (tcm *TemporaryConfigurationManager) PromoteMhighToMvalid() error {
	tcm.mu.Lock()
	defer tcm.mu.Unlock()
	if tcm.Mhigh == nil {
		return fmt.Errorf("Mhigh is nil")
	}
	tcm.Mvalid = tcm.Mhigh.Copy()
	tcm.currentConfigID++
	tcm.reservedConfigID = tcm.currentConfigID
	tcm.Mvalid.ID = tcm.currentConfigID
	tcm.history = append(tcm.history, tcm.Mvalid)
	tcm.pendingJoins = make(map[uint64]*MemberRequest)
	tcm.pendingLeaves = make(map[uint64]*MemberRequest)
	logger.Info("Promoted Mhigh to Mvalid: configID %d validators %d", tcm.currentConfigID, len(tcm.Mvalid.Validators))
	return nil
}

func (tcm *TemporaryConfigurationManager) InstallCommittedConfig(config *Configuration) error {
	if config == nil {
		return fmt.Errorf("configuration is nil")
	}

	tcm.mu.Lock()
	defer tcm.mu.Unlock()

	if tcm.Mvalid != nil && tcm.Mvalid.ID == config.ID {
		if configsEqualHydra(tcm.Mvalid, config) {
			return nil
		}
		return fmt.Errorf("conflicting committed config for id %d", config.ID)
	}

	tcm.Mvalid = config.Copy()
	tcm.Mhigh = config.Copy()
	tcm.currentConfigID = config.ID
	if tcm.reservedConfigID < config.ID {
		tcm.reservedConfigID = config.ID
	}
	tcm.history = append(tcm.history, tcm.Mvalid.Copy())

	for id := range tcm.pendingJoins {
		if _, exists := config.Validators[id]; exists {
			delete(tcm.pendingJoins, id)
		}
	}
	for id := range tcm.pendingLeaves {
		if _, exists := config.Validators[id]; !exists {
			delete(tcm.pendingLeaves, id)
		}
	}

	return nil
}

func (tcm *TemporaryConfigurationManager) GetMvalid() *Configuration {
	tcm.mu.RLock()
	defer tcm.mu.RUnlock()
	return tcm.Mvalid.Copy()
}

func (tcm *TemporaryConfigurationManager) GetMhigh() *Configuration {
	tcm.mu.RLock()
	defer tcm.mu.RUnlock()
	if tcm.Mhigh == nil {
		return tcm.Mvalid.Copy()
	}
	return tcm.Mhigh.Copy()
}

func (tcm *TemporaryConfigurationManager) GetConfigurationByID(configID uint64) *Configuration {
	tcm.mu.RLock()
	defer tcm.mu.RUnlock()
	for i := len(tcm.history) - 1; i >= 0; i-- {
		config := tcm.history[i]
		if config != nil && config.ID == configID {
			return config.Copy()
		}
	}
	if tcm.Mvalid != nil && tcm.Mvalid.ID == configID {
		return tcm.Mvalid.Copy()
	}
	if tcm.Mhigh != nil && tcm.Mhigh.ID == configID {
		return tcm.Mhigh.Copy()
	}
	return nil
}

func (tcm *TemporaryConfigurationManager) GetHistory() []*Configuration {
	tcm.mu.RLock()
	defer tcm.mu.RUnlock()
	result := make([]*Configuration, len(tcm.history))
	for i, config := range tcm.history {
		result[i] = config.Copy()
	}
	return result
}

func (tcm *TemporaryConfigurationManager) ReserveNextConfigID() uint64 {
	tcm.mu.Lock()
	defer tcm.mu.Unlock()
	tcm.reservedConfigID++
	return tcm.reservedConfigID
}

func (tcm *TemporaryConfigurationManager) ObserveReservedConfigID(id uint64) {
	tcm.mu.Lock()
	defer tcm.mu.Unlock()
	if id > tcm.reservedConfigID {
		tcm.reservedConfigID = id
	}
}

func (tcm *TemporaryConfigurationManager) AddJoinRequest(req *MemberRequest) error {
	tcm.mu.Lock()
	defer tcm.mu.Unlock()
	if _, exists := tcm.pendingJoins[req.ID]; exists {
		return fmt.Errorf("join request already pending for %d", req.ID)
	}
	tcm.pendingJoins[req.ID] = cloneMemberRequest(req)
	logger.Info("Join request added to pending: id %d", req.ID)
	return nil
}

func (tcm *TemporaryConfigurationManager) AddLeaveRequest(req *MemberRequest) error {
	tcm.mu.Lock()
	defer tcm.mu.Unlock()
	if _, exists := tcm.pendingLeaves[req.ID]; exists {
		return fmt.Errorf("leave request already pending for %d", req.ID)
	}
	tcm.pendingLeaves[req.ID] = cloneMemberRequest(req)
	logger.Info("Leave request added to pending: id %d", req.ID)
	return nil
}

func (tcm *TemporaryConfigurationManager) RemoveJoinRequest(id uint64) {
	tcm.mu.Lock()
	defer tcm.mu.Unlock()
	delete(tcm.pendingJoins, id)
}

func (tcm *TemporaryConfigurationManager) RemoveLeaveRequest(id uint64) {
	tcm.mu.Lock()
	defer tcm.mu.Unlock()
	delete(tcm.pendingLeaves, id)
}

func (tcm *TemporaryConfigurationManager) GetPendingJoins() []*MemberRequest {
	tcm.mu.RLock()
	defer tcm.mu.RUnlock()
	result := make([]*MemberRequest, 0, len(tcm.pendingJoins))
	for _, req := range tcm.pendingJoins {
		result = append(result, cloneMemberRequest(req))
	}
	return result
}

func (tcm *TemporaryConfigurationManager) GetPendingLeaves() []*MemberRequest {
	tcm.mu.RLock()
	defer tcm.mu.RUnlock()
	result := make([]*MemberRequest, 0, len(tcm.pendingLeaves))
	for _, req := range tcm.pendingLeaves {
		result = append(result, cloneMemberRequest(req))
	}
	return result
}

func (tcm *TemporaryConfigurationManager) IsValidConfiguration(config *Configuration) bool {
	n := len(config.Validators)
	if n < 3 {
		return false
	}
	return n >= 3*((n-1)/3)+1
}

type TCM = TemporaryConfigurationManager
