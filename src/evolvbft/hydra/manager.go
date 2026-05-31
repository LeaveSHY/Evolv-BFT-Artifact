// Copyright 2024 Evolv-BFT Project
// Licensed under Apache License 2.0

package hydra

import (
	"fmt"
	"sort"
	"sync"

	"evolvbft/evolvbft/types"
)

// ResponsibilityTracker tracks which configuration is responsible for each request
type ResponsibilityTracker struct {
	mu sync.RWMutex

	// Map: request hash -> responsible config ID
	requestToConfig map[string]uint64

	// Map: config ID -> set of responsible validators
	responsibleValidators map[uint64]map[uint64]bool
}

// HydraManager integrates all Hydra features into Evolv-BFT
type HydraManager struct {
	mu sync.RWMutex

	// Core components
	LSetManager           *LSetManager
	TempConfigManager     *TCM
	AutoTransitionManager *ATM
	DiscoveryManager      *CDManager

	// Credential verification (§III-D Sybil resistance)
	CredentialVerifier CredentialVerifier

	// Replica Responsibility tracking
	responsibilityTracker *ResponsibilityTracker

	// Configuration
	nodeID  uint64
	network NetworkInterface

	// State
	isRunning bool
}

// NewHydraManager creates a new Hydra manager integrating all features
func NewHydraManager(
	nodeID uint64,
	initialValidators map[uint64]*Validator,
	network NetworkInterface,
	privateKey ...types.PrivateKey,
) (*HydraManager, error) {

	validators := make(map[uint64]*Validator, len(initialValidators))
	for id, validator := range initialValidators {
		validators[id] = cloneValidator(validator)
	}

	// Create initial configuration
	initialConfig := &Configuration{
		ID:         0,
		Validators: validators,
		QuorumSize: (2*len(validators))/3 + 1,
	}

	// Initialize L-set manager
	lSetManager, err := NewLSetManager(validators)
	if err != nil {
		return nil, err
	}

	// Initialize temporary configuration manager
	tempConfigManager := NewTemporaryConfigurationManager(initialConfig)

	// Initialize auto-transition manager
	autoTransitionManager := NewAutoTransitionManager(
		lSetManager,
		tempConfigManager,
		network,
	)

	// Initialize discovery manager
	discoveryManager := NewConfigurationDiscoveryManager(nodeID, initialConfig, network)

	// Initialize responsibility tracker
	respTracker := &ResponsibilityTracker{
		requestToConfig:       make(map[string]uint64),
		responsibleValidators: make(map[uint64]map[uint64]bool),
	}

	hm := &HydraManager{
		nodeID:                nodeID,
		LSetManager:           lSetManager,
		TempConfigManager:     tempConfigManager,
		AutoTransitionManager: autoTransitionManager,
		DiscoveryManager:      discoveryManager,
		responsibilityTracker: respTracker,
		CredentialVerifier:    &NoopCredentialVerifier{},
		network:               network,
	}

	logger.Info("Hydra manager initialized: node %d validators %d lSet %d", nodeID, len(initialValidators), len(lSetManager.LSet))

	// Wire nodeID into ATM so auto-votes use the correct sender identity
	autoTransitionManager.SetNodeID(nodeID)
	if len(privateKey) > 0 {
		autoTransitionManager.SetPrivateKey(privateKey[0])
	}

	return hm, nil
}

// HM is shorthand
type HM = HydraManager

// Start starts all Hydra components
func (hm *HM) Start() {
	hm.mu.Lock()
	hm.isRunning = true
	hm.mu.Unlock()

	hm.AutoTransitionManager.Start()
	hm.DiscoveryManager.Start()
	hm.DiscoveryManager.SyncInBackground()

	logger.Info("Hydra manager started")
}

// Stop stops all Hydra components
func (hm *HM) Stop() {
	hm.mu.Lock()
	defer hm.mu.Unlock()

	hm.AutoTransitionManager.Stop()
	hm.DiscoveryManager.Stop()
	hm.isRunning = false

	logger.Info("Hydra manager stopped")
}

// GetCurrentConfiguration returns the committed configuration currently installed
// on the authoritative runtime path.
func (hm *HM) GetCurrentConfiguration() *Configuration {
	return hm.TempConfigManager.GetMvalid()
}

// GetHighestKnownConfiguration returns Hydra's highest discovered contiguous
// configuration view for non-authoritative catch-up/observation use.
func (hm *HM) GetHighestKnownConfiguration() *Configuration {
	if hm == nil || hm.DiscoveryManager == nil {
		return nil
	}
	return hm.DiscoveryManager.GetParticipatingConfig()
}

// SubmitJoinRequest submits a join request after credential verification (§III-D).
func (hm *HM) SubmitJoinRequest(validatorID uint64, pubKey []byte, power uint64) error {
	// Admission filter: verify CA-issued credential (Algorithm 3, Sybil resistance)
	if hm.CredentialVerifier != nil {
		if err := hm.CredentialVerifier.VerifyCredential(validatorID, pubKey); err != nil {
			return fmt.Errorf("join rejected: %w", err)
		}
	}

	req := &MemberRequest{
		Type:      RequestJoin,
		ID:        validatorID,
		PublicKey: pubKey,
		Power:     power,
	}

	// Add to temporary configuration
	return hm.TempConfigManager.AddJoinRequest(req)
}

// SubmitLeaveRequest submits a leave request
func (hm *HM) SubmitLeaveRequest(validatorID uint64) error {
	req := &MemberRequest{
		Type: RequestLeave,
		ID:   validatorID,
	}

	return hm.TempConfigManager.AddLeaveRequest(req)
}

// TrackResponsibility tracks which config is responsible for a request
func (hm *HM) TrackResponsibility(requestHash string, configID uint64) {
	hm.responsibilityTracker.mu.Lock()
	defer hm.responsibilityTracker.mu.Unlock()

	hm.responsibilityTracker.requestToConfig[requestHash] = configID

	// Track responsible validators for the specific configuration being bound.
	if hm.TempConfigManager == nil {
		return
	}
	config := hm.TempConfigManager.GetConfigurationByID(configID)
	if config == nil && hm.DiscoveryManager != nil {
		config = hm.DiscoveryManager.GetConfigurationByID(configID)
	}
	if config == nil {
		return
	}
	validators := make(map[uint64]bool, len(config.Validators))
	for id := range config.Validators {
		validators[id] = true
	}
	hm.responsibilityTracker.responsibleValidators[configID] = validators
}

// GetResponsibleConfig returns the config responsible for a request
func (hm *HM) GetResponsibleConfig(requestHash string) (uint64, bool) {
	hm.responsibilityTracker.mu.RLock()
	defer hm.responsibilityTracker.mu.RUnlock()

	configID, exists := hm.responsibilityTracker.requestToConfig[requestHash]
	return configID, exists
}

// IsResponsibleValidator checks if this node is responsible for a request
func (hm *HM) IsResponsibleValidator(requestHash string) bool {
	configID, exists := hm.GetResponsibleConfig(requestHash)
	if !exists {
		return false
	}

	hm.responsibilityTracker.mu.RLock()
	defer hm.responsibilityTracker.mu.RUnlock()

	if validators, ok := hm.responsibilityTracker.responsibleValidators[configID]; ok {
		return validators[hm.nodeID]
	}

	return false
}

// HandleMessage handles Hydra-related messages
func (hm *HM) HandleMessage(msg interface{}) error {
	// Dispatch to appropriate handler
	switch m := msg.(type) {
	case *AutoTransitionMessage:
		return hm.AutoTransitionManager.HandleAutoTransitionMessage(m)
	case *DiscoveryRequest:
		return hm.DiscoveryManager.HandleDiscoveryRequest(m)
	case *DiscoveryResponseMessage:
		return hm.DiscoveryManager.HandleDiscoveryResponse(m.Configs)
	default:
		return nil
	}
}

// TriggerAutoTransition triggers auto-transition when leader times out
func (hm *HM) TriggerAutoTransition(
	leaderID uint64,
	view uint64,
) error {
	return hm.AutoTransitionManager.TriggerAutoTransition(
		leaderID,
		view,
	)
}

// RequestConfigDiscovery requests missing configurations (non-blocking)
func (hm *HM) RequestConfigDiscovery(targetConfigID uint64) error {
	return hm.DiscoveryManager.RequestDiscovery(targetConfigID)
}

// CanParticipate checks if node can participate in consensus
// Returns true even if not fully caught up (non-blocking)
func (hm *HM) CanParticipate() bool {
	return hm.DiscoveryManager.CanParticipate()
}

// GetClarity returns clarity info for a request (which config is responsible)
func (hm *HM) GetClarity(requestHash string) (configID uint64, validators map[uint64]bool, exists bool) {
	configID, exists = hm.GetResponsibleConfig(requestHash)
	if !exists {
		return 0, nil, false
	}

	hm.responsibilityTracker.mu.RLock()
	defer hm.responsibilityTracker.mu.RUnlock()

	trackedValidators := hm.responsibilityTracker.responsibleValidators[configID]
	if trackedValidators == nil {
		return configID, nil, true
	}
	copyValidators := make(map[uint64]bool, len(trackedValidators))
	for id, allowed := range trackedValidators {
		copyValidators[id] = allowed
	}
	return configID, copyValidators, true
}

func (hm *HM) InstallCommittedConfiguration(config *types.Configuration) error {
	if config == nil {
		return nil
	}
	if hm == nil {
		return nil
	}

	hm.mu.Lock()
	defer hm.mu.Unlock()

	if hm.TempConfigManager == nil {
		return fmt.Errorf("hydra temp config manager is nil")
	}
	if hm.LSetManager == nil {
		return fmt.Errorf("hydra l-set manager is nil")
	}
	if hm.DiscoveryManager == nil {
		return fmt.Errorf("hydra discovery manager is nil")
	}

	committed := configFromTypes(config)
	existing := hm.TempConfigManager.GetMvalid()
	if existing != nil {
		if committed.ID < existing.ID {
			return nil
		}
		if existing.ID == committed.ID {
			if configsEqualHydra(existing, committed) {
				return nil
			}
			return fmt.Errorf("conflicting committed hydra config for id %d", committed.ID)
		}
	}
	if err := hm.DiscoveryManager.AddConfiguration(committed); err != nil {
		return err
	}
	if err := hm.TempConfigManager.InstallCommittedConfig(committed); err != nil {
		return err
	}
	hm.LSetManager.InstallConfiguration(committed)
	return nil
}

func (hm *HM) AllowedLeaders() []uint64 {
	if hm == nil || hm.LSetManager == nil {
		return nil
	}
	lset := hm.LSetManager.GetLSet()
	ids := make([]uint64, 0, len(lset))
	for id, validator := range lset {
		if validator == nil || !validator.IsActive {
			continue
		}
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	return ids
}

func (hm *HM) IsAllowedLeader(id uint64) bool {
	allowed := hm.AllowedLeaders()
	if len(allowed) == 0 {
		return true
	}
	for _, allowedID := range allowed {
		if allowedID == id {
			return true
		}
	}
	return false
}

func configsEqualHydra(a *Configuration, b *Configuration) bool {
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
		if av.ID != bv.ID || av.Power != bv.Power || av.IsActive != bv.IsActive || string(av.PublicKey) != string(bv.PublicKey) || string(av.VRFPublicKey) != string(bv.VRFPublicKey) {
			return false
		}
	}
	return true
}

func configFromTypes(config *types.Configuration) *Configuration {
	if config == nil {
		return nil
	}

	validators := make(map[uint64]*Validator, len(config.Validators))
	for id, v := range config.Validators {
		if v == nil {
			continue
		}
		copyVal := *v
		copyVal.PublicKey = append([]byte(nil), v.PublicKey...)
		copyVal.VRFPublicKey = append([]byte(nil), v.VRFPublicKey...)
		validators[id] = &copyVal
	}

	return &Configuration{
		ID:         config.ID,
		Validators: validators,
		QuorumSize: int(config.QuorumSize),
	}
}
