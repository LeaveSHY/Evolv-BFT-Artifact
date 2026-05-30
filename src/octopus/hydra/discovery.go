// Copyright 2024 Octopus Project
// Licensed under Apache License 2.0

package hydra

import (
	"sync"
	"time"
)

type ConfigurationDiscoveryManager struct {
	mu sync.RWMutex

	history            []*Configuration
	highestValidConfig *Configuration
	pendingRequests    map[uint64]*DiscoveryRequest
	network            NetworkInterface
	isRunning          bool
	nodeID             uint64
}

type DiscoveryRequest struct {
	Type      DiscoveryType
	SenderID  uint64
	ConfigIDs []uint64
	Timestamp time.Time
}

type DiscoveryType int

const (
	DiscoveryRequest_ DiscoveryType = iota
	DiscoveryResponse
)

func NewConfigurationDiscoveryManager(nodeID uint64, initialConfig *Configuration, network NetworkInterface) *CDManager {
	initialCopy := initialConfig.Copy()
	return &CDManager{
		nodeID:             nodeID,
		history:            []*Configuration{initialCopy.Copy()},
		highestValidConfig: initialCopy,
		pendingRequests:    make(map[uint64]*DiscoveryRequest),
		network:            network,
	}
}

type CDManager = ConfigurationDiscoveryManager

type DiscoveryResponseMessage struct {
	Type      DiscoveryType
	SenderID  uint64
	Configs   []*Configuration
	Timestamp time.Time
}

func (cdm *CDManager) Start() {
	cdm.mu.Lock()
	cdm.isRunning = true
	cdm.mu.Unlock()
	logger.Info("Configuration discovery manager started")
}

func (cdm *CDManager) Stop() {
	cdm.mu.Lock()
	defer cdm.mu.Unlock()
	cdm.isRunning = false
	logger.Info("Configuration discovery manager stopped")
}

func (cdm *CDManager) GetHighestValidConfig() *Configuration {
	cdm.mu.RLock()
	defer cdm.mu.RUnlock()
	return cdm.highestValidConfig.Copy()
}

func (cdm *CDManager) GetHistory() []*Configuration {
	cdm.mu.RLock()
	defer cdm.mu.RUnlock()
	result := make([]*Configuration, len(cdm.history))
	for i, config := range cdm.history {
		result[i] = config.Copy()
	}
	return result
}

func (cdm *CDManager) GetConfigurationByID(configID uint64) *Configuration {
	cdm.mu.RLock()
	defer cdm.mu.RUnlock()
	for i := len(cdm.history) - 1; i >= 0; i-- {
		config := cdm.history[i]
		if config != nil && config.ID == configID {
			return config.Copy()
		}
	}
	if cdm.highestValidConfig != nil && cdm.highestValidConfig.ID == configID {
		return cdm.highestValidConfig.Copy()
	}
	return nil
}

func (cdm *CDManager) AddConfiguration(config *Configuration) error {
	cdm.mu.Lock()
	defer cdm.mu.Unlock()
	if config == nil {
		return nil
	}
	copyConfig := config.Copy()

	if copyConfig.ID > 0 && len(cdm.history) > 0 {
		lastConfig := cdm.history[len(cdm.history)-1]
		if copyConfig.ID != lastConfig.ID+1 {
			logger.Warn("Non-contiguous configuration: last %d new %d", lastConfig.ID, copyConfig.ID)
		}
	}

	cdm.history = append(cdm.history, copyConfig)

	if cdm.highestValidConfig == nil || copyConfig.ID > cdm.highestValidConfig.ID {
		cdm.highestValidConfig = copyConfig
	}

	logger.Info("Configuration added to history: configID %d validators %d", copyConfig.ID, len(copyConfig.Validators))
	return nil
}

func (cdm *CDManager) RequestDiscovery(targetConfigID uint64) error {
	cdm.mu.Lock()
	defer cdm.mu.Unlock()

	if cdm.highestValidConfig != nil && targetConfigID <= cdm.highestValidConfig.ID {
		return nil
	}

	missing := cdm.findMissingConfigs(targetConfigID)

	req := &DiscoveryRequest{
		Type:      DiscoveryRequest_,
		SenderID:  cdm.nodeID,
		ConfigIDs: missing,
		Timestamp: time.Now(),
	}

	cdm.pendingRequests[req.SenderID] = req

	if cdm.network != nil {
		cdm.network.Broadcast(req)
	}

	logger.Info("Discovery requested (non-blocking): target %d missing %d", targetConfigID, len(missing))
	return nil
}

func (cdm *CDManager) findMissingConfigs(targetID uint64) []uint64 {
	var missing []uint64
	for i := uint64(1); i <= targetID; i++ {
		if !cdm.hasConfig(i) {
			missing = append(missing, i)
		}
	}
	return missing
}

func (cdm *CDManager) hasConfig(id uint64) bool {
	for _, config := range cdm.history {
		if config.ID == id {
			return true
		}
	}
	return false
}

func (cdm *CDManager) HandleDiscoveryRequest(req *DiscoveryRequest) error {
	logger.Info("Handling discovery request from %d requested %d", req.SenderID, len(req.ConfigIDs))

	cdm.mu.RLock()
	defer cdm.mu.RUnlock()

	configs := make([]*Configuration, 0, len(req.ConfigIDs))
	for _, id := range req.ConfigIDs {
		for _, config := range cdm.history {
			if config.ID == id {
				configs = append(configs, config.Copy())
				break
			}
		}
	}

	if cdm.network != nil {
		logger.Info("Sending configuration response to %d count %d", req.SenderID, len(configs))
		cdm.network.Send(req.SenderID, &DiscoveryResponseMessage{
			Type:      DiscoveryResponse,
			SenderID:  cdm.nodeID,
			Configs:   configs,
			Timestamp: time.Now(),
		})
	}
	return nil
}

func (cdm *CDManager) HandleDiscoveryResponse(configs []*Configuration) error {
	cdm.mu.Lock()
	defer cdm.mu.Unlock()

	logger.Info("Handling discovery response: received %d", len(configs))

	for _, config := range configs {
		if !cdm.isValidConfig(config) {
			logger.Warn("Invalid configuration received: configID %d", config.ID)
			continue
		}

		if !cdm.hasConfig(config.ID) {
			copyConfig := config.Copy()
			cdm.history = append(cdm.history, copyConfig)

			if cdm.highestValidConfig == nil || copyConfig.ID > cdm.highestValidConfig.ID {
				if cdm.isContinuous(copyConfig) {
					cdm.highestValidConfig = copyConfig
				}
			}
		}
	}

	logger.Info("Configuration history updated: highest %d history %d", cdm.highestValidConfig.ID, len(cdm.history))
	return nil
}

func (cdm *CDManager) isValidConfig(config *Configuration) bool {
	n := len(config.Validators)
	if n < 3 {
		return false
	}
	return n >= 3*((n-1)/3)+1
}

func (cdm *CDManager) isContinuous(config *Configuration) bool {
	if cdm.highestValidConfig == nil {
		return config.ID == 0
	}
	return config.ID == cdm.highestValidConfig.ID+1
}

func (cdm *CDManager) CanParticipate() bool {
	cdm.mu.RLock()
	defer cdm.mu.RUnlock()
	return cdm.highestValidConfig != nil
}

func (cdm *CDManager) GetParticipatingConfig() *Configuration {
	cdm.mu.RLock()
	defer cdm.mu.RUnlock()
	if cdm.highestValidConfig == nil {
		return nil
	}
	return cdm.highestValidConfig.Copy()
}

func (cdm *CDManager) SyncInBackground() {
	go func() {
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()

		for {
			cdm.mu.RLock()
			running := cdm.isRunning
			cdm.mu.RUnlock()
			if !running {
				return
			}
			select {
			case <-ticker.C:
				cdm.mu.RLock()
				targetID := cdm.highestValidConfig.ID + 10
				cdm.mu.RUnlock()
				_ = cdm.RequestDiscovery(targetID)
			}
		}
	}()
}

func (dt DiscoveryType) String() string {
	switch dt {
	case DiscoveryRequest_:
		return "CDIS"
	case DiscoveryResponse:
		return "DIS"
	default:
		return "UNKNOWN"
	}
}
