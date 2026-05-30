// Copyright 2024 Octopus Project
// Licensed under Apache License 2.0

package storage

import (
	"fmt"
	"sync"

	"octopus-bft/octopus/types"
)

var logger = struct {
	Info func(format string, args ...interface{})
}{
	Info: func(format string, args ...interface{}) { fmt.Printf(format+"\n", args...) },
}

// StorageManager manages all storage
type StorageManager struct {
	mu sync.RWMutex

	// In-memory blocks
	blocks map[string]*types.Block
	
	// Height to Hash mapping
	heightToHash map[uint64]string

	// State
	nodeID uint64
}

// NewStorageManager creates a new storage manager
func NewStorageManager(nodeID uint64) *StorageManager {
	return &StorageManager{
		nodeID:       nodeID,
		blocks:       make(map[string]*types.Block),
		heightToHash: make(map[uint64]string),
	}
}

// PutBlock stores a block
func (sm *StorageManager) PutBlock(block *types.Block) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if block == nil || block.Hash == nil {
		return fmt.Errorf("invalid block")
	}

	hashStr := string(block.Hash)
	sm.blocks[hashStr] = block
	sm.heightToHash[block.Height] = hashStr
	
	logger.Info("Stored block at height %d for node %d", block.Height, sm.nodeID)
	return nil
}

// GetBlock retrieves a block by hash
func (sm *StorageManager) GetBlock(hash []byte) (*types.Block, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	block, exists := sm.blocks[string(hash)]
	if !exists {
		return nil, fmt.Errorf("block not found")
	}

	return block, nil
}

// GetBlockByHeight retrieves a block by height
func (sm *StorageManager) GetBlockByHeight(height uint64) (*types.Block, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	hashStr, exists := sm.heightToHash[height]
	if !exists {
		return nil, fmt.Errorf("block at height %d not found", height)
	}

	return sm.blocks[hashStr], nil
}

// HasBlock checks if a block exists
func (sm *StorageManager) HasBlock(hash []byte) bool {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	_, exists := sm.blocks[string(hash)]
	return exists
}

// GetLatestBlockHeight returns the latest block height
func (sm *StorageManager) GetLatestBlockHeight() uint64 {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	maxHeight := uint64(0)
	for height := range sm.heightToHash {
		if height > maxHeight {
			maxHeight = height
		}
	}

	return maxHeight
}

// Close closes the storage
func (sm *StorageManager) Close() error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	sm.blocks = make(map[string]*types.Block)
	sm.heightToHash = make(map[uint64]string)
	logger.Info("Storage closed for node %d", sm.nodeID)
	return nil
}
