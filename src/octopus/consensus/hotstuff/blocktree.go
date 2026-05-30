// Copyright 2024 Octopus Project
// Licensed under Apache License 2.0

package hotstuff

import (
	"bytes"
	"fmt"
	"octopus-bft/octopus/storage"
	"octopus-bft/octopus/types"
	"sync"
)

var logger = struct {
	Info  func(format string, args ...interface{})
	Error func(format string, args ...interface{})
}{
	Info:  func(format string, args ...interface{}) { fmt.Printf(format+"\n", args...) },
	Error: func(format string, args ...interface{}) { fmt.Printf("ERROR: "+format+"\n", args...) },
}

// BlockTree manages the block DAG and QC state
type BlockTree struct {
	mu sync.RWMutex

	storage             *storage.StorageManager
	executor            *Executor
	onCommit            func(*types.Block)
	onOptimisticConfirm func(*types.Block)

	// HotStuff Chain State
	highQC    *types.QuorumCertificate // Highest QC known
	genericQC *types.QuorumCertificate // QC for the current view
	lockedQC  *types.QuorumCertificate // Highest 2-chain QC
	commitQC  *types.QuorumCertificate // Highest 3-chain QC

	fastPathThreshold uint64 // Number of signatures required for optimistic 2-chain fast commit
}

// NewBlockTree creates a new block tree
func NewBlockTree(storage *storage.StorageManager, executor *Executor) *BlockTree {
	return NewBlockTreeWithEpoch(storage, executor, 0)
}

// NewBlockTreeWithEpoch creates a new block tree with an explicit genesis epoch
func NewBlockTreeWithEpoch(storage *storage.StorageManager, executor *Executor, genesisEpoch uint64) *BlockTree {
	genesisQC := types.NewQuorumCertificate(nil, 0, genesisEpoch, types.PhaseDecide)

	return &BlockTree{
		storage:   storage,
		executor:  executor,
		highQC:    genesisQC,
		genericQC: genesisQC,
		lockedQC:  genesisQC,
		commitQC:  genesisQC,
	}
}

// ProcessBlock processes a new block and updates chain state
func (bt *BlockTree) ProcessBlock(block *types.Block) error {
	bt.mu.Lock()
	defer bt.mu.Unlock()

	if err := bt.storage.PutBlock(block); err != nil {
		return err
	}

	if block.Justify != nil && isQCNewer(block.Justify, bt.highQC) {
		bt.highQC = block.Justify
	}

	if block.Justify == nil || len(block.Justify.BlockHash) == 0 {
		return nil
	}

	genericBlock, err := bt.storage.GetBlock(block.Justify.BlockHash)
	if err != nil {
		return nil
	}
	bt.genericQC = block.Justify

	// ⚡ Optimistic Confirmation: notify clients after 1-QC (~50ms)
	// Full BFT finality requires 3-chain, but for latency-sensitive
	// applications, a single QC from 2f+1 validators provides strong
	// probabilistic safety (violated only if >f validators equivocate).
	if bt.onOptimisticConfirm != nil {
		bt.onOptimisticConfirm(genericBlock)
	}

	if genericBlock.Justify == nil || len(genericBlock.Justify.BlockHash) == 0 {
		return nil
	}
	lockedBlock, err := bt.storage.GetBlock(genericBlock.Justify.BlockHash)
	if err != nil {
		return nil
	}

	lockCandidate := genericBlock.Justify
	if lockCandidate.View >= lockedBlock.View &&
		genericBlock.View > lockedBlock.View &&
		(bt.lockedQC == nil || lockCandidate.View > bt.lockedQC.View) {
		bt.lockedQC = lockCandidate
	}

	// ⚡ Optimistic 2-chain Fast-Commit (90% supermajority threshold)
	if bt.fastPathThreshold > 0 && uint64(block.Justify.NumSignatures) >= bt.fastPathThreshold {
		if bt.commitQC == nil || lockCandidate.View > bt.commitQC.View {
			bt.commitQC = lockCandidate
			logger.Info("⚡ FAST-COMMIT (2-chain) block at height %d (signatures=%d threshold=%d)", lockedBlock.Height, block.Justify.NumSignatures, bt.fastPathThreshold)
			bt.commit(lockedBlock)
		}
	}

	if lockedBlock.Justify == nil || len(lockedBlock.Justify.BlockHash) == 0 {
		return nil
	}
	commitBlock, err := bt.storage.GetBlock(lockedBlock.Justify.BlockHash)
	if err != nil {
		return nil
	}

	commitCandidate := lockedBlock.Justify
	if commitCandidate.View >= commitBlock.View &&
		lockedBlock.View > commitBlock.View &&
		(bt.commitQC == nil || commitCandidate.View > bt.commitQC.View) {
		bt.commitQC = commitCandidate
		bt.commit(commitBlock)
	}

	return nil
}

func (bt *BlockTree) commit(block *types.Block) {
	if block == nil {
		return
	}
	logger.Info("🔥 COMMITTING block at height %d, View %d, Epoch %d, Hash %x", block.Height, block.View, block.Epoch, block.Hash[:4])
	block.SetPhase(types.PhaseCommit)

	if bt.onCommit != nil {
		bt.onCommit(block)
	}
}

func (bt *BlockTree) SetCommitCallback(callback func(*types.Block)) {
	bt.mu.Lock()
	defer bt.mu.Unlock()
	bt.onCommit = callback
}

func (bt *BlockTree) SetOptimisticConfirmCallback(callback func(*types.Block)) {
	bt.mu.Lock()
	defer bt.mu.Unlock()
	bt.onOptimisticConfirm = callback
}

func (bt *BlockTree) SetFastPathThreshold(n uint64) {
	bt.mu.Lock()
	defer bt.mu.Unlock()
	bt.fastPathThreshold = n * 9 / 10
}

func (bt *BlockTree) OnVoteQC(qc *types.QuorumCertificate) {
	if qc == nil {
		return
	}
	bt.mu.Lock()
	defer bt.mu.Unlock()
	if isQCNewer(qc, bt.highQC) {
		bt.highQC = qc
	}
	if bt.genericQC == nil || qc.Epoch > bt.genericQC.Epoch || (qc.Epoch == bt.genericQC.Epoch && qc.View >= bt.genericQC.View) {
		bt.genericQC = qc
	}
}

// SafeNode checks if a block is safe to vote for
func (bt *BlockTree) SafeNode(block *types.Block) bool {
	bt.mu.RLock()
	defer bt.mu.RUnlock()

	if block == nil {
		return false
	}
	if bt.lockedQC == nil {
		return true
	}
	if block.Justify != nil && block.Justify.View > bt.lockedQC.View {
		return true
	}
	return bt.extendsLockedBlock(block, bt.lockedQC)
}

func (bt *BlockTree) extendsLockedBlock(block *types.Block, lockedQC *types.QuorumCertificate) bool {
	if block == nil {
		return false
	}
	if lockedQC == nil || len(lockedQC.BlockHash) == 0 {
		return true
	}
	if block.Justify != nil && bytes.Equal(block.Justify.BlockHash, lockedQC.BlockHash) {
		return true
	}
	current := block.Parent
	if len(current) == 0 && block.Justify != nil {
		current = block.Justify.BlockHash
	}
	visited := make(map[string]struct{})
	for len(current) > 0 {
		if bytes.Equal(current, lockedQC.BlockHash) {
			return true
		}
		key := string(current)
		if _, exists := visited[key]; exists {
			return false
		}
		visited[key] = struct{}{}

		parentBlock, err := bt.storage.GetBlock(current)
		if err != nil {
			return false
		}
		next := parentBlock.Parent
		if len(next) == 0 && parentBlock.Justify != nil {
			next = parentBlock.Justify.BlockHash
		}
		current = next
	}
	return false
}

// GetHighQC returns the highest QC
func (bt *BlockTree) GetHighQC() *types.QuorumCertificate {
	bt.mu.RLock()
	defer bt.mu.RUnlock()
	return bt.highQC
}

// GetLockedQC returns the locked QC
func (bt *BlockTree) GetLockedQC() *types.QuorumCertificate {
	bt.mu.RLock()
	defer bt.mu.RUnlock()
	return bt.lockedQC
}

func (bt *BlockTree) GetCommitQC() *types.QuorumCertificate {
	bt.mu.RLock()
	defer bt.mu.RUnlock()
	return bt.commitQC
}

func isQCNewer(candidate *types.QuorumCertificate, current *types.QuorumCertificate) bool {
	if candidate == nil {
		return false
	}
	if current == nil {
		return true
	}
	if candidate.Epoch != current.Epoch {
		return candidate.Epoch > current.Epoch
	}
	return candidate.View > current.View
}
