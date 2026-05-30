// Copyright 2024 Octopus Project
// Licensed under Apache License 2.0

package hotstuff

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"

	"octopus-bft/octopus/crypto"
	"octopus-bft/octopus/types"
)

// VertexResolver resolves vertex certificates to actual vertices.
// Implemented by Mempool.
type VertexResolver interface {
	GetVertex(hash types.Hash) *types.Vertex
}

// ExecutionResult holds the result of executing a block
type ExecutionResult struct {
	BlockHeight    uint64
	TxsExecuted    int
	ReconfigEvents []*types.ReconfigData
}

// Executor manages transaction execution and validator set updates.
//
// G4 fix (epoch safety barrier): Reconfig events discovered during ExecuteBlock
// are NOT applied immediately. Instead they are queued in pendingReconfigs and
// only applied when CommitReconfigs() is called from the globally ordered commit
// path. This guarantees that a configuration change takes effect only after the
// block containing it has been globally confirmed, preserving quorum
// intersection across epochs.
type Executor struct {
	mu       sync.RWMutex
	commitMu sync.Mutex

	// Current Validator Set
	currentValSet *types.ValidatorSet

	// Pending Reconfiguration queue (G4 fix: applied only at commit points)
	pendingReconfigs []*pendingReconfigEntry

	// Vertex resolution (injected, references Mempool)
	vertexResolver VertexResolver

	// Callbacks
	onEpochChange      func(newValSet *types.ValidatorSet, transitions []*types.EpochTransition) error
	reconfigAuthorizer func(data *types.ReconfigData) bool

	// Ordered commit deduplication
	executedOrderedBlocks map[string]struct{}
	appliedOrderedBlocks  map[string]struct{}

	// Stats
	totalBlocksExecuted atomic.Uint64
	totalTxsExecuted    atomic.Uint64
}

// pendingReconfigEntry pairs a reconfig event with the block height that produced it.
// Reconfigs are only applied when that height has been committed.
type pendingReconfigEntry struct {
	blockHeight uint64
	blockHash   []byte
	data        *types.ReconfigData
}

// NewExecutor creates a new executor
func NewExecutor(initialValSet *types.ValidatorSet) *Executor {
	return &Executor{
		currentValSet:         initialValSet,
		executedOrderedBlocks: make(map[string]struct{}),
		appliedOrderedBlocks:  make(map[string]struct{}),
	}
}

// SetVertexResolver injects a VertexResolver (typically the Mempool)
func (e *Executor) SetVertexResolver(vr VertexResolver) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.vertexResolver = vr
}

// SetEpochChangeCallback sets the callback for epoch changes
func (e *Executor) SetEpochChangeCallback(cb func(newValSet *types.ValidatorSet, transitions []*types.EpochTransition) error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.onEpochChange = cb
}

// SetReconfigAuthorizer sets the callback for validating externally authorized
// reconfiguration intents before they are allowed onto the committed path.
func (e *Executor) SetReconfigAuthorizer(cb func(data *types.ReconfigData) bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.reconfigAuthorizer = cb
}

// ExecuteBlock executes transactions in a committed block.
// In Multi-Leader BFT, the block contains VertexCertificates pointing to DAG vertices.
// We resolve each cert to its vertex, then execute the transactions within.
//
// G4 fix: Reconfig events are queued but NOT applied here. They are applied
// only when CommitReconfigs(blockHeight) is called from the commit path.
func (e *Executor) ExecuteBlock(block *types.Block) error {
	e.commitMu.Lock()
	defer e.commitMu.Unlock()
	return e.executeBlockLocked(block)
}

func (e *Executor) executeBlockLocked(block *types.Block) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if block == nil {
		return fmt.Errorf("nil block")
	}

	var txsExecuted int64
	var reconfigEvents []*types.ReconfigData

	// Path 1: Block has DAG vertex certificates (multi-leader path).
	// Resolve all referenced vertices before executing any tx so missing DAG data
	// cannot leave partial execution state behind on retry.
	if len(block.Payload) > 0 && e.vertexResolver != nil {
		vertices, err := e.resolvePayloadVertices(block)
		if err != nil {
			return err
		}
		for _, vertex := range vertices {
			if vertex == nil {
				continue
			}
			for _, tx := range vertex.Txs {
				if tx == nil {
					continue
				}
				reconfig := e.executeTx(tx, block)
				if reconfig != nil {
					reconfigEvents = append(reconfigEvents, reconfig)
				}
				txsExecuted++
			}
		}
	}

	// Path 2: Block has raw Data (single transaction, used by main.go applyReconfigFromBlock)
	if len(block.Data) > 0 && len(block.Payload) == 0 {
		var tx types.Transaction
		if err := json.Unmarshal(block.Data, &tx); err == nil {
			reconfig := e.executeTx(&tx, block)
			if reconfig != nil {
				reconfigEvents = append(reconfigEvents, reconfig)
			}
			txsExecuted++
		}
	}

	e.totalBlocksExecuted.Add(1)
	e.totalTxsExecuted.Add(uint64(txsExecuted))

	logger.Info("Executed block height=%d epoch=%d certs=%d txs_executed=%d reconfigs=%d",
		block.Height, block.Epoch, len(block.Payload), txsExecuted, len(reconfigEvents))

	// G4 fix: Queue reconfig events instead of applying immediately.
	// They will be applied when CommitReconfigs() is called from the commit path.
	for _, reconfig := range reconfigEvents {
		e.pendingReconfigs = append(e.pendingReconfigs, &pendingReconfigEntry{
			blockHeight: block.Height,
			blockHash:   append([]byte(nil), block.Hash...),
			data:        reconfig,
		})
	}

	return nil
}

func (e *Executor) resolvePayloadVertices(block *types.Block) ([]*types.Vertex, error) {
	if block == nil || len(block.Payload) == 0 || e.vertexResolver == nil {
		return nil, nil
	}

	numCerts := len(block.Payload)
	vertices := make([]*types.Vertex, numCerts)
	if numCerts <= 2 {
		for i, cert := range block.Payload {
			if cert == nil {
				continue
			}
			vertex := e.vertexResolver.GetVertex(cert.VertexHash)
			if vertex == nil {
				logger.Error("Executor: vertex %x not found for cert in block height %d", cert.VertexHash[:4], block.Height)
				return nil, fmt.Errorf("missing vertex for cert in block height %d", block.Height)
			}
			vertices[i] = vertex
		}
		return vertices, nil
	}

	resolved := make([]*types.Vertex, numCerts)
	errCh := make(chan error, numCerts)
	var wg sync.WaitGroup
	for i, cert := range block.Payload {
		if cert == nil {
			continue
		}
		wg.Add(1)
		go func(idx int, c *types.VertexCertificate) {
			defer wg.Done()
			vertex := e.vertexResolver.GetVertex(c.VertexHash)
			if vertex == nil {
				logger.Error("Executor: vertex %x not found for cert in block height %d", c.VertexHash[:4], block.Height)
				errCh <- fmt.Errorf("missing vertex for cert in block height %d", block.Height)
				return
			}
			resolved[idx] = vertex
		}(i, cert)
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		if err != nil {
			return nil, err
		}
	}
	return resolved, nil
}

// CommitReconfigs applies only the pending reconfigurations produced by the
// currently globally ordered block. This keeps epoch activation tied to the
// exact ordered barrier that delivered the block.
func (e *Executor) CommitReconfigs(committedBlock *types.Block, activationRank uint64) ([]*types.EpochTransition, error) {
	if committedBlock == nil {
		return nil, nil
	}

	e.commitMu.Lock()
	defer e.commitMu.Unlock()
	return e.commitReconfigsLocked(committedBlock, activationRank)
}

func (e *Executor) commitReconfigsLocked(committedBlock *types.Block, activationRank uint64) ([]*types.EpochTransition, error) {
	if committedBlock == nil {
		return nil, nil
	}
	if len(committedBlock.Hash) == 0 {
		return nil, fmt.Errorf("ordered block missing hash")
	}

	hashKey := string(committedBlock.Hash)
	e.mu.RLock()
	if _, alreadyApplied := e.appliedOrderedBlocks[hashKey]; alreadyApplied {
		e.mu.RUnlock()
		return nil, nil
	}
	pendingSnapshot := clonePendingReconfigs(e.pendingReconfigs)
	baseValSet := e.currentValSet.Copy()
	cb := e.onEpochChange
	_, wasExecuted := e.executedOrderedBlocks[hashKey]
	e.mu.RUnlock()

	var matching []*pendingReconfigEntry
	remaining := make([]*pendingReconfigEntry, 0, len(pendingSnapshot))
	for _, entry := range pendingSnapshot {
		if entry == nil {
			continue
		}
		if entry.blockHeight == committedBlock.Height && bytes.Equal(entry.blockHash, committedBlock.Hash) {
			matching = append(matching, entry)
		} else {
			remaining = append(remaining, entry)
		}
	}
	if len(matching) == 0 {
		if !wasExecuted {
			return nil, nil
		}
		e.mu.Lock()
		if _, alreadyApplied := e.appliedOrderedBlocks[hashKey]; !alreadyApplied {
			e.appliedOrderedBlocks[hashKey] = struct{}{}
		}
		e.mu.Unlock()
		return nil, nil
	}

	candidateValSet := baseValSet
	transitions := make([]*types.EpochTransition, 0, len(matching))
	for _, entry := range matching {
		var transition *types.EpochTransition
		candidateValSet, transition = applyReconfig(candidateValSet, entry.data, entry.blockHeight, activationRank)
		if transition != nil {
			transitions = append(transitions, transition)
		}
	}
	if len(transitions) == 0 {
		e.mu.Lock()
		if _, alreadyApplied := e.appliedOrderedBlocks[hashKey]; !alreadyApplied {
			e.pendingReconfigs = remaining
			e.appliedOrderedBlocks[hashKey] = struct{}{}
		}
		e.mu.Unlock()
		return nil, nil
	}

	transitionClones := cloneEpochTransitions(transitions)
	if cb != nil && candidateValSet != nil {
		if err := cb(candidateValSet.Copy(), transitionClones); err != nil {
			return nil, err
		}
	}

	e.mu.Lock()
	defer e.mu.Unlock()
	if _, alreadyApplied := e.appliedOrderedBlocks[hashKey]; alreadyApplied {
		return nil, nil
	}
	e.currentValSet = candidateValSet
	e.pendingReconfigs = remaining
	e.appliedOrderedBlocks[hashKey] = struct{}{}
	return transitionClones, nil
}

// executeTx executes a single transaction and returns ReconfigData if it's a reconfig tx
func (e *Executor) executeTx(tx *types.Transaction, block *types.Block) *types.ReconfigData {
	switch tx.Type {
	case types.TxTypeNormal:
		// Normal transaction: in this prototype, we just count it.
		// A real system would update application state.
		return nil

	case types.TxTypeReconfig:
		var data types.ReconfigData
		if err := json.Unmarshal(tx.Payload, &data); err != nil {
			logger.Error("Executor: failed to unmarshal reconfig data: %v", err)
			return nil
		}
		switch data.Type {
		case types.ReconfigAutoLeave:
			if !e.verifyAutoLeaveProof(&data, block) {
				logger.Error("Executor: rejected unauthorized auto-leave for node %d", data.NodeID)
				return nil
			}
		default:
			if !e.verifyManualReconfigSignature(&data) {
				logger.Error("Executor: rejected unauthorized reconfig for node %d type=%d", data.NodeID, data.Type)
				return nil
			}
			if e.currentValSet != nil && data.Epoch != e.currentValSet.Epoch {
				logger.Error("Executor: rejected stale reconfig for node %d type=%d payload_epoch=%d current_epoch=%d", data.NodeID, data.Type, data.Epoch, e.currentValSet.Epoch)
				return nil
			}
			if data.Type == types.ReconfigJoin && (e.reconfigAuthorizer == nil || !e.reconfigAuthorizer(&data)) {
				logger.Error("Executor: rejected unauthorized reconfig for node %d type=%d", data.NodeID, data.Type)
				return nil
			}
			if data.Type == types.ReconfigLeave && (e.reconfigAuthorizer == nil || !e.reconfigAuthorizer(&data)) {
				logger.Error("Executor: rejected unauthorized reconfig for node %d type=%d", data.NodeID, data.Type)
				return nil
			}
		}
		return &data

	default:
		return nil
	}
}

func (e *Executor) verifyManualReconfigSignature(data *types.ReconfigData) bool {
	if data == nil {
		return false
	}
	switch data.Type {
	case types.ReconfigJoin:
		return data.VerifySignature()
	case types.ReconfigLeave:
		if e.currentValSet == nil {
			return false
		}
		validator, exists := e.currentValSet.Validators[data.NodeID]
		if !exists || validator == nil || !bytes.Equal(data.PublicKey, validator.PublicKey) {
			return false
		}
		return data.VerifySignature()
	default:
		return false
	}
}

func (e *Executor) verifyAutoLeaveProof(data *types.ReconfigData, block *types.Block) bool {
	if data == nil || e.currentValSet == nil || block == nil {
		return false
	}
	proof := data.AutoLeaveProof
	if proof == nil || len(proof.Leaves) == 0 || len(proof.AutoVotes) == 0 {
		return false
	}
	if data.Epoch != e.currentValSet.Epoch {
		return false
	}
	if !bytes.Equal(proof.BlockHash, block.Hash) {
		return false
	}
	if proof.NewConfigID == 0 {
		return false
	}
	effectiveConfigID := proof.NewConfigID
	leaves := append([]uint64(nil), proof.Leaves...)
	sort.Slice(leaves, func(i, j int) bool { return leaves[i] < leaves[j] })
	pos := sort.Search(len(leaves), func(i int) bool { return leaves[i] >= data.NodeID })
	if pos >= len(leaves) || leaves[pos] != data.NodeID {
		return false
	}
	expectedDigest := autoLeaveProofDigest(proof.View, leaves, proof.BlockHash, effectiveConfigID)
	var accumulatedPower uint64
	seen := make(map[uint64]struct{}, len(proof.AutoVotes))
	for voterID, vote := range proof.AutoVotes {
		if vote == nil || vote.SenderID != voterID {
			return false
		}
		if _, duplicated := seen[voterID]; duplicated {
			return false
		}
		seen[voterID] = struct{}{}
		validator, exists := e.currentValSet.Validators[voterID]
		if !exists || validator == nil || !validator.IsActive {
			return false
		}
		if !bytes.Equal(vote.Digest, expectedDigest) {
			return false
		}
		if !crypto.Verify(expectedDigest, vote.Signature, validator.PublicKey) {
			return false
		}
		power := validator.Power
		if validator.IsActive && power == 0 {
			power = 1
		}
		accumulatedPower += power
	}
	return accumulatedPower >= e.currentValSet.QuorumSize
}

func autoLeaveProofDigest(view uint64, leaves []uint64, blockHash []byte, newConfigID uint64) []byte {
	hasher := sha256.New()
	var buf [8]byte
	binary.BigEndian.PutUint64(buf[:], view)
	hasher.Write(buf[:])
	binary.BigEndian.PutUint64(buf[:], newConfigID)
	hasher.Write(buf[:])
	if len(blockHash) > 0 {
		hasher.Write(blockHash)
	}
	sortedLeaves := append([]uint64(nil), leaves...)
	sort.Slice(sortedLeaves, func(i, j int) bool { return sortedLeaves[i] < sortedLeaves[j] })
	for _, leaf := range sortedLeaves {
		binary.BigEndian.PutUint64(buf[:], leaf)
		hasher.Write(buf[:])
	}
	return hasher.Sum(nil)
}

func cloneEpochTransitions(transitions []*types.EpochTransition) []*types.EpochTransition {
	if len(transitions) == 0 {
		return nil
	}
	clones := make([]*types.EpochTransition, 0, len(transitions))
	for _, transition := range transitions {
		if transition == nil {
			continue
		}
		clone := *transition
		clone.Added = append([]uint64(nil), transition.Added...)
		clone.Removed = append([]uint64(nil), transition.Removed...)
		clone.ConfigHash = append([]byte(nil), transition.ConfigHash...)
		clones = append(clones, &clone)
	}
	return clones
}

func clonePendingReconfigs(entries []*pendingReconfigEntry) []*pendingReconfigEntry {
	if len(entries) == 0 {
		return nil
	}
	clones := make([]*pendingReconfigEntry, 0, len(entries))
	for _, entry := range entries {
		if entry == nil {
			continue
		}
		clone := &pendingReconfigEntry{
			blockHeight: entry.blockHeight,
			blockHash:   append([]byte(nil), entry.blockHash...),
		}
		if entry.data != nil {
			dataCopy := *entry.data
			dataCopy.PublicKey = append([]byte(nil), entry.data.PublicKey...)
			dataCopy.VRFPublicKey = append([]byte(nil), entry.data.VRFPublicKey...)
			dataCopy.Signature = append([]byte(nil), entry.data.Signature...)
			if entry.data.AutoLeaveProof != nil {
				proofCopy := *entry.data.AutoLeaveProof
				proofCopy.Leaves = append([]uint64(nil), entry.data.AutoLeaveProof.Leaves...)
				proofCopy.BlockHash = append([]byte(nil), entry.data.AutoLeaveProof.BlockHash...)
				if entry.data.AutoLeaveProof.AutoVotes != nil {
					proofCopy.AutoVotes = make(map[uint64]*types.HydraAutoVote, len(entry.data.AutoLeaveProof.AutoVotes))
					for voterID, vote := range entry.data.AutoLeaveProof.AutoVotes {
						if vote == nil {
							continue
						}
						voteCopy := *vote
						voteCopy.Signature = append([]byte(nil), vote.Signature...)
						voteCopy.Digest = append([]byte(nil), vote.Digest...)
						proofCopy.AutoVotes[voterID] = &voteCopy
					}
				}
				dataCopy.AutoLeaveProof = &proofCopy
			}
			clone.data = &dataCopy
		}
		clones = append(clones, clone)
	}
	return clones
}

// applyReconfig applies a reconfiguration to the provided validator set and
// returns the resulting validator set plus the globally ordered epoch transition
// produced by that commit barrier.
func applyReconfig(currentValSet *types.ValidatorSet, data *types.ReconfigData, activationHeight uint64, activationRank uint64) (*types.ValidatorSet, *types.EpochTransition) {
	if data == nil || currentValSet == nil {
		return currentValSet, nil
	}

	oldValSet := currentValSet
	newValSet := oldValSet.Copy()
	if newValSet == nil {
		return currentValSet, nil
	}
	newEpoch := data.TargetEpoch
	if newEpoch <= oldValSet.Epoch {
		newEpoch = oldValSet.Epoch + 1
	}

	transition := &types.EpochTransition{
		OldEpoch:         oldValSet.Epoch,
		ActivationHeight: activationHeight,
		ActivationRank:   activationRank,
		Added:            make([]uint64, 0, 1),
		Removed:          make([]uint64, 0, 1),
	}

	switch data.Type {
	case types.ReconfigJoin:
		if _, exists := newValSet.Validators[data.NodeID]; exists {
			return currentValSet, nil
		}
		power := data.Power
		if power == 0 {
			power = 1
		}
		newValSet.Validators[data.NodeID] = &types.Validator{
			ID:           data.NodeID,
			PublicKey:    data.PublicKey,
			VRFPublicKey: append([]byte(nil), data.VRFPublicKey...),
			Power:        power,
			IsActive:     true,
		}
		transition.Added = append(transition.Added, data.NodeID)

	case types.ReconfigLeave:
		if _, exists := newValSet.Validators[data.NodeID]; !exists {
			return currentValSet, nil
		}
		delete(newValSet.Validators, data.NodeID)
		transition.Removed = append(transition.Removed, data.NodeID)
	case types.ReconfigAutoLeave:
		removed := make([]uint64, 0, 1)
		if data.AutoLeaveProof != nil && len(data.AutoLeaveProof.Leaves) > 0 {
			for _, nodeID := range data.AutoLeaveProof.Leaves {
				if _, exists := newValSet.Validators[nodeID]; !exists {
					continue
				}
				delete(newValSet.Validators, nodeID)
				removed = append(removed, nodeID)
			}
		} else {
			if _, exists := newValSet.Validators[data.NodeID]; !exists {
				return currentValSet, nil
			}
			delete(newValSet.Validators, data.NodeID)
			removed = append(removed, data.NodeID)
		}
		if len(removed) == 0 {
			return currentValSet, nil
		}
		sort.Slice(removed, func(i, j int) bool { return removed[i] < removed[j] })
		transition.Removed = append(transition.Removed, removed...)
	default:
		return currentValSet, nil
	}

	// Defense-in-depth: BFT safety invariant guard (Theorem reconfig-safety).
	// Reject any reconfiguration that would drop below minimum BFT threshold
	// (n >= 4 for f >= 1) or remove more than f validators in a single block.
	if data.Type == types.ReconfigLeave || data.Type == types.ReconfigAutoLeave {
		nAfter := len(newValSet.Validators)
		if nAfter < 4 {
			logger.Error("Executor: BFT invariant guard REJECTED reconfig: n_after=%d < 4", nAfter)
			return currentValSet, nil
		}
		maxRemovable := (len(oldValSet.Validators) - 1) / 3
		nRemoved := len(oldValSet.Validators) - nAfter
		if nRemoved > maxRemovable {
			logger.Error("Executor: BFT invariant guard REJECTED: removing %d > f=%d in one block", nRemoved, maxRemovable)
			return currentValSet, nil
		}
	}

	newValSet = types.NewValidatorSet(newEpoch, newValSet.Validators)
	transition.NewEpoch = newValSet.Epoch
	transition.QuorumSize = newValSet.QuorumSize
	transition.ConfigHash = append([]byte(nil), newValSet.Hash()...)
	if len(transition.Added) > 0 {
		logger.Info("Executor: Node %d joined at epoch %d (validators=%d quorum=%d total_power=%d)",
			data.NodeID, newValSet.Epoch, len(newValSet.Validators), newValSet.QuorumSize, newValSet.TotalPower)
	} else {
		logger.Info("Executor: Node %d left (type=%d) at epoch %d (validators=%d quorum=%d total_power=%d)",
			data.NodeID, data.Type, newValSet.Epoch, len(newValSet.Validators), newValSet.QuorumSize, newValSet.TotalPower)
	}
	return newValSet, transition
}

// ApplyOrderedBlock executes and commits a globally ordered block exactly once.
func (e *Executor) ApplyOrderedBlock(block *types.Block, activationRank uint64) ([]*types.EpochTransition, error) {
	if block == nil {
		return nil, nil
	}
	if len(block.Hash) == 0 {
		return nil, fmt.Errorf("ordered block missing hash")
	}

	hashKey := string(block.Hash)
	e.commitMu.Lock()
	defer e.commitMu.Unlock()

	e.mu.Lock()
	if _, alreadyApplied := e.appliedOrderedBlocks[hashKey]; alreadyApplied {
		e.mu.Unlock()
		return nil, nil
	}
	if _, alreadyExecuted := e.executedOrderedBlocks[hashKey]; alreadyExecuted {
		e.mu.Unlock()
		return e.commitReconfigsLocked(block, activationRank)
	}
	e.executedOrderedBlocks[hashKey] = struct{}{}
	e.mu.Unlock()

	if err := e.executeBlockLocked(block); err != nil {
		e.mu.Lock()
		delete(e.executedOrderedBlocks, hashKey)
		e.mu.Unlock()
		return nil, err
	}
	return e.commitReconfigsLocked(block, activationRank)
}

// GetCurrentValidatorSet returns a copy of the current validator set.
func (e *Executor) GetCurrentValidatorSet() *types.ValidatorSet {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.currentValSet == nil {
		return nil
	}
	return e.currentValSet.Copy()
}

// Stats returns execution statistics
func (e *Executor) Stats() (blocksExecuted, txsExecuted uint64) {
	return e.totalBlocksExecuted.Load(), e.totalTxsExecuted.Load()
}

func (e *Executor) PendingReconfigCount() int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return len(e.pendingReconfigs)
}
