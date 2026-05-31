// Copyright 2024 Evolv-BFT Project
// Licensed under Apache License 2.0

package hydra

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"sort"
	"sync"
	"time"

	"evolvbft/evolvbft/crypto"
	"evolvbft/evolvbft/types"
)

var logger = struct {
	Info  func(format string, args ...interface{})
	Warn  func(format string, args ...interface{})
	Error func(format string, args ...interface{})
}{
	Info:  func(format string, args ...interface{}) {},
	Warn:  func(format string, args ...interface{}) {},
	Error: func(format string, args ...interface{}) {},
}

// autoVoteCollector 收集 auto-transition 的投票并验证 quorum。
// 只有在收集到 quorumSize 个不重复的有效投票后才允许 commit。
type autoVoteCollector struct {
	view         uint64
	voters       map[uint64]struct{} // 已验证并绑定到候选配置的 validator ID（去重）
	votes        map[uint64]*Vote
	pendingVotes map[uint64]*AutoTransitionMessage
	done         bool
	quorum       int
	timestamp    time.Time
}

type pendingAutoTransition struct {
	view      uint64
	config    *Configuration
	leaves    []uint64
	blockHash []byte
}

// AutoTransitionManager implements Hydra's configuration auto-transition protocol
// Triggered when leader times out and cannot make progress.
//
// 安全模型：auto-transition 采用与共识相同的 2f+1 quorum 收集机制。
// 单个 AutoVote 不可触发配置变更（防止拜占庭节点随意踢除诚实节点）。
type AutoTransitionManager struct {
	mu sync.RWMutex

	// Components
	lSetManager       *LSetManager
	tempConfigManager *TCM
	network           NetworkInterface
	privateKey        types.PrivateKey

	// State
	pendingAutoConfigs map[uint64]*pendingAutoTransition
	isRunning          bool
	nodeID             uint64 // 当前节点 ID

	// Quorum-based vote collectors (keyed by view)
	voteCollectors     map[uint64]*autoVoteCollector
	recentMessages     map[autoTransitionKey]struct{}
	finalizedViews     map[uint64]struct{}
	finalizedConfigIDs map[uint64]struct{}
	lastCommittedView  uint64

	// Configuration
	maxSeq          int // maximum sequence attempts
	timeoutDuration time.Duration

	// Callbacks
	onTransition func(*Configuration, *TransitionProof)
}

type autoTransitionKey struct {
	messageType AutoTransitionType
	senderID    uint64
	view        uint64
}

func autoTransitionDigest(view uint64, leaves []uint64, blockHash []byte, newConfigID uint64) []byte {
	hasher := sha256.New()
	var buf [8]byte
	binary.BigEndian.PutUint64(buf[:], view)
	hasher.Write(buf[:])
	binary.BigEndian.PutUint64(buf[:], newConfigID)
	hasher.Write(buf[:])
	if len(blockHash) > 0 {
		hasher.Write(blockHash)
	}
	orderedLeaves := append([]uint64(nil), leaves...)
	sort.Slice(orderedLeaves, func(i, j int) bool { return orderedLeaves[i] < orderedLeaves[j] })
	for _, leaf := range orderedLeaves {
		binary.BigEndian.PutUint64(buf[:], leaf)
		hasher.Write(buf[:])
	}
	return hasher.Sum(nil)
}

// AutoTransitionMessage represents AUTO message
type AutoTransitionMessage struct {
	Type        AutoTransitionType
	SenderID    uint64
	View        uint64
	BlockHash   []byte
	Leaves      []uint64 // IDs to automatically leave
	NewConfigID uint64
	Proof       *TransitionProof
	Signature   []byte
}

func (msg *AutoTransitionMessage) SigningBytes() ([]byte, error) {
	if msg == nil {
		return nil, nil
	}
	copyMsg := *msg
	copyMsg.Signature = nil
	return json.Marshal(&copyMsg)
}

func (msg *AutoTransitionMessage) VoteDigest() []byte {
	if msg == nil || msg.Type != AutoVote || msg.NewConfigID == 0 {
		return nil
	}
	return autoTransitionDigest(msg.View, msg.Leaves, msg.BlockHash, msg.NewConfigID)
}

func (msg *AutoTransitionMessage) Sign(privateKey types.PrivateKey) error {
	if msg == nil {
		return fmt.Errorf("auto-transition message is nil")
	}
	if len(privateKey) == 0 {
		return fmt.Errorf("auto-transition private key is not configured")
	}
	payload := msg.VoteDigest()
	if len(payload) == 0 {
		var err error
		payload, err = msg.SigningBytes()
		if err != nil {
			return err
		}
	}
	msg.Signature = crypto.Sign(payload, privateKey)
	return nil
}

func (msg *AutoTransitionMessage) VerifySignature(publicKey types.PublicKey) bool {
	if msg == nil || len(msg.Signature) == 0 || len(publicKey) != 32 {
		return false
	}
	payload := msg.VoteDigest()
	if len(payload) == 0 {
		var err error
		payload, err = msg.SigningBytes()
		if err != nil {
			return false
		}
	}
	return crypto.Verify(payload, msg.Signature, publicKey)
}

// AutoTransitionType enum
type AutoTransitionType int

const (
	AutoPropose AutoTransitionType = iota
	AutoVote
	AutoCommit
)

// TransitionProof contains proofs for auto-transition
type TransitionProof struct {
	View        uint64
	AutoVotes   map[uint64]*Vote
	QuorumProof []byte
	NewConfigID uint64
	Leaves      []uint64
	BlockHash   []byte
}

// NewAutoTransitionManager creates a new auto-transition manager
func NewAutoTransitionManager(
	lSet *LSetManager,
	tcm *TCM,
	network NetworkInterface,
) *ATM {
	return &ATM{
		lSetManager:        lSet,
		tempConfigManager:  tcm,
		network:            network,
		pendingAutoConfigs: make(map[uint64]*pendingAutoTransition),
		voteCollectors:     make(map[uint64]*autoVoteCollector),
		recentMessages:     make(map[autoTransitionKey]struct{}),
		finalizedViews:     make(map[uint64]struct{}),
		finalizedConfigIDs: make(map[uint64]struct{}),
		lastCommittedView:  0,
		maxSeq:             3, // retry 3 times before giving up
		timeoutDuration:    10 * time.Second,
	}
}

// ATM is shorthand
type ATM = AutoTransitionManager

// SetNodeID 设置当前节点 ID（由 HydraManager 在初始化时调用）
func (atm *ATM) SetNodeID(id uint64) {
	atm.mu.Lock()
	defer atm.mu.Unlock()
	atm.nodeID = id
}

func (atm *ATM) SetPrivateKey(privateKey types.PrivateKey) {
	atm.mu.Lock()
	defer atm.mu.Unlock()
	atm.privateKey = append(types.PrivateKey(nil), privateKey...)
}

// Start starts the auto-transition manager
func (atm *ATM) Start() {
	atm.mu.Lock()
	atm.isRunning = true
	atm.mu.Unlock()

	logger.Info("Auto-transition manager started")
}

// Stop stops the auto-transition manager
func (atm *ATM) Stop() {
	atm.mu.Lock()
	defer atm.mu.Unlock()
	atm.isRunning = false
	logger.Info("Auto-transition manager stopped")
}

// TriggerAutoTransition triggers auto-transition when leader times out
func (atm *ATM) TriggerAutoTransition(
	leaderID uint64,
	view uint64,
) error {

	logger.Info("Triggering auto-transition for leader %d view %d", leaderID, view)

	atm.mu.RLock()
	localNodeID := atm.nodeID
	privateKey := append(types.PrivateKey(nil), atm.privateKey...)
	network := atm.network
	atm.mu.RUnlock()

	// Only the local node, if it belongs to the committed L-set, may initiate
	// auto-transition for a timed-out leader.
	if !atm.lSetManager.IsInL(localNodeID) {
		return fmt.Errorf("only L-set members can trigger auto-transition")
	}

	// Identify misbehaving replicas from mmtable
	misbehaving := atm.identifyMisbehaving()
	if len(misbehaving) == 0 {
		return fmt.Errorf("auto-transition requires at least one evictable target")
	}

	// Create auto-transition message
	msg := &AutoTransitionMessage{
		Type:     AutoPropose,
		SenderID: localNodeID,
		View:     view,
		Leaves:   make([]uint64, len(misbehaving)),
	}

	for i, id := range misbehaving {
		msg.Leaves[i] = id
	}

	if network != nil {
		if err := msg.Sign(privateKey); err != nil {
			return err
		}
		network.Broadcast(msg)
	}

	return nil
}

// identifyMisbehaving identifies replicas to be removed
func (atm *ATM) identifyMisbehaving() []uint64 {
	var result []uint64

	// Get L-set
	lSet := atm.lSetManager.GetLSet()

	// Check mmtable for evictable replicas only
	for id := range lSet {
		if atm.lSetManager.HasEvictableFault(id) {
			result = append(result, id)
		}
	}

	return result
}

// HandleAutoTransitionMessage handles received auto-transition message
func (atm *ATM) HandleAutoTransitionMessage(msg *AutoTransitionMessage) error {
	if msg == nil {
		return fmt.Errorf("auto-transition message is nil")
	}
	logger.Info("Handling auto-transition message from %d type %s", msg.SenderID, msg.Type.String())
	if err := atm.verifyMessageSignature(msg); err != nil {
		return err
	}

	switch msg.Type {
	case AutoPropose:
		return atm.handleAutoPropose(msg)
	case AutoVote:
		return atm.handleAutoVote(msg)
	case AutoCommit:
		return atm.handleAutoCommit(msg)
	}

	return nil
}

func (atm *ATM) rejectStaleView(view uint64) error {
	if _, finalized := atm.finalizedViews[view]; finalized {
		return fmt.Errorf("finalized auto-transition view %d rejected", view)
	}
	if view <= atm.lastCommittedView {
		return fmt.Errorf("stale auto-transition view %d rejected; last committed view %d", view, atm.lastCommittedView)
	}
	return nil
}

func (atm *ATM) rejectReplay(msgType AutoTransitionType, senderID uint64, view uint64) error {
	key := autoTransitionKey{messageType: msgType, senderID: senderID, view: view}
	if _, exists := atm.recentMessages[key]; exists {
		return fmt.Errorf("replayed %s from sender %d at view %d rejected", msgType.String(), senderID, view)
	}
	atm.recentMessages[key] = struct{}{}
	return nil
}

func (atm *ATM) validateActiveValidator(senderID uint64, msgType AutoTransitionType) error {
	currentConfig := atm.tempConfigManager.GetMvalid()
	if _, isValidator := currentConfig.Validators[senderID]; !isValidator {
		return fmt.Errorf("%s from non-validator %d rejected", msgType.String(), senderID)
	}
	return nil
}

func (atm *ATM) verifyMessageSignature(msg *AutoTransitionMessage) error {
	if msg == nil {
		return fmt.Errorf("auto-transition message is nil")
	}
	currentConfig := atm.tempConfigManager.GetMvalid()
	validator, exists := currentConfig.Validators[msg.SenderID]
	if !exists || validator == nil {
		return fmt.Errorf("%s from non-validator %d rejected", msg.Type.String(), msg.SenderID)
	}
	if !msg.VerifySignature(validator.PublicKey) {
		return fmt.Errorf("invalid %s signature from sender %d", msg.Type.String(), msg.SenderID)
	}
	return nil
}

func (atm *ATM) pendingAutoConfigForView(view uint64) *pendingAutoTransition {
	return atm.pendingAutoConfigs[view]
}

func (atm *ATM) storePendingAutoConfig(view uint64, config *Configuration, leaves []uint64, blockHash []byte) *pendingAutoTransition {
	pending := &pendingAutoTransition{
		view:      view,
		config:    config.Copy(),
		leaves:    append([]uint64(nil), leaves...),
		blockHash: append([]byte(nil), blockHash...),
	}
	atm.pendingAutoConfigs[view] = pending
	return pending
}

func copyAutoVoteMessage(msg *AutoTransitionMessage) *AutoTransitionMessage {
	if msg == nil {
		return nil
	}
	return &AutoTransitionMessage{
		Type:        msg.Type,
		SenderID:    msg.SenderID,
		View:        msg.View,
		BlockHash:   append([]byte(nil), msg.BlockHash...),
		Leaves:      append([]uint64(nil), msg.Leaves...),
		NewConfigID: msg.NewConfigID,
		Signature:   append([]byte(nil), msg.Signature...),
	}
}

func validateAutoVoteForPending(msg *AutoTransitionMessage, pending *pendingAutoTransition) ([]byte, error) {
	if pending == nil || pending.config == nil {
		return nil, fmt.Errorf("missing pending auto configuration")
	}
	expectedDigest := autoTransitionDigest(msg.View, pending.leaves, pending.blockHash, pending.config.ID)
	if msg.NewConfigID != pending.config.ID {
		return nil, fmt.Errorf("auto-vote config mismatch: %d != %d", msg.NewConfigID, pending.config.ID)
	}
	if !bytes.Equal(msg.BlockHash, pending.blockHash) {
		return nil, fmt.Errorf("auto-vote block hash mismatch")
	}
	if !sameUint64Set(msg.Leaves, pending.leaves) {
		return nil, fmt.Errorf("auto-vote leaves mismatch")
	}
	if !bytes.Equal(msg.VoteDigest(), expectedDigest) {
		return nil, fmt.Errorf("auto-vote digest mismatch")
	}
	return expectedDigest, nil
}

func (atm *ATM) validateAutoProposeLeaves(msg *AutoTransitionMessage) error {
	if msg == nil {
		return fmt.Errorf("auto-propose message is nil")
	}
	if len(msg.Leaves) == 0 {
		return fmt.Errorf("auto-propose missing leave targets")
	}
	lSet := atm.lSetManager.GetLSet()
	seen := make(map[uint64]struct{}, len(msg.Leaves))
	for _, id := range msg.Leaves {
		if _, duplicated := seen[id]; duplicated {
			return fmt.Errorf("auto-propose contains duplicate leave target %d", id)
		}
		seen[id] = struct{}{}
		if _, exists := lSet[id]; !exists {
			return fmt.Errorf("auto-propose targets node %d outside current L-set", id)
		}
		faultClass := atm.lSetManager.GetFaultClass(id)
		if !faultClass.Evictable() {
			return fmt.Errorf("auto-propose targets non-evictable node %d fault_class=%s", id, faultClass.String())
		}
	}
	return nil
}

func validateAutoCommitLeaves(currentConfig *Configuration, lSetManager *LSetManager, leaves []uint64) error {
	if len(leaves) == 0 {
		return fmt.Errorf("auto-commit missing leave targets")
	}
	if currentConfig == nil {
		return fmt.Errorf("current configuration is nil")
	}
	if lSetManager == nil {
		return fmt.Errorf("l-set manager is nil")
	}
	seen := make(map[uint64]struct{}, len(leaves))
	for _, id := range leaves {
		if _, duplicated := seen[id]; duplicated {
			return fmt.Errorf("auto-commit contains duplicate leave target %d", id)
		}
		seen[id] = struct{}{}
		if _, exists := currentConfig.Validators[id]; !exists {
			return fmt.Errorf("auto-commit targets unknown validator %d", id)
		}
		if !lSetManager.IsInL(id) {
			return fmt.Errorf("auto-commit targets node %d outside current L-set", id)
		}
		faultClass := lSetManager.GetFaultClass(id)
		if !faultClass.Evictable() {
			return fmt.Errorf("auto-commit targets non-evictable node %d fault_class=%s", id, faultClass.String())
		}
	}
	return nil
}

func buildPendingAutoTransitionForCommit(currentConfig *Configuration, lSetManager *LSetManager, view uint64, leaves []uint64, blockHash []byte, newConfigID uint64) (*pendingAutoTransition, error) {
	if currentConfig == nil {
		return nil, fmt.Errorf("current configuration is nil")
	}
	if newConfigID == 0 {
		return nil, fmt.Errorf("auto-commit missing new config id")
	}
	if newConfigID <= currentConfig.ID {
		return nil, fmt.Errorf("auto-commit new config id %d must exceed committed config id %d", newConfigID, currentConfig.ID)
	}
	if err := validateAutoCommitLeaves(currentConfig, lSetManager, leaves); err != nil {
		return nil, err
	}
	candidate := currentConfig.Copy()
	for _, id := range leaves {
		delete(candidate.Validators, id)
	}
	candidate.QuorumSize = (2*len(candidate.Validators))/3 + 1
	candidate.ID = newConfigID
	return &pendingAutoTransition{
		view:      view,
		config:    candidate,
		leaves:    append([]uint64(nil), leaves...),
		blockHash: append([]byte(nil), blockHash...),
	}, nil
}

func (atm *ATM) bindPendingVotesLocked(view uint64) {
	collector := atm.voteCollectors[view]
	pending := atm.pendingAutoConfigForView(view)
	if collector == nil || pending == nil || pending.config == nil || len(collector.pendingVotes) == 0 {
		return
	}
	for senderID, msg := range collector.pendingVotes {
		if msg == nil {
			delete(collector.pendingVotes, senderID)
			continue
		}
		digest, err := validateAutoVoteForPending(msg, pending)
		if err != nil {
			delete(collector.pendingVotes, senderID)
			continue
		}
		collector.voters[senderID] = struct{}{}
		collector.votes[senderID] = &Vote{
			SenderID:  msg.SenderID,
			Signature: append([]byte(nil), msg.Signature...),
			Digest:    append([]byte(nil), digest...),
		}
		delete(collector.pendingVotes, senderID)
	}
}

func (atm *ATM) transitionProofForView(view uint64) *TransitionProof {
	collector := atm.voteCollectors[view]
	pending := atm.pendingAutoConfigForView(view)
	if collector == nil || pending == nil || pending.config == nil {
		return nil
	}
	proof := &TransitionProof{
		View:        view,
		AutoVotes:   make(map[uint64]*Vote, len(collector.votes)),
		NewConfigID: pending.config.ID,
		Leaves:      append([]uint64(nil), pending.leaves...),
		BlockHash:   append([]byte(nil), pending.blockHash...),
	}
	for voterID, vote := range collector.votes {
		if vote == nil {
			continue
		}
		dup := *vote
		if vote.Signature != nil {
			dup.Signature = append([]byte(nil), vote.Signature...)
		}
		if vote.Digest != nil {
			dup.Digest = append([]byte(nil), vote.Digest...)
		}
		proof.AutoVotes[voterID] = &dup
	}
	return proof
}

func (atm *ATM) commitReadyForView(view uint64) (*Configuration, func(*Configuration, *TransitionProof), *TransitionProof, error) {
	collector := atm.voteCollectors[view]
	if collector == nil || collector.done || len(collector.votes) < collector.quorum {
		return nil, nil, nil, nil
	}
	pending := atm.pendingAutoConfigForView(view)
	if pending == nil {
		return nil, nil, nil, fmt.Errorf("no pending auto configuration for view %d", view)
	}
	proof := atm.transitionProofForView(view)
	committedConfig, callback, err := atm.commitAutoTransitionLocked(&AutoTransitionMessage{View: view})
	if err != nil {
		return nil, nil, nil, err
	}
	collector.done = true
	return committedConfig, callback, proof, nil
}

func (atm *ATM) commitPendingView(view uint64) (*Configuration, func(*Configuration, *TransitionProof), error) {
	pending := atm.pendingAutoConfigForView(view)
	if pending == nil {
		return nil, nil, fmt.Errorf("no pending auto configuration for view %d", view)
	}
	return atm.commitAutoTransitionLocked(&AutoTransitionMessage{View: view})
}

// handleAutoPropose handles auto-propose
func (atm *ATM) handleAutoPropose(msg *AutoTransitionMessage) error {
	var (
		committedConfig *Configuration
		callback        func(*Configuration, *TransitionProof)
		proof           *TransitionProof
		err             error
	)

	// Verify sender is from L-set
	if !atm.lSetManager.IsInL(msg.SenderID) {
		return fmt.Errorf("sender not in L-set")
	}

	func() {
		atm.mu.Lock()
		defer atm.mu.Unlock()

		if err = atm.rejectStaleView(msg.View); err != nil {
			return
		}
		if err = atm.rejectReplay(msg.Type, msg.SenderID, msg.View); err != nil {
			return
		}
		if err = atm.validateActiveValidator(msg.SenderID, msg.Type); err != nil {
			return
		}
		if err = atm.validateAutoProposeLeaves(msg); err != nil {
			return
		}

		existing := atm.pendingAutoConfigForView(msg.View)
		if existing != nil {
			if !sameUint64Set(existing.leaves, msg.Leaves) || !bytes.Equal(existing.blockHash, msg.BlockHash) {
				err = fmt.Errorf("conflicting auto-propose candidate for view %d", msg.View)
				return
			}
		} else {
			newConfig := atm.tempConfigManager.GetMvalid()
			for _, id := range msg.Leaves {
				delete(newConfig.Validators, id)
			}
			newConfig.QuorumSize = (2*len(newConfig.Validators))/3 + 1
			newConfig.ID = atm.tempConfigManager.ReserveNextConfigID()
			atm.storePendingAutoConfig(msg.View, newConfig, msg.Leaves, msg.BlockHash)
		}

		currentConfig := atm.tempConfigManager.GetMvalid()
		quorum := (2*len(currentConfig.Validators))/3 + 1
		if _, exists := atm.voteCollectors[msg.View]; !exists {
			atm.voteCollectors[msg.View] = &autoVoteCollector{
				view:         msg.View,
				voters:       make(map[uint64]struct{}),
				votes:        make(map[uint64]*Vote),
				pendingVotes: make(map[uint64]*AutoTransitionMessage),
				quorum:       quorum,
				timestamp:    time.Now(),
			}
		}

		atm.bindPendingVotesLocked(msg.View)
		committedConfig, callback, proof, err = atm.commitReadyForView(msg.View)
		if err != nil {
			return
		}
	}()

	if err != nil {
		return err
	}
	if committedConfig == nil {
		atm.mu.RLock()
		localNodeID := atm.nodeID
		privateKey := append(types.PrivateKey(nil), atm.privateKey...)
		network := atm.network
		pending := atm.pendingAutoConfigForView(msg.View)
		var voteBlockHash []byte
		var voteLeaves []uint64
		var voteConfigID uint64
		if pending != nil && pending.config != nil {
			voteBlockHash = append([]byte(nil), pending.blockHash...)
			voteLeaves = append([]uint64(nil), pending.leaves...)
			voteConfigID = pending.config.ID
		}
		atm.mu.RUnlock()
		if network != nil {
			if voteConfigID == 0 {
				return fmt.Errorf("missing pending auto configuration for vote view %d", msg.View)
			}
			vote := &AutoTransitionMessage{
				Type:        AutoVote,
				SenderID:    localNodeID,
				View:        msg.View,
				BlockHash:   voteBlockHash,
				Leaves:      voteLeaves,
				NewConfigID: voteConfigID,
			}
			if err := vote.Sign(privateKey); err != nil {
				return err
			}
			network.Broadcast(vote)
		}
	}
	if callback != nil {
		callback(committedConfig, proof)
	}
	return nil
}

// handleAutoVote 收集 auto-transition 投票，只有在达到 2f+1 quorum 后才 commit。
// 这是 G2 修复的核心：防止拜占庭节点通过单个 AutoVote 触发配置变更。
func (atm *ATM) handleAutoVote(msg *AutoTransitionMessage) error {
	var (
		committedConfig *Configuration
		callback        func(*Configuration, *TransitionProof)
		proof           *TransitionProof
		err             error
	)

	func() {
		atm.mu.Lock()
		defer atm.mu.Unlock()

		if err = atm.rejectStaleView(msg.View); err != nil {
			return
		}
		if err = atm.validateActiveValidator(msg.SenderID, msg.Type); err != nil {
			return
		}

		pending := atm.pendingAutoConfigForView(msg.View)
		if pending == nil || pending.config == nil {
			if len(msg.Leaves) == 0 {
				err = fmt.Errorf("auto-vote missing leave targets")
				return
			}
			if msg.NewConfigID == 0 {
				err = fmt.Errorf("auto-vote missing new config id")
				return
			}
			err = fmt.Errorf("auto-vote without pending proposal rejected")
			return
		}
		if err = atm.rejectReplay(msg.Type, msg.SenderID, msg.View); err != nil {
			return
		}

		collector, exists := atm.voteCollectors[msg.View]
		if !exists {
			currentConfig := atm.tempConfigManager.GetMvalid()
			quorum := (2*len(currentConfig.Validators))/3 + 1
			collector = &autoVoteCollector{
				view:         msg.View,
				voters:       make(map[uint64]struct{}),
				votes:        make(map[uint64]*Vote),
				pendingVotes: make(map[uint64]*AutoTransitionMessage),
				quorum:       quorum,
				timestamp:    time.Now(),
			}
			atm.voteCollectors[msg.View] = collector
		}

		if collector.done {
			return
		}
		if _, voted := collector.voters[msg.SenderID]; voted {
			err = fmt.Errorf("replayed %s from sender %d at view %d rejected", msg.Type.String(), msg.SenderID, msg.View)
			return
		}

		expectedDigest, validateErr := validateAutoVoteForPending(msg, pending)
		if validateErr != nil {
			err = validateErr
			return
		}
		collector.voters[msg.SenderID] = struct{}{}
		collector.votes[msg.SenderID] = &Vote{
			SenderID:  msg.SenderID,
			Signature: append([]byte(nil), msg.Signature...),
			Digest:    append([]byte(nil), expectedDigest...),
		}

		logger.Info("Auto-transition vote collected: view=%d voter=%d votes=%d/%d",
			msg.View, msg.SenderID, len(collector.votes), collector.quorum)

		if len(collector.votes) < collector.quorum {
			return
		}

		committedConfig, callback, proof, err = atm.commitReadyForView(msg.View)
	}()

	if err != nil {
		return err
	}
	if callback != nil {
		callback(committedConfig, proof)
	}
	return nil
}

// handleAutoCommit handles auto-commit message from peers
func (atm *ATM) handleAutoCommit(msg *AutoTransitionMessage) error {
	var (
		committedConfig *Configuration
		callback        func(*Configuration, *TransitionProof)
		proof           *TransitionProof
		err             error
	)

	func() {
		atm.mu.Lock()
		defer atm.mu.Unlock()

		if err = atm.rejectStaleView(msg.View); err != nil {
			return
		}
		if err = atm.rejectReplay(msg.Type, msg.SenderID, msg.View); err != nil {
			return
		}
		if err = atm.validateActiveValidator(msg.SenderID, msg.Type); err != nil {
			return
		}
		if !atm.lSetManager.IsInL(msg.SenderID) {
			err = fmt.Errorf("sender not in L-set")
			return
		}

		if msg.Proof == nil || len(msg.Proof.AutoVotes) == 0 {
			err = fmt.Errorf("auto-commit without proof rejected")
			return
		}
		if msg.Proof.View != msg.View {
			err = fmt.Errorf("auto-commit proof view mismatch")
			return
		}
		if msg.NewConfigID != 0 && msg.Proof.NewConfigID != 0 && msg.NewConfigID != msg.Proof.NewConfigID {
			err = fmt.Errorf("auto-commit top-level/proof config id mismatch: %d != %d", msg.NewConfigID, msg.Proof.NewConfigID)
			return
		}
		proofLeaves := append([]uint64(nil), msg.Proof.Leaves...)
		if len(msg.Leaves) > 0 {
			if len(proofLeaves) > 0 && !sameUint64Set(msg.Leaves, proofLeaves) {
				err = fmt.Errorf("auto-commit top-level/proof leaves mismatch")
				return
			}
			if len(proofLeaves) == 0 {
				proofLeaves = append([]uint64(nil), msg.Leaves...)
			}
		}
		if !bytes.Equal(msg.Proof.BlockHash, msg.BlockHash) {
			err = fmt.Errorf("auto-commit proof block hash mismatch")
			return
		}

		currentConfig := atm.tempConfigManager.GetMvalid()
		quorum := (2*len(currentConfig.Validators))/3 + 1
		if len(msg.Proof.AutoVotes) < quorum {
			err = fmt.Errorf("auto-commit proof has insufficient votes: %d < %d", len(msg.Proof.AutoVotes), quorum)
			return
		}

		pending := atm.pendingAutoConfigForView(msg.View)
		newConfigID := msg.Proof.NewConfigID
		if _, finalized := atm.finalizedConfigIDs[newConfigID]; finalized {
			err = fmt.Errorf("auto-commit proof config id %d already finalized", newConfigID)
			return
		}
		if pending == nil {
			pending, err = buildPendingAutoTransitionForCommit(currentConfig, atm.lSetManager, msg.View, proofLeaves, msg.Proof.BlockHash, newConfigID)
			if err != nil {
				return
			}
			for view, otherPending := range atm.pendingAutoConfigs {
				if view == msg.View || otherPending == nil || otherPending.config == nil {
					continue
				}
				if otherPending.config.ID == pending.config.ID {
					err = fmt.Errorf("auto-commit proof config id %d collides with pending view %d", pending.config.ID, view)
					return
				}
			}
			atm.pendingAutoConfigs[msg.View] = pending
		} else if !sameUint64Set(proofLeaves, pending.leaves) {
			err = fmt.Errorf("auto-commit leaves mismatch for pending view %d", msg.View)
			return
		} else if !bytes.Equal(pending.blockHash, msg.Proof.BlockHash) {
			err = fmt.Errorf("auto-commit block hash mismatch for pending view %d", msg.View)
			return
		}
		if newConfigID == 0 {
			newConfigID = pending.config.ID
		} else {
			if newConfigID <= currentConfig.ID {
				err = fmt.Errorf("auto-commit proof config id %d must exceed committed config id %d", newConfigID, currentConfig.ID)
				return
			}
			if pending.config != nil && pending.config.ID != 0 && newConfigID != pending.config.ID {
				err = fmt.Errorf("auto-commit proof config id mismatch: %d != %d", newConfigID, pending.config.ID)
				return
			}
			for view, otherPending := range atm.pendingAutoConfigs {
				if view == msg.View || otherPending == nil || otherPending.config == nil {
					continue
				}
				if otherPending.config.ID == newConfigID {
					err = fmt.Errorf("auto-commit proof config id %d collides with pending view %d", newConfigID, view)
					return
				}
			}
			atm.tempConfigManager.ObserveReservedConfigID(newConfigID)
			pending.config.ID = newConfigID
		}
		pending.blockHash = append([]byte(nil), msg.Proof.BlockHash...)
		expectedDigest := autoTransitionDigest(msg.View, pending.leaves, pending.blockHash, newConfigID)

		seen := make(map[uint64]struct{}, len(msg.Proof.AutoVotes))
		for voterID, vote := range msg.Proof.AutoVotes {
			validator, exists := currentConfig.Validators[voterID]
			if !exists || validator == nil {
				err = fmt.Errorf("auto-commit proof includes non-validator %d", voterID)
				return
			}
			if _, duplicated := seen[voterID]; duplicated {
				err = fmt.Errorf("auto-commit proof includes duplicate voter %d", voterID)
				return
			}
			seen[voterID] = struct{}{}
			if vote == nil {
				err = fmt.Errorf("auto-commit proof missing vote for %d", voterID)
				return
			}
			if vote.SenderID != voterID {
				err = fmt.Errorf("auto-commit proof signer mismatch for voter %d", voterID)
				return
			}
			if len(vote.Digest) == 0 {
				err = fmt.Errorf("auto-commit proof missing digest for voter %d", voterID)
				return
			}
			if !bytes.Equal(vote.Digest, expectedDigest) {
				err = fmt.Errorf("auto-commit proof digest mismatch for voter %d", voterID)
				return
			}
			if !crypto.Verify(expectedDigest, vote.Signature, validator.PublicKey) {
				err = fmt.Errorf("auto-commit proof signature invalid for voter %d", voterID)
				return
			}
		}

		if collector := atm.voteCollectors[msg.View]; collector == nil {
			atm.voteCollectors[msg.View] = &autoVoteCollector{
				view:         msg.View,
				voters:       make(map[uint64]struct{}, len(msg.Proof.AutoVotes)),
				votes:        make(map[uint64]*Vote, len(msg.Proof.AutoVotes)),
				pendingVotes: make(map[uint64]*AutoTransitionMessage),
				quorum:       quorum,
				timestamp:    time.Now(),
			}
		}
		for voterID, vote := range msg.Proof.AutoVotes {
			atm.voteCollectors[msg.View].voters[voterID] = struct{}{}
			if vote == nil {
				continue
			}
			dup := *vote
			if vote.Signature != nil {
				dup.Signature = append([]byte(nil), vote.Signature...)
			}
			if vote.Digest != nil {
				dup.Digest = append([]byte(nil), vote.Digest...)
			}
			atm.voteCollectors[msg.View].votes[voterID] = &dup
		}

		proof = &TransitionProof{
			View:        msg.Proof.View,
			AutoVotes:   make(map[uint64]*Vote, len(msg.Proof.AutoVotes)),
			QuorumProof: append([]byte(nil), msg.Proof.QuorumProof...),
			NewConfigID: newConfigID,
			Leaves:      append([]uint64(nil), pending.leaves...),
			BlockHash:   append([]byte(nil), msg.Proof.BlockHash...),
		}
		for voterID, vote := range msg.Proof.AutoVotes {
			if vote == nil {
				continue
			}
			dup := *vote
			dup.Signature = append([]byte(nil), vote.Signature...)
			dup.Digest = append([]byte(nil), vote.Digest...)
			proof.AutoVotes[voterID] = &dup
		}
		committedConfig, callback, err = atm.commitPendingView(msg.View)
	}()

	if err != nil {
		return err
	}
	if callback != nil {
		callback(committedConfig, proof)
	}
	return nil
}

// commitAutoTransitionLocked commits the auto-transition (must hold atm.mu)
func (atm *ATM) commitAutoTransitionLocked(msg *AutoTransitionMessage) (*Configuration, func(*Configuration, *TransitionProof), error) {
	if msg == nil {
		return nil, nil, fmt.Errorf("auto-transition message is nil")
	}
	pending := atm.pendingAutoConfigForView(msg.View)
	if pending == nil || pending.config == nil {
		return nil, nil, fmt.Errorf("no pending auto configuration for view %d", msg.View)
	}

	logger.Info("Auto-transition candidate reached quorum: view %d configID %d validators %d (awaiting committed installation)",
		msg.View, pending.config.ID, len(pending.config.Validators))

	committedConfig := pending.config.Copy()
	callback := atm.onTransition
	atm.finalizedViews[msg.View] = struct{}{}
	atm.finalizedConfigIDs[committedConfig.ID] = struct{}{}
	if msg.View > atm.lastCommittedView {
		atm.lastCommittedView = msg.View
	}
	delete(atm.pendingAutoConfigs, msg.View)
	delete(atm.voteCollectors, msg.View)

	return committedConfig, callback, nil
}

// GCCollectors 清理过期的 vote collector，防止内存泄漏
func (atm *ATM) GCCollectors(maxAge time.Duration) {
	atm.mu.Lock()
	defer atm.mu.Unlock()
	now := time.Now()
	for view, collector := range atm.voteCollectors {
		if now.Sub(collector.timestamp) > maxAge {
			delete(atm.voteCollectors, view)
		}
	}
	for key := range atm.recentMessages {
		if key.view <= atm.lastCommittedView {
			delete(atm.recentMessages, key)
		}
	}
}

// OnTransition sets the transition callback
func sameUint64Set(left []uint64, right []uint64) bool {
	if len(left) != len(right) {
		return false
	}
	leftCopy := append([]uint64(nil), left...)
	rightCopy := append([]uint64(nil), right...)
	sort.Slice(leftCopy, func(i, j int) bool { return leftCopy[i] < leftCopy[j] })
	sort.Slice(rightCopy, func(i, j int) bool { return rightCopy[i] < rightCopy[j] })
	for i := range leftCopy {
		if leftCopy[i] != rightCopy[i] {
			return false
		}
	}
	return true
}

func (atm *ATM) OnTransition(callback func(*Configuration, *TransitionProof)) {
	atm.mu.Lock()
	defer atm.mu.Unlock()
	atm.onTransition = callback
}

// String returns string representation
func (at AutoTransitionType) String() string {
	switch at {
	case AutoPropose:
		return "AUTO_PROPOSE"
	case AutoVote:
		return "AUTO_VOTE"
	case AutoCommit:
		return "AUTO_COMMIT"
	default:
		return "UNKNOWN"
	}
}

// Vote represents a vote in auto-transition
type Vote struct {
	SenderID  uint64
	Signature []byte
	Digest    []byte
}
