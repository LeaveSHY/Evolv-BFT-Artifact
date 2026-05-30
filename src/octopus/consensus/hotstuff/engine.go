// Copyright 2024 Octopus Project
// Licensed under Apache License 2.0

package hotstuff

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"sort"
	"sync"
	"time"

	"go.dedis.ch/kyber/v3"

	"octopus-bft/octopus/consensus/beacon"
	"octopus-bft/octopus/consensus/gbc"
	"octopus-bft/octopus/consensus/mempool"
	"octopus-bft/octopus/consensus/pacemaker"
	"octopus-bft/octopus/consensus/viewchange"
	"octopus-bft/octopus/crypto"
	"octopus-bft/octopus/hydra"
	"octopus-bft/octopus/network/libp2p"
	"octopus-bft/octopus/storage"
	"octopus-bft/octopus/types"
)

// Engine is the main consensus engine
type Engine struct {
	mu sync.RWMutex

	nodeID         uint64
	keypair        *types.Keypair
	instanceID     uint64
	numInstances   uint64
	consensusTopic string
	consensusBuf   int
	timeoutMs      int64
	outputChan     chan<- InstanceOutput
	rankState      *RankState

	// State
	currentEpoch uint64
	valSet       *types.ValidatorSet // Current active validator set

	// Components
	blockTree  *BlockTree
	pacemaker  *pacemaker.Pacemaker
	vcm        *viewchange.ViewChangeManager // View change / TC aggregation manager
	network    *libp2p.P2PNetwork            // Updated to Libp2p
	storage    *storage.StorageManager
	executor   *Executor
	mempool    *mempool.Mempool
	beacon     *beacon.RandomBeacon
	blsBeacon  *beacon.BLSBeacon // BLS aggregate signatures for verifiable random beacon
	reputation *LeaderReputation
	hydra      *hydra.HydraManager // Hydra dynamic membership manager (optional)
	gbcLog     *gbc.Log            // Global Beacon Chain log (optional, §III-C)
	gbcNode    *gbc.Node           // GBC Node for attestation protocol (optional, G4)

	// VRF committee selection keys (Phase 4b)
	vrfPrivKey    kyber.Scalar
	vrfPubKey     kyber.Point
	committeeSize int                    // Target committee size for VRF sortition (0 = disabled, all vote)
	vrfPubKeys    map[uint64]kyber.Point // VRF public keys indexed by validator ID

	// Channels
	voteChan       chan *types.Vote
	voteCollectors map[string]*voteCollector

	lastCommittedHeight uint64

	// Equivocation detection: tracks the first proposal hash seen
	// per (view, leaderID) to detect conflicting proposals from
	// the same leader in the same view (Byzantine behavior).
	seenProposals map[uint64]map[uint64][]byte // view -> leaderID -> blockHash

	// Buffered mempool certs for when node becomes leader
	pendingCerts []*types.VertexCertificate

	// Track last view we proposed in to prevent equivocation
	lastProposedView uint64

	isRunning bool
	rejected  map[string]uint64
}

type AdaptiveTuning struct {
	CommitteeSize int
	TimeoutMs     int64
}

type voteCollector struct {
	qc      *types.QuorumCertificate
	signers map[uint64]struct{}
	done    bool
}

type proposalSnapshot struct {
	view          uint64
	epoch         uint64
	configID      uint64
	leaderSetHash []byte
	highQC        *types.QuorumCertificate
	latestHeight  uint64
	rank          uint64
	seed          []byte
}

type EngineOptions struct {
	PacemakerTimeoutMs int64
	ConsensusBuffer    int
	Mempool            mempool.Options
	CommitteeSize      int // VRF committee size (0 = disabled, all validators vote)
}

func DefaultEngineOptions() EngineOptions {
	return EngineOptions{
		PacemakerTimeoutMs: 500,  // 500ms timeout for faster view rotation at 1000-node scale
		ConsensusBuffer:    4096, // Larger buffer for 1000-node scale
		Mempool:            mempool.DefaultOptions(),
		CommitteeSize:      25, // VRF committee: 25 voters per view reduces O(n²)→O(k²)
	}
}

func normalizeEngineOptions(opts EngineOptions) EngineOptions {
	def := DefaultEngineOptions()
	if opts.PacemakerTimeoutMs <= 0 {
		opts.PacemakerTimeoutMs = def.PacemakerTimeoutMs
	}
	if opts.ConsensusBuffer <= 0 {
		opts.ConsensusBuffer = def.ConsensusBuffer
	}
	return opts
}

// NewEngine creates a new consensus engine
func NewEngine(
	nodeID uint64,
	keypair *types.Keypair,
	initialValSet *types.ValidatorSet,
	net *libp2p.P2PNetwork,
	store *storage.StorageManager,
) *Engine {
	return NewEngineWithInstanceAndOptions(nodeID, keypair, initialValSet, net, store, 0, 1, "octopus-consensus", nil, DefaultEngineOptions())
}

func NewEngineWithInstance(
	nodeID uint64,
	keypair *types.Keypair,
	initialValSet *types.ValidatorSet,
	net *libp2p.P2PNetwork,
	store *storage.StorageManager,
	instanceID uint64,
	numInstances uint64,
	consensusTopicBase string,
	outputChan chan<- InstanceOutput,
) *Engine {
	return NewEngineWithInstanceAndOptions(nodeID, keypair, initialValSet, net, store, instanceID, numInstances, consensusTopicBase, outputChan, DefaultEngineOptions())
}

func NewEngineWithInstanceAndOptions(
	nodeID uint64,
	keypair *types.Keypair,
	initialValSet *types.ValidatorSet,
	net *libp2p.P2PNetwork,
	store *storage.StorageManager,
	instanceID uint64,
	numInstances uint64,
	consensusTopicBase string,
	outputChan chan<- InstanceOutput,
	options EngineOptions,
) *Engine {
	options = normalizeEngineOptions(options)
	// Initialize Executor
	exec := NewExecutor(initialValSet)

	// Initialize BlockTree
	bt := NewBlockTreeWithEpoch(store, exec, initialValSet.Epoch)
	if initialValSet != nil && initialValSet.Validators != nil {
		bt.SetFastPathThreshold(uint64(len(initialValSet.Validators)))
	}

	mp := mempool.NewMempoolWithOptions(nodeID, keypair, initialValSet, net, options.Mempool)

	// Wire VertexResolver: Mempool implements GetVertex(hash) for DAG vertex resolution
	exec.SetVertexResolver(mp)

	// Initialize Random Beacon
	rb := beacon.NewRandomBeacon([]byte("genesis-seed"))

	// Initialize Leader Reputation tracker for straggler detection
	repConfig := DefaultReputationConfig()
	repConfig.BaseTimeoutMs = options.PacemakerTimeoutMs
	rep := NewLeaderReputation(repConfig)

	// Generate VRF key pair for committee selection (Phase 4b)
	vrfPriv, vrfPub := crypto.GenerateVRFKey()

	// Initialize BLS beacon for verifiable random beacon (threshold BLS aggregate)
	blsBeaconModule := beacon.NewBLSBeacon()

	// Create engine
	if numInstances == 0 {
		numInstances = 1
	}
	consensusTopic := libp2p.InstanceConsensusTopic(consensusTopicBase, instanceID)
	e := &Engine{
		nodeID:         nodeID,
		keypair:        keypair,
		instanceID:     instanceID,
		numInstances:   numInstances,
		consensusTopic: consensusTopic,
		consensusBuf:   options.ConsensusBuffer,
		timeoutMs:      options.PacemakerTimeoutMs,
		outputChan:     outputChan,
		rankState:      NewRankState(instanceID, numInstances),
		currentEpoch:   initialValSet.Epoch,
		valSet:         initialValSet,
		network:        net,
		storage:        store,
		blockTree:      bt,
		executor:       exec,
		mempool:        mp,
		beacon:         rb,
		blsBeacon:      blsBeaconModule,
		reputation:     rep,
		vrfPrivKey:     vrfPriv,
		vrfPubKey:      vrfPub,
		committeeSize:  options.CommitteeSize,
		vrfPubKeys:     make(map[uint64]kyber.Point),
		voteChan:       make(chan *types.Vote, 2048), // Large buffer for 1000-node vote flood
		voteCollectors: make(map[string]*voteCollector),
		seenProposals:  make(map[uint64]map[uint64][]byte),
		rejected:       make(map[string]uint64),
	}
	e.rankState.SetHighestLocalHeight(store.GetLatestBlockHeight())

	// Register own VRF public key
	e.vrfPubKeys[nodeID] = vrfPub

	// Register own BLS key for verifiable beacon
	if _, err := e.blsBeacon.GenerateKey(nodeID); err != nil {
		logger.Error("Failed to generate BLS beacon key: %v", err)
	}

	// Initial Pacemaker
	e.resetPacemaker(initialValSet)

	// Initialize ViewChange/TC manager
	vcm := viewchange.NewViewChangeManager(nodeID, time.Duration(options.PacemakerTimeoutMs)*time.Millisecond, func() uint64 {
		return e.valSet.QuorumSize
	})
	e.vcm = vcm

	bt.SetCommitCallback(e.onBlockCommitted)

	// Setup Network Streams
	if e.network != nil {
		e.network.SetupStreamHandler()
	}

	return e
}

func (e *Engine) resetPacemaker(valSet *types.ValidatorSet) {
	valIDs := validatorIDsFromSet(valSet)
	if len(valIDs) == 0 {
		valIDs = e.allowedLeaderIDsSnapshot(valSet)
	}

	timeoutMs := e.timeoutMs
	if timeoutMs <= 0 {
		timeoutMs = 2000
	}
	e.pacemaker = pacemaker.NewPacemakerWithLane(valIDs, timeoutMs, e.instanceID)
	e.refreshLeaderSelectorFor(valSet)
}

func (e *Engine) onEpochChangeWithVRFKeys(newValSet *types.ValidatorSet, refreshedVRFPubKeys map[uint64]kyber.Point) {
	if newValSet == nil {
		return
	}
	e.mu.Lock()
	logger.Info("Engine: Processing Epoch Change %d -> %d", e.currentEpoch, newValSet.Epoch)

	e.currentEpoch = newValSet.Epoch
	e.valSet = newValSet
	if newValSet != nil {
		retained := make(map[uint64]kyber.Point, len(newValSet.Validators))
		for validatorID, validator := range newValSet.Validators {
			if validator == nil || !validator.IsActive {
				continue
			}
			if pubKey, ok := e.vrfPubKeys[validatorID]; ok && pubKey != nil {
				retained[validatorID] = pubKey
			}
			if refreshedVRFPubKeys != nil {
				if pubKey, ok := refreshedVRFPubKeys[validatorID]; ok && pubKey != nil {
					retained[validatorID] = pubKey
				}
			}
		}
		e.vrfPubKeys = retained
	}

	if e.blockTree != nil && newValSet != nil && newValSet.Validators != nil {
		e.blockTree.SetFastPathThreshold(uint64(len(newValSet.Validators)))
	}
	e.mu.Unlock()

	// Reset Pacemaker for new Epoch (View 1)
	e.resetPacemaker(newValSet)
	e.pacemaker.Start()

	// Try to propose if we are the new leader
	e.tryPropose()
}

// UpdateValidatorSet is a compatibility entrypoint for tests or callers that do
// not need committed VRF key refresh. The authoritative committed runtime path
// must not use this entrypoint, because it advances epoch/config state without
// atomically applying refreshed committed VRF verifier keys.
func (e *Engine) UpdateValidatorSet(newValSet *types.ValidatorSet) {
	if newValSet == nil {
		return
	}
	e.onEpochChangeWithVRFKeys(newValSet, nil)
}

// UpdateValidatorSetWithVRFKeys atomically advances epoch/config state together
// with any refreshed VRF public keys from the committed validator set, while
// preserving already-registered keys for unchanged active validators.
func (e *Engine) UpdateValidatorSetWithVRFKeys(newValSet *types.ValidatorSet, refreshedVRFPubKeys map[uint64]kyber.Point) {
	if newValSet == nil {
		return
	}
	e.onEpochChangeWithVRFKeys(newValSet, refreshedVRFPubKeys)
}

// CurrentLeader returns the leader of this instance for the current view.
func (e *Engine) CurrentLeader() uint64 {
	e.mu.RLock()
	pm := e.pacemaker
	e.mu.RUnlock()
	if pm == nil {
		return 0
	}
	return pm.GetLeader(pm.GetCurrentView())
}

func (e *Engine) GetCurrentValidatorSet() *types.ValidatorSet {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.valSet
}

func (e *Engine) currentEpochSnapshot() uint64 {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.currentEpoch
}

func (e *Engine) currentConfigIDSnapshot() uint64 {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.valSet == nil {
		return 0
	}
	return e.valSet.Epoch
}

func (e *Engine) leaderSetHashSnapshot() []byte {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.valSet == nil {
		return nil
	}
	return append([]byte(nil), e.valSet.Hash()...)
}

func (e *Engine) leaderSetHashForConfigID(configID uint64) []byte {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.valSet == nil || e.valSet.Epoch != configID {
		return nil
	}
	return append([]byte(nil), e.valSet.Hash()...)
}

func (e *Engine) resolveValidatorPubKeyByID(id uint64) (types.PublicKey, bool) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.valSet == nil {
		return nil, false
	}
	validator, exists := e.valSet.Validators[id]
	if !exists || validator == nil {
		return nil, false
	}
	return validator.PublicKey, true
}

// Start starts the consensus engine
func (e *Engine) Start() {
	e.isRunning = true
	if e.network != nil {
		instanceID := e.instanceID
		err := e.network.RegisterTopicPolicy(e.consensusTopic, libp2p.TopicValidationPolicy{
			ExpectedInstance: &instanceID,
			EpochProvider:    e.currentEpochSnapshot,
			KeyProvider:      e.resolveValidatorPubKeyByID,
		})
		if err != nil {
			logger.Error("Failed to register consensus validator: %v", err)
		}
	}
	e.pacemaker.Start()
	e.mempool.Start()

	// Start message processing loop
	go e.runLoop()
}

func (e *Engine) runLoop() {
	if e.network == nil {
		logger.Error("network is nil")
		return
	}
	consensusChan, err := e.network.SubscribeTopic(e.consensusTopic, e.consensusBuf)
	if err != nil {
		logger.Error("Failed to subscribe consensus: %v", err)
		return
	}

	proposalChan := e.mempool.GetProposalChan()

	// Phase 5: Batch vote processing.
	// At 1000 nodes, the leader receives hundreds of votes per view.
	// We batch-drain up to maxVoteBatch votes before yielding to other
	// select cases, reducing per-message select overhead.
	const maxVoteBatch = 64

	for e.isRunning {
		timeoutChan := e.pacemaker.TimeoutChan()

		select {
		case data := <-consensusChan:
			msg, err := types.DecodeMessage(data)
			if err != nil {
				logger.Error("Failed to decode consensus message: %v", err)
				continue
			}
			e.handleMessage(msg)

			// Batch-drain: if more messages are already buffered, process
			// them immediately without re-entering select (avoids overhead).
			e.drainConsensusBatch(consensusChan, maxVoteBatch)

		case certs := <-proposalChan:
			e.handleMempoolProposal(certs)

		case view := <-timeoutChan:
			if e.pacemaker.GetCurrentView() <= view {
				// Record timeout against the expected leader for this view
				timedOutLeader := e.pacemaker.GetLeader(view)
				if e.reputation != nil {
					e.reputation.RecordTimeout(timedOutLeader)

					if e.reputation.IsCrashed(timedOutLeader) {
						logger.Info("Leader %d appears crashed (%d consecutive timeouts) at View %d",
							timedOutLeader, e.reputation.GetStats(timedOutLeader).ConsecutiveTimeouts, view)

						// Hydra integration: trigger auto-transition to remove crashed leader
						if e.hydra != nil {
							e.hydra.LSetManager.MarkFault(timedOutLeader, hydra.FaultClassUnavailable)
							if err := e.hydra.TriggerAutoTransition(timedOutLeader, view); err != nil {
								logger.Error("Hydra auto-transition failed for crashed leader %d: %v", timedOutLeader, err)
							}
						}
					} else if e.reputation.IsStraggler(timedOutLeader) {
						logger.Info("Leader %d is straggler (failure_rate=%.2f) at View %d",
							timedOutLeader, e.reputation.GetStats(timedOutLeader).FailureRate, view)

						// Hydra integration: mark straggler in L-set mmtable for future auto-transition
						if e.hydra != nil {
							e.hydra.LSetManager.MarkFault(timedOutLeader, hydra.FaultClassDegraded)
						}
					}

					logger.Info("Timeout in View %d (Epoch %d), adaptive_timeout=%v",
						view, e.currentEpoch, e.reputation.AdaptiveTimeout(timedOutLeader))

					// Wire adaptive timeout into pacemaker: adjust the next view's
					// timeout based on the timed-out leader's reputation history.
					adaptiveTimeout := e.reputation.AdaptiveTimeout(timedOutLeader)
					e.pacemaker.SetTimeout(adaptiveTimeout)
				} else {
					logger.Info("Timeout in View %d (Epoch %d)", view, e.currentEpoch)
				}

				// Broadcast a timeout vote for TC aggregation.
				// Instead of just locally advancing, we participate in the
				// distributed view-change protocol: when 2f+1 nodes time out
				// on the same view, a TC is formed enabling the next leader
				// to safely propose.
				e.broadcastTimeoutVote(view)
				e.pacemaker.AdvanceView(view)
				e.tryPropose()
			}

		case <-time.After(100 * time.Millisecond):
			// Yield
		}
	}
}

// drainConsensusBatch processes up to maxBatch additional messages from the
// consensus channel without re-entering the select loop. This reduces per-message
// scheduling overhead when the leader is under heavy vote load (1000 nodes).
func (e *Engine) drainConsensusBatch(ch <-chan []byte, maxBatch int) {
	for i := 0; i < maxBatch; i++ {
		select {
		case data := <-ch:
			msg, err := types.DecodeMessage(data)
			if err != nil {
				continue
			}
			e.handleMessage(msg)
		default:
			return
		}
	}
}

func (e *Engine) handleMempoolProposal(certs []*types.VertexCertificate) {
	currentView := e.pacemaker.GetCurrentView()
	leader := e.pacemaker.GetLeader(currentView)

	// Check if we are the primary leader
	isMyTurn := leader == e.nodeID

	// If primary leader is a straggler, check if we are the fallback leader
	if !isMyTurn && e.reputation != nil && (e.reputation.IsStraggler(leader) || e.reputation.IsCrashed(leader)) {
		valIDs := e.sortedValidatorIDs()
		fallback := e.reputation.SelectFallbackLeader(leader, valIDs, currentView)
		if fallback == e.nodeID {
			isMyTurn = true
			logger.Info("Taking over as fallback leader for View %d (primary=%d is straggler)", currentView, leader)
		}
	}

	if isMyTurn {
		// Include any previously buffered certs
		allCerts := append(e.pendingCerts, certs...)
		e.pendingCerts = nil
		logger.Info("Got %d certs from Mempool, creating block...", len(allCerts))
		e.proposeBlock(allCerts)
	} else {
		// Buffer certs for when we become leader
		e.pendingCerts = append(e.pendingCerts, certs...)
	}
}

func (e *Engine) handleMessage(msg *types.Message) {
	if msg == nil {
		e.incRejected("nil_message")
		return
	}
	if msg.Instance != e.instanceID {
		e.incRejected("wrong_instance")
		return
	}
	if msg.Lane != e.instanceID {
		e.incRejected("wrong_lane")
		return
	}
	pubKey, ok := e.resolveValidatorPubKeyByID(msg.SenderID)
	if !ok || !msg.VerifySignature(pubKey) {
		e.incRejected("invalid_message_signature")
		return
	}
	if msg.Epoch < e.currentEpoch {
		e.incRejected("stale_epoch")
		return
	}
	if msg.ConfigID != e.currentConfigIDSnapshot() {
		e.incRejected("message_config_mismatch")
		return
	}
	currentLeaderSetHash := e.leaderSetHashSnapshot()
	requireLeaderSetHash := func() bool {
		switch msg.Type {
		case types.MsgProposal, types.MsgVote, types.MsgTimeout, types.MsgNewView:
			return true
		default:
			return false
		}
	}()
	if requireLeaderSetHash && len(msg.LeaderSetHash) == 0 {
		e.incRejected("missing_leader_set_hash")
		return
	}
	if len(msg.LeaderSetHash) > 0 && !types.LeaderSetHashEqual(msg.LeaderSetHash, currentLeaderSetHash) {
		e.incRejected("leader_set_mismatch")
		return
	}
	switch msg.Type {
	case types.MsgProposal:
		if msg.Block == nil {
			e.incRejected("nil_proposal_payload")
			return
		}
		if msg.Block.LeaderID != msg.SenderID {
			e.incRejected("proposal_sender_mismatch")
			return
		}
		if msg.Block.View != msg.View {
			e.incRejected("proposal_view_wrapper_mismatch")
			return
		}
		if msg.Block.Epoch != msg.Epoch {
			e.incRejected("proposal_epoch_wrapper_mismatch")
			return
		}
		if msg.Block.ConfigID != msg.ConfigID {
			e.incRejected("proposal_config_wrapper_mismatch")
			return
		}
		if msg.Block.LaneID != msg.Lane {
			e.incRejected("proposal_lane_wrapper_mismatch")
			return
		}
		if msg.BarrierView != msg.View {
			e.incRejected("proposal_barrier_view_mismatch")
			return
		}
		e.handleProposal(msg.Block)
	case types.MsgVote:
		if msg.Vote == nil {
			e.incRejected("nil_vote_payload")
			return
		}
		validatorID, ok := e.resolveValidatorID(msg.Vote.Author)
		if !ok || validatorID != msg.SenderID {
			e.incRejected("vote_sender_mismatch")
			return
		}
		if msg.View != msg.Vote.View {
			e.incRejected("vote_view_mismatch")
			return
		}
		if msg.Epoch != msg.Vote.Epoch {
			e.incRejected("vote_epoch_mismatch")
			return
		}
		if msg.ConfigID != msg.Vote.ConfigID {
			e.incRejected("vote_config_mismatch")
			return
		}
		if msg.Lane != msg.Vote.Lane {
			e.incRejected("vote_lane_mismatch")
			return
		}
		if msg.BarrierView != msg.View {
			e.incRejected("vote_barrier_view_mismatch")
			return
		}
		e.handleVote(msg.Vote)
	case types.MsgTimeout:
		if msg.TimeoutVote == nil {
			e.incRejected("nil_timeout_vote_payload")
			return
		}
		if msg.TimeoutVote.VoterID != msg.SenderID {
			e.incRejected("timeout_vote_sender_mismatch")
			return
		}
		if msg.TimeoutVote.View != msg.View {
			e.incRejected("timeout_vote_view_wrapper_mismatch")
			return
		}
		if msg.TimeoutVote.Epoch != msg.Epoch {
			e.incRejected("timeout_vote_epoch_wrapper_mismatch")
			return
		}
		if msg.TimeoutVote.ConfigID != msg.ConfigID {
			e.incRejected("timeout_vote_config_wrapper_mismatch")
			return
		}
		if msg.TimeoutVote.Lane != msg.Lane {
			e.incRejected("timeout_vote_lane_wrapper_mismatch")
			return
		}
		if msg.BarrierView != msg.View {
			e.incRejected("timeout_vote_barrier_view_mismatch")
			return
		}
		e.handleTimeoutVote(msg.TimeoutVote)
	case types.MsgNewView:
		if msg.TC != nil {
			if msg.View != msg.TC.View+1 {
				e.incRejected("newview_tc_view_wrapper_mismatch")
				return
			}
			if msg.BarrierView != msg.View {
				e.incRejected("newview_tc_barrier_view_mismatch")
				return
			}
			if msg.Epoch != msg.TC.Epoch {
				e.incRejected("newview_tc_epoch_wrapper_mismatch")
				return
			}
			if msg.ConfigID != msg.TC.ConfigID {
				e.incRejected("newview_tc_config_wrapper_mismatch")
				return
			}
			if msg.Lane != msg.TC.Lane {
				e.incRejected("newview_tc_lane_wrapper_mismatch")
				return
			}
		}
		if msg.QC != nil {
			if msg.View != msg.QC.View+1 {
				e.incRejected("newview_qc_view_wrapper_mismatch")
				return
			}
			if msg.BarrierView != msg.View {
				e.incRejected("newview_qc_barrier_view_mismatch")
				return
			}
			if msg.Epoch != msg.QC.Epoch {
				e.incRejected("newview_qc_epoch_wrapper_mismatch")
				return
			}
			if msg.ConfigID != msg.QC.ConfigID {
				e.incRejected("newview_qc_config_wrapper_mismatch")
				return
			}
			if msg.Lane != msg.QC.Lane {
				e.incRejected("newview_qc_lane_wrapper_mismatch")
				return
			}
		}
		if msg.TC == nil && msg.QC == nil {
			e.incRejected("newview_missing_certificate")
			return
		}
		if msg.TC != nil && msg.QC != nil {
			e.incRejected("newview_redundant_top_level_qc")
			return
		}
		e.handleNewView(msg)
	}
}

func (e *Engine) handleProposal(block *types.Block) {
	if block == nil {
		return
	}
	if ok, reason := e.validateProposalConstraint(block); !ok {
		e.incRejected(reason)
		return
	}

	if block.Epoch != e.currentEpoch {
		e.incRejected("proposal_epoch_mismatch")
		logger.Error("Block Epoch %d mismatch with Engine Epoch %d", block.Epoch, e.currentEpoch)
		return
	}

	// Cryptographically verify the Justify QC signatures. Without this check,
	// a Byzantine node could forge a QC with invalid signatures and trick
	// honest nodes into accepting an unsafe proposal.
	// Genesis QC (View=0, no signatures) is trusted by all participants by definition.
	if block.Justify != nil && block.View > 1 && block.Justify.View > 0 {
		if !e.verifyQC(block.Justify) {
			e.incRejected("proposal_invalid_justify_qc")
			logger.Error("Block %x Justify QC failed signature verification", block.Hash[:4])
			return
		}
	}

	if e.storage.HasBlock(block.Hash) {
		e.incRejected("duplicate_block")
		return
	}
	if block.Height <= e.rankState.HighestLocalHeight() {
		e.incRejected("stale_height")
		return
	}
	if !e.rankState.VerifyRank(block.Height, block.Rank) {
		e.incRejected("rank_mismatch")
		logger.Error("Block rank %d mismatch at height %d for instance %d", block.Rank, block.Height, e.instanceID)
		return
	}

	logger.Info("Received Proposal for View %d (Epoch %d) with %d certs", block.View, block.Epoch, len(block.Payload))

	// Equivocation detection: reject conflicting proposals from the same
	// leader in the same view. A Byzantine leader may send different blocks
	// to different nodes; we must detect and reject the second one.
	if leaders, ok := e.seenProposals[block.View]; ok {
		if prevHash, seen := leaders[block.LeaderID]; seen {
			if !bytes.Equal(prevHash, block.Hash) {
				e.incRejected("equivocation_detected")
				logger.Error("EQUIVOCATION: Leader %d sent conflicting proposals in View %d (prev=%x, new=%x)",
					block.LeaderID, block.View, prevHash[:4], block.Hash[:4])
				// Mark leader as crashed in reputation system
				if e.reputation != nil {
					e.reputation.RecordTimeout(block.LeaderID) // penalize
				}
				if e.hydra != nil {
					e.hydra.LSetManager.MarkFault(block.LeaderID, hydra.FaultClassByzantine)
				}
				return
			}
			// Same hash = duplicate proposal, already processed
			e.incRejected("duplicate_proposal")
			return
		}
	} else {
		e.seenProposals[block.View] = make(map[uint64][]byte)
	}
	e.seenProposals[block.View][block.LeaderID] = block.Hash

	// Phase 5: Update Random Beacon
	if block.Justify != nil && len(block.Justify.AggregatedSig) > 0 {
		e.beacon.UpdateRandomness(block.Justify.AggregatedSig)
	}

	if !e.blockTree.SafeNode(block) {
		e.incRejected("safety_rule")
		logger.Error("Block %x rejected by Safety Rules", block.Hash[:4])
		return
	}

	if err := e.blockTree.ProcessBlock(block); err != nil {
		e.incRejected("process_block_error")
		logger.Error("Failed to process block: %v", err)
		return
	}

	e.pacemaker.AdvanceView(block.View)

	// Phase 4b: VRF Committee Selection
	// If committeeSize > 0, only VRF-selected committee members vote.
	// Otherwise, all validators vote (classic HotStuff).
	var vrfProof []byte
	if e.committeeSize > 0 && e.vrfPrivKey != nil {
		totalWeight := e.valSet.TotalPower
		selected, _, proof := e.beacon.AmICommitteeMember(e.vrfPrivKey, totalWeight, e.committeeSize)
		if !selected {
			logger.Info("VRF: Not selected for committee in View %d (committee_size=%d, total=%d), skipping vote",
				block.View, e.committeeSize, totalWeight)
			e.tryPropose()
			return
		}
		vrfProof = proof
		logger.Info("VRF: Selected for committee in View %d (committee_size=%d)", block.View, e.committeeSize)
	}

	var blockID types.Hash
	copy(blockID[:], block.Hash)
	vote, err := types.NewVoteWithIdentity(blockID, block.View, block.Epoch, block.ConfigID, block.LaneID, e.keypair.PublicKey, e.keypair.PrivateKey, vrfProof)
	if err != nil {
		logger.Error("Failed to sign vote: %v", err)
		return
	}

	voteMsg := &types.Message{
		Type:          types.MsgVote,
		SenderID:      e.nodeID,
		View:          vote.View,
		Epoch:         e.currentEpoch,
		ConfigID:      block.ConfigID,
		Lane:          block.LaneID,
		LeaderSetHash: e.leaderSetHashSnapshot(),
		BarrierView:   block.View,
		Instance:      e.instanceID,
		Vote:          vote,
	}
	if err := voteMsg.Sign(e.keypair.PrivateKey); err != nil {
		logger.Error("Failed to sign vote message: %v", err)
		return
	}
	voteData, err := types.EncodeMessage(voteMsg)
	if err != nil {
		logger.Error("Failed to encode vote message: %v", err)
		return
	}
	// Phase 2a: Send votes directly to the leader via unicast (not broadcast).
	// At 1000 nodes, broadcasting votes is O(n²). Unicast is O(n).
	leaderID := block.LeaderID
	if err := e.network.SendVoteToLeader(leaderID, voteData); err != nil {
		// Fallback to PubSub broadcast if unicast fails (leader not in address book, etc.)
		if pubErr := e.network.PublishTopic(e.consensusTopic, voteData); pubErr != nil {
			logger.Error("Failed to send vote (unicast and broadcast failed): %v / %v", err, pubErr)
		}
	}

	logger.Info("Voted for block %x in View %d (to leader %d)", block.Hash[:4], block.View, leaderID)

	e.tryPropose()
}

func (e *Engine) validateProposalConstraint(block *types.Block) (bool, string) {
	currentView := e.pacemaker.GetCurrentView()
	if currentView > 1 && block.View+1 < currentView {
		return false, "proposal_stale_view"
	}
	if block.View > currentView+1 {
		return false, "proposal_future_view"
	}
	if block.View == 0 {
		return false, "proposal_zero_view"
	}
	if block.LaneID != e.instanceID {
		return false, "proposal_wrong_lane"
	}
	if block.ConfigID != e.currentConfigIDSnapshot() {
		return false, "proposal_config_mismatch"
	}

	expectedLeader := e.pacemaker.GetLeader(block.View)
	if block.LeaderID != expectedLeader {
		return false, "proposal_wrong_leader"
	}
	if e.hydra != nil && !e.hydra.IsAllowedLeader(block.LeaderID) {
		return false, "proposal_leader_not_allowed"
	}

	if block.Justify == nil {
		if block.View == 1 {
			return true, ""
		}
		return false, "proposal_missing_qc"
	}
	if block.Justify.Epoch != block.Epoch {
		return false, "proposal_qc_epoch_mismatch"
	}
	if block.Justify.View >= block.View {
		return false, "proposal_qc_view_non_monotonic"
	}
	if len(block.Justify.BlockHash) > 0 && !bytes.Equal(block.Parent, block.Justify.BlockHash) {
		return false, "proposal_parent_qc_mismatch"
	}
	return true, ""
}

func (e *Engine) handleVote(vote *types.Vote) {
	currentView := e.pacemaker.GetCurrentView()
	if vote == nil || (vote.View != currentView && vote.View+1 != currentView) {
		e.incRejected("vote_view_mismatch")
		return
	}
	if vote.Epoch != e.currentEpoch {
		e.incRejected("vote_epoch_mismatch")
		return
	}
	if vote.ConfigID != e.currentConfigIDSnapshot() {
		e.incRejected("vote_config_mismatch")
		return
	}
	if vote.Lane != e.instanceID {
		e.incRejected("vote_lane_mismatch")
		return
	}
	if !vote.Verify() {
		e.incRejected("invalid_vote_signature")
		return
	}

	validatorID, ok := e.resolveValidatorID(vote.Author)
	if !ok {
		e.incRejected("unknown_vote_author")
		return
	}

	// Phase 4b: Verify VRF committee membership proof
	if e.committeeSize > 0 {
		if len(vote.VRFProof) == 0 {
			e.incRejected("missing_vrf_proof")
			return
		}
		vrfPubKey, hasKey := e.vrfPubKeys[validatorID]
		if !hasKey {
			e.incRejected("missing_vrf_pubkey")
			logger.Error("VRF: Vote from validator %d rejected — missing registered VRF public key", validatorID)
			return
		}
		totalWeight := e.valSet.TotalPower
		if !e.beacon.VerifyCommitteeMember(vrfPubKey, vote.VRFProof, totalWeight, e.committeeSize) {
			e.incRejected("invalid_vrf_proof")
			logger.Error("VRF: Vote from validator %d rejected — invalid committee proof", validatorID)
			return
		}
	}

	blockHash := vote.BlockID[:]
	collectorKey := e.collectorKey(vote.View, vote.Epoch, vote.ConfigID, vote.Lane, blockHash)
	collector, exists := e.voteCollectors[collectorKey]
	if !exists {
		qc := types.NewQuorumCertificateWithIdentity(blockHash, vote.View, vote.Epoch, vote.ConfigID, vote.Lane, types.PhasePrepare)
		collector = &voteCollector{
			qc:      qc,
			signers: make(map[uint64]struct{}),
		}
		e.voteCollectors[collectorKey] = collector
	}
	if collector.done {
		return
	}
	if _, duplicated := collector.signers[validatorID]; duplicated {
		return
	}

	collector.signers[validatorID] = struct{}{}
	collector.qc.AddSignature(validatorID, vote.Signature)

	// Phase 4b: When VRF committee is active, quorum is 2/3+1 of committeeSize,
	// not of the full validator set.
	quorumSize := e.valSet.QuorumSize
	if e.committeeSize > 0 {
		quorumSize = uint64(e.committeeSize)*2/3 + 1
	}
	if !collector.qc.IsComplete(quorumSize) {
		return
	}

	// NOTE: verifyQC checks each individual Ed25519 signature in the
	// Signatures map — this is the real cryptographic safety check.
	// AggregatedSig is only used as a deterministic beacon seed.
	if !e.verifyQC(collector.qc) {
		e.incRejected("invalid_qc")
		return
	}
	collector.qc.AggregatedSig = e.aggregateSignatures(collector.qc.Signatures)
	collector.done = true
	e.blockTree.OnVoteQC(collector.qc)
	e.pacemaker.AdvanceView(collector.qc.View)
	e.tryPropose()
}

func (e *Engine) verifyQC(qc *types.QuorumCertificate) bool {
	validators := make(map[uint64]types.PublicKey, len(e.valSet.Validators))
	for id, validator := range e.valSet.Validators {
		if validator == nil {
			continue
		}
		validators[id] = validator.PublicKey
	}
	// Use batch parallel verification for 1000-node scale performance
	return crypto.VerifyQuorumBatch(qc, validators, int(e.valSet.QuorumSize))
}

// verifyTC cryptographically verifies a Timeout Certificate by checking that
// at least QuorumSize timeout vote signatures are valid. Each signature in
// TC.Signatures was produced by crypto.Sign(BigEndian(view), privateKey).
func (e *Engine) verifyTC(tc *types.TimeoutCertificate) bool {
	if tc == nil || len(tc.Signatures) == 0 {
		return false
	}
	if tc.Epoch != e.currentEpoch {
		return false
	}
	if tc.ConfigID != e.currentConfigIDSnapshot() {
		return false
	}
	if tc.Lane != e.instanceID {
		return false
	}

	signingBytes := types.TimeoutSigningBytes(tc.View, tc.Epoch, tc.ConfigID, tc.Lane)

	validSigs := 0
	for voterID, sig := range tc.Signatures {
		validator, exists := e.valSet.Validators[voterID]
		if !exists || validator == nil {
			continue
		}
		if crypto.Verify(signingBytes, sig, validator.PublicKey) {
			validSigs++
		}
	}

	return uint64(validSigs) >= e.valSet.QuorumSize
}

func (e *Engine) resolveValidatorID(author types.PublicKey) (uint64, bool) {
	for id, validator := range e.valSet.Validators {
		if validator == nil {
			continue
		}
		if bytes.Equal(validator.PublicKey, author) {
			return id, true
		}
	}
	return 0, false
}

func (e *Engine) collectorKey(view uint64, epoch uint64, configID uint64, lane uint64, blockHash []byte) string {
	return hex.EncodeToString(blockHash) + "-" + hex.EncodeToString([]byte{
		byte(view >> 56), byte(view >> 48), byte(view >> 40), byte(view >> 32),
		byte(view >> 24), byte(view >> 16), byte(view >> 8), byte(view),
		byte(epoch >> 56), byte(epoch >> 48), byte(epoch >> 40), byte(epoch >> 32),
		byte(epoch >> 24), byte(epoch >> 16), byte(epoch >> 8), byte(epoch),
		byte(configID >> 56), byte(configID >> 48), byte(configID >> 40), byte(configID >> 32),
		byte(configID >> 24), byte(configID >> 16), byte(configID >> 8), byte(configID),
		byte(lane >> 56), byte(lane >> 48), byte(lane >> 40), byte(lane >> 32),
		byte(lane >> 24), byte(lane >> 16), byte(lane >> 8), byte(lane),
	})
}

// aggregateSignatures produces a verifiable BLS aggregate signature from the
// QC's individual Ed25519 signatures. When BLS keys are available for signers,
// it uses BLS aggregation (cryptographically sound, verifiable). Otherwise, it
// falls back to a deterministic SHA256 concatenation of Ed25519 signatures.
//
// The aggregate serves as the Random Beacon seed input. The BLS aggregate is
// unpredictable because an adversary cannot predict honest nodes' BLS signatures
// before they are broadcast, and verifiable because it can be checked against
// the aggregate public key.
//
// Security model:
//   - QC validity is verified by checking each individual Ed25519 signature in
//     crypto.VerifyQuorumBatch(), NOT by checking this aggregate.
//   - This aggregate is used only as Random Beacon input.
func (e *Engine) aggregateSignatures(signatures map[uint64][]byte) []byte {
	// Try BLS aggregation first (paper-grade verifiable beacon)
	if e.blsBeacon != nil {
		// Collect BLS signature shares: sign the QC message with each signer's BLS key.
		// We use the deterministic canonical hash of the Ed25519 signatures as the BLS message
		// to bind the BLS aggregate to the specific QC content.
		ids := make([]uint64, 0, len(signatures))
		for id := range signatures {
			ids = append(ids, id)
		}
		sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })

		// Canonical message = SHA256(sorted Ed25519 signatures)
		hasher := sha256.New()
		idBuf := make([]byte, 8)
		for _, id := range ids {
			binary.BigEndian.PutUint64(idBuf, id)
			hasher.Write(idBuf)
			hasher.Write(signatures[id])
		}
		blsMessage := hasher.Sum(nil)

		// Collect BLS signatures from validators that have registered BLS keys
		blsSigs := make(map[uint64][]byte, len(ids))
		hasBLS := false
		for _, id := range ids {
			sig, err := e.blsBeacon.Sign(id, blsMessage)
			if err == nil {
				blsSigs[id] = sig
				hasBLS = true
			}
		}

		if hasBLS {
			aggSig, err := e.blsBeacon.AggregateAndVerify(blsMessage, blsSigs)
			if err == nil {
				return aggSig
			}
			logger.Error("BLS aggregate failed, falling back to SHA256: %v", err)
		}
	}

	// Fallback: deterministic SHA256 concatenation (legacy path)
	ids := make([]uint64, 0, len(signatures))
	for id := range signatures {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })

	hasher := sha256.New()
	idBuf := make([]byte, 8)
	for _, id := range ids {
		binary.BigEndian.PutUint64(idBuf, id)
		hasher.Write(idBuf)
		hasher.Write(signatures[id])
	}
	return hasher.Sum(nil)
}

func (e *Engine) tryPropose() {
	currentView := e.pacemaker.GetCurrentView()
	leader := e.pacemaker.GetLeader(currentView)

	// Fallback leader selection: if the primary leader is a known straggler/crashed node,
	// pick the next healthy validator as the effective leader for this view.
	effectiveLeader := leader
	if e.reputation != nil && (e.reputation.IsStraggler(leader) || e.reputation.IsCrashed(leader)) {
		valIDs := e.sortedValidatorIDs()
		fallback := e.reputation.SelectFallbackLeader(leader, valIDs, currentView)
		if fallback != leader {
			logger.Info("Fallback leader: primary=%d is straggler, using fallback=%d for View %d", leader, fallback, currentView)
			effectiveLeader = fallback
		}
	}

	if effectiveLeader == e.nodeID {
		// If we have buffered certs, propose immediately
		if len(e.pendingCerts) > 0 {
			certs := e.pendingCerts
			e.pendingCerts = nil
			logger.Info("Got %d buffered certs, creating block...", len(certs))
			e.proposeBlock(certs)
		} else {
			logger.Info("I am leader for View %d (Epoch %d), waiting for Mempool...", currentView, e.currentEpoch)
		}
	}
}

func (e *Engine) snapshotProposalStateLocked() proposalSnapshot {
	snapshot := proposalSnapshot{
		view:  e.pacemaker.GetCurrentView(),
		epoch: e.currentEpoch,
		seed:  append([]byte(nil), e.beacon.GetCurrentSeed()...),
	}
	if e.valSet != nil {
		snapshot.configID = e.valSet.Epoch
		snapshot.leaderSetHash = append([]byte(nil), e.valSet.Hash()...)
	}
	if highQC := e.blockTree.GetHighQC(); highQC != nil {
		snapshot.highQC = highQC
	}
	snapshot.latestHeight = e.storage.GetLatestBlockHeight()
	snapshot.rank = e.rankState.ExpectedRank(snapshot.latestHeight + 1)
	return snapshot
}

func (e *Engine) snapshotProposalState() proposalSnapshot {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.snapshotProposalStateLocked()
}

func (e *Engine) buildProposalBlockWithSnapshot(certs []*types.VertexCertificate, snapshot proposalSnapshot) *types.Block {
	parentHash := make([]byte, 32)
	if snapshot.highQC != nil && snapshot.highQC.BlockHash != nil {
		parentHash = snapshot.highQC.BlockHash
	}

	block := types.NewBlock(
		snapshot.latestHeight+1,
		parentHash,
		e.nextBlockDataFromMempool(certs),
		snapshot.view,
		snapshot.epoch,
		e.nodeID,
		int64(snapshot.rank),
		snapshot.highQC,
		snapshot.seed, // Phase 5: Include Randomness
	)
	block.ConfigID = snapshot.configID
	block.LaneID = e.instanceID
	block.Payload = certs
	block.Hash = block.ComputeHash()
	return block
}

func (e *Engine) buildProposalBlock(certs []*types.VertexCertificate) *types.Block {
	return e.buildProposalBlockWithSnapshot(certs, e.snapshotProposalState())
}

func (e *Engine) proposeBlock(certs []*types.VertexCertificate) {
	// Phase 1: Build block and encode under lock
	e.mu.Lock()
	snapshot := e.snapshotProposalStateLocked()

	// Prevent equivocation: only propose once per view
	if snapshot.view <= e.lastProposedView {
		e.mu.Unlock()
		return
	}
	e.lastProposedView = snapshot.view

	block := e.buildProposalBlockWithSnapshot(certs, snapshot)

	msg := &types.Message{
		Type:          types.MsgProposal,
		SenderID:      e.nodeID,
		View:          snapshot.view,
		Epoch:         snapshot.epoch,
		ConfigID:      block.ConfigID,
		Lane:          block.LaneID,
		LeaderSetHash: snapshot.leaderSetHash,
		BarrierView:   snapshot.view,
		Instance:      e.instanceID,
		Block:         block,
	}
	if err := msg.Sign(e.keypair.PrivateKey); err != nil {
		e.mu.Unlock()
		logger.Error("Failed to sign proposal message: %v", err)
		return
	}

	data, err := types.EncodeMessage(msg)
	if err != nil {
		e.mu.Unlock()
		logger.Error("Failed to encode proposal message: %v", err)
		return
	}
	// Capture state needed after publish while still under lock
	consensusTopic := e.consensusTopic
	committeeSize := e.committeeSize
	var totalPower uint64
	if e.valSet != nil {
		totalPower = e.valSet.TotalPower
	}
	e.mu.Unlock()

	// Phase 2: Publish WITHOUT holding e.mu (validator callbacks need e.mu.RLock)
	if err := e.network.PublishTopic(consensusTopic, data); err != nil {
		logger.Error("Failed to publish proposal message: %v", err)
		return
	}

	// Phase 3: Process locally (no lock needed — same pattern as handleProposal)
	if err := e.blockTree.ProcessBlock(block); err != nil {
		logger.Error("Failed to process local proposal block: %v", err)
		return
	}

	var blockID types.Hash
	copy(blockID[:], block.Hash)

	// Phase 4: Leader generates own VRF proof for self-vote
	var selfVRFProof []byte
	if committeeSize > 0 && e.vrfPrivKey != nil {
		_, _, proof := e.beacon.AmICommitteeMember(e.vrfPrivKey, totalPower, committeeSize)
		selfVRFProof = proof
	}
	selfVote, err := types.NewVoteWithIdentity(blockID, snapshot.view, snapshot.epoch, block.ConfigID, block.LaneID, e.keypair.PublicKey, e.keypair.PrivateKey, selfVRFProof)
	if err != nil {
		logger.Error("Failed to create self vote: %v", err)
		return
	}
	e.handleVote(selfVote)
}

func (e *Engine) onBlockCommitted(block *types.Block) {
	if block == nil {
		return
	}
	if block.Height > e.lastCommittedHeight {
		e.lastCommittedHeight = block.Height
	}

	// Track leader reputation: success vs nil-block
	if e.reputation != nil {
		if len(block.Payload) == 0 && len(block.Data) == 0 {
			e.reputation.RecordNilBlock(block.LeaderID)
		} else {
			e.reputation.RecordSuccess(block.LeaderID)
		}
	}

	// Phase 5: GC old vote collectors to prevent memory leak at 1000-node scale.
	e.gcVoteCollectors()

	// §III-C / Algorithm II line 9: publish CommitQC to the Global Beacon Chain.
	// The GBC records cross-instance metadata (QCs, membership, checkpoints, policy).
	// Publishing after commit ensures the entry reflects a finalized decision.
	// Prefer gbc.Node.Propose() for G4 attestation protocol; fall back to Log.Publish().
	if e.gbcNode != nil {
		payload := block.Hash
		if payload == nil {
			payload = []byte{}
		}
		if _, err := e.gbcNode.Propose(gbc.EntryQC, payload); err != nil {
			if !gbc.IsNotProposer(err) {
				logger.Error("GBC propose failed for block height %d: %v", block.Height, err)
			}
		}
	} else if e.gbcLog != nil {
		payload := block.Hash
		if payload == nil {
			payload = []byte{}
		}
		entry := gbc.Entry{
			Height:  e.gbcLog.Height(),
			Type:    gbc.EntryQC,
			Payload: payload,
		}
		if err := e.gbcLog.Publish(entry); err != nil {
			logger.Error("GBC publish failed for block height %d: %v", block.Height, err)
		}
	}

	// G4 fix: Apply pending reconfigurations now that this height is committed.
	// This is the epoch safety barrier — reconfigs queued during ExecuteBlock()
	// only take effect after the block is committed via 3-chain or 2-chain fast path.
	e.rankState.OnCommit(block.Height)
	if e.outputChan != nil {
		rank := e.rankState.ExpectedRank(block.Height)
		output := InstanceOutput{
			InstanceID:       e.instanceID,
			LocalHeight:      block.Height,
			Rank:             rank,
			BlockHash:        block.Hash,
			Block:            block,
			EpochTransitions: nil,
		}
		e.outputChan <- output
	}
	logger.Info("Engine: committed state updated to height %d at epoch %d", e.lastCommittedHeight, e.currentEpoch)
}

// gcVoteCollectors removes completed or stale vote collectors.
// A collector is stale if its view is more than gcViewLag behind the current view.
// This prevents unbounded memory growth at 1000-node scale where each view
// creates new collector entries.
func (e *Engine) gcVoteCollectors() {
	const gcViewLag uint64 = 10

	currentView := e.pacemaker.GetCurrentView()
	if currentView <= gcViewLag {
		return
	}
	threshold := currentView - gcViewLag

	for key, collector := range e.voteCollectors {
		// Remove completed collectors (QC already formed) or stale ones
		if collector.done || collector.qc.View < threshold {
			delete(e.voteCollectors, key)
		}
	}

	// Also GC stale equivocation detection entries
	for view := range e.seenProposals {
		if view < threshold {
			delete(e.seenProposals, view)
		}
	}

	// GC stale TC collectors
	if e.vcm != nil {
		e.vcm.GCCollectors(threshold)
	}
}

// broadcastTimeoutVote broadcasts a timeout vote for the given view.
// When 2f+1 nodes broadcast timeout votes for the same view, a TC is formed.
func (e *Engine) broadcastTimeoutVote(view uint64) {
	highQC := e.blockTree.GetHighQC()

	tv := &types.TimeoutVote{
		View:      view,
		Epoch:     e.currentEpoch,
		ConfigID:  e.currentConfigIDSnapshot(),
		Lane:      e.instanceID,
		VoterID:   e.nodeID,
		HighestQC: highQC,
	}
	// Sign the timeout vote
	signingBytes := types.TimeoutSigningBytes(tv.View, tv.Epoch, tv.ConfigID, tv.Lane)
	tv.Signature = crypto.Sign(signingBytes, e.keypair.PrivateKey)

	// Handle own timeout vote locally
	if e.vcm != nil {
		e.vcm.HandleTimeoutVote(tv)
	}

	// Broadcast via network
	msg := &types.Message{
		Type:          types.MsgTimeout,
		SenderID:      e.nodeID,
		View:          view,
		Epoch:         e.currentEpoch,
		ConfigID:      e.currentConfigIDSnapshot(),
		Lane:          e.instanceID,
		LeaderSetHash: e.leaderSetHashSnapshot(),
		BarrierView:   view,
		Instance:      e.instanceID,
		TimeoutVote:   tv,
	}
	if err := msg.Sign(e.keypair.PrivateKey); err != nil {
		logger.Error("Failed to sign timeout vote message: %v", err)
		return
	}
	data, err := types.EncodeMessage(msg)
	if err != nil {
		logger.Error("Failed to encode timeout vote message: %v", err)
		return
	}
	if err := e.network.PublishTopic(e.consensusTopic, data); err != nil {
		logger.Error("Failed to broadcast timeout vote: %v", err)
	}
}

// handleTimeoutVote processes an incoming timeout vote and attempts to form a TC.
func (e *Engine) handleTimeoutVote(tv *types.TimeoutVote) {
	if tv == nil || e.vcm == nil {
		return
	}
	currentView := e.pacemaker.GetCurrentView()
	if tv.View < currentView {
		e.incRejected("stale_timeout_vote")
		return
	}
	if tv.View > currentView {
		e.incRejected("future_timeout_vote")
		return
	}
	if tv.Epoch != e.currentEpoch {
		e.incRejected("timeout_vote_epoch_mismatch")
		return
	}
	if tv.ConfigID != e.currentConfigIDSnapshot() {
		e.incRejected("timeout_vote_config_mismatch")
		return
	}
	if tv.Lane != e.instanceID {
		e.incRejected("timeout_vote_lane_mismatch")
		return
	}
	validator, exists := e.valSet.Validators[tv.VoterID]
	if !exists || validator == nil {
		e.incRejected("unknown_timeout_voter")
		return
	}
	signingBytes := types.TimeoutSigningBytes(tv.View, tv.Epoch, tv.ConfigID, tv.Lane)
	if !crypto.Verify(signingBytes, tv.Signature, validator.PublicKey) {
		e.incRejected("invalid_timeout_vote_signature")
		return
	}
	if tv.HighestQC != nil {
		if tv.HighestQC.Epoch != tv.Epoch || tv.HighestQC.ConfigID != tv.ConfigID || tv.HighestQC.Lane != tv.Lane {
			e.incRejected("timeout_vote_highqc_mismatch")
			return
		}
		if tv.HighestQC.View > tv.View {
			e.incRejected("timeout_vote_highqc_future_view")
			return
		}
		if !e.verifyQC(tv.HighestQC) {
			e.incRejected("timeout_vote_invalid_highqc")
			return
		}
	}

	tc := e.vcm.HandleTimeoutVote(tv)
	if tc == nil {
		return // Not yet quorum
	}

	// TC formed — advance to the next view and try to propose.
	// The TC proves that 2f+1 nodes timed out, so it's safe to skip
	// the failed view's QC.
	logger.Info("TC formed for View %d, advancing to View %d", tc.View, tc.View+1)
	e.pacemaker.AdvanceView(tc.View)

	// Broadcast NewView with TC so other nodes can advance too
	newViewMsg := e.newTCBackedNewViewMessage(tc)
	if newViewMsg == nil {
		logger.Error("Failed to build NewView message for TC view %d config %d", tc.View, tc.ConfigID)
		return
	}
	if err := newViewMsg.Sign(e.keypair.PrivateKey); err != nil {
		logger.Error("Failed to sign NewView message: %v", err)
		return
	}
	data, err := types.EncodeMessage(newViewMsg)
	if err != nil {
		logger.Error("Failed to encode NewView message: %v", err)
		return
	}
	if err := e.network.PublishTopic(e.consensusTopic, data); err != nil {
		logger.Error("Failed to broadcast NewView: %v", err)
	}

	e.tryPropose()
}

// handleNewView processes an incoming NewView message containing a TC.
// This allows nodes that haven't collected enough timeout votes locally
// to advance to the new view using the TC proof.
//
// G3 fix: When a TC is received, we merge TC.HighestQC with our local highQC.
// This ensures that after a network partition recovery, the new leader's proposal
// uses the most recent QC as justify, preventing SafeNode failures and liveness loss.
func (e *Engine) newTCBackedNewViewMessage(tc *types.TimeoutCertificate) *types.Message {
	if tc == nil {
		return nil
	}
	leaderSetHash := e.leaderSetHashForConfigID(tc.ConfigID)
	if len(leaderSetHash) == 0 {
		return nil
	}
	return &types.Message{
		Type:          types.MsgNewView,
		SenderID:      e.nodeID,
		View:          tc.View + 1,
		Epoch:         tc.Epoch,
		ConfigID:      tc.ConfigID,
		Lane:          tc.Lane,
		LeaderSetHash: leaderSetHash,
		BarrierView:   tc.View + 1,
		Instance:      tc.Lane,
		TC:            tc,
	}
}

func (e *Engine) handleNewView(msg *types.Message) {
	if msg == nil {
		return
	}

	if msg.TC != nil {
		// Verify TC has enough valid signatures (cryptographic check)
		if !e.verifyTC(msg.TC) {
			e.incRejected("newview_invalid_tc_signatures")
			logger.Error("NewView TC for View %d failed signature verification", msg.TC.View)
			return
		}
		if msg.TC.HighestQC != nil {
			if msg.TC.HighestQC.Epoch != msg.TC.Epoch || msg.TC.HighestQC.ConfigID != msg.TC.ConfigID || msg.TC.HighestQC.Lane != msg.TC.Lane {
				e.incRejected("newview_tc_highqc_mismatch")
				return
			}
			if msg.TC.HighestQC.View > msg.TC.View {
				e.incRejected("newview_tc_highqc_future_view")
				return
			}
			if !e.verifyQC(msg.TC.HighestQC) {
				e.incRejected("newview_invalid_tc_highqc")
				return
			}
		}

		currentView := e.pacemaker.GetCurrentView()
		advancedHighQC := false
		// G3 fix: Merge TC.HighestQC with local highQC.
		// After a partition, different nodes may have seen different QCs.
		// The TC carries the highest QC observed by any of the 2f+1 timeout voters.
		// We must adopt it if it's higher than our local highQC, otherwise our
		// next proposal may use a stale justify and be rejected by SafeNode.
		if msg.TC.HighestQC != nil {
			localHighQC := e.blockTree.GetHighQC()
			if localHighQC == nil || msg.TC.HighestQC.View > localHighQC.View {
				logger.Info("NewView: adopting TC.HighestQC (view=%d) over local highQC (view=%d)",
					msg.TC.HighestQC.View, func() uint64 {
						if localHighQC != nil {
							return localHighQC.View
						}
						return 0
					}())
				e.blockTree.OnVoteQC(msg.TC.HighestQC)
				advancedHighQC = true
			}
		}
		if msg.TC.View+1 <= currentView {
			if !advancedHighQC {
				e.incRejected("stale_newview_tc")
			}
			return
		}

		logger.Info("Received NewView with TC for View %d, advancing", msg.TC.View)
		e.pacemaker.AdvanceView(msg.TC.View)
		e.tryPropose()
	} else if msg.QC != nil {
		// NewView with QC only (standard HotStuff view advance)
		if !e.verifyQC(msg.QC) {
			e.incRejected("newview_invalid_qc")
			return
		}
		currentView := e.pacemaker.GetCurrentView()
		localHighQC := e.blockTree.GetHighQC()
		advancedHighQC := false
		if localHighQC == nil || msg.QC.View > localHighQC.View {
			e.blockTree.OnVoteQC(msg.QC)
			advancedHighQC = true
		}
		if msg.QC.View+1 <= currentView {
			if !advancedHighQC {
				e.incRejected("stale_newview_qc")
			}
			return
		}
		e.pacemaker.AdvanceView(msg.QC.View)
		e.tryPropose()
	}
}

func (e *Engine) AddTransaction(tx *types.Transaction) error {
	if tx == nil {
		e.incRejected("nil_transaction")
		return nil
	}
	err := e.mempool.SubmitTransaction(tx)
	if err != nil {
		logger.Info("AddTransaction FAILED instance=%d: %v", e.instanceID, err)
	}
	return err
}

func (e *Engine) nextBlockDataFromMempool(certs []*types.VertexCertificate) []byte {
	if len(certs) == 0 {
		return nil
	}
	if e.mempool == nil {
		return nil
	}
	vertex := e.mempool.GetVertex(certs[0].VertexHash)
	if vertex == nil || len(vertex.Txs) == 0 {
		return nil
	}
	tx := vertex.Txs[0]
	payload, err := json.Marshal(tx)
	if err != nil {
		return nil
	}
	return payload
}

func (e *Engine) incRejected(reason string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.rejected == nil {
		e.rejected = make(map[string]uint64)
	}
	e.rejected[reason]++
}

func (e *Engine) GetRejectedStats() map[string]uint64 {
	e.mu.RLock()
	defer e.mu.RUnlock()
	stats := make(map[string]uint64, len(e.rejected))
	for k, v := range e.rejected {
		stats[k] = v
	}
	return stats
}

// sortedValidatorIDs returns the current validator IDs sorted ascending.
func validatorIDsFromSet(valSet *types.ValidatorSet) []uint64 {
	if valSet == nil {
		return nil
	}
	ids := make([]uint64, 0, len(valSet.Validators))
	for id, v := range valSet.Validators {
		if v != nil && v.IsActive {
			ids = append(ids, id)
		}
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	return ids
}

func (e *Engine) sortedValidatorIDs() []uint64 {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return validatorIDsFromSet(e.valSet)
}

func (e *Engine) allowedLeaderIDsSnapshot(valSet *types.ValidatorSet) []uint64 {
	ids := validatorIDsFromSet(valSet)
	if e.hydra == nil {
		return ids
	}
	hydraIDs := e.hydra.AllowedLeaders()
	if len(hydraIDs) == 0 {
		return ids
	}
	return hydraIDs
}

func (e *Engine) allowedLeaderIDs() []uint64 {
	e.mu.RLock()
	valSet := e.valSet
	e.mu.RUnlock()
	return e.allowedLeaderIDsSnapshot(valSet)
}

func (e *Engine) refreshLeaderSelector() {
	e.mu.RLock()
	valSet := e.valSet
	e.mu.RUnlock()
	e.refreshLeaderSelectorFor(valSet)
}

func (e *Engine) refreshLeaderSelectorFor(valSet *types.ValidatorSet) {
	if e.pacemaker == nil {
		return
	}
	allowed := e.allowedLeaderIDsSnapshot(valSet)
	if len(allowed) == 0 {
		allowed = validatorIDsFromSet(valSet)
	}
	if len(allowed) > 0 {
		e.pacemaker.UpdateValidators(allowed)
	}
	// Use deterministic round-robin when VRF committee is disabled or beacon unavailable.
	// Beacon-based selection requires all nodes to have identical beacon state;
	// in small clusters or early views, divergence causes leader disagreement.
	if e.beacon == nil || e.committeeSize == 0 {
		e.pacemaker.SetLeaderSelector(nil)
		return
	}
	allowedCopy := append([]uint64(nil), allowed...)
	laneID := e.instanceID
	e.pacemaker.SetLeaderSelector(func(view uint64) uint64 {
		if len(allowedCopy) == 0 {
			return 0
		}
		return e.beacon.SelectLeader(view, laneID, allowedCopy)
	})
}

// GetLeaderReputation returns the reputation tracker for external inspection.
func (e *Engine) GetLeaderReputation() *LeaderReputation {
	return e.reputation
}

// SetHydraManager attaches a HydraManager to this engine.
// Hydra advisory state may influence leader eligibility, but auto-transition
// transaction injection is wired explicitly by cmd/octopus so engine attachment
// order cannot create a hidden configuration-effect path.
func (e *Engine) SetHydraManager(hm *hydra.HydraManager) {
	if hm == nil {
		return
	}
	e.hydra = hm
	e.refreshLeaderSelector()
}

// SetGBCLog attaches the Global Beacon Chain log (§III-C).
// When set, onBlockCommitted publishes CommitQC entries to the GBC.
func (e *Engine) SetGBCLog(log *gbc.Log) {
	e.gbcLog = log
}

// SetGBCNode attaches a GBC Node for the full Propose→Attest→Commit protocol (G4).
// When set, proposals go through the Node's attestation pathway instead of direct Log.Publish.
func (e *Engine) SetGBCNode(node *gbc.Node) {
	e.gbcNode = node
	e.gbcLog = node.Log() // keep log reference for reads
}

// GetHydraManager returns the attached HydraManager (may be nil).
func (e *Engine) GetHydraManager() *hydra.HydraManager {
	return e.hydra
}

// RegisterVRFPubKey registers a VRF public key for a validator.
// This is needed so the leader can verify VRF proofs in incoming votes.
func (e *Engine) RegisterVRFPubKey(validatorID uint64, pubKey kyber.Point) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.vrfPubKeys == nil {
		e.vrfPubKeys = make(map[uint64]kyber.Point)
	}
	e.vrfPubKeys[validatorID] = pubKey
}

// RegisterVRFPubKeyFromBytes decodes and registers a serialized VRF public key.
func (e *Engine) RegisterVRFPubKeyFromBytes(validatorID uint64, raw []byte) error {
	if len(raw) == 0 {
		return nil
	}
	pubKey, err := DecodeVRFPublicKey(raw)
	if err != nil {
		return err
	}
	e.RegisterVRFPubKey(validatorID, pubKey)
	return nil
}

// SetLocalVRFKeypair installs the local VRF keypair loaded from stable bootstrap.
func (e *Engine) SetLocalVRFKeypair(privKey kyber.Scalar, pubKey kyber.Point) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.vrfPrivKey = privKey
	e.vrfPubKey = pubKey
	if e.vrfPubKeys == nil {
		e.vrfPubKeys = make(map[uint64]kyber.Point)
	}
	if pubKey != nil {
		e.vrfPubKeys[e.nodeID] = pubKey
	}
}

// GetVRFPubKey returns this engine's VRF public key for distribution to peers.
func (e *Engine) GetVRFPubKey() kyber.Point {
	return e.vrfPubKey
}

// GetCommitteeSize returns the configured committee size.
func (e *Engine) GetCommitteeSize() int {
	return e.committeeSize
}

// SetCommitteeSize updates the VRF committee size at runtime.
// Set to 0 to disable VRF committee selection (all validators vote).
func (e *Engine) SetCommitteeSize(size int) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.committeeSize = size
}

func (e *Engine) SetAdaptiveTuning(tuning AdaptiveTuning) {
	e.mu.Lock()
	if tuning.CommitteeSize >= 0 {
		e.committeeSize = tuning.CommitteeSize
	}
	if tuning.TimeoutMs > 0 {
		e.timeoutMs = tuning.TimeoutMs
		if e.reputation != nil {
			e.reputation.SetBaseTimeoutMs(tuning.TimeoutMs)
		}
	}
	pm := e.pacemaker
	timeoutMs := e.timeoutMs
	committeeSize := e.committeeSize
	e.mu.Unlock()

	if pm != nil && timeoutMs > 0 {
		pm.SetTimeout(time.Duration(timeoutMs) * time.Millisecond)
	}
	_ = committeeSize
}

func (e *Engine) GetAdaptiveTuning() AdaptiveTuning {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return AdaptiveTuning{
		CommitteeSize: e.committeeSize,
		TimeoutMs:     e.timeoutMs,
	}
}

func (e *Engine) SetMempoolAdaptiveTuning(tuning mempool.AdaptiveTuning) {
	e.mu.RLock()
	mp := e.mempool
	e.mu.RUnlock()
	if mp != nil {
		mp.SetAdaptiveTuning(tuning)
	}
}

func (e *Engine) GetMempoolAdaptiveTuning() mempool.AdaptiveTuning {
	e.mu.RLock()
	mp := e.mempool
	e.mu.RUnlock()
	if mp == nil {
		return mempool.AdaptiveTuning{}
	}
	return mp.GetAdaptiveTuning()
}

func (e *Engine) GetInstanceID() uint64 {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.instanceID
}

// hydraConfigToValidatorSet converts a hydra.Configuration to types.ValidatorSet.
func hydraConfigToValidatorSet(config *hydra.Configuration) *types.ValidatorSet {
	validators := make(map[uint64]*types.Validator, len(config.Validators))
	for id, v := range config.Validators {
		validators[id] = &types.Validator{
			ID:        v.ID,
			PublicKey: v.PublicKey,
			Power:     v.Power,
			IsActive:  v.IsActive,
		}
	}
	return &types.ValidatorSet{
		Validators: validators,
		QuorumSize: uint64(config.QuorumSize),
		Epoch:      config.ID,
	}
}
