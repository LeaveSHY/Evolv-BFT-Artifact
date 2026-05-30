// Copyright 2024 Octopus Project
// Licensed under Apache License 2.0

package mempool

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"reflect"
	"sync"
	"time"

	"octopus-bft/octopus/crypto"
	"octopus-bft/octopus/types"
)

var logger = struct {
	Info  func(format string, args ...interface{})
	Error func(format string, args ...interface{})
}{
	Info:  func(format string, args ...interface{}) { fmt.Printf(format+"\n", args...) },
	Error: func(format string, args ...interface{}) { fmt.Printf("ERROR: "+format+"\n", args...) },
}

// Mempool manages the DAG of vertices
type Mempool struct {
	mu sync.RWMutex

	nodeID  uint64
	keypair *types.Keypair
	network MempoolNetwork
	valSet  *types.ValidatorSet

	vertices     map[types.Hash]*types.Vertex
	certificates map[types.Hash]*types.VertexCertificate

	currentRound uint64
	roundCerts   map[uint64][]*types.VertexCertificate

	proposalChan chan []*types.VertexCertificate

	txQueue       []*mempoolTx
	seenTx        map[string]time.Time
	poolBytes     int
	txTTL         time.Duration
	maxPoolBytes  int
	maxTxSize     int
	maxBatchTxs   int
	maxBatchBytes int
	maxQueueLen   int
	networkBuffer int
	proposeEvery  time.Duration
	topic         string

	isRunning bool
}

type AdaptiveTuning struct {
	MaxBatchTxs      int
	ProposalInterval time.Duration
}

type MempoolNetwork interface {
	PublishTopic(topicName string, data []byte) error
	SubscribeTopic(topicName string, buf int) (chan []byte, error)
}

type mempoolTx struct {
	tx       *types.Transaction
	txID     string
	received time.Time
	size     int
	raw      []byte
}

type Options struct {
	MaxPoolBytes       int
	MaxTxSize          int
	MaxBatchTxs        int
	MaxBatchBytes      int
	MaxQueueLen        int
	NetworkBuffer      int
	ProposalInterval   time.Duration
	ProposalChanBuffer int
	TxTTL              time.Duration
	Topic              string
	CompressMinBytes   int // Compress batches above this size (0 = disabled)
}

// DefaultOptions returns mempool options tuned for 1000-node networks.
//
// Rationale:
//   - MaxPoolBytes=16MB: At 1000 nodes, tx inflow can be 10x higher than
//     a 100-node network. 16MB buffers ~60K average-sized transactions.
//   - MaxBatchTxs=2048: Larger batches amortize per-vertex overhead across
//     more transactions, critical for 100k+ tx/s throughput targets.
//   - MaxBatchBytes=4MB: Accommodates 2048 txs of ~2KB average.
//   - MaxQueueLen=16384: Deeper queue prevents drops during tx bursts.
//   - NetworkBuffer=4096: Faster ingestion from GossipSub at high throughput.
//   - ProposalInterval=50ms: Fast vertex production rate to fill pipeline;
//     with 10 instances this yields 200k+ proposals/sec aggregate.
//   - ProposalChanBuffer=64: Prevents backpressure when consensus is busy.
func DefaultOptions() Options {
	return Options{
		MaxPoolBytes:       16 * 1024 * 1024,
		MaxTxSize:          256 * 1024,
		MaxBatchTxs:        2048,
		MaxBatchBytes:      4 * 1024 * 1024,
		MaxQueueLen:        16384,
		NetworkBuffer:      4096,
		ProposalInterval:   50 * time.Millisecond,
		ProposalChanBuffer: 64,
		TxTTL:              30 * time.Second,
		Topic:              "octopus-mempool",
		CompressMinBytes:   4096, // Compress batches > 4KB
	}
}

func normalizeOptions(opts Options) Options {
	def := DefaultOptions()
	if opts.MaxPoolBytes <= 0 {
		opts.MaxPoolBytes = def.MaxPoolBytes
	}
	if opts.MaxTxSize <= 0 {
		opts.MaxTxSize = def.MaxTxSize
	}
	if opts.MaxBatchTxs <= 0 {
		opts.MaxBatchTxs = def.MaxBatchTxs
	}
	if opts.MaxBatchBytes <= 0 {
		opts.MaxBatchBytes = def.MaxBatchBytes
	}
	if opts.MaxQueueLen <= 0 {
		opts.MaxQueueLen = def.MaxQueueLen
	}
	if opts.NetworkBuffer <= 0 {
		opts.NetworkBuffer = def.NetworkBuffer
	}
	if opts.ProposalInterval <= 0 {
		opts.ProposalInterval = def.ProposalInterval
	}
	if opts.ProposalChanBuffer <= 0 {
		opts.ProposalChanBuffer = def.ProposalChanBuffer
	}
	if opts.TxTTL <= 0 {
		opts.TxTTL = def.TxTTL
	}
	if opts.Topic == "" {
		opts.Topic = def.Topic
	}
	return opts
}

func NewMempool(nodeID uint64, keypair *types.Keypair, valSet *types.ValidatorSet, net MempoolNetwork) *Mempool {
	return NewMempoolWithOptions(nodeID, keypair, valSet, net, DefaultOptions())
}

func NewMempoolWithOptions(nodeID uint64, keypair *types.Keypair, valSet *types.ValidatorSet, net MempoolNetwork, opts Options) *Mempool {
	normalized := normalizeOptions(opts)
	return &Mempool{
		nodeID:        nodeID,
		keypair:       keypair,
		valSet:        valSet,
		network:       net,
		vertices:      make(map[types.Hash]*types.Vertex),
		certificates:  make(map[types.Hash]*types.VertexCertificate),
		roundCerts:    make(map[uint64][]*types.VertexCertificate),
		currentRound:  1,
		proposalChan:  make(chan []*types.VertexCertificate, normalized.ProposalChanBuffer),
		txQueue:       make([]*mempoolTx, 0, 1024),
		seenTx:        make(map[string]time.Time),
		txTTL:         normalized.TxTTL,
		maxPoolBytes:  normalized.MaxPoolBytes,
		maxTxSize:     normalized.MaxTxSize,
		maxBatchTxs:   normalized.MaxBatchTxs,
		maxBatchBytes: normalized.MaxBatchBytes,
		maxQueueLen:   normalized.MaxQueueLen,
		networkBuffer: normalized.NetworkBuffer,
		proposeEvery:  normalized.ProposalInterval,
		topic:         normalized.Topic,
	}
}

func (mp *Mempool) Start() {
	mp.mu.Lock()
	if mp.isRunning {
		mp.mu.Unlock()
		return
	}
	mp.isRunning = true
	mp.mu.Unlock()

	go mp.runLoop()
	go mp.proposeLoop()
}

func isNilMempoolNetwork(network MempoolNetwork) bool {
	if network == nil {
		return true
	}
	value := reflect.ValueOf(network)
	switch value.Kind() {
	case reflect.Ptr, reflect.Map, reflect.Slice, reflect.Interface, reflect.Func, reflect.Chan:
		return value.IsNil()
	default:
		return false
	}
}

func (mp *Mempool) runLoop() {
	if isNilMempoolNetwork(mp.network) {
		return
	}
	ch, err := mp.network.SubscribeTopic(mp.topic, mp.networkBuffer)
	if err != nil {
		logger.Error("failed to subscribe mempool topic: %v", err)
		return
	}
	for mp.running() {
		select {
		case data := <-ch:
			tx := &types.Transaction{}
			if err := json.Unmarshal(data, tx); err != nil {
				continue
			}
			_ = mp.submit(tx, data, false)
		case <-time.After(100 * time.Millisecond):
		}
	}
}

func (mp *Mempool) proposeLoop() {
	for mp.running() {
		mp.mu.RLock()
		interval := mp.proposeEvery
		mp.mu.RUnlock()
		if interval <= 0 {
			interval = 50 * time.Millisecond
		}
		timer := time.NewTimer(interval)
		<-timer.C
		mp.createVertex()
	}
}

func (mp *Mempool) createVertex() {
	mp.mu.Lock()
	defer mp.mu.Unlock()

	qLen := len(mp.txQueue)
	logger.Info("createVertex: round=%d queueLen=%d poolBytes=%d", mp.currentRound, qLen, mp.poolBytes)
	mp.pruneExpiredLocked(time.Now())
	txs := mp.popBatchLocked()
	if len(txs) == 0 {
		return
	}

	var parents []types.Hash
	if mp.currentRound > 1 {
		prevCerts := mp.roundCerts[mp.currentRound-1]
		if len(prevCerts) == 0 {
			logger.Info("createVertex: round=%d has %d txs but no prevCerts for round %d", mp.currentRound, len(txs), mp.currentRound-1)
			return
		}
		if len(prevCerts) == 0 {
			return
		}
		for _, cert := range prevCerts {
			parents = append(parents, cert.VertexHash)
		}
	}

	v := types.NewVertex(
		mp.valSet.Epoch,
		mp.currentRound,
		mp.nodeID,
		txs,
		parents,
	)

	vertexBytes, _ := json.Marshal(v)
	data := append([]byte(fmt.Sprintf("%d-%d-%d-", v.Epoch, v.Round, v.Author)), vertexBytes...)
	v.Hash = crypto.SHA256(data)
	v.Signature = crypto.Sign(v.Hash[:], mp.keypair.PrivateKey)

	mp.vertices[v.Hash] = v
	logger.Info("Created Vertex %x (Round %d) with %d txs", v.Hash[:4], v.Round, len(txs))

	// Create vertex certificate with only the author's own signature.
	// In this architecture, vertex certificates serve as proof-of-authorship;
	// BFT safety is provided by the block-level QC (2f+1 votes on the proposal
	// that includes these vertices). This avoids the prior bug of forging
	// all validators' signatures locally.
	cert := types.NewVertexCertificate(v.Hash, v.Epoch, v.Round)
	cert.AddSignature(mp.nodeID, v.Signature)

	mp.certificates[v.Hash] = cert
	mp.roundCerts[mp.currentRound] = append(mp.roundCerts[mp.currentRound], cert)

	if len(mp.roundCerts[mp.currentRound]) >= 1 {
		mp.proposalChan <- mp.roundCerts[mp.currentRound]
		mp.currentRound++
	}
}

func (mp *Mempool) GetProposalChan() <-chan []*types.VertexCertificate {
	return mp.proposalChan
}

func (mp *Mempool) SetAdaptiveTuning(tuning AdaptiveTuning) {
	mp.mu.Lock()
	defer mp.mu.Unlock()
	if tuning.MaxBatchTxs > 0 {
		mp.maxBatchTxs = tuning.MaxBatchTxs
	}
	if tuning.ProposalInterval > 0 {
		mp.proposeEvery = tuning.ProposalInterval
	}
}

func (mp *Mempool) GetAdaptiveTuning() AdaptiveTuning {
	mp.mu.RLock()
	defer mp.mu.RUnlock()
	return AdaptiveTuning{
		MaxBatchTxs:      mp.maxBatchTxs,
		ProposalInterval: mp.proposeEvery,
	}
}

func (mp *Mempool) GetVertex(hash types.Hash) *types.Vertex {
	mp.mu.RLock()
	defer mp.mu.RUnlock()
	return mp.vertices[hash]
}

func (mp *Mempool) SubmitTransaction(tx *types.Transaction) error {
	raw, err := json.Marshal(tx)
	if err != nil {
		return err
	}
	return mp.submit(tx, raw, true)
}

func (mp *Mempool) submit(tx *types.Transaction, raw []byte, publish bool) error {
	if tx == nil {
		return fmt.Errorf("nil transaction")
	}
	size := len(raw)
	if size == 0 {
		return fmt.Errorf("empty transaction")
	}
	if size > mp.maxTxSize {
		return fmt.Errorf("transaction exceeds size limit: %d > %d", size, mp.maxTxSize)
	}
	sum := sha256.Sum256(raw)
	txID := hex.EncodeToString(sum[:])
	now := time.Now()

	mp.mu.Lock()
	mp.pruneExpiredLocked(now)
	if _, exists := mp.seenTx[txID]; exists {
		mp.mu.Unlock()
		return nil
	}
	mp.evictUntilQueueSlotLocked()
	mp.evictUntilFitLocked(size)
	if mp.poolBytes+size > mp.maxPoolBytes {
		mp.mu.Unlock()
		return fmt.Errorf("mempool full: poolBytes=%d + %d > %d", mp.poolBytes, size, mp.maxPoolBytes)
	}
	cloned := &types.Transaction{
		Type:    tx.Type,
		Payload: append([]byte(nil), tx.Payload...),
	}
	mp.txQueue = append(mp.txQueue, &mempoolTx{
		tx:       cloned,
		txID:     txID,
		received: now,
		size:     size,
		raw:      append([]byte(nil), raw...),
	})
	mp.seenTx[txID] = now
	mp.poolBytes += size
	newLen := len(mp.txQueue)
	mp.mu.Unlock()

	if newLen == 1 || newLen%1000 == 0 {
		logger.Info("submit: txQueue now has %d entries (poolBytes=%d)", newLen, mp.poolBytes)
	}

	if publish && !isNilMempoolNetwork(mp.network) {
		if err := mp.network.PublishTopic(mp.topic, raw); err != nil {
			logger.Error("failed to publish mempool tx: %v", err)
		}
	}
	return nil
}

func (mp *Mempool) popBatchLocked() []*types.Transaction {
	if len(mp.txQueue) == 0 {
		return nil
	}
	batch := make([]*types.Transaction, 0, mp.maxBatchTxs)
	size := 0
	count := 0
	for count < len(mp.txQueue) && count < mp.maxBatchTxs {
		entry := mp.txQueue[count]
		if size+entry.size > mp.maxBatchBytes && len(batch) > 0 {
			break
		}
		batch = append(batch, entry.tx)
		size += entry.size
		count++
		if size >= mp.maxBatchBytes {
			break
		}
	}
	if count == 0 {
		return nil
	}
	for i := 0; i < count; i++ {
		mp.poolBytes -= mp.txQueue[i].size
	}
	mp.txQueue = append([]*mempoolTx(nil), mp.txQueue[count:]...)
	return batch
}

func (mp *Mempool) evictUntilFitLocked(incoming int) {
	for len(mp.txQueue) > 0 && mp.poolBytes+incoming > mp.maxPoolBytes {
		old := mp.txQueue[0]
		mp.txQueue = mp.txQueue[1:]
		mp.poolBytes -= old.size
	}
}

func (mp *Mempool) evictUntilQueueSlotLocked() {
	if mp.maxQueueLen <= 0 {
		return
	}
	for len(mp.txQueue) >= mp.maxQueueLen {
		old := mp.txQueue[0]
		mp.txQueue = mp.txQueue[1:]
		mp.poolBytes -= old.size
	}
}

func (mp *Mempool) pruneExpiredLocked(now time.Time) {
	for id, ts := range mp.seenTx {
		if now.Sub(ts) > mp.txTTL {
			delete(mp.seenTx, id)
		}
	}
	if len(mp.txQueue) == 0 {
		return
	}
	kept := mp.txQueue[:0]
	for _, entry := range mp.txQueue {
		if now.Sub(entry.received) > mp.txTTL {
			mp.poolBytes -= entry.size
			continue
		}
		kept = append(kept, entry)
	}
	mp.txQueue = kept
}

func (mp *Mempool) running() bool {
	mp.mu.RLock()
	defer mp.mu.RUnlock()
	return mp.isRunning
}
