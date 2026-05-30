package mempool

import (
	"encoding/json"
	"fmt"
	"testing"
	"time"

	octcrypto "octopus-bft/octopus/crypto"
	"octopus-bft/octopus/types"
)

// --- 1. DefaultOptions ---

func TestDefaultOptionsAllFieldsNonZero(t *testing.T) {
	opts := DefaultOptions()
	if opts.MaxPoolBytes <= 0 {
		t.Errorf("MaxPoolBytes should be positive, got %d", opts.MaxPoolBytes)
	}
	if opts.MaxTxSize <= 0 {
		t.Errorf("MaxTxSize should be positive, got %d", opts.MaxTxSize)
	}
	if opts.MaxBatchTxs <= 0 {
		t.Errorf("MaxBatchTxs should be positive, got %d", opts.MaxBatchTxs)
	}
	if opts.MaxBatchBytes <= 0 {
		t.Errorf("MaxBatchBytes should be positive, got %d", opts.MaxBatchBytes)
	}
	if opts.MaxQueueLen <= 0 {
		t.Errorf("MaxQueueLen should be positive, got %d", opts.MaxQueueLen)
	}
	if opts.NetworkBuffer <= 0 {
		t.Errorf("NetworkBuffer should be positive, got %d", opts.NetworkBuffer)
	}
	if opts.ProposalInterval <= 0 {
		t.Errorf("ProposalInterval should be positive, got %v", opts.ProposalInterval)
	}
	if opts.ProposalChanBuffer <= 0 {
		t.Errorf("ProposalChanBuffer should be positive, got %d", opts.ProposalChanBuffer)
	}
	if opts.TxTTL <= 0 {
		t.Errorf("TxTTL should be positive, got %v", opts.TxTTL)
	}
	if opts.Topic == "" {
		t.Errorf("Topic should be non-empty")
	}
}

// --- 2. normalizeOptions ---

func TestNormalizeOptionsZeroInputGetsDefaults(t *testing.T) {
	zero := Options{}
	got := normalizeOptions(zero)
	def := DefaultOptions()

	if got.MaxPoolBytes != def.MaxPoolBytes {
		t.Errorf("MaxPoolBytes: got %d, want %d", got.MaxPoolBytes, def.MaxPoolBytes)
	}
	if got.MaxTxSize != def.MaxTxSize {
		t.Errorf("MaxTxSize: got %d, want %d", got.MaxTxSize, def.MaxTxSize)
	}
	if got.MaxBatchTxs != def.MaxBatchTxs {
		t.Errorf("MaxBatchTxs: got %d, want %d", got.MaxBatchTxs, def.MaxBatchTxs)
	}
	if got.MaxBatchBytes != def.MaxBatchBytes {
		t.Errorf("MaxBatchBytes: got %d, want %d", got.MaxBatchBytes, def.MaxBatchBytes)
	}
	if got.MaxQueueLen != def.MaxQueueLen {
		t.Errorf("MaxQueueLen: got %d, want %d", got.MaxQueueLen, def.MaxQueueLen)
	}
	if got.NetworkBuffer != def.NetworkBuffer {
		t.Errorf("NetworkBuffer: got %d, want %d", got.NetworkBuffer, def.NetworkBuffer)
	}
	if got.ProposalInterval != def.ProposalInterval {
		t.Errorf("ProposalInterval: got %v, want %v", got.ProposalInterval, def.ProposalInterval)
	}
	if got.ProposalChanBuffer != def.ProposalChanBuffer {
		t.Errorf("ProposalChanBuffer: got %d, want %d", got.ProposalChanBuffer, def.ProposalChanBuffer)
	}
	if got.TxTTL != def.TxTTL {
		t.Errorf("TxTTL: got %v, want %v", got.TxTTL, def.TxTTL)
	}
	if got.Topic != def.Topic {
		t.Errorf("Topic: got %q, want %q", got.Topic, def.Topic)
	}
}

func TestNormalizeOptionsPartialPreservesExplicit(t *testing.T) {
	partial := Options{
		MaxPoolBytes: 999,
		MaxTxSize:    555,
		Topic:        "custom-topic",
	}
	got := normalizeOptions(partial)
	def := DefaultOptions()

	if got.MaxPoolBytes != 999 {
		t.Errorf("MaxPoolBytes should preserve explicit 999, got %d", got.MaxPoolBytes)
	}
	if got.MaxTxSize != 555 {
		t.Errorf("MaxTxSize should preserve explicit 555, got %d", got.MaxTxSize)
	}
	if got.Topic != "custom-topic" {
		t.Errorf("Topic should preserve explicit, got %q", got.Topic)
	}
	// Zero fields should get defaults
	if got.MaxBatchTxs != def.MaxBatchTxs {
		t.Errorf("MaxBatchTxs: got %d, want default %d", got.MaxBatchTxs, def.MaxBatchTxs)
	}
	if got.ProposalInterval != def.ProposalInterval {
		t.Errorf("ProposalInterval: got %v, want default %v", got.ProposalInterval, def.ProposalInterval)
	}
}

// --- 3. isNilMempoolNetwork ---

func TestIsNilMempoolNetworkNil(t *testing.T) {
	if !isNilMempoolNetwork(nil) {
		t.Error("expected true for nil interface")
	}
}

func TestIsNilMempoolNetworkTypedNil(t *testing.T) {
	var net *fakeNetwork = nil
	// Passing a typed nil pointer as interface
	if !isNilMempoolNetwork(net) {
		t.Error("expected true for typed-nil *fakeNetwork")
	}
}

func TestIsNilMempoolNetworkValid(t *testing.T) {
	net := &fakeNetwork{}
	if isNilMempoolNetwork(net) {
		t.Error("expected false for valid network")
	}
}

// --- 4. submit error paths ---

func TestSubmitNilTransaction(t *testing.T) {
	mp := buildTestMempool(nil)
	err := mp.submit(nil, []byte("data"), false)
	if err == nil {
		t.Fatal("expected error for nil transaction")
	}
}

func TestSubmitEmptyRaw(t *testing.T) {
	mp := buildTestMempool(nil)
	tx := &types.Transaction{Type: types.TxTypeNormal, Payload: []byte("x")}
	err := mp.submit(tx, []byte{}, false)
	if err == nil {
		t.Fatal("expected error for empty raw data")
	}
}

func TestSubmitOversizedTransaction(t *testing.T) {
	mp := buildTestMempool(nil)
	mp.maxTxSize = 10
	tx := &types.Transaction{Type: types.TxTypeNormal, Payload: []byte("x")}
	oversized := make([]byte, 100)
	err := mp.submit(tx, oversized, false)
	if err == nil {
		t.Fatal("expected error for oversized transaction")
	}
}

func TestSubmitMempoolFull(t *testing.T) {
	mp := buildTestMempool(nil)
	// Set pool to tiny size that can't fit even one tx
	mp.maxPoolBytes = 1
	mp.maxQueueLen = 0 // disable queue-slot eviction
	tx := &types.Transaction{Type: types.TxTypeNormal, Payload: []byte("data")}
	raw, _ := json.Marshal(tx)
	err := mp.submit(tx, raw, false)
	if err == nil {
		t.Fatal("expected error for full mempool")
	}
}

// --- 5. GetVertex ---

func TestGetVertexFound(t *testing.T) {
	mp := buildTestMempool(nil)
	tx := &types.Transaction{Type: types.TxTypeNormal, Payload: []byte("gv-test")}
	if err := mp.SubmitTransaction(tx); err != nil {
		t.Fatalf("submit failed: %v", err)
	}
	mp.createVertex()
	if len(mp.vertices) == 0 {
		t.Fatal("expected at least one vertex after createVertex")
	}
	// Get the hash of the created vertex
	var hash types.Hash
	for h := range mp.vertices {
		hash = h
		break
	}
	got := mp.GetVertex(hash)
	if got == nil {
		t.Fatal("GetVertex returned nil for existing hash")
	}
	if len(got.Txs) != 1 {
		t.Fatalf("expected 1 tx in vertex, got %d", len(got.Txs))
	}
}

func TestGetVertexNotFound(t *testing.T) {
	mp := buildTestMempool(nil)
	var missing types.Hash
	missing[0] = 0xFF
	got := mp.GetVertex(missing)
	if got != nil {
		t.Fatal("GetVertex should return nil for non-existent hash")
	}
}

// --- 6. Start idempotency ---

func TestStartIdempotent(t *testing.T) {
	net := &fakeNetwork{subChan: make(chan []byte, 1)}
	mp := buildTestMempool(net)
	// Should not panic on double-start
	mp.Start()
	mp.Start()
	// Verify still running
	if !mp.running() {
		t.Fatal("expected mempool to be running after Start()")
	}
	// Cleanup
	mp.mu.Lock()
	mp.isRunning = false
	mp.mu.Unlock()
	time.Sleep(20 * time.Millisecond)
}

// --- 7. running() ---

func TestRunningFalseByDefault(t *testing.T) {
	mp := buildTestMempool(nil)
	if mp.running() {
		t.Error("expected running() = false before Start()")
	}
}

func TestRunningTrueAfterSet(t *testing.T) {
	mp := buildTestMempool(nil)
	mp.mu.Lock()
	mp.isRunning = true
	mp.mu.Unlock()
	if !mp.running() {
		t.Error("expected running() = true after setting isRunning")
	}
}

// --- 8. popBatchLocked ---

func TestPopBatchLockedEmptyQueue(t *testing.T) {
	mp := buildTestMempool(nil)
	mp.mu.Lock()
	batch := mp.popBatchLocked()
	mp.mu.Unlock()
	if batch != nil {
		t.Fatalf("expected nil batch from empty queue, got %d txs", len(batch))
	}
}

func TestPopBatchLockedRespectsBytesLimit(t *testing.T) {
	mp := buildTestMempool(nil)
	mp.maxBatchTxs = 100 // high tx count limit
	// Submit 3 txs, set byte limit so only 2 fit
	tx1 := &types.Transaction{Type: types.TxTypeNormal, Payload: []byte("aa")}
	tx2 := &types.Transaction{Type: types.TxTypeNormal, Payload: []byte("bb")}
	tx3 := &types.Transaction{Type: types.TxTypeNormal, Payload: []byte("cc")}
	_ = mp.SubmitTransaction(tx1)
	_ = mp.SubmitTransaction(tx2)
	_ = mp.SubmitTransaction(tx3)

	// Each marshaled tx is about the same size, compute it
	raw, _ := json.Marshal(tx1)
	txSize := len(raw)
	// Allow exactly 2 txs worth of bytes
	mp.mu.Lock()
	mp.maxBatchBytes = txSize*2 + 1
	batch := mp.popBatchLocked()
	remaining := len(mp.txQueue)
	mp.mu.Unlock()

	if len(batch) != 2 {
		t.Fatalf("expected batch of 2 txs (byte limit), got %d", len(batch))
	}
	if remaining != 1 {
		t.Fatalf("expected 1 remaining in queue, got %d", remaining)
	}
}

func TestPopBatchLockedRespectsTxCountLimit(t *testing.T) {
	mp := buildTestMempool(nil)
	mp.maxBatchTxs = 2
	mp.maxBatchBytes = 1024 * 1024 // large byte limit
	_ = mp.SubmitTransaction(&types.Transaction{Type: types.TxTypeNormal, Payload: []byte("p1")})
	_ = mp.SubmitTransaction(&types.Transaction{Type: types.TxTypeNormal, Payload: []byte("p2")})
	_ = mp.SubmitTransaction(&types.Transaction{Type: types.TxTypeNormal, Payload: []byte("p3")})

	mp.mu.Lock()
	batch := mp.popBatchLocked()
	remaining := len(mp.txQueue)
	mp.mu.Unlock()

	if len(batch) != 2 {
		t.Fatalf("expected batch of 2 txs (count limit), got %d", len(batch))
	}
	if remaining != 1 {
		t.Fatalf("expected 1 remaining in queue, got %d", remaining)
	}
}

// --- 9. NewMempoolWithOptions ---

func TestNewMempoolWithOptionsCustomValues(t *testing.T) {
	kp, _ := octcrypto.GenerateKeyPair()
	validators := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: []byte("v0"), Power: 1, IsActive: true},
	}
	valSet := types.NewValidatorSet(1, validators)
	net := &fakeNetwork{}

	custom := Options{
		MaxPoolBytes:       8 * 1024 * 1024,
		MaxTxSize:          128 * 1024,
		MaxBatchTxs:        512,
		MaxBatchBytes:      2 * 1024 * 1024,
		MaxQueueLen:        8192,
		NetworkBuffer:      2048,
		ProposalInterval:   100 * time.Millisecond,
		ProposalChanBuffer: 32,
		TxTTL:              60 * time.Second,
		Topic:              "test-topic",
	}

	mp := NewMempoolWithOptions(0, &types.Keypair{PublicKey: kp.PublicKey, PrivateKey: kp.PrivateKey}, valSet, net, custom)

	if mp.maxPoolBytes != 8*1024*1024 {
		t.Errorf("maxPoolBytes: got %d, want %d", mp.maxPoolBytes, 8*1024*1024)
	}
	if mp.maxTxSize != 128*1024 {
		t.Errorf("maxTxSize: got %d, want %d", mp.maxTxSize, 128*1024)
	}
	if mp.maxBatchTxs != 512 {
		t.Errorf("maxBatchTxs: got %d, want 512", mp.maxBatchTxs)
	}
	if mp.maxBatchBytes != 2*1024*1024 {
		t.Errorf("maxBatchBytes: got %d, want %d", mp.maxBatchBytes, 2*1024*1024)
	}
	if mp.maxQueueLen != 8192 {
		t.Errorf("maxQueueLen: got %d, want 8192", mp.maxQueueLen)
	}
	if mp.networkBuffer != 2048 {
		t.Errorf("networkBuffer: got %d, want 2048", mp.networkBuffer)
	}
	if mp.proposeEvery != 100*time.Millisecond {
		t.Errorf("proposeEvery: got %v, want 100ms", mp.proposeEvery)
	}
	if mp.txTTL != 60*time.Second {
		t.Errorf("txTTL: got %v, want 60s", mp.txTTL)
	}
	if mp.topic != "test-topic" {
		t.Errorf("topic: got %q, want %q", mp.topic, "test-topic")
	}
	if mp.nodeID != 0 {
		t.Errorf("nodeID: got %d, want 0", mp.nodeID)
	}
	if mp.currentRound != 1 {
		t.Errorf("currentRound: got %d, want 1", mp.currentRound)
	}
}

func TestNewMempoolWithOptionsZeroGetDefaults(t *testing.T) {
	kp, _ := octcrypto.GenerateKeyPair()
	validators := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: []byte("v0"), Power: 1, IsActive: true},
	}
	valSet := types.NewValidatorSet(1, validators)

	mp := NewMempoolWithOptions(1, &types.Keypair{PublicKey: kp.PublicKey, PrivateKey: kp.PrivateKey}, valSet, nil, Options{})
	def := DefaultOptions()

	if mp.maxPoolBytes != def.MaxPoolBytes {
		t.Errorf("maxPoolBytes: got %d, want default %d", mp.maxPoolBytes, def.MaxPoolBytes)
	}
	if mp.maxBatchTxs != def.MaxBatchTxs {
		t.Errorf("maxBatchTxs: got %d, want default %d", mp.maxBatchTxs, def.MaxBatchTxs)
	}
	if mp.topic != def.Topic {
		t.Errorf("topic: got %q, want default %q", mp.topic, def.Topic)
	}
}

// --- 10. proposeLoop coverage ---

func TestProposeLoopExecutesOneIteration(t *testing.T) {
	mp := buildTestMempool(nil)
	mp.proposeEvery = 10 * time.Millisecond
	// Pre-load a transaction so createVertex has work
	tx := &types.Transaction{Type: types.TxTypeNormal, Payload: []byte("propose-loop-test")}
	_ = mp.SubmitTransaction(tx)

	mp.mu.Lock()
	mp.isRunning = true
	mp.mu.Unlock()

	go mp.proposeLoop()
	// Wait enough time for one iteration
	time.Sleep(30 * time.Millisecond)

	mp.mu.Lock()
	mp.isRunning = false
	mp.mu.Unlock()
	time.Sleep(20 * time.Millisecond)

	// Should have created a vertex
	mp.mu.RLock()
	vCount := len(mp.vertices)
	mp.mu.RUnlock()
	if vCount == 0 {
		t.Fatal("expected proposeLoop to create at least one vertex")
	}
}

func TestProposeLoopZeroInterval(t *testing.T) {
	mp := buildTestMempool(nil)
	mp.proposeEvery = 0 // triggers the <= 0 fallback to 50ms
	tx := &types.Transaction{Type: types.TxTypeNormal, Payload: []byte("zero-interval")}
	_ = mp.SubmitTransaction(tx)

	mp.mu.Lock()
	mp.isRunning = true
	mp.mu.Unlock()

	go mp.proposeLoop()
	time.Sleep(80 * time.Millisecond)

	mp.mu.Lock()
	mp.isRunning = false
	mp.mu.Unlock()
	time.Sleep(20 * time.Millisecond)

	mp.mu.RLock()
	vCount := len(mp.vertices)
	mp.mu.RUnlock()
	if vCount == 0 {
		t.Fatal("expected proposeLoop to create vertex with zero-interval fallback")
	}
}

// --- 11. runLoop subscribe error path ---

type errorNetwork struct {
	fakeNetwork
	subErr error
}

func (e *errorNetwork) SubscribeTopic(topicName string, buf int) (chan []byte, error) {
	return nil, e.subErr
}

func TestRunLoopSubscribeError(t *testing.T) {
	net := &errorNetwork{subErr: fmt.Errorf("subscribe failed")}
	mp := buildTestMempool(net)
	mp.mu.Lock()
	mp.isRunning = true
	mp.mu.Unlock()

	// runLoop should return immediately on subscribe error
	done := make(chan struct{})
	go func() {
		mp.runLoop()
		close(done)
	}()
	select {
	case <-done:
		// success - runLoop returned
	case <-time.After(500 * time.Millisecond):
		t.Fatal("runLoop did not exit after subscribe error")
	}
}

func TestRunLoopNilNetworkReturns(t *testing.T) {
	mp := buildTestMempool(nil)
	mp.mu.Lock()
	mp.isRunning = true
	mp.mu.Unlock()

	done := make(chan struct{})
	go func() {
		mp.runLoop()
		close(done)
	}()
	select {
	case <-done:
		// success - runLoop returned for nil network
	case <-time.After(500 * time.Millisecond):
		t.Fatal("runLoop did not exit for nil network")
	}
}

func TestRunLoopTimeoutBranch(t *testing.T) {
	// Create a network with a channel that never receives - forces timeout path
	net := &fakeNetwork{subChan: make(chan []byte, 1)}
	mp := buildTestMempool(net)
	mp.mu.Lock()
	mp.isRunning = true
	mp.mu.Unlock()

	go mp.runLoop()
	// Let the timeout (100ms) fire at least once
	time.Sleep(200 * time.Millisecond)

	mp.mu.Lock()
	mp.isRunning = false
	mp.mu.Unlock()
	time.Sleep(150 * time.Millisecond)
}

// --- 12. createVertex at round > 1 with no prev certs ---

func TestCreateVertexNoPrevCertsReturnsEarly(t *testing.T) {
	mp := buildTestMempool(nil)
	tx := &types.Transaction{Type: types.TxTypeNormal, Payload: []byte("round2")}
	_ = mp.SubmitTransaction(tx)

	// Advance round without having any certs for previous round
	mp.mu.Lock()
	mp.currentRound = 5
	mp.mu.Unlock()

	mp.createVertex()
	// Should have returned early, no vertex created
	mp.mu.RLock()
	vCount := len(mp.vertices)
	mp.mu.RUnlock()
	if vCount != 0 {
		t.Fatalf("expected no vertex when prev round has no certs, got %d", vCount)
	}
}

func TestCreateVertexWithParents(t *testing.T) {
	mp := buildTestMempool(nil)
	// Submit a tx and create vertex at round 1 (no parents needed)
	tx1 := &types.Transaction{Type: types.TxTypeNormal, Payload: []byte("r1")}
	_ = mp.SubmitTransaction(tx1)
	mp.createVertex()

	// Now round should be 2, and roundCerts[1] should have an entry
	mp.mu.RLock()
	round := mp.currentRound
	prevCerts := mp.roundCerts[round-1]
	mp.mu.RUnlock()

	if round != 2 {
		t.Fatalf("expected round 2 after first vertex, got %d", round)
	}
	if len(prevCerts) == 0 {
		t.Fatal("expected certs in round 1")
	}

	// Submit another tx, createVertex should succeed with parents
	tx2 := &types.Transaction{Type: types.TxTypeNormal, Payload: []byte("r2")}
	_ = mp.SubmitTransaction(tx2)
	mp.createVertex()

	mp.mu.RLock()
	vCount := len(mp.vertices)
	mp.mu.RUnlock()
	if vCount != 2 {
		t.Fatalf("expected 2 vertices (round 1 and 2), got %d", vCount)
	}
}

// --- 13. popBatchLocked: first tx exceeds byte limit (still taken) ---

func TestPopBatchLockedFirstTxExceedsBytesStillTaken(t *testing.T) {
	mp := buildTestMempool(nil)
	// Submit one large tx
	bigPayload := make([]byte, 1000)
	bigTx := &types.Transaction{Type: types.TxTypeNormal, Payload: bigPayload}
	_ = mp.SubmitTransaction(bigTx)

	// Set maxBatchBytes smaller than the tx itself
	mp.mu.Lock()
	mp.maxBatchBytes = 10 // tiny limit
	batch := mp.popBatchLocked()
	mp.mu.Unlock()

	// The first tx should still be taken even if it exceeds byte limit
	// (the break condition checks len(batch) > 0 first)
	if len(batch) != 1 {
		t.Fatalf("expected first tx taken even if exceeds byte limit, got %d", len(batch))
	}
}

// --- 14. SubmitTransaction publish error logging (just coverage) ---

type failPublishNetwork struct {
	published int
}

func (f *failPublishNetwork) PublishTopic(topicName string, data []byte) error {
	f.published++
	return fmt.Errorf("publish failed")
}

func (f *failPublishNetwork) SubscribeTopic(topicName string, buf int) (chan []byte, error) {
	return make(chan []byte, buf), nil
}

func TestSubmitTransactionPublishError(t *testing.T) {
	net := &failPublishNetwork{}
	mp := buildTestMempool(net)
	tx := &types.Transaction{Type: types.TxTypeNormal, Payload: []byte("pub-fail")}
	// Should not return error even if publish fails (publish error is just logged)
	err := mp.SubmitTransaction(tx)
	if err != nil {
		t.Fatalf("SubmitTransaction should succeed even if publish fails: %v", err)
	}
	if net.published != 1 {
		t.Fatalf("expected 1 publish attempt, got %d", net.published)
	}
}
