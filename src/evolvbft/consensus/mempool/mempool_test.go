package mempool

import (
	"encoding/json"
	"testing"
	"time"

	octcrypto "evolvbft/evolvbft/crypto"
	"evolvbft/evolvbft/types"
)

type fakeNetwork struct {
	subChan      chan []byte
	published    [][]byte
	subscribeErr error
}

func (f *fakeNetwork) PublishTopic(topicName string, data []byte) error {
	f.published = append(f.published, append([]byte(nil), data...))
	return nil
}

func (f *fakeNetwork) SubscribeTopic(topicName string, buf int) (chan []byte, error) {
	if f.subChan == nil {
		f.subChan = make(chan []byte, buf)
	}
	return f.subChan, f.subscribeErr
}

func buildTestMempool(net MempoolNetwork) *Mempool {
	kp, _ := octcrypto.GenerateKeyPair()
	validators := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: []byte("v0"), Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: []byte("v1"), Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: []byte("v2"), Power: 1, IsActive: true},
		3: {ID: 3, PublicKey: []byte("v3"), Power: 1, IsActive: true},
	}
	return NewMempool(
		0,
		&types.Keypair{PublicKey: kp.PublicKey, PrivateKey: kp.PrivateKey},
		types.NewValidatorSet(1, validators),
		net,
	)
}

func TestMempoolDedupAndBatch(t *testing.T) {
	net := &fakeNetwork{}
	mp := buildTestMempool(net)
	tx := &types.Transaction{Type: types.TxTypeNormal, Payload: []byte("tx-a")}
	if err := mp.SubmitTransaction(tx); err != nil {
		t.Fatalf("submit first tx failed: %v", err)
	}
	if err := mp.SubmitTransaction(tx); err != nil {
		t.Fatalf("submit duplicate tx failed: %v", err)
	}
	if len(mp.txQueue) != 1 {
		t.Fatalf("expected 1 tx after dedup, got %d", len(mp.txQueue))
	}
	mp.createVertex()
	if len(mp.vertices) != 1 {
		t.Fatalf("expected 1 vertex, got %d", len(mp.vertices))
	}
	var created *types.Vertex
	for _, v := range mp.vertices {
		created = v
	}
	if created == nil || len(created.Txs) != 1 {
		t.Fatalf("expected vertex with 1 tx")
	}
	select {
	case certs := <-mp.GetProposalChan():
		if len(certs) == 0 {
			t.Fatalf("expected certs in proposal")
		}
	case <-time.After(300 * time.Millisecond):
		t.Fatalf("expected proposal emitted")
	}
	if len(net.published) != 1 {
		t.Fatalf("expected 1 published tx, got %d", len(net.published))
	}
}

func TestMempoolSizeLimitEvictsOldest(t *testing.T) {
	mp := buildTestMempool(nil)
	payload1 := make([]byte, 64)
	payload2 := make([]byte, 64)
	payload3 := make([]byte, 64)
	payload2[0] = 1
	payload3[0] = 2
	tx1 := &types.Transaction{Type: types.TxTypeNormal, Payload: payload1}
	raw, _ := json.Marshal(tx1)
	size := len(raw)
	mp.maxPoolBytes = size*2 + 8
	mp.maxTxSize = size + 16
	tx2 := &types.Transaction{Type: types.TxTypeNormal, Payload: payload2}
	tx3 := &types.Transaction{Type: types.TxTypeNormal, Payload: payload3}
	if err := mp.SubmitTransaction(tx1); err != nil {
		t.Fatalf("submit tx1 failed: %v", err)
	}
	if err := mp.SubmitTransaction(tx2); err != nil {
		t.Fatalf("submit tx2 failed: %v", err)
	}
	if err := mp.SubmitTransaction(tx3); err != nil {
		t.Fatalf("submit tx3 failed: %v", err)
	}
	if len(mp.txQueue) != 2 {
		t.Fatalf("expected 2 txs after eviction, got %d", len(mp.txQueue))
	}
	if mp.poolBytes > mp.maxPoolBytes {
		t.Fatalf("pool bytes exceeded limit: %d > %d", mp.poolBytes, mp.maxPoolBytes)
	}
}

func TestMempoolTTLPrunesExpired(t *testing.T) {
	mp := buildTestMempool(nil)
	mp.txTTL = 20 * time.Millisecond
	tx := &types.Transaction{Type: types.TxTypeNormal, Payload: []byte("ttl")}
	if err := mp.SubmitTransaction(tx); err != nil {
		t.Fatalf("submit tx failed: %v", err)
	}
	time.Sleep(30 * time.Millisecond)
	mp.mu.Lock()
	mp.pruneExpiredLocked(time.Now())
	mp.mu.Unlock()
	if len(mp.txQueue) != 0 {
		t.Fatalf("expected tx queue empty after ttl prune, got %d", len(mp.txQueue))
	}
}

func TestMempoolRunLoopAcceptsNetworkTx(t *testing.T) {
	net := &fakeNetwork{subChan: make(chan []byte, 1)}
	mp := buildTestMempool(net)
	mp.mu.Lock()
	mp.isRunning = true
	mp.mu.Unlock()
	go mp.runLoop()
	tx := &types.Transaction{Type: types.TxTypeNormal, Payload: []byte("from-net")}
	raw, _ := json.Marshal(tx)
	net.subChan <- raw
	time.Sleep(50 * time.Millisecond)
	mp.mu.RLock()
	queued := len(mp.txQueue)
	mp.mu.RUnlock()
	if queued == 0 {
		t.Fatalf("expected network tx queued")
	}
	mp.mu.Lock()
	mp.isRunning = false
	mp.mu.Unlock()
}

func TestMempoolQueueLimitDropsOldest(t *testing.T) {
	mp := buildTestMempool(nil)
	mp.maxQueueLen = 2
	tx1 := &types.Transaction{Type: types.TxTypeNormal, Payload: []byte("tx-1")}
	tx2 := &types.Transaction{Type: types.TxTypeNormal, Payload: []byte("tx-2")}
	tx3 := &types.Transaction{Type: types.TxTypeNormal, Payload: []byte("tx-3")}
	if err := mp.SubmitTransaction(tx1); err != nil {
		t.Fatalf("submit tx1 failed: %v", err)
	}
	if err := mp.SubmitTransaction(tx2); err != nil {
		t.Fatalf("submit tx2 failed: %v", err)
	}
	if err := mp.SubmitTransaction(tx3); err != nil {
		t.Fatalf("submit tx3 failed: %v", err)
	}
	if len(mp.txQueue) != 2 {
		t.Fatalf("expected queue len 2, got %d", len(mp.txQueue))
	}
	if string(mp.txQueue[0].tx.Payload) != "tx-2" || string(mp.txQueue[1].tx.Payload) != "tx-3" {
		t.Fatalf("expected oldest dropped, got payloads %q, %q", mp.txQueue[0].tx.Payload, mp.txQueue[1].tx.Payload)
	}
}

func TestMempoolRuntimeTuningSetters(t *testing.T) {
	mp := buildTestMempool(nil)
	mp.SetAdaptiveTuning(AdaptiveTuning{
		MaxBatchTxs:      512,
		ProposalInterval: 75 * time.Millisecond,
	})
	got := mp.GetAdaptiveTuning()
	if got.MaxBatchTxs != 512 {
		t.Fatalf("unexpected batch tuning: %d", got.MaxBatchTxs)
	}
	if got.ProposalInterval != 75*time.Millisecond {
		t.Fatalf("unexpected proposal interval: %v", got.ProposalInterval)
	}
}
