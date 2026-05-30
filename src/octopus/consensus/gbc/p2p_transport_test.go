package gbc

import (
	"encoding/json"
	"sync"
	"testing"
	"time"
)

// mockP2PNetwork is an in-memory implementation of P2PNetwork for testing.
type mockP2PNetwork struct {
	mu          sync.Mutex
	subscribers map[string][]chan []byte
	directMsgs  map[uint64]chan []byte
}

func newMockP2PNetwork(peerIDs []uint64) *mockP2PNetwork {
	direct := make(map[uint64]chan []byte, len(peerIDs))
	for _, id := range peerIDs {
		direct[id] = make(chan []byte, 256)
	}
	return &mockP2PNetwork{
		subscribers: make(map[string][]chan []byte),
		directMsgs:  direct,
	}
}

func (m *mockP2PNetwork) PublishToTopic(topic string, data []byte) error {
	m.mu.Lock()
	subs := m.subscribers[topic]
	m.mu.Unlock()
	for _, ch := range subs {
		select {
		case ch <- append([]byte(nil), data...):
		default:
		}
	}
	return nil
}

func (m *mockP2PNetwork) SubscribeToTopic(topic string, bufSize int) (<-chan []byte, error) {
	ch := make(chan []byte, bufSize)
	m.mu.Lock()
	m.subscribers[topic] = append(m.subscribers[topic], ch)
	m.mu.Unlock()
	return ch, nil
}

func (m *mockP2PNetwork) SendDirect(peerID uint64, data []byte) error {
	m.mu.Lock()
	ch, ok := m.directMsgs[peerID]
	m.mu.Unlock()
	if !ok {
		return nil // swallow for unknown peers
	}
	select {
	case ch <- append([]byte(nil), data...):
	default:
	}
	return nil
}

func TestP2PTransport_BroadcastAndReceive(t *testing.T) {
	peers := []uint64{0, 1, 2}
	net := newMockP2PNetwork(peers)

	// Create 3 transports
	transports := make([]*P2PTransport, 3)
	for i := uint64(0); i < 3; i++ {
		tr, err := NewP2PTransport(P2PTransportConfig{
			NodeID:      i,
			PeerIDs:     peers,
			Network:     net,
			Topic:       "test-gbc",
			InboxBuffer: 64,
		})
		if err != nil {
			t.Fatalf("create transport %d: %v", i, err)
		}
		defer tr.Close()
		transports[i] = tr
	}

	// Node 0 broadcasts a proposal
	entry := Entry{Height: 1, Type: EntryQC, Payload: []byte("test-payload")}
	err := transports[0].Broadcast(Message{
		Type:     MsgPropose,
		SenderID: 0,
		Entry:    entry,
	})
	if err != nil {
		t.Fatalf("broadcast: %v", err)
	}

	// Node 1 and 2 should receive it
	for _, nodeID := range []int{1, 2} {
		select {
		case msg := <-transports[nodeID].inbox:
			if msg.Type != MsgPropose {
				t.Fatalf("node %d: expected MsgPropose, got %d", nodeID, msg.Type)
			}
			if msg.SenderID != 0 {
				t.Fatalf("node %d: expected sender 0, got %d", nodeID, msg.SenderID)
			}
			if msg.Entry.Height != 1 {
				t.Fatalf("node %d: expected height 1, got %d", nodeID, msg.Entry.Height)
			}
		case <-time.After(time.Second):
			t.Fatalf("node %d: timeout waiting for broadcast", nodeID)
		}
	}

	// Node 0 should NOT receive its own broadcast
	select {
	case <-transports[0].inbox:
		t.Fatal("node 0 received its own broadcast")
	case <-time.After(100 * time.Millisecond):
		// Good - no self-delivery
	}
}

func TestP2PTransport_DirectSend(t *testing.T) {
	peers := []uint64{1, 2, 3}
	net := newMockP2PNetwork(peers)

	transports := make(map[uint64]*P2PTransport)
	for _, id := range peers {
		tr, err := NewP2PTransport(P2PTransportConfig{
			NodeID: id, PeerIDs: peers, Network: net, Topic: "test-gbc-direct", InboxBuffer: 64,
		})
		if err != nil {
			t.Fatalf("create transport %d: %v", id, err)
		}
		defer tr.Close()
		transports[id] = tr
	}

	// Node 2 sends attestation directly to node 1 (via topic with TargetID)
	att := Attestation{SignerID: 2, Signature: []byte("sig-2")}

	wire := WireMessage{
		Type:        MsgAttest,
		SenderID:    2,
		TargetID:    1,
		Attestation: &att,
		Entry:       &WireEntry{Height: 5},
	}
	data, _ := json.Marshal(wire)

	// Deliver through topic
	net.PublishToTopic("test-gbc-direct", data)

	// Node 1 should receive it
	select {
	case msg := <-transports[1].inbox:
		if msg.Type != MsgAttest {
			t.Fatalf("expected MsgAttest, got %d", msg.Type)
		}
		if msg.Attestation.SignerID != 2 {
			t.Fatalf("expected signer 2, got %d", msg.Attestation.SignerID)
		}
	case <-time.After(time.Second):
		t.Fatal("timeout waiting for directed message")
	}

	// Node 3 should NOT receive it (targeted at node 1)
	select {
	case <-transports[3].inbox:
		t.Fatal("node 3 received message targeted at node 1")
	case <-time.After(100 * time.Millisecond):
		// Good
	}
}

func TestP2PTransport_Close(t *testing.T) {
	net := newMockP2PNetwork([]uint64{0, 1})
	tr, err := NewP2PTransport(P2PTransportConfig{
		NodeID: 0, PeerIDs: []uint64{0, 1}, Network: net, Topic: "test-close", InboxBuffer: 64,
	})
	if err != nil {
		t.Fatal(err)
	}

	tr.Close()

	// Operations after close should fail
	err = tr.Broadcast(Message{Type: MsgPropose})
	if err == nil {
		t.Fatal("expected error after close")
	}

	_, err = tr.Receive()
	if err == nil {
		t.Fatal("expected error on receive after close")
	}
}

func TestP2PTransport_PeerManagement(t *testing.T) {
	net := newMockP2PNetwork([]uint64{0, 1, 2})
	tr, err := NewP2PTransport(P2PTransportConfig{
		NodeID: 0, PeerIDs: []uint64{0, 1}, Network: net, Topic: "test-peers", InboxBuffer: 64,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer tr.Close()

	if tr.PeerCount() != 1 {
		t.Fatalf("expected 1 peer, got %d", tr.PeerCount())
	}

	tr.AddPeer(2)
	if tr.PeerCount() != 2 {
		t.Fatalf("expected 2 peers after add, got %d", tr.PeerCount())
	}

	tr.RemovePeer(1)
	if tr.PeerCount() != 1 {
		t.Fatalf("expected 1 peer after remove, got %d", tr.PeerCount())
	}

	// Adding self should be no-op
	tr.AddPeer(0)
	if tr.PeerCount() != 1 {
		t.Fatalf("expected 1 peer after self-add, got %d", tr.PeerCount())
	}
}
