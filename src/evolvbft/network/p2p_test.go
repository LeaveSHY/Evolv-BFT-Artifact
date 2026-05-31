package network

import (
	"testing"

	"evolvbft/evolvbft/types"
)

func TestNewP2PNetwork(t *testing.T) {
	peers := map[uint64]string{
		1: "localhost:9001",
		2: "localhost:9002",
		3: "localhost:9003",
	}
	p2p := NewP2PNetwork(1, nil, "localhost:9001", peers)
	if p2p == nil {
		t.Fatal("NewP2PNetwork should not return nil")
	}
	if p2p.nodeID != 1 {
		t.Errorf("expected nodeID 1, got %d", p2p.nodeID)
	}
}

func TestGetPeerCount(t *testing.T) {
	peers := map[uint64]string{
		1: "localhost:9001",
		2: "localhost:9002",
	}
	p2p := NewP2PNetwork(1, nil, "localhost:9001", peers)
	if c := p2p.GetPeerCount(); c != 2 {
		t.Errorf("expected 2 peers, got %d", c)
	}
}

func TestReceiveChan(t *testing.T) {
	p2p := NewP2PNetwork(1, nil, ":0", nil)
	ch := p2p.ReceiveChan()
	if ch == nil {
		t.Error("ReceiveChan should not be nil")
	}
	if cap(ch) != 1000 {
		t.Errorf("expected buffer 1000, got %d", cap(ch))
	}
}

type mockHandler struct {
	received []*types.Message
}

func (m *mockHandler) HandleMessage(msg *types.Message) {
	m.received = append(m.received, msg)
}

func TestSetMessageHandler(t *testing.T) {
	p2p := NewP2PNetwork(1, nil, ":0", nil)
	h := &mockHandler{}
	p2p.SetMessageHandler(h)
	if p2p.messageHandler == nil {
		t.Error("handler should be set")
	}
}

func TestSend_NoPeer(t *testing.T) {
	p2p := NewP2PNetwork(1, nil, ":0", nil)
	msg := &types.Message{Type: types.MsgVote, SenderID: 1}
	err := p2p.Send(99, msg)
	if err == nil {
		t.Error("sending to non-existent peer should fail")
	}
}

func TestStop_NotRunning(t *testing.T) {
	p2p := NewP2PNetwork(1, nil, ":0", nil)
	err := p2p.Stop()
	if err == nil {
		t.Error("stopping a non-running network should return error")
	}
}

func TestStartStop(t *testing.T) {
	p2p := NewP2PNetwork(1, nil, "localhost:0", nil)
	if err := p2p.Start(); err != nil {
		t.Fatalf("Start failed: %v", err)
	}

	// Double start should fail
	if err := p2p.Start(); err == nil {
		t.Error("double start should fail")
	}

	if err := p2p.Stop(); err != nil {
		t.Fatalf("Stop failed: %v", err)
	}
}
