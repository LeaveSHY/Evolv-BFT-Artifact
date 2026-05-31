package gbc

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// P2PTransport implements the Transport interface using a generic pub/sub network.
// It bridges the GBC protocol to a real network layer (libp2p, TCP, gRPC, etc.)
// for production multi-machine deployments.
//
// Architecture:
//   - Uses a dedicated GBC topic for broadcast (Propose, Committed, SyncResp)
//   - Uses point-to-point messaging for directed sends (Attest, SyncReq)
//   - Messages are JSON-serialized with type-length-value framing
//
// This replaces ChannelTransport for production deployments with m primaries
// running on separate physical machines.
type P2PTransport struct {
	mu      sync.Mutex
	nodeID  uint64
	peers   map[uint64]struct{} // known GBC peer IDs
	network P2PNetwork          // underlying network abstraction
	topic   string              // pub/sub topic for GBC broadcasts
	inbox   chan Message        // buffered incoming messages
	closed  chan struct{}
	ctx     context.Context
	cancel  context.CancelFunc
}

// P2PNetwork is the minimal interface required from the underlying network layer.
// Both libp2p and gRPC implementations can satisfy this interface.
type P2PNetwork interface {
	// PublishToTopic broadcasts data on a named topic.
	PublishToTopic(topic string, data []byte) error

	// SubscribeToTopic subscribes to a named topic and returns a channel of messages.
	SubscribeToTopic(topic string, bufSize int) (<-chan []byte, error)

	// SendDirect sends data directly to a specific peer by ID.
	SendDirect(peerID uint64, data []byte) error
}

// P2PTransportConfig configures the P2P transport.
type P2PTransportConfig struct {
	NodeID      uint64
	PeerIDs     []uint64 // IDs of all GBC primaries (including self)
	Network     P2PNetwork
	Topic       string // GBC topic name (default: "gbc-protocol")
	InboxBuffer int    // incoming message buffer size (default: 1024)
}

// WireMessage is the JSON-serialized envelope for GBC messages on the wire.
type WireMessage struct {
	Type        MsgType      `json:"type"`
	SenderID    uint64       `json:"sender_id"`
	TargetID    uint64       `json:"target_id,omitempty"` // 0 = broadcast
	Entry       *WireEntry   `json:"entry,omitempty"`
	Attestation *Attestation `json:"attestation,omitempty"`
	SyncFrom    uint64       `json:"sync_from,omitempty"`
	SyncEntries []WireEntry  `json:"sync_entries,omitempty"`
}

// WireEntry is the JSON-serializable form of an Entry.
type WireEntry struct {
	Height       uint64        `json:"height"`
	Type         EntryType     `json:"type"`
	Payload      []byte        `json:"payload"`
	Attestations []Attestation `json:"attestations,omitempty"`
}

func entryToWire(e Entry) *WireEntry {
	return &WireEntry{
		Height:       e.Height,
		Type:         e.Type,
		Payload:      e.Payload,
		Attestations: e.Attestations,
	}
}

func wireToEntry(w *WireEntry) Entry {
	if w == nil {
		return Entry{}
	}
	return Entry{
		Height:       w.Height,
		Type:         w.Type,
		Payload:      w.Payload,
		Attestations: w.Attestations,
	}
}

func wireEntriesToEntries(wires []WireEntry) []Entry {
	if len(wires) == 0 {
		return nil
	}
	entries := make([]Entry, len(wires))
	for i, w := range wires {
		entries[i] = Entry{Height: w.Height, Type: w.Type, Payload: w.Payload, Attestations: w.Attestations}
	}
	return entries
}

func entriesToWireEntries(entries []Entry) []WireEntry {
	if len(entries) == 0 {
		return nil
	}
	wires := make([]WireEntry, len(entries))
	for i, e := range entries {
		wires[i] = WireEntry{Height: e.Height, Type: e.Type, Payload: e.Payload, Attestations: e.Attestations}
	}
	return wires
}

// NewP2PTransport creates a production-ready GBC transport over a P2P network.
func NewP2PTransport(cfg P2PTransportConfig) (*P2PTransport, error) {
	if cfg.Network == nil {
		return nil, fmt.Errorf("gbc: P2PNetwork is required")
	}
	if cfg.Topic == "" {
		cfg.Topic = "gbc-protocol"
	}
	if cfg.InboxBuffer <= 0 {
		cfg.InboxBuffer = 1024
	}

	ctx, cancel := context.WithCancel(context.Background())

	peers := make(map[uint64]struct{}, len(cfg.PeerIDs))
	for _, id := range cfg.PeerIDs {
		if id != cfg.NodeID {
			peers[id] = struct{}{}
		}
	}

	t := &P2PTransport{
		nodeID:  cfg.NodeID,
		peers:   peers,
		network: cfg.Network,
		topic:   cfg.Topic,
		inbox:   make(chan Message, cfg.InboxBuffer),
		closed:  make(chan struct{}),
		ctx:     ctx,
		cancel:  cancel,
	}

	// Subscribe to broadcast topic
	sub, err := cfg.Network.SubscribeToTopic(cfg.Topic, cfg.InboxBuffer)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("gbc: subscribe to topic %q: %w", cfg.Topic, err)
	}

	// Start message receiving goroutine
	go t.receiveLoop(sub)

	return t, nil
}

func (t *P2PTransport) receiveLoop(sub <-chan []byte) {
	for {
		select {
		case <-t.closed:
			return
		case data, ok := <-sub:
			if !ok {
				return
			}
			var wire WireMessage
			if err := json.Unmarshal(data, &wire); err != nil {
				continue // skip malformed messages
			}
			// Skip our own broadcasts
			if wire.SenderID == t.nodeID {
				continue
			}
			// Skip messages targeted at other nodes
			if wire.TargetID != 0 && wire.TargetID != t.nodeID {
				continue
			}

			msg := t.wireToMessage(wire)
			select {
			case t.inbox <- msg:
			case <-t.closed:
				return
			default:
				// Drop if inbox full (backpressure)
			}
		}
	}
}

func (t *P2PTransport) wireToMessage(wire WireMessage) Message {
	msg := Message{
		Type:     wire.Type,
		SenderID: wire.SenderID,
		SyncFrom: wire.SyncFrom,
	}
	if wire.Entry != nil {
		msg.Entry = wireToEntry(wire.Entry)
	}
	if wire.Attestation != nil {
		msg.Attestation = *wire.Attestation
	}
	if len(wire.SyncEntries) > 0 {
		msg.SyncEntries = wireEntriesToEntries(wire.SyncEntries)
	}
	return msg
}

// Send delivers a message to a specific GBC primary.
func (t *P2PTransport) Send(to uint64, msg Message) error {
	select {
	case <-t.closed:
		return fmt.Errorf("gbc: transport closed")
	default:
	}

	wire := WireMessage{
		Type:     msg.Type,
		SenderID: t.nodeID,
		TargetID: to,
	}
	t.fillWire(&wire, msg)

	data, err := json.Marshal(wire)
	if err != nil {
		return fmt.Errorf("gbc: marshal message: %w", err)
	}

	// For directed messages, use SendDirect if available
	if err := t.network.SendDirect(to, data); err != nil {
		// Fallback: publish on topic (recipient filters by TargetID)
		return t.network.PublishToTopic(t.topic, data)
	}
	return nil
}

// Broadcast sends a message to all GBC primaries except self.
func (t *P2PTransport) Broadcast(msg Message) error {
	select {
	case <-t.closed:
		return fmt.Errorf("gbc: transport closed")
	default:
	}

	wire := WireMessage{
		Type:     msg.Type,
		SenderID: t.nodeID,
		TargetID: 0, // broadcast
	}
	t.fillWire(&wire, msg)

	data, err := json.Marshal(wire)
	if err != nil {
		return fmt.Errorf("gbc: marshal broadcast: %w", err)
	}

	return t.network.PublishToTopic(t.topic, data)
}

// Receive returns the next inbound message. Blocks until available or closed.
func (t *P2PTransport) Receive() (Message, error) {
	select {
	case msg := <-t.inbox:
		return msg, nil
	case <-t.closed:
		return Message{}, fmt.Errorf("gbc: transport closed")
	}
}

// ReceiveTimeout returns the next message or times out.
func (t *P2PTransport) ReceiveTimeout(timeout time.Duration) (Message, error) {
	select {
	case msg := <-t.inbox:
		return msg, nil
	case <-time.After(timeout):
		return Message{}, fmt.Errorf("gbc: receive timeout after %s", timeout)
	case <-t.closed:
		return Message{}, fmt.Errorf("gbc: transport closed")
	}
}

// Close shuts down the transport.
func (t *P2PTransport) Close() error {
	select {
	case <-t.closed:
		return nil
	default:
		close(t.closed)
		t.cancel()
		return nil
	}
}

// PeerCount returns the number of known GBC peers.
func (t *P2PTransport) PeerCount() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return len(t.peers)
}

// AddPeer registers a new GBC primary peer.
func (t *P2PTransport) AddPeer(peerID uint64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if peerID != t.nodeID {
		t.peers[peerID] = struct{}{}
	}
}

// RemovePeer unregisters a GBC primary peer.
func (t *P2PTransport) RemovePeer(peerID uint64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	delete(t.peers, peerID)
}

func (t *P2PTransport) fillWire(wire *WireMessage, msg Message) {
	switch msg.Type {
	case MsgPropose, MsgCommitted:
		wire.Entry = entryToWire(msg.Entry)
	case MsgAttest:
		att := msg.Attestation
		wire.Attestation = &att
		wire.Entry = &WireEntry{Height: msg.Entry.Height}
	case MsgSyncReq:
		wire.SyncFrom = msg.SyncFrom
	case MsgSyncResp:
		wire.SyncEntries = entriesToWireEntries(msg.SyncEntries)
	}
}
