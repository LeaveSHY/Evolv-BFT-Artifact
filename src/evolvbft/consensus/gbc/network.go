package gbc

import "fmt"

// ═══════════════════════════════════════════════════════════════════════════════
// GBC Network Transport Layer
//
// Provides the transport abstraction for distributing GBC entries among
// m instance primaries. The protocol runs three message types:
//   Propose   - proposer broadcasts a new entry to all primaries
//   Attest    - each primary signs and returns an attestation
//   Committed - proposer broadcasts the entry with quorum attestations
//
// This corresponds to Section III-C of the paper: "The GBC maintains a
// replicated, append-only log among the m instance primaries."
// ═══════════════════════════════════════════════════════════════════════════════

// MsgType identifies the GBC network message class.
type MsgType uint8

const (
	MsgPropose   MsgType = 1 // proposer → all: new entry without quorum
	MsgAttest    MsgType = 2 // follower → proposer: attestation for entry
	MsgCommitted MsgType = 3 // proposer → all: entry with quorum attestations
	MsgSyncReq   MsgType = 4 // lagging node → any: request entries after height
	MsgSyncResp  MsgType = 5 // any → requester: batch of committed entries
)

// Message is the wire format for GBC inter-primary communication.
type Message struct {
	Type        MsgType
	SenderID    uint64
	Entry       Entry       // for Propose/Committed
	Attestation Attestation // for Attest
	SyncFrom    uint64      // for SyncReq: entries from this height onward
	SyncEntries []Entry     // for SyncResp: batch of entries
}

// Transport is the interface for sending and receiving GBC messages
// among the m instance primaries.
//
// Implementations:
//   - ChannelTransport (in-memory, for testing)
//   - future: TCP/gRPC transport for deployment
type Transport interface {
	// Send delivers a message to a specific primary.
	Send(to uint64, msg Message) error

	// Broadcast sends a message to all primaries except the sender.
	Broadcast(msg Message) error

	// Receive returns the next inbound message for this node.
	// Blocks until a message is available or the transport is closed.
	Receive() (Message, error)

	// Close shuts down the transport.
	Close() error
}

// ChannelTransport is an in-memory transport using Go channels.
// Used for testing and simulation without real networking.
type ChannelTransport struct {
	nodeID uint64
	peers  map[uint64]chan Message
	inbox  chan Message
	closed chan struct{}
}

// NewChannelTransportSet creates a connected set of in-memory transports
// for numNodes GBC primaries (IDs 0..numNodes-1).
func NewChannelTransportSet(numNodes int, bufferSize int) []*ChannelTransport {
	if bufferSize <= 0 {
		bufferSize = 256
	}

	// Create shared inbox channels
	inboxes := make(map[uint64]chan Message, numNodes)
	for i := 0; i < numNodes; i++ {
		inboxes[uint64(i)] = make(chan Message, bufferSize)
	}

	transports := make([]*ChannelTransport, numNodes)
	for i := 0; i < numNodes; i++ {
		transports[i] = &ChannelTransport{
			nodeID: uint64(i),
			peers:  inboxes,
			inbox:  inboxes[uint64(i)],
			closed: make(chan struct{}),
		}
	}
	return transports
}

func (t *ChannelTransport) Send(to uint64, msg Message) error {
	select {
	case <-t.closed:
		return fmt.Errorf("gbc: transport closed")
	default:
	}

	ch, ok := t.peers[to]
	if !ok {
		return fmt.Errorf("gbc: unknown peer %d", to)
	}

	select {
	case ch <- msg:
		return nil
	case <-t.closed:
		return fmt.Errorf("gbc: transport closed")
	}
}

func (t *ChannelTransport) Broadcast(msg Message) error {
	select {
	case <-t.closed:
		return fmt.Errorf("gbc: transport closed")
	default:
	}

	for id, ch := range t.peers {
		if id == t.nodeID {
			continue
		}
		select {
		case ch <- msg:
		case <-t.closed:
			return fmt.Errorf("gbc: transport closed during broadcast")
		}
	}
	return nil
}

func (t *ChannelTransport) Receive() (Message, error) {
	select {
	case msg := <-t.inbox:
		return msg, nil
	case <-t.closed:
		return Message{}, fmt.Errorf("gbc: transport closed")
	}
}

func (t *ChannelTransport) Close() error {
	select {
	case <-t.closed:
		return nil // already closed
	default:
		close(t.closed)
		return nil
	}
}
