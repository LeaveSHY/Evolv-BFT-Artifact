// Copyright 2024 Octopus Project
// Licensed under Apache License 2.0

package libp2p

import (
	"fmt"
	"io"
	"log"
	"strings"
	"time"

	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/protocol"
)

const (
	ProtocolID         = "/octopus/1.0.0"
	ProtocolVoteID     = "/octopus/vote/1.0.0"
	streamWriteTimeout = 5 * time.Second
	streamReadMaxBytes = 1 << 20 // 1MB max message
)

// handleStream handles incoming direct streams
func (n *P2PNetwork) handleStream(s network.Stream) {
	defer s.Close()

	// Read with size limit to prevent memory exhaustion
	reader := io.LimitReader(s, streamReadMaxBytes)
	buf, err := io.ReadAll(reader)
	if err != nil {
		log.Printf("Stream: read error from %s: %v", s.Conn().RemotePeer(), err)
		return
	}
	if len(buf) == 0 {
		return
	}

	n.totalBytesRecv.Add(uint64(len(buf)))

	// Collect per-instance channels under lock, then send without holding lock.
	n.mu.Lock()
	var targets []chan []byte
	for topic, ch := range n.chans {
		if isInstanceConsensusTopic(topic) {
			targets = append(targets, ch)
		}
	}
	n.mu.Unlock()

	delivered := false
	for _, ch := range targets {
		select {
		case ch <- buf:
			delivered = true
		default:
			n.incValidatorStat("stream_inbound_drop")
		}
	}
	if !delivered {
		select {
		case n.ConsensusChan <- buf:
		default:
			n.incValidatorStat("stream_inbound_drop")
		}
	}
}

// handleVoteStream handles incoming vote streams (unicast votes from validators)
func (n *P2PNetwork) handleVoteStream(s network.Stream) {
	defer s.Close()

	reader := io.LimitReader(s, streamReadMaxBytes)
	buf, err := io.ReadAll(reader)
	if err != nil {
		log.Printf("VoteStream: read error from %s: %v", s.Conn().RemotePeer(), err)
		return
	}
	if len(buf) == 0 {
		return
	}

	n.totalBytesRecv.Add(uint64(len(buf)))

	// Collect per-instance channels under lock, then send without holding lock.
	n.mu.Lock()
	var targets []chan []byte
	for topic, ch := range n.chans {
		if isInstanceConsensusTopic(topic) {
			targets = append(targets, ch)
		}
	}
	n.mu.Unlock()

	delivered := false
	for _, ch := range targets {
		select {
		case ch <- buf:
			delivered = true
		default:
			n.incValidatorStat("vote_inbound_drop")
		}
	}
	if !delivered {
		select {
		case n.ConsensusChan <- buf:
		default:
			n.incValidatorStat("vote_inbound_drop")
		}
	}
}

// SetupStreamHandler registers protocol handlers for direct streams
func (n *P2PNetwork) SetupStreamHandler() {
	n.Host.SetStreamHandler(protocol.ID(ProtocolID), n.handleStream)
	n.Host.SetStreamHandler(protocol.ID(ProtocolVoteID), n.handleVoteStream)
}

// Send sends a message to a specific peer by PeerID.
// If not connected, attempts DHT lookup and connection.
func (n *P2PNetwork) Send(peerID peer.ID, data []byte) error {
	// Check connectivity and reconnect if needed
	if n.Host.Network().Connectedness(peerID) != network.Connected {
		addrInfo, err := n.DHT.FindPeer(n.ctx, peerID)
		if err != nil {
			return fmt.Errorf("peer %s not found in DHT: %w", peerID, err)
		}

		if err := n.Host.Connect(n.ctx, addrInfo); err != nil {
			return fmt.Errorf("failed to connect to peer %s: %w", peerID, err)
		}
	}

	// Open stream
	s, err := n.Host.NewStream(n.ctx, peerID, protocol.ID(ProtocolID))
	if err != nil {
		return fmt.Errorf("failed to open stream to %s: %w", peerID, err)
	}
	defer s.Close()

	// Write with deadline
	s.SetWriteDeadline(time.Now().Add(streamWriteTimeout))
	_, err = s.Write(data)
	if err == nil {
		n.totalBytesSent.Add(uint64(len(data)))
	}
	return err
}

// SendVoteToLeader sends a vote message directly to the leader node via unicast stream.
// This implements Phase 7 vote delivery: votes go directly to the leader instead of broadcast.
func (n *P2PNetwork) SendVoteToLeader(leaderNodeID uint64, data []byte) error {
	return n.SendToNode(leaderNodeID, ProtocolVoteID, data)
}

// FindProviders finds peers providing a specific key (for DAS)
func (n *P2PNetwork) FindProviders(key string) ([]peer.AddrInfo, error) {
	// Placeholder for Phase 6 DAS integration
	return nil, nil
}

// Provide announces that this node provides a key
func (n *P2PNetwork) Provide(key string) error {
	// Placeholder for Phase 6 DAS integration
	return nil
}

// isInstanceConsensusTopic returns true if the topic name matches the
// per-instance consensus topic pattern (e.g. "octopus-consensus/instance/0").
func isInstanceConsensusTopic(topic string) bool {
	return strings.Contains(topic, "/instance/")
}
