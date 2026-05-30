// Copyright 2024 Octopus Project
// Licensed under Apache License 2.0

package network

import (
	"context"
	"fmt"
	"net"
	"sync"
	"time"

	"google.golang.org/grpc"

	pb "octopus-bft/octopus/network/proto"
	"octopus-bft/octopus/types"
)

var logger = struct {
	Info  func(format string, args ...interface{})
	Error func(format string, args ...interface{})
}{
	Info:  func(format string, args ...interface{}) { fmt.Printf(format+"\n", args...) },
	Error: func(format string, args ...interface{}) { fmt.Printf("ERROR: "+format+"\n", args...) },
}

// P2PNetwork represents the P2P network layer
type P2PNetwork struct {
	pb.UnimplementedConsensusServiceServer
	mu sync.RWMutex

	// Node identity
	nodeID  uint64
	keypair *types.Keypair

	// Network configuration
	listenAddress string
	peerAddresses map[uint64]string

	// Connections
	connections map[uint64]*PeerConnection
	listener    net.Listener
	grpcServer  *grpc.Server

	// Message handlers
	messageHandler MessageHandler

	// State
	isRunning bool

	// Channels
	receiveChan chan *types.Message
}

// PeerConnection represents a connection to a peer
type PeerConnection struct {
	nodeID   uint64
	address  string
	conn     *grpc.ClientConn
	client   pb.ConsensusServiceClient
	isActive bool
	lastSeen time.Time
}

// MessageHandler handles incoming messages
type MessageHandler interface {
	HandleMessage(msg *types.Message)
}

// NewP2PNetwork creates a new P2P network
func NewP2PNetwork(nodeID uint64, keypair *types.Keypair, listenAddress string, peerAddresses map[uint64]string) *P2PNetwork {
	return &P2PNetwork{
		nodeID:        nodeID,
		keypair:       keypair,
		listenAddress: listenAddress,
		peerAddresses: peerAddresses,
		connections:   make(map[uint64]*PeerConnection),
		receiveChan:   make(chan *types.Message, 1000),
	}
}

// Start starts the P2P network
func (p2p *P2PNetwork) Start() error {
	p2p.mu.Lock()
	if p2p.isRunning {
		p2p.mu.Unlock()
		return fmt.Errorf("network already running")
	}

	// Start listener
	lis, err := net.Listen("tcp", p2p.listenAddress)
	if err != nil {
		p2p.mu.Unlock()
		return fmt.Errorf("failed to listen: %v", err)
	}
	p2p.listener = lis

	// Create gRPC server
	p2p.grpcServer = grpc.NewServer()
	pb.RegisterConsensusServiceServer(p2p.grpcServer, p2p)

	p2p.isRunning = true
	p2p.mu.Unlock()

	// Connect to peers
	go p2p.connectToPeers()

	// Serve gRPC
	go func() {
		logger.Info("Starting P2P network for node %d at %s", p2p.nodeID, p2p.listenAddress)
		if err := p2p.grpcServer.Serve(lis); err != nil {
			logger.Error("Failed to serve: %v", err)
		}
	}()

	return nil
}

func (p2p *P2PNetwork) connectToPeers() {
	for id, addr := range p2p.peerAddresses {
		if id == p2p.nodeID {
			continue
		}
		go p2p.connectToPeer(id, addr)
	}
}

func (p2p *P2PNetwork) connectToPeer(id uint64, addr string) {
	conn, err := grpc.Dial(addr, grpc.WithInsecure(), grpc.WithBlock(), grpc.WithTimeout(5*time.Second))
	if err != nil {
		logger.Error("Failed to connect to peer %d at %s: %v", id, addr, err)
		return
	}

	client := pb.NewConsensusServiceClient(conn)

	p2p.mu.Lock()
	p2p.connections[id] = &PeerConnection{
		nodeID:   id,
		address:  addr,
		conn:     conn,
		client:   client,
		isActive: true,
		lastSeen: time.Now(),
	}
	p2p.mu.Unlock()

	logger.Info("Connected to peer %d at %s", id, addr)
}

// Stop stops the P2P network
func (p2p *P2PNetwork) Stop() error {
	p2p.mu.Lock()
	defer p2p.mu.Unlock()

	if !p2p.isRunning {
		return fmt.Errorf("network not running")
	}

	p2p.isRunning = false

	if p2p.grpcServer != nil {
		p2p.grpcServer.GracefulStop()
	}

	for _, conn := range p2p.connections {
		if conn.conn != nil {
			conn.conn.Close()
		}
	}

	logger.Info("P2P network stopped for node %d", p2p.nodeID)
	return nil
}

// Send sends a message to a specific peer
func (p2p *P2PNetwork) Send(to uint64, msg *types.Message) error {
	p2p.mu.RLock()
	conn, exists := p2p.connections[to]
	p2p.mu.RUnlock()

	if !exists || !conn.isActive {
		return fmt.Errorf("peer %d not connected", to)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	var err error
	switch msg.Type {
	case types.MsgProposal:
		data, encErr := types.EncodeMessage(msg)
		if encErr != nil {
			return fmt.Errorf("failed to serialize block: %v", encErr)
		}
		_, err = conn.client.ProposeBlock(ctx, &pb.BlockMsg{BlockData: data})
	case types.MsgVote:
		data, encErr := types.EncodeMessage(msg)
		if encErr != nil {
			return fmt.Errorf("failed to serialize vote: %v", encErr)
		}
		_, err = conn.client.SubmitVote(ctx, &pb.VoteMsg{VoteData: data})
	default:
		// For now only support Propose and Vote via gRPC
		logger.Info("Sending message type %s to node %d (simulated)", msg.Type, to)
		return nil
	}

	if err != nil {
		return fmt.Errorf("failed to send message: %v", err)
	}

	return nil
}

// Broadcast sends a message to all peers
func (p2p *P2PNetwork) Broadcast(msg *types.Message) error {
	p2p.mu.RLock()
	defer p2p.mu.RUnlock()

	for peerID := range p2p.peerAddresses {
		if peerID == p2p.nodeID {
			continue
		}
		// Async send to avoid blocking
		go func(pid uint64) {
			if err := p2p.Send(pid, msg); err != nil {
				logger.Error("Failed to broadcast to %d: %v", pid, err)
			}
		}(peerID)
	}

	return nil
}

// ReceiveChan returns the channel for receiving messages
func (p2p *P2PNetwork) ReceiveChan() chan *types.Message {
	return p2p.receiveChan
}

// SetMessageHandler sets the message handler
func (p2p *P2PNetwork) SetMessageHandler(handler MessageHandler) {
	p2p.mu.Lock()
	defer p2p.mu.Unlock()
	p2p.messageHandler = handler
}

// GetPeerCount returns the number of peers
func (p2p *P2PNetwork) GetPeerCount() int {
	p2p.mu.RLock()
	defer p2p.mu.RUnlock()
	return len(p2p.peerAddresses)
}

// gRPC Server Implementations

func (p2p *P2PNetwork) ProposeBlock(ctx context.Context, req *pb.BlockMsg) (*pb.Ack, error) {
	msg, err := types.DecodeMessage(req.BlockData)
	if err != nil {
		return &pb.Ack{Success: false, Error: err.Error()}, nil
	}
	p2p.receiveChan <- msg
	return &pb.Ack{Success: true}, nil
}

func (p2p *P2PNetwork) SubmitVote(ctx context.Context, req *pb.VoteMsg) (*pb.Ack, error) {
	msg, err := types.DecodeMessage(req.VoteData)
	if err != nil {
		return &pb.Ack{Success: false, Error: err.Error()}, nil
	}
	p2p.receiveChan <- msg
	return &pb.Ack{Success: true}, nil
}

func (p2p *P2PNetwork) NewView(ctx context.Context, req *pb.NewViewMsg) (*pb.Ack, error) {
	return &pb.Ack{Success: true}, nil
}
