// Copyright 2024 Octopus Project
// Licensed under Apache License 2.0

package libp2p

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/libp2p/go-libp2p"
	dht "github.com/libp2p/go-libp2p-kad-dht"
	pubsub "github.com/libp2p/go-libp2p-pubsub"
	libp2pcrypto "github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/protocol"
	"github.com/libp2p/go-libp2p/p2p/discovery/mdns"
	"github.com/libp2p/go-libp2p/p2p/net/connmgr"

	"octopus-bft/octopus/types"
)

const (
	defaultPubSubMaxMessageSize = 1 << 20
	defaultReplayTTL            = 30 * time.Second
	defaultRateLimitWindow      = time.Second
	// At 1000 nodes a leader receives up to ~1000 vote messages per view.
	// Each node also gossips proposals/metadata. 1000/window prevents
	// false-positive rate limiting while still capping abuse.
	defaultRateLimitPerWindow = 1000

	// Connection manager limits for 1000-node scale.
	// LowWater=200: keep at least 200 connections alive.
	// HighWater=800: aggressively prune beyond 800 to control FD usage.
	// Grace=30s: newly-opened connections get 30s before being eligible for pruning.
	connMgrLowWater  = 200
	connMgrHighWater = 800
	connMgrGrace     = 30 * time.Second

	// Reconnection parameters
	reconnectBaseDelay = 1 * time.Second
	reconnectMaxDelay  = 30 * time.Second
	reconnectInterval  = 10 * time.Second
)

type TopicValidationPolicy struct {
	ExpectedInstance   *uint64
	EpochProvider      func() uint64
	KeyProvider        func(senderID uint64) (types.PublicKey, bool)
	MaxMessageBytes    int
	ReplayTTL          time.Duration
	RateLimitPerWindow int
	RateLimitWindow    time.Duration
}

type senderWindow struct {
	start time.Time
	count int
}

// ConnectionStats holds connection-level statistics
type ConnectionStats struct {
	NumConnections int `json:"num_connections"`
	NumStreams     int `json:"num_streams"`
	EstimatedFDs   int `json:"estimated_fds"`
	ConnectedPeers int `json:"connected_peers"`
	KnownPeers     int `json:"known_peers"`
	ConnMgrLow     int `json:"connmgr_low"`
	ConnMgrHigh    int `json:"connmgr_high"`
}

// ResourceMetrics holds resource usage metrics
type ResourceMetrics struct {
	GoroutineCount int     `json:"goroutine_count"`
	HeapAllocMB    float64 `json:"heap_alloc_mb"`
	HeapSysMB      float64 `json:"heap_sys_mb"`
	NumConnections int     `json:"num_connections"`
	NumStreams     int     `json:"num_streams"`
	NumTopics      int     `json:"num_topics"`
	NumPeers       int     `json:"num_peers"`
}

// NetworkStats holds network-level statistics
type NetworkStats struct {
	ConnectedPeers    int    `json:"connected_peers"`
	KnownPeers        int    `json:"known_peers"`
	TotalBytesSent    uint64 `json:"total_bytes_sent"`
	TotalBytesRecv    uint64 `json:"total_bytes_recv"`
	ReconnectAttempts uint64 `json:"reconnect_attempts"`
	ReconnectSuccess  uint64 `json:"reconnect_success"`
}

// PropagationMetrics tracks message propagation timing
type PropagationMetrics struct {
	mu       sync.Mutex
	samples  []float64
	maxStore int
}

// P2PNetwork represents the libp2p network layer
type P2PNetwork struct {
	ctx    context.Context
	cancel context.CancelFunc
	Host   host.Host
	DHT    *dht.IpfsDHT
	PubSub *pubsub.PubSub

	consensusTopic *pubsub.Topic
	mempoolTopic   *pubsub.Topic

	consensusSub *pubsub.Subscription
	mempoolSub   *pubsub.Subscription

	// Channels for upper layers
	ConsensusChan chan []byte
	MempoolChan   chan []byte

	topics map[string]*pubsub.Topic
	subs   map[string]*pubsub.Subscription
	chans  map[string]chan []byte

	topicPolicies       map[string]TopicValidationPolicy
	validatorsInstalled map[string]struct{}
	seenDigests         map[string]map[string]time.Time
	senderWindows       map[string]map[uint64]*senderWindow
	validatorStats      map[string]uint64

	// Phase 7: NodeID → peer.AddrInfo address book (replaces old map[uint64]string)
	addressBook map[uint64]peer.AddrInfo
	peerIDMap   map[peer.ID]uint64 // reverse: PeerID → NodeID

	// Phase 7: Reconnection & resource tracking
	reconnectAttempts atomic.Uint64
	reconnectSuccess  atomic.Uint64
	totalBytesSent    atomic.Uint64
	totalBytesRecv    atomic.Uint64
	propagation       *PropagationMetrics

	mu sync.Mutex
}

// gossipSubParams returns optimized GossipSub parameters for 1000-node scale.
//
// Rationale for 1000 nodes:
//   - D=8: Higher mesh degree for reliable propagation (log₂(1000)≈10).
//     With D=8, message propagation takes ~4 hops to reach all nodes.
//   - Dlo=5, Dhi=16: Wider range absorbs churn in large networks without
//     excessive GRAFT/PRUNE traffic.
//   - Dlazy=8: More lazy gossip peers accelerates metadata dissemination
//     so nodes discover missing messages faster via IHAVE/IWANT.
//   - HeartbeatInterval=500ms: Faster heartbeat detects mesh disruptions
//     sooner, critical when any of 1000 nodes may churn.
//   - HistoryLength=6, HistoryGossip=4: Deeper history helps catching up
//     across multiple heartbeat intervals.
//   - Dscore=5: More score-tracked peers improves Sybil resistance at scale.
func gossipSubParams() pubsub.GossipSubParams {
	params := pubsub.DefaultGossipSubParams()
	params.D = 8
	params.Dlo = 5
	params.Dhi = 16
	params.Dscore = 5
	params.Dlazy = 8
	params.HeartbeatInterval = 500 * time.Millisecond
	params.HistoryLength = 6
	params.HistoryGossip = 4
	return params
}

// NewP2PNetwork creates a new libp2p network with connection management and optimized GossipSub
func NewP2PNetwork(ctx context.Context, port int, privKey []byte) (*P2PNetwork, error) {
	listenAddr := fmt.Sprintf("/ip4/0.0.0.0/tcp/%d", port)

	childCtx, cancel := context.WithCancel(ctx)

	// Phase 7: Connection manager for FD control
	cmgr, err := connmgr.NewConnManager(
		connMgrLowWater,
		connMgrHighWater,
		connmgr.WithGracePeriod(connMgrGrace),
	)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("failed to create connection manager: %w", err)
	}

	opts := []libp2p.Option{
		libp2p.ListenAddrStrings(listenAddr),
		libp2p.EnableNATService(),
		libp2p.ConnectionManager(cmgr),
	}
	if len(privKey) > 0 {
		hostPrivKey, err := libp2pcrypto.UnmarshalEd25519PrivateKey(privKey)
		if err != nil {
			cancel()
			return nil, fmt.Errorf("failed to decode libp2p identity key: %w", err)
		}
		opts = append(opts, libp2p.Identity(hostPrivKey))
	}

	h, err := libp2p.New(opts...)
	if err != nil {
		cancel()
		return nil, err
	}

	log.Printf("P2P Node started. PeerID: %s, Addrs: %v", h.ID(), h.Addrs())

	// Setup DHT
	kademliaDHT, err := dht.New(childCtx, h)
	if err != nil {
		cancel()
		return nil, err
	}

	if err = kademliaDHT.Bootstrap(childCtx); err != nil {
		cancel()
		return nil, err
	}

	// Phase 7: GossipSub with optimized mesh degree D=8 for 1000-node scale
	ps, err := pubsub.NewGossipSub(childCtx, h,
		pubsub.WithGossipSubParams(gossipSubParams()),
		pubsub.WithMaxMessageSize(defaultPubSubMaxMessageSize),
	)
	if err != nil {
		cancel()
		return nil, err
	}

	n := &P2PNetwork{
		ctx:                 childCtx,
		cancel:              cancel,
		Host:                h,
		DHT:                 kademliaDHT,
		PubSub:              ps,
		ConsensusChan:       make(chan []byte, 8192),  // Larger buffer for 1000-node vote/proposal flood
		MempoolChan:         make(chan []byte, 16384), // Larger buffer for 1000-node tx dissemination
		topics:              make(map[string]*pubsub.Topic),
		subs:                make(map[string]*pubsub.Subscription),
		chans:               make(map[string]chan []byte),
		topicPolicies:       make(map[string]TopicValidationPolicy),
		validatorsInstalled: make(map[string]struct{}),
		seenDigests:         make(map[string]map[string]time.Time),
		senderWindows:       make(map[string]map[uint64]*senderWindow),
		validatorStats:      make(map[string]uint64),
		addressBook:         make(map[uint64]peer.AddrInfo),
		peerIDMap:           make(map[peer.ID]uint64),
		propagation:         newPropagationMetrics(10000),
	}

	if err := n.setupDiscovery(); err != nil {
		cancel()
		return nil, err
	}

	return n, nil
}

// setupDiscovery sets up mDNS discovery
func (n *P2PNetwork) setupDiscovery() error {
	s := mdns.NewMdnsService(n.Host, "octopus-discovery", n)
	return s.Start()
}

// HandlePeerFound is called when mDNS discovers a peer
func (n *P2PNetwork) HandlePeerFound(pi peer.AddrInfo) {
	if err := n.Host.Connect(n.ctx, pi); err != nil {
		log.Printf("mDNS: failed to connect to discovered peer %s: %v", pi.ID, err)
	}
}

func InstanceConsensusTopic(base string, instance uint64) string {
	root := strings.TrimSuffix(strings.TrimSpace(base), "/")
	if root == "" {
		root = "octopus-consensus"
	}
	return fmt.Sprintf("%s/instance/%d", root, instance)
}

func GlobalTopic(base string) string {
	root := strings.TrimSuffix(strings.TrimSpace(base), "/")
	if root == "" {
		root = "octopus-consensus"
	}
	return root + "/global"
}

func ReconfigTopic(base string) string {
	root := strings.TrimSuffix(strings.TrimSpace(base), "/")
	if root == "" {
		root = "octopus-consensus"
	}
	return root + "/reconfig"
}

func (n *P2PNetwork) RegisterTopicPolicy(topicName string, policy TopicValidationPolicy) error {
	normalized := policy
	if normalized.MaxMessageBytes <= 0 {
		normalized.MaxMessageBytes = defaultPubSubMaxMessageSize
	}
	if normalized.ReplayTTL <= 0 {
		normalized.ReplayTTL = defaultReplayTTL
	}
	if normalized.RateLimitPerWindow <= 0 {
		normalized.RateLimitPerWindow = defaultRateLimitPerWindow
	}
	if normalized.RateLimitWindow <= 0 {
		normalized.RateLimitWindow = defaultRateLimitWindow
	}
	n.mu.Lock()
	n.topicPolicies[topicName] = normalized
	n.mu.Unlock()
	return n.ensureTopicValidator(topicName)
}

func (n *P2PNetwork) GetPubSubValidatorStats() map[string]uint64 {
	n.mu.Lock()
	defer n.mu.Unlock()
	out := make(map[string]uint64, len(n.validatorStats))
	for k, v := range n.validatorStats {
		out[k] = v
	}
	return out
}

func (n *P2PNetwork) PublishGlobal(base string, data []byte) error {
	return n.PublishTopic(GlobalTopic(base), data)
}

func (n *P2PNetwork) PublishReconfig(base string, data []byte) error {
	return n.PublishTopic(ReconfigTopic(base), data)
}

func (n *P2PNetwork) SubscribeGlobal(base string, buf int) (chan []byte, error) {
	return n.SubscribeTopic(GlobalTopic(base), buf)
}

func (n *P2PNetwork) SubscribeReconfig(base string, buf int) (chan []byte, error) {
	return n.SubscribeTopic(ReconfigTopic(base), buf)
}

func (n *P2PNetwork) ensureTopicValidator(topicName string) error {
	n.mu.Lock()
	_, hasPolicy := n.topicPolicies[topicName]
	if !hasPolicy {
		n.mu.Unlock()
		return nil
	}
	if _, exists := n.validatorsInstalled[topicName]; exists {
		n.mu.Unlock()
		return nil
	}
	n.mu.Unlock()

	err := n.PubSub.RegisterTopicValidator(topicName, func(ctx context.Context, from peer.ID, msg *pubsub.Message) pubsub.ValidationResult {
		_ = ctx
		_ = from
		if n.validatePubSubMessage(topicName, msg.Data) {
			return pubsub.ValidationAccept
		}
		return pubsub.ValidationReject
	})
	if err != nil {
		return err
	}
	n.mu.Lock()
	n.validatorsInstalled[topicName] = struct{}{}
	n.mu.Unlock()
	return nil
}

func (n *P2PNetwork) validatePubSubMessage(topicName string, data []byte) bool {
	n.mu.Lock()
	policy, ok := n.topicPolicies[topicName]
	n.mu.Unlock()
	if !ok {
		return true
	}
	if policy.MaxMessageBytes > 0 && len(data) > policy.MaxMessageBytes {
		n.incValidatorStat("pubsub_reject_size")
		return false
	}
	msg, err := types.DecodeMessage(data)
	if err != nil {
		n.incValidatorStat("pubsub_reject_decode")
		return false
	}
	if msg == nil {
		n.incValidatorStat("pubsub_reject_nil")
		return false
	}
	expectedInstance, hasInstance := instanceFromTopic(topicName)
	if policy.ExpectedInstance != nil {
		expectedInstance = *policy.ExpectedInstance
		hasInstance = true
	}
	if hasInstance && msg.Instance != expectedInstance {
		n.incValidatorStat("pubsub_reject_instance")
		return false
	}
	if policy.EpochProvider != nil {
		currentEpoch := policy.EpochProvider()
		if msg.Epoch < currentEpoch || msg.Epoch > currentEpoch+1 {
			n.incValidatorStat("pubsub_reject_epoch")
			return false
		}
	}
	if policy.KeyProvider != nil {
		pubKey, exists := policy.KeyProvider(msg.SenderID)
		if !exists || !msg.VerifySignature(pubKey) {
			n.incValidatorStat("pubsub_reject_signature")
			return false
		}
	}
	now := time.Now()
	if n.isRateLimited(topicName, msg.SenderID, now, policy) {
		n.incValidatorStat("pubsub_rate_limited")
		return false
	}
	if n.isReplay(topicName, data, now, policy.ReplayTTL) {
		n.incValidatorStat("pubsub_replay")
		return false
	}
	return true
}

func (n *P2PNetwork) isRateLimited(topicName string, senderID uint64, now time.Time, policy TopicValidationPolicy) bool {
	n.mu.Lock()
	defer n.mu.Unlock()
	if n.senderWindows[topicName] == nil {
		n.senderWindows[topicName] = make(map[uint64]*senderWindow)
	}
	win, ok := n.senderWindows[topicName][senderID]
	if !ok {
		n.senderWindows[topicName][senderID] = &senderWindow{start: now, count: 1}
		return false
	}
	if now.Sub(win.start) >= policy.RateLimitWindow {
		win.start = now
		win.count = 1
		return false
	}
	win.count++
	return win.count > policy.RateLimitPerWindow
}

func (n *P2PNetwork) isReplay(topicName string, data []byte, now time.Time, ttl time.Duration) bool {
	sum := sha256.Sum256(data)
	digest := hex.EncodeToString(sum[:])
	n.mu.Lock()
	defer n.mu.Unlock()
	if n.seenDigests[topicName] == nil {
		n.seenDigests[topicName] = make(map[string]time.Time)
	}
	for k, ts := range n.seenDigests[topicName] {
		if now.Sub(ts) > ttl {
			delete(n.seenDigests[topicName], k)
		}
	}
	if _, exists := n.seenDigests[topicName][digest]; exists {
		return true
	}
	n.seenDigests[topicName][digest] = now
	return false
}

func (n *P2PNetwork) incValidatorStat(reason string) {
	n.mu.Lock()
	defer n.mu.Unlock()
	n.validatorStats[reason]++
}

func instanceFromTopic(topic string) (uint64, bool) {
	parts := strings.Split(strings.Trim(topic, "/"), "/")
	for i := 0; i < len(parts)-1; i++ {
		if parts[i] != "instance" {
			continue
		}
		var id uint64
		_, err := fmt.Sscanf(parts[i+1], "%d", &id)
		if err == nil {
			return id, true
		}
	}
	return 0, false
}

// JoinConsensusTopic joins the consensus gossip topic
func (n *P2PNetwork) JoinConsensusTopic() error {
	topicName := InstanceConsensusTopic("octopus-consensus", 0)
	ch, err := n.SubscribeTopic(topicName, 4096)
	if err != nil {
		return err
	}
	n.ConsensusChan = ch
	n.mu.Lock()
	n.consensusTopic = n.topics[topicName]
	n.consensusSub = n.subs[topicName]
	n.mu.Unlock()
	return nil
}

// JoinMempoolTopic joins the mempool gossip topic
func (n *P2PNetwork) JoinMempoolTopic() error {
	ch, err := n.SubscribeTopic("octopus-mempool", 8192)
	if err != nil {
		return err
	}
	n.MempoolChan = ch
	n.mu.Lock()
	n.mempoolTopic = n.topics["octopus-mempool"]
	n.mempoolSub = n.subs["octopus-mempool"]
	n.mu.Unlock()
	return nil
}

func (n *P2PNetwork) readSubscription(sub *pubsub.Subscription, ch chan []byte) {
	for {
		msg, err := sub.Next(n.ctx)
		if err != nil {
			return
		}
		if msg.ReceivedFrom == n.Host.ID() {
			continue
		}
		n.totalBytesRecv.Add(uint64(len(msg.Data)))
		select {
		case ch <- msg.Data:
		default:
			n.incValidatorStat("inbound_queue_drop")
		}
	}
}

func (n *P2PNetwork) PublishTopic(topicName string, data []byte) error {
	if err := n.ensureTopicValidator(topicName); err != nil {
		return err
	}
	n.mu.Lock()
	topic := n.topics[topicName]
	n.mu.Unlock()
	if topic == nil {
		var err error
		topic, err = n.PubSub.Join(topicName)
		if err != nil {
			return err
		}
		n.mu.Lock()
		n.topics[topicName] = topic
		n.mu.Unlock()
	}
	n.totalBytesSent.Add(uint64(len(data)))
	pubCtx, pubCancel := context.WithTimeout(n.ctx, 3*time.Second)
	err := topic.Publish(pubCtx, data)
	pubCancel()
	return err
}

func (n *P2PNetwork) SubscribeTopic(topicName string, buf int) (chan []byte, error) {
	if err := n.ensureTopicValidator(topicName); err != nil {
		return nil, err
	}
	n.mu.Lock()
	if ch, ok := n.chans[topicName]; ok {
		n.mu.Unlock()
		return ch, nil
	}
	n.mu.Unlock()

	topic, err := n.PubSub.Join(topicName)
	if err != nil {
		return nil, err
	}
	sub, err := topic.Subscribe()
	if err != nil {
		return nil, err
	}
	if buf <= 0 {
		buf = 256
	}
	ch := make(chan []byte, buf)

	n.mu.Lock()
	n.topics[topicName] = topic
	n.subs[topicName] = sub
	n.chans[topicName] = ch
	n.mu.Unlock()

	go n.readSubscription(sub, ch)
	return ch, nil
}

// BroadcastConsensus broadcasts a message to the consensus topic
func (n *P2PNetwork) BroadcastConsensus(data []byte) error {
	if n.consensusTopic == nil {
		return fmt.Errorf("consensus topic not joined")
	}
	n.totalBytesSent.Add(uint64(len(data)))
	return n.consensusTopic.Publish(n.ctx, data)
}

// SetStreamHandler sets the handler for direct streams
func (n *P2PNetwork) SetStreamHandler(protocolID string, handler func(network.Stream)) {
	n.Host.SetStreamHandler(protocol.ID(protocolID), handler)
}

// SendDirect sends a message directly to a peer via stream
func (n *P2PNetwork) SendDirect(peerID peer.ID, protocolID string, data []byte) error {
	s, err := n.Host.NewStream(n.ctx, peerID, protocol.ID(protocolID))
	if err != nil {
		return err
	}
	defer s.Close()

	_, err = s.Write(data)
	if err == nil {
		n.totalBytesSent.Add(uint64(len(data)))
	}
	return err
}

// --- Phase 7: Address Book with NodeID → PeerID → Multiaddr ---

// SetNodeAddressBook sets the peer address book using peer.AddrInfo entries.
// This replaces the old map[uint64]string address book with proper NodeID→PeerID→Multiaddr routing.
func (n *P2PNetwork) SetNodeAddressBook(peers map[uint64]peer.AddrInfo) {
	n.mu.Lock()
	defer n.mu.Unlock()
	n.addressBook = make(map[uint64]peer.AddrInfo, len(peers))
	n.peerIDMap = make(map[peer.ID]uint64, len(peers))
	for id, info := range peers {
		n.addressBook[id] = info
		n.peerIDMap[info.ID] = id
	}
}

// SetNodeAddressBookLegacy accepts old-style map[uint64]string for backward compatibility.
// The string values are treated as "host:port" and converted to multiaddrs.
func (n *P2PNetwork) SetNodeAddressBookLegacy(peers map[uint64]string) {
	n.mu.Lock()
	defer n.mu.Unlock()
	n.addressBook = make(map[uint64]peer.AddrInfo, len(peers))
	// Legacy entries cannot resolve PeerID, so they remain address-only placeholders
	for id, addr := range peers {
		// Store as empty AddrInfo — will need DHT to resolve PeerID
		_ = addr
		n.addressBook[id] = peer.AddrInfo{}
	}
}

// ResolvePeer looks up a peer's AddrInfo by NodeID
func (n *P2PNetwork) ResolvePeer(nodeID uint64) (peer.AddrInfo, bool) {
	n.mu.Lock()
	info, ok := n.addressBook[nodeID]
	n.mu.Unlock()
	if !ok {
		return peer.AddrInfo{}, false
	}
	if info.ID == "" {
		return peer.AddrInfo{}, false
	}
	return info, true
}

// ResolveNodeID looks up a NodeID from a PeerID
func (n *P2PNetwork) ResolveNodeID(peerID peer.ID) (uint64, bool) {
	n.mu.Lock()
	defer n.mu.Unlock()
	nodeID, ok := n.peerIDMap[peerID]
	return nodeID, ok
}

// RegisterPeer registers or updates a peer in the address book
func (n *P2PNetwork) RegisterPeer(nodeID uint64, info peer.AddrInfo) {
	n.mu.Lock()
	defer n.mu.Unlock()
	n.addressBook[nodeID] = info
	if info.ID != "" {
		n.peerIDMap[info.ID] = nodeID
	}
}

// SendToNode sends data directly to a node identified by NodeID via stream protocol.
// This implements the NodeID→PeerID→Multiaddr routing chain for unicast delivery.
func (n *P2PNetwork) SendToNode(nodeID uint64, protocolID string, data []byte) error {
	info, ok := n.ResolvePeer(nodeID)
	if !ok {
		return fmt.Errorf("unknown node %d: not in address book or missing PeerID", nodeID)
	}

	// Ensure we're connected
	if n.Host.Network().Connectedness(info.ID) != network.Connected {
		if err := n.Host.Connect(n.ctx, info); err != nil {
			// Fallback: try DHT peer lookup
			dhtInfo, dhtErr := n.DHT.FindPeer(n.ctx, info.ID)
			if dhtErr != nil {
				return fmt.Errorf("failed to connect to node %d (peer %s): %w", nodeID, info.ID, err)
			}
			if connErr := n.Host.Connect(n.ctx, dhtInfo); connErr != nil {
				return fmt.Errorf("failed to connect to node %d via DHT: %w", nodeID, connErr)
			}
		}
	}

	s, err := n.Host.NewStream(n.ctx, info.ID, protocol.ID(protocolID))
	if err != nil {
		return fmt.Errorf("failed to open stream to node %d: %w", nodeID, err)
	}
	defer s.Close()

	_, err = s.Write(data)
	if err == nil {
		n.totalBytesSent.Add(uint64(len(data)))
	}
	return err
}

// ConnectBootstrapPeers connects to all peers in the address book.
// This should be called after SetNodeAddressBook to establish initial connectivity.
func (n *P2PNetwork) ConnectBootstrapPeers() {
	n.mu.Lock()
	peers := make([]peer.AddrInfo, 0, len(n.addressBook))
	for _, info := range n.addressBook {
		if info.ID != "" && info.ID != n.Host.ID() {
			peers = append(peers, info)
		}
	}
	n.mu.Unlock()

	for _, pi := range peers {
		go func(info peer.AddrInfo) {
			if err := n.Host.Connect(n.ctx, info); err != nil {
				log.Printf("Bootstrap: failed to connect to %s: %v", info.ID, err)
			}
		}(pi)
	}
}

// --- Phase 7: Connection Monitor with Reconnection ---

// StartConnectionMonitor starts a background goroutine that monitors peer connectivity
// and attempts reconnection with exponential backoff on disconnection.
func (n *P2PNetwork) StartConnectionMonitor() {
	go n.connectionMonitorLoop()
}

func (n *P2PNetwork) connectionMonitorLoop() {
	ticker := time.NewTicker(reconnectInterval)
	defer ticker.Stop()

	for {
		select {
		case <-n.ctx.Done():
			return
		case <-ticker.C:
			n.checkAndReconnect()
		}
	}
}

func (n *P2PNetwork) checkAndReconnect() {
	n.mu.Lock()
	peers := make(map[uint64]peer.AddrInfo, len(n.addressBook))
	for id, info := range n.addressBook {
		peers[id] = info
	}
	n.mu.Unlock()

	for nodeID, info := range peers {
		if info.ID == "" || info.ID == n.Host.ID() {
			continue
		}
		if n.Host.Network().Connectedness(info.ID) == network.Connected {
			continue
		}

		// Attempt reconnection with exponential backoff
		go n.reconnectWithBackoff(nodeID, info)
	}
}

func (n *P2PNetwork) reconnectWithBackoff(nodeID uint64, info peer.AddrInfo) {
	delay := reconnectBaseDelay
	for attempt := 0; attempt < 5; attempt++ {
		n.reconnectAttempts.Add(1)

		select {
		case <-n.ctx.Done():
			return
		default:
		}

		if err := n.Host.Connect(n.ctx, info); err != nil {
			log.Printf("Reconnect: attempt %d to node %d (peer %s) failed: %v", attempt+1, nodeID, info.ID.String()[:8], err)
			select {
			case <-n.ctx.Done():
				return
			case <-time.After(delay):
			}
			delay *= 2
			if delay > reconnectMaxDelay {
				delay = reconnectMaxDelay
			}
			continue
		}

		n.reconnectSuccess.Add(1)
		log.Printf("Reconnect: successfully reconnected to node %d (peer %s)", nodeID, info.ID.String()[:8])
		return
	}
}

// --- Phase 7: Resource Metrics ---

// GetConnectionStats returns connection-level statistics including FD estimates
func (n *P2PNetwork) GetConnectionStats() ConnectionStats {
	conns := n.Host.Network().Conns()
	numStreams := 0
	for _, c := range conns {
		numStreams += len(c.GetStreams())
	}

	n.mu.Lock()
	knownPeers := len(n.addressBook)
	n.mu.Unlock()

	return ConnectionStats{
		NumConnections: len(conns),
		NumStreams:     numStreams,
		EstimatedFDs:   len(conns) + numStreams + 10, // base FDs for listeners etc
		ConnectedPeers: len(n.Host.Network().Peers()),
		KnownPeers:     knownPeers,
		ConnMgrLow:     connMgrLowWater,
		ConnMgrHigh:    connMgrHighWater,
	}
}

// GetResourceMetrics returns resource usage metrics including goroutines, memory, and network stats
func (n *P2PNetwork) GetResourceMetrics() ResourceMetrics {
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	conns := n.Host.Network().Conns()
	numStreams := 0
	for _, c := range conns {
		numStreams += len(c.GetStreams())
	}

	n.mu.Lock()
	numTopics := len(n.topics)
	n.mu.Unlock()

	return ResourceMetrics{
		GoroutineCount: runtime.NumGoroutine(),
		HeapAllocMB:    float64(memStats.HeapAlloc) / (1024 * 1024),
		HeapSysMB:      float64(memStats.HeapSys) / (1024 * 1024),
		NumConnections: len(conns),
		NumStreams:     numStreams,
		NumTopics:      numTopics,
		NumPeers:       len(n.Host.Network().Peers()),
	}
}

// GetNetworkStats returns network-level statistics
func (n *P2PNetwork) GetNetworkStats() NetworkStats {
	n.mu.Lock()
	knownPeers := len(n.addressBook)
	n.mu.Unlock()

	return NetworkStats{
		ConnectedPeers:    len(n.Host.Network().Peers()),
		KnownPeers:        knownPeers,
		TotalBytesSent:    n.totalBytesSent.Load(),
		TotalBytesRecv:    n.totalBytesRecv.Load(),
		ReconnectAttempts: n.reconnectAttempts.Load(),
		ReconnectSuccess:  n.reconnectSuccess.Load(),
	}
}

// --- Phase 7: Propagation Metrics ---

func newPropagationMetrics(maxStore int) *PropagationMetrics {
	return &PropagationMetrics{
		samples:  make([]float64, 0, maxStore),
		maxStore: maxStore,
	}
}

// RecordPropagation records a propagation delay sample in milliseconds
func (n *P2PNetwork) RecordPropagation(delayMs float64) {
	n.propagation.mu.Lock()
	defer n.propagation.mu.Unlock()
	if len(n.propagation.samples) >= n.propagation.maxStore {
		// Ring buffer: overwrite oldest
		copy(n.propagation.samples, n.propagation.samples[1:])
		n.propagation.samples[len(n.propagation.samples)-1] = delayMs
	} else {
		n.propagation.samples = append(n.propagation.samples, delayMs)
	}
}

// GetPropagationStats returns p50/p95/p99 propagation delay in milliseconds
func (n *P2PNetwork) GetPropagationStats() (p50, p95, p99 float64) {
	n.propagation.mu.Lock()
	defer n.propagation.mu.Unlock()

	count := len(n.propagation.samples)
	if count == 0 {
		return 0, 0, 0
	}

	// Copy and sort
	sorted := make([]float64, count)
	copy(sorted, n.propagation.samples)
	sortFloat64s(sorted)

	p50 = sorted[count*50/100]
	idx95 := count * 95 / 100
	if idx95 >= count {
		idx95 = count - 1
	}
	p95 = sorted[idx95]
	idx99 := count * 99 / 100
	if idx99 >= count {
		idx99 = count - 1
	}
	p99 = sorted[idx99]
	return
}

func sortFloat64s(a []float64) {
	// Simple insertion sort, sufficient for metrics sampling
	for i := 1; i < len(a); i++ {
		key := a[i]
		j := i - 1
		for j >= 0 && a[j] > key {
			a[j+1] = a[j]
			j--
		}
		a[j+1] = key
	}
}

func (n *P2PNetwork) Close() error {
	if n.cancel != nil {
		n.cancel()
	}
	n.mu.Lock()
	defer n.mu.Unlock()
	if n.consensusSub != nil {
		n.consensusSub.Cancel()
	}
	if n.mempoolSub != nil {
		n.mempoolSub.Cancel()
	}
	for _, sub := range n.subs {
		if sub != nil {
			sub.Cancel()
		}
	}
	return n.Host.Close()
}
