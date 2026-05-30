package gbc

import (
	"bytes"
	"crypto/ed25519"
	"crypto/sha256"
	"errors"
	"fmt"
	"sync"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════════
// GBC Node: Distributed GBC Protocol
//
// Implements the replicated append-only log protocol among m instance primaries
// as described in Section III-C. The protocol for each entry:
//
//   1. Propose:  The round-robin proposer creates an entry and broadcasts it.
//   2. Attest:   Each honest primary verifies the entry and sends an attestation.
//   3. Commit:   Once 2f+1 attestations are collected, the entry is committed
//                and broadcast to all (G4 satisfied).
//
// Consistency properties:
//   G1: Append-only integrity (height monotonicity enforced by local Log)
//   G2: Honest-primary common prefix (single proposer per height, quorum commit)
//   G3: Bounded retrieval delay (sync protocol after GST)
//   G4: 2f_GBC + 1 attestation quorum on every committed entry
// ═══════════════════════════════════════════════════════════════════════════════

// NodeConfig configures a GBC Node.
type NodeConfig struct {
	NodeID     uint64                                                  // this primary's validator ID
	NumMembers int                                                     // total number of GBC primaries (m)
	MemberIDs  []uint64                                                // ordered validator IDs of GBC primaries
	PrivateKey ed25519.PrivateKey                                      // optional Ed25519 signing key
	PublicKeys map[uint64]ed25519.PublicKey                            // optional Ed25519 verifier set
	SignFunc   func(digest [32]byte) []byte                            // signing function for attestations
	VerifyFunc func(signerID uint64, digest [32]byte, sig []byte) bool // signature verification
}

var errNotProposer = errors.New("gbc: not proposer")

// IsNotProposer reports whether an error means this node is not the GBC proposer
// for the current height. Callers should wait for the proposer broadcast instead
// of falling back to uncertified publication.
func IsNotProposer(err error) bool {
	return errors.Is(err, errNotProposer)
}

// defaultSignFunc produces a deterministic pseudo-signature for testing.
func defaultSignFunc(nodeID uint64) func([32]byte) []byte {
	return func(digest [32]byte) []byte {
		return pseudoSignature(nodeID, digest)
	}
}

func pseudoSignature(nodeID uint64, digest [32]byte) []byte {
	h := sha256.New()
	h.Write(digest[:])
	h.Write([]byte(fmt.Sprintf("signer-%d", nodeID)))
	return h.Sum(nil)
}

// defaultVerifyFunc verifies the deterministic pseudo-signature used by tests.
func defaultVerifyFunc(signerID uint64, digest [32]byte, sig []byte) bool {
	return bytes.Equal(sig, pseudoSignature(signerID, digest))
}

func ed25519SignFunc(privateKey ed25519.PrivateKey) func([32]byte) []byte {
	return func(digest [32]byte) []byte {
		if len(privateKey) != ed25519.PrivateKeySize {
			return nil
		}
		return ed25519.Sign(privateKey, digest[:])
	}
}

func ed25519VerifyFunc(publicKeys map[uint64]ed25519.PublicKey) func(uint64, [32]byte, []byte) bool {
	return func(signerID uint64, digest [32]byte, sig []byte) bool {
		pub, ok := publicKeys[signerID]
		if !ok || len(pub) != ed25519.PublicKeySize {
			return false
		}
		return ed25519.Verify(pub, digest[:], sig)
	}
}

// Node is a single GBC primary participating in the distributed protocol.
// It maintains a local Log and communicates with peers via Transport.
type Node struct {
	mu        sync.Mutex
	config    NodeConfig
	log       *Log
	transport Transport

	// Pending proposals awaiting quorum attestations
	pending map[uint64]*pendingEntry // height → pending

	// Committed entry callback (optional)
	onCommit func(Entry)

	// Running state
	stopCh chan struct{}
	wg     sync.WaitGroup
}

type pendingEntry struct {
	entry        Entry
	attestations map[uint64]Attestation // signerID → attestation
	committed    chan Entry
}

// NewNode creates a GBC node. Call Start() to begin processing messages.
func NewNode(cfg NodeConfig, transport Transport) *Node {
	cfg.MemberIDs = normalizeMemberIDs(cfg.MemberIDs, cfg.NumMembers)
	cfg.NumMembers = len(cfg.MemberIDs)
	if cfg.SignFunc == nil {
		if len(cfg.PrivateKey) == ed25519.PrivateKeySize {
			cfg.SignFunc = ed25519SignFunc(cfg.PrivateKey)
		} else {
			cfg.SignFunc = defaultSignFunc(cfg.NodeID)
		}
	}
	if cfg.VerifyFunc == nil {
		if len(cfg.PublicKeys) > 0 {
			cfg.VerifyFunc = ed25519VerifyFunc(cfg.PublicKeys)
		} else {
			cfg.VerifyFunc = defaultVerifyFunc
		}
	}
	log := NewLogWithMembers(cfg.NumMembers)
	log.SetVerifier(cfg.VerifyFunc)
	return &Node{
		config:    cfg,
		log:       log,
		transport: transport,
		pending:   make(map[uint64]*pendingEntry),
		stopCh:    make(chan struct{}),
	}
}

func normalizeMemberIDs(memberIDs []uint64, numMembers int) []uint64 {
	if len(memberIDs) > 0 {
		return append([]uint64(nil), memberIDs...)
	}
	if numMembers < 0 {
		numMembers = 0
	}
	members := make([]uint64, numMembers)
	for i := range members {
		members[i] = uint64(i)
	}
	return members
}

func (n *Node) proposerForHeight(height uint64) uint64 {
	if n.config.NumMembers == 0 || len(n.config.MemberIDs) == 0 {
		return 0
	}
	idx := height % uint64(n.config.NumMembers)
	return n.config.MemberIDs[idx]
}

// OnCommit registers a callback invoked when an entry achieves quorum commit.
func (n *Node) OnCommit(fn func(Entry)) {
	n.mu.Lock()
	defer n.mu.Unlock()
	n.onCommit = fn
}

// Log returns the underlying local log for read access.
func (n *Node) Log() *Log {
	return n.log
}

// Start begins the message processing loop in a background goroutine.
func (n *Node) Start() {
	n.wg.Add(1)
	go n.processLoop()
}

// Stop shuts down the node's message processing.
func (n *Node) Stop() {
	select {
	case <-n.stopCh:
		return
	default:
		close(n.stopCh)
	}
	n.transport.Close()
	n.wg.Wait()
}

func (n *Node) Propose(entryType EntryType, payload []byte) (uint64, error) {
	height, _, err := n.propose(entryType, payload)
	return height, err
}

// ProposeAndWait creates a new entry and waits until it obtains a G4 quorum
// commit. This is the authoritative API for callers that must implement
// commit-then-apply semantics.
func (n *Node) ProposeAndWait(entryType EntryType, payload []byte, timeout time.Duration) (uint64, Entry, error) {
	if timeout <= 0 {
		timeout = 2 * time.Second
	}
	height, committed, err := n.propose(entryType, payload)
	if err != nil {
		return 0, Entry{}, err
	}
	select {
	case entry := <-committed:
		return height, entry, nil
	case <-time.After(timeout):
		return height, Entry{}, fmt.Errorf("gbc: quorum commit timeout at height %d", height)
	}
}

// Propose creates a new entry and broadcasts it to all primaries.
// Only the current proposer (round-robin by height) should call this.
// Returns the proposed height.
func (n *Node) propose(entryType EntryType, payload []byte) (uint64, <-chan Entry, error) {
	n.mu.Lock()
	defer n.mu.Unlock()

	height := n.log.Height()
	if n.config.NumMembers == 0 {
		return 0, nil, fmt.Errorf("gbc: no members configured")
	}

	// Verify this node is the proposer for this height
	proposer := n.proposerForHeight(height)
	if proposer != n.config.NodeID {
		return 0, nil, fmt.Errorf("%w: node %d is not proposer for height %d (proposer=%d)", errNotProposer,
			n.config.NodeID, height, proposer)
	}

	entry := Entry{
		Height:  height,
		Type:    entryType,
		Payload: append([]byte(nil), payload...),
	}

	// Self-attest
	digest := EntryDigest(entry.Height, entry.Type, entry.Payload)
	selfAtt := Attestation{
		SignerID:  n.config.NodeID,
		Signature: n.config.SignFunc(digest),
	}
	entry.Attestations = []Attestation{selfAtt}

	// Publish locally (validates G1 contiguous height)
	if err := n.log.Publish(entry); err != nil {
		return 0, nil, fmt.Errorf("gbc: local publish failed: %w", err)
	}

	// Track pending attestations
	pe := &pendingEntry{
		entry:        entry,
		attestations: map[uint64]Attestation{n.config.NodeID: selfAtt},
		committed:    make(chan Entry, 1),
	}
	n.pending[height] = pe

	// Broadcast proposal
	if err := n.transport.Broadcast(Message{
		Type:     MsgPropose,
		SenderID: n.config.NodeID,
		Entry:    entry,
	}); err != nil {
		return 0, nil, fmt.Errorf("gbc: broadcast propose failed: %w", err)
	}

	// Check if self-attestation alone reaches quorum (m=1 case)
	if n.checkAndCommit(pe, height) {
		return height, pe.committed, nil
	}

	return height, pe.committed, nil
}

// processLoop handles incoming messages until stopped.
func (n *Node) processLoop() {
	defer n.wg.Done()
	for {
		select {
		case <-n.stopCh:
			return
		default:
		}

		msg, err := n.transport.Receive()
		if err != nil {
			select {
			case <-n.stopCh:
				return
			default:
				continue
			}
		}

		switch msg.Type {
		case MsgPropose:
			n.handlePropose(msg)
		case MsgAttest:
			n.handleAttest(msg)
		case MsgCommitted:
			n.handleCommitted(msg)
		case MsgSyncReq:
			n.handleSyncReq(msg)
		case MsgSyncResp:
			n.handleSyncResp(msg)
		}
	}
}

// handlePropose processes a proposal from the proposer for this height.
func (n *Node) handlePropose(msg Message) {
	n.mu.Lock()
	defer n.mu.Unlock()

	entry := msg.Entry

	// Verify proposer is correct for this height
	expectedProposer := n.proposerForHeight(entry.Height)
	if msg.SenderID != expectedProposer {
		return // wrong proposer, ignore
	}
	if !n.hasValidProposerAttestation(entry, msg.SenderID) {
		return
	}

	// Verify height matches our expectation (may need sync if behind)
	localHeight := n.log.Height()
	if entry.Height < localHeight {
		return // already have this entry
	}
	if entry.Height > localHeight {
		// We're behind; request sync
		n.requestSync(msg.SenderID, localHeight)
		return
	}

	// Publish locally
	cleanEntry := Entry{
		Height:  entry.Height,
		Type:    entry.Type,
		Payload: append([]byte(nil), entry.Payload...),
	}
	if err := n.log.Publish(cleanEntry); err != nil {
		return // G1 violation
	}

	// Create attestation
	digest := EntryDigest(entry.Height, entry.Type, entry.Payload)
	att := Attestation{
		SignerID:  n.config.NodeID,
		Signature: n.config.SignFunc(digest),
	}

	// Send attestation back to proposer
	n.transport.Send(msg.SenderID, Message{
		Type:        MsgAttest,
		SenderID:    n.config.NodeID,
		Entry:       Entry{Height: entry.Height},
		Attestation: att,
	})
}

// handleAttest processes an attestation from a peer for a pending proposal.
func (n *Node) handleAttest(msg Message) {
	n.mu.Lock()
	defer n.mu.Unlock()

	height := msg.Entry.Height
	pe, ok := n.pending[height]
	if !ok {
		return // not tracking this height
	}

	// Verify attestation signature
	digest := EntryDigest(pe.entry.Height, pe.entry.Type, pe.entry.Payload)
	if !n.config.VerifyFunc(msg.Attestation.SignerID, digest, msg.Attestation.Signature) {
		return // invalid signature
	}

	// Collect attestation (dedup by signer)
	if _, exists := pe.attestations[msg.Attestation.SignerID]; exists {
		return
	}
	pe.attestations[msg.Attestation.SignerID] = msg.Attestation

	// Also add to local log
	n.log.Attest(height, msg.Attestation)

	// Check quorum
	n.checkAndCommit(pe, height)
}

// handleCommitted processes a committed entry broadcast (with quorum attestations).
func (n *Node) handleCommitted(msg Message) {
	n.mu.Lock()
	defer n.mu.Unlock()

	entry := msg.Entry
	localHeight := n.log.Height()

	if entry.Height < localHeight {
		// Already have it; just add any new attestations
		for _, att := range entry.Attestations {
			_, _ = n.log.Attest(entry.Height, att)
		}
		return
	}

	if entry.Height > localHeight {
		// We're behind; request sync
		n.requestSync(msg.SenderID, localHeight)
		return
	}

	if err := n.log.VerifyEntry(entry, true); err != nil {
		return
	}

	// Publish and add all attestations
	cleanEntry := Entry{
		Height:  entry.Height,
		Type:    entry.Type,
		Payload: append([]byte(nil), entry.Payload...),
	}
	if err := n.log.Publish(cleanEntry); err != nil {
		return
	}
	for _, att := range entry.Attestations {
		_, _ = n.log.Attest(entry.Height, att)
	}

	// Fire commit callback
	if n.onCommit != nil {
		n.onCommit(entry)
	}

	// Clean up pending if we were tracking
	delete(n.pending, entry.Height)
}

// handleSyncReq responds with missing entries.
func (n *Node) handleSyncReq(msg Message) {
	n.mu.Lock()
	defer n.mu.Unlock()

	fromHeight := msg.SyncFrom
	localHeight := n.log.Height()

	var entries []Entry
	for h := fromHeight; h < localHeight; h++ {
		if entry, ok := n.log.Retrieve(h); ok {
			entries = append(entries, entry)
		}
	}

	if len(entries) > 0 {
		n.transport.Send(msg.SenderID, Message{
			Type:        MsgSyncResp,
			SenderID:    n.config.NodeID,
			SyncEntries: entries,
		})
	}
}

// handleSyncResp applies received sync entries to catch up.
func (n *Node) handleSyncResp(msg Message) {
	n.mu.Lock()
	defer n.mu.Unlock()

	for _, entry := range msg.SyncEntries {
		localHeight := n.log.Height()
		if entry.Height != localHeight {
			continue // not the next expected entry
		}
		if err := n.log.VerifyEntry(entry, true); err != nil {
			continue
		}

		cleanEntry := Entry{
			Height:  entry.Height,
			Type:    entry.Type,
			Payload: append([]byte(nil), entry.Payload...),
		}
		if err := n.log.Publish(cleanEntry); err != nil {
			break
		}
		for _, att := range entry.Attestations {
			_, _ = n.log.Attest(entry.Height, att)
		}
	}
}

// checkAndCommit checks if a pending entry has quorum and broadcasts commit if so.
// Returns true if committed. Must be called with n.mu held.
func (n *Node) checkAndCommit(pe *pendingEntry, height uint64) bool {
	needed := QuorumSize(n.config.NumMembers)
	if len(pe.attestations) < needed {
		return false
	}

	// Build committed entry with all attestations
	committed := Entry{
		Height:  pe.entry.Height,
		Type:    pe.entry.Type,
		Payload: append([]byte(nil), pe.entry.Payload...),
	}
	for _, att := range pe.attestations {
		committed.Attestations = append(committed.Attestations, att)
	}
	if err := n.log.VerifyEntry(committed, true); err != nil {
		return false
	}

	// Broadcast committed entry to all peers
	n.transport.Broadcast(Message{
		Type:     MsgCommitted,
		SenderID: n.config.NodeID,
		Entry:    committed,
	})

	// Fire commit callback
	if n.onCommit != nil {
		n.onCommit(committed)
	}
	if pe.committed != nil {
		select {
		case pe.committed <- committed:
		default:
		}
	}

	// Clean up
	delete(n.pending, height)
	return true
}

func (n *Node) hasValidProposerAttestation(entry Entry, proposerID uint64) bool {
	for _, att := range entry.Attestations {
		if att.SignerID != proposerID {
			continue
		}
		if n.config.VerifyFunc(att.SignerID, EntryDigest(entry.Height, entry.Type, entry.Payload), att.Signature) {
			return true
		}
	}
	return false
}

// requestSync sends a sync request to a peer. Must be called with n.mu held.
func (n *Node) requestSync(peer uint64, fromHeight uint64) {
	n.transport.Send(peer, Message{
		Type:     MsgSyncReq,
		SenderID: n.config.NodeID,
		SyncFrom: fromHeight,
	})
}

// IsProposer returns true if this node is the proposer for the current height.
func (n *Node) IsProposer() bool {
	n.mu.Lock()
	defer n.mu.Unlock()
	height := n.log.Height()
	return height%uint64(n.config.NumMembers) == n.config.NodeID
}

// Height returns the next expected height from this node's local log.
func (n *Node) Height() uint64 {
	return n.log.Height()
}
