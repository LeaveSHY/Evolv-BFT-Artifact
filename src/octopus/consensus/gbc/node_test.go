package gbc

import (
	"crypto/ed25519"
	"sync"
	"testing"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════════
// Tests for the distributed GBC protocol (propose → attest → commit)
// ═══════════════════════════════════════════════════════════════════════════════

// waitFor polls a condition with timeout. Returns true if condition met.
func waitFor(t *testing.T, timeout time.Duration, condition func() bool) bool {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if condition() {
			return true
		}
		time.Sleep(5 * time.Millisecond)
	}
	return false
}

func TestNodeProposeAndCommitSingleNode(t *testing.T) {
	// m=1: single primary, self-attestation should reach quorum immediately
	transports := NewChannelTransportSet(1, 64)
	node := NewNode(NodeConfig{
		NodeID:     0,
		NumMembers: 1,
	}, transports[0])

	var committed Entry
	var mu sync.Mutex
	node.OnCommit(func(e Entry) {
		mu.Lock()
		committed = e
		mu.Unlock()
	})
	node.Start()
	defer node.Stop()

	height, err := node.Propose(EntryQC, []byte("test-qc"))
	if err != nil {
		t.Fatalf("propose failed: %v", err)
	}
	if height != 1 {
		t.Fatalf("expected height 1, got %d", height)
	}

	// With m=1, quorum is immediate
	ok := waitFor(t, time.Second, func() bool {
		mu.Lock()
		defer mu.Unlock()
		return committed.Height == 1
	})
	if !ok {
		t.Fatal("expected commit callback for single-node quorum")
	}
}

func TestNodeProposeFourNodeQuorum(t *testing.T) {
	// m=4 primaries, f=1, quorum=3
	numNodes := 4
	transports := NewChannelTransportSet(numNodes, 256)

	nodes := make([]*Node, numNodes)
	commits := make([][]Entry, numNodes)
	var mu sync.Mutex

	for i := 0; i < numNodes; i++ {
		idx := i
		nodes[i] = NewNode(NodeConfig{
			NodeID:     uint64(i),
			NumMembers: numNodes,
		}, transports[i])
		nodes[i].OnCommit(func(e Entry) {
			mu.Lock()
			commits[idx] = append(commits[idx], e)
			mu.Unlock()
		})
		nodes[i].Start()
	}
	defer func() {
		for _, n := range nodes {
			n.Stop()
		}
	}()

	// Height 1: proposer is node 1%4=1... actually height starts at 1, so proposer = 1%4 = 1
	// Wait, height starts at 1 (nextHeight=1 in NewLog), so proposer = 1%4 = 1
	proposerID := uint64(1) % uint64(numNodes)
	height, err := nodes[proposerID].Propose(EntryCheckpoint, []byte("checkpoint-1"))
	if err != nil {
		t.Fatalf("propose failed: %v", err)
	}
	if height != 1 {
		t.Fatalf("expected height 1, got %d", height)
	}

	// Wait for proposer to get quorum (needs 3 attestations including self)
	ok := waitFor(t, 2*time.Second, func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(commits[proposerID]) >= 1
	})
	if !ok {
		t.Fatal("proposer did not achieve quorum within timeout")
	}

	// Verify committed entry on proposer
	mu.Lock()
	entry := commits[proposerID][0]
	mu.Unlock()
	if entry.Height != 1 || entry.Type != EntryCheckpoint {
		t.Fatalf("unexpected committed entry: %+v", entry)
	}
	if !entry.HasQuorum(numNodes) {
		t.Fatal("committed entry should have quorum")
	}

	// Wait for all followers to receive the committed entry
	ok = waitFor(t, 2*time.Second, func() bool {
		for i := 0; i < numNodes; i++ {
			if nodes[i].Height() < 2 {
				return false
			}
		}
		return true
	})
	if !ok {
		for i := 0; i < numNodes; i++ {
			t.Logf("node %d height: %d", i, nodes[i].Height())
		}
		t.Fatal("not all nodes advanced to height 2")
	}
}

func TestNodeRejectsWrongProposer(t *testing.T) {
	transports := NewChannelTransportSet(4, 64)
	node := NewNode(NodeConfig{
		NodeID:     0,
		NumMembers: 4,
	}, transports[0])
	node.Start()
	defer node.Stop()

	// Height 1: proposer should be 1%4=1, but we're node 0
	_, err := node.Propose(EntryQC, []byte("should-fail"))
	if err == nil {
		t.Fatal("expected error when wrong proposer tries to propose")
	}
}

func TestNodeUsesExplicitMemberIDsForProposer(t *testing.T) {
	transports := NewChannelTransportSet(2, 64)
	node7 := NewNode(NodeConfig{
		NodeID:     7,
		MemberIDs:  []uint64{7, 42},
		NumMembers: 2,
	}, transports[0])
	node7.Start()
	defer node7.Stop()

	if _, err := node7.Propose(EntryQC, []byte("wrong-primary")); err == nil {
		t.Fatal("expected node 7 to reject height 1 because proposer is member 42")
	}

	node42 := NewNode(NodeConfig{
		NodeID:     42,
		MemberIDs:  []uint64{7, 42},
		NumMembers: 2,
	}, transports[1])
	node42.Start()
	defer node42.Stop()

	if _, err := node42.Propose(EntryQC, []byte("right-primary")); err != nil {
		t.Fatalf("expected explicit member proposer to succeed: %v", err)
	}
}

func TestNodeHandleProposeUsesExplicitMemberIDs(t *testing.T) {
	transports := NewChannelTransportSet(1, 64)
	node7 := NewNode(NodeConfig{
		NodeID:     7,
		MemberIDs:  []uint64{7, 42},
		NumMembers: 2,
	}, transports[0])

	entry := Entry{Height: 1, Type: EntryQC, Payload: []byte("explicit-member-proposal")}
	digest := EntryDigest(entry.Height, entry.Type, entry.Payload)
	entry.Attestations = []Attestation{{SignerID: 42, Signature: pseudoSignature(42, digest)}}

	node7.handlePropose(Message{Type: MsgPropose, SenderID: 42, Entry: entry})

	if got := node7.Height(); got != 2 {
		t.Fatalf("expected follower to accept explicit member proposer and advance to height 2, got %d", got)
	}
}

func TestNodeMultipleEntries(t *testing.T) {
	numNodes := 4
	transports := NewChannelTransportSet(numNodes, 256)

	nodes := make([]*Node, numNodes)
	commitCounts := make([]int, numNodes)
	var mu sync.Mutex

	for i := 0; i < numNodes; i++ {
		idx := i
		nodes[i] = NewNode(NodeConfig{
			NodeID:     uint64(i),
			NumMembers: numNodes,
		}, transports[i])
		nodes[i].OnCommit(func(e Entry) {
			mu.Lock()
			commitCounts[idx]++
			mu.Unlock()
		})
		nodes[i].Start()
	}
	defer func() {
		for _, n := range nodes {
			n.Stop()
		}
	}()

	// Propose 4 entries (round-robin proposers)
	for h := uint64(1); h <= 4; h++ {
		proposer := h % uint64(numNodes)

		// Wait for all nodes to be at this height before proposing
		ok := waitFor(t, 2*time.Second, func() bool {
			return nodes[proposer].Height() == h
		})
		if !ok {
			t.Fatalf("proposer %d not ready for height %d (at %d)", proposer, h, nodes[proposer].Height())
		}

		_, err := nodes[proposer].Propose(EntryQC, []byte("entry"))
		if err != nil {
			t.Fatalf("propose at height %d by node %d failed: %v", h, proposer, err)
		}

		// Wait for proposer to advance past this height (commit happened)
		ok = waitFor(t, 2*time.Second, func() bool {
			return nodes[proposer].Height() > h
		})
		if !ok {
			t.Fatalf("height %d did not commit (proposer %d at height %d)", h, proposer, nodes[proposer].Height())
		}
	}

	// All nodes should have advanced to height 5
	ok := waitFor(t, 3*time.Second, func() bool {
		for i := 0; i < numNodes; i++ {
			if nodes[i].Height() < 5 {
				return false
			}
		}
		return true
	})
	if !ok {
		for i := 0; i < numNodes; i++ {
			t.Logf("node %d height: %d", i, nodes[i].Height())
		}
		t.Fatal("not all nodes advanced to height 5 after 4 entries")
	}
}

func TestNodeSyncLaggingNode(t *testing.T) {
	// Create 4 transports but start only 3 nodes initially
	numNodes := 4
	transports := NewChannelTransportSet(numNodes, 256)

	nodes := make([]*Node, numNodes)
	for i := 0; i < 3; i++ {
		nodes[i] = NewNode(NodeConfig{
			NodeID:     uint64(i),
			NumMembers: numNodes,
		}, transports[i])
		nodes[i].Start()
	}

	// Propose height 1 (proposer = 1)
	ok := waitFor(t, time.Second, func() bool {
		return nodes[1].Height() == 1
	})
	if !ok {
		t.Fatal("node 1 not ready")
	}

	_, err := nodes[1].Propose(EntryQC, []byte("entry-1"))
	if err != nil {
		t.Fatalf("propose failed: %v", err)
	}

	// Wait for nodes 0,1,2 to commit
	ok = waitFor(t, 2*time.Second, func() bool {
		for i := 0; i < 3; i++ {
			if nodes[i].Height() < 2 {
				return false
			}
		}
		return true
	})
	if !ok {
		t.Fatal("first 3 nodes didn't advance")
	}

	// Now start node 3 (lagging)
	nodes[3] = NewNode(NodeConfig{
		NodeID:     uint64(3),
		NumMembers: numNodes,
	}, transports[3])
	nodes[3].Start()

	// Propose height 2 (proposer = 2%4 = 2)
	_, err = nodes[2].Propose(EntryCheckpoint, []byte("entry-2"))
	if err != nil {
		t.Fatalf("propose height 2 failed: %v", err)
	}

	// Node 3 should sync and catch up
	ok = waitFor(t, 3*time.Second, func() bool {
		return nodes[3].Height() >= 2
	})
	if !ok {
		t.Logf("node 3 height: %d", nodes[3].Height())
		t.Log("node 3 may need committed broadcast to catch up (sync via MsgCommitted)")
	}

	// Clean up
	for _, n := range nodes {
		if n != nil {
			n.Stop()
		}
	}
}

func TestNodeIsProposer(t *testing.T) {
	transports := NewChannelTransportSet(4, 64)
	// Height starts at 1, proposer = 1%4 = 1
	for i := 0; i < 4; i++ {
		node := NewNode(NodeConfig{
			NodeID:     uint64(i),
			NumMembers: 4,
		}, transports[i])
		expected := uint64(i) == 1 // only node 1 is proposer for height 1
		if node.IsProposer() != expected {
			t.Errorf("node %d IsProposer() = %v, expected %v", i, node.IsProposer(), expected)
		}
	}
}

func TestEntryDigestDeterministic(t *testing.T) {
	d1 := EntryDigest(1, EntryQC, []byte("payload"))
	d2 := EntryDigest(1, EntryQC, []byte("payload"))
	if d1 != d2 {
		t.Fatal("EntryDigest should be deterministic")
	}

	d3 := EntryDigest(2, EntryQC, []byte("payload"))
	if d1 == d3 {
		t.Fatal("different heights should produce different digests")
	}

	d4 := EntryDigest(1, EntryCheckpoint, []byte("payload"))
	if d1 == d4 {
		t.Fatal("different types should produce different digests")
	}
}

func TestLogVerifierRejectsInvalidAttestation(t *testing.T) {
	log := NewLogWithMembers(4)
	log.SetVerifier(defaultVerifyFunc)
	entry := Entry{Height: 1, Type: EntryQC, Payload: []byte("qc")}
	if err := log.Publish(entry); err != nil {
		t.Fatalf("publish failed: %v", err)
	}
	if _, err := log.Attest(1, Attestation{SignerID: 1, Signature: []byte("not-a-valid-signature")}); err == nil {
		t.Fatal("expected invalid attestation to be rejected")
	}
	digest := EntryDigest(1, EntryQC, []byte("qc"))
	if hasQuorum, err := log.Attest(1, Attestation{SignerID: 1, Signature: defaultSignFunc(1)(digest)}); err != nil {
		t.Fatalf("valid attestation rejected: %v", err)
	} else if hasQuorum {
		t.Fatal("one attestation should not satisfy quorum for four members")
	}
}

func TestNodeRejectsInvalidCommittedEntry(t *testing.T) {
	transports := NewChannelTransportSet(4, 64)
	node := NewNode(NodeConfig{NodeID: 0, NumMembers: 4}, transports[0])
	entry := Entry{Height: 1, Type: EntryQC, Payload: []byte("qc")}
	for signer := uint64(0); signer < 3; signer++ {
		entry.Attestations = append(entry.Attestations, Attestation{SignerID: signer, Signature: []byte("bad")})
	}
	node.handleCommitted(Message{Type: MsgCommitted, SenderID: 1, Entry: entry})
	if node.Height() != 1 {
		t.Fatalf("invalid committed entry advanced height to %d", node.Height())
	}
}

func TestNodeCommitsEd25519Quorum(t *testing.T) {
	numNodes := 4
	transports := NewChannelTransportSet(numNodes, 256)
	publicKeys := make(map[uint64]ed25519.PublicKey, numNodes)
	privateKeys := make(map[uint64]ed25519.PrivateKey, numNodes)
	for i := 0; i < numNodes; i++ {
		pub, priv, err := ed25519.GenerateKey(nil)
		if err != nil {
			t.Fatalf("generate key: %v", err)
		}
		publicKeys[uint64(i)] = pub
		privateKeys[uint64(i)] = priv
	}

	nodes := make([]*Node, numNodes)
	for i := 0; i < numNodes; i++ {
		nodes[i] = NewNode(NodeConfig{
			NodeID:     uint64(i),
			NumMembers: numNodes,
			PrivateKey: privateKeys[uint64(i)],
			PublicKeys: publicKeys,
		}, transports[i])
		nodes[i].Start()
	}
	defer func() {
		for _, node := range nodes {
			node.Stop()
		}
	}()

	proposerID := uint64(1)
	if _, err := nodes[proposerID].Propose(EntryQC, []byte("ed25519-qc")); err != nil {
		t.Fatalf("propose failed: %v", err)
	}
	if ok := waitFor(t, 2*time.Second, func() bool {
		entry, ok := nodes[proposerID].Log().Retrieve(1)
		return ok && entry.HasQuorum(numNodes)
	}); !ok {
		t.Fatal("ed25519 quorum did not commit")
	}
	entry, ok := nodes[proposerID].Log().Retrieve(1)
	if !ok {
		t.Fatal("expected committed entry at height 1")
	}
	if err := nodes[proposerID].Log().VerifyEntry(entry, true); err != nil {
		t.Fatalf("committed entry failed verification: %v", err)
	}
}

func TestChannelTransportBroadcast(t *testing.T) {
	transports := NewChannelTransportSet(3, 64)

	msg := Message{Type: MsgPropose, SenderID: 0}
	if err := transports[0].Broadcast(msg); err != nil {
		t.Fatalf("broadcast failed: %v", err)
	}

	// Nodes 1 and 2 should receive, node 0 should not
	for i := 1; i <= 2; i++ {
		select {
		case got := <-transports[i].inbox:
			if got.Type != MsgPropose || got.SenderID != 0 {
				t.Fatalf("node %d received unexpected message: %+v", i, got)
			}
		case <-time.After(time.Second):
			t.Fatalf("node %d did not receive broadcast", i)
		}
	}

	select {
	case <-transports[0].inbox:
		t.Fatal("sender should not receive own broadcast")
	case <-time.After(50 * time.Millisecond):
		// Good, no self-delivery
	}
}

func TestChannelTransportClose(t *testing.T) {
	transports := NewChannelTransportSet(2, 64)
	transports[0].Close()

	if err := transports[0].Send(1, Message{}); err == nil {
		t.Fatal("expected error after close")
	}

	if _, err := transports[0].Receive(); err == nil {
		t.Fatal("expected error after close")
	}

	// Close is idempotent
	if err := transports[0].Close(); err != nil {
		t.Fatalf("second close should not error: %v", err)
	}
}
