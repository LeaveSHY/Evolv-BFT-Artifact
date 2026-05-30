package gbc

import (
	"encoding/json"
	"testing"
)

func TestOrderer_GlobalOrdering(t *testing.T) {
	log := NewLogWithMembers(4)
	orderer := NewOrderer(log, 4)

	var orderedResult []InstanceCommit
	orderer.OnGlobalOrder(func(epoch uint64, ordered []InstanceCommit) {
		orderedResult = ordered
	})

	// Submit commits from 4 instances in non-sorted order
	commits := []InstanceCommit{
		{InstanceID: 2, LocalHeight: 1, Rank: 3, Epoch: 0, BlockHash: []byte("b2")},
		{InstanceID: 0, LocalHeight: 1, Rank: 1, Epoch: 0, BlockHash: []byte("b0")},
		{InstanceID: 3, LocalHeight: 1, Rank: 4, Epoch: 0, BlockHash: []byte("b3")},
		{InstanceID: 1, LocalHeight: 1, Rank: 2, Epoch: 0, BlockHash: []byte("b1")},
	}
	for _, c := range commits {
		if err := orderer.SubmitCommit(c); err != nil {
			t.Fatalf("SubmitCommit failed: %v", err)
		}
	}

	if orderedResult == nil {
		t.Fatal("expected global order callback to fire")
	}
	if len(orderedResult) != 4 {
		t.Fatalf("expected 4 ordered commits, got %d", len(orderedResult))
	}

	// Verify rank ordering: 1, 2, 3, 4
	expectedRanks := []uint64{1, 2, 3, 4}
	for i, c := range orderedResult {
		if c.Rank != expectedRanks[i] {
			t.Fatalf("position %d: expected rank %d, got %d", i, expectedRanks[i], c.Rank)
		}
	}

	// Verify checkpoint was published to GBC log
	entry, ok := log.Retrieve(1)
	if !ok {
		t.Fatal("expected checkpoint entry at height 1")
	}
	if entry.Type != EntryCheckpoint {
		t.Fatalf("expected checkpoint type, got %s", entry.Type)
	}

	var checkpoint struct {
		Epoch   uint64           `json:"epoch"`
		Ordered []InstanceCommit `json:"ordered"`
	}
	if err := json.Unmarshal(entry.Payload, &checkpoint); err != nil {
		t.Fatalf("unmarshal checkpoint: %v", err)
	}
	if checkpoint.Epoch != 0 {
		t.Fatalf("expected epoch 0, got %d", checkpoint.Epoch)
	}
	if len(checkpoint.Ordered) != 4 {
		t.Fatalf("expected 4 ordered entries in checkpoint, got %d", len(checkpoint.Ordered))
	}
}

func TestOrderer_PartialEpochNoCallback(t *testing.T) {
	log := NewLogWithMembers(3)
	orderer := NewOrderer(log, 3)

	called := false
	orderer.OnGlobalOrder(func(epoch uint64, ordered []InstanceCommit) {
		called = true
	})

	// Submit only 2 of 3 instances
	orderer.SubmitCommit(InstanceCommit{InstanceID: 0, LocalHeight: 1, Rank: 1, Epoch: 0})
	orderer.SubmitCommit(InstanceCommit{InstanceID: 1, LocalHeight: 1, Rank: 2, Epoch: 0})

	if called {
		t.Fatal("callback should not fire until all instances contribute")
	}
	if orderer.PendingCount(0) != 2 {
		t.Fatalf("expected 2 pending, got %d", orderer.PendingCount(0))
	}
}

func TestOrderer_DedupSameInstance(t *testing.T) {
	log := NewLogWithMembers(2)
	orderer := NewOrderer(log, 2)

	orderer.SubmitCommit(InstanceCommit{InstanceID: 0, LocalHeight: 1, Rank: 1, Epoch: 0})
	orderer.SubmitCommit(InstanceCommit{InstanceID: 0, LocalHeight: 1, Rank: 1, Epoch: 0}) // dup

	if orderer.PendingCount(0) != 1 {
		t.Fatalf("expected 1 (deduped), got %d", orderer.PendingCount(0))
	}
}
