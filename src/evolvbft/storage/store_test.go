package storage

import (
	"testing"

	"evolvbft/evolvbft/types"
)

func makeBlock(height uint64) *types.Block {
	return types.NewBlock(height, make([]byte, 32), []byte("data"), height, 1, 0, 0,
		types.NewQuorumCertificate(nil, 0, 0, types.PhaseDecide), nil)
}

func TestNewStorageManager(t *testing.T) {
	sm := NewStorageManager(1)
	if sm == nil {
		t.Fatal("NewStorageManager should not return nil")
	}
}

func TestPutBlock_And_GetBlock(t *testing.T) {
	sm := NewStorageManager(1)
	block := makeBlock(1)

	if err := sm.PutBlock(block); err != nil {
		t.Fatalf("PutBlock failed: %v", err)
	}

	got, err := sm.GetBlock(block.Hash)
	if err != nil {
		t.Fatalf("GetBlock failed: %v", err)
	}
	if got.Height != 1 {
		t.Errorf("expected height 1, got %d", got.Height)
	}
}

func TestPutBlock_NilBlock(t *testing.T) {
	sm := NewStorageManager(1)
	if err := sm.PutBlock(nil); err == nil {
		t.Error("nil block should fail")
	}
}

func TestPutBlock_NilHash(t *testing.T) {
	sm := NewStorageManager(1)
	block := &types.Block{Height: 1, Hash: nil}
	if err := sm.PutBlock(block); err == nil {
		t.Error("block with nil hash should fail")
	}
}

func TestGetBlock_NotFound(t *testing.T) {
	sm := NewStorageManager(1)
	_, err := sm.GetBlock([]byte("nonexistent"))
	if err == nil {
		t.Error("should fail for nonexistent block")
	}
}

func TestGetBlockByHeight(t *testing.T) {
	sm := NewStorageManager(1)
	block := makeBlock(5)
	sm.PutBlock(block)

	got, err := sm.GetBlockByHeight(5)
	if err != nil {
		t.Fatalf("GetBlockByHeight failed: %v", err)
	}
	if got.Height != 5 {
		t.Errorf("expected height 5, got %d", got.Height)
	}
}

func TestGetBlockByHeight_NotFound(t *testing.T) {
	sm := NewStorageManager(1)
	_, err := sm.GetBlockByHeight(99)
	if err == nil {
		t.Error("should fail for nonexistent height")
	}
}

func TestHasBlock(t *testing.T) {
	sm := NewStorageManager(1)
	block := makeBlock(1)
	sm.PutBlock(block)

	if !sm.HasBlock(block.Hash) {
		t.Error("should find stored block")
	}
	if sm.HasBlock([]byte("missing")) {
		t.Error("should not find missing block")
	}
}

func TestGetLatestBlockHeight(t *testing.T) {
	sm := NewStorageManager(1)
	if h := sm.GetLatestBlockHeight(); h != 0 {
		t.Errorf("empty store should have height 0, got %d", h)
	}

	sm.PutBlock(makeBlock(3))
	sm.PutBlock(makeBlock(7))
	sm.PutBlock(makeBlock(5))

	if h := sm.GetLatestBlockHeight(); h != 7 {
		t.Errorf("expected latest height 7, got %d", h)
	}
}

func TestClose(t *testing.T) {
	sm := NewStorageManager(1)
	sm.PutBlock(makeBlock(1))

	if err := sm.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	// After close, store should be empty
	if sm.HasBlock([]byte("anything")) {
		t.Error("store should be empty after close")
	}
	if h := sm.GetLatestBlockHeight(); h != 0 {
		t.Errorf("should be 0 after close, got %d", h)
	}
}
