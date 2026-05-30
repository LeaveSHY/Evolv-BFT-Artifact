package coding

import (
	"bytes"
	"testing"
)

func TestNewEncoder(t *testing.T) {
	enc, err := NewEncoder(4, 2)
	if err != nil {
		t.Fatalf("NewEncoder(4,2) failed: %v", err)
	}
	if enc.DataShards != 4 {
		t.Errorf("expected 4 data shards, got %d", enc.DataShards)
	}
	if enc.ParityShards != 2 {
		t.Errorf("expected 2 parity shards, got %d", enc.ParityShards)
	}
}

func TestNewEncoder_InvalidParams(t *testing.T) {
	_, err := NewEncoder(0, 0)
	if err == nil {
		t.Error("should fail with 0 shards")
	}
}

func TestEncodeAndReconstruct_RoundTrip(t *testing.T) {
	enc, _ := NewEncoder(4, 2)
	data := []byte("This is test data for reed-solomon encoding round trip test!!")

	shards, err := enc.Encode(data)
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}
	if len(shards) != 6 {
		t.Fatalf("expected 6 shards (4 data + 2 parity), got %d", len(shards))
	}

	// Reconstruct from all shards
	recovered, err := enc.Reconstruct(shards)
	if err != nil {
		t.Fatalf("Reconstruct failed: %v", err)
	}

	// recovered may have padding, but should start with original data
	if !bytes.HasPrefix(recovered, data) {
		t.Error("reconstructed data should start with original data")
	}
}

func TestReconstruct_WithMissingShards(t *testing.T) {
	enc, _ := NewEncoder(4, 2)
	data := []byte("Reed-Solomon can tolerate up to parityShards missing shards!!")

	shards, _ := enc.Encode(data)

	// Lose 2 shards (parity tolerance)
	shards[0] = nil
	shards[3] = nil

	recovered, err := enc.Reconstruct(shards)
	if err != nil {
		t.Fatalf("Reconstruct with 2 missing shards should succeed: %v", err)
	}
	if !bytes.HasPrefix(recovered, data) {
		t.Error("should recover original data")
	}
}

func TestReconstruct_TooManyMissing(t *testing.T) {
	enc, _ := NewEncoder(4, 2)
	data := []byte("Cannot recover with more than parityShards missing!!!!!!!!!!!!")

	shards, _ := enc.Encode(data)

	// Lose 3 shards (more than 2 parity can handle)
	shards[0] = nil
	shards[1] = nil
	shards[2] = nil

	_, err := enc.Reconstruct(shards)
	if err == nil {
		t.Error("should fail with 3 missing shards (parity=2)")
	}
}

func TestEncode_EmptyData(t *testing.T) {
	enc, _ := NewEncoder(2, 1)
	_, err := enc.Encode([]byte{})
	// reedsolomon.Split may return an error for empty data
	if err == nil {
		t.Log("empty data did not error (implementation-dependent)")
	}
}
