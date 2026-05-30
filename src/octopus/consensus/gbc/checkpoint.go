package gbc

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"fmt"
)

// Checkpoint records the minimal committed metadata currently published into the local GBC scaffold.
// It reflects only local checkpoint sink semantics, not a paper-grade global beacon chain checkpoint proof.
type Checkpoint struct {
	InstanceID      uint64 `json:"instance_id"`
	LocalHeight     uint64 `json:"local_height"`
	Rank            uint64 `json:"rank"`
	Epoch           uint64 `json:"epoch"`
	IsNil           bool   `json:"is_nil"`
	TransitionCount int    `json:"transition_count"`
	BlockHashHex    string `json:"block_hash_hex"`
}

// GetLatestCheckpoint decodes the latest locally stored checkpoint entry if one exists.
func GetLatestCheckpoint(log *Log) (Checkpoint, bool, error) {
	if log == nil {
		return Checkpoint{}, false, nil
	}
	entry, ok := log.LatestByType(EntryCheckpoint)
	if !ok {
		return Checkpoint{}, false, nil
	}
	var checkpoint Checkpoint
	if err := json.Unmarshal(entry.Payload, &checkpoint); err != nil {
		return Checkpoint{}, false, fmt.Errorf("gbc: decode latest checkpoint: %w", err)
	}
	if _, err := decodeCheckpointHash(checkpoint.BlockHashHex); err != nil {
		return Checkpoint{}, true, err
	}
	return checkpoint, true, nil
}

// ValidateLatestCheckpointAgainst ensures the latest locally stored checkpoint matches expected committed metadata.
func ValidateLatestCheckpointAgainst(log *Log, expected Checkpoint) error {
	latest, ok, err := GetLatestCheckpoint(log)
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("gbc: latest checkpoint not found")
	}
	return compareCheckpoint(latest, expected)
}

func compareCheckpoint(actual, expected Checkpoint) error {
	actualHash, err := decodeCheckpointHash(actual.BlockHashHex)
	if err != nil {
		return err
	}
	expectedHash, err := decodeCheckpointHash(expected.BlockHashHex)
	if err != nil {
		return err
	}
	if actual.InstanceID != expected.InstanceID {
		return fmt.Errorf("gbc: checkpoint instance_id mismatch: got %d want %d", actual.InstanceID, expected.InstanceID)
	}
	if actual.LocalHeight != expected.LocalHeight {
		return fmt.Errorf("gbc: checkpoint local_height mismatch: got %d want %d", actual.LocalHeight, expected.LocalHeight)
	}
	if actual.Rank != expected.Rank {
		return fmt.Errorf("gbc: checkpoint rank mismatch: got %d want %d", actual.Rank, expected.Rank)
	}
	if actual.Epoch != expected.Epoch {
		return fmt.Errorf("gbc: checkpoint epoch mismatch: got %d want %d", actual.Epoch, expected.Epoch)
	}
	if actual.IsNil != expected.IsNil {
		return fmt.Errorf("gbc: checkpoint is_nil mismatch: got %t want %t", actual.IsNil, expected.IsNil)
	}
	if actual.TransitionCount != expected.TransitionCount {
		return fmt.Errorf("gbc: checkpoint transition_count mismatch: got %d want %d", actual.TransitionCount, expected.TransitionCount)
	}
	if !bytes.Equal(actualHash, expectedHash) {
		return fmt.Errorf("gbc: checkpoint block_hash mismatch: got %s want %s", actual.BlockHashHex, expected.BlockHashHex)
	}
	return nil
}

func decodeCheckpointHash(hashHex string) ([]byte, error) {
	if hashHex == "" {
		return nil, fmt.Errorf("gbc: checkpoint block_hash_hex is empty")
	}
	decoded, err := hex.DecodeString(hashHex)
	if err != nil {
		return nil, fmt.Errorf("gbc: decode checkpoint block_hash_hex: %w", err)
	}
	return decoded, nil
}
