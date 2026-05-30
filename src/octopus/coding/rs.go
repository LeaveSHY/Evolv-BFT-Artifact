// Copyright 2024 Octopus Project
// Licensed under Apache License 2.0

package coding

import (
	"fmt"
	"github.com/klauspost/reedsolomon"
)

// Encoder wraps Reed-Solomon encoding
type Encoder struct {
	DataShards   int
	ParityShards int
	enc          reedsolomon.Encoder
}

// NewEncoder creates a new Reed-Solomon encoder
func NewEncoder(dataShards, parityShards int) (*Encoder, error) {
	enc, err := reedsolomon.New(dataShards, parityShards)
	if err != nil {
		return nil, err
	}
	return &Encoder{
		DataShards:   dataShards,
		ParityShards: parityShards,
		enc:          enc,
	}, nil
}

// Encode splits data into shards and adds parity
func (e *Encoder) Encode(data []byte) ([][]byte, error) {
	// Split data into equal-sized shards
	shards, err := e.enc.Split(data)
	if err != nil {
		return nil, err
	}
	
	// Encode parity
	if err := e.enc.Encode(shards); err != nil {
		return nil, err
	}
	
	return shards, nil
}

// Reconstruct recovers the original data from a subset of shards
// The input 'shards' must have length DataShards + ParityShards
// Missing shards should be nil
func (e *Encoder) Reconstruct(shards [][]byte) ([]byte, error) {
	// Verify we have enough shards
	ok, err := e.enc.Verify(shards)
	if !ok || err != nil {
		// Try to reconstruct
		if err := e.enc.Reconstruct(shards); err != nil {
			return nil, err
		}
		// Verify again
		if ok, err := e.enc.Verify(shards); !ok || err != nil {
			return nil, fmt.Errorf("reconstruction failed: %v", err)
		}
	}
	
	// Join shards back to data
	var buf []byte
	// Join only data shards
	for i := 0; i < e.DataShards; i++ {
		buf = append(buf, shards[i]...)
	}
	
	return buf, nil
}
