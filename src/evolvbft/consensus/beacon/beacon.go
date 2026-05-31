// Copyright 2024 Evolv-BFT Project
// Licensed under Apache License 2.0

package beacon

import (
	"crypto/sha256"

	"evolvbft/evolvbft/crypto"
	// "evolvbft/evolvbft/types"
	"go.dedis.ch/kyber/v3"
)

// RandomBeacon manages the generation of randomness for committee selection
type RandomBeacon struct {
	currentSeed []byte
}

// NewRandomBeacon creates a new random beacon
func NewRandomBeacon(initialSeed []byte) *RandomBeacon {
	return &RandomBeacon{
		currentSeed: initialSeed,
	}
}

// UpdateRandomness updates the seed using the aggregated signature from the previous block
func (rb *RandomBeacon) UpdateRandomness(aggregatedSig []byte) {
	// In a real system using BLS, the aggregated signature is unique and unpredictable.
	// H(prev_seed || aggregated_sig) is a good next seed.

	hasher := sha256.New()
	hasher.Write(rb.currentSeed)
	hasher.Write(aggregatedSig)
	rb.currentSeed = hasher.Sum(nil)
}

// GetCurrentSeed returns the current seed
func (rb *RandomBeacon) GetCurrentSeed() []byte {
	return rb.currentSeed
}

// AmICommitteeMember checks if the local node is selected for the committee
func (rb *RandomBeacon) AmICommitteeMember(vrfPrivKey kyber.Scalar, totalWeight uint64, committeeSize int) (bool, []byte, []byte) {
	// 1. Evaluate VRF
	result, err := crypto.EvaluateVRF(vrfPrivKey, rb.currentSeed)
	if err != nil {
		return false, nil, nil
	}

	// 2. Check Sortition
	selected := crypto.Sortition(result.Hash, totalWeight, committeeSize)

	return selected, result.Hash, result.Proof
}

// VerifyCommitteeMember checks if a remote node is a valid committee member
func (rb *RandomBeacon) VerifyCommitteeMember(vrfPubKey kyber.Point, proof []byte, totalWeight uint64, committeeSize int) bool {
	// 1. Verify VRF Proof
	hash, valid := crypto.VerifyVRF(vrfPubKey, rb.currentSeed, proof)
	if !valid {
		return false
	}

	// 2. Check Sortition
	return crypto.Sortition(hash, totalWeight, committeeSize)
}

// SelectLeader deterministically selects a leader from the validator set using
// the beacon seed, the current view, and the lane ID. All honest nodes with
// the same seed, view, and lane will select the same leader. Including laneID
// ensures that different parallel lanes elect different leaders in each view,
// enabling true multi-leader parallelism.
//
// Algorithm: H(seed || view_bytes || lane_bytes) mod len(validators) → index into sorted validator list.
func (rb *RandomBeacon) SelectLeader(view uint64, laneID uint64, validators []uint64) uint64 {
	if len(validators) == 0 {
		return 0
	}
	if len(validators) == 1 {
		return validators[0]
	}

	// Deterministic hash: SHA-256(currentSeed || big-endian view || big-endian laneID)
	hasher := sha256.New()
	hasher.Write(rb.currentSeed)
	viewBytes := make([]byte, 8)
	viewBytes[0] = byte(view >> 56)
	viewBytes[1] = byte(view >> 48)
	viewBytes[2] = byte(view >> 40)
	viewBytes[3] = byte(view >> 32)
	viewBytes[4] = byte(view >> 24)
	viewBytes[5] = byte(view >> 16)
	viewBytes[6] = byte(view >> 8)
	viewBytes[7] = byte(view)
	hasher.Write(viewBytes)
	laneBytes := make([]byte, 8)
	laneBytes[0] = byte(laneID >> 56)
	laneBytes[1] = byte(laneID >> 48)
	laneBytes[2] = byte(laneID >> 40)
	laneBytes[3] = byte(laneID >> 32)
	laneBytes[4] = byte(laneID >> 24)
	laneBytes[5] = byte(laneID >> 16)
	laneBytes[6] = byte(laneID >> 8)
	laneBytes[7] = byte(laneID)
	hasher.Write(laneBytes)
	h := hasher.Sum(nil)

	// Use first 8 bytes as uint64, mod validator count
	val := uint64(h[0])<<56 | uint64(h[1])<<48 | uint64(h[2])<<40 | uint64(h[3])<<32 |
		uint64(h[4])<<24 | uint64(h[5])<<16 | uint64(h[6])<<8 | uint64(h[7])
	index := val % uint64(len(validators))
	return validators[index]
}
