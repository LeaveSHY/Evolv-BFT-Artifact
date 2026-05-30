// Copyright 2024 Octopus Project
// Licensed under Apache License 2.0

package beacon

import (
	"fmt"
	"sort"
	"sync"

	"octopus-bft/octopus/crypto"

	"go.dedis.ch/kyber/v3"
)

// BLSBeacon extends the random beacon with real BLS aggregate signatures.
// It collects per-validator BLS signature shares on QC messages and produces
// a verifiable aggregate signature that serves as the beacon seed input.
//
// This replaces the SHA256 concat approach in engine.aggregateSignatures()
// with cryptographically sound BLS aggregation, matching the paper's
// "aggregated quorum certificate signatures" description.
type BLSBeacon struct {
	mu   sync.RWMutex
	bls  *crypto.BLSFull
	keys map[uint64]BLSKeyPair // validatorID -> key pair
}

// BLSKeyPair holds a validator's BLS key pair for beacon participation.
type BLSKeyPair struct {
	Private kyber.Scalar
	Public  kyber.Point
}

// NewBLSBeacon creates a BLS beacon module.
func NewBLSBeacon() *BLSBeacon {
	return &BLSBeacon{
		bls:  crypto.NewBLSFull(),
		keys: make(map[uint64]BLSKeyPair),
	}
}

// GenerateKey generates and registers a BLS key pair for the given validator.
func (b *BLSBeacon) GenerateKey(validatorID uint64) (BLSKeyPair, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	priv, pub, err := b.bls.GenerateKeyPair()
	if err != nil {
		return BLSKeyPair{}, fmt.Errorf("bls keygen for validator %d: %w", validatorID, err)
	}
	kp := BLSKeyPair{Private: priv, Public: pub}
	b.keys[validatorID] = kp
	return kp, nil
}

// RegisterKey registers an externally generated BLS public key for a validator.
func (b *BLSBeacon) RegisterKey(validatorID uint64, pub kyber.Point) {
	b.mu.Lock()
	defer b.mu.Unlock()
	existing, ok := b.keys[validatorID]
	if ok {
		existing.Public = pub
		b.keys[validatorID] = existing
	} else {
		b.keys[validatorID] = BLSKeyPair{Public: pub}
	}
}

// Sign creates a BLS signature share on the given message using the validator's private key.
func (b *BLSBeacon) Sign(validatorID uint64, message []byte) ([]byte, error) {
	b.mu.RLock()
	kp, ok := b.keys[validatorID]
	b.mu.RUnlock()
	if !ok || kp.Private == nil {
		return nil, fmt.Errorf("no BLS private key for validator %d", validatorID)
	}
	return b.bls.Sign(kp.Private, message)
}

// AggregateAndVerify aggregates BLS signature shares from the QC signers and
// verifies the aggregate against their public keys. Returns the aggregated
// signature bytes for use as random beacon seed.
func (b *BLSBeacon) AggregateAndVerify(message []byte, signerSigs map[uint64][]byte) ([]byte, error) {
	if len(signerSigs) == 0 {
		return nil, fmt.Errorf("no signatures to aggregate")
	}

	b.mu.RLock()
	defer b.mu.RUnlock()

	// Sort signer IDs for deterministic ordering
	ids := make([]uint64, 0, len(signerSigs))
	for id := range signerSigs {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })

	sigs := make([][]byte, 0, len(ids))
	pubKeys := make([]kyber.Point, 0, len(ids))
	for _, id := range ids {
		kp, ok := b.keys[id]
		if !ok || kp.Public == nil {
			continue // skip validators without registered BLS keys
		}
		sigs = append(sigs, signerSigs[id])
		pubKeys = append(pubKeys, kp.Public)
	}

	if len(sigs) == 0 {
		return nil, fmt.Errorf("no valid BLS signatures found")
	}

	// Aggregate signatures
	aggSig, err := b.bls.AggregateSignatures(sigs...)
	if err != nil {
		return nil, fmt.Errorf("bls aggregate: %w", err)
	}

	// Verify the aggregate
	if err := b.bls.VerifyAggregate(pubKeys, message, aggSig); err != nil {
		return nil, fmt.Errorf("bls aggregate verify: %w", err)
	}

	return aggSig, nil
}

// HasKey returns whether a validator has a registered BLS key.
func (b *BLSBeacon) HasKey(validatorID uint64) bool {
	b.mu.RLock()
	defer b.mu.RUnlock()
	_, ok := b.keys[validatorID]
	return ok
}

// GetPublicKey returns a validator's BLS public key.
func (b *BLSBeacon) GetPublicKey(validatorID uint64) (kyber.Point, bool) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	kp, ok := b.keys[validatorID]
	if !ok {
		return nil, false
	}
	return kp.Public, true
}

// RetainKeys removes BLS keys for validators not in the given set.
func (b *BLSBeacon) RetainKeys(activeIDs map[uint64]bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	for id := range b.keys {
		if !activeIDs[id] {
			delete(b.keys, id)
		}
	}
}
