// Copyright 2024 Evolv-BFT Project
// Licensed under Apache License 2.0

package crypto

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"

	"evolvbft/evolvbft/types"
)

// Keypair represents a node's key pair
type Keypair struct {
	PublicKey  types.PublicKey
	PrivateKey types.PrivateKey
}

// Hash computes SHA256 hash
func SHA256(data []byte) types.Hash {
	return sha256.Sum256(data)
}

// HashString computes SHA256 hash and returns hex string
func HashString(data []byte) string {
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:])
}

// Sign signs data with a private key using Ed25519
func Sign(data []byte, key types.PrivateKey) types.Signature {
	return ed25519.Sign(key, data)
}

// Verify verifies a signature using Ed25519
func Verify(data []byte, sig types.Signature, key types.PublicKey) bool {
	return ed25519.Verify(key, data, sig)
}

// GenerateKeyPair generates a new Ed25519 key pair
func GenerateKeyPair() (*Keypair, error) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return nil, err
	}
	return &Keypair{
		PublicKey:  pub,
		PrivateKey: priv,
	}, nil
}

// VerifyQuorum checks if a QC has enough valid signatures
// validators is a map of ValidatorID -> PublicKey
func VerifyQuorum(qc *types.QuorumCertificate, validators map[uint64]types.PublicKey, quorumSize int) bool {
	if qc == nil {
		return false
	}

	validSignatures := 0
	msg := types.QCSigningBytes(qc.BlockHash, qc.View, qc.Epoch, qc.ConfigID, qc.Lane)

	for validatorID, sig := range qc.Signatures {
		pubKey, exists := validators[validatorID]
		if !exists {
			continue
		}

		if ed25519.Verify(pubKey, msg, sig) {
			validSignatures++
		}
	}

	return validSignatures >= quorumSize
}

// VerifyQuorumBatch verifies QC signatures in parallel using goroutines.
// At 1000 nodes, sequential verification of ~1000 Ed25519 signatures takes
// ~50ms. Parallel verification with GOMAXPROCS workers reduces this to ~5ms.
// Returns true if at least quorumSize valid signatures are found.
func VerifyQuorumBatch(qc *types.QuorumCertificate, validators map[uint64]types.PublicKey, quorumSize int) bool {
	if qc == nil || len(qc.Signatures) == 0 {
		return false
	}

	msg := types.QCSigningBytes(qc.BlockHash, qc.View, qc.Epoch, qc.ConfigID, qc.Lane)

	type verifyResult struct {
		valid bool
	}

	results := make(chan verifyResult, len(qc.Signatures))

	for validatorID, sig := range qc.Signatures {
		pubKey, exists := validators[validatorID]
		if !exists {
			results <- verifyResult{valid: false}
			continue
		}
		go func(pk types.PublicKey, s []byte) {
			results <- verifyResult{valid: ed25519.Verify(pk, msg, s)}
		}(pubKey, sig)
	}

	validCount := 0
	for i := 0; i < len(qc.Signatures); i++ {
		r := <-results
		if r.valid {
			validCount++
			// Early exit: if we already have enough valid sigs, don't wait for the rest
			if validCount >= quorumSize {
				return true
			}
		}
	}

	return validCount >= quorumSize
}
