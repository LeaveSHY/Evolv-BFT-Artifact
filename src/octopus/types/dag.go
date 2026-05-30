// Copyright 2024 Octopus Project
// Licensed under Apache License 2.0

package types

import (
	"crypto/sha256"
	"encoding/binary"
	"sync"
)

// Vertex represents a node in the Mempool DAG
type Vertex struct {
	mu sync.RWMutex

	// Metadata
	Epoch  uint64
	Round  uint64
	Author uint64

	// Payload
	Txs []*Transaction

	// DAG Links (References to vertices in Round-1)
	Parents []Hash

	// Authenticity
	Signature Signature
	Hash      Hash
}

// NewVertex creates a new vertex
func NewVertex(epoch, round, author uint64, txs []*Transaction, parents []Hash) *Vertex {
	v := &Vertex{
		Epoch:   epoch,
		Round:   round,
		Author:  author,
		Txs:     txs,
		Parents: parents,
	}
	v.Hash = v.computeHash()
	return v
}

// computeHash computes SHA256 over (epoch, round, author, parents, tx hashes).
func (v *Vertex) computeHash() Hash {
	h := sha256.New()
	buf := make([]byte, 8)
	binary.BigEndian.PutUint64(buf, v.Epoch)
	h.Write(buf)
	binary.BigEndian.PutUint64(buf, v.Round)
	h.Write(buf)
	binary.BigEndian.PutUint64(buf, v.Author)
	h.Write(buf)
	for _, p := range v.Parents {
		h.Write(p[:])
	}
	for _, tx := range v.Txs {
		txHash := sha256.Sum256(tx.Payload)
		h.Write(txHash[:])
	}
	var out Hash
	copy(out[:], h.Sum(nil))
	return out
}

// GetHash returns the vertex hash
func (v *Vertex) GetHash() Hash {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.Hash
}

// VertexCertificate represents 2f+1 signatures for a vertex
type VertexCertificate struct {
	mu sync.RWMutex

	VertexHash Hash
	Epoch      uint64
	Round      uint64

	// Signatures from validators
	Signatures map[uint64]Signature
}

// NewVertexCertificate creates a new certificate
func NewVertexCertificate(vertexHash Hash, epoch, round uint64) *VertexCertificate {
	return &VertexCertificate{
		VertexHash: vertexHash,
		Epoch:      epoch,
		Round:      round,
		Signatures: make(map[uint64]Signature),
	}
}

// AddSignature adds a signature to the certificate
func (vc *VertexCertificate) AddSignature(validatorID uint64, sig Signature) {
	vc.mu.Lock()
	defer vc.mu.Unlock()
	vc.Signatures[validatorID] = sig
}

// GetSignatures returns the signatures
func (vc *VertexCertificate) GetSignatures() map[uint64]Signature {
	vc.mu.RLock()
	defer vc.mu.RUnlock()
	return vc.Signatures
}
