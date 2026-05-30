// Copyright 2024 Octopus Project
// Licensed under Apache License 2.0

package types

import (
	"crypto/ed25519"
)

// Vote represents a vote for a block
type Vote struct {
	BlockID   Hash
	View      uint64
	Epoch     uint64
	ConfigID  uint64
	Lane      uint64
	Author    PublicKey
	Signature Signature

	// Phase 5: VRF Proof for Committee Eligibility
	VRFProof []byte
}

// NewVote creates a new vote
func NewVote(blockID Hash, view uint64, author PublicKey, priv PrivateKey, vrfProof []byte) (*Vote, error) {
	return NewVoteWithIdentity(blockID, view, 0, 0, 0, author, priv, vrfProof)
}

func NewVoteWithIdentity(blockID Hash, view uint64, epoch uint64, configID uint64, lane uint64, author PublicKey, priv PrivateKey, vrfProof []byte) (*Vote, error) {
	msg := VoteSigningBytes(blockID, view, epoch, configID, lane)
	sig := ed25519.Sign(priv, msg)
	return &Vote{
		BlockID:   blockID,
		View:      view,
		Epoch:     epoch,
		ConfigID:  configID,
		Lane:      lane,
		Author:    author,
		Signature: sig,
		VRFProof:  vrfProof,
	}, nil
}

// Verify verifies the vote signature
func (v *Vote) Verify() bool {
	msg := VoteSigningBytes(v.BlockID, v.View, v.Epoch, v.ConfigID, v.Lane)
	return ed25519.Verify(v.Author, msg, v.Signature)
}
