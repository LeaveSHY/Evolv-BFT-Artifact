// Copyright 2024 Evolv-BFT Project
// Licensed under Apache License 2.0

package hydra

import "errors"

// CredentialVerifier is the admission filter interface for Sybil resistance
// (§III-D, Algorithm 3). In production, implementations verify CA-issued
// hardware-bound credentials with attested device identifiers. For simulation
// and testing, use NoopCredentialVerifier which admits all join requests.
type CredentialVerifier interface {
	// VerifyCredential checks a CA-issued credential presented during a
	// join request. Returns nil if the credential is valid and the node
	// is authorized to join; returns an error describing the rejection reason
	// otherwise.
	VerifyCredential(nodeID uint64, credential []byte) error
}

// NoopCredentialVerifier admits all join requests without verification.
// Used in simulation environments where no real CA infrastructure exists.
type NoopCredentialVerifier struct{}

func (n *NoopCredentialVerifier) VerifyCredential(_ uint64, _ []byte) error {
	return nil
}

// ErrInvalidCredential is returned when credential verification fails.
var ErrInvalidCredential = errors.New("invalid or missing CA credential")
