// Package gbc implements the Global Beacon Chain metadata log and attestation protocol.
package gbc

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"sync"
)

// Log is the Global Beacon Chain append-only log recording cross-instance metadata
// among the m instance primaries (Section III-C of the paper).
//
// It records five categories of verifiable metadata:
//
//	(i)   Aggregated quorum certificates (EntryQC)
//	(ii)  Membership-change transactions (EntryMembership)
//	(iii) Epoch-anchored checkpoints (EntryCheckpoint)
//	(iv)  MARL policy updates (EntryPolicyUpdate)
//	(v)   Cross-instance trust reports (EntryTrust) — Limitation (i) defense
//
// Properties:
//
//	G1: Append-only integrity (enforced by contiguous height check)
//	G2: Honest-primary agreement on common prefix (single-writer per height)
//	G3: Bounded retrieval delay (caller responsibility after GST)
//	G4: >= 2f_GBC + 1 QC verifiability on every entry (Attestation quorum)
//
// Thread-safe for concurrent Publish/Retrieve/Attest calls.
type Log struct {
	mu           sync.RWMutex
	entries      map[uint64]Entry
	latestByType map[EntryType]Entry
	nextHeight   uint64
	numMembers   int // number of GBC members (m instance primaries)
	verifier     AttestationVerifier
}

// AttestationVerifier verifies one signer over an entry digest.
type AttestationVerifier func(signerID uint64, digest [32]byte, sig []byte) bool

// NewLog constructs an empty GBC log. numMembers is the number of instance primaries
// participating in attestation (m). The first valid publish is at height 1.
func NewLog() *Log {
	return NewLogWithMembers(0)
}

// NewLogWithMembers constructs a GBC log with a known member count for quorum checks.
func NewLogWithMembers(numMembers int) *Log {
	return &Log{
		entries:      make(map[uint64]Entry),
		latestByType: make(map[EntryType]Entry),
		nextHeight:   1,
		numMembers:   numMembers,
	}
}

// SetNumMembers updates the GBC member count (e.g., after a primary rotation).
func (l *Log) SetNumMembers(n int) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.numMembers = n
}

// SetVerifier installs the signature verifier used by Publish, Attest, and
// VerifyEntry. A nil verifier preserves append-only behavior for legacy tests.
func (l *Log) SetVerifier(verifier AttestationVerifier) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.verifier = verifier
}

// NumMembers returns the current GBC member count.
func (l *Log) NumMembers() int {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.numMembers
}

// Publish appends an entry if the height is contiguous (G1: append-only).
func (l *Log) Publish(entry Entry) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	if entry.Height != l.nextHeight {
		return fmt.Errorf("gbc: expected height %d, got %d (G1 violation)", l.nextHeight, entry.Height)
	}
	if err := verifyEntryWith(l.numMembers, l.verifier, entry, false); err != nil {
		return err
	}

	stored := Entry{
		Height:       entry.Height,
		Type:         entry.Type,
		Payload:      append([]byte(nil), entry.Payload...),
		Attestations: copyAttestations(entry.Attestations),
	}
	l.entries[stored.Height] = stored
	l.latestByType[stored.Type] = stored
	l.nextHeight++
	return nil
}

// Attest adds an attestation to an existing entry at the given height.
// Returns true if the entry now has a quorum (G4).
func (l *Log) Attest(height uint64, att Attestation) (bool, error) {
	l.mu.Lock()
	defer l.mu.Unlock()

	entry, ok := l.entries[height]
	if !ok {
		return false, fmt.Errorf("gbc: no entry at height %d", height)
	}
	if len(att.Signature) == 0 {
		return false, fmt.Errorf("gbc: empty signature from signer %d", att.SignerID)
	}
	if err := verifyAttestation(l.verifier, entry, att); err != nil {
		return false, err
	}

	// Deduplicate: one attestation per signer
	for _, existing := range entry.Attestations {
		if existing.SignerID == att.SignerID {
			// Already attested by this signer
			hasQ := entry.HasQuorum(l.numMembers)
			return hasQ, nil
		}
	}

	entry.Attestations = append(entry.Attestations, Attestation{
		SignerID:  att.SignerID,
		Signature: append([]byte(nil), att.Signature...),
	})
	l.entries[height] = entry
	l.latestByType[entry.Type] = entry

	hasQuorum := entry.HasQuorum(l.numMembers)
	return hasQuorum, nil
}

// VerifyEntry checks that all attestations on entry are valid and, when
// requireQuorum is true, that they form the G4 quorum certificate.
func (l *Log) VerifyEntry(entry Entry, requireQuorum bool) error {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return verifyEntryWith(l.numMembers, l.verifier, entry, requireQuorum)
}

func verifyEntryWith(numMembers int, verifier AttestationVerifier, entry Entry, requireQuorum bool) error {
	seen := make(map[uint64]struct{}, len(entry.Attestations))
	for _, att := range entry.Attestations {
		if _, exists := seen[att.SignerID]; exists {
			return fmt.Errorf("gbc: duplicate attestation from signer %d", att.SignerID)
		}
		seen[att.SignerID] = struct{}{}
		if err := verifyAttestation(verifier, entry, att); err != nil {
			return err
		}
	}
	if requireQuorum && !entry.HasQuorum(numMembers) {
		return fmt.Errorf("gbc: entry height %d has %d attestations, quorum requires %d", entry.Height, len(seen), QuorumSize(numMembers))
	}
	return nil
}

func verifyAttestation(verifier AttestationVerifier, entry Entry, att Attestation) error {
	if len(att.Signature) == 0 {
		return fmt.Errorf("gbc: empty signature from signer %d", att.SignerID)
	}
	if verifier == nil {
		return nil
	}
	digest := EntryDigest(entry.Height, entry.Type, entry.Payload)
	if !verifier(att.SignerID, digest, att.Signature) {
		return fmt.Errorf("gbc: invalid signature from signer %d at height %d", att.SignerID, entry.Height)
	}
	return nil
}

// Retrieve returns the entry at the requested height.
func (l *Log) Retrieve(height uint64) (Entry, bool) {
	l.mu.RLock()
	defer l.mu.RUnlock()

	entry, ok := l.entries[height]
	if !ok {
		return Entry{}, false
	}
	return cloneEntry(entry), true
}

// LatestByType returns the latest entry for a given type.
func (l *Log) LatestByType(entryType EntryType) (Entry, bool) {
	l.mu.RLock()
	defer l.mu.RUnlock()

	entry, ok := l.latestByType[entryType]
	if !ok {
		return Entry{}, false
	}
	return cloneEntry(entry), true
}

// Height returns the next expected height (current log length + 1).
func (l *Log) Height() uint64 {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.nextHeight
}

// EntryDigest computes the canonical digest of an entry for signature verification.
// digest = sha256(height || type || sha256(payload))
func EntryDigest(height uint64, entryType EntryType, payload []byte) [32]byte {
	var buf [8]byte
	binary.BigEndian.PutUint64(buf[:], height)
	payloadHash := sha256.Sum256(payload)

	h := sha256.New()
	h.Write(buf[:])
	h.Write([]byte(entryType))
	h.Write(payloadHash[:])

	var digest [32]byte
	copy(digest[:], h.Sum(nil))
	return digest
}

func cloneEntry(entry Entry) Entry {
	return Entry{
		Height:       entry.Height,
		Type:         entry.Type,
		Payload:      append([]byte(nil), entry.Payload...),
		Attestations: copyAttestations(entry.Attestations),
	}
}

func copyAttestations(atts []Attestation) []Attestation {
	if len(atts) == 0 {
		return nil
	}
	out := make([]Attestation, len(atts))
	for i, a := range atts {
		out[i] = Attestation{
			SignerID:  a.SignerID,
			Signature: append([]byte(nil), a.Signature...),
		}
	}
	return out
}
