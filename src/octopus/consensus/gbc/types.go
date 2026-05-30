package gbc

// EntryType identifies the semantic class carried by the GBC log.
type EntryType string

const (
	EntryQC           EntryType = "qc"
	EntryCheckpoint   EntryType = "checkpoint"
	EntryPolicyUpdate EntryType = "policy_update"
	EntryMembership   EntryType = "membership"
	EntryTrust        EntryType = "trust" // cross-instance trust reports (Limitation i defense)
)

// Attestation is a signature from one GBC member (instance primary) over an entry digest.
type Attestation struct {
	SignerID  uint64 // instance primary ID
	Signature []byte // Ed25519 or threshold signature over (Height || Type || sha256(Payload))
}

// Entry is one record in the GBC append-only log.
// When Attestations contains >= 2f_GBC + 1 valid signatures, property G4 holds.
type Entry struct {
	Height       uint64
	Type         EntryType
	Payload      []byte
	Attestations []Attestation // multi-signature attestation set (G4)
}

// QuorumSize returns the minimum number of attestations needed for G4 (2f+1 out of n members).
func QuorumSize(numMembers int) int {
	f := (numMembers - 1) / 3
	return 2*f + 1
}

// HasQuorum returns true if the entry carries at least QuorumSize(numMembers) distinct attestations.
func (e Entry) HasQuorum(numMembers int) bool {
	needed := QuorumSize(numMembers)
	if len(e.Attestations) < needed {
		return false
	}
	// Count distinct signers
	seen := make(map[uint64]bool, len(e.Attestations))
	for _, att := range e.Attestations {
		seen[att.SignerID] = true
	}
	return len(seen) >= needed
}
