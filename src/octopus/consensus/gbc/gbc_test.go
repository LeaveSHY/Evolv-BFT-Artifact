package gbc

import (
	"encoding/hex"
	"encoding/json"
	"testing"
)

func TestLogPublishesAndRetrievesEntries(t *testing.T) {
	log := NewLog()
	entry := Entry{Height: 1, Type: EntryQC, Payload: []byte("qc")}
	if err := log.Publish(entry); err != nil {
		t.Fatalf("publish failed: %v", err)
	}
	got, ok := log.Retrieve(1)
	if !ok {
		t.Fatalf("expected entry at height 1")
	}
	if got.Type != EntryQC || string(got.Payload) != "qc" {
		t.Fatalf("unexpected entry: %+v", got)
	}
}

func TestLogRejectsOutOfOrderHeight(t *testing.T) {
	log := NewLog()
	if err := log.Publish(Entry{Height: 2, Type: EntryQC}); err == nil {
		t.Fatalf("expected out-of-order publish to fail")
	}
}

func TestLogStoresLatestPolicyUpdate(t *testing.T) {
	log := NewLog()
	_ = log.Publish(Entry{Height: 1, Type: EntryPolicyUpdate, Payload: []byte("policy-v1")})
	got, ok := log.LatestByType(EntryPolicyUpdate)
	if !ok {
		t.Fatalf("expected policy update")
	}
	if string(got.Payload) != "policy-v1" {
		t.Fatalf("unexpected payload: %s", string(got.Payload))
	}
}

type checkpointRecord struct {
	InstanceID      uint64 `json:"instance_id"`
	LocalHeight     uint64 `json:"local_height"`
	Rank            uint64 `json:"rank"`
	Epoch           uint64 `json:"epoch"`
	IsNil           bool   `json:"is_nil"`
	TransitionCount int    `json:"transition_count"`
	BlockHashHex    string `json:"block_hash_hex"`
}

func TestGetLatestCheckpointReadsCheckpointPayload(t *testing.T) {
	log := NewLog()
	payload, err := json.Marshal(checkpointRecord{
		InstanceID:      1,
		LocalHeight:     7,
		Rank:            13,
		Epoch:           4,
		IsNil:           false,
		TransitionCount: 2,
		BlockHashHex:    hex.EncodeToString([]byte("block-a")),
	})
	if err != nil {
		t.Fatalf("marshal checkpoint payload: %v", err)
	}
	if err := log.Publish(Entry{Height: 1, Type: EntryCheckpoint, Payload: payload}); err != nil {
		t.Fatalf("publish checkpoint entry: %v", err)
	}

	record, ok, err := GetLatestCheckpoint(log)
	if err != nil {
		t.Fatalf("get latest checkpoint: %v", err)
	}
	if !ok {
		t.Fatal("expected checkpoint record")
	}
	if record.InstanceID != 1 || record.LocalHeight != 7 || record.Rank != 13 || record.Epoch != 4 {
		t.Fatalf("unexpected checkpoint record: %+v", record)
	}
	if record.IsNil {
		t.Fatalf("unexpected nil checkpoint record: %+v", record)
	}
	if record.TransitionCount != 2 {
		t.Fatalf("unexpected transition count: %+v", record)
	}
}

func TestGetLatestCheckpointReturnsErrorForInvalidPayload(t *testing.T) {
	log := NewLog()
	if err := log.Publish(Entry{Height: 1, Type: EntryCheckpoint, Payload: []byte("not-json")}); err != nil {
		t.Fatalf("publish checkpoint entry: %v", err)
	}

	if _, ok, err := GetLatestCheckpoint(log); err == nil {
		t.Fatal("expected invalid payload error")
	} else if ok {
		t.Fatal("expected malformed JSON payload not to decode into checkpoint")
	}
}

func TestGetLatestCheckpointReturnsErrorForInvalidBlockHashHex(t *testing.T) {
	log := NewLog()
	payload, err := json.Marshal(checkpointRecord{InstanceID: 1, LocalHeight: 7, Rank: 13, Epoch: 4, BlockHashHex: "zz"})
	if err != nil {
		t.Fatalf("marshal checkpoint payload: %v", err)
	}
	if err := log.Publish(Entry{Height: 1, Type: EntryCheckpoint, Payload: payload}); err != nil {
		t.Fatalf("publish checkpoint entry: %v", err)
	}

	if _, ok, err := GetLatestCheckpoint(log); err == nil {
		t.Fatal("expected invalid block hash error")
	} else if !ok {
		t.Fatal("expected invalid checkpoint to report latest checkpoint present")
	}
}

func TestGetLatestCheckpointReturnsErrorForEmptyBlockHash(t *testing.T) {
	log := NewLog()
	payload, err := json.Marshal(checkpointRecord{InstanceID: 1, LocalHeight: 7, Rank: 13, Epoch: 4})
	if err != nil {
		t.Fatalf("marshal checkpoint payload: %v", err)
	}
	if err := log.Publish(Entry{Height: 1, Type: EntryCheckpoint, Payload: payload}); err != nil {
		t.Fatalf("publish checkpoint entry: %v", err)
	}

	if _, ok, err := GetLatestCheckpoint(log); err == nil {
		t.Fatal("expected empty block hash error")
	} else if !ok {
		t.Fatal("expected invalid checkpoint to report latest checkpoint present")
	}
}

func TestGetLatestCheckpointReturnsNotFoundForNilLog(t *testing.T) {
	if _, ok, err := GetLatestCheckpoint(nil); err != nil {
		t.Fatalf("get latest checkpoint: %v", err)
	} else if ok {
		t.Fatal("expected nil log to return not found")
	}
}

func TestGetLatestCheckpointReturnsNotFoundWithoutCheckpoint(t *testing.T) {
	log := NewLog()
	if _, ok, err := GetLatestCheckpoint(log); err != nil {
		t.Fatalf("get latest checkpoint: %v", err)
	} else if ok {
		t.Fatal("expected no checkpoint record")
	}
}

func TestValidateLatestCheckpointAgainstReturnsErrorForNilLog(t *testing.T) {
	if err := ValidateLatestCheckpointAgainst(nil, Checkpoint{BlockHashHex: hex.EncodeToString([]byte("block-a"))}); err == nil {
		t.Fatal("expected nil log to fail validation")
	}
}

func TestValidateLatestCheckpointAgainstReturnsErrorWithoutCheckpoint(t *testing.T) {
	log := NewLog()
	if err := ValidateLatestCheckpointAgainst(log, Checkpoint{BlockHashHex: hex.EncodeToString([]byte("block-a"))}); err == nil {
		t.Fatal("expected missing checkpoint error")
	}
}

func TestGetLatestCheckpointReturnsLatestCheckpoint(t *testing.T) {
	log := NewLog()
	firstPayload, err := json.Marshal(checkpointRecord{InstanceID: 1, LocalHeight: 7, Rank: 13, Epoch: 4, BlockHashHex: hex.EncodeToString([]byte("block-a"))})
	if err != nil {
		t.Fatalf("marshal first checkpoint payload: %v", err)
	}
	secondPayload, err := json.Marshal(checkpointRecord{InstanceID: 1, LocalHeight: 8, Rank: 14, Epoch: 4, BlockHashHex: hex.EncodeToString([]byte("block-b"))})
	if err != nil {
		t.Fatalf("marshal second checkpoint payload: %v", err)
	}
	if err := log.Publish(Entry{Height: 1, Type: EntryCheckpoint, Payload: firstPayload}); err != nil {
		t.Fatalf("publish first checkpoint entry: %v", err)
	}
	if err := log.Publish(Entry{Height: 2, Type: EntryCheckpoint, Payload: secondPayload}); err != nil {
		t.Fatalf("publish second checkpoint entry: %v", err)
	}

	record, ok, err := GetLatestCheckpoint(log)
	if err != nil {
		t.Fatalf("get latest checkpoint: %v", err)
	}
	if !ok {
		t.Fatal("expected latest checkpoint record")
	}
	if record.LocalHeight != 8 || record.Rank != 14 {
		t.Fatalf("expected latest checkpoint record, got %+v", record)
	}
}

func TestValidateLatestCheckpointAgainstMatchesCommittedHead(t *testing.T) {
	log := NewLog()
	record := checkpointRecord{
		InstanceID:      2,
		LocalHeight:     9,
		Rank:            21,
		Epoch:           6,
		IsNil:           false,
		TransitionCount: 1,
		BlockHashHex:    hex.EncodeToString([]byte("block-c")),
	}
	payload, err := json.Marshal(record)
	if err != nil {
		t.Fatalf("marshal checkpoint payload: %v", err)
	}
	if err := log.Publish(Entry{Height: 1, Type: EntryCheckpoint, Payload: payload}); err != nil {
		t.Fatalf("publish checkpoint entry: %v", err)
	}
	if err := ValidateLatestCheckpointAgainst(log, Checkpoint(record)); err != nil {
		t.Fatalf("validate latest checkpoint: %v", err)
	}
}

func TestValidateLatestCheckpointAgainstRejectsMismatches(t *testing.T) {
	log := NewLog()
	record := checkpointRecord{
		InstanceID:      2,
		LocalHeight:     9,
		Rank:            21,
		Epoch:           6,
		IsNil:           false,
		TransitionCount: 1,
		BlockHashHex:    hex.EncodeToString([]byte("block-c")),
	}
	payload, err := json.Marshal(record)
	if err != nil {
		t.Fatalf("marshal checkpoint payload: %v", err)
	}
	if err := log.Publish(Entry{Height: 1, Type: EntryCheckpoint, Payload: payload}); err != nil {
		t.Fatalf("publish checkpoint entry: %v", err)
	}

	cases := []checkpointRecord{
		{InstanceID: 99, LocalHeight: 9, Rank: 21, Epoch: 6, IsNil: false, TransitionCount: 1, BlockHashHex: hex.EncodeToString([]byte("block-c"))},
		{InstanceID: 2, LocalHeight: 9, Rank: 21, Epoch: 6, IsNil: true, TransitionCount: 1, BlockHashHex: hex.EncodeToString([]byte("block-c"))},
		{InstanceID: 2, LocalHeight: 9, Rank: 21, Epoch: 6, IsNil: false, TransitionCount: 9, BlockHashHex: hex.EncodeToString([]byte("block-c"))},
		{InstanceID: 2, LocalHeight: 9, Rank: 21, Epoch: 6, IsNil: false, TransitionCount: 1, BlockHashHex: hex.EncodeToString([]byte("other"))},
		{InstanceID: 2, LocalHeight: 9, Rank: 99, Epoch: 6, IsNil: false, TransitionCount: 1, BlockHashHex: hex.EncodeToString([]byte("block-c"))},
		{InstanceID: 2, LocalHeight: 77, Rank: 21, Epoch: 6, IsNil: false, TransitionCount: 1, BlockHashHex: hex.EncodeToString([]byte("block-c"))},
		{InstanceID: 2, LocalHeight: 9, Rank: 21, Epoch: 99, IsNil: false, TransitionCount: 1, BlockHashHex: hex.EncodeToString([]byte("block-c"))},
	}
	for _, candidate := range cases {
		if err := ValidateLatestCheckpointAgainst(log, Checkpoint(candidate)); err == nil {
			t.Fatalf("expected mismatch validation to fail for %+v", candidate)
		}
	}
}

func TestValidateLatestCheckpointAgainstReturnsErrorForInvalidExpectedBlockHashHex(t *testing.T) {
	log := NewLog()
	record := checkpointRecord{InstanceID: 2, LocalHeight: 9, Rank: 21, Epoch: 6, BlockHashHex: hex.EncodeToString([]byte("block-c"))}
	payload, err := json.Marshal(record)
	if err != nil {
		t.Fatalf("marshal checkpoint payload: %v", err)
	}
	if err := log.Publish(Entry{Height: 1, Type: EntryCheckpoint, Payload: payload}); err != nil {
		t.Fatalf("publish checkpoint entry: %v", err)
	}

	if err := ValidateLatestCheckpointAgainst(log, Checkpoint{InstanceID: 2, LocalHeight: 9, Rank: 21, Epoch: 6, BlockHashHex: "zz"}); err == nil {
		t.Fatal("expected invalid expected block hash error")
	}
}

// ─── Attestation and quorum tests (G4) ────────────────────────────────────

func TestQuorumSizeCalculation(t *testing.T) {
	cases := []struct {
		members  int
		expected int
	}{
		{1, 1},  // f=0, quorum=1
		{4, 3},  // f=1, quorum=3
		{7, 5},  // f=2, quorum=5
		{10, 7}, // f=3, quorum=7
		{13, 9}, // f=4, quorum=9
	}
	for _, tc := range cases {
		got := QuorumSize(tc.members)
		if got != tc.expected {
			t.Errorf("QuorumSize(%d) = %d, want %d", tc.members, got, tc.expected)
		}
	}
}

func TestHasQuorumRequires2fPlus1DistinctSigners(t *testing.T) {
	entry := Entry{Height: 1, Type: EntryQC, Payload: []byte("qc")}
	// m=4 primaries, f=1, need 3 attestations
	if entry.HasQuorum(4) {
		t.Fatal("entry with no attestations should not have quorum")
	}

	entry.Attestations = []Attestation{
		{SignerID: 0, Signature: []byte("sig0")},
		{SignerID: 1, Signature: []byte("sig1")},
	}
	if entry.HasQuorum(4) {
		t.Fatal("entry with 2 attestations should not have quorum for m=4")
	}

	entry.Attestations = append(entry.Attestations, Attestation{SignerID: 2, Signature: []byte("sig2")})
	if !entry.HasQuorum(4) {
		t.Fatal("entry with 3 attestations should have quorum for m=4")
	}
}

func TestHasQuorumDeduplicatesSigners(t *testing.T) {
	entry := Entry{
		Height: 1, Type: EntryQC, Payload: []byte("qc"),
		Attestations: []Attestation{
			{SignerID: 0, Signature: []byte("sig0")},
			{SignerID: 0, Signature: []byte("sig0-dup")},
			{SignerID: 1, Signature: []byte("sig1")},
		},
	}
	// m=4, need 3 distinct signers, only 2 distinct here
	if entry.HasQuorum(4) {
		t.Fatal("duplicate signers should not count toward quorum")
	}
}

func TestLogAttestCollectsSignatures(t *testing.T) {
	log := NewLogWithMembers(4) // m=4, f=1, quorum=3
	entry := Entry{Height: 1, Type: EntryQC, Payload: []byte("qc")}
	if err := log.Publish(entry); err != nil {
		t.Fatalf("publish failed: %v", err)
	}

	// First attestation
	hasQ, err := log.Attest(1, Attestation{SignerID: 0, Signature: []byte("sig0")})
	if err != nil {
		t.Fatalf("attest failed: %v", err)
	}
	if hasQ {
		t.Fatal("should not have quorum after 1 attestation")
	}

	// Second attestation
	hasQ, err = log.Attest(1, Attestation{SignerID: 1, Signature: []byte("sig1")})
	if err != nil {
		t.Fatalf("attest failed: %v", err)
	}
	if hasQ {
		t.Fatal("should not have quorum after 2 attestations")
	}

	// Third attestation reaches quorum
	hasQ, err = log.Attest(1, Attestation{SignerID: 2, Signature: []byte("sig2")})
	if err != nil {
		t.Fatalf("attest failed: %v", err)
	}
	if !hasQ {
		t.Fatal("should have quorum after 3 attestations for m=4")
	}

	// Verify attestations are persisted in retrieved entry
	got, ok := log.Retrieve(1)
	if !ok {
		t.Fatal("expected entry at height 1")
	}
	if len(got.Attestations) != 3 {
		t.Fatalf("expected 3 attestations, got %d", len(got.Attestations))
	}
}

func TestLogAttestDeduplicatesSameSigner(t *testing.T) {
	log := NewLogWithMembers(4)
	if err := log.Publish(Entry{Height: 1, Type: EntryQC, Payload: []byte("qc")}); err != nil {
		t.Fatalf("publish failed: %v", err)
	}

	log.Attest(1, Attestation{SignerID: 0, Signature: []byte("sig0")})
	log.Attest(1, Attestation{SignerID: 0, Signature: []byte("sig0-again")}) // duplicate

	got, _ := log.Retrieve(1)
	if len(got.Attestations) != 1 {
		t.Fatalf("expected 1 attestation after dedup, got %d", len(got.Attestations))
	}
}

func TestLogAttestRejectsEmptySignature(t *testing.T) {
	log := NewLogWithMembers(4)
	if err := log.Publish(Entry{Height: 1, Type: EntryQC, Payload: []byte("qc")}); err != nil {
		t.Fatalf("publish failed: %v", err)
	}

	if _, err := log.Attest(1, Attestation{SignerID: 0, Signature: nil}); err == nil {
		t.Fatal("expected empty signature to be rejected")
	}
}

func TestLogAttestRejectsNonExistentHeight(t *testing.T) {
	log := NewLogWithMembers(4)
	if _, err := log.Attest(99, Attestation{SignerID: 0, Signature: []byte("sig")}); err == nil {
		t.Fatal("expected non-existent height to fail")
	}
}

func TestEntryDigestIsDeterministic(t *testing.T) {
	d1 := EntryDigest(1, EntryQC, []byte("payload"))
	d2 := EntryDigest(1, EntryQC, []byte("payload"))
	if d1 != d2 {
		t.Fatal("digest should be deterministic")
	}

	d3 := EntryDigest(2, EntryQC, []byte("payload"))
	if d1 == d3 {
		t.Fatal("different height should produce different digest")
	}

	d4 := EntryDigest(1, EntryCheckpoint, []byte("payload"))
	if d1 == d4 {
		t.Fatal("different type should produce different digest")
	}
}

func TestLogConcurrentPublishAndRetrieve(t *testing.T) {
	log := NewLogWithMembers(4)
	// Publish sequentially (append-only constraint)
	for i := uint64(1); i <= 100; i++ {
		if err := log.Publish(Entry{Height: i, Type: EntryQC, Payload: []byte("qc")}); err != nil {
			t.Fatalf("publish %d failed: %v", i, err)
		}
	}
	// Concurrent reads
	done := make(chan bool, 10)
	for g := 0; g < 10; g++ {
		go func() {
			for i := uint64(1); i <= 100; i++ {
				if _, ok := log.Retrieve(i); !ok {
					t.Errorf("expected entry at height %d", i)
				}
			}
			done <- true
		}()
	}
	for g := 0; g < 10; g++ {
		<-done
	}
}

func TestLogMembershipEntryType(t *testing.T) {
	log := NewLog()
	entry := Entry{Height: 1, Type: EntryMembership, Payload: []byte(`{"join":"node-5"}`)}
	if err := log.Publish(entry); err != nil {
		t.Fatalf("publish membership entry failed: %v", err)
	}
	got, ok := log.LatestByType(EntryMembership)
	if !ok {
		t.Fatal("expected membership entry")
	}
	if got.Type != EntryMembership {
		t.Fatalf("expected membership type, got %s", got.Type)
	}
}
