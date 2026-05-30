package types

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"testing"
)

// ===========================================================================
// Phase
// ===========================================================================

func TestPhase_String(t *testing.T) {
	tests := []struct {
		phase Phase
		want  string
	}{
		{PhasePrepare, "PREPARE"},
		{PhasePrecommit, "PRECOMMIT"},
		{PhaseCommit, "COMMIT"},
		{PhaseDecide, "DECIDE"},
		{Phase(99), "UNKNOWN"},
	}
	for _, tt := range tests {
		if got := tt.phase.String(); got != tt.want {
			t.Errorf("Phase(%d).String() = %q, want %q", tt.phase, got, tt.want)
		}
	}
}

// ===========================================================================
// MessageType
// ===========================================================================

func TestMessageType_String(t *testing.T) {
	tests := []struct {
		mt   MessageType
		want string
	}{
		{MsgProposal, "PROPOSAL"},
		{MsgVote, "VOTE"},
		{MsgNewView, "NEW_VIEW"},
		{MsgTimeout, "TIMEOUT"},
		{MsgJoin, "JOIN"},
		{MsgLeave, "LEAVE"},
		{MsgSync, "SYNC"},
		{MessageType(99), "UNKNOWN"},
	}
	for _, tt := range tests {
		if got := tt.mt.String(); got != tt.want {
			t.Errorf("MessageType(%d).String() = %q, want %q", tt.mt, got, tt.want)
		}
	}
}

// ===========================================================================
// Block
// ===========================================================================

func TestNewBlock_ComputesHash(t *testing.T) {
	b := NewBlock(1, make([]byte, 32), []byte("data"), 1, 1, 0, 0,
		NewQuorumCertificate(nil, 0, 0, PhaseDecide), nil)

	if len(b.Hash) != 32 {
		t.Fatalf("expected 32-byte hash, got %d", len(b.Hash))
	}
}

func TestBlock_ComputeHash_Deterministic(t *testing.T) {
	parent := make([]byte, 32)
	qc := NewQuorumCertificate(nil, 0, 0, PhaseDecide)

	b1 := &Block{Height: 1, Parent: parent, Data: []byte("x"), View: 1, Epoch: 1, Justify: qc}
	b2 := &Block{Height: 1, Parent: parent, Data: []byte("x"), View: 1, Epoch: 1, Justify: qc}

	h1 := b1.ComputeHash()
	h2 := b2.ComputeHash()
	if !bytes.Equal(h1, h2) {
		t.Error("identical blocks should produce identical hashes")
	}
}

func TestBlock_ComputeHash_DifferentData(t *testing.T) {
	parent := make([]byte, 32)
	qc := NewQuorumCertificate(nil, 0, 0, PhaseDecide)

	b1 := &Block{Height: 1, Parent: parent, Data: []byte("a"), View: 1, Epoch: 1, Justify: qc}
	b2 := &Block{Height: 1, Parent: parent, Data: []byte("b"), View: 1, Epoch: 1, Justify: qc}

	if bytes.Equal(b1.ComputeHash(), b2.ComputeHash()) {
		t.Error("blocks with different data should have different hashes")
	}
}

func TestBlock_Getters(t *testing.T) {
	b := NewBlock(5, []byte("parent"), []byte("data"), 10, 2, 3, 7,
		NewQuorumCertificate(nil, 0, 0, PhaseDecide), []byte("rand"))
	b.LaneID = 4
	b.ConfigID = 8

	if b.GetHeight() != 5 {
		t.Errorf("GetHeight: got %d", b.GetHeight())
	}
	if b.GetView() != 10 {
		t.Errorf("GetView: got %d", b.GetView())
	}
	if b.GetEpoch() != 2 {
		t.Errorf("GetEpoch: got %d", b.GetEpoch())
	}
	if b.GetLeaderID() != 3 {
		t.Errorf("GetLeaderID: got %d", b.GetLeaderID())
	}
	if b.GetRank() != 7 {
		t.Errorf("GetRank: got %d", b.GetRank())
	}
	if b.GetLaneID() != 4 {
		t.Errorf("GetLaneID: got %d", b.GetLaneID())
	}
	if b.GetConfigID() != 8 {
		t.Errorf("GetConfigID: got %d", b.GetConfigID())
	}
	if b.GetPhase() != PhasePrepare {
		t.Errorf("GetPhase: got %v", b.GetPhase())
	}
}

func TestBlock_SetPhase(t *testing.T) {
	b := NewBlock(1, nil, nil, 1, 1, 0, 0, nil, nil)
	b.SetPhase(PhaseCommit)
	if b.GetPhase() != PhaseCommit {
		t.Errorf("expected COMMIT, got %v", b.GetPhase())
	}
}

// ===========================================================================
// Vote
// ===========================================================================

func TestVote_NewAndVerify(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	var blockID Hash
	copy(blockID[:], []byte("test-block-id-32bytes-padded...."))

	vote, err := NewVote(blockID, 1, pub, priv, nil)
	if err != nil {
		t.Fatalf("NewVote failed: %v", err)
	}
	if !vote.Verify() {
		t.Error("vote should verify with correct key")
	}
}

func TestVote_VerifyFailsWithWrongKey(t *testing.T) {
	pub1, priv1, _ := ed25519.GenerateKey(rand.Reader)
	pub2, _, _ := ed25519.GenerateKey(rand.Reader)
	var blockID Hash

	vote, _ := NewVote(blockID, 1, pub1, priv1, nil)
	vote.Author = pub2 // swap key
	if vote.Verify() {
		t.Error("vote should not verify with wrong key")
	}
}

func TestVoteWithIdentity(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	var blockID Hash

	vote, err := NewVoteWithIdentity(blockID, 5, 2, 7, 3, pub, priv, []byte("vrf-proof"))
	if err != nil {
		t.Fatalf("NewVoteWithIdentity failed: %v", err)
	}
	if vote.View != 5 || vote.Epoch != 2 || vote.ConfigID != 7 || vote.Lane != 3 {
		t.Error("identity fields mismatch")
	}
	if !vote.Verify() {
		t.Error("identity vote should verify")
	}
}

// ===========================================================================
// QuorumCertificate
// ===========================================================================

func TestQC_AddSignature_Deduplicates(t *testing.T) {
	qc := NewQuorumCertificate([]byte("block"), 1, 1, PhasePrepare)
	qc.AddSignature(1, []byte("sig1"))
	qc.AddSignature(1, []byte("sig1-dup"))

	if qc.GetNumSignatures() != 1 {
		t.Errorf("expected 1 signature after dedup, got %d", qc.GetNumSignatures())
	}
}

func TestQC_IsComplete(t *testing.T) {
	qc := NewQuorumCertificate([]byte("block"), 1, 1, PhasePrepare)
	qc.AddSignature(1, []byte("s1"))
	qc.AddSignature(2, []byte("s2"))

	if qc.IsComplete(3) {
		t.Error("should not be complete with 2/3")
	}
	qc.AddSignature(3, []byte("s3"))
	if !qc.IsComplete(3) {
		t.Error("should be complete with 3/3")
	}
}

func TestQC_WithIdentity(t *testing.T) {
	qc := NewQuorumCertificateWithIdentity([]byte("block"), 5, 2, 7, 3, PhasePrecommit)
	if qc.View != 5 || qc.Epoch != 2 || qc.ConfigID != 7 || qc.Lane != 3 || qc.Phase != PhasePrecommit {
		t.Error("identity fields mismatch")
	}
}

// ===========================================================================
// ValidatorSet
// ===========================================================================

func TestNewValidatorSet_QuorumCalculation(t *testing.T) {
	validators := map[uint64]*Validator{
		1: {ID: 1, Power: 1, IsActive: true},
		2: {ID: 2, Power: 1, IsActive: true},
		3: {ID: 3, Power: 1, IsActive: true},
		4: {ID: 4, Power: 1, IsActive: true},
	}
	vs := NewValidatorSet(1, validators)

	if vs.TotalPower != 4 {
		t.Errorf("expected total power 4, got %d", vs.TotalPower)
	}
	// quorum = 4*2/3 + 1 = 3
	if vs.QuorumSize != 3 {
		t.Errorf("expected quorum size 3, got %d", vs.QuorumSize)
	}
}

func TestNewValidatorSet_SkipsInactiveValidators(t *testing.T) {
	validators := map[uint64]*Validator{
		1: {ID: 1, Power: 1, IsActive: true},
		2: {ID: 2, Power: 1, IsActive: false},
		3: {ID: 3, Power: 1, IsActive: true},
	}
	vs := NewValidatorSet(1, validators)

	if vs.TotalPower != 2 {
		t.Errorf("expected total power 2 (skipping inactive), got %d", vs.TotalPower)
	}
}

func TestNewValidatorSet_FallbackForNoActiveValidators(t *testing.T) {
	validators := map[uint64]*Validator{
		1: {ID: 1, Power: 0, IsActive: false},
		2: {ID: 2, Power: 0, IsActive: false},
	}
	vs := NewValidatorSet(1, validators)

	// Fallback: totalPower = len(validators) = 2
	if vs.TotalPower != 2 {
		t.Errorf("expected fallback total power 2, got %d", vs.TotalPower)
	}
}

func TestValidatorSet_Copy(t *testing.T) {
	pub, _, _ := ed25519.GenerateKey(rand.Reader)
	validators := map[uint64]*Validator{
		1: {ID: 1, PublicKey: pub, VRFPublicKey: []byte("vrf"), Power: 1, IsActive: true},
	}
	vs := NewValidatorSet(1, validators)
	cp := vs.Copy()

	if cp.Epoch != vs.Epoch || cp.TotalPower != vs.TotalPower {
		t.Error("copy should preserve epoch and total power")
	}
	if &cp.Validators[1].PublicKey[0] == &vs.Validators[1].PublicKey[0] {
		t.Error("copy should deep-copy public keys")
	}
}

func TestValidatorSet_Hash_Deterministic(t *testing.T) {
	validators := map[uint64]*Validator{
		1: {ID: 1, PublicKey: []byte("pk1"), Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: []byte("pk2"), Power: 1, IsActive: true},
	}
	vs1 := NewValidatorSet(1, validators)
	vs2 := NewValidatorSet(1, validators)

	if !bytes.Equal(vs1.Hash(), vs2.Hash()) {
		t.Error("identical validator sets should produce identical hashes")
	}
}

// ===========================================================================
// ReconfigData signing
// ===========================================================================

func TestReconfigData_SignAndVerify(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	data := &ReconfigData{
		Type:        ReconfigJoin,
		NodeID:      1,
		PublicKey:   pub,
		Power:       1,
		Epoch:       1,
		TargetEpoch: 2,
	}

	data.Sign(priv)
	if !data.VerifySignature() {
		t.Error("signature should verify")
	}
}

func TestReconfigData_VerifyFailsForAutoLeave(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	data := &ReconfigData{
		Type:      ReconfigAutoLeave,
		NodeID:    1,
		PublicKey: pub,
	}
	data.Sign(priv)
	// AutoLeave uses quorum proof, not individual signature
	if data.VerifySignature() {
		t.Error("AutoLeave should not verify via individual signature")
	}
}

func TestReconfigData_VerifyFailsForNil(t *testing.T) {
	var data *ReconfigData
	if data.VerifySignature() {
		t.Error("nil data should not verify")
	}
}

// ===========================================================================
// Signing bytes helpers
// ===========================================================================

func TestVoteSigningBytes_Deterministic(t *testing.T) {
	var blockID Hash
	copy(blockID[:], []byte("block"))
	b1 := VoteSigningBytes(blockID, 1, 2, 3, 4)
	b2 := VoteSigningBytes(blockID, 1, 2, 3, 4)
	if !bytes.Equal(b1, b2) {
		t.Error("same inputs should produce same signing bytes")
	}
}

func TestQCSigningBytes_MatchesVoteSigningBytes(t *testing.T) {
	blockHash := make([]byte, 32)
	copy(blockHash, []byte("qc-block"))
	var blockID Hash
	copy(blockID[:], blockHash)

	qcBytes := QCSigningBytes(blockHash, 1, 2, 3, 4)
	voteBytes := VoteSigningBytes(blockID, 1, 2, 3, 4)
	if !bytes.Equal(qcBytes, voteBytes) {
		t.Error("QCSigningBytes should match VoteSigningBytes for same input")
	}
}

func TestTimeoutSigningBytes_Deterministic(t *testing.T) {
	b1 := TimeoutSigningBytes(5, 2, 3, 4)
	b2 := TimeoutSigningBytes(5, 2, 3, 4)
	if !bytes.Equal(b1, b2) {
		t.Error("same inputs should produce same timeout signing bytes")
	}
}

// ===========================================================================
// Message signing
// ===========================================================================

func TestMessage_SignAndVerify(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	msg := &Message{
		Type:     MsgVote,
		SenderID: 1,
		View:     5,
		Epoch:    1,
	}
	if err := msg.Sign(priv); err != nil {
		t.Fatalf("Sign failed: %v", err)
	}
	if !msg.VerifySignature(pub) {
		t.Error("should verify with correct key")
	}
}

func TestMessage_VerifyFailsWithWrongKey(t *testing.T) {
	_, priv1, _ := ed25519.GenerateKey(rand.Reader)
	pub2, _, _ := ed25519.GenerateKey(rand.Reader)
	msg := &Message{Type: MsgVote, SenderID: 1, View: 1}
	msg.Sign(priv1)
	if msg.VerifySignature(pub2) {
		t.Error("should not verify with wrong key")
	}
}

func TestMessage_VerifyNilMessage(t *testing.T) {
	var msg *Message
	if msg.VerifySignature(nil) {
		t.Error("nil message should not verify")
	}
}

// ===========================================================================
// DAG types
// ===========================================================================

func TestNewVertex(t *testing.T) {
	txs := []*Transaction{{Type: TxTypeNormal, Payload: []byte("tx1")}}
	v := NewVertex(1, 2, 42, txs, nil)
	if v.Epoch != 1 || v.Round != 2 || v.Author != 42 {
		t.Error("vertex fields mismatch")
	}
	if len(v.Txs) != 1 {
		t.Error("expected 1 tx")
	}
}

func TestVertexCertificate_AddSignature(t *testing.T) {
	var hash Hash
	vc := NewVertexCertificate(hash, 1, 2)
	vc.AddSignature(1, []byte("sig1"))
	vc.AddSignature(2, []byte("sig2"))

	sigs := vc.GetSignatures()
	if len(sigs) != 2 {
		t.Errorf("expected 2 signatures, got %d", len(sigs))
	}
}

// ===========================================================================
// Configuration
// ===========================================================================

func TestConfiguration_Hash_Deterministic(t *testing.T) {
	cfg := &Configuration{
		ID: 1,
		Validators: map[uint64]*Validator{
			1: {ID: 1, PublicKey: []byte("pk1"), Power: 1, IsActive: true},
			2: {ID: 2, PublicKey: []byte("pk2"), Power: 1, IsActive: true},
		},
		QuorumSize: 2,
	}
	h1 := cfg.Hash()
	h2 := cfg.Hash()
	if !bytes.Equal(h1, h2) {
		t.Error("configuration hash should be deterministic")
	}
}

func TestConfiguration_ToValidatorSet(t *testing.T) {
	cfg := &Configuration{
		ID: 3,
		Validators: map[uint64]*Validator{
			1: {ID: 1, PublicKey: []byte("pk"), Power: 1, IsActive: true},
		},
		QuorumSize: 1,
	}
	vs := cfg.ToValidatorSet()
	if vs == nil {
		t.Fatal("ToValidatorSet should not return nil")
	}
	if vs.Epoch != 3 {
		t.Errorf("expected epoch 3, got %d", vs.Epoch)
	}
	if vs.QuorumSize != 1 {
		t.Errorf("expected quorum 1, got %d", vs.QuorumSize)
	}
}

func TestConfiguration_NilHash(t *testing.T) {
	var cfg *Configuration
	if cfg.Hash() != nil {
		t.Error("nil config should return nil hash")
	}
}

func TestConfiguration_NilToValidatorSet(t *testing.T) {
	var cfg *Configuration
	if cfg.ToValidatorSet() != nil {
		t.Error("nil config should return nil validator set")
	}
}

// ===========================================================================
// LeaderSetHashEqual
// ===========================================================================

func TestLeaderSetHashEqual(t *testing.T) {
	if !LeaderSetHashEqual([]byte("abc"), []byte("abc")) {
		t.Error("equal hashes should match")
	}
	if LeaderSetHashEqual([]byte("abc"), []byte("xyz")) {
		t.Error("different hashes should not match")
	}
	if !LeaderSetHashEqual(nil, nil) {
		t.Error("nil hashes should match")
	}
}
