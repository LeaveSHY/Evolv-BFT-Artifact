package types

import (
	"crypto/ed25519"
	"crypto/rand"
	"testing"
)

func generateTestEd25519Keys() (PublicKey, PrivateKey, error) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	return PublicKey(pub), PrivateKey(priv), err
}

// --- Vertex.GetHash ---

func TestVertexGetHash(t *testing.T) {
	v := NewVertex(1, 2, 3, nil, nil)
	var expected Hash
	expected[0] = 0xAB
	v.Hash = expected
	got := v.GetHash()
	if got != expected {
		t.Fatalf("GetHash mismatch: got %x, want %x", got, expected)
	}
}

func TestVertexGetHashZero(t *testing.T) {
	v := NewVertex(1, 1, 0, nil, nil)
	got := v.GetHash()
	var zero Hash
	if got == zero {
		t.Fatal("expected non-zero hash for vertex with computed hash")
	}
}

// --- Block.GetHash / GetParent ---

func TestBlockGetHash(t *testing.T) {
	b := NewBlock(1, []byte("parent"), []byte("data"), 1, 1, 0, 0, nil, nil)
	hash := b.GetHash()
	if len(hash) == 0 {
		t.Fatal("GetHash should return non-empty hash after NewBlock")
	}
}

func TestBlockGetParent(t *testing.T) {
	parent := []byte("parenthash")
	b := NewBlock(1, parent, nil, 1, 1, 0, 0, nil, nil)
	got := b.GetParent()
	if string(got) != string(parent) {
		t.Fatalf("GetParent: got %q, want %q", got, parent)
	}
}

// --- Block.GetJoinRequests / GetLeaveRequests ---

func TestBlockGetJoinRequests(t *testing.T) {
	b := NewBlock(1, nil, nil, 1, 1, 0, 0, nil, nil)
	b.JoinRequests = []PublicKey{[]byte("pk1"), []byte("pk2")}
	got := b.GetJoinRequests()
	if len(got) != 2 {
		t.Fatalf("expected 2 join requests, got %d", len(got))
	}
}

func TestBlockGetJoinRequestsNil(t *testing.T) {
	b := NewBlock(1, nil, nil, 1, 1, 0, 0, nil, nil)
	got := b.GetJoinRequests()
	if got != nil {
		t.Fatalf("expected nil join requests, got %v", got)
	}
}

func TestBlockGetLeaveRequests(t *testing.T) {
	b := NewBlock(1, nil, nil, 1, 1, 0, 0, nil, nil)
	b.LeaveRequests = []PublicKey{[]byte("leave1")}
	got := b.GetLeaveRequests()
	if len(got) != 1 {
		t.Fatalf("expected 1 leave request, got %d", len(got))
	}
}

func TestBlockGetLeaveRequestsNil(t *testing.T) {
	b := NewBlock(1, nil, nil, 1, 1, 0, 0, nil, nil)
	got := b.GetLeaveRequests()
	if got != nil {
		t.Fatalf("expected nil leave requests, got %v", got)
	}
}

// --- QuorumCertificate.GetBlockHash / GetView / GetEpoch ---

func TestQCGetBlockHash(t *testing.T) {
	qc := NewQuorumCertificateWithIdentity([]byte("blockhash"), 5, 2, 0, 0, PhasePrepare)
	got := qc.GetBlockHash()
	if string(got) != "blockhash" {
		t.Fatalf("GetBlockHash: got %q, want %q", got, "blockhash")
	}
}

func TestQCGetView(t *testing.T) {
	qc := NewQuorumCertificateWithIdentity(nil, 7, 3, 0, 0, PhasePrepare)
	if qc.GetView() != 7 {
		t.Fatalf("GetView: got %d, want 7", qc.GetView())
	}
}

func TestQCGetEpoch(t *testing.T) {
	qc := NewQuorumCertificateWithIdentity(nil, 1, 4, 0, 0, PhasePrepare)
	if qc.GetEpoch() != 4 {
		t.Fatalf("GetEpoch: got %d, want 4", qc.GetEpoch())
	}
}

// --- Block.ComputeHash with payload and justify ---

func TestBlockComputeHashWithPayloadAndJustify(t *testing.T) {
	justify := NewQuorumCertificateWithIdentity([]byte("justifyhash"), 1, 1, 0, 0, PhasePrepare)
	b := NewBlock(2, []byte("p"), []byte("d"), 2, 1, 1, 5, justify, []byte("rand"))
	// Add a vertex certificate to payload
	var vHash Hash
	vHash[0] = 0xFF
	cert := NewVertexCertificate(vHash, 1, 1)
	b.Payload = append(b.Payload, cert)
	hash := b.ComputeHash()
	if len(hash) == 0 {
		t.Fatal("ComputeHash with payload should return non-empty hash")
	}
	// Adding nil cert to payload
	b.Payload = append(b.Payload, nil)
	hash2 := b.ComputeHash()
	if len(hash2) == 0 {
		t.Fatal("ComputeHash with nil cert should still work")
	}
}

// --- ReconfigData.Sign and VerifySignature ---

func TestReconfigDataSignAndVerify(t *testing.T) {
	pub, priv, _ := generateTestEd25519Keys()
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
		t.Fatal("signature verification failed for valid ReconfigJoin")
	}
}

func TestReconfigDataVerifySignatureNil(t *testing.T) {
	var d *ReconfigData
	if d.VerifySignature() {
		t.Fatal("nil ReconfigData should not verify")
	}
}

func TestReconfigDataSignNil(t *testing.T) {
	var d *ReconfigData
	d.Sign(nil) // should not panic
}

func TestReconfigDataVerifySignatureLeave(t *testing.T) {
	pub, priv, _ := generateTestEd25519Keys()
	data := &ReconfigData{
		Type:        ReconfigLeave,
		NodeID:      2,
		PublicKey:   pub,
		Power:       1,
		Epoch:       1,
		TargetEpoch: 2,
	}
	data.Sign(priv)
	if !data.VerifySignature() {
		t.Fatal("leave verification should pass")
	}
}

func TestReconfigDataVerifySignatureUnknownType(t *testing.T) {
	pub, priv, _ := generateTestEd25519Keys()
	data := &ReconfigData{
		Type:      ReconfigType(99),
		NodeID:    1,
		PublicKey: pub,
	}
	data.Sign(priv)
	if data.VerifySignature() {
		t.Fatal("unknown reconfig type should not verify")
	}
}

func TestReconfigSigningBytesNil(t *testing.T) {
	got := ReconfigSigningBytes(nil)
	if got != nil {
		t.Fatal("expected nil for nil input")
	}
}

// --- VertexCertificate.GetSignatures ---

func TestVertexCertificateGetSignatures(t *testing.T) {
	var h Hash
	h[0] = 1
	vc := NewVertexCertificate(h, 1, 1)
	vc.AddSignature(0, []byte("sig0"))
	vc.AddSignature(1, []byte("sig1"))
	sigs := vc.GetSignatures()
	if len(sigs) != 2 {
		t.Fatalf("expected 2 signatures, got %d", len(sigs))
	}
}
