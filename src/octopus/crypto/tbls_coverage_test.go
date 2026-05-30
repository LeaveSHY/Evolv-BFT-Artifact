package crypto

import (
	"testing"

	"go.dedis.ch/kyber/v3/pairing"
)

// --- TBLSThreshold additional coverage ---

func TestTBLSVerify(t *testing.T) {
	tbls, err := NewTBLSThreshold(2, 3)
	if err != nil {
		t.Fatalf("new tbls: %v", err)
	}
	msg := []byte("verify-test")
	shareKey, err := tbls.GenerateShare(0)
	if err != nil {
		t.Fatalf("generate share: %v", err)
	}
	sig, err := tbls.Sign(msg, shareKey)
	if err != nil {
		t.Fatalf("sign: %v", err)
	}
	// Derive public key from share
	priv := tbls.suite.G1().Scalar()
	if err := priv.UnmarshalBinary(shareKey); err != nil {
		t.Fatalf("unmarshal share key: %v", err)
	}
	pub := tbls.suite.G2().Point().Mul(priv, nil)
	if err := tbls.Verify(msg, sig, pub); err != nil {
		t.Fatalf("verify failed: %v", err)
	}
}

func TestTBLSVerifyBadSig(t *testing.T) {
	tbls, err := NewTBLSThreshold(2, 3)
	if err != nil {
		t.Fatalf("new tbls: %v", err)
	}
	msg := []byte("verify-bad")
	shareKey, err := tbls.GenerateShare(0)
	if err != nil {
		t.Fatalf("generate share: %v", err)
	}
	priv := tbls.suite.G1().Scalar()
	_ = priv.UnmarshalBinary(shareKey)
	pub := tbls.suite.G2().Point().Mul(priv, nil)
	verErr := tbls.Verify(msg, []byte("invalid-signature"), pub)
	if verErr == nil {
		t.Fatal("expected error for bad signature")
	}
}

func TestTBLSRecover(t *testing.T) {
	tbls, err := NewTBLSThreshold(2, 3)
	if err != nil {
		t.Fatalf("new tbls: %v", err)
	}
	msg := []byte("recover-test")

	shares := make([][]byte, 3)
	for i := 0; i < 3; i++ {
		shareKey, err := tbls.GenerateShare(i)
		if err != nil {
			t.Fatalf("generate share %d: %v", i, err)
		}
		sig, err := tbls.Sign(msg, shareKey)
		if err != nil {
			t.Fatalf("sign share %d: %v", i, err)
		}
		shares[i] = sig
	}

	combined, err := tbls.Recover(msg, shares)
	if err != nil {
		t.Fatalf("recover: %v", err)
	}
	if len(combined) == 0 {
		t.Fatal("combined signature is empty")
	}
}

func TestTBLSRecoverNotEnoughShares(t *testing.T) {
	tbls, err := NewTBLSThreshold(2, 3)
	if err != nil {
		t.Fatalf("new tbls: %v", err)
	}
	_, recErr := tbls.Recover([]byte("msg"), [][]byte{[]byte("one")})
	if recErr == nil {
		t.Fatal("expected error for insufficient shares")
	}
}

func TestTBLSGetPublicKey(t *testing.T) {
	tbls, err := NewTBLSThreshold(2, 3)
	if err != nil {
		t.Fatalf("new tbls: %v", err)
	}
	pk := tbls.GetPublicKey()
	_ = pk // may be nil without distributed setup, just ensure no panic
}

// --- BLSFull additional coverage ---

func TestRandBytes(t *testing.T) {
	b := randBytes(16)
	if len(b) != 16 {
		t.Fatalf("expected 16 bytes, got %d", len(b))
	}
}

// --- Unmarshal/Marshal round-trip ---

func TestMarshalUnmarshalPointRoundTrip(t *testing.T) {
	suite := pairing.NewSuiteBn256()
	point := suite.G1().Point().Pick(suite.RandomStream())
	data, err := MarshalPoint(point)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	recovered, err := UnmarshalPoint(suite, data)
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !recovered.Equal(point) {
		t.Fatal("round-trip failed: points differ")
	}
}

func TestMarshalUnmarshalScalarRoundTrip(t *testing.T) {
	suite := pairing.NewSuiteBn256()
	scalar := suite.G1().Scalar().Pick(suite.RandomStream())
	data, err := MarshalScalar(scalar)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	recovered, err := UnmarshalScalar(suite, data)
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !recovered.Equal(scalar) {
		t.Fatal("round-trip failed: scalars differ")
	}
}

func TestUnmarshalPointInvalid(t *testing.T) {
	suite := pairing.NewSuiteBn256()
	_, err := UnmarshalPoint(suite, []byte("bad"))
	if err == nil {
		t.Fatal("expected error for invalid point data")
	}
}

func TestUnmarshalScalarInvalid(t *testing.T) {
	suite := pairing.NewSuiteBn256()
	_, err := UnmarshalScalar(suite, []byte("bad"))
	if err == nil {
		t.Fatal("expected error for invalid scalar data")
	}
}

// --- ThresholdSignature ---

func TestNewThresholdSignature(t *testing.T) {
	ts := NewThresholdSignature([]byte("sig"), 3)
	if ts == nil {
		t.Fatal("expected non-nil")
	}
	if ts.NumShares != 3 {
		t.Fatalf("expected 3 shares, got %d", ts.NumShares)
	}
}

func TestThresholdSignatureVerifyEmpty(t *testing.T) {
	ts := NewThresholdSignature(nil, 0)
	kp, _ := GenerateKeyPair()
	if ts.Verify(kp.PublicKey, []byte("msg")) {
		t.Fatal("empty signature should not verify")
	}
}

func TestThresholdSignatureVerifyBad(t *testing.T) {
	kp, _ := GenerateKeyPair()
	wrongMsg := []byte("wrong-message")
	sig := Sign([]byte("correct-message"), kp.PrivateKey)
	ts := NewThresholdSignature(sig, 1)
	if ts.Verify(kp.PublicKey, wrongMsg) {
		t.Fatal("bad message should not verify")
	}
}

func TestThresholdSignatureVerifyValid(t *testing.T) {
	kp, _ := GenerateKeyPair()
	msg := []byte("threshold-verify-msg")
	sig := Sign(msg, kp.PrivateKey)
	ts := NewThresholdSignature(sig, 1)
	if !ts.Verify(kp.PublicKey, msg) {
		t.Fatal("valid threshold signature should verify")
	}
}

// --- CombinedSignature ---

func TestCombinedSignatureInsufficient(t *testing.T) {
	_, err := CombinedSignature([]SignatureShare{
		{Share: []byte("s1"), Sender: 0, Index: 1},
	}, 3)
	if err == nil {
		t.Fatal("expected error for insufficient shares")
	}
}

func TestCombinedSignatureEmptyShares(t *testing.T) {
	shares := []SignatureShare{
		{Share: nil, Sender: 0, Index: 1},
		{Share: nil, Sender: 1, Index: 2},
	}
	_, err := CombinedSignature(shares, 2)
	if err == nil {
		t.Fatal("expected error for empty shares")
	}
}

func TestCombinedSignatureValid(t *testing.T) {
	blsFull := NewBLSFull()
	msg := []byte("combine-test")

	priv1, _, _ := blsFull.GenerateKeyPair()
	priv2, _, _ := blsFull.GenerateKeyPair()
	sig1, _ := blsFull.Sign(priv1, msg)
	sig2, _ := blsFull.Sign(priv2, msg)

	shares := []SignatureShare{
		{Share: sig1, Sender: 0, Index: 1},
		{Share: sig2, Sender: 1, Index: 2},
	}
	combined, err := CombinedSignature(shares, 2)
	if err != nil {
		t.Fatalf("CombinedSignature: %v", err)
	}
	if len(combined) == 0 {
		t.Fatal("expected non-empty combined signature")
	}
}

// --- VerifyShare ---

func TestVerifyShareEmpty(t *testing.T) {
	kp, _ := GenerateKeyPair()
	if VerifyShare(kp.PublicKey, []byte("msg"), SignatureShare{Share: []byte("bad")}) {
		t.Fatal("bad share should return false")
	}
}

func TestVerifyShareValid(t *testing.T) {
	kp, _ := GenerateKeyPair()
	msg := []byte("verify-share-msg")
	sig := Sign(msg, kp.PrivateKey)
	share := SignatureShare{Share: sig, Sender: 0, Index: 1}
	if !VerifyShare(kp.PublicKey, msg, share) {
		t.Fatal("valid share should verify")
	}
}

func TestVerifyShareInvalid(t *testing.T) {
	kp, _ := GenerateKeyPair()
	msg := []byte("verify-share-bad")
	share := SignatureShare{Share: []byte("bad"), Sender: 0, Index: 1}
	if VerifyShare(kp.PublicKey, msg, share) {
		t.Fatal("invalid share should not verify")
	}
}

// --- DistributeShares ---

func TestDistributeShares(t *testing.T) {
	shares, err := DistributeShares([]byte("secret"), 5, 3)
	if err != nil {
		t.Fatalf("DistributeShares: %v", err)
	}
	if len(shares) != 5 {
		t.Fatalf("expected 5 shares, got %d", len(shares))
	}
	seen := make(map[string]struct{})
	for _, s := range shares {
		key := string(s.Share)
		if _, exists := seen[key]; exists {
			t.Fatal("shares should be unique")
		}
		seen[key] = struct{}{}
	}
}

func TestDistributeSharesThresholdTooHigh(t *testing.T) {
	_, err := DistributeShares([]byte("secret"), 2, 5)
	if err == nil {
		t.Fatal("expected error when threshold > numShares")
	}
}

func TestDistributeSharesZero(t *testing.T) {
	_, err := DistributeShares([]byte("secret"), 0, 0)
	if err == nil {
		t.Fatal("expected error for zero shares")
	}
}

// --- scalarFromBigInt ---

func TestScalarFromBigInt(t *testing.T) {
	suite := pairing.NewSuiteBn256()
	val := bn256Order()
	s := scalarFromBigInt(suite, val)
	if s == nil {
		t.Fatal("expected non-nil scalar")
	}
}
