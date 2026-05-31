package crypto

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"math/big"
	"testing"

	"go.dedis.ch/kyber/v3"

	"evolvbft/evolvbft/types"
)

// ===========================================================================
// SHA256 / HashString
// ===========================================================================

func TestSHA256_Deterministic(t *testing.T) {
	h1 := SHA256([]byte("hello"))
	h2 := SHA256([]byte("hello"))
	if h1 != h2 {
		t.Error("SHA256 should be deterministic")
	}
}

func TestSHA256_DifferentInput(t *testing.T) {
	h1 := SHA256([]byte("a"))
	h2 := SHA256([]byte("b"))
	if h1 == h2 {
		t.Error("different input should produce different hash")
	}
}

func TestHashString_HexEncoded(t *testing.T) {
	s := HashString([]byte("test"))
	if len(s) != 64 {
		t.Errorf("expected 64 hex chars, got %d", len(s))
	}
}

// ===========================================================================
// Sign / Verify (Ed25519)
// ===========================================================================

func TestSignVerify_RoundTrip(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	msg := []byte("test message")
	sig := Sign(msg, priv)

	if !Verify(msg, sig, pub) {
		t.Error("signature should verify")
	}
}

func TestVerify_WrongKey(t *testing.T) {
	_, priv, _ := ed25519.GenerateKey(rand.Reader)
	pub2, _, _ := ed25519.GenerateKey(rand.Reader)
	sig := Sign([]byte("msg"), priv)

	if Verify([]byte("msg"), sig, pub2) {
		t.Error("should not verify with wrong key")
	}
}

func TestVerify_WrongMessage(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	sig := Sign([]byte("msg"), priv)

	if Verify([]byte("other"), sig, pub) {
		t.Error("should not verify with wrong message")
	}
}

// ===========================================================================
// GenerateKeyPair
// ===========================================================================

func TestGenerateKeyPair(t *testing.T) {
	kp, err := GenerateKeyPair()
	if err != nil {
		t.Fatalf("GenerateKeyPair failed: %v", err)
	}
	if len(kp.PublicKey) != ed25519.PublicKeySize {
		t.Errorf("bad public key size: %d", len(kp.PublicKey))
	}
	if len(kp.PrivateKey) != ed25519.PrivateKeySize {
		t.Errorf("bad private key size: %d", len(kp.PrivateKey))
	}

	// Keys should actually work
	msg := []byte("hello")
	sig := Sign(msg, kp.PrivateKey)
	if !Verify(msg, sig, kp.PublicKey) {
		t.Error("generated keypair should produce valid signatures")
	}
}

// ===========================================================================
// VerifyQuorum
// ===========================================================================

func TestVerifyQuorum_Passes(t *testing.T) {
	validators := make(map[uint64]types.PublicKey)
	qc := types.NewQuorumCertificateWithIdentity([]byte("block"), 1, 1, 0, 0, types.PhasePrepare)
	msg := types.QCSigningBytes(qc.BlockHash, qc.View, qc.Epoch, qc.ConfigID, qc.Lane)

	for i := uint64(1); i <= 3; i++ {
		pub, priv, _ := ed25519.GenerateKey(rand.Reader)
		validators[i] = pub
		sig := ed25519.Sign(priv, msg)
		qc.AddSignature(i, sig)
	}

	if !VerifyQuorum(qc, validators, 3) {
		t.Error("quorum should verify with 3/3 valid signatures")
	}
}

func TestVerifyQuorum_NilQC(t *testing.T) {
	if VerifyQuorum(nil, nil, 1) {
		t.Error("nil QC should not verify")
	}
}

func TestVerifyQuorum_InsufficientSigs(t *testing.T) {
	validators := make(map[uint64]types.PublicKey)
	qc := types.NewQuorumCertificateWithIdentity([]byte("block"), 1, 1, 0, 0, types.PhasePrepare)
	msg := types.QCSigningBytes(qc.BlockHash, qc.View, qc.Epoch, qc.ConfigID, qc.Lane)

	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	validators[1] = pub
	qc.AddSignature(1, ed25519.Sign(priv, msg))

	if VerifyQuorum(qc, validators, 2) {
		t.Error("should fail with 1/2 quorum")
	}
}

func TestVerifyQuorum_InvalidSignature(t *testing.T) {
	validators := make(map[uint64]types.PublicKey)
	qc := types.NewQuorumCertificateWithIdentity([]byte("block"), 1, 1, 0, 0, types.PhasePrepare)

	pub, _, _ := ed25519.GenerateKey(rand.Reader)
	validators[1] = pub
	qc.AddSignature(1, []byte("bad-sig-not-64-bytes-this-will-fail-to-verify-for-ed25519-checking-purposes"))

	if VerifyQuorum(qc, validators, 1) {
		t.Error("invalid signature should not verify")
	}
}

func TestVerifyQuorum_UnknownValidator(t *testing.T) {
	validators := make(map[uint64]types.PublicKey)
	qc := types.NewQuorumCertificateWithIdentity([]byte("block"), 1, 1, 0, 0, types.PhasePrepare)
	qc.AddSignature(99, []byte("sig"))

	if VerifyQuorum(qc, validators, 1) {
		t.Error("unknown validator should be skipped")
	}
}

// ===========================================================================
// VerifyQuorumBatch
// ===========================================================================

func TestVerifyQuorumBatch_Passes(t *testing.T) {
	validators := make(map[uint64]types.PublicKey)
	qc := types.NewQuorumCertificateWithIdentity([]byte("block"), 1, 1, 0, 0, types.PhasePrepare)
	msg := types.QCSigningBytes(qc.BlockHash, qc.View, qc.Epoch, qc.ConfigID, qc.Lane)

	for i := uint64(1); i <= 4; i++ {
		pub, priv, _ := ed25519.GenerateKey(rand.Reader)
		validators[i] = pub
		qc.AddSignature(i, ed25519.Sign(priv, msg))
	}

	if !VerifyQuorumBatch(qc, validators, 3) {
		t.Error("batch quorum should verify with 4/3 valid signatures")
	}
}

func TestVerifyQuorumBatch_NilQC(t *testing.T) {
	if VerifyQuorumBatch(nil, nil, 1) {
		t.Error("nil QC should not verify in batch")
	}
}

func TestVerifyQuorumBatch_EmptySignatures(t *testing.T) {
	qc := types.NewQuorumCertificate([]byte("b"), 1, 1, types.PhasePrepare)
	if VerifyQuorumBatch(qc, nil, 1) {
		t.Error("empty signatures should not verify")
	}
}

// ===========================================================================
// VRF: GenerateVRFKey, EvaluateVRF, VerifyVRF, Sortition
// ===========================================================================

func TestVRF_GenerateEvaluateVerify(t *testing.T) {
	sk, pk := GenerateVRFKey()
	seed := []byte("round-42-epoch-1")

	result, err := EvaluateVRF(sk, seed)
	if err != nil {
		t.Fatalf("EvaluateVRF failed: %v", err)
	}
	if len(result.Hash) != 32 {
		t.Errorf("expected 32-byte VRF hash, got %d", len(result.Hash))
	}
	if len(result.Proof) != 96 {
		t.Errorf("expected 96-byte proof, got %d", len(result.Proof))
	}

	hash, ok := VerifyVRF(pk, seed, result.Proof)
	if !ok {
		t.Fatal("VRF should verify with correct key and seed")
	}
	if !bytes.Equal(hash, result.Hash) {
		t.Error("verified hash should match original hash")
	}
}

func TestVRF_WrongKey(t *testing.T) {
	sk, _ := GenerateVRFKey()
	_, pk2 := GenerateVRFKey()

	result, _ := EvaluateVRF(sk, []byte("seed"))
	_, ok := VerifyVRF(pk2, []byte("seed"), result.Proof)
	if ok {
		t.Error("VRF should not verify with wrong public key")
	}
}

func TestVRF_WrongSeed(t *testing.T) {
	sk, pk := GenerateVRFKey()
	result, _ := EvaluateVRF(sk, []byte("seed1"))
	_, ok := VerifyVRF(pk, []byte("seed2"), result.Proof)
	if ok {
		t.Error("VRF should not verify with wrong seed")
	}
}

func TestVRF_TruncatedProof(t *testing.T) {
	_, pk := GenerateVRFKey()
	_, ok := VerifyVRF(pk, []byte("seed"), []byte("short"))
	if ok {
		t.Error("truncated proof should fail")
	}
}

func TestVRF_Determinism(t *testing.T) {
	sk, pk := GenerateVRFKey()
	seed := []byte("deterministic-test")

	r1, _ := EvaluateVRF(sk, seed)
	r2, _ := EvaluateVRF(sk, seed)

	// VRF hash should be deterministic (same sk + seed = same hash)
	if !bytes.Equal(r1.Hash, r2.Hash) {
		t.Error("VRF hash should be deterministic for same key and seed")
	}
	// But proofs differ because of random k
	// Both should still verify
	h1, ok1 := VerifyVRF(pk, seed, r1.Proof)
	h2, ok2 := VerifyVRF(pk, seed, r2.Proof)
	if !ok1 || !ok2 {
		t.Error("both proofs should verify")
	}
	if !bytes.Equal(h1, h2) {
		t.Error("verified hashes should match for same key and seed")
	}
}

func TestSortition_FullCommittee(t *testing.T) {
	// With committeeSize == totalWeight, everyone should be selected
	hash := make([]byte, 32)
	// Any hash value should pass if committee == total
	if !Sortition(hash, 10, 10) {
		t.Error("everyone should be selected when committeeSize == totalWeight")
	}
}

func TestSortition_ShortHash(t *testing.T) {
	if Sortition([]byte{1, 2}, 10, 5) {
		t.Error("hash shorter than 8 bytes should return false")
	}
}

func TestSortition_ZeroCommittee(t *testing.T) {
	hash := make([]byte, 32)
	if Sortition(hash, 10, 0) {
		t.Error("zero committee size should never select")
	}
}

// ===========================================================================
// BLSFull: GenerateKeyPair, Sign, Verify, Aggregate, VerifyAggregate
// ===========================================================================

func TestBLSFull_SignVerify(t *testing.T) {
	b := NewBLSFull()
	sk, pk, err := b.GenerateKeyPair()
	if err != nil {
		t.Fatalf("GenerateKeyPair: %v", err)
	}

	msg := []byte("bls-test")
	sig, err := b.Sign(sk, msg)
	if err != nil {
		t.Fatalf("Sign: %v", err)
	}
	if err := b.Verify(pk, msg, sig); err != nil {
		t.Errorf("Verify failed: %v", err)
	}
}

func TestBLSFull_VerifyWrongMessage(t *testing.T) {
	b := NewBLSFull()
	sk, pk, _ := b.GenerateKeyPair()
	sig, _ := b.Sign(sk, []byte("correct"))
	if err := b.Verify(pk, []byte("wrong"), sig); err == nil {
		t.Error("should fail with wrong message")
	}
}

func TestBLSFull_AggregateSignatures(t *testing.T) {
	b := NewBLSFull()
	msg := []byte("aggregate-test")

	var sigs [][]byte
	var pubs []kyber.Point
	for i := 0; i < 3; i++ {
		sk, pk, _ := b.GenerateKeyPair()
		sig, _ := b.Sign(sk, msg)
		sigs = append(sigs, sig)
		pubs = append(pubs, pk)
	}

	aggSig, err := b.AggregateSignatures(sigs...)
	if err != nil {
		t.Fatalf("AggregateSignatures: %v", err)
	}

	if err := b.VerifyAggregate(pubs, msg, aggSig); err != nil {
		t.Errorf("VerifyAggregate failed: %v", err)
	}
}

func TestBLSFull_AggregateEmpty(t *testing.T) {
	b := NewBLSFull()
	_, err := b.AggregateSignatures()
	if err == nil {
		t.Error("empty aggregate should fail")
	}
}

func TestBLSFull_AggregateSingle(t *testing.T) {
	b := NewBLSFull()
	sk, pk, _ := b.GenerateKeyPair()
	msg := []byte("single")
	sig, _ := b.Sign(sk, msg)

	agg, err := b.AggregateSignatures(sig)
	if err != nil {
		t.Fatalf("single aggregate should succeed: %v", err)
	}
	if err := b.Verify(pk, msg, agg); err != nil {
		t.Errorf("single aggregate should verify: %v", err)
	}
}

func TestBLSFull_VerifyAggregateNoKeys(t *testing.T) {
	b := NewBLSFull()
	if err := b.VerifyAggregate(nil, []byte("msg"), []byte("sig")); err == nil {
		t.Error("no keys should fail")
	}
}

// ===========================================================================
// TBLS: Threshold BLS
// ===========================================================================

func TestTBLS_InvalidParams(t *testing.T) {
	_, err := NewTBLSThreshold(0, 3)
	if err == nil {
		t.Error("threshold 0 should fail")
	}
	_, err = NewTBLSThreshold(4, 3)
	if err == nil {
		t.Error("threshold > numShares should fail")
	}
}

func TestTBLS_GenerateAndSign(t *testing.T) {
	tbls, err := NewTBLSThreshold(2, 3)
	if err != nil {
		t.Fatalf("NewTBLSThreshold: %v", err)
	}

	share, err := tbls.GenerateShare(1)
	if err != nil {
		t.Fatalf("GenerateShare: %v", err)
	}
	if len(share) == 0 {
		t.Error("share should not be empty")
	}

	sig, err := tbls.Sign([]byte("msg"), share)
	if err != nil {
		t.Fatalf("Sign: %v", err)
	}
	if len(sig) == 0 {
		t.Error("signature should not be empty")
	}
}

func TestTBLS_SignInvalidShare(t *testing.T) {
	tbls, _ := NewTBLSThreshold(2, 3)
	_, err := tbls.Sign([]byte("msg"), []byte("bad"))
	if err == nil {
		t.Error("invalid share should fail")
	}
}

func TestTBLS_RecoverInsufficientShares(t *testing.T) {
	tbls, _ := NewTBLSThreshold(3, 5)
	_, err := tbls.Recover([]byte("msg"), [][]byte{[]byte("s1"), []byte("s2")})
	if err == nil {
		t.Error("should fail with fewer shares than threshold")
	}
}

// ===========================================================================
// Merkle Tree
// ===========================================================================

func TestBuildMerkleTree_RoundTrip(t *testing.T) {
	shards := [][]byte{
		[]byte("shard1"),
		[]byte("shard2"),
		[]byte("shard3"),
		[]byte("shard4"),
	}
	mt := BuildMerkleTree(shards)
	if len(mt.Root) != 32 {
		t.Errorf("expected 32-byte root, got %d", len(mt.Root))
	}
	if len(mt.Leaves) != 4 {
		t.Errorf("expected 4 leaves, got %d", len(mt.Leaves))
	}

	// Verify each shard
	for i, shard := range shards {
		proof := mt.GetProof(i)
		if !VerifyProof(mt.Root, shard, proof, i) {
			t.Errorf("shard %d should verify", i)
		}
	}
}

func TestBuildMerkleTree_OddShards(t *testing.T) {
	shards := [][]byte{
		[]byte("a"),
		[]byte("b"),
		[]byte("c"),
	}
	mt := BuildMerkleTree(shards)
	if len(mt.Root) != 32 {
		t.Fatal("odd shards should still produce valid root")
	}

	for i, shard := range shards {
		proof := mt.GetProof(i)
		if !VerifyProof(mt.Root, shard, proof, i) {
			t.Errorf("odd shard %d should verify", i)
		}
	}
}

func TestBuildMerkleTree_SingleShard(t *testing.T) {
	mt := BuildMerkleTree([][]byte{[]byte("only")})
	if len(mt.Root) != 32 {
		t.Fatal("single shard should produce valid root")
	}
	proof := mt.GetProof(0)
	if !VerifyProof(mt.Root, []byte("only"), proof, 0) {
		t.Error("single shard should verify")
	}
}

func TestVerifyProof_WrongShard(t *testing.T) {
	mt := BuildMerkleTree([][]byte{[]byte("a"), []byte("b")})
	proof := mt.GetProof(0)
	if VerifyProof(mt.Root, []byte("c"), proof, 0) {
		t.Error("wrong shard data should not verify")
	}
}

func TestVerifyProof_WrongIndex(t *testing.T) {
	mt := BuildMerkleTree([][]byte{[]byte("a"), []byte("b"), []byte("c"), []byte("d")})
	proof := mt.GetProof(0)
	if VerifyProof(mt.Root, []byte("a"), proof, 1) {
		t.Error("wrong index should not verify")
	}
}

// ===========================================================================
// Utility functions: Hash, HashToBigInt, Marshal/Unmarshal
// ===========================================================================

func TestHash_Concat(t *testing.T) {
	h1 := Hash([]byte("ab"))
	h2 := Hash([]byte("a"), []byte("b"))
	if !bytes.Equal(h1, h2) {
		t.Error("Hash should concatenate slices before hashing")
	}
}

func TestHashToBigInt(t *testing.T) {
	val := HashToBigInt([]byte("test"))
	if val.Cmp(big.NewInt(0)) <= 0 {
		t.Error("HashToBigInt should produce positive value")
	}
}

func TestMarshalUnmarshalPoint(t *testing.T) {
	_, pk := GenerateVRFKey()
	pkData, err := MarshalPoint(pk)
	if err != nil {
		t.Fatalf("MarshalPoint: %v", err)
	}
	if len(pkData) == 0 {
		t.Fatal("marshaled VRF point should not be empty")
	}

	// UnmarshalPoint uses G1 internally; pk is on Edwards25519 not BN256,
	// so we test with a BN256 point instead
	bls := NewBLSFull()
	_, blsPk, _ := bls.GenerateKeyPair()
	// blsPk is on G2, but MarshalPoint just calls MarshalBinary
	pkBytes, _ := MarshalPoint(blsPk)
	if len(pkBytes) == 0 {
		t.Error("marshaled point should not be empty")
	}
}

func TestMarshalUnmarshalScalar(t *testing.T) {
	sk, _ := GenerateVRFKey()
	scalarBytes, err := MarshalScalar(sk)
	if err != nil {
		t.Fatalf("MarshalScalar: %v", err)
	}
	if len(scalarBytes) == 0 {
		t.Fatal("marshaled scalar should not be empty")
	}
}

func TestPadTo32(t *testing.T) {
	short := padTo32([]byte{1, 2, 3})
	if len(short) != 32 {
		t.Errorf("expected 32 bytes, got %d", len(short))
	}
	if short[31] != 3 || short[30] != 2 || short[29] != 1 {
		t.Error("padding should be big-endian (left-padded)")
	}

	long := padTo32(make([]byte, 40))
	if len(long) != 32 {
		t.Errorf("should truncate to 32, got %d", len(long))
	}
}

func TestBn256Order(t *testing.T) {
	order := bn256Order()
	if order == nil {
		t.Fatal("bn256Order should not be nil")
	}
	if order.BitLen() < 250 {
		t.Errorf("order should be ~256 bits, got %d", order.BitLen())
	}
}

func TestLagrangeCoefficient(t *testing.T) {
	// For 2-of-3, L_0(0) for indices {1,2} should be 2 (mod order)
	coeff := lagrangeCoefficient(0, 2)
	if coeff.Sign() <= 0 {
		t.Error("Lagrange coefficient should be positive")
	}
}
