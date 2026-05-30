package beacon

import (
	"bytes"
	"sync"
	"testing"

	"octopus-bft/octopus/crypto"
)

// ---------------------------------------------------------------------------
// RandomBeacon: UpdateRandomness / GetCurrentSeed
// ---------------------------------------------------------------------------

func TestGetCurrentSeed_ReturnsInitial(t *testing.T) {
	seed := []byte("initial-seed-value")
	rb := NewRandomBeacon(seed)
	got := rb.GetCurrentSeed()
	if !bytes.Equal(got, seed) {
		t.Errorf("expected initial seed %x, got %x", seed, got)
	}
}

func TestUpdateRandomness_ChangesSeed(t *testing.T) {
	seed := []byte("some-seed")
	rb := NewRandomBeacon(seed)

	rb.UpdateRandomness([]byte("aggregated-sig"))
	after := rb.GetCurrentSeed()
	if bytes.Equal(after, seed) {
		t.Error("seed should change after UpdateRandomness")
	}
	// SHA256 output should be 32 bytes
	if len(after) != 32 {
		t.Errorf("expected 32-byte seed, got %d bytes", len(after))
	}
}

func TestUpdateRandomness_Deterministic(t *testing.T) {
	sig := []byte("same-sig")
	rb1 := NewRandomBeacon([]byte("seed"))
	rb2 := NewRandomBeacon([]byte("seed"))

	rb1.UpdateRandomness(sig)
	rb2.UpdateRandomness(sig)

	if !bytes.Equal(rb1.GetCurrentSeed(), rb2.GetCurrentSeed()) {
		t.Error("same initial seed + same sig should produce identical seeds")
	}
}

func TestUpdateRandomness_DifferentSigsDifferentSeeds(t *testing.T) {
	rb1 := NewRandomBeacon([]byte("seed"))
	rb2 := NewRandomBeacon([]byte("seed"))

	rb1.UpdateRandomness([]byte("sig-A"))
	rb2.UpdateRandomness([]byte("sig-B"))

	if bytes.Equal(rb1.GetCurrentSeed(), rb2.GetCurrentSeed()) {
		t.Error("different sigs should yield different seeds")
	}
}

func TestUpdateRandomness_Chained(t *testing.T) {
	rb := NewRandomBeacon([]byte("chain-test"))
	seeds := make([][]byte, 0, 5)

	for i := 0; i < 5; i++ {
		rb.UpdateRandomness([]byte{byte(i)})
		s := make([]byte, len(rb.GetCurrentSeed()))
		copy(s, rb.GetCurrentSeed())
		seeds = append(seeds, s)
	}

	// All 5 seeds should be distinct
	for i := 0; i < len(seeds); i++ {
		for j := i + 1; j < len(seeds); j++ {
			if bytes.Equal(seeds[i], seeds[j]) {
				t.Errorf("chained seeds %d and %d collide", i, j)
			}
		}
	}
}

// ---------------------------------------------------------------------------
// RandomBeacon: AmICommitteeMember / VerifyCommitteeMember
// ---------------------------------------------------------------------------

func TestAmICommitteeMember_RoundTrip(t *testing.T) {
	rb := NewRandomBeacon([]byte("vrf-test-seed"))
	privKey, pubKey := crypto.GenerateVRFKey()

	// Large committeeSize / totalWeight ratio to make selection very likely
	totalWeight := uint64(10)
	committeeSize := 10 // everyone selected

	selected, hash, proof := rb.AmICommitteeMember(privKey, totalWeight, committeeSize)
	if !selected {
		t.Skip("VRF did not select us (committeeSize==totalWeight should be near-certain)")
	}
	if hash == nil || proof == nil {
		t.Fatal("selected but hash or proof is nil")
	}

	// Verify with matching public key
	valid := rb.VerifyCommitteeMember(pubKey, proof, totalWeight, committeeSize)
	if !valid {
		t.Error("VerifyCommitteeMember should accept valid proof")
	}
}

func TestVerifyCommitteeMember_InvalidProof(t *testing.T) {
	rb := NewRandomBeacon([]byte("tamper-test"))
	_, pubKey := crypto.GenerateVRFKey()

	// Tampered proof (random bytes, wrong length or content)
	badProof := make([]byte, 96)
	for i := range badProof {
		badProof[i] = byte(i)
	}

	valid := rb.VerifyCommitteeMember(pubKey, badProof, 10, 10)
	if valid {
		t.Error("VerifyCommitteeMember should reject invalid proof")
	}
}

func TestVerifyCommitteeMember_WrongSeed(t *testing.T) {
	rb1 := NewRandomBeacon([]byte("seed-1"))
	rb2 := NewRandomBeacon([]byte("seed-2"))
	privKey, pubKey := crypto.GenerateVRFKey()

	selected, _, proof := rb1.AmICommitteeMember(privKey, 10, 10)
	if !selected {
		t.Skip("VRF not selected")
	}

	// Verify with rb2 (different seed) should fail
	valid := rb2.VerifyCommitteeMember(pubKey, proof, 10, 10)
	if valid {
		t.Error("VerifyCommitteeMember should reject proof from different seed")
	}
}

func TestVerifyCommitteeMember_WrongPubKey(t *testing.T) {
	rb := NewRandomBeacon([]byte("wrong-key-test"))
	privKey, _ := crypto.GenerateVRFKey()
	_, otherPub := crypto.GenerateVRFKey() // different key pair

	selected, _, proof := rb.AmICommitteeMember(privKey, 10, 10)
	if !selected {
		t.Skip("VRF not selected")
	}

	valid := rb.VerifyCommitteeMember(otherPub, proof, 10, 10)
	if valid {
		t.Error("VerifyCommitteeMember should reject proof for wrong public key")
	}
}

func TestAmICommitteeMember_SmallCommittee(t *testing.T) {
	rb := NewRandomBeacon([]byte("small-committee"))
	privKey, _ := crypto.GenerateVRFKey()

	// committeeSize=1, totalWeight=1000 → very low selection probability
	totalWeight := uint64(1000)
	committeeSize := 1

	// Run multiple times to verify it doesn't panic; selection is unlikely
	selected := false
	for i := 0; i < 10; i++ {
		s, _, _ := rb.AmICommitteeMember(privKey, totalWeight, committeeSize)
		if s {
			selected = true
		}
		rb.UpdateRandomness([]byte{byte(i)})
	}
	// Just ensure no panic; selection outcome is probabilistic
	_ = selected
}

// ---------------------------------------------------------------------------
// BLSBeacon
// ---------------------------------------------------------------------------

func TestBLSBeacon_GenerateKey(t *testing.T) {
	b := NewBLSBeacon()
	kp, err := b.GenerateKey(1)
	if err != nil {
		t.Fatalf("GenerateKey failed: %v", err)
	}
	if kp.Private == nil || kp.Public == nil {
		t.Fatal("key pair should have non-nil private and public keys")
	}
	if !b.HasKey(1) {
		t.Error("HasKey should return true after GenerateKey")
	}
	pub, ok := b.GetPublicKey(1)
	if !ok || pub == nil {
		t.Error("GetPublicKey should return the generated key")
	}
}

func TestBLSBeacon_GenerateKey_Overwrite(t *testing.T) {
	b := NewBLSBeacon()
	kp1, _ := b.GenerateKey(1)
	kp2, err := b.GenerateKey(1)
	if err != nil {
		t.Fatalf("second GenerateKey failed: %v", err)
	}
	// Keys should differ (fresh random each time)
	pub1, _ := kp1.Public.MarshalBinary()
	pub2, _ := kp2.Public.MarshalBinary()
	if bytes.Equal(pub1, pub2) {
		t.Error("regenerated key should differ from original")
	}
}

func TestBLSBeacon_RegisterKey(t *testing.T) {
	b := NewBLSBeacon()

	// Generate externally and register only pubkey
	kp, _ := b.GenerateKey(99)
	b.RegisterKey(42, kp.Public)

	if !b.HasKey(42) {
		t.Error("HasKey should be true after RegisterKey")
	}
	pub, ok := b.GetPublicKey(42)
	if !ok || pub == nil {
		t.Error("GetPublicKey should return registered key")
	}
}

func TestBLSBeacon_Sign_Success(t *testing.T) {
	b := NewBLSBeacon()
	b.GenerateKey(1)

	sig, err := b.Sign(1, []byte("hello"))
	if err != nil {
		t.Fatalf("Sign failed: %v", err)
	}
	if len(sig) == 0 {
		t.Error("signature should not be empty")
	}
}

func TestBLSBeacon_Sign_NoKey(t *testing.T) {
	b := NewBLSBeacon()
	_, err := b.Sign(999, []byte("hello"))
	if err == nil {
		t.Error("Sign should fail for unregistered validator")
	}
}

func TestBLSBeacon_Sign_NilPrivate(t *testing.T) {
	b := NewBLSBeacon()
	// Register only a public key (no private)
	kp, _ := b.GenerateKey(1)
	b.RegisterKey(2, kp.Public) // only pubkey for validator 2

	_, err := b.Sign(2, []byte("hello"))
	if err == nil {
		t.Error("Sign should fail when only public key is registered")
	}
}

func TestBLSBeacon_AggregateAndVerify_Success(t *testing.T) {
	b := NewBLSBeacon()
	msg := []byte("consensus-round-42")
	ids := []uint64{1, 2, 3}

	for _, id := range ids {
		if _, err := b.GenerateKey(id); err != nil {
			t.Fatalf("GenerateKey(%d): %v", id, err)
		}
	}

	sigs := make(map[uint64][]byte)
	for _, id := range ids {
		sig, err := b.Sign(id, msg)
		if err != nil {
			t.Fatalf("Sign(%d): %v", id, err)
		}
		sigs[id] = sig
	}

	aggSig, err := b.AggregateAndVerify(msg, sigs)
	if err != nil {
		t.Fatalf("AggregateAndVerify failed: %v", err)
	}
	if len(aggSig) == 0 {
		t.Error("aggregated signature should not be empty")
	}
}

func TestBLSBeacon_AggregateAndVerify_EmptySigs(t *testing.T) {
	b := NewBLSBeacon()
	_, err := b.AggregateAndVerify([]byte("msg"), map[uint64][]byte{})
	if err == nil {
		t.Error("AggregateAndVerify should fail with empty sigs")
	}
}

func TestBLSBeacon_AggregateAndVerify_PartialSigners(t *testing.T) {
	b := NewBLSBeacon()
	msg := []byte("partial-test")

	// Generate keys for 1,2,3 but provide sig for 1,2,999
	for _, id := range []uint64{1, 2, 3} {
		b.GenerateKey(id)
	}

	sigs := make(map[uint64][]byte)
	for _, id := range []uint64{1, 2} {
		sig, _ := b.Sign(id, msg)
		sigs[id] = sig
	}
	// Add a sig from an unregistered validator (999)
	sigs[999] = []byte("fake-sig")

	// Should still succeed with the 2 valid signers (999 skipped)
	aggSig, err := b.AggregateAndVerify(msg, sigs)
	if err != nil {
		t.Fatalf("AggregateAndVerify should succeed with partial signers: %v", err)
	}
	if len(aggSig) == 0 {
		t.Error("aggregated signature should not be empty")
	}
}

func TestBLSBeacon_HasKey_NotRegistered(t *testing.T) {
	b := NewBLSBeacon()
	if b.HasKey(42) {
		t.Error("HasKey should be false for unregistered validator")
	}
}

func TestBLSBeacon_GetPublicKey_NotRegistered(t *testing.T) {
	b := NewBLSBeacon()
	pub, ok := b.GetPublicKey(42)
	if ok || pub != nil {
		t.Error("GetPublicKey should return (nil, false) for unregistered validator")
	}
}

func TestBLSBeacon_RetainKeys(t *testing.T) {
	b := NewBLSBeacon()
	for _, id := range []uint64{1, 2, 3, 4, 5} {
		b.GenerateKey(id)
	}

	active := map[uint64]bool{1: true, 3: true, 5: true}
	b.RetainKeys(active)

	for _, id := range []uint64{1, 3, 5} {
		if !b.HasKey(id) {
			t.Errorf("validator %d should still have key", id)
		}
	}
	for _, id := range []uint64{2, 4} {
		if b.HasKey(id) {
			t.Errorf("validator %d key should have been removed", id)
		}
	}
}

func TestBLSBeacon_RetainKeys_EmptySet(t *testing.T) {
	b := NewBLSBeacon()
	b.GenerateKey(1)
	b.GenerateKey(2)

	b.RetainKeys(map[uint64]bool{})

	if b.HasKey(1) || b.HasKey(2) {
		t.Error("all keys should be removed when retaining empty set")
	}
}

func TestBLSBeacon_Concurrency(t *testing.T) {
	b := NewBLSBeacon()
	var wg sync.WaitGroup
	const n = 20

	// Concurrent GenerateKey
	for i := uint64(0); i < n; i++ {
		wg.Add(1)
		go func(id uint64) {
			defer wg.Done()
			b.GenerateKey(id)
		}(i)
	}
	wg.Wait()

	// Concurrent Sign + HasKey + GetPublicKey
	for i := uint64(0); i < n; i++ {
		wg.Add(1)
		go func(id uint64) {
			defer wg.Done()
			b.HasKey(id)
			b.GetPublicKey(id)
			b.Sign(id, []byte("concurrent-msg"))
		}(i)
	}
	wg.Wait()

	// All keys should exist
	for i := uint64(0); i < n; i++ {
		if !b.HasKey(i) {
			t.Errorf("validator %d key missing after concurrent GenerateKey", i)
		}
	}
}
