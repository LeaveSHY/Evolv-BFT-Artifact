package hotstuff

import (
	"testing"

	"octopus-bft/octopus/crypto"
)

func TestEncodeDecodeVRFRegistration_RoundTrip(t *testing.T) {
	_, pubKey := crypto.GenerateVRFKey()
	const validatorID uint64 = 42

	data, err := EncodeVRFRegistration(validatorID, pubKey)
	if err != nil {
		t.Fatalf("encode failed: %v", err)
	}
	if len(data) == 0 {
		t.Fatal("encoded data should not be empty")
	}

	gotID, gotPub, err := DecodeVRFRegistration(data)
	if err != nil {
		t.Fatalf("decode failed: %v", err)
	}
	if gotID != validatorID {
		t.Errorf("expected validator ID %d, got %d", validatorID, gotID)
	}

	// Compare marshaled bytes
	origBytes, _ := pubKey.MarshalBinary()
	gotBytes, _ := gotPub.MarshalBinary()
	if len(origBytes) != len(gotBytes) {
		t.Fatalf("public key length mismatch: %d vs %d", len(origBytes), len(gotBytes))
	}
	for i := range origBytes {
		if origBytes[i] != gotBytes[i] {
			t.Fatalf("public key byte %d mismatch", i)
		}
	}
}

func TestDecodeVRFRegistration_InvalidData(t *testing.T) {
	_, _, err := DecodeVRFRegistration([]byte("not-json"))
	if err == nil {
		t.Error("should fail on invalid JSON")
	}
}

func TestDecodeVRFRegistration_InvalidPubKey(t *testing.T) {
	data := []byte(`{"validator_id":1,"vrf_pub_key":"AAAA"}`)
	_, _, err := DecodeVRFRegistration(data)
	if err == nil {
		t.Error("should fail on invalid pubkey bytes")
	}
}

func TestDecodeVRFPublicKey_Valid(t *testing.T) {
	_, pubKey := crypto.GenerateVRFKey()
	raw, _ := pubKey.MarshalBinary()

	decoded, err := DecodeVRFPublicKey(raw)
	if err != nil {
		t.Fatalf("decode failed: %v", err)
	}
	if decoded == nil {
		t.Fatal("decoded key should not be nil")
	}
}

func TestDecodeVRFPublicKey_Invalid(t *testing.T) {
	_, err := DecodeVRFPublicKey([]byte{0, 1, 2})
	if err == nil {
		t.Error("should fail on invalid raw bytes")
	}
}
