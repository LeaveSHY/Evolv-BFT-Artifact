// Copyright 2024 Octopus Project
// Licensed under Apache License 2.0

package crypto

import (
	"encoding/binary"
	"math/big"

	"go.dedis.ch/kyber/v3"
	"go.dedis.ch/kyber/v3/group/edwards25519"
	"go.dedis.ch/kyber/v3/util/random"
)

var suite = edwards25519.NewBlakeSHA256Ed25519()

// VRFResult contains the hash and proof of a VRF evaluation
type VRFResult struct {
	Hash  []byte
	Proof []byte
}

// GenerateVRFKey generates a new key pair for VRF
func GenerateVRFKey() (kyber.Scalar, kyber.Point) {
	scalar := suite.Scalar().Pick(random.New())
	point := suite.Point().Mul(scalar, nil)
	return scalar, point
}

// EvaluateVRF calculates the VRF hash and proof
func EvaluateVRF(privateKey kyber.Scalar, seed []byte) (*VRFResult, error) {
	// Map seed to point M
	M := suite.Point().Pick(suite.XOF(seed))

	// Calculate V = M * sk
	V := suite.Point().Mul(privateKey, M)

	// k = random scalar
	k := suite.Scalar().Pick(random.New())

	// U = k * G
	U := suite.Point().Mul(k, nil)

	// W = k * M
	W := suite.Point().Mul(k, M)

	// PK = sk * G
	PK := suite.Point().Mul(privateKey, nil)

	// Calculate challenge c = Hash(G, M, PK, V, U, W)
	xof := suite.XOF(nil)
	gBytes, _ := suite.Point().Base().MarshalBinary()
	mBytes, _ := M.MarshalBinary()
	pkBytes, _ := PK.MarshalBinary()
	vBytes, _ := V.MarshalBinary()
	uBytes, _ := U.MarshalBinary()
	wBytes, _ := W.MarshalBinary()

	xof.Write(gBytes)
	xof.Write(mBytes)
	xof.Write(pkBytes)
	xof.Write(vBytes)
	xof.Write(uBytes)
	xof.Write(wBytes)

	c := suite.Scalar().Pick(xof)

	// s = k - c * sk
	csk := suite.Scalar().Mul(c, privateKey)
	s := suite.Scalar().Sub(k, csk)

	// Proof = V (32 bytes) + c (32 bytes) + s (32 bytes)
	cBytes, _ := c.MarshalBinary()
	sBytes, _ := s.MarshalBinary()

	proof := make([]byte, 0, 96)
	proof = append(proof, vBytes...)
	proof = append(proof, cBytes...)
	proof = append(proof, sBytes...)

	// Output hash = Hash(V)
	hash := suite.XOF(vBytes)
	h := make([]byte, 32)
	hash.Read(h)

	return &VRFResult{
		Hash:  h,
		Proof: proof,
	}, nil
}

// VerifyVRF verifies the VRF output
// Returns true if valid, false otherwise
func VerifyVRF(publicKey kyber.Point, seed []byte, proof []byte) ([]byte, bool) {
	if len(proof) != 96 {
		return nil, false
	}

	// Parse proof into V, c, s
	V := suite.Point()
	if err := V.UnmarshalBinary(proof[:32]); err != nil {
		return nil, false
	}

	c := suite.Scalar()
	if err := c.UnmarshalBinary(proof[32:64]); err != nil {
		return nil, false
	}

	s := suite.Scalar()
	if err := s.UnmarshalBinary(proof[64:96]); err != nil {
		return nil, false
	}

	// Reconstruct M from seed
	M := suite.Point().Pick(suite.XOF(seed))

	// U' = s*G + c*PK
	sG := suite.Point().Mul(s, nil)
	cPK := suite.Point().Mul(c, publicKey)
	U_prime := suite.Point().Add(sG, cPK)

	// W' = s*M + c*V
	sM := suite.Point().Mul(s, M)
	cV := suite.Point().Mul(c, V)
	W_prime := suite.Point().Add(sM, cV)

	// Recompute c' = Hash(G, M, PK, V, U', W')
	xof := suite.XOF(nil)
	gBytes, _ := suite.Point().Base().MarshalBinary()
	mBytes, _ := M.MarshalBinary()
	pkBytes, _ := publicKey.MarshalBinary()
	vBytes, _ := V.MarshalBinary()
	uBytes, _ := U_prime.MarshalBinary()
	wBytes, _ := W_prime.MarshalBinary()

	xof.Write(gBytes)
	xof.Write(mBytes)
	xof.Write(pkBytes)
	xof.Write(vBytes)
	xof.Write(uBytes)
	xof.Write(wBytes)

	c_prime := suite.Scalar().Pick(xof)

	// Verify c' == c
	if !c.Equal(c_prime) {
		return nil, false
	}

	// Hash V to get the output
	hash := suite.XOF(vBytes)
	h := make([]byte, 32)
	hash.Read(h)

	return h, true
}

// Sortition checks if the VRF hash allows the node to be in the committee
// Returns true if selected
func Sortition(vrfHash []byte, totalWeight uint64, committeeSize int) bool {
	// Interpret hash as integer
	// We use the first 8 bytes for a uint64
	if len(vrfHash) < 8 {
		return false
	}

	val := binary.BigEndian.Uint64(vrfHash[:8])

	// Threshold = (2^64 * k) / N
	// We use big.Int to avoid overflow
	max := new(big.Int).Lsh(big.NewInt(1), 64)
	k := big.NewInt(int64(committeeSize))
	n := big.NewInt(int64(totalWeight))

	// threshold = (max * k) / n
	threshold := new(big.Int).Mul(max, k)
	threshold.Div(threshold, n)

	// Check if val < threshold
	valBig := new(big.Int).SetUint64(val)
	return valBig.Cmp(threshold) < 0
}
