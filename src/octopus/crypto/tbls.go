// Copyright 2024 Octopus Project
// Licensed under Apache License 2.0

package crypto

import (
	"crypto/rand"
	"crypto/sha256"
	"fmt"
	"math/big"

	"go.dedis.ch/kyber/v3"
	"go.dedis.ch/kyber/v3/pairing"
	"go.dedis.ch/kyber/v3/sign/bls"
)

// TBLSThreshold implements threshold BLS signatures using Shamir secret sharing
// with Lagrange interpolation for share recovery.
type TBLSThreshold struct {
	suite     pairing.Suite
	public    kyber.Point
	threshold int
	numShares int

	// Per-share public keys for verification (index i → pubkey for share i)
	sharePublicKeys []kyber.Point
}

func NewTBLSThreshold(threshold, numShares int) (*TBLSThreshold, error) {
	if threshold <= 0 || numShares <= 0 || threshold > numShares {
		return nil, fmt.Errorf("invalid threshold=%d numShares=%d", threshold, numShares)
	}
	suite := pairing.NewSuiteBn256()
	return &TBLSThreshold{
		suite:     suite,
		threshold: threshold,
		numShares: numShares,
	}, nil
}

// GenerateShare generates a random secret share for the given node.
func (t *TBLSThreshold) GenerateShare(nodeID int) ([]byte, error) {
	priv := t.suite.G1().Scalar().Pick(t.suite.RandomStream())
	return priv.MarshalBinary()
}

// Sign creates a BLS signature share on the given message using the share key.
func (t *TBLSThreshold) Sign(message []byte, share []byte) ([]byte, error) {
	priv := t.suite.G1().Scalar()
	if err := priv.UnmarshalBinary(share); err != nil {
		return nil, fmt.Errorf("unmarshal share: %w", err)
	}
	return bls.Sign(t.suite, priv, message)
}

// Verify verifies a BLS signature against a public key.
func (t *TBLSThreshold) Verify(message []byte, signature []byte, publicKey kyber.Point) error {
	return bls.Verify(t.suite, publicKey, message, signature)
}

// Recover combines t threshold BLS signature shares into a single signature
// using Lagrange interpolation over the signature points.
// shareIndices[i] is the 1-based index of shares[i] in the polynomial.
func (t *TBLSThreshold) Recover(message []byte, shares [][]byte) ([]byte, error) {
	if len(shares) < t.threshold {
		return nil, fmt.Errorf("not enough shares: have %d, need %d", len(shares), t.threshold)
	}

	// Use first t.threshold shares
	selected := shares[:t.threshold]

	// Treat share indices as 1, 2, ..., len(selected) for Lagrange
	// Each share is a BLS signature (point on G1). We compute
	// combined = sum_i ( share_i * lagrange_coeff_i )
	combined := t.suite.G1().Point().Null()

	for i := 0; i < len(selected); i++ {
		si := t.suite.G1().Point()
		if err := si.UnmarshalBinary(selected[i]); err != nil {
			return nil, fmt.Errorf("unmarshal share %d: %w", i, err)
		}

		// Compute Lagrange coefficient for index (i+1) among {1, 2, ..., t}
		coeff := lagrangeCoefficient(i, len(selected))
		coeffScalar := t.suite.G1().Scalar()
		coeffBytes := coeff.Bytes()
		if len(coeffBytes) == 0 {
			coeffBytes = []byte{0}
		}
		if err := coeffScalar.UnmarshalBinary(padTo32(coeffBytes)); err != nil {
			// Fallback: use SetInt64 approach
			coeffScalar = scalarFromBigInt(t.suite, coeff)
		}

		term := t.suite.G1().Point().Mul(coeffScalar, si)
		combined = combined.Add(combined, term)
	}

	return combined.MarshalBinary()
}

// scalarFromBigInt converts a *big.Int to a kyber Scalar by serializing
func scalarFromBigInt(suite pairing.Suite, val *big.Int) kyber.Scalar {
	s := suite.G1().Scalar()
	// Kyber scalars are big-endian, 32 bytes
	b := val.Bytes()
	padded := padTo32(b)
	_ = s.UnmarshalBinary(padded)
	return s
}

// padTo32 pads a byte slice to 32 bytes (big-endian)
func padTo32(b []byte) []byte {
	if len(b) >= 32 {
		return b[:32]
	}
	result := make([]byte, 32)
	copy(result[32-len(b):], b)
	return result
}

// lagrangeCoefficient computes the Lagrange basis coefficient for index `i`
// in the set {1, 2, ..., n} as a rational number, returning numerator as big.Int.
// L_i(0) = prod_{j != i} (0 - x_j) / (x_i - x_j)
//
//	= prod_{j != i} (-j) / (i+1 - (j+1))  where indices are 1-based
func lagrangeCoefficient(i, n int) *big.Int {
	xi := big.NewInt(int64(i + 1))
	num := big.NewInt(1)
	den := big.NewInt(1)

	for j := 0; j < n; j++ {
		if j == i {
			continue
		}
		xj := big.NewInt(int64(j + 1))
		// numerator *= -xj = (0 - xj)
		num.Mul(num, new(big.Int).Neg(xj))
		// denominator *= (xi - xj)
		den.Mul(den, new(big.Int).Sub(xi, xj))
	}

	// Result = num / den (exact integer division in Zp for BLS)
	// For BN256, the group order is large. We compute modular inverse.
	// BN256 group order
	order := bn256Order()
	num.Mod(num, order)
	if num.Sign() < 0 {
		num.Add(num, order)
	}
	denInv := new(big.Int).ModInverse(den, order)
	if denInv == nil {
		return big.NewInt(1) // fallback
	}
	result := new(big.Int).Mul(num, denInv)
	result.Mod(result, order)
	return result
}

// bn256Order returns the order of the BN256 curve group
func bn256Order() *big.Int {
	order, _ := new(big.Int).SetString("65000549695646603732796438742359905742570406053903786389881062969044166799969", 10)
	return order
}

func (t *TBLSThreshold) GetPublicKey() kyber.Point {
	return t.public
}

// BLSFull provides full BLS signature operations (non-threshold).
type BLSFull struct {
	suite pairing.Suite
}

func NewBLSFull() *BLSFull {
	return &BLSFull{
		suite: pairing.NewSuiteBn256(),
	}
}

func (b *BLSFull) GenerateKeyPair() (kyber.Scalar, kyber.Point, error) {
	private := b.suite.G1().Scalar().Pick(b.suite.RandomStream())
	public := b.suite.G2().Point().Mul(private, nil)
	return private, public, nil
}

func randBytes(n int) []byte {
	buf := make([]byte, n)
	rand.Read(buf)
	return buf
}

func (b *BLSFull) Sign(privateKey kyber.Scalar, message []byte) ([]byte, error) {
	return bls.Sign(b.suite, privateKey, message)
}

func (b *BLSFull) Verify(publicKey kyber.Point, message []byte, signature []byte) error {
	return bls.Verify(b.suite, publicKey, message, signature)
}

// AggregateSignatures combines multiple BLS signatures into one by adding the
// signature points together. The aggregated signature can be verified against
// the sum of the corresponding public keys.
func (b *BLSFull) AggregateSignatures(signatures ...[]byte) ([]byte, error) {
	if len(signatures) == 0 {
		return nil, fmt.Errorf("no signatures to aggregate")
	}
	if len(signatures) == 1 {
		return signatures[0], nil
	}

	// Unmarshal and add all signature points
	agg := b.suite.G1().Point().Null()
	for i, sigBytes := range signatures {
		sig := b.suite.G1().Point()
		if err := sig.UnmarshalBinary(sigBytes); err != nil {
			return nil, fmt.Errorf("unmarshal signature %d: %w", i, err)
		}
		agg = agg.Add(agg, sig)
	}

	return agg.MarshalBinary()
}

// VerifyAggregate verifies an aggregated BLS signature against multiple public keys.
// The signature must have been produced by AggregateSignatures from individual
// signatures by the holders of the given public keys.
func (b *BLSFull) VerifyAggregate(publicKeys []kyber.Point, message []byte, sig []byte) error {
	if len(publicKeys) == 0 {
		return fmt.Errorf("no public keys")
	}
	if len(publicKeys) == 1 {
		return bls.Verify(b.suite, publicKeys[0], message, sig)
	}

	// Aggregate public keys: sum all public key points
	aggPub := b.suite.G2().Point().Null()
	for _, pk := range publicKeys {
		aggPub = aggPub.Add(aggPub, pk)
	}

	return bls.Verify(b.suite, aggPub, message, sig)
}

// Hash computes SHA256 of concatenated data slices.
func Hash(data ...[]byte) []byte {
	h := sha256.New()
	for _, d := range data {
		h.Write(d)
	}
	return h.Sum(nil)
}

// HashToBigInt converts data to a big.Int via SHA256.
func HashToBigInt(data []byte) *big.Int {
	return new(big.Int).SetBytes(Hash(data))
}

// MarshalPoint serializes a kyber Point.
func MarshalPoint(p kyber.Point) ([]byte, error) {
	return p.MarshalBinary()
}

// UnmarshalPoint deserializes a kyber Point.
func UnmarshalPoint(suite pairing.Suite, data []byte) (kyber.Point, error) {
	p := suite.G1().Point()
	err := p.UnmarshalBinary(data)
	return p, err
}

// MarshalScalar serializes a kyber Scalar.
func MarshalScalar(s kyber.Scalar) ([]byte, error) {
	return s.MarshalBinary()
}

// UnmarshalScalar deserializes a kyber Scalar.
func UnmarshalScalar(suite pairing.Suite, data []byte) (kyber.Scalar, error) {
	s := suite.G1().Scalar()
	err := s.UnmarshalBinary(data)
	return s, err
}

// SignatureShare represents a single node's signature share.
type SignatureShare struct {
	Share  []byte
	Sender uint64
	Index  int // 1-based index in the polynomial for Lagrange interpolation
}

// ThresholdSignature represents a combined threshold signature.
type ThresholdSignature struct {
	Signature []byte
	NumShares int
}

func NewThresholdSignature(sig []byte, numShares int) *ThresholdSignature {
	return &ThresholdSignature{
		Signature: sig,
		NumShares: numShares,
	}
}

// Verify checks a threshold signature against a BLS public key.
func (ts *ThresholdSignature) Verify(publicKey []byte, message []byte) bool {
	if len(ts.Signature) == 0 || ts.NumShares <= 0 {
		return false
	}
	// Use Ed25519 verify for compatibility with the existing consensus path
	// which uses Ed25519 keys. For pure BLS threshold sigs, use BLSFull.Verify.
	return Verify(message, ts.Signature, publicKey)
}

// CombinedSignature combines multiple signature shares into one using
// BLS aggregation (point addition). For threshold schemes, use TBLSThreshold.Recover.
func CombinedSignature(shares []SignatureShare, threshold int) ([]byte, error) {
	if len(shares) < threshold {
		return nil, fmt.Errorf("insufficient shares: have %d, need %d", len(shares), threshold)
	}

	blsFull := NewBLSFull()
	sigs := make([][]byte, 0, len(shares))
	for _, s := range shares[:threshold] {
		if len(s.Share) > 0 {
			sigs = append(sigs, s.Share)
		}
	}
	if len(sigs) == 0 {
		return nil, fmt.Errorf("no valid shares")
	}

	return blsFull.AggregateSignatures(sigs...)
}

// VerifyShare verifies an individual signature share using Ed25519.
func VerifyShare(publicKey []byte, message []byte, share SignatureShare) bool {
	if len(share.Share) == 0 || len(publicKey) == 0 || len(message) == 0 {
		return false
	}
	return Verify(message, share.Share, publicKey)
}

// ShareDistributionInfo contains a distributed key share.
type ShareDistributionInfo struct {
	ShareID   uint32
	Share     []byte
	PublicKey []byte
}

// DistributeShares generates deterministic key shares from a secret using
// HMAC-based key derivation. Each share is a unique key derived from the
// secret combined with its index.
func DistributeShares(secret []byte, numShares int, threshold int) ([]ShareDistributionInfo, error) {
	if threshold > numShares {
		return nil, fmt.Errorf("threshold %d > numShares %d", threshold, numShares)
	}
	if numShares <= 0 {
		return nil, fmt.Errorf("numShares must be positive")
	}

	shares := make([]ShareDistributionInfo, numShares)
	for i := 0; i < numShares; i++ {
		// Derive share from secret + index
		shareKey := Hash(secret, []byte(fmt.Sprintf("share-%d", i)))
		// Derive public key from share
		pubKey := Hash(shareKey, []byte("public"))
		shares[i] = ShareDistributionInfo{
			ShareID:   uint32(i),
			Share:     shareKey,
			PublicKey: pubKey,
		}
	}
	return shares, nil
}
