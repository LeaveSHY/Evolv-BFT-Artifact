// Copyright 2024 Octopus Project
// Licensed under Apache License 2.0

package crypto

import (
	"bytes"
	"crypto/sha256"
	// "fmt"
)

// MerkleTree represents a Merkle Tree
type MerkleTree struct {
	Root   []byte
	Leaves [][]byte
	Layers [][][]byte
}

// BuildMerkleTree builds a Merkle Tree from data shards
func BuildMerkleTree(shards [][]byte) *MerkleTree {
	leaves := make([][]byte, len(shards))
	for i, shard := range shards {
		hash := sha256.Sum256(shard)
		leaves[i] = hash[:]
	}

	layers := [][][]byte{leaves}
	currentLayer := leaves

	for len(currentLayer) > 1 {
		var nextLayer [][]byte
		for i := 0; i < len(currentLayer); i += 2 {
			var combined []byte
			if i+1 < len(currentLayer) {
				combined = append(currentLayer[i], currentLayer[i+1]...)
			} else {
				combined = append(currentLayer[i], currentLayer[i]...) // Duplicate last if odd
			}
			hash := sha256.Sum256(combined)
			nextLayer = append(nextLayer, hash[:])
		}
		layers = append(layers, nextLayer)
		currentLayer = nextLayer
	}

	root := currentLayer[0]
	return &MerkleTree{
		Root:   root,
		Leaves: leaves,
		Layers: layers,
	}
}

// GetProof generates a Merkle Proof for a given leaf index
func (mt *MerkleTree) GetProof(index int) [][]byte {
	var proof [][]byte
	currentLayerIndex := index

	for _, layer := range mt.Layers[:len(mt.Layers)-1] { // Skip root layer
		var siblingIndex int
		if currentLayerIndex%2 == 0 {
			siblingIndex = currentLayerIndex + 1
		} else {
			siblingIndex = currentLayerIndex - 1
		}

		if siblingIndex < len(layer) {
			proof = append(proof, layer[siblingIndex])
		} else {
			// If odd, sibling is self
			proof = append(proof, layer[currentLayerIndex])
		}
		currentLayerIndex /= 2
	}
	return proof
}

// VerifyProof verifies a Merkle Proof
func VerifyProof(root []byte, shard []byte, proof [][]byte, index int) bool {
	hash := sha256.Sum256(shard)
	currentHash := hash[:]

	for _, sibling := range proof {
		var combined []byte
		if index%2 == 0 {
			combined = append(currentHash, sibling...)
		} else {
			combined = append(sibling, currentHash...)
		}
		newHash := sha256.Sum256(combined)
		currentHash = newHash[:]
		index /= 2
	}

	return bytes.Equal(currentHash, root)
}
