package bootstrap

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// TestMultiNodeManifest_SharedIdentityEnablesCrossSigning verifies that
// when multiple nodes load the same genesis manifest, they share consistent
// public keys, making cross-node signature verification possible.
// This is the core fix for the ephemeral bootstrap problem where each node
// generated independent dummy keys, causing signature verification failures.
func TestMultiNodeManifest_SharedIdentityEnablesCrossSigning(t *testing.T) {
	numNodes := 4
	manifest := generateDeterministicManifest(t, numNodes, "crosssign-test")
	manifestPath := writeManifestFile(t, manifest)

	// Each node loads the manifest independently.
	keypairs := make(map[uint64]*struct {
		pub  ed25519.PublicKey
		priv ed25519.PrivateKey
	}, numNodes)

	for i := 0; i < numNodes; i++ {
		loaded, err := LoadGenesisManifest(manifestPath)
		if err != nil {
			t.Fatalf("node %d: load manifest: %v", i, err)
		}
		kp, err := loaded.LocalKeypair(uint64(i))
		if err != nil {
			t.Fatalf("node %d: local keypair: %v", i, err)
		}
		keypairs[uint64(i)] = &struct {
			pub  ed25519.PublicKey
			priv ed25519.PrivateKey
		}{
			pub:  ed25519.PublicKey(kp.PublicKey),
			priv: ed25519.PrivateKey(kp.PrivateKey),
		}
	}

	// Build validators from each node's perspective — they must all agree.
	for loader := 0; loader < numNodes; loader++ {
		loaded, err := LoadGenesisManifest(manifestPath)
		if err != nil {
			t.Fatalf("loader %d: load manifest: %v", loader, err)
		}
		validators, err := loaded.BuildValidators()
		if err != nil {
			t.Fatalf("loader %d: build validators: %v", loader, err)
		}
		if len(validators) != numNodes {
			t.Fatalf("loader %d: expected %d validators, got %d", loader, numNodes, len(validators))
		}
		for id, v := range validators {
			expected := keypairs[id]
			if !ed25519.PublicKey(v.PublicKey).Equal(expected.pub) {
				t.Fatalf("loader %d: validator %d public key mismatch", loader, id)
			}
		}
	}

	// Cross-signing test: node i signs, node j verifies using the shared validator public key.
	message := []byte("consensus-block-hash-placeholder")
	for signer := uint64(0); signer < uint64(numNodes); signer++ {
		sig := ed25519.Sign(keypairs[signer].priv, message)

		for verifier := uint64(0); verifier < uint64(numNodes); verifier++ {
			// Verifier uses signer's public key from the shared validator set.
			ok := ed25519.Verify(keypairs[signer].pub, message, sig)
			if !ok {
				t.Fatalf("node %d's signature failed verification by node %d", signer, verifier)
			}
		}
	}
}

// TestMultiNodeManifest_PeerMapConsistency verifies that all nodes build
// the same peer map with proper multiaddrs from the shared manifest.
func TestMultiNodeManifest_PeerMapConsistency(t *testing.T) {
	numNodes := 7
	manifest := generateDeterministicManifestWithMultiaddr(t, numNodes, "peermap-test", "127.0.0.1", 8080)
	manifestPath := writeManifestFile(t, manifest)

	loaded, err := LoadGenesisManifest(manifestPath)
	if err != nil {
		t.Fatalf("load manifest: %v", err)
	}

	peers, err := loaded.BuildPeerMap()
	if err != nil {
		t.Fatalf("build peer map: %v", err)
	}
	if len(peers) != numNodes {
		t.Fatalf("expected %d peers, got %d", numNodes, len(peers))
	}

	// Each peer ID should be derived from the same Ed25519 key.
	for _, node := range loaded.Nodes {
		info, ok := peers[node.ID]
		if !ok {
			t.Fatalf("node %d missing from peer map", node.ID)
		}
		if info.ID.String() != node.PeerID {
			t.Fatalf("node %d: peer ID mismatch: %s vs %s", node.ID, info.ID, node.PeerID)
		}
		if len(info.Addrs) == 0 {
			t.Fatalf("node %d: no addresses in peer info", node.ID)
		}
	}
}

// TestMultiNodeManifest_QuorumSize verifies n=4 → q=3, n=7 → q=5.
func TestMultiNodeManifest_QuorumSize(t *testing.T) {
	tests := []struct {
		nodes      int
		wantQuorum uint64
	}{
		{4, 3},  // f=1, q = 2f+1 = 3
		{7, 5},  // f=2, q = 2f+1 = 5
		{10, 7}, // f=3, q = 2f+1 = 7
	}

	for _, tt := range tests {
		t.Run(fmt.Sprintf("n=%d", tt.nodes), func(t *testing.T) {
			manifest := generateDeterministicManifest(t, tt.nodes, fmt.Sprintf("quorum-test-%d", tt.nodes))
			manifestPath := writeManifestFile(t, manifest)

			loaded, err := LoadGenesisManifest(manifestPath)
			if err != nil {
				t.Fatalf("load manifest: %v", err)
			}
			validators, err := loaded.BuildValidators()
			if err != nil {
				t.Fatalf("build validators: %v", err)
			}

			// Compute expected quorum: sum(power)*2/3 + 1
			var totalPower uint64
			for _, v := range validators {
				totalPower += v.Power
			}
			expected := totalPower*2/3 + 1
			if expected != tt.wantQuorum {
				t.Fatalf("expected quorum %d, computed %d", tt.wantQuorum, expected)
			}
		})
	}
}

// TestMultiNodeManifest_DeterministicReproducibility verifies that the same
// seed produces identical manifests across runs.
func TestMultiNodeManifest_DeterministicReproducibility(t *testing.T) {
	m1 := generateDeterministicManifest(t, 4, "repro-test")
	m2 := generateDeterministicManifest(t, 4, "repro-test")

	if len(m1.Nodes) != len(m2.Nodes) {
		t.Fatal("manifest node count differs")
	}
	for i := range m1.Nodes {
		if m1.Nodes[i].PublicKeyHex != m2.Nodes[i].PublicKeyHex {
			t.Fatalf("node %d: public key differs across runs", i)
		}
		if m1.Nodes[i].PrivateKeyHex != m2.Nodes[i].PrivateKeyHex {
			t.Fatalf("node %d: private key differs across runs", i)
		}
		if m1.Nodes[i].PeerID != m2.Nodes[i].PeerID {
			t.Fatalf("node %d: peer ID differs across runs", i)
		}
	}
}

// ── Helpers ────────────────────────────────────────────────────────────────

func generateDeterministicManifest(t *testing.T, numNodes int, seed string) *GenesisManifest {
	t.Helper()
	manifest := &GenesisManifest{Version: 1, Nodes: make([]ManifestNode, 0, numNodes)}
	for i := 0; i < numNodes; i++ {
		seedHash := sha256.Sum256([]byte(fmt.Sprintf("%s:%d", seed, i)))
		priv := ed25519.NewKeyFromSeed(seedHash[:])
		pub := priv.Public().(ed25519.PublicKey)
		vrfSeedHash := sha256.Sum256([]byte(fmt.Sprintf("vrf:%s:%d", seed, i)))
		vrfPriv := manifestVRFSuite.Scalar().Pick(manifestVRFSuite.XOF(vrfSeedHash[:]))
		vrfPub := manifestVRFSuite.Point().Mul(vrfPriv, nil)
		vrfPrivRaw, err := vrfPriv.MarshalBinary()
		if err != nil {
			t.Fatalf("marshal vrf private key: %v", err)
		}
		vrfPubRaw, err := vrfPub.MarshalBinary()
		if err != nil {
			t.Fatalf("marshal vrf public key: %v", err)
		}
		manifest.Nodes = append(manifest.Nodes, ManifestNode{
			ID:               uint64(i),
			Power:            1,
			IsActive:         true,
			PublicKeyHex:     hex.EncodeToString(pub),
			PrivateKeyHex:    hex.EncodeToString(priv),
			VRFPublicKeyHex:  hex.EncodeToString(vrfPubRaw),
			VRFPrivateKeyHex: hex.EncodeToString(vrfPrivRaw),
		})
	}
	if err := manifest.Normalize(); err != nil {
		t.Fatalf("normalize manifest: %v", err)
	}
	return manifest
}

func generateDeterministicManifestWithMultiaddr(t *testing.T, numNodes int, seed, host string, basePort int) *GenesisManifest {
	t.Helper()
	manifest := &GenesisManifest{Version: 1, Nodes: make([]ManifestNode, 0, numNodes)}
	for i := 0; i < numNodes; i++ {
		seedHash := sha256.Sum256([]byte(fmt.Sprintf("%s:%d", seed, i)))
		priv := ed25519.NewKeyFromSeed(seedHash[:])
		pub := priv.Public().(ed25519.PublicKey)
		peerID, err := peerIDFromEd25519PublicKey(pub)
		if err != nil {
			t.Fatalf("peer id for node %d: %v", i, err)
		}
		vrfSeedHash := sha256.Sum256([]byte(fmt.Sprintf("vrf:%s:%d", seed, i)))
		vrfPriv := manifestVRFSuite.Scalar().Pick(manifestVRFSuite.XOF(vrfSeedHash[:]))
		vrfPub := manifestVRFSuite.Point().Mul(vrfPriv, nil)
		vrfPrivRaw, err := vrfPriv.MarshalBinary()
		if err != nil {
			t.Fatalf("marshal vrf private key: %v", err)
		}
		vrfPubRaw, err := vrfPub.MarshalBinary()
		if err != nil {
			t.Fatalf("marshal vrf public key: %v", err)
		}
		manifest.Nodes = append(manifest.Nodes, ManifestNode{
			ID:               uint64(i),
			Power:            1,
			IsActive:         true,
			PublicKeyHex:     hex.EncodeToString(pub),
			PrivateKeyHex:    hex.EncodeToString(priv),
			VRFPublicKeyHex:  hex.EncodeToString(vrfPubRaw),
			VRFPrivateKeyHex: hex.EncodeToString(vrfPrivRaw),
			PeerID:           peerID.String(),
			P2PMultiaddr:     fmt.Sprintf("/ip4/%s/tcp/%d/p2p/%s", host, basePort+i, peerID),
		})
	}
	if err := manifest.Normalize(); err != nil {
		t.Fatalf("normalize manifest: %v", err)
	}
	return manifest
}

func writeManifestFile(t *testing.T, manifest *GenesisManifest) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "genesis.json")
	payload, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		t.Fatalf("marshal manifest: %v", err)
	}
	if err := os.WriteFile(path, payload, 0o600); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	return path
}
