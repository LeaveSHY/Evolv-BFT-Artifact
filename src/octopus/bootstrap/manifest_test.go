package bootstrap

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"go.dedis.ch/kyber/v3"
)

func TestLoadGenesisManifest_BuildsValidatorsAndPeers(t *testing.T) {
	manifestPath := writeTestManifest(t, []ManifestNode{
		makeTestManifestNode(t, 0, "127.0.0.1", 8080),
		makeTestManifestNode(t, 1, "127.0.0.1", 8081),
	})

	manifest, err := LoadGenesisManifest(manifestPath)
	if err != nil {
		t.Fatalf("load manifest: %v", err)
	}

	validators, err := manifest.BuildValidators()
	if err != nil {
		t.Fatalf("build validators: %v", err)
	}
	if len(validators) != 2 {
		t.Fatalf("unexpected validator count: %d", len(validators))
	}
	if !validators[0].IsActive || !validators[1].IsActive {
		t.Fatalf("validators should be active")
	}

	peers, err := manifest.BuildPeerMap()
	if err != nil {
		t.Fatalf("build peers: %v", err)
	}
	if len(peers) != 2 {
		t.Fatalf("unexpected peer count: %d", len(peers))
	}
	if peers[0].ID.String() == "" || peers[1].ID.String() == "" {
		t.Fatalf("peer IDs should be populated")
	}
}

func TestLoadGenesisManifest_LocalKeypair(t *testing.T) {
	node := makeTestManifestNode(t, 7, "127.0.0.1", 9007)
	manifestPath := writeTestManifest(t, []ManifestNode{node})

	manifest, err := LoadGenesisManifest(manifestPath)
	if err != nil {
		t.Fatalf("load manifest: %v", err)
	}
	keypair, err := manifest.LocalKeypair(7)
	if err != nil {
		t.Fatalf("local keypair: %v", err)
	}
	if hex.EncodeToString(keypair.PublicKey) != node.PublicKeyHex {
		t.Fatalf("public key mismatch")
	}
	if hex.EncodeToString(keypair.PrivateKey) != node.PrivateKeyHex {
		t.Fatalf("private key mismatch")
	}

	vrfPriv, vrfPub, err := manifest.LocalVRFKeypair(7)
	if err != nil {
		t.Fatalf("local vrf keypair: %v", err)
	}
	vrfPubRaw, err := vrfPub.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal vrf public key: %v", err)
	}
	vrfPrivRaw, err := vrfPriv.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal vrf private key: %v", err)
	}
	if hex.EncodeToString(vrfPubRaw) != node.VRFPublicKeyHex {
		t.Fatalf("vrf public key mismatch")
	}
	if hex.EncodeToString(vrfPrivRaw) != node.VRFPrivateKeyHex {
		t.Fatalf("vrf private key mismatch")
	}
}

func TestLoadGenesisManifest_RejectsPeerIDMismatch(t *testing.T) {
	node := makeTestManifestNode(t, 3, "127.0.0.1", 9010)
	node.PeerID = "12D3KooWbadpeerid"
	manifestPath := writeTestManifest(t, []ManifestNode{node})

	if _, err := LoadGenesisManifest(manifestPath); err == nil {
		t.Fatalf("expected manifest peer id mismatch")
	}
}

func TestGenesisManifestValidateRuntimeBootstrap_RejectsMissingPeerMultiaddr(t *testing.T) {
	manifestPath := writeTestManifest(t, []ManifestNode{
		makeTestManifestNode(t, 0, "127.0.0.1", 8080),
		func() ManifestNode {
			node := makeTestManifestNode(t, 1, "127.0.0.1", 8081)
			node.P2PMultiaddr = ""
			return node
		}(),
	})

	manifest, err := LoadGenesisManifest(manifestPath)
	if err != nil {
		t.Fatalf("load manifest: %v", err)
	}
	if err := manifest.ValidateRuntimeBootstrap(0); err == nil {
		t.Fatalf("expected runtime bootstrap validation to fail without peer multiaddr")
	}
}

func TestGenesisManifestValidateRuntimeBootstrap_RejectsMissingVRFMaterial(t *testing.T) {
	node0 := makeTestManifestNode(t, 0, "127.0.0.1", 8080)
	node1 := makeTestManifestNode(t, 1, "127.0.0.1", 8081)
	node1.VRFPublicKeyHex = ""
	node1.VRFPrivateKeyHex = ""
	manifestPath := writeTestManifest(t, []ManifestNode{node0, node1})

	manifest, err := LoadGenesisManifest(manifestPath)
	if err != nil {
		t.Fatalf("load manifest: %v", err)
	}
	if err := manifest.ValidateRuntimeBootstrap(0); err == nil {
		t.Fatalf("expected runtime bootstrap validation to fail without vrf material")
	}
}

func TestLoadGenesisManifest_AcceptsDNSMultiaddr(t *testing.T) {
	node0 := makeTestManifestNode(t, 0, "127.0.0.1", 8080)
	node1 := makeTestManifestNode(t, 1, "127.0.0.1", 8081)
	node0.P2PMultiaddr = "/dns/localhost/tcp/8080/p2p/" + node0.PeerID
	node1.P2PMultiaddr = "/dns/localhost/tcp/8081/p2p/" + node1.PeerID
	manifestPath := writeTestManifest(t, []ManifestNode{node0, node1})

	manifest, err := LoadGenesisManifest(manifestPath)
	if err != nil {
		t.Fatalf("load manifest: %v", err)
	}
	if err := manifest.ValidateRuntimeBootstrap(0); err != nil {
		t.Fatalf("validate runtime bootstrap: %v", err)
	}
}

func TestLoadGenesisManifest_RejectsMultiaddrPeerIDMismatch(t *testing.T) {
	node0 := makeTestManifestNode(t, 0, "127.0.0.1", 8080)
	node1 := makeTestManifestNode(t, 1, "127.0.0.1", 8081)
	node1.P2PMultiaddr = "/ip4/127.0.0.1/tcp/8081/p2p/" + node0.PeerID
	manifestPath := writeTestManifest(t, []ManifestNode{node0, node1})

	if _, err := LoadGenesisManifest(manifestPath); err == nil {
		t.Fatalf("expected manifest multiaddr peer id mismatch")
	}
}

func writeTestManifest(t *testing.T, nodes []ManifestNode) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "genesis.json")
	payload, err := json.MarshalIndent(&GenesisManifest{
		Version: 1,
		Nodes:   nodes,
	}, "", "  ")
	if err != nil {
		t.Fatalf("marshal manifest: %v", err)
	}
	if err := os.WriteFile(path, payload, 0o600); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	return path
}

func makeTestManifestNode(t *testing.T, nodeID uint64, host string, port int) ManifestNode {
	t.Helper()
	seed := sha256.Sum256([]byte{byte(nodeID), byte(nodeID >> 8), byte(nodeID >> 16), byte(nodeID >> 24)})
	priv := ed25519.NewKeyFromSeed(seed[:])
	pub := priv.Public().(ed25519.PublicKey)
	peerID, err := peerIDFromEd25519PublicKey(pub)
	if err != nil {
		t.Fatalf("peer id: %v", err)
	}
	vrfSeed := sha256.Sum256([]byte{'v', 'r', 'f', byte(nodeID), byte(nodeID >> 8), byte(nodeID >> 16), byte(nodeID >> 24)})
	vrfPriv := vrfTestScalar(t, vrfSeed[:])
	vrfPub := vrfTestPoint(vrfPriv)
	vrfPrivRaw, err := vrfPriv.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal vrf private key: %v", err)
	}
	vrfPubRaw, err := vrfPub.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal vrf public key: %v", err)
	}
	return ManifestNode{
		ID:               nodeID,
		Power:            1,
		IsActive:         true,
		PublicKeyHex:     hex.EncodeToString(pub),
		PrivateKeyHex:    hex.EncodeToString(priv),
		VRFPublicKeyHex:  hex.EncodeToString(vrfPubRaw),
		VRFPrivateKeyHex: hex.EncodeToString(vrfPrivRaw),
		PeerID:           peerID.String(),
		P2PMultiaddr:     "/ip4/" + host + "/tcp/" + strconv.Itoa(port) + "/p2p/" + peerID.String(),
	}
}

func vrfTestScalar(t *testing.T, seed []byte) kyber.Scalar {
	t.Helper()
	scalar := manifestVRFSuite.Scalar().Pick(manifestVRFSuite.XOF(seed))
	return scalar
}

func vrfTestPoint(scalar kyber.Scalar) kyber.Point {
	return manifestVRFSuite.Point().Mul(scalar, nil)
}
