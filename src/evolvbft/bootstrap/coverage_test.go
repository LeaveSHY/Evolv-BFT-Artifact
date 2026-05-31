package bootstrap

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// --- ParseEngineConfig validation error paths ---

func TestParseEngineConfig_RejectsZeroTotalNodes(t *testing.T) {
	_, err := ParseEngineConfig([]string{"-total-nodes", "0"})
	if err == nil {
		t.Fatal("expected error for total-nodes=0")
	}
}

func TestParseEngineConfig_RejectsZeroInitialValidators(t *testing.T) {
	_, err := ParseEngineConfig([]string{"-initial-validators", "0"})
	if err == nil {
		t.Fatal("expected error for initial-validators=0")
	}
}

func TestParseEngineConfig_RejectsInitialValidatorsExceedsTotalNodes(t *testing.T) {
	_, err := ParseEngineConfig([]string{"-total-nodes", "2", "-initial-validators", "3"})
	if err == nil {
		t.Fatal("expected error when initial-validators > total-nodes")
	}
}

func TestParseEngineConfig_RejectsZeroInstances(t *testing.T) {
	_, err := ParseEngineConfig([]string{"-instances", "0"})
	if err == nil {
		t.Fatal("expected error for instances=0")
	}
}

func TestParseEngineConfig_RejectsZeroBatchTxs(t *testing.T) {
	_, err := ParseEngineConfig([]string{"-batch-txs", "0"})
	if err == nil {
		t.Fatal("expected error for batch-txs=0")
	}
}

func TestParseEngineConfig_RejectsZeroTimeoutMs(t *testing.T) {
	_, err := ParseEngineConfig([]string{"-timeout-ms", "0"})
	if err == nil {
		t.Fatal("expected error for timeout-ms=0")
	}
}

func TestParseEngineConfig_RejectsZeroInboundMsgQueue(t *testing.T) {
	_, err := ParseEngineConfig([]string{"-inbound-msg-queue", "0"})
	if err == nil {
		t.Fatal("expected error for inbound-msg-queue=0")
	}
}

func TestParseEngineConfig_RejectsZeroInboundTxQueue(t *testing.T) {
	_, err := ParseEngineConfig([]string{"-inbound-tx-queue", "0"})
	if err == nil {
		t.Fatal("expected error for inbound-tx-queue=0")
	}
}

func TestParseEngineConfig_RejectsZeroOrdererPendingCap(t *testing.T) {
	_, err := ParseEngineConfig([]string{"-orderer-pending-cap", "0"})
	if err == nil {
		t.Fatal("expected error for orderer-pending-cap=0")
	}
}

func TestParseEngineConfig_RejectsEmptyHTTPListenAddr(t *testing.T) {
	_, err := ParseEngineConfig([]string{"-http-listen-addr", "  "})
	if err == nil {
		t.Fatal("expected error for empty http-listen-addr")
	}
}

func TestParseEngineConfig_RejectsEmptyConsensusTopic(t *testing.T) {
	_, err := ParseEngineConfig([]string{"-consensus-topic", "  "})
	if err == nil {
		t.Fatal("expected error for empty consensus-topic")
	}
}

func TestParseEngineConfig_RejectsZeroAdaptiveIntervalMs(t *testing.T) {
	_, err := ParseEngineConfig([]string{"-adaptive-interval-ms", "0"})
	if err == nil {
		t.Fatal("expected error for adaptive-interval-ms=0")
	}
}

func TestParseEngineConfig_RejectsBadFlagSyntax(t *testing.T) {
	_, err := ParseEngineConfig([]string{"-unknownflag", "xyz"})
	if err == nil {
		t.Fatal("expected error for unknown flag")
	}
}

// --- BuildPeerMap / BuildPeerMapLegacy / parsePeersMultiaddr / parsePeersLegacy ---

func TestBuildPeerMapEmptyPeers(t *testing.T) {
	cfg, _ := ParseEngineConfig([]string{})
	peers, err := cfg.BuildPeerMap()
	if err != nil {
		t.Fatalf("BuildPeerMap with empty peers: %v", err)
	}
	if len(peers) != 0 {
		t.Fatalf("expected empty peer map, got %d", len(peers))
	}
}

func TestBuildPeerMapWithMultiaddr(t *testing.T) {
	// Generate a real peer ID from a known key
	seed := sha256.Sum256([]byte("test-peer-0"))
	priv := ed25519.NewKeyFromSeed(seed[:])
	pub := priv.Public().(ed25519.PublicKey)
	peerID, err := peerIDFromEd25519PublicKey(pub)
	if err != nil {
		t.Fatalf("generate peer id: %v", err)
	}
	peersStr := "0=/ip4/127.0.0.1/tcp/8080/p2p/" + peerID.String()

	cfg, err := ParseEngineConfig([]string{"-peers", peersStr, "-total-nodes", "1", "-initial-validators", "1"})
	if err != nil {
		t.Fatalf("parse config: %v", err)
	}
	peers, err := cfg.BuildPeerMap()
	if err != nil {
		t.Fatalf("BuildPeerMap: %v", err)
	}
	if len(peers) != 1 {
		t.Fatalf("expected 1 peer, got %d", len(peers))
	}
	info, ok := peers[0]
	if !ok {
		t.Fatal("peer 0 not in map")
	}
	if info.ID.String() != peerID.String() {
		t.Fatalf("unexpected peer ID: %s", info.ID)
	}
}

func TestBuildPeerMapLegacyEmptyPeers(t *testing.T) {
	cfg, _ := ParseEngineConfig([]string{"-total-nodes", "2", "-initial-validators", "2"})
	peers, err := cfg.BuildPeerMapLegacy()
	if err != nil {
		t.Fatalf("BuildPeerMapLegacy: %v", err)
	}
	if len(peers) != 2 {
		t.Fatalf("expected 2 legacy peers, got %d", len(peers))
	}
	if peers[0] != "127.0.0.1:8080" {
		t.Fatalf("peer 0 address: %s", peers[0])
	}
}

func TestBuildPeerMapLegacyWithExplicitPeers(t *testing.T) {
	cfg, err := ParseEngineConfig([]string{
		"-peers", "0=10.0.0.1:9000,1=10.0.0.2:9001",
		"-total-nodes", "2",
		"-initial-validators", "2",
	})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	peers, err := cfg.BuildPeerMapLegacy()
	if err != nil {
		t.Fatalf("BuildPeerMapLegacy: %v", err)
	}
	if peers[0] != "10.0.0.1:9000" || peers[1] != "10.0.0.2:9001" {
		t.Fatalf("unexpected peers: %v", peers)
	}
}

func TestParsePeersMultiaddrErrors(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{"no equals", "badformat"},
		{"non-numeric id", "abc=/ip4/127.0.0.1/tcp/8080/p2p/12D3KooW"},
		{"empty address", "0="},
		{"invalid multiaddr", "0=/bad/multiaddr"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := parsePeersMultiaddr(tc.input)
			if err == nil {
				t.Fatalf("expected error for input %q", tc.input)
			}
		})
	}
}

func TestParsePeersMultiaddrSkipsEmpty(t *testing.T) {
	seed := sha256.Sum256([]byte("parse-test"))
	priv := ed25519.NewKeyFromSeed(seed[:])
	pub := priv.Public().(ed25519.PublicKey)
	peerID, _ := peerIDFromEd25519PublicKey(pub)
	input := "0=/ip4/127.0.0.1/tcp/8080/p2p/" + peerID.String() + ",,"
	result, err := parsePeersMultiaddr(input)
	if err != nil {
		t.Fatalf("parsePeersMultiaddr: %v", err)
	}
	if len(result) != 1 {
		t.Fatalf("expected 1 peer after skipping empties, got %d", len(result))
	}
}

func TestParsePeersLegacyErrors(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{"no equals", "badformat"},
		{"non-numeric id", "abc=host:1234"},
		{"empty address", "0="},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := parsePeersLegacy(tc.input)
			if err == nil {
				t.Fatalf("expected error for input %q", tc.input)
			}
		})
	}
}

func TestParsePeersLegacyOnlyEmpty(t *testing.T) {
	_, err := parsePeersLegacy(",,")
	if err == nil {
		t.Fatal("expected error for all-empty peer entries")
	}
}

// --- RequiresManifest edge: nil config ---

func TestRequiresManifestNilConfig(t *testing.T) {
	var cfg *EngineConfig
	if cfg.RequiresManifest() {
		t.Fatal("nil config should not require manifest")
	}
}

// --- Manifest error paths ---

func TestLoadGenesisManifest_EmptyPath(t *testing.T) {
	_, err := LoadGenesisManifest("")
	if err == nil {
		t.Fatal("expected error for empty path")
	}
}

func TestLoadGenesisManifest_NonExistentFile(t *testing.T) {
	_, err := LoadGenesisManifest("/nonexistent/path/genesis.json")
	if err == nil {
		t.Fatal("expected error for nonexistent file")
	}
}

func TestLoadGenesisManifest_InvalidJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.json")
	os.WriteFile(path, []byte("{not valid json"), 0o600)
	_, err := LoadGenesisManifest(path)
	if err == nil {
		t.Fatal("expected error for invalid json")
	}
}

func TestLoadGenesisManifest_EmptyNodes(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.json")
	data, _ := json.Marshal(GenesisManifest{Version: 1, Nodes: []ManifestNode{}})
	os.WriteFile(path, data, 0o600)
	_, err := LoadGenesisManifest(path)
	if err == nil {
		t.Fatal("expected error for manifest with no nodes")
	}
}

func TestNormalize_DuplicateNodeID(t *testing.T) {
	node := makeTestManifestNode(t, 0, "127.0.0.1", 8080)
	m := &GenesisManifest{
		Version: 1,
		Nodes:   []ManifestNode{node, node},
	}
	if err := m.Normalize(); err == nil {
		t.Fatal("expected error for duplicate node id")
	}
}

func TestNormalize_NilManifest(t *testing.T) {
	var m *GenesisManifest
	if err := m.Normalize(); err == nil {
		t.Fatal("expected error for nil manifest")
	}
}

func TestNodeByID_NilManifest(t *testing.T) {
	var m *GenesisManifest
	_, err := m.NodeByID(0)
	if err == nil {
		t.Fatal("expected error for nil manifest")
	}
}

func TestNodeByID_NotFound(t *testing.T) {
	node := makeTestManifestNode(t, 0, "127.0.0.1", 8080)
	path := writeTestManifest(t, []ManifestNode{node})
	m, _ := LoadGenesisManifest(path)
	_, err := m.NodeByID(99)
	if err == nil {
		t.Fatal("expected error for non-existent node id")
	}
}

func TestLocalKeypair_NoPrivateKey(t *testing.T) {
	node := makeTestManifestNode(t, 0, "127.0.0.1", 8080)
	node.PrivateKeyHex = ""
	// Must supply public key only
	path := writeTestManifest(t, []ManifestNode{node})
	m, err := LoadGenesisManifest(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	_, err = m.LocalKeypair(0)
	if err == nil {
		t.Fatal("expected error for missing private key")
	}
}

func TestLocalVRFKeypair_NoPrivateKey(t *testing.T) {
	node := makeTestManifestNode(t, 0, "127.0.0.1", 8080)
	node.VRFPrivateKeyHex = ""
	path := writeTestManifest(t, []ManifestNode{node})
	m, err := LoadGenesisManifest(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	_, _, err = m.LocalVRFKeypair(0)
	if err == nil {
		t.Fatal("expected error for missing VRF private key")
	}
}

func TestBuildVRFPublicKeys_MissingKey(t *testing.T) {
	node := makeTestManifestNode(t, 0, "127.0.0.1", 8080)
	// Create a manifest, then mutate to remove VRF pub after normalization
	path := writeTestManifest(t, []ManifestNode{node})
	m, _ := LoadGenesisManifest(path)
	m.Nodes[0].VRFPublicKeyHex = ""
	_, err := m.BuildVRFPublicKeys()
	if err == nil {
		t.Fatal("expected error for missing VRF public key")
	}
}

func TestBuildValidators_NilManifest(t *testing.T) {
	var m *GenesisManifest
	_, err := m.BuildValidators()
	if err == nil {
		t.Fatal("expected error for nil manifest")
	}
}

func TestBuildPeerMap_NilManifest(t *testing.T) {
	var m *GenesisManifest
	_, err := m.BuildPeerMap()
	if err == nil {
		t.Fatal("expected error for nil manifest")
	}
}

func TestBuildVRFPublicKeys_NilManifest(t *testing.T) {
	var m *GenesisManifest
	_, err := m.BuildVRFPublicKeys()
	if err == nil {
		t.Fatal("expected error for nil manifest")
	}
}

func TestValidateRuntimeBootstrap_NilManifest(t *testing.T) {
	var m *GenesisManifest
	if err := m.ValidateRuntimeBootstrap(0); err == nil {
		t.Fatal("expected error for nil manifest")
	}
}

func TestValidateRuntimeBootstrap_SingleNodeValid(t *testing.T) {
	node := makeTestManifestNode(t, 0, "127.0.0.1", 8080)
	path := writeTestManifest(t, []ManifestNode{node})
	m, _ := LoadGenesisManifest(path)
	if err := m.ValidateRuntimeBootstrap(0); err != nil {
		t.Fatalf("expected single-node validation to pass: %v", err)
	}
}

// --- decodeSizedHex error path ---

func TestDecodeSizedHex_BadHex(t *testing.T) {
	_, err := decodeSizedHex("not-valid-hex!!", 32, "test key")
	if err == nil {
		t.Fatal("expected error for invalid hex")
	}
}

func TestDecodeSizedHex_WrongSize(t *testing.T) {
	_, err := decodeSizedHex(hex.EncodeToString(make([]byte, 16)), 32, "test key")
	if err == nil {
		t.Fatal("expected error for wrong size")
	}
}

// --- normalizeEd25519Keys edge cases ---

func TestNormalizeEd25519Keys_NilNode(t *testing.T) {
	_, _, err := normalizeEd25519Keys(nil)
	if err == nil {
		t.Fatal("expected error for nil node")
	}
}

func TestNormalizeEd25519Keys_NoKeys(t *testing.T) {
	node := &ManifestNode{ID: 0}
	_, _, err := normalizeEd25519Keys(node)
	if err == nil {
		t.Fatal("expected error for node with no keys")
	}
}

func TestNormalizeEd25519Keys_PublicKeyOnly(t *testing.T) {
	seed := sha256.Sum256([]byte("pubonly"))
	priv := ed25519.NewKeyFromSeed(seed[:])
	pub := priv.Public().(ed25519.PublicKey)
	node := &ManifestNode{
		ID:           0,
		PublicKeyHex: hex.EncodeToString(pub),
	}
	gotPub, gotPriv, err := normalizeEd25519Keys(node)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gotPriv != nil {
		t.Fatal("expected nil private key")
	}
	if hex.EncodeToString(gotPub) != hex.EncodeToString(pub) {
		t.Fatal("public key mismatch")
	}
}

func TestNormalizeEd25519Keys_PublicPrivateMismatch(t *testing.T) {
	seed1 := sha256.Sum256([]byte("key1"))
	seed2 := sha256.Sum256([]byte("key2"))
	priv := ed25519.NewKeyFromSeed(seed1[:])
	wrongPub := ed25519.NewKeyFromSeed(seed2[:]).Public().(ed25519.PublicKey)
	node := &ManifestNode{
		ID:            0,
		PrivateKeyHex: hex.EncodeToString(priv),
		PublicKeyHex:  hex.EncodeToString(wrongPub),
	}
	_, _, err := normalizeEd25519Keys(node)
	if err == nil {
		t.Fatal("expected error for public/private key mismatch")
	}
}

// --- normalizeVRFKeys edge cases ---

func TestNormalizeVRFKeys_NilNode(t *testing.T) {
	_, _, err := normalizeVRFKeys(nil)
	if err == nil {
		t.Fatal("expected error for nil node")
	}
}

func TestNormalizeVRFKeys_NeitherKey(t *testing.T) {
	node := &ManifestNode{ID: 0}
	pub, priv, err := normalizeVRFKeys(node)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if pub != nil || priv != nil {
		t.Fatal("expected nil keys when neither is provided")
	}
}

func TestNormalizeVRFKeys_PublicOnly(t *testing.T) {
	// Generate a valid VRF public key
	seed := sha256.Sum256([]byte("vrfpubonly"))
	scalar := manifestVRFSuite.Scalar().Pick(manifestVRFSuite.XOF(seed[:]))
	point := manifestVRFSuite.Point().Mul(scalar, nil)
	pubRaw, _ := point.MarshalBinary()
	node := &ManifestNode{
		ID:              0,
		VRFPublicKeyHex: hex.EncodeToString(pubRaw),
	}
	pub, priv, err := normalizeVRFKeys(node)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if pub == nil {
		t.Fatal("expected non-nil public key")
	}
	if priv != nil {
		t.Fatal("expected nil private key")
	}
}

func TestNormalizeVRFKeys_PublicPrivateMismatch(t *testing.T) {
	seed1 := sha256.Sum256([]byte("vrfkey1"))
	seed2 := sha256.Sum256([]byte("vrfkey2"))
	scalar1 := manifestVRFSuite.Scalar().Pick(manifestVRFSuite.XOF(seed1[:]))
	scalar2 := manifestVRFSuite.Scalar().Pick(manifestVRFSuite.XOF(seed2[:]))
	wrongPub := manifestVRFSuite.Point().Mul(scalar2, nil)
	privRaw, _ := scalar1.MarshalBinary()
	wrongPubRaw, _ := wrongPub.MarshalBinary()
	node := &ManifestNode{
		ID:               0,
		VRFPrivateKeyHex: hex.EncodeToString(privRaw),
		VRFPublicKeyHex:  hex.EncodeToString(wrongPubRaw),
	}
	_, _, err := normalizeVRFKeys(node)
	if err == nil {
		t.Fatal("expected error for VRF public/private key mismatch")
	}
}

// --- decodeVRFPrivateKey / decodeVRFPublicKey error paths ---

func TestDecodeVRFPrivateKey_BadHex(t *testing.T) {
	_, err := decodeVRFPrivateKey("not-hex!!")
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestDecodeVRFPrivateKey_BadBinary(t *testing.T) {
	_, err := decodeVRFPrivateKey(hex.EncodeToString([]byte("tooshort")))
	if err == nil {
		t.Fatal("expected error for invalid scalar binary")
	}
}

func TestDecodeVRFPublicKey_BadHex(t *testing.T) {
	_, err := decodeVRFPublicKey("not-hex!!")
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestDecodeVRFPublicKey_BadBinary(t *testing.T) {
	_, err := decodeVRFPublicKey(hex.EncodeToString([]byte("tooshort")))
	if err == nil {
		t.Fatal("expected error for invalid point binary")
	}
}
