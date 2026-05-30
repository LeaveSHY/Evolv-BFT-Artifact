package bootstrap

import (
	"crypto/ed25519"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"

	libp2pcrypto "github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/peer"
	ma "github.com/multiformats/go-multiaddr"
	"go.dedis.ch/kyber/v3"
	"go.dedis.ch/kyber/v3/suites"

	"octopus-bft/octopus/types"
)

// GenesisManifest defines the stable node identities used by an Octopus cluster.
// This is intentionally experiment-oriented and may contain private keys so that
// multi-node runs can be reproduced exactly.
type GenesisManifest struct {
	Version int            `json:"version"`
	Nodes   []ManifestNode `json:"nodes"`
}

type ManifestNode struct {
	ID               uint64 `json:"id"`
	Power            uint64 `json:"power"`
	IsActive         bool   `json:"is_active"`
	PublicKeyHex     string `json:"public_key_hex"`
	PrivateKeyHex    string `json:"private_key_hex"`
	VRFPublicKeyHex  string `json:"vrf_public_key_hex,omitempty"`
	VRFPrivateKeyHex string `json:"vrf_private_key_hex,omitempty"`
	PeerID           string `json:"peer_id"`
	P2PMultiaddr     string `json:"p2p_multiaddr,omitempty"`
}

var manifestVRFSuite = suites.MustFind("Ed25519")

func LoadGenesisManifest(path string) (*GenesisManifest, error) {
	if strings.TrimSpace(path) == "" {
		return nil, fmt.Errorf("manifest path is empty")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read manifest: %w", err)
	}
	var manifest GenesisManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return nil, fmt.Errorf("decode manifest: %w", err)
	}
	if err := manifest.Normalize(); err != nil {
		return nil, err
	}
	return &manifest, nil
}

func (m *GenesisManifest) ValidateRuntimeBootstrap(nodeID uint64) error {
	if m == nil {
		return fmt.Errorf("manifest is nil")
	}
	if _, err := m.LocalKeypair(nodeID); err != nil {
		return err
	}
	if _, _, err := m.LocalVRFKeypair(nodeID); err != nil {
		return err
	}
	if _, err := m.BuildValidators(); err != nil {
		return err
	}
	if _, err := m.BuildVRFPublicKeys(); err != nil {
		return err
	}
	if len(m.Nodes) <= 1 {
		return nil
	}
	for _, node := range m.Nodes {
		if strings.TrimSpace(node.P2PMultiaddr) == "" {
			return fmt.Errorf("node %d has no p2p multiaddr in manifest", node.ID)
		}
	}
	peers, err := m.BuildPeerMap()
	if err != nil {
		return err
	}
	if len(peers) != len(m.Nodes) {
		return fmt.Errorf("manifest peer map incomplete: have %d entries, want %d", len(peers), len(m.Nodes))
	}
	return nil
}

func (m *GenesisManifest) Normalize() error {
	if m == nil {
		return fmt.Errorf("manifest is nil")
	}
	if len(m.Nodes) == 0 {
		return fmt.Errorf("manifest has no nodes")
	}
	seen := make(map[uint64]struct{}, len(m.Nodes))
	for i := range m.Nodes {
		node := &m.Nodes[i]
		if _, exists := seen[node.ID]; exists {
			return fmt.Errorf("duplicate node id %d in manifest", node.ID)
		}
		seen[node.ID] = struct{}{}
		if node.Power == 0 {
			node.Power = 1
		}
		pub, priv, err := normalizeEd25519Keys(node)
		if err != nil {
			return fmt.Errorf("normalize node %d: %w", node.ID, err)
		}
		node.PublicKeyHex = hex.EncodeToString(pub)
		if len(priv) > 0 {
			node.PrivateKeyHex = hex.EncodeToString(priv)
		}
		vrfPub, vrfPriv, err := normalizeVRFKeys(node)
		if err != nil {
			return fmt.Errorf("normalize node %d vrf keys: %w", node.ID, err)
		}
		if vrfPub != nil {
			vrfPubRaw, err := vrfPub.MarshalBinary()
			if err != nil {
				return fmt.Errorf("marshal node %d vrf public key: %w", node.ID, err)
			}
			node.VRFPublicKeyHex = hex.EncodeToString(vrfPubRaw)
		}
		if vrfPriv != nil {
			vrfPrivRaw, err := vrfPriv.MarshalBinary()
			if err != nil {
				return fmt.Errorf("marshal node %d vrf private key: %w", node.ID, err)
			}
			node.VRFPrivateKeyHex = hex.EncodeToString(vrfPrivRaw)
		}
		peerID, err := peerIDFromEd25519PublicKey(pub)
		if err != nil {
			return fmt.Errorf("derive peer id for node %d: %w", node.ID, err)
		}
		if strings.TrimSpace(node.PeerID) != "" && node.PeerID != peerID.String() {
			return fmt.Errorf("manifest peer id mismatch for node %d", node.ID)
		}
		node.PeerID = peerID.String()
		if strings.TrimSpace(node.P2PMultiaddr) != "" {
			maddr, err := ma.NewMultiaddr(node.P2PMultiaddr)
			if err != nil {
				return fmt.Errorf("invalid multiaddr for node %d: %w", node.ID, err)
			}
			info, err := peer.AddrInfoFromP2pAddr(maddr)
			if err != nil {
				return fmt.Errorf("invalid p2p multiaddr for node %d: %w", node.ID, err)
			}
			if info.ID.String() != node.PeerID {
				return fmt.Errorf("multiaddr peer id mismatch for node %d", node.ID)
			}
		}
	}
	sort.Slice(m.Nodes, func(i, j int) bool { return m.Nodes[i].ID < m.Nodes[j].ID })
	return nil
}

func (m *GenesisManifest) NodeByID(nodeID uint64) (*ManifestNode, error) {
	if m == nil {
		return nil, fmt.Errorf("manifest is nil")
	}
	for i := range m.Nodes {
		if m.Nodes[i].ID == nodeID {
			return &m.Nodes[i], nil
		}
	}
	return nil, fmt.Errorf("node %d not found in manifest", nodeID)
}

func (m *GenesisManifest) LocalKeypair(nodeID uint64) (*types.Keypair, error) {
	node, err := m.NodeByID(nodeID)
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(node.PrivateKeyHex) == "" {
		return nil, fmt.Errorf("node %d has no private key in manifest", nodeID)
	}
	priv, err := decodeSizedHex(node.PrivateKeyHex, ed25519.PrivateKeySize, "ed25519 private key")
	if err != nil {
		return nil, err
	}
	pub := ed25519.PrivateKey(priv).Public().(ed25519.PublicKey)
	return &types.Keypair{
		PublicKey:  append(types.PublicKey(nil), pub...),
		PrivateKey: append(types.PrivateKey(nil), priv...),
	}, nil
}

func (m *GenesisManifest) BuildValidators() (map[uint64]*types.Validator, error) {
	if m == nil {
		return nil, fmt.Errorf("manifest is nil")
	}
	validators := make(map[uint64]*types.Validator, len(m.Nodes))
	for _, node := range m.Nodes {
		pub, err := decodeSizedHex(node.PublicKeyHex, ed25519.PublicKeySize, "ed25519 public key")
		if err != nil {
			return nil, fmt.Errorf("node %d public key: %w", node.ID, err)
		}
		validators[node.ID] = &types.Validator{
			ID:        node.ID,
			PublicKey: append(types.PublicKey(nil), pub...),
			Power:     node.Power,
			IsActive:  node.IsActive,
		}
	}
	return validators, nil
}

func (m *GenesisManifest) BuildPeerMap() (map[uint64]peer.AddrInfo, error) {
	if m == nil {
		return nil, fmt.Errorf("manifest is nil")
	}
	peers := make(map[uint64]peer.AddrInfo)
	for _, node := range m.Nodes {
		if strings.TrimSpace(node.P2PMultiaddr) == "" {
			continue
		}
		maddr, err := ma.NewMultiaddr(node.P2PMultiaddr)
		if err != nil {
			return nil, fmt.Errorf("invalid multiaddr for node %d: %w", node.ID, err)
		}
		info, err := peer.AddrInfoFromP2pAddr(maddr)
		if err != nil {
			return nil, fmt.Errorf("invalid p2p multiaddr for node %d: %w", node.ID, err)
		}
		peers[node.ID] = *info
	}
	return peers, nil
}

func (m *GenesisManifest) LocalVRFKeypair(nodeID uint64) (kyber.Scalar, kyber.Point, error) {
	node, err := m.NodeByID(nodeID)
	if err != nil {
		return nil, nil, err
	}
	if strings.TrimSpace(node.VRFPrivateKeyHex) == "" {
		return nil, nil, fmt.Errorf("node %d has no vrf private key in manifest", nodeID)
	}
	priv, err := decodeVRFPrivateKey(node.VRFPrivateKeyHex)
	if err != nil {
		return nil, nil, err
	}
	pub := manifestVRFSuite.Point().Mul(priv, nil)
	return priv, pub, nil
}

func (m *GenesisManifest) BuildVRFPublicKeys() (map[uint64]kyber.Point, error) {
	if m == nil {
		return nil, fmt.Errorf("manifest is nil")
	}
	keys := make(map[uint64]kyber.Point, len(m.Nodes))
	for _, node := range m.Nodes {
		if strings.TrimSpace(node.VRFPublicKeyHex) == "" {
			return nil, fmt.Errorf("node %d has no vrf public key in manifest", node.ID)
		}
		pub, err := decodeVRFPublicKey(node.VRFPublicKeyHex)
		if err != nil {
			return nil, fmt.Errorf("node %d vrf public key: %w", node.ID, err)
		}
		keys[node.ID] = pub
	}
	return keys, nil
}

func normalizeEd25519Keys(node *ManifestNode) (ed25519.PublicKey, ed25519.PrivateKey, error) {
	if node == nil {
		return nil, nil, fmt.Errorf("manifest node is nil")
	}
	switch {
	case strings.TrimSpace(node.PrivateKeyHex) != "":
		priv, err := decodeSizedHex(node.PrivateKeyHex, ed25519.PrivateKeySize, "ed25519 private key")
		if err != nil {
			return nil, nil, err
		}
		pub := ed25519.PrivateKey(priv).Public().(ed25519.PublicKey)
		if strings.TrimSpace(node.PublicKeyHex) != "" {
			expected, err := decodeSizedHex(node.PublicKeyHex, ed25519.PublicKeySize, "ed25519 public key")
			if err != nil {
				return nil, nil, err
			}
			if !ed25519.PublicKey(expected).Equal(pub) {
				return nil, nil, fmt.Errorf("public key does not match private key")
			}
		}
		return pub, priv, nil
	case strings.TrimSpace(node.PublicKeyHex) != "":
		pub, err := decodeSizedHex(node.PublicKeyHex, ed25519.PublicKeySize, "ed25519 public key")
		if err != nil {
			return nil, nil, err
		}
		return ed25519.PublicKey(pub), nil, nil
	default:
		return nil, nil, fmt.Errorf("node %d must provide at least a public key", node.ID)
	}
}

func normalizeVRFKeys(node *ManifestNode) (kyber.Point, kyber.Scalar, error) {
	if node == nil {
		return nil, nil, fmt.Errorf("manifest node is nil")
	}
	hasPrivate := strings.TrimSpace(node.VRFPrivateKeyHex) != ""
	hasPublic := strings.TrimSpace(node.VRFPublicKeyHex) != ""
	if !hasPrivate && !hasPublic {
		return nil, nil, nil
	}
	if hasPrivate {
		priv, err := decodeVRFPrivateKey(node.VRFPrivateKeyHex)
		if err != nil {
			return nil, nil, err
		}
		pub := manifestVRFSuite.Point().Mul(priv, nil)
		if hasPublic {
			expected, err := decodeVRFPublicKey(node.VRFPublicKeyHex)
			if err != nil {
				return nil, nil, err
			}
			derivedRaw, err := pub.MarshalBinary()
			if err != nil {
				return nil, nil, err
			}
			expectedRaw, err := expected.MarshalBinary()
			if err != nil {
				return nil, nil, err
			}
			if hex.EncodeToString(derivedRaw) != hex.EncodeToString(expectedRaw) {
				return nil, nil, fmt.Errorf("vrf public key does not match private key")
			}
		}
		return pub, priv, nil
	}
	pub, err := decodeVRFPublicKey(node.VRFPublicKeyHex)
	if err != nil {
		return nil, nil, err
	}
	return pub, nil, nil
}

func decodeSizedHex(raw string, size int, label string) ([]byte, error) {
	decoded, err := hex.DecodeString(strings.TrimSpace(raw))
	if err != nil {
		return nil, fmt.Errorf("decode %s: %w", label, err)
	}
	if len(decoded) != size {
		return nil, fmt.Errorf("%s has size %d, want %d", label, len(decoded), size)
	}
	return decoded, nil
}

func decodeVRFPrivateKey(raw string) (kyber.Scalar, error) {
	decoded, err := hex.DecodeString(strings.TrimSpace(raw))
	if err != nil {
		return nil, fmt.Errorf("decode vrf private key: %w", err)
	}
	scalar := manifestVRFSuite.Scalar()
	if err := scalar.UnmarshalBinary(decoded); err != nil {
		return nil, fmt.Errorf("decode vrf private key: %w", err)
	}
	return scalar, nil
}

func decodeVRFPublicKey(raw string) (kyber.Point, error) {
	decoded, err := hex.DecodeString(strings.TrimSpace(raw))
	if err != nil {
		return nil, fmt.Errorf("decode vrf public key: %w", err)
	}
	point := manifestVRFSuite.Point()
	if err := point.UnmarshalBinary(decoded); err != nil {
		return nil, fmt.Errorf("decode vrf public key: %w", err)
	}
	return point, nil
}

func peerIDFromEd25519PublicKey(pub ed25519.PublicKey) (peer.ID, error) {
	libp2pPub, err := libp2pcrypto.UnmarshalEd25519PublicKey(pub)
	if err != nil {
		return "", err
	}
	return peer.IDFromPublicKey(libp2pPub)
}
