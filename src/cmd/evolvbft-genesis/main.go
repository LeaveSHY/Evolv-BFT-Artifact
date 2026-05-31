package main

import (
	"crypto/ed25519"
	crand "crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"net"
	"os"
	"strings"

	libp2pcrypto "github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/peer"
	"go.dedis.ch/kyber/v3"
	"go.dedis.ch/kyber/v3/suites"

	"evolvbft/evolvbft/bootstrap"
)

var genesisVRFSuite = suites.MustFind("Ed25519")

func main() {
	var (
		nodes    = flag.Int("nodes", 4, "Number of nodes in the genesis manifest")
		out      = flag.String("out", "genesis.json", "Output manifest path")
		seed     = flag.String("seed", "", "Optional deterministic cluster seed")
		power    = flag.Uint64("power", 1, "Voting power per node")
		baseHost = flag.String("base-host", "", "Base IPv4 or hostname for P2P multiaddr generation (e.g. 127.0.0.1 or localhost)")
		basePort = flag.Int("base-port", 8080, "Base TCP port; node i listens on base-port+i")
		verbose  = flag.Bool("verbose", false, "Print generated peer IDs")
	)
	flag.Parse()

	if *nodes <= 0 {
		fmt.Fprintln(os.Stderr, "nodes must be > 0")
		os.Exit(1)
	}

	manifest := &bootstrap.GenesisManifest{
		Version: 1,
		Nodes:   make([]bootstrap.ManifestNode, 0, *nodes),
	}

	for i := 0; i < *nodes; i++ {
		seedBytes, err := deriveSeed(*seed, uint64(i))
		if err != nil {
			fmt.Fprintf(os.Stderr, "derive seed for node %d: %v\n", i, err)
			os.Exit(1)
		}
		priv := ed25519.NewKeyFromSeed(seedBytes)
		pub := priv.Public().(ed25519.PublicKey)

		hostPrivKey, err := libp2pcrypto.UnmarshalEd25519PrivateKey(priv)
		if err != nil {
			fmt.Fprintf(os.Stderr, "decode libp2p key for node %d: %v\n", i, err)
			os.Exit(1)
		}
		peerID, err := peer.IDFromPrivateKey(hostPrivKey)
		if err != nil {
			fmt.Fprintf(os.Stderr, "peer id for node %d: %v\n", i, err)
			os.Exit(1)
		}

		vrfSeedBytes, err := deriveVRFSeed(*seed, uint64(i))
		if err != nil {
			fmt.Fprintf(os.Stderr, "derive vrf seed for node %d: %v\n", i, err)
			os.Exit(1)
		}
		vrfPriv := genesisVRFScalar(vrfSeedBytes)
		vrfPub := genesisVRFSuite.Point().Mul(vrfPriv, nil)
		vrfPrivRaw, err := vrfPriv.MarshalBinary()
		if err != nil {
			fmt.Fprintf(os.Stderr, "marshal vrf private key for node %d: %v\n", i, err)
			os.Exit(1)
		}
		vrfPubRaw, err := vrfPub.MarshalBinary()
		if err != nil {
			fmt.Fprintf(os.Stderr, "marshal vrf public key for node %d: %v\n", i, err)
			os.Exit(1)
		}

		node := bootstrap.ManifestNode{
			ID:               uint64(i),
			Power:            *power,
			IsActive:         true,
			PublicKeyHex:     hex.EncodeToString(pub),
			PrivateKeyHex:    hex.EncodeToString(priv),
			VRFPublicKeyHex:  hex.EncodeToString(vrfPubRaw),
			VRFPrivateKeyHex: hex.EncodeToString(vrfPrivRaw),
			PeerID:           peerID.String(),
		}
		if *baseHost != "" {
			multiaddr, err := buildP2PMultiaddr(*baseHost, *basePort+i, peerID.String())
			if err != nil {
				fmt.Fprintf(os.Stderr, "build p2p multiaddr for node %d: %v\n", i, err)
				os.Exit(1)
			}
			node.P2PMultiaddr = multiaddr
		}
		manifest.Nodes = append(manifest.Nodes, node)
		if *verbose {
			fmt.Printf("node=%d peer_id=%s multiaddr=%s\n", i, peerID, node.P2PMultiaddr)
		}
	}

	if err := manifest.Normalize(); err != nil {
		fmt.Fprintf(os.Stderr, "normalize manifest: %v\n", err)
		os.Exit(1)
	}
	payload, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "marshal manifest: %v\n", err)
		os.Exit(1)
	}
	if err := os.WriteFile(*out, payload, 0o600); err != nil {
		fmt.Fprintf(os.Stderr, "write manifest: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Wrote genesis manifest: %s\n", *out)
}

func deriveSeed(clusterSeed string, nodeID uint64) ([]byte, error) {
	if clusterSeed == "" {
		seed := make([]byte, ed25519.SeedSize)
		_, err := crand.Read(seed)
		return seed, err
	}
	sum := sha256.Sum256([]byte(fmt.Sprintf("%s:%d", clusterSeed, nodeID)))
	seed := make([]byte, ed25519.SeedSize)
	copy(seed, sum[:])
	return seed, nil
}

func deriveVRFSeed(clusterSeed string, nodeID uint64) ([]byte, error) {
	if clusterSeed == "" {
		seed := make([]byte, 32)
		_, err := crand.Read(seed)
		return seed, err
	}
	sum := sha256.Sum256([]byte(fmt.Sprintf("vrf:%s:%d", clusterSeed, nodeID)))
	seed := make([]byte, 32)
	copy(seed, sum[:])
	return seed, nil
}

func genesisVRFScalar(seed []byte) kyber.Scalar {
	return genesisVRFSuite.Scalar().Pick(genesisVRFSuite.XOF(seed))
}

func buildP2PMultiaddr(host string, port int, peerID string) (string, error) {
	trimmedHost := strings.TrimSpace(host)
	if trimmedHost == "" {
		return "", fmt.Errorf("base-host is empty")
	}
	if ip := net.ParseIP(trimmedHost); ip != nil {
		if v4 := ip.To4(); v4 != nil {
			return fmt.Sprintf("/ip4/%s/tcp/%d/p2p/%s", v4.String(), port, peerID), nil
		}
		return "", fmt.Errorf("ipv6 base-host is not supported yet: %s", host)
	}
	return fmt.Sprintf("/dns/%s/tcp/%d/p2p/%s", trimmedHost, port, peerID), nil
}
