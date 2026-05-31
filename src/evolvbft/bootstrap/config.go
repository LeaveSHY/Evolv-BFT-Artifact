package bootstrap

import (
	"errors"
	"flag"
	"fmt"
	"strconv"
	"strings"

	"github.com/libp2p/go-libp2p/core/peer"
	ma "github.com/multiformats/go-multiaddr"
)

var allowedAdaptivePolicies = map[string]struct{}{
	"off":           {},
	"safe-baseline": {},
	"scripted":      {},
	"http":          {},
	"facmac-http":   {},
	"sfac":          {}, // SFAC local policy (§III-D)
}

type EngineConfig struct {
	NodeID             uint64
	Port               int
	HTTPPort           int
	HTTPListenAddr     string
	BasePort           int
	TotalNodes         uint64
	InitialValidators  uint64
	Peers              string
	Manifest           string
	Instances          uint64
	BatchTxs           int
	TimeoutMs          int
	InboundMsgQueue    int
	InboundTxQueue     int
	OrdererPendingCap  int
	ConsensusTopic     string
	AdaptiveEnabled    bool
	AdaptivePolicy     string
	AdaptiveIntervalMs int
	AdaptiveScript     string
	AdaptivePolicyURL  string
	AdaptiveTracePath  string
	AdminPprofEnabled  bool
}

func ParseEngineConfig(args []string) (*EngineConfig, error) {
	cfg := &EngineConfig{}
	fs := flag.NewFlagSet("evolvbft", flag.ContinueOnError)
	fs.Uint64Var(&cfg.NodeID, "id", 0, "Node ID")
	fs.IntVar(&cfg.Port, "port", 8080, "Port to listen on")
	fs.IntVar(&cfg.HTTPPort, "http", 9000, "HTTP port for admin API")
	fs.StringVar(&cfg.HTTPListenAddr, "http-listen-addr", "127.0.0.1", "Listen address for admin API")
	fs.IntVar(&cfg.BasePort, "base-port", 8080, "Base port for default peer map")
	fs.Uint64Var(&cfg.TotalNodes, "total-nodes", 4, "Total nodes in cluster")
	fs.Uint64Var(&cfg.InitialValidators, "initial-validators", 3, "Initial validators in epoch 1")
	fs.StringVar(&cfg.Peers, "peers", "", "Peer map: id=/ip4/host/tcp/port/p2p/PeerID,...")
	fs.StringVar(&cfg.Manifest, "manifest", "", "Path to genesis manifest with stable validator identities")
	fs.Uint64Var(&cfg.Instances, "instances", 10, "Number of consensus instances (M)")
	fs.IntVar(&cfg.BatchTxs, "batch-txs", 4096, "Maximum transactions per proposal batch")
	fs.IntVar(&cfg.TimeoutMs, "timeout-ms", 500, "Base timeout in milliseconds for pacemaker/orderer")
	fs.IntVar(&cfg.InboundMsgQueue, "inbound-msg-queue", 8192, "Inbound consensus/global message queue capacity")
	fs.IntVar(&cfg.InboundTxQueue, "inbound-tx-queue", 65536, "Inbound transaction queue capacity")
	fs.IntVar(&cfg.OrdererPendingCap, "orderer-pending-cap", 65536, "Global orderer pending rank queue capacity")
	fs.StringVar(&cfg.ConsensusTopic, "consensus-topic", "evolvbft-consensus", "Consensus topic name (prefix)")
	fs.BoolVar(&cfg.AdaptiveEnabled, "adaptive-enabled", false, "Enable adaptive MARL control plane")
	fs.StringVar(&cfg.AdaptivePolicy, "adaptive-policy", "off", "Adaptive policy: off|safe-baseline|scripted|sfac|http|facmac-http")
	fs.IntVar(&cfg.AdaptiveIntervalMs, "adaptive-interval-ms", 1000, "Adaptive control loop interval in milliseconds")
	fs.StringVar(&cfg.AdaptiveScript, "adaptive-script", "", "Path to external scripted adaptive action JSON")
	fs.StringVar(&cfg.AdaptivePolicyURL, "adaptive-policy-url", "", "HTTP endpoint for external MARL/FACMAC policy inference")
	fs.StringVar(&cfg.AdaptiveTracePath, "adaptive-trace-path", "", "Path to JSONL adaptive trajectory log for offline MARL training")
	fs.BoolVar(&cfg.AdminPprofEnabled, "admin-pprof-enabled", false, "Expose pprof handlers on the admin API")
	if err := fs.Parse(args); err != nil {
		return nil, err
	}
	if cfg.TotalNodes == 0 {
		return nil, errors.New("total-nodes must be greater than 0")
	}
	if cfg.InitialValidators == 0 {
		return nil, errors.New("initial-validators must be greater than 0")
	}
	if cfg.InitialValidators > cfg.TotalNodes {
		return nil, errors.New("initial-validators cannot exceed total-nodes")
	}
	if cfg.Instances == 0 {
		return nil, errors.New("instances must be greater than 0")
	}
	if cfg.BatchTxs <= 0 {
		return nil, errors.New("batch-txs must be greater than 0")
	}
	if cfg.TimeoutMs <= 0 {
		return nil, errors.New("timeout-ms must be greater than 0")
	}
	if cfg.InboundMsgQueue <= 0 {
		return nil, errors.New("inbound-msg-queue must be greater than 0")
	}
	if cfg.InboundTxQueue <= 0 {
		return nil, errors.New("inbound-tx-queue must be greater than 0")
	}
	if cfg.OrdererPendingCap <= 0 {
		return nil, errors.New("orderer-pending-cap must be greater than 0")
	}
	if strings.TrimSpace(cfg.HTTPListenAddr) == "" {
		return nil, errors.New("http-listen-addr must not be empty")
	}
	if strings.TrimSpace(cfg.ConsensusTopic) == "" {
		return nil, errors.New("consensus-topic must not be empty")
	}
	if cfg.AdaptiveIntervalMs <= 0 {
		return nil, errors.New("adaptive-interval-ms must be greater than 0")
	}
	adaptivePolicy := strings.TrimSpace(cfg.AdaptivePolicy)
	if adaptivePolicy == "" {
		adaptivePolicy = "off"
	}
	cfg.AdaptivePolicy = adaptivePolicy
	if _, ok := allowedAdaptivePolicies[adaptivePolicy]; !ok {
		return nil, fmt.Errorf("unsupported adaptive-policy %q", cfg.AdaptivePolicy)
	}
	if cfg.AdaptiveEnabled {
		switch adaptivePolicy {
		case "scripted":
			if strings.TrimSpace(cfg.AdaptiveScript) == "" {
				return nil, errors.New("adaptive-script must not be empty when adaptive-policy=scripted")
			}
		case "http", "facmac-http":
			if strings.TrimSpace(cfg.AdaptivePolicyURL) == "" {
				return nil, fmt.Errorf("adaptive-policy-url must not be empty when adaptive-policy=%s", adaptivePolicy)
			}
		}
	}
	return cfg, nil
}

// BuildPeerMap builds a NodeID → peer.AddrInfo map from --peers flag or DHT-only mode.
// Format: --peers "0=/ip4/127.0.0.1/tcp/8080/p2p/QmPeerID,1=/ip4/127.0.0.1/tcp/8081/p2p/QmPeerID2"
// When --peers is empty, returns an empty map (DHT/mDNS only discovery, no hardcoded IPs).
func (c *EngineConfig) RequiresManifest() bool {
	if c == nil {
		return false
	}
	return strings.TrimSpace(c.Manifest) == "" && c.TotalNodes > 1
}

func (c *EngineConfig) BuildPeerMap() (map[uint64]peer.AddrInfo, error) {
	if strings.TrimSpace(c.Peers) == "" {
		// Phase 7: No hardcoded peers — rely on DHT/mDNS discovery only
		return make(map[uint64]peer.AddrInfo), nil
	}
	return parsePeersMultiaddr(c.Peers)
}

// BuildPeerMapLegacy returns a legacy host:port peer map for backward compatibility
func (c *EngineConfig) BuildPeerMapLegacy() (map[uint64]string, error) {
	if strings.TrimSpace(c.Peers) == "" {
		peerMap := make(map[uint64]string, c.TotalNodes)
		for i := uint64(0); i < c.TotalNodes; i++ {
			peerMap[i] = fmt.Sprintf("127.0.0.1:%d", c.BasePort+int(i))
		}
		return peerMap, nil
	}
	return parsePeersLegacy(c.Peers)
}

// parsePeersMultiaddr parses multiaddr-format peer entries:
// "0=/ip4/127.0.0.1/tcp/8080/p2p/QmPeerID,1=/ip4/127.0.0.1/tcp/8081/p2p/QmPeerID2"
func parsePeersMultiaddr(peers string) (map[uint64]peer.AddrInfo, error) {
	result := make(map[uint64]peer.AddrInfo)
	items := strings.Split(peers, ",")
	for _, item := range items {
		part := strings.TrimSpace(item)
		if part == "" {
			continue
		}
		pair := strings.SplitN(part, "=", 2)
		if len(pair) != 2 {
			return nil, fmt.Errorf("invalid peer entry: %s", part)
		}
		id, err := strconv.ParseUint(strings.TrimSpace(pair[0]), 10, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid peer id: %s", pair[0])
		}
		addrStr := strings.TrimSpace(pair[1])
		if addrStr == "" {
			return nil, fmt.Errorf("empty address for peer %d", id)
		}

		maddr, err := ma.NewMultiaddr(addrStr)
		if err != nil {
			return nil, fmt.Errorf("invalid multiaddr for peer %d: %w", id, err)
		}

		info, err := peer.AddrInfoFromP2pAddr(maddr)
		if err != nil {
			return nil, fmt.Errorf("invalid p2p address for peer %d: %w", id, err)
		}
		result[id] = *info
	}
	return result, nil
}

// parsePeersLegacy parses old-style peer entries: "0=host:port,1=host:port"
func parsePeersLegacy(peers string) (map[uint64]string, error) {
	result := make(map[uint64]string)
	items := strings.Split(peers, ",")
	for _, item := range items {
		part := strings.TrimSpace(item)
		if part == "" {
			continue
		}
		pair := strings.SplitN(part, "=", 2)
		if len(pair) != 2 {
			return nil, fmt.Errorf("invalid peer entry: %s", part)
		}
		id, err := strconv.ParseUint(strings.TrimSpace(pair[0]), 10, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid peer id: %s", pair[0])
		}
		addr := strings.TrimSpace(pair[1])
		if addr == "" {
			return nil, fmt.Errorf("empty address for peer %d", id)
		}
		result[id] = addr
	}
	if len(result) == 0 {
		return nil, errors.New("peer map is empty")
	}
	return result, nil
}
