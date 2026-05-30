package main

import (
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"net/http"
	"net/http/pprof"
	"os"
	"os/signal"
	"runtime"
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"

	libp2pnetwork "github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"go.dedis.ch/kyber/v3"

	"octopus-bft/octopus/adaptive"
	"octopus-bft/octopus/bootstrap"
	"octopus-bft/octopus/consensus/gbc"
	"octopus-bft/octopus/consensus/hotstuff"
	"octopus-bft/octopus/crypto"
	"octopus-bft/octopus/hydra"
	"octopus-bft/octopus/membership"
	"octopus-bft/octopus/network/libp2p"
	"octopus-bft/octopus/storage"
	"octopus-bft/octopus/trust"
	"octopus-bft/octopus/types"
)

type gbcReadView struct {
	store *gbc.Log
}

func newGBCReadView(store *gbc.Log) *gbcReadView {
	return &gbcReadView{store: store}
}

func (v *gbcReadView) LatestCheckpoint() (gbc.Checkpoint, bool, error) {
	if v == nil {
		return gbc.Checkpoint{}, false, nil
	}
	return gbc.GetLatestCheckpoint(v.store)
}

func main() {
	args := os.Args[1:]
	if !hasInstancesFlag(args) {
		// Number of parallel consensus instances (default 10 for 1000-node throughput targets)
		args = append(args, "-instances=10")
	}

	cfg, err := bootstrap.ParseEngineConfig(args)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Starting Octopus Node %d on port %d (HTTP %d), instances=%d, batch=%d, timeout_ms=%d\n", cfg.NodeID, cfg.Port, cfg.HTTPPort, cfg.Instances, cfg.BatchTxs, cfg.TimeoutMs)

	manifest, err := loadManifest(cfg)
	if err != nil {
		log.Fatal(err)
	}
	typesKp, initialValidators, peerMap, err := loadBootstrapState(cfg, manifest)
	if err != nil {
		log.Fatal(err)
	}
	vrfPriv, vrfPub, vrfPubKeys, err := loadManifestVRFState(cfg, manifest)
	if err != nil {
		log.Fatal(err)
	}

	memberMgr := membership.NewMembershipManager(initialValidators)

	ctx := context.Background()
	net, err := libp2p.NewP2PNetwork(ctx, cfg.Port, typesKp.PrivateKey)
	if err != nil {
		log.Fatalf("Failed to initialize libp2p network: %v", err)
	}
	defer net.Close()
	net.SetNodeAddressBook(peerMap)
	net.SetupStreamHandler()
	net.ConnectBootstrapPeers()
	net.StartConnectionMonitor()
	_, _ = net.SubscribeGlobal(cfg.ConsensusTopic, cfg.InboundMsgQueue)
	_, _ = net.SubscribeReconfig(cfg.ConsensusTopic, cfg.InboundMsgQueue)

	// Initialize Hydra dynamic membership manager
	hydraNet := hydra.NewNetworkAdapter(net, cfg.NodeID)
	hydraValidators := make(map[uint64]*types.Validator, len(initialValidators))
	for id, v := range initialValidators {
		hydraValidators[id] = &types.Validator{
			ID:        v.ID,
			PublicKey: v.PublicKey,
			Power:     v.Power,
			IsActive:  v.IsActive,
		}
	}
	hydraMgr, err := hydra.NewHydraManager(cfg.NodeID, hydraValidators, hydraNet)
	if err != nil {
		log.Fatalf("Failed to initialize Hydra manager: %v", err)
	}

	store := storage.NewStorageManager(cfg.NodeID)
	stores := make([]*storage.StorageManager, 0, cfg.Instances)
	stores = append(stores, store)
	for i := uint64(1); i < cfg.Instances; i++ {
		stores = append(stores, storage.NewStorageManager(cfg.NodeID*100000+i))
	}
	defer func() {
		for _, s := range stores {
			_ = s.Close()
		}
	}()

	outputChan := make(chan hotstuff.InstanceOutput, 4096)
	barTimeout := time.Duration(cfg.TimeoutMs) * time.Millisecond
	orderer := hotstuff.NewGlobalOrdererWithLimit(cfg.Instances, barTimeout, cfg.OrdererPendingCap)
	metricsCollector := hotstuff.NewGlobalConfirmedMetrics(barTimeout)
	gbcLog := gbc.NewLog()
	var gbcNode *gbc.Node
	useLocalGBCFallback := false
	orderedChan := orderer.Start(outputChan)
	exec := hotstuff.NewExecutor(memberMgr.GetCurrentConfig().ToValidatorSet())
	engines := make([]*hotstuff.Engine, 0, cfg.Instances)

	for i := uint64(0); i < cfg.Instances; i++ {
		engineOpts := hotstuff.DefaultEngineOptions()
		engineOpts.PacemakerTimeoutMs = int64(cfg.TimeoutMs)
		engineOpts.ConsensusBuffer = cfg.InboundMsgQueue
		engineOpts.Mempool.MaxBatchTxs = cfg.BatchTxs
		engineOpts.Mempool.MaxQueueLen = cfg.InboundTxQueue
		engineOpts.Mempool.NetworkBuffer = cfg.InboundTxQueue
		// Disable VRF committee when fewer nodes than committee size
		if len(initialValidators) < engineOpts.CommitteeSize {
			engineOpts.CommitteeSize = 0
		}
		proposalIntervalMs := cfg.TimeoutMs / 2
		if proposalIntervalMs <= 0 {
			proposalIntervalMs = 1
		}
		engineOpts.Mempool.ProposalInterval = time.Duration(proposalIntervalMs) * time.Millisecond
		engine := hotstuff.NewEngineWithInstanceAndOptions(cfg.NodeID, typesKp, memberMgr.GetCurrentConfig().ToValidatorSet(), net, stores[i], i, cfg.Instances, cfg.ConsensusTopic, outputChan, engineOpts)
		if vrfPriv != nil {
			engine.SetLocalVRFKeypair(vrfPriv, vrfPub)
		}
		for validatorID, pubKey := range vrfPubKeys {
			engine.RegisterVRFPubKey(validatorID, pubKey)
		}
		engine.SetHydraManager(hydraMgr)
		engines = append(engines, engine)
	}
	gbcPrimaries := currentGBCPrimaries(engines, initialValidators)
	gbcNode, err = newRuntimeGBCNode(cfg.NodeID, typesKp.PrivateKey, gbcPrimaries, net, cfg.ConsensusTopic, cfg.InboundMsgQueue)
	if err != nil {
		if isNotGBCMember(err) {
			log.Printf("Certified GBC node inactive: local node %d is not a current instance primary", cfg.NodeID)
		} else {
			log.Printf("Warning: certified GBC node unavailable, using local GBC log: %v", err)
			useLocalGBCFallback = true
		}
	} else {
		gbcNode.Start()
		defer gbcNode.Stop()
		gbcLog = gbcNode.Log()
	}
	// Wire the certified GBC path when available so CommitQC entries collect
	// Propose-Attest-Commit quorum evidence (§III-C, G4).
	if useLocalGBCFallback {
		if len(gbcPrimaries) > 0 {
			gbcLog.SetNumMembers(len(gbcPrimaries))
		} else {
			gbcLog.SetNumMembers(int(cfg.Instances))
		}
	}
	for _, engine := range engines {
		if gbcNode != nil {
			engine.SetGBCNode(gbcNode)
		} else if useLocalGBCFallback {
			engine.SetGBCLog(gbcLog)
		}
	}

	wireHydraAutoLeaveInjection(hydraMgr, memberMgr, engines)
	if err := setupHydraNetworking(net, hydraMgr); err != nil {
		log.Printf("Warning: failed to setup Hydra networking: %v", err)
	}
	hydraMgr.Start()
	defer hydraMgr.Stop()
	exec.SetReconfigAuthorizer(func(data *types.ReconfigData) bool {
		return reconfigAuthorizedByHydraPendingIntent(data, hydraMgr)
	})
	exec.SetEpochChangeCallback(func(newValSet *types.ValidatorSet, transitions []*types.EpochTransition) error {
		return installCommittedValidatorSet(newValSet, transitions, memberMgr, engines, net, cfg.ConsensusTopic, hydraMgr)
	})
	// Instantiate paper-grade Bayesian trust estimator (§III-D, Eq. 5–6).
	// W=50 matches the paper's Appendix hyperparameter table.
	trustEstimator := trust.NewBayesianEstimator(trust.BayesianConfig{
		WindowSize:   50,
		MinSamples:   3,
		MaxLatencyMs: float64(cfg.TimeoutMs),
		Weights: trust.ClassifierWeights{
			W: [5]float64{2.0, 3.0, 1.5, 1.0, 0.5}, // calibrated via MARL training
			B: -1.0,
		},
	})

	adaptiveRuntime := &octopusAdaptiveRuntime{
		nodeID:    cfg.NodeID,
		keypair:   typesKp,
		engines:   engines,
		memberMgr: memberMgr,
		orderer:   orderer,
		metrics:   metricsCollector,
		net:       net,
		hydraMgr:  hydraMgr,
		trustEst:   trustEstimator,
		trustAgg:   trust.NewAggregator(),
		trustDecay: trust.NewTrustDecay(trust.DefaultDecayConfig()),
		rejectStats: func() map[string]uint64 {
			return aggregateRejectedStats(engines, net)
		},
	}
	gbcReadPath := newGBCReadView(gbcLog)
	go func() {
		for out := range orderedChan {
			retryDelay := 100 * time.Millisecond
			for {
				appliedOut, err := applyOrderedOutput(out, exec)
				if err != nil {
					log.Printf("Ordered block execution failed, retrying in %s: %v", retryDelay, err)
					time.Sleep(retryDelay)
					if retryDelay < 5*time.Second {
						retryDelay *= 2
						if retryDelay > 5*time.Second {
							retryDelay = 5 * time.Second
						}
					}
					continue
				}
				metricsCollector.ObserveGlobalConfirmed(appliedOut, time.Now())
				adaptiveRuntime.recordOrderedOutput(appliedOut)
				if err := publishOrderedCheckpoint(gbcLog, gbcNode, useLocalGBCFallback, appliedOut); err != nil {
					log.Printf("Warning: failed to publish ordered checkpoint: %v", err)
				}
				if payload, err := json.Marshal(appliedOut); err == nil {
					_ = net.PublishGlobal(cfg.ConsensusTopic, payload)
				}
				break
			}
		}
	}()
	for _, engine := range engines {
		go engine.Start()
	}
	adaptiveController := buildAdaptiveController(cfg, adaptiveRuntime)
	if adaptiveController != nil {
		adaptiveRuntime.ctrl = adaptiveController // back-reference for epoch gate
		adaptiveController.Start()
		defer adaptiveController.Stop()
	}

	go startMetricsLogLoop(metricsCollector, orderer, engines, net)
	go startAdminServer(cfg.HTTPListenAddr, cfg.HTTPPort, cfg.NodeID, typesKp, engines[0], memberMgr, engines, orderer, metricsCollector, net, hydraMgr, adaptiveController, adaptiveRuntime, gbcReadPath, cfg.AdminPprofEnabled)

	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	<-c

	fmt.Println("\nShutting down...")
}

func loadManifest(cfg *bootstrap.EngineConfig) (*bootstrap.GenesisManifest, error) {
	if cfg == nil {
		return nil, fmt.Errorf("engine config is nil")
	}
	if cfg.Manifest == "" {
		return nil, nil
	}
	manifest, err := bootstrap.LoadGenesisManifest(cfg.Manifest)
	if err != nil {
		return nil, err
	}
	if err := manifest.ValidateRuntimeBootstrap(cfg.NodeID); err != nil {
		return nil, err
	}
	return manifest, nil
}

func loadBootstrapState(cfg *bootstrap.EngineConfig, manifest *bootstrap.GenesisManifest) (*types.Keypair, map[uint64]*types.Validator, map[uint64]peer.AddrInfo, error) {
	if cfg == nil {
		return nil, nil, nil, fmt.Errorf("engine config is nil")
	}
	if manifest != nil {
		keypair, err := manifest.LocalKeypair(cfg.NodeID)
		if err != nil {
			return nil, nil, nil, err
		}
		validators, err := manifest.BuildValidators()
		if err != nil {
			return nil, nil, nil, err
		}
		peerMap, err := manifest.BuildPeerMap()
		if err != nil {
			return nil, nil, nil, err
		}
		return keypair, validators, peerMap, nil
	}
	if cfg.RequiresManifest() {
		return nil, nil, nil, fmt.Errorf("refusing ephemeral bootstrap for total-nodes=%d: supply -manifest with stable validator identities", cfg.TotalNodes)
	}

	kp, err := crypto.GenerateKeyPair()
	if err != nil {
		return nil, nil, nil, err
	}
	typesKp := &types.Keypair{
		PublicKey:  kp.PublicKey,
		PrivateKey: kp.PrivateKey,
	}

	initialValidators := make(map[uint64]*types.Validator)
	for i := uint64(0); i < cfg.InitialValidators; i++ {
		dummyKey, _ := crypto.GenerateKeyPair()
		initialValidators[i] = &types.Validator{
			ID:        i,
			PublicKey: dummyKey.PublicKey,
			Power:     1,
			IsActive:  true,
		}
	}
	if val, ok := initialValidators[cfg.NodeID]; ok {
		val.PublicKey = kp.PublicKey
	}

	peerMap, err := cfg.BuildPeerMap()
	if err != nil {
		return nil, nil, nil, err
	}
	return typesKp, initialValidators, peerMap, nil
}

func loadManifestVRFState(cfg *bootstrap.EngineConfig, manifest *bootstrap.GenesisManifest) (kyber.Scalar, kyber.Point, map[uint64]kyber.Point, error) {
	if cfg == nil {
		return nil, nil, nil, fmt.Errorf("engine config is nil")
	}
	if manifest == nil {
		return nil, nil, nil, nil
	}
	vrfPriv, vrfPub, err := manifest.LocalVRFKeypair(cfg.NodeID)
	if err != nil {
		return nil, nil, nil, err
	}
	vrfPubKeys, err := manifest.BuildVRFPublicKeys()
	if err != nil {
		return nil, nil, nil, err
	}
	return vrfPriv, vrfPub, vrfPubKeys, nil
}

func hasInstancesFlag(args []string) bool {
	for _, arg := range args {
		if arg == "-instances" || arg == "--instances" {
			return true
		}
		if strings.HasPrefix(arg, "-instances=") || strings.HasPrefix(arg, "--instances=") {
			return true
		}
	}
	return false
}

const (
	adminMaxTxBodyBytes              = 1 << 20
	adminMaxAdaptiveContextBodyBytes = 8 << 10
	maxScenarioJitterMs              = 60000
	adminReadHeaderTimeout           = 5 * time.Second
	adminReadTimeout                 = 10 * time.Second
	adminWriteTimeout                = 15 * time.Second
	adminIdleTimeout                 = 60 * time.Second
)

func methodGuard(method string, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != method {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		next(w, r)
	}
}

func validateScenarioContext(ctx adaptive.ScenarioContext) error {
	scores := []struct {
		name  string
		value float64
	}{
		{name: "heterogeneity_score", value: ctx.HeterogeneityScore},
		{name: "churn_rate", value: ctx.ChurnRate},
		{name: "adversary_score", value: ctx.AdversaryScore},
		{name: "ai_load_score", value: ctx.AILoadScore},
	}
	for _, score := range scores {
		if math.IsNaN(score.value) || math.IsInf(score.value, 0) {
			return fmt.Errorf("%s must be finite", score.name)
		}
		if score.value < 0 || score.value > 1 {
			return fmt.Errorf("%s must be within [0,1]", score.name)
		}
	}
	if math.IsNaN(ctx.NetworkJitterMs) || math.IsInf(ctx.NetworkJitterMs, 0) {
		return errors.New("network_jitter_ms must be finite")
	}
	if ctx.NetworkJitterMs < 0 {
		return errors.New("network_jitter_ms must be >= 0")
	}
	if ctx.NetworkJitterMs > maxScenarioJitterMs {
		return fmt.Errorf("network_jitter_ms must be <= %d", maxScenarioJitterMs)
	}
	return nil
}

func rejectDuplicateScenarioContextKeys(raw []byte) error {
	var payload map[string]json.RawMessage
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	if err := dec.Decode(&payload); err != nil {
		return nil
	}
	seen := make(map[string]struct{}, len(payload))
	keyDec := json.NewDecoder(strings.NewReader(string(raw)))
	for {
		tok, err := keyDec.Token()
		if err != nil {
			break
		}
		delim, ok := tok.(json.Delim)
		if !ok || delim != '{' {
			break
		}
		for keyDec.More() {
			keyToken, err := keyDec.Token()
			if err != nil {
				return nil
			}
			key, ok := keyToken.(string)
			if !ok {
				return nil
			}
			if _, exists := seen[key]; exists {
				return fmt.Errorf("duplicate field %q", key)
			}
			seen[key] = struct{}{}
			var discard json.RawMessage
			if err := keyDec.Decode(&discard); err != nil {
				return nil
			}
		}
		break
	}
	return nil
}

func decodeScenarioContext(body io.Reader) (adaptive.ScenarioContext, error) {
	const maxDuplicateCheckBytes = adminMaxAdaptiveContextBodyBytes + 1
	rawBody, err := io.ReadAll(io.LimitReader(body, maxDuplicateCheckBytes))
	if err != nil {
		return adaptive.ScenarioContext{}, err
	}
	if err := rejectDuplicateScenarioContextKeys(rawBody); err != nil {
		return adaptive.ScenarioContext{}, err
	}
	type scenarioContextInput struct {
		HeterogeneityScore *float64 `json:"heterogeneity_score"`
		ChurnRate          *float64 `json:"churn_rate"`
		AdversaryScore     *float64 `json:"adversary_score"`
		NetworkJitterMs    *float64 `json:"network_jitter_ms"`
		AILoadScore        *float64 `json:"ai_load_score"`
	}
	var input scenarioContextInput
	dec := json.NewDecoder(strings.NewReader(string(rawBody)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&input); err != nil {
		return adaptive.ScenarioContext{}, err
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		if err == nil {
			return adaptive.ScenarioContext{}, errors.New("request body must contain exactly one JSON object")
		}
		return adaptive.ScenarioContext{}, errors.New("request body must contain exactly one JSON object")
	}
	required := []struct {
		name  string
		value *float64
	}{
		{name: "heterogeneity_score", value: input.HeterogeneityScore},
		{name: "churn_rate", value: input.ChurnRate},
		{name: "adversary_score", value: input.AdversaryScore},
		{name: "network_jitter_ms", value: input.NetworkJitterMs},
		{name: "ai_load_score", value: input.AILoadScore},
	}
	for _, field := range required {
		if field.value == nil {
			return adaptive.ScenarioContext{}, fmt.Errorf("%s is required", field.name)
		}
	}
	ctx := adaptive.ScenarioContext{
		HeterogeneityScore: *input.HeterogeneityScore,
		ChurnRate:          *input.ChurnRate,
		AdversaryScore:     *input.AdversaryScore,
		NetworkJitterMs:    *input.NetworkJitterMs,
		AILoadScore:        *input.AILoadScore,
	}
	if err := validateScenarioContext(ctx); err != nil {
		return adaptive.ScenarioContext{}, err
	}
	return ctx, nil
}

func buildAdminMux(nodeID uint64, kp *types.Keypair, engine *hotstuff.Engine, memberMgr *membership.MembershipManager, engines []*hotstuff.Engine, orderer *hotstuff.GlobalOrderer, metricsCollector *hotstuff.GlobalConfirmedMetrics, net *libp2p.P2PNetwork, hydraMgr *hydra.HydraManager, adaptiveController *adaptive.Controller, adaptiveRuntime *octopusAdaptiveRuntime, gbcReadPath *gbcReadView, adminPprofEnabled bool) *http.ServeMux {
	mux := http.NewServeMux()

	mux.HandleFunc("/join", methodGuard(http.MethodPost, func(w http.ResponseWriter, r *http.Request) {
		if err := submitLocalJoinIntent(nodeID, kp, engine, memberMgr, hydraMgr); err != nil {
			http.Error(w, fmt.Sprintf("Failed to process join request: %v", err), http.StatusInternalServerError)
			return
		}
		fmt.Fprintf(w, "Join intent processed for Node %d", nodeID)
	}))

	mux.HandleFunc("/leave", methodGuard(http.MethodPost, func(w http.ResponseWriter, r *http.Request) {
		if err := submitLocalLeaveIntent(nodeID, kp, engine, memberMgr, hydraMgr); err != nil {
			http.Error(w, fmt.Sprintf("Failed to process leave request: %v", err), http.StatusInternalServerError)
			return
		}
		fmt.Fprintf(w, "Leave intent processed for Node %d", nodeID)
	}))

	mux.HandleFunc("/tx", methodGuard(http.MethodPost, func(w http.ResponseWriter, r *http.Request) { // Added for benchmarking
		r.Body = http.MaxBytesReader(w, r.Body, adminMaxTxBodyBytes)
		body, err := io.ReadAll(r.Body)
		if err != nil {
			status := http.StatusBadRequest
			var maxBytesErr *http.MaxBytesError
			if errors.As(err, &maxBytesErr) {
				status = http.StatusRequestEntityTooLarge
			}
			http.Error(w, err.Error(), status)
			return
		}
		tx := &types.Transaction{
			Type:    types.TxTypeNormal,
			Payload: body,
		}
		// Pick a random instance to submit to load balance
		idx := rand.Intn(len(engines))
		if err := engines[idx].AddTransaction(tx); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))

	mux.HandleFunc("/config", methodGuard(http.MethodGet, func(w http.ResponseWriter, r *http.Request) {
		type configResp struct {
			Epoch      uint64 `json:"epoch"`
			Validators uint64 `json:"validators"`
			QuorumSize uint64 `json:"quorum_size"`
			Hash       string `json:"hash"`
		}
		cfg := memberMgr.GetCurrentConfig()
		resp := configResp{
			Epoch:      cfg.ID,
			Validators: uint64(len(cfg.Validators)),
			QuorumSize: cfg.QuorumSize,
			Hash:       fmt.Sprintf("%x", cfg.Hash()),
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))

	mux.HandleFunc("/metrics", methodGuard(http.MethodGet, func(w http.ResponseWriter, r *http.Request) {
		rejectStats := aggregateRejectedStats(engines, net)
		snapshot := metricsCollector.Snapshot(orderer, rejectStats)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(snapshot)
	}))

	mux.HandleFunc("/network", methodGuard(http.MethodGet, func(w http.ResponseWriter, r *http.Request) {
		type networkResp struct {
			Connection libp2p.ConnectionStats `json:"connection"`
			Resources  libp2p.ResourceMetrics `json:"resources"`
			Network    libp2p.NetworkStats    `json:"network"`
			PropP50Ms  float64                `json:"propagation_p50_ms"`
			PropP95Ms  float64                `json:"propagation_p95_ms"`
			PropP99Ms  float64                `json:"propagation_p99_ms"`
		}
		p50, p95, p99 := net.GetPropagationStats()
		resp := networkResp{
			Connection: net.GetConnectionStats(),
			Resources:  net.GetResourceMetrics(),
			Network:    net.GetNetworkStats(),
			PropP50Ms:  p50,
			PropP95Ms:  p95,
			PropP99Ms:  p99,
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))

	mux.HandleFunc("/hydra", methodGuard(http.MethodGet, func(w http.ResponseWriter, r *http.Request) {
		type hydraResp struct {
			Enabled              bool   `json:"enabled"`
			CanParticipate       bool   `json:"can_participate"`
			ConfigID             uint64 `json:"config_id"`
			HighestKnownConfigID uint64 `json:"highest_known_config_id"`
			Validators           int    `json:"validators"`
			QuorumSize           int    `json:"quorum_size"`
			LSetSize             int    `json:"lset_size"`
			PendingJoins         int    `json:"pending_joins"`
			PendingLeaves        int    `json:"pending_leaves"`
		}
		resp := hydraResp{Enabled: hydraMgr != nil}
		if hydraMgr != nil {
			resp.CanParticipate = hydraMgr.CanParticipate()
			if config := memberMgr.GetCurrentConfig(); config != nil {
				resp.ConfigID = config.ID
				resp.Validators = len(config.Validators)
				resp.QuorumSize = int(config.QuorumSize)
			}
			if config := hydraMgr.GetHighestKnownConfiguration(); config != nil {
				resp.HighestKnownConfigID = config.ID
			}
			if hydraMgr.LSetManager != nil {
				resp.LSetSize = len(hydraMgr.LSetManager.GetLSet())
			}
			if hydraMgr.TempConfigManager != nil {
				resp.PendingJoins = len(hydraMgr.TempConfigManager.GetPendingJoins())
				resp.PendingLeaves = len(hydraMgr.TempConfigManager.GetPendingLeaves())
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))

	mux.HandleFunc("/adaptive", methodGuard(http.MethodGet, func(w http.ResponseWriter, r *http.Request) {
		type adaptiveResp struct {
			Enabled               bool                           `json:"enabled"`
			SchemaVersion         string                         `json:"schema_version"`
			Schema                map[string]any                 `json:"schema"`
			HasLastDecision       bool                           `json:"has_last_decision"`
			LastDecision          adaptive.Decision              `json:"last_decision"`
			ClaimBoundary         string                         `json:"claim_boundary"`
			OrganizationSemantics adaptive.OrganizationSemantics `json:"organization_semantics"`
			Context               adaptive.ScenarioContext       `json:"context"`
		}
		resp := adaptiveResp{
			Enabled:       adaptiveController != nil,
			SchemaVersion: adaptive.SchemaVersion,
			Schema:        adaptive.SchemaSnapshot(),
			ClaimBoundary: adaptive.AdminClaimBoundary,
			OrganizationSemantics: adaptive.OrganizationSemantics{
				Status:        adaptive.OrganizationSemanticsAbsent,
				ClaimBoundary: adaptive.OrganizationSemanticsBoundary,
			},
		}
		if adaptiveController != nil {
			resp.HasLastDecision = adaptiveController.HasLastDecision()
			resp.LastDecision = adaptiveController.LastDecision()
		}
		if adaptiveRuntime != nil {
			resp.Context = adaptiveRuntime.GetScenarioContext()
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))

	mux.HandleFunc("/gbc/checkpoint/latest", methodGuard(http.MethodGet, func(w http.ResponseWriter, r *http.Request) {
		if gbcReadPath == nil {
			http.Error(w, "gbc read path unavailable", http.StatusServiceUnavailable)
			return
		}
		checkpoint, ok, err := gbcReadPath.LatestCheckpoint()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if !ok {
			http.Error(w, "latest checkpoint not found", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(checkpoint)
	}))

	mux.HandleFunc("/adaptive/context", func(w http.ResponseWriter, r *http.Request) {
		if adaptiveRuntime == nil {
			http.Error(w, "adaptive runtime unavailable", http.StatusServiceUnavailable)
			return
		}
		switch r.Method {
		case http.MethodGet:
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(adaptiveRuntime.GetScenarioContext())
		case http.MethodPost:
			r.Body = http.MaxBytesReader(w, r.Body, adminMaxAdaptiveContextBodyBytes)
			ctx, err := decodeScenarioContext(r.Body)
			if err != nil {
				status := http.StatusBadRequest
				if strings.Contains(err.Error(), "http: request body too large") {
					status = http.StatusRequestEntityTooLarge
				}
				http.Error(w, err.Error(), status)
				return
			}
			adaptiveRuntime.SetScenarioContext(ctx)
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(adaptiveRuntime.GetScenarioContext())
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	})

	if adminPprofEnabled {
		mux.HandleFunc("/debug/pprof/", pprof.Index)
		mux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
		mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
		mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
		mux.HandleFunc("/debug/pprof/trace", pprof.Trace)
	}

	// gc endpoint for manual GC trigger and memory stats
	mux.HandleFunc("/debug/gc", methodGuard(http.MethodPost, func(w http.ResponseWriter, r *http.Request) {
		runtime.GC()
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		resp := map[string]interface{}{
			"alloc_mb":       float64(m.Alloc) / 1024 / 1024,
			"total_alloc_mb": float64(m.TotalAlloc) / 1024 / 1024,
			"sys_mb":         float64(m.Sys) / 1024 / 1024,
			"num_gc":         m.NumGC,
			"goroutines":     runtime.NumGoroutine(),
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))

	return mux
}

func newAdminHTTPServer(addr string, handler http.Handler) *http.Server {
	return &http.Server{
		Addr:              addr,
		Handler:           handler,
		ReadHeaderTimeout: adminReadHeaderTimeout,
		ReadTimeout:       adminReadTimeout,
		WriteTimeout:      adminWriteTimeout,
		IdleTimeout:       adminIdleTimeout,
	}
}

func startAdminServer(listenAddr string, port int, nodeID uint64, kp *types.Keypair, engine *hotstuff.Engine, memberMgr *membership.MembershipManager, engines []*hotstuff.Engine, orderer *hotstuff.GlobalOrderer, metricsCollector *hotstuff.GlobalConfirmedMetrics, net *libp2p.P2PNetwork, hydraMgr *hydra.HydraManager, adaptiveController *adaptive.Controller, adaptiveRuntime *octopusAdaptiveRuntime, gbcReadPath *gbcReadView, adminPprofEnabled bool) {
	mux := buildAdminMux(nodeID, kp, engine, memberMgr, engines, orderer, metricsCollector, net, hydraMgr, adaptiveController, adaptiveRuntime, gbcReadPath, adminPprofEnabled)
	addr := fmt.Sprintf("%s:%d", listenAddr, port)
	log.Printf("Admin server listening on %s", addr)
	srv := newAdminHTTPServer(addr, mux)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Printf("admin server stopped: %v", err)
	}
}

func reconfigAuthorizedByHydraPendingIntent(data *types.ReconfigData, hydraMgr *hydra.HydraManager) bool {
	if data == nil || hydraMgr == nil || hydraMgr.TempConfigManager == nil {
		return false
	}
	switch data.Type {
	case types.ReconfigJoin:
		for _, req := range hydraMgr.TempConfigManager.GetPendingJoins() {
			if req == nil || req.ID != data.NodeID {
				continue
			}
			if !bytes.Equal(req.PublicKey, data.PublicKey) {
				return false
			}
			power := req.Power
			if power == 0 {
				power = 1
			}
			dataPower := data.Power
			if dataPower == 0 {
				dataPower = 1
			}
			return power == dataPower
		}
		return false
	case types.ReconfigLeave:
		for _, req := range hydraMgr.TempConfigManager.GetPendingLeaves() {
			if req != nil && req.ID == data.NodeID {
				return true
			}
		}
		return false
	default:
		return false
	}
}

func aggregateRejectedStats(engines []*hotstuff.Engine, net *libp2p.P2PNetwork) map[string]uint64 {
	agg := make(map[string]uint64)
	for _, engine := range engines {
		if engine == nil {
			continue
		}
		stats := engine.GetRejectedStats()
		for reason, count := range stats {
			agg[reason] += count
		}
	}
	if net != nil {
		for reason, count := range net.GetPubSubValidatorStats() {
			agg[reason] += count
		}
	}
	return agg
}

func startMetricsLogLoop(metricsCollector *hotstuff.GlobalConfirmedMetrics, orderer *hotstuff.GlobalOrderer, engines []*hotstuff.Engine, net *libp2p.P2PNetwork) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	for range ticker.C {
		rejectStats := aggregateRejectedStats(engines, net)
		snapshot := metricsCollector.Snapshot(orderer, rejectStats)
		netStats := net.GetNetworkStats()
		connStats := net.GetConnectionStats()
		p50, p95, p99 := net.GetPropagationStats()
		log.Printf("global-confirmed total=%d nil=%d tps=%.2f latency_ms(p50/p95/p99)=%.2f/%.2f/%.2f recovery_ms(p50/p95/p99)=%.2f/%.2f/%.2f backlog(pending/missing)=%d/%d reject_total=%d net(peers=%d/%d sent=%d recv=%d reconn=%d/%d) prop_ms(p50/p95/p99)=%.2f/%.2f/%.2f",
			snapshot.GlobalConfirmedTotal,
			snapshot.GlobalConfirmedNil,
			snapshot.ThroughputTPS,
			snapshot.LatencyP50Ms,
			snapshot.LatencyP95Ms,
			snapshot.LatencyP99Ms,
			snapshot.RecoveryP50Ms,
			snapshot.RecoveryP95Ms,
			snapshot.RecoveryP99Ms,
			snapshot.BacklogPending,
			snapshot.BacklogMissing,
			snapshot.RejectTotal,
			connStats.ConnectedPeers,
			connStats.KnownPeers,
			netStats.TotalBytesSent,
			netStats.TotalBytesRecv,
			netStats.ReconnectAttempts,
			netStats.ReconnectSuccess,
			p50, p95, p99,
		)
	}
}

func applyOrderedOutput(out hotstuff.InstanceOutput, exec *hotstuff.Executor) (hotstuff.InstanceOutput, error) {
	if exec == nil {
		return out, nil
	}
	if out.IsNil || out.Block == nil {
		return out, nil
	}
	transitions, err := exec.ApplyOrderedBlock(out.Block, out.Rank)
	if err != nil {
		return out, err
	}
	out.EpochTransitions = append(out.EpochTransitions, transitions...)
	return out, nil
}

func publishOrderedCheckpoint(logStore *gbc.Log, node *gbc.Node, allowLocalFallback bool, out hotstuff.InstanceOutput) error {
	if logStore == nil || out.IsNil || out.Block == nil {
		return nil
	}
	if latest, ok := logStore.LatestByType(gbc.EntryCheckpoint); ok {
		var payload struct {
			BlockHashHex string `json:"block_hash_hex"`
		}
		if err := json.Unmarshal(latest.Payload, &payload); err == nil && payload.BlockHashHex == hex.EncodeToString(out.BlockHash) {
			return nil
		}
	}
	payload, err := json.Marshal(map[string]any{
		"instance_id":      out.InstanceID,
		"local_height":     out.LocalHeight,
		"rank":             out.Rank,
		"epoch":            out.Block.Epoch,
		"is_nil":           out.IsNil,
		"transition_count": len(out.EpochTransitions),
		"block_hash_hex":   hex.EncodeToString(out.BlockHash),
	})
	if err != nil {
		return err
	}
	if node != nil {
		_, err := node.Propose(gbc.EntryCheckpoint, payload)
		if gbc.IsNotProposer(err) {
			return nil
		}
		return err
	}
	if !allowLocalFallback {
		return nil
	}
	return logStore.Publish(gbc.Entry{
		Height:  logStore.Height(),
		Type:    gbc.EntryCheckpoint,
		Payload: payload,
	})
}

func wireHydraAutoLeaveInjection(hydraMgr *hydra.HydraManager, memberMgr *membership.MembershipManager, engines []*hotstuff.Engine) {
	if hydraMgr == nil || hydraMgr.AutoTransitionManager == nil {
		return
	}
	hydraMgr.AutoTransitionManager.OnTransition(func(config *hydra.Configuration, proof *hydra.TransitionProof) {
		injectHydraAutoLeaves(config, proof, memberMgr, engines)
	})
}

func injectHydraAutoLeaves(config *hydra.Configuration, proof *hydra.TransitionProof, memberMgr *membership.MembershipManager, engines []*hotstuff.Engine) {
	if config == nil || memberMgr == nil || proof == nil {
		return
	}
	currentConfig := memberMgr.GetCurrentConfig()
	if currentConfig == nil {
		return
	}
	engine := selectHydraInjectionEngine(engines)
	if engine == nil {
		return
	}
	if len(proof.Leaves) == 0 {
		return
	}
	logger := log.Printf
	logger("Hydra ATM transition candidate observed (ID=%d, validators=%d); injecting ReconfigAutoLeave transactions via engine instance %d",
		config.ID, len(config.Validators), engine.GetInstanceID())

	proofLeaves := append([]uint64(nil), proof.Leaves...)
	authorizedLeaves := make([]uint64, 0, len(proofLeaves))
	for _, id := range proofLeaves {
		if _, exists := currentConfig.Validators[id]; exists {
			authorizedLeaves = append(authorizedLeaves, id)
		}
	}
	if len(authorizedLeaves) == 0 {
		return
	}
	proofVotes := make(map[uint64]*types.HydraAutoVote, len(proof.AutoVotes))
	for voterID, vote := range proof.AutoVotes {
		if vote == nil {
			continue
		}
		proofVotes[voterID] = &types.HydraAutoVote{
			SenderID:  vote.SenderID,
			Signature: append([]byte(nil), vote.Signature...),
			Digest:    append([]byte(nil), vote.Digest...),
		}
	}
	reconfigData := types.ReconfigData{
		Type:        types.ReconfigAutoLeave,
		NodeID:      authorizedLeaves[0],
		Epoch:       currentConfig.ID,
		TargetEpoch: currentConfig.ID + 1,
		AutoLeaveProof: &types.HydraTransitionProof{
			View:        proof.View,
			NewConfigID: proof.NewConfigID,
			Leaves:      append([]uint64(nil), authorizedLeaves...),
			BlockHash:   append([]byte(nil), proof.BlockHash...),
			AutoVotes:   proofVotes,
		},
	}
	payload, err := json.Marshal(reconfigData)
	if err != nil {
		log.Printf("Failed to marshal auto-leave reconfig for nodes %v: %v", authorizedLeaves, err)
		return
	}
	tx := &types.Transaction{Type: types.TxTypeReconfig, Payload: payload}
	if err := engine.AddTransaction(tx); err != nil {
		log.Printf("Failed to inject auto-leave reconfig for nodes %v: %v", authorizedLeaves, err)
		return
	}
	log.Printf("Injected ReconfigAutoLeave for nodes %v (epoch %d -> %d, hydra_config_id=%d) via engine instance %d", authorizedLeaves, currentConfig.ID, currentConfig.ID+1, proof.NewConfigID, engine.GetInstanceID())
}

func selectHydraInjectionEngine(engines []*hotstuff.Engine) *hotstuff.Engine {
	for _, engine := range engines {
		if engine != nil {
			return engine
		}
	}
	return nil
}

var installCommittedValidatorSetMu sync.Mutex

func installCommittedValidatorSet(newValSet *types.ValidatorSet, transitions []*types.EpochTransition, memberMgr *membership.MembershipManager, engines []*hotstuff.Engine, net *libp2p.P2PNetwork, topicBase string, hydraMgr *hydra.HydraManager) error {
	if newValSet == nil {
		return fmt.Errorf("validator set is nil")
	}
	installCommittedValidatorSetMu.Lock()
	defer installCommittedValidatorSetMu.Unlock()
	decodedVRFPubKeys := make(map[uint64]kyber.Point, len(newValSet.Validators))
	for validatorID, validator := range newValSet.Validators {
		if validator == nil || len(validator.VRFPublicKey) == 0 {
			continue
		}
		pubKey, err := hotstuff.DecodeVRFPublicKey(validator.VRFPublicKey)
		if err != nil {
			return fmt.Errorf("decode vrf pubkey for validator %d: %w", validatorID, err)
		}
		decodedVRFPubKeys[validatorID] = pubKey
	}

	targetConfig := &types.Configuration{
		ID:         newValSet.Epoch,
		Validators: make(map[uint64]*types.Validator, len(newValSet.Validators)),
		QuorumSize: newValSet.QuorumSize,
	}
	for id, validator := range newValSet.Validators {
		if validator == nil {
			continue
		}
		copyVal := *validator
		targetConfig.Validators[id] = &copyVal
	}
	if targetConfig.QuorumSize == 0 {
		targetConfig.QuorumSize = uint64((2*len(targetConfig.Validators))/3 + 1)
	}
	currentConfig := memberMgr.GetCurrentConfig()
	if currentConfig != nil {
		if bytes.Equal(currentConfig.Hash(), targetConfig.Hash()) {
			if hydraMgr != nil {
				if err := hydraMgr.InstallCommittedConfiguration(targetConfig); err != nil {
					return fmt.Errorf("reconcile hydra committed configuration: %w", err)
				}
			}
			for _, engine := range engines {
				if engine == nil {
					continue
				}
				installedSet := newValSet.Copy()
				engine.UpdateValidatorSetWithVRFKeys(installedSet, decodedVRFPubKeys)
			}
			return nil
		}
		if targetConfig.ID < currentConfig.ID {
			return nil
		}
		if targetConfig.ID == currentConfig.ID {
			return fmt.Errorf("conflicting committed validator set for epoch %d", targetConfig.ID)
		}
	}
	if len(transitions) > 0 {
		transition := transitions[len(transitions)-1]
		if transition != nil {
			oldEpoch := uint64(0)
			if currentConfig != nil {
				oldEpoch = currentConfig.ID
			}
			added, removed := membershipDiffValidatorIDs(currentConfig, targetConfig)
			if transition.OldEpoch != oldEpoch || transition.NewEpoch != targetConfig.ID || transition.QuorumSize != targetConfig.QuorumSize || !bytes.Equal(transition.ConfigHash, targetConfig.Hash()) || !uint64SlicesEqual(transition.Added, added) || !uint64SlicesEqual(transition.Removed, removed) {
				return fmt.Errorf("transition metadata does not match committed validator set for epoch %d", targetConfig.ID)
			}
		}
	}
	if hydraMgr != nil {
		if err := hydraMgr.InstallCommittedConfiguration(targetConfig); err != nil {
			return fmt.Errorf("install hydra committed configuration: %w", err)
		}
	}
	config, event, changed, err := memberMgr.InstallValidatorSetFromTransitions(newValSet, transitions)
	if err != nil || !changed || config == nil {
		return err
	}

	if err != nil || !changed || config == nil {
		return err
	}
	for _, engine := range engines {
		if engine == nil {
			continue
		}
		installedSet := newValSet.Copy()
		engine.UpdateValidatorSetWithVRFKeys(installedSet, decodedVRFPubKeys)
	}
	if event != nil {
		if payload, err := json.Marshal(event); err == nil && net != nil {
			_ = net.PublishReconfig(topicBase, payload)
		}
		log.Printf("Config changed epoch %d -> %d added=%v removed=%v quorum=%d", event.OldEpoch, event.NewEpoch, event.Added, event.Removed, event.QuorumSize)
	}
	return nil
}

func membershipDiffValidatorIDs(oldConfig *types.Configuration, newConfig *types.Configuration) ([]uint64, []uint64) {
	added := make([]uint64, 0)
	removed := make([]uint64, 0)
	if newConfig != nil {
		for id := range newConfig.Validators {
			if oldConfig == nil {
				added = append(added, id)
				continue
			}
			if _, exists := oldConfig.Validators[id]; !exists {
				added = append(added, id)
			}
		}
	}
	if oldConfig != nil {
		for id := range oldConfig.Validators {
			if newConfig == nil {
				removed = append(removed, id)
				continue
			}
			if _, exists := newConfig.Validators[id]; !exists {
				removed = append(removed, id)
			}
		}
	}
	sort.Slice(added, func(i, j int) bool { return added[i] < added[j] })
	sort.Slice(removed, func(i, j int) bool { return removed[i] < removed[j] })
	return added, removed
}

func uint64SlicesEqual(a []uint64, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func setupHydraNetworking(net *libp2p.P2PNetwork, hydraMgr *hydra.HydraManager) error {
	if net == nil || hydraMgr == nil {
		return nil
	}

	hydraTopicChan, err := net.SubscribeTopic(hydra.HydraTopicName, 256)
	if err != nil {
		return err
	}
	go func() {
		for data := range hydraTopicChan {
			if err := handleHydraMessageBytes(data, hydraMgr); err != nil {
				log.Printf("Hydra topic message handling error: %v", err)
			}
		}
	}()

	net.SetStreamHandler(hydra.HydraProtocolID, func(stream libp2pnetwork.Stream) {
		defer stream.Close()
		data, err := io.ReadAll(io.LimitReader(stream, 1<<20))
		if err != nil || len(data) == 0 {
			return
		}
		if err := handleHydraMessageBytes(data, hydraMgr); err != nil {
			log.Printf("Hydra stream message handling error: %v", err)
		}
	})

	return nil
}

func handleHydraMessageBytes(data []byte, hydraMgr *hydra.HydraManager) error {
	if hydraMgr == nil || len(data) == 0 {
		return nil
	}

	var atMsg hydra.AutoTransitionMessage
	if err := json.Unmarshal(data, &atMsg); err == nil && atMsg.View > 0 {
		return hydraMgr.HandleMessage(&atMsg)
	}

	var disReq hydra.DiscoveryRequest
	if err := json.Unmarshal(data, &disReq); err == nil && !disReq.Timestamp.IsZero() && disReq.Type == hydra.DiscoveryRequest_ {
		return hydraMgr.HandleMessage(&disReq)
	}

	var disResp hydra.DiscoveryResponseMessage
	if err := json.Unmarshal(data, &disResp); err == nil && !disResp.Timestamp.IsZero() && disResp.Type == hydra.DiscoveryResponse {
		return hydraMgr.HandleMessage(&disResp)
	}

	return nil
}
