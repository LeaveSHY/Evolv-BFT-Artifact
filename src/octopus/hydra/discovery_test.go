package hydra

import (
	"encoding/hex"
	"sync"
	"testing"
	"time"

	"octopus-bft/octopus/crypto"
	"octopus-bft/octopus/types"
)

var testValidatorPrivateKeys = map[string]types.PrivateKey{}

func storeTestValidatorKey(publicKey types.PublicKey, privateKey types.PrivateKey) {
	testValidatorPrivateKeys[hex.EncodeToString(publicKey)] = append(types.PrivateKey(nil), privateKey...)
}

func lookupTestValidatorKey(publicKey types.PublicKey) (types.PrivateKey, bool) {
	privateKey, ok := testValidatorPrivateKeys[hex.EncodeToString(publicKey)]
	if !ok {
		return nil, false
	}
	return append(types.PrivateKey(nil), privateKey...), true
}

type recordingNetwork struct {
	broadcasts []interface{}
	sends      []sentMessage
}

type sentMessage struct {
	to  uint64
	msg interface{}
}

func (n *recordingNetwork) Broadcast(msg interface{}) {
	n.broadcasts = append(n.broadcasts, msg)
}

func (n *recordingNetwork) Send(to uint64, msg interface{}) {
	n.sends = append(n.sends, sentMessage{to: to, msg: msg})
}

func TestRequestDiscoveryBroadcastsDiscoveryRequest(t *testing.T) {
	net := &recordingNetwork{}
	manager := NewConfigurationDiscoveryManager(1, testHydraConfig(0, 0, 1, 2, 3), net)

	if err := manager.RequestDiscovery(2); err != nil {
		t.Fatalf("request discovery: %v", err)
	}
	if len(net.broadcasts) != 1 {
		t.Fatalf("expected 1 broadcast, got %d", len(net.broadcasts))
	}
	req, ok := net.broadcasts[0].(*DiscoveryRequest)
	if !ok {
		t.Fatalf("expected DiscoveryRequest broadcast, got %T", net.broadcasts[0])
	}
	if len(req.ConfigIDs) != 2 || req.ConfigIDs[0] != 1 || req.ConfigIDs[1] != 2 {
		t.Fatalf("unexpected requested config IDs: %#v", req.ConfigIDs)
	}
}

func TestHandleDiscoveryRequestRespondsWithConfigurations(t *testing.T) {
	net := &recordingNetwork{}
	manager := NewConfigurationDiscoveryManager(1, testHydraConfig(0, 0, 1, 2, 3), net)
	if err := manager.AddConfiguration(testHydraConfig(1, 0, 1, 2, 3, 4)); err != nil {
		t.Fatalf("add config: %v", err)
	}

	req := &DiscoveryRequest{
		Type:      DiscoveryRequest_,
		SenderID:  9,
		ConfigIDs: []uint64{1},
		Timestamp: time.Now(),
	}
	if err := manager.HandleDiscoveryRequest(req); err != nil {
		t.Fatalf("handle request: %v", err)
	}
	if len(net.sends) != 1 {
		t.Fatalf("expected 1 direct response, got %d", len(net.sends))
	}
	resp, ok := net.sends[0].msg.(*DiscoveryResponseMessage)
	if !ok {
		t.Fatalf("expected DiscoveryResponseMessage, got %T", net.sends[0].msg)
	}
	if net.sends[0].to != req.SenderID {
		t.Fatalf("unexpected response target: got=%d want=%d", net.sends[0].to, req.SenderID)
	}
	if len(resp.Configs) != 1 || resp.Configs[0].ID != 1 {
		t.Fatalf("unexpected response configs: %#v", resp.Configs)
	}
}

func TestInstallCommittedConfigurationIsIdempotentAndRejectsConflictingSameID(t *testing.T) {
	initial := testHydraConfig(0, 0, 1, 2, 3)
	hm, err := NewHydraManager(0, initial.Validators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	target := testHydraConfig(1, 0, 1, 2, 3, 4)
	target.Validators[0].VRFPublicKey = []byte{1, 2, 3}
	committed := &types.Configuration{ID: target.ID, Validators: target.Validators, QuorumSize: uint64(target.QuorumSize)}

	beforeHistory := len(hm.DiscoveryManager.GetHistory())
	if err := hm.InstallCommittedConfiguration(committed); err != nil {
		t.Fatalf("install committed config: %v", err)
	}
	if got := hm.GetCurrentConfiguration(); got == nil || got.ID != 1 {
		t.Fatalf("expected committed hydra config id 1, got %+v", got)
	}
	if got := hm.GetHighestKnownConfiguration(); got == nil || got.ID != 1 {
		t.Fatalf("expected highest-known hydra config id 1, got %+v", got)
	}
	if got := len(hm.DiscoveryManager.GetHistory()); got != beforeHistory+1 {
		t.Fatalf("expected discovery history to advance by one, got %d entries", got)
	}

	if err := hm.InstallCommittedConfiguration(committed); err != nil {
		t.Fatalf("repeat committed install should be idempotent: %v", err)
	}
	if got := len(hm.DiscoveryManager.GetHistory()); got != beforeHistory+1 {
		t.Fatalf("expected repeat committed install to avoid duplicate discovery history, got %d entries", got)
	}

	updated := testHydraConfig(1, 0, 1, 2, 3, 4)
	updated.Validators[0].VRFPublicKey = []byte{9, 9, 9}
	updatedCommitted := &types.Configuration{ID: updated.ID, Validators: updated.Validators, QuorumSize: uint64(updated.QuorumSize)}
	if err := hm.InstallCommittedConfiguration(updatedCommitted); err == nil {
		t.Fatalf("expected conflicting committed config with same id to be rejected")
	}
	current := hm.GetCurrentConfiguration()
	if current == nil || string(current.Validators[0].VRFPublicKey) != string([]byte{1, 2, 3}) {
		t.Fatalf("expected original committed hydra config to remain authoritative, got %+v", current)
	}
	if got := len(hm.DiscoveryManager.GetHistory()); got != beforeHistory+1 {
		t.Fatalf("expected conflicting same-id install to avoid duplicate discovery history, got %d entries", got)
	}
}

func TestInstallCommittedConfigurationClearsCommittedPendingLeave(t *testing.T) {
	initial := testHydraConfig(0, 0, 1, 2, 3)
	hm, err := NewHydraManager(0, initial.Validators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	if err := hm.SubmitLeaveRequest(0); err != nil {
		t.Fatalf("submit leave request: %v", err)
	}
	if got := hm.TempConfigManager.GetPendingLeaves(); len(got) != 1 || got[0] == nil || got[0].ID != 0 {
		t.Fatalf("expected pending leave for validator 0, got %#v", got)
	}

	committed := testHydraConfig(1, 1, 2, 3)
	committedCfg := &types.Configuration{ID: committed.ID, Validators: committed.Validators, QuorumSize: uint64(committed.QuorumSize)}
	if err := hm.InstallCommittedConfiguration(committedCfg); err != nil {
		t.Fatalf("install committed config: %v", err)
	}
	if got := hm.TempConfigManager.GetPendingLeaves(); len(got) != 0 {
		t.Fatalf("expected committed leave to clear pending leave, got %#v", got)
	}
}

func TestInstallCommittedConfigurationClearsCommittedPendingJoin(t *testing.T) {
	initial := testHydraConfig(0, 0, 1, 2, 3)
	hm, err := NewHydraManager(0, initial.Validators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	joinerKey, err := crypto.GenerateKeyPair()
	if err != nil {
		t.Fatalf("generate joiner key: %v", err)
	}
	if err := hm.SubmitJoinRequest(4, joinerKey.PublicKey, 1); err != nil {
		t.Fatalf("submit join request: %v", err)
	}
	if got := hm.TempConfigManager.GetPendingJoins(); len(got) != 1 || got[0] == nil || got[0].ID != 4 {
		t.Fatalf("expected pending join for validator 4, got %#v", got)
	}

	committed := testHydraConfig(1, 0, 1, 2, 3, 4)
	committed.Validators[4].PublicKey = append([]byte(nil), joinerKey.PublicKey...)
	committedCfg := &types.Configuration{ID: committed.ID, Validators: committed.Validators, QuorumSize: uint64(committed.QuorumSize)}
	if err := hm.InstallCommittedConfiguration(committedCfg); err != nil {
		t.Fatalf("install committed config: %v", err)
	}
	if got := hm.TempConfigManager.GetPendingJoins(); len(got) != 0 {
		t.Fatalf("expected committed join to clear pending join, got %#v", got)
	}
}

func TestInstallCommittedConfigurationIgnoresStaleLowerConfigWithoutClearingPendingIntents(t *testing.T) {
	initial := testHydraConfig(0, 0, 1, 2, 3)
	hm, err := NewHydraManager(0, initial.Validators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}

	newer := testHydraConfig(2, 0, 1, 2, 3, 4)
	newerCommitted := &types.Configuration{ID: newer.ID, Validators: newer.Validators, QuorumSize: uint64(newer.QuorumSize)}
	if err := hm.InstallCommittedConfiguration(newerCommitted); err != nil {
		t.Fatalf("install newer committed config: %v", err)
	}
	beforeCurrent := hm.GetCurrentConfiguration()
	beforeHighest := hm.GetHighestKnownConfiguration()
	beforeDiscoveryHistory := len(hm.DiscoveryManager.GetHistory())
	beforeTempHistory := len(hm.TempConfigManager.GetHistory())
	beforeLSet := hm.LSetManager.GetLSet()
	joinerKey, err := crypto.GenerateKeyPair()
	if err != nil {
		t.Fatalf("generate joiner key: %v", err)
	}
	if err := hm.SubmitJoinRequest(5, joinerKey.PublicKey, 1); err != nil {
		t.Fatalf("submit join request: %v", err)
	}
	if err := hm.SubmitLeaveRequest(0); err != nil {
		t.Fatalf("submit leave request: %v", err)
	}

	stale := testHydraConfig(1, 0, 1, 3, 9)
	staleCommitted := &types.Configuration{ID: stale.ID, Validators: stale.Validators, QuorumSize: uint64(stale.QuorumSize)}
	if err := hm.InstallCommittedConfiguration(staleCommitted); err != nil {
		t.Fatalf("install stale committed config: %v", err)
	}
	if got := hm.TempConfigManager.GetPendingJoins(); len(got) != 1 || got[0] == nil || got[0].ID != 5 {
		t.Fatalf("expected stale committed install to preserve pending join, got %#v", got)
	}
	if got := hm.TempConfigManager.GetPendingLeaves(); len(got) != 1 || got[0] == nil || got[0].ID != 0 {
		t.Fatalf("expected stale committed install to preserve pending leave, got %#v", got)
	}
	current := hm.GetCurrentConfiguration()
	if current == nil || beforeCurrent == nil || current.ID != beforeCurrent.ID {
		t.Fatalf("expected stale committed install to preserve current config id %d, got %+v", beforeCurrent.ID, current)
	}
	if _, exists := current.Validators[4]; !exists {
		t.Fatalf("expected stale committed install to preserve current validator 4")
	}
	highest := hm.GetHighestKnownConfiguration()
	if highest == nil || beforeHighest == nil || highest.ID != beforeHighest.ID {
		t.Fatalf("expected stale committed install to preserve highest-known config id %d, got %+v", beforeHighest.ID, highest)
	}
	if got := len(hm.DiscoveryManager.GetHistory()); got != beforeDiscoveryHistory {
		t.Fatalf("expected stale committed install to preserve discovery history length %d, got %d", beforeDiscoveryHistory, got)
	}
	if got := len(hm.TempConfigManager.GetHistory()); got != beforeTempHistory {
		t.Fatalf("expected stale committed install to preserve temp history length %d, got %d", beforeTempHistory, got)
	}
	afterLSet := hm.LSetManager.GetLSet()
	if len(afterLSet) != len(beforeLSet) {
		t.Fatalf("expected stale committed install to preserve L-set size %d, got %d", len(beforeLSet), len(afterLSet))
	}
	for id := range beforeLSet {
		if _, exists := afterLSet[id]; !exists {
			t.Fatalf("expected stale committed install to preserve L-set member %d", id)
		}
	}
	if _, exists := afterLSet[9]; exists {
		t.Fatalf("did not expect stale committed install to add validator 9 to L-set")
	}
}

func TestInstallCommittedConfigurationRejectsConflictingSameIDWithoutClearingPendingIntents(t *testing.T) {
	initial := testHydraConfig(0, 0, 1, 2, 3)
	hm, err := NewHydraManager(0, initial.Validators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}

	committed := testHydraConfig(1, 0, 1, 2, 3, 4)
	committedCfg := &types.Configuration{ID: committed.ID, Validators: committed.Validators, QuorumSize: uint64(committed.QuorumSize)}
	if err := hm.InstallCommittedConfiguration(committedCfg); err != nil {
		t.Fatalf("install committed config: %v", err)
	}
	beforeCurrent := hm.GetCurrentConfiguration()
	beforeHighest := hm.GetHighestKnownConfiguration()
	beforeDiscoveryHistory := len(hm.DiscoveryManager.GetHistory())
	beforeTempHistory := len(hm.TempConfigManager.GetHistory())
	beforeLSet := hm.LSetManager.GetLSet()
	joinerKey, err := crypto.GenerateKeyPair()
	if err != nil {
		t.Fatalf("generate joiner key: %v", err)
	}
	if err := hm.SubmitJoinRequest(5, joinerKey.PublicKey, 1); err != nil {
		t.Fatalf("submit join request: %v", err)
	}
	if err := hm.SubmitLeaveRequest(0); err != nil {
		t.Fatalf("submit leave request: %v", err)
	}

	conflicting := testHydraConfig(1, 0, 1, 2, 3, 9)
	conflictingCfg := &types.Configuration{ID: conflicting.ID, Validators: conflicting.Validators, QuorumSize: uint64(conflicting.QuorumSize)}
	if err := hm.InstallCommittedConfiguration(conflictingCfg); err == nil {
		t.Fatal("expected conflicting committed config to be rejected")
	}
	if got := hm.TempConfigManager.GetPendingJoins(); len(got) != 1 || got[0] == nil || got[0].ID != 5 {
		t.Fatalf("expected conflicting committed install to preserve pending join, got %#v", got)
	}
	if got := hm.TempConfigManager.GetPendingLeaves(); len(got) != 1 || got[0] == nil || got[0].ID != 0 {
		t.Fatalf("expected conflicting committed install to preserve pending leave, got %#v", got)
	}
	current := hm.GetCurrentConfiguration()
	if current == nil || beforeCurrent == nil || current.ID != beforeCurrent.ID {
		t.Fatalf("expected conflicting committed install to preserve current config id %d, got %+v", beforeCurrent.ID, current)
	}
	if _, exists := current.Validators[4]; !exists {
		t.Fatalf("expected conflicting committed install to preserve current validator 4")
	}
	highest := hm.GetHighestKnownConfiguration()
	if highest == nil || beforeHighest == nil || highest.ID != beforeHighest.ID {
		t.Fatalf("expected conflicting committed install to preserve highest-known config id %d, got %+v", beforeHighest.ID, highest)
	}
	if got := len(hm.DiscoveryManager.GetHistory()); got != beforeDiscoveryHistory {
		t.Fatalf("expected conflicting committed install to preserve discovery history length %d, got %d", beforeDiscoveryHistory, got)
	}
	if got := len(hm.TempConfigManager.GetHistory()); got != beforeTempHistory {
		t.Fatalf("expected conflicting committed install to preserve temp history length %d, got %d", beforeTempHistory, got)
	}
	afterLSet := hm.LSetManager.GetLSet()
	if len(afterLSet) != len(beforeLSet) {
		t.Fatalf("expected conflicting committed install to preserve L-set size %d, got %d", len(beforeLSet), len(afterLSet))
	}
	for id := range beforeLSet {
		if _, exists := afterLSet[id]; !exists {
			t.Fatalf("expected conflicting committed install to preserve L-set member %d", id)
		}
	}
	if _, exists := afterLSet[9]; exists {
		t.Fatalf("did not expect conflicting committed install to add validator 9 to L-set")
	}
}

func TestTrackResponsibilityUsesTrackedConfigValidatorsInsteadOfCurrentConfig(t *testing.T) {
	initial := testHydraConfig(0, 0, 1, 2, 3)
	hm, err := NewHydraManager(0, initial.Validators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}

	committed0 := &types.Configuration{ID: initial.ID, Validators: initial.Validators, QuorumSize: uint64(initial.QuorumSize)}
	if err := hm.InstallCommittedConfiguration(committed0); err != nil {
		t.Fatalf("install initial committed config: %v", err)
	}

	next := testHydraConfig(1, 0, 1, 2, 4)
	committed1 := &types.Configuration{ID: next.ID, Validators: next.Validators, QuorumSize: uint64(next.QuorumSize)}
	if err := hm.InstallCommittedConfiguration(committed1); err != nil {
		t.Fatalf("install next committed config: %v", err)
	}

	requestHash := "req-0"
	hm.TrackResponsibility(requestHash, 0)

	configID, validators, exists := hm.GetClarity(requestHash)
	if !exists {
		t.Fatalf("expected tracked clarity for request %q", requestHash)
	}
	if configID != 0 {
		t.Fatalf("expected request to remain bound to config 0, got %d", configID)
	}
	if !validators[3] {
		t.Fatalf("expected validator 3 from tracked config to remain responsible")
	}
	if validators[4] {
		t.Fatalf("did not expect validator 4 from newer config to become responsible")
	}
}

func TestTrackResponsibilityFallsBackToDiscoveryHistory(t *testing.T) {
	initial := testHydraConfig(0, 0, 1, 2, 3)
	hm, err := NewHydraManager(0, initial.Validators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	discovered := testHydraConfig(1, 0, 1, 2, 4)
	if err := hm.DiscoveryManager.AddConfiguration(discovered); err != nil {
		t.Fatalf("add discovered config: %v", err)
	}

	requestHash := "req-discovered"
	hm.TrackResponsibility(requestHash, 1)
	configID, validators, exists := hm.GetClarity(requestHash)
	if !exists {
		t.Fatalf("expected tracked clarity for request %q", requestHash)
	}
	if configID != 1 {
		t.Fatalf("expected request to bind to discovered config 1, got %d", configID)
	}
	if !validators[4] {
		t.Fatalf("expected validator 4 from discovered config to be responsible")
	}
	if validators[3] {
		t.Fatalf("did not expect validator 3 from older config to remain responsible")
	}
}

func TestInstallCommittedConfigurationDetachesValidatorKeyMaterial(t *testing.T) {
	initial := testHydraConfig(0, 0, 1, 2, 3)
	hm, err := NewHydraManager(0, initial.Validators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	target := testHydraConfig(1, 0, 1, 2, 3, 4)
	target.Validators[0].VRFPublicKey = []byte{7, 8, 9}
	committed := &types.Configuration{ID: target.ID, Validators: target.Validators, QuorumSize: uint64(target.QuorumSize)}
	beforePubKey := append([]byte(nil), committed.Validators[0].PublicKey...)
	beforeVRFKey := append([]byte(nil), committed.Validators[0].VRFPublicKey...)
	if err := hm.InstallCommittedConfiguration(committed); err != nil {
		t.Fatalf("install committed config: %v", err)
	}
	committed.Validators[0].PublicKey[0] ^= 0xff
	committed.Validators[0].VRFPublicKey[0] ^= 0xff

	current := hm.GetCurrentConfiguration()
	if current == nil {
		t.Fatalf("expected committed hydra config")
	}
	if string(current.Validators[0].PublicKey) != string(beforePubKey) {
		t.Fatalf("expected hydra committed config to detach public key material from caller mutation")
	}
	if string(current.Validators[0].VRFPublicKey) != string(beforeVRFKey) {
		t.Fatalf("expected hydra committed config to detach vrf key material from caller mutation")
	}
	history := hm.DiscoveryManager.GetHistory()
	if len(history) == 0 {
		t.Fatalf("expected hydra discovery history")
	}
	latest := history[len(history)-1]
	if string(latest.Validators[0].PublicKey) != string(beforePubKey) {
		t.Fatalf("expected hydra discovery history to detach public key material from caller mutation")
	}
	if string(latest.Validators[0].VRFPublicKey) != string(beforeVRFKey) {
		t.Fatalf("expected hydra discovery history to detach vrf key material from caller mutation")
	}
}

func TestHandleDiscoveryResponseDetachesConfigObjects(t *testing.T) {
	manager := NewConfigurationDiscoveryManager(1, testHydraConfig(0, 0, 1, 2, 3), nil)
	incoming := testHydraConfig(1, 0, 1, 2, 4)
	incoming.Validators[0].VRFPublicKey = []byte{5, 6, 7}
	beforePubKey := append([]byte(nil), incoming.Validators[0].PublicKey...)
	beforeVRFKey := append([]byte(nil), incoming.Validators[0].VRFPublicKey...)
	if err := manager.HandleDiscoveryResponse([]*Configuration{incoming}); err != nil {
		t.Fatalf("handle discovery response: %v", err)
	}
	incoming.Validators[0].PublicKey[0] ^= 0xff
	incoming.Validators[0].VRFPublicKey[0] ^= 0xff

	history := manager.GetHistory()
	latest := history[len(history)-1]
	if string(latest.Validators[0].PublicKey) != string(beforePubKey) {
		t.Fatalf("expected discovery response history to detach public key material")
	}
	if string(latest.Validators[0].VRFPublicKey) != string(beforeVRFKey) {
		t.Fatalf("expected discovery response history to detach vrf key material")
	}
	participating := manager.GetParticipatingConfig()
	if participating == nil {
		t.Fatalf("expected participating config")
	}
	if string(participating.Validators[0].PublicKey) != string(beforePubKey) {
		t.Fatalf("expected participating config to detach public key material")
	}
	if string(participating.Validators[0].VRFPublicKey) != string(beforeVRFKey) {
		t.Fatalf("expected participating config to detach vrf key material")
	}
}

func TestInstallCommittedConfigurationDeduplicatesConcurrentDuplicateInstalls(t *testing.T) {
	initial := testHydraConfig(0, 0, 1, 2, 3)
	hm, err := NewHydraManager(0, initial.Validators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}
	beforeDiscoveryHistory := len(hm.DiscoveryManager.GetHistory())
	beforeTempHistory := len(hm.TempConfigManager.GetHistory())

	target := testHydraConfig(1, 0, 1, 2, 3, 4)
	target.Validators[0].VRFPublicKey = []byte{1, 2, 3}
	committed := &types.Configuration{ID: target.ID, Validators: target.Validators, QuorumSize: uint64(target.QuorumSize)}

	const workers = 16
	var wg sync.WaitGroup
	errCh := make(chan error, workers)
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			errCh <- hm.InstallCommittedConfiguration(committed)
		}()
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		if err != nil {
			t.Fatalf("concurrent install committed config: %v", err)
		}
	}

	current := hm.GetCurrentConfiguration()
	if current == nil || current.ID != committed.ID {
		t.Fatalf("expected committed current config id %d, got %+v", committed.ID, current)
	}
	highest := hm.GetHighestKnownConfiguration()
	if highest == nil || highest.ID != committed.ID {
		t.Fatalf("expected highest-known config id %d, got %+v", committed.ID, highest)
	}
	if got := len(hm.DiscoveryManager.GetHistory()); got != beforeDiscoveryHistory+1 {
		t.Fatalf("expected one discovery history append, got %d entries", got)
	}
	if got := len(hm.TempConfigManager.GetHistory()); got != beforeTempHistory+1 {
		t.Fatalf("expected one temp config history append, got %d entries", got)
	}
}

func TestInstallCommittedConfigurationIgnoresStaleLowerConfigID(t *testing.T) {
	initial := testHydraConfig(0, 0, 1, 2, 3)
	hm, err := NewHydraManager(0, initial.Validators, nil)
	if err != nil {
		t.Fatalf("new hydra manager: %v", err)
	}

	newer := testHydraConfig(2, 0, 1, 2, 3, 4)
	newer.Validators[0].VRFPublicKey = []byte{2, 2, 2}
	newerCommitted := &types.Configuration{ID: newer.ID, Validators: newer.Validators, QuorumSize: uint64(newer.QuorumSize)}
	if err := hm.InstallCommittedConfiguration(newerCommitted); err != nil {
		t.Fatalf("install newer committed config: %v", err)
	}
	beforeDiscoveryHistory := len(hm.DiscoveryManager.GetHistory())
	beforeTempHistory := len(hm.TempConfigManager.GetHistory())

	stale := testHydraConfig(1, 0, 1, 3, 9)
	stale.Validators[0].VRFPublicKey = []byte{9, 9, 9}
	staleCommitted := &types.Configuration{ID: stale.ID, Validators: stale.Validators, QuorumSize: uint64(stale.QuorumSize)}
	if err := hm.InstallCommittedConfiguration(staleCommitted); err != nil {
		t.Fatalf("install stale committed config: %v", err)
	}

	current := hm.GetCurrentConfiguration()
	if current == nil || current.ID != newerCommitted.ID {
		t.Fatalf("expected current config to remain at id %d, got %+v", newerCommitted.ID, current)
	}
	if _, exists := current.Validators[4]; !exists {
		t.Fatalf("expected newer validator set to remain installed")
	}
	if _, exists := current.Validators[9]; exists {
		t.Fatalf("did not expect stale validator set to replace newer committed config")
	}
	highest := hm.GetHighestKnownConfiguration()
	if highest == nil || highest.ID != newerCommitted.ID {
		t.Fatalf("expected highest-known config to remain at id %d, got %+v", newerCommitted.ID, highest)
	}
	if got := len(hm.DiscoveryManager.GetHistory()); got != beforeDiscoveryHistory {
		t.Fatalf("expected stale install to avoid discovery history append, got %d entries", got)
	}
	if got := len(hm.TempConfigManager.GetHistory()); got != beforeTempHistory {
		t.Fatalf("expected stale install to avoid temp config history append, got %d entries", got)
	}
	if !hm.LSetManager.IsInL(2) {
		t.Fatalf("expected newer L-set membership to remain installed")
	}
	if hm.LSetManager.IsInL(3) {
		t.Fatalf("did not expect stale L-set membership to replace newer L-set")
	}
}

func testHydraConfig(id uint64, validatorIDs ...uint64) *Configuration {
	validators := make(map[uint64]*Validator, len(validatorIDs))
	for _, validatorID := range validatorIDs {
		kp, err := crypto.GenerateKeyPair()
		if err != nil {
			panic(err)
		}
		storeTestValidatorKey(kp.PublicKey, kp.PrivateKey)
		validators[validatorID] = &Validator{
			ID:        validatorID,
			PublicKey: append([]byte(nil), kp.PublicKey...),
			Power:     1,
			IsActive:  true,
		}
	}
	return &Configuration{
		ID:         id,
		Validators: validators,
		QuorumSize: (2 * len(validators)) / 3 + 1,
	}
}

func TestPendingRequestsAreDetachedFromCallers(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	tcm := NewTemporaryConfigurationManager(initial)
	join := &MemberRequest{Type: RequestJoin, ID: 9, PublicKey: []byte{1, 2, 3}, Signature: []byte{4, 5, 6}, Power: 2}
	if err := tcm.AddJoinRequest(join); err != nil {
		t.Fatalf("add join request: %v", err)
	}
	join.PublicKey[0] = 99
	join.Signature[0] = 88

	stored := tcm.GetPendingJoins()
	if len(stored) != 1 {
		t.Fatalf("expected 1 pending join, got %d", len(stored))
	}
	if stored[0].PublicKey[0] != 1 || stored[0].Signature[0] != 4 {
		t.Fatal("expected stored join request to be detached from caller mutation")
	}

	stored[0].PublicKey[0] = 77
	stored[0].Signature[0] = 66
	fresh := tcm.GetPendingJoins()
	if fresh[0].PublicKey[0] != 1 || fresh[0].Signature[0] != 4 {
		t.Fatal("expected returned join request snapshot to be detached from internal state")
	}
}

func TestApplyMemberRequestsDeepCopiesJoinPublicKey(t *testing.T) {
	initial := testHydraConfig(1, 1, 2, 3, 4)
	tcm := NewTemporaryConfigurationManager(initial)
	join := &MemberRequest{Type: RequestJoin, ID: 9, PublicKey: []byte{7, 8, 9}, Power: 1}
	if err := tcm.AddJoinRequest(join); err != nil {
		t.Fatalf("add join request: %v", err)
	}
	config, err := tcm.ApplyMemberRequests()
	if err != nil {
		t.Fatalf("apply member requests: %v", err)
	}
	config.Validators[9].PublicKey[0] = 55
	fresh, err := tcm.ApplyMemberRequests()
	if err != nil {
		t.Fatalf("re-apply member requests: %v", err)
	}
	if fresh.Validators[9].PublicKey[0] != 7 {
		t.Fatal("expected applied join public key to be detached from prior returned config")
	}
}
