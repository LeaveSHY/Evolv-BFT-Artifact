package hydra

import (
	"testing"
	"time"

	"evolvbft/evolvbft/types"
)

// --- LSetManager: IsMarked, ClearMarks ---

func TestLSetManagerIsMarked(t *testing.T) {
	cfg := testHydraConfig(1, 0, 1, 2, 3)
	lm, err := NewLSetManager(cfg.Validators)
	if err != nil {
		t.Fatalf("NewLSetManager: %v", err)
	}
	if lm.IsMarked(0) {
		t.Fatal("should not be marked before any fault")
	}
	lm.MarkFault(0, FaultClassUnavailable)
	if !lm.IsMarked(0) {
		t.Fatal("should be marked after fault")
	}
}

func TestLSetManagerClearMarks(t *testing.T) {
	cfg := testHydraConfig(1, 0, 1, 2, 3)
	lm, _ := NewLSetManager(cfg.Validators)
	lm.MarkFault(0, FaultClassByzantine)
	// Marks are very fresh, ClearMarks with a long duration should NOT clear them
	lm.ClearMarks(1 * time.Hour)
	if !lm.IsMarked(0) {
		t.Fatal("mark should survive when duration not exceeded")
	}
	// Wait briefly so that the mark timestamp is in the past
	time.Sleep(20 * time.Millisecond)
	// ClearMarks with a tiny duration should now clear the mark
	lm.ClearMarks(1 * time.Millisecond)
	if lm.IsMarked(0) {
		t.Fatal("mark should be cleared after duration exceeded")
	}
}

// --- TemporaryConfigurationManager: RemoveJoinRequest, RemoveLeaveRequest ---

func TestTCMRemoveJoinRequest(t *testing.T) {
	cfg := testHydraConfig(1, 0, 1, 2, 3)
	tcm := NewTemporaryConfigurationManager(cfg)
	req := &MemberRequest{Type: RequestJoin, ID: 9, PublicKey: []byte("pk"), Power: 1, Signature: []byte("sig")}
	if err := tcm.AddJoinRequest(req); err != nil {
		t.Fatalf("add: %v", err)
	}
	tcm.RemoveJoinRequest(9)
	joins := tcm.GetPendingJoins()
	if len(joins) != 0 {
		t.Fatalf("expected 0 pending joins after remove, got %d", len(joins))
	}
}

func TestTCMRemoveLeaveRequest(t *testing.T) {
	cfg := testHydraConfig(1, 0, 1, 2, 3)
	tcm := NewTemporaryConfigurationManager(cfg)
	req := &MemberRequest{Type: RequestLeave, ID: 2, PublicKey: []byte("pk"), Power: 1, Signature: []byte("sig")}
	if err := tcm.AddLeaveRequest(req); err != nil {
		t.Fatalf("add: %v", err)
	}
	tcm.RemoveLeaveRequest(2)
	leaves := tcm.GetPendingLeaves()
	if len(leaves) != 0 {
		t.Fatalf("expected 0 pending leaves after remove, got %d", len(leaves))
	}
}

// --- TemporaryConfigurationManager: IsValidConfiguration ---

func TestTCMIsValidConfiguration(t *testing.T) {
	cfg := testHydraConfig(1, 0, 1, 2, 3)
	tcm := NewTemporaryConfigurationManager(cfg)

	valid := testHydraConfig(2, 0, 1, 2, 3) // n=4
	if !tcm.IsValidConfiguration(valid) {
		t.Fatal("config with 4 validators should be valid")
	}

	tooSmall := &Configuration{ID: 3, Validators: map[uint64]*Validator{0: {ID: 0, IsActive: true}}, QuorumSize: 1}
	if tcm.IsValidConfiguration(tooSmall) {
		t.Fatal("config with 1 validator should be invalid")
	}
}

// --- TemporaryConfigurationManager: PromoteMhighToMvalid ---

func TestTCMPromoteMhighToMvalid(t *testing.T) {
	cfg := testHydraConfig(1, 0, 1, 2, 3)
	tcm := NewTemporaryConfigurationManager(cfg)
	// Add a join request and apply to produce Mhigh
	tcm.AddJoinRequest(&MemberRequest{
		ID:        99,
		Type:      RequestJoin,
		PublicKey: []byte("key99"),
		Power:     1,
	})
	_, err := tcm.ApplyMemberRequests()
	if err != nil {
		t.Fatalf("ApplyMemberRequests: %v", err)
	}
	// Promote: currentConfigID was 1, increments to 2
	if err := tcm.PromoteMhighToMvalid(); err != nil {
		t.Fatalf("PromoteMhighToMvalid: %v", err)
	}
	mvalid := tcm.GetMvalid()
	if mvalid == nil || mvalid.ID != 2 {
		t.Fatalf("expected Mvalid.ID=2, got %d", mvalid.ID)
	}
}

// --- TemporaryConfigurationManager: GetMhigh ---

func TestTCMGetMhigh(t *testing.T) {
	cfg := testHydraConfig(1, 0, 1, 2, 3)
	tcm := NewTemporaryConfigurationManager(cfg)
	mhigh := tcm.GetMhigh()
	// Initially Mhigh should equal initial config
	if mhigh == nil {
		t.Fatal("expected non-nil Mhigh")
	}
	if mhigh.ID != 1 {
		t.Fatalf("expected Mhigh ID=1, got %d", mhigh.ID)
	}
}

// --- CDManager: GetHighestValidConfig, CanParticipate ---

func TestCDManagerGetHighestValidConfig(t *testing.T) {
	cfg := testHydraConfig(0, 0, 1, 2, 3)
	net := &recordingNetwork{}
	cdm := NewConfigurationDiscoveryManager(0, cfg, net)
	got := cdm.GetHighestValidConfig()
	if got == nil || got.ID != 0 {
		t.Fatal("expected highest valid config = initial config")
	}
}

func TestCDManagerCanParticipate(t *testing.T) {
	cfg := testHydraConfig(0, 0, 1, 2, 3)
	net := &recordingNetwork{}
	cdm := NewConfigurationDiscoveryManager(0, cfg, net)
	if !cdm.CanParticipate() {
		t.Fatal("should be able to participate when highestValidConfig is set")
	}
}

// --- DiscoveryType String ---

func TestDiscoveryTypeString(t *testing.T) {
	if DiscoveryRequest_.String() != "CDIS" {
		t.Fatalf("unexpected string for DiscoveryRequest_: %s", DiscoveryRequest_.String())
	}
	if DiscoveryResponse.String() != "DIS" {
		t.Fatalf("unexpected string for DiscoveryResponse: %s", DiscoveryResponse.String())
	}
	unknown := DiscoveryType(99)
	if unknown.String() != "UNKNOWN" {
		t.Fatalf("unexpected string for unknown: %s", unknown.String())
	}
}

// --- HydraManager: HandleMessage, AllowedLeaders, IsAllowedLeader, CanParticipate ---

func TestHydraManagerHandleMessageNil(t *testing.T) {
	hm := buildTestHydraManager(t)
	// nil message should return nil
	if err := hm.HandleMessage("string-type"); err != nil {
		t.Fatalf("unexpected error for unrecognized message type: %v", err)
	}
}

func TestHydraManagerAllowedLeaders(t *testing.T) {
	hm := buildTestHydraManager(t)
	leaders := hm.AllowedLeaders()
	if len(leaders) == 0 {
		t.Fatal("expected non-empty allowed leaders")
	}
}

func TestHydraManagerIsAllowedLeader(t *testing.T) {
	hm := buildTestHydraManager(t)
	// Initially all active validators are allowed
	if !hm.IsAllowedLeader(0) {
		t.Fatal("validator 0 should be allowed leader")
	}
}

func TestHydraManagerIsAllowedLeaderNilManager(t *testing.T) {
	var hm *HM
	leaders := hm.AllowedLeaders()
	if leaders != nil {
		t.Fatal("nil HM should return nil leaders")
	}
}

func TestHydraManagerCanParticipate(t *testing.T) {
	hm := buildTestHydraManager(t)
	if !hm.CanParticipate() {
		t.Fatal("should be able to participate")
	}
}

func TestHydraManagerIsResponsibleValidatorUnknownHash(t *testing.T) {
	hm := buildTestHydraManager(t)
	if hm.IsResponsibleValidator("unknown-hash") {
		t.Fatal("should not be responsible for unknown request")
	}
}

// --- ATM: GCCollectors ---

func TestATMGCCollectors(t *testing.T) {
	cfg := testHydraConfig(1, 0, 1, 2, 3)
	lm, _ := NewLSetManager(cfg.Validators)
	tcm := NewTemporaryConfigurationManager(cfg)
	net := &recordingNetwork{}
	atm := NewAutoTransitionManager(lm, tcm, net)
	// GCCollectors should not panic on empty state
	atm.GCCollectors(1 * time.Second)
}

// --- FaultClass: String ---

func TestFaultClassStringAll(t *testing.T) {
	tests := []struct {
		fc   FaultClass
		want string
	}{
		{FaultClassNone, "none"},
		{FaultClassDegraded, "degraded"},
		{FaultClassUnavailable, "unavailable"},
		{FaultClassByzantine, "byzantine"},
	}
	for _, tc := range tests {
		if tc.fc.String() != tc.want {
			t.Errorf("FaultClass(%d).String() = %q, want %q", tc.fc, tc.fc.String(), tc.want)
		}
	}
}

// --- configsEqualHydra ---

func TestConfigsEqualHydraNils(t *testing.T) {
	if !configsEqualHydra(nil, nil) {
		t.Fatal("nil == nil should be true")
	}
	cfg := testHydraConfig(1, 0, 1, 2, 3)
	if configsEqualHydra(cfg, nil) {
		t.Fatal("cfg == nil should be false")
	}
	if configsEqualHydra(nil, cfg) {
		t.Fatal("nil == cfg should be false")
	}
}

func TestConfigsEqualHydraDifferentIDs(t *testing.T) {
	a := testHydraConfig(1, 0, 1, 2, 3)
	b := testHydraConfig(2, 0, 1, 2, 3)
	if configsEqualHydra(a, b) {
		t.Fatal("different IDs should not be equal")
	}
}

// --- configFromTypes nil ---

func TestConfigFromTypesNil(t *testing.T) {
	got := configFromTypes(nil)
	if got != nil {
		t.Fatal("expected nil for nil input")
	}
}

func TestConfigFromTypesWithNilValidator(t *testing.T) {
	cfg := &types.Configuration{
		ID:         1,
		QuorumSize: 3,
		Validators: map[uint64]*types.Validator{
			0: nil,
			1: {ID: 1, PublicKey: []byte("pk"), Power: 1, IsActive: true},
		},
	}
	got := configFromTypes(cfg)
	if got == nil {
		t.Fatal("expected non-nil config")
	}
	if len(got.Validators) != 1 {
		t.Fatalf("expected 1 non-nil validator, got %d", len(got.Validators))
	}
}

// --- helper ---

func buildTestHydraManager(t *testing.T) *HM {
	t.Helper()
	cfg := testHydraConfig(1, 0, 1, 2, 3)
	net := &recordingNetwork{}
	hm, err := NewHydraManager(0, cfg.Validators, net)
	if err != nil {
		t.Fatalf("NewHydraManager: %v", err)
	}
	return hm
}

// --- ATM: Start/Stop ---

func TestATMStartStop(t *testing.T) {
	cfg := testHydraConfig(1, 0, 1, 2, 3)
	lm, _ := NewLSetManager(cfg.Validators)
	tcm := NewTemporaryConfigurationManager(cfg)
	net := &recordingNetwork{}
	atm := NewAutoTransitionManager(lm, tcm, net)
	atm.Start()
	atm.Stop()
}

// --- CDManager: Start/Stop ---

func TestCDManagerStartStop(t *testing.T) {
	cfg := testHydraConfig(0, 0, 1, 2, 3)
	net := &recordingNetwork{}
	cdm := NewConfigurationDiscoveryManager(0, cfg, net)
	cdm.Start()
	cdm.Stop()
}

// --- HydraManager: Start/Stop ---

func TestHydraManagerStartStop(t *testing.T) {
	hm := buildTestHydraManager(t)
	hm.Start()
	// Give SyncInBackground goroutine a moment to start, then stop
	time.Sleep(10 * time.Millisecond)
	hm.Stop()
}

// --- copyAutoVoteMessage ---

func TestCopyAutoVoteMessageNil(t *testing.T) {
	got := copyAutoVoteMessage(nil)
	if got != nil {
		t.Fatal("expected nil for nil input")
	}
}

func TestCopyAutoVoteMessageNonNil(t *testing.T) {
	msg := &AutoTransitionMessage{
		Type:        AutoPropose,
		SenderID:    1,
		View:        10,
		BlockHash:   []byte("hash"),
		Leaves:      []uint64{2, 3},
		NewConfigID: 5,
		Signature:   []byte("sig"),
	}
	cp := copyAutoVoteMessage(msg)
	if cp == nil {
		t.Fatal("expected non-nil copy")
	}
	if cp.SenderID != 1 || cp.View != 10 || cp.NewConfigID != 5 {
		t.Fatal("fields mismatch")
	}
	// Ensure deep copy
	msg.BlockHash[0] = 'X'
	if cp.BlockHash[0] == 'X' {
		t.Fatal("block hash not deep copied")
	}
}

// --- HydraManager: TriggerAutoTransition ---

func TestHydraManagerTriggerAutoTransition(t *testing.T) {
	hm := buildTestHydraManager(t)
	hm.Start()
	defer hm.Stop()
	// Node 0 is in L-set but no one is marked, so trigger should fail
	err := hm.TriggerAutoTransition(1, 5)
	if err == nil {
		t.Fatal("expected error when no evictable targets")
	}
}

// --- HydraManager: RequestConfigDiscovery ---

func TestHydraManagerRequestConfigDiscovery(t *testing.T) {
	hm := buildTestHydraManager(t)
	// Request discovery for a future config
	err := hm.RequestConfigDiscovery(10)
	if err != nil {
		t.Fatalf("RequestConfigDiscovery: %v", err)
	}
}

// --- HydraManager: HandleMessage with DiscoveryResponseMessage ---

func TestHydraManagerHandleMessageDiscoveryResponse(t *testing.T) {
	hm := buildTestHydraManager(t)
	resp := &DiscoveryResponseMessage{
		Configs: []*Configuration{
			testHydraConfig(2, 0, 1, 2, 3, 4),
		},
	}
	if err := hm.HandleMessage(resp); err != nil {
		t.Fatalf("HandleMessage(DiscoveryResponseMessage): %v", err)
	}
}

func TestHydraManagerHandleMessageUnknownType(t *testing.T) {
	hm := buildTestHydraManager(t)
	// Unknown type should return nil
	if err := hm.HandleMessage("unknown"); err != nil {
		t.Fatalf("HandleMessage(unknown type): %v", err)
	}
}

// --- CDManager: SyncInBackground ---

func TestCDManagerSyncInBackground(t *testing.T) {
	cfg := testHydraConfig(0, 0, 1, 2, 3)
	net := &recordingNetwork{}
	cdm := NewConfigurationDiscoveryManager(0, cfg, net)
	cdm.Start()
	cdm.SyncInBackground()
	// Let it tick once with a very short sleep, then stop
	time.Sleep(10 * time.Millisecond)
	cdm.Stop()
}

// --- CDManager: GetParticipatingConfig ---

func TestCDManagerGetParticipatingConfig(t *testing.T) {
	cfg := testHydraConfig(0, 0, 1, 2, 3)
	net := &recordingNetwork{}
	cdm := NewConfigurationDiscoveryManager(0, cfg, net)
	pc := cdm.GetParticipatingConfig()
	if pc == nil {
		t.Fatal("expected non-nil participating config")
	}
}

// --- AutoTransitionType: String ---

func TestAutoTransitionTypeString(t *testing.T) {
	if AutoPropose.String() != "AUTO_PROPOSE" {
		t.Errorf("AutoPropose.String() = %q", AutoPropose.String())
	}
	if AutoVote.String() != "AUTO_VOTE" {
		t.Errorf("AutoVote.String() = %q", AutoVote.String())
	}
	if AutoCommit.String() != "AUTO_COMMIT" {
		t.Errorf("AutoCommit.String() = %q", AutoCommit.String())
	}
	// Unknown
	if AutoTransitionType(99).String() != "UNKNOWN" {
		t.Errorf("AutoTransitionType(99).String() = %q", AutoTransitionType(99).String())
	}
}

// --- SigningBytes nil ---

func TestSigningBytesNil(t *testing.T) {
	var msg *AutoTransitionMessage
	b, err := msg.SigningBytes()
	if err != nil || b != nil {
		t.Fatal("nil message should return nil, nil")
	}
}

// --- GetMhigh nil branch ---

func TestTCMGetMhighNilFallback(t *testing.T) {
	cfg := testHydraConfig(1, 0, 1, 2, 3)
	tcm := NewTemporaryConfigurationManager(cfg)
	// Force Mhigh to nil (TCM normally sets Mhigh = Mvalid in constructor)
	tcm.mu.Lock()
	tcm.Mhigh = nil
	tcm.mu.Unlock()
	mhigh := tcm.GetMhigh()
	if mhigh == nil {
		t.Fatal("GetMhigh should fallback to Mvalid when Mhigh is nil")
	}
	if mhigh.ID != 1 {
		t.Fatalf("expected fallback to Mvalid (ID=1), got %d", mhigh.ID)
	}
}

// --- HandleMessage: all branches via HydraManager ---

func TestHydraManagerHandleMessageAutoTransition(t *testing.T) {
	hm := buildTestHydraManager(t)
	msg := &AutoTransitionMessage{
		Type:     AutoPropose,
		SenderID: 0,
		View:     1,
	}
	// Should not panic; may return error if message is invalid
	_ = hm.HandleMessage(msg)
}

func TestHydraManagerHandleMessageDiscoveryRequest(t *testing.T) {
	hm := buildTestHydraManager(t)
	req := &DiscoveryRequest{
		Type:      DiscoveryRequest_,
		SenderID:  1,
		ConfigIDs: []uint64{2, 3},
	}
	err := hm.HandleMessage(req)
	if err != nil {
		t.Fatalf("HandleMessage(DiscoveryRequest): %v", err)
	}
}

// --- NetworkAdapter: NewNetworkAdapter ---

func TestNewNetworkAdapter(t *testing.T) {
	na := NewNetworkAdapter(nil, 42)
	if na == nil {
		t.Fatal("expected non-nil NetworkAdapter")
	}
	if na.nodeID != 42 {
		t.Fatalf("expected nodeID=42, got %d", na.nodeID)
	}
}
