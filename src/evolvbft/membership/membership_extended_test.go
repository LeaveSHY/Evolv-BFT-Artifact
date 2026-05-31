package membership

import (
	"testing"

	"evolvbft/evolvbft/crypto"
	"evolvbft/evolvbft/types"
)

// ─── helpers ─────────────────────────────────────────────────────────────────

func makeValidator(id uint64) *types.Validator {
	kp, _ := crypto.GenerateKeyPair()
	return &types.Validator{ID: id, PublicKey: kp.PublicKey, Power: 1, IsActive: true}
}

func makeManager(ids ...uint64) *MembershipManager {
	vals := make(map[uint64]*types.Validator, len(ids))
	for _, id := range ids {
		vals[id] = makeValidator(id)
	}
	return NewMembershipManager(vals)
}

// ─── SubmitJoinRequest ───────────────────────────────────────────────────────

func TestSubmitJoinRequest_NewNode(t *testing.T) {
	mm := makeManager(1, 2, 3)
	kp, _ := crypto.GenerateKeyPair()
	err := mm.SubmitJoinRequest(&types.JoinRequest{ID: 10, PublicKey: kp.PublicKey, Power: 1})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, ok := mm.pendingJoins[10]; !ok {
		t.Fatal("expected pending join for node 10")
	}
}

func TestSubmitJoinRequest_AlreadyValidator(t *testing.T) {
	mm := makeManager(1, 2, 3)
	kp, _ := crypto.GenerateKeyPair()
	err := mm.SubmitJoinRequest(&types.JoinRequest{ID: 1, PublicKey: kp.PublicKey})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Should be silently ignored (already a validator)
	if _, ok := mm.pendingJoins[1]; ok {
		t.Fatal("should not add existing validator to pending joins")
	}
}

func TestSubmitJoinRequest_Duplicate(t *testing.T) {
	mm := makeManager(1)
	kp, _ := crypto.GenerateKeyPair()
	_ = mm.SubmitJoinRequest(&types.JoinRequest{ID: 5, PublicKey: kp.PublicKey})
	_ = mm.SubmitJoinRequest(&types.JoinRequest{ID: 5, PublicKey: kp.PublicKey})
	if len(mm.pendingJoins) != 1 {
		t.Fatalf("expected 1 pending join, got %d", len(mm.pendingJoins))
	}
}

// ─── SubmitLeaveRequest ──────────────────────────────────────────────────────

func TestSubmitLeaveRequest_ExistingValidator(t *testing.T) {
	mm := makeManager(1, 2, 3)
	pubKey := mm.currentConfig.Validators[2].PublicKey
	err := mm.SubmitLeaveRequest(&types.LeaveRequest{ID: 2, PublicKey: pubKey})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, ok := mm.pendingLeaves[2]; !ok {
		t.Fatal("expected pending leave for node 2")
	}
}

func TestSubmitLeaveRequest_NonValidator(t *testing.T) {
	mm := makeManager(1, 2)
	kp, _ := crypto.GenerateKeyPair()
	err := mm.SubmitLeaveRequest(&types.LeaveRequest{ID: 99, PublicKey: kp.PublicKey})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, ok := mm.pendingLeaves[99]; ok {
		t.Fatal("should not add non-validator to pending leaves")
	}
}

// ─── ApplyMembershipChanges ─────────────────────────────────────────────────

func TestApplyMembershipChanges_JoinViaBlock(t *testing.T) {
	mm := makeManager(1, 2, 3)
	kp, _ := crypto.GenerateKeyPair()
	_ = mm.SubmitJoinRequest(&types.JoinRequest{ID: 10, PublicKey: kp.PublicKey, Power: 1})

	block := &types.Block{
		JoinRequests: []types.PublicKey{kp.PublicKey},
	}
	if err := mm.ApplyMembershipChanges(block); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !mm.IsValidator(10) {
		t.Fatal("node 10 should be a validator after join")
	}
	if mm.GetValidatorCount() != 4 {
		t.Fatalf("expected 4 validators, got %d", mm.GetValidatorCount())
	}
}

func TestApplyMembershipChanges_LeaveViaBlock(t *testing.T) {
	mm := makeManager(1, 2, 3, 4)
	pubKey := mm.currentConfig.Validators[3].PublicKey
	_ = mm.SubmitLeaveRequest(&types.LeaveRequest{ID: 3, PublicKey: pubKey})

	block := &types.Block{
		LeaveRequests: []types.PublicKey{pubKey},
	}
	if err := mm.ApplyMembershipChanges(block); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mm.IsValidator(3) {
		t.Fatal("node 3 should not be a validator after leave")
	}
	if mm.GetValidatorCount() != 3 {
		t.Fatalf("expected 3 validators, got %d", mm.GetValidatorCount())
	}
}

func TestApplyMembershipChanges_NoChanges(t *testing.T) {
	mm := makeManager(1, 2)
	initialID := mm.currentConfig.ID
	block := &types.Block{}
	if err := mm.ApplyMembershipChanges(block); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Config ID should not change
	if mm.currentConfig.ID != initialID {
		t.Fatalf("expected config ID %d, got %d", initialID, mm.currentConfig.ID)
	}
}

// ─── GetValidatorCount / GetQuorumSize / IsValidator ────────────────────────

func TestGetValidatorCount(t *testing.T) {
	mm := makeManager(1, 2, 3, 4, 5)
	if mm.GetValidatorCount() != 5 {
		t.Fatalf("expected 5, got %d", mm.GetValidatorCount())
	}
}

func TestGetQuorumSize(t *testing.T) {
	mm := makeManager(1, 2, 3, 4)
	q := mm.GetQuorumSize()
	// 2/3*4 + 1 = 3
	if q != 3 {
		t.Fatalf("expected quorum 3, got %d", q)
	}
}

func TestIsValidator(t *testing.T) {
	mm := makeManager(1, 2, 3)
	if !mm.IsValidator(1) {
		t.Fatal("node 1 should be a validator")
	}
	if mm.IsValidator(99) {
		t.Fatal("node 99 should not be a validator")
	}
}

// ─── GetLatestEvent ─────────────────────────────────────────────────────────

func TestGetLatestEvent_InitiallyNil(t *testing.T) {
	mm := makeManager(1, 2)
	if mm.GetLatestEvent() != nil {
		t.Fatal("expected nil initial event")
	}
}

func TestGetLatestEvent_AfterChange(t *testing.T) {
	mm := makeManager(1, 2, 3)
	kp, _ := crypto.GenerateKeyPair()
	_ = mm.SubmitJoinRequest(&types.JoinRequest{ID: 10, PublicKey: kp.PublicKey, Power: 1})
	block := &types.Block{JoinRequests: []types.PublicKey{kp.PublicKey}}
	_ = mm.ApplyMembershipChanges(block)

	event := mm.GetLatestEvent()
	if event == nil {
		t.Fatal("expected non-nil event")
	}
	if len(event.Added) != 1 || event.Added[0] != 10 {
		t.Fatalf("expected Added=[10], got %v", event.Added)
	}
}

// ─── InstallValidatorSet ────────────────────────────────────────────────────

func TestInstallValidatorSet_NewEpoch(t *testing.T) {
	mm := makeManager(1, 2, 3)

	// Build new validator set reusing existing keys + adding node 4
	newVals := make(map[uint64]*types.Validator)
	for id, v := range mm.currentConfig.Validators {
		copyV := *v
		newVals[id] = &copyV
	}
	newVals[4] = makeValidator(4)

	valSet := &types.ValidatorSet{
		Epoch:      mm.currentConfig.ID + 1,
		Validators: newVals,
	}

	cfg, event, changed, err := mm.InstallValidatorSet(valSet)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !changed {
		t.Fatal("expected config to change")
	}
	if len(cfg.Validators) != 4 {
		t.Fatalf("expected 4 validators, got %d", len(cfg.Validators))
	}
	if event == nil {
		t.Fatal("expected non-nil event")
	}
	if len(event.Added) != 1 {
		t.Fatalf("expected 1 added, got %d", len(event.Added))
	}
}

func TestInstallValidatorSet_NilReturnsError(t *testing.T) {
	mm := makeManager(1, 2)
	_, _, _, err := mm.InstallValidatorSet(nil)
	if err == nil {
		t.Fatal("expected error for nil validator set")
	}
}

func TestInstallValidatorSet_OldEpochIgnored(t *testing.T) {
	mm := makeManager(1, 2, 3)
	// First install epoch 2
	newVals := make(map[uint64]*types.Validator)
	for _, id := range []uint64{1, 2, 3} {
		newVals[id] = makeValidator(id)
	}
	mm.InstallValidatorSet(&types.ValidatorSet{Epoch: 2, Validators: newVals})

	// Try installing epoch 1 (old) — should be ignored
	_, _, changed, err := mm.InstallValidatorSet(&types.ValidatorSet{Epoch: 1, Validators: newVals})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if changed {
		t.Fatal("old epoch should not cause change")
	}
}
