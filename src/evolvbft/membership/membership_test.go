package membership

import (
	"testing"

	"evolvbft/evolvbft/crypto"
	"evolvbft/evolvbft/types"
)

func TestApplyReconfigData_IdempotentJoinLeave(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	initial := map[uint64]*types.Validator{
		0: {
			ID:        0,
			PublicKey: kp0.PublicKey,
			Power:     1,
			IsActive:  true,
		},
	}
	mm := NewMembershipManager(initial)

	kp1, _ := crypto.GenerateKeyPair()
	join := &types.ReconfigData{
		Type:         types.ReconfigJoin,
		NodeID:       1,
		PublicKey:    kp1.PublicKey,
		VRFPublicKey: []byte("vrf-1"),
		Power:        1,
	}
	cfg, _, changed, err := mm.ApplyReconfigData(join)
	if err != nil || !changed {
		t.Fatalf("join failed: changed=%v err=%v", changed, err)
	}
	if _, ok := cfg.Validators[1]; !ok {
		t.Fatalf("joined validator missing")
	}
	if got := string(cfg.Validators[1].VRFPublicKey); got != "vrf-1" {
		t.Fatalf("unexpected joined validator vrf public key: %q", got)
	}

	cfg, _, changed, err = mm.ApplyReconfigData(join)
	if err != nil || changed {
		t.Fatalf("duplicate join should be idempotent changed=%v err=%v", changed, err)
	}
	if _, ok := cfg.Validators[1]; !ok {
		t.Fatalf("validator missing after duplicate join")
	}

	leave := &types.ReconfigData{
		Type:      types.ReconfigLeave,
		NodeID:    1,
		PublicKey: kp1.PublicKey,
	}
	cfg, _, changed, err = mm.ApplyReconfigData(leave)
	if err != nil || !changed {
		t.Fatalf("leave failed: changed=%v err=%v", changed, err)
	}
	if _, ok := cfg.Validators[1]; ok {
		t.Fatalf("validator should be removed")
	}

	cfg, _, changed, err = mm.ApplyReconfigData(leave)
	if err != nil || changed {
		t.Fatalf("duplicate leave should be idempotent changed=%v err=%v", changed, err)
	}
}

func TestApplyReconfigData_OrderConsistency(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	kp2, _ := crypto.GenerateKeyPair()
	initial := map[uint64]*types.Validator{
		0: {
			ID:        0,
			PublicKey: kp0.PublicKey,
			Power:     1,
			IsActive:  true,
		},
	}
	mmA := NewMembershipManager(initial)
	mmB := NewMembershipManager(initial)

	join1 := &types.ReconfigData{Type: types.ReconfigJoin, NodeID: 1, PublicKey: kp1.PublicKey, Power: 1}
	join2 := &types.ReconfigData{Type: types.ReconfigJoin, NodeID: 2, PublicKey: kp2.PublicKey, Power: 1}

	_, _, _, _ = mmA.ApplyReconfigData(join1)
	cfgA, _, _, _ := mmA.ApplyReconfigData(join2)

	_, _, _, _ = mmB.ApplyReconfigData(join2)
	cfgB, _, _, _ := mmB.ApplyReconfigData(join1)

	if string(cfgA.Hash()) != string(cfgB.Hash()) {
		t.Fatalf("config hash mismatch under different arrival order")
	}
}

func TestInstallValidatorSetFromTransitionsUsesOrderedTransitionMetadata(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	initial := map[uint64]*types.Validator{
		0: {
			ID:        0,
			PublicKey: kp0.PublicKey,
			Power:     1,
			IsActive:  true,
		},
	}
	mm := NewMembershipManager(initial)
	valSet := types.NewValidatorSet(2, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
	})
	transitions := []*types.EpochTransition{{
		OldEpoch:         1,
		NewEpoch:         2,
		ActivationHeight: 7,
		ActivationRank:   11,
		Added:            []uint64{1},
		QuorumSize:       valSet.QuorumSize,
		ConfigHash:       valSet.Hash(),
	}}

	cfg, event, changed, err := mm.InstallValidatorSetFromTransitions(valSet, transitions)
	if err != nil || !changed {
		t.Fatalf("install validator set failed: changed=%v err=%v", changed, err)
	}
	if cfg.ID != 2 {
		t.Fatalf("unexpected installed config id: %d", cfg.ID)
	}
	if event == nil {
		t.Fatalf("expected config change event")
	}
	if event.OldEpoch != 1 || event.NewEpoch != 2 {
		t.Fatalf("unexpected event epochs: %+v", event)
	}
	if len(event.Added) != 1 || event.Added[0] != 1 {
		t.Fatalf("unexpected event added set: %+v", event.Added)
	}
	if string(event.ConfigHash) != string(valSet.Hash()) {
		t.Fatalf("unexpected config hash in event")
	}
}

func TestInstallValidatorSetFromTransitionsIgnoresStaleLowerEpoch(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	kp2, _ := crypto.GenerateKeyPair()
	initial := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
	}
	mm := NewMembershipManager(initial)

	newer := types.NewValidatorSet(2, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: kp2.PublicKey, Power: 1, IsActive: true},
	})
	if _, _, changed, err := mm.InstallValidatorSetFromTransitions(newer, []*types.EpochTransition{{OldEpoch: 1, NewEpoch: 2, Added: []uint64{2}, QuorumSize: newer.QuorumSize, ConfigHash: newer.Hash()}}); err != nil || !changed {
		t.Fatalf("install newer validator set failed: changed=%v err=%v", changed, err)
	}
	beforeHistory := len(mm.GetConfigHistory())

	stale := types.NewValidatorSet(1, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
	})
		cfg, _, changed, err := mm.InstallValidatorSetFromTransitions(stale, []*types.EpochTransition{{OldEpoch: 0, NewEpoch: 1, Added: []uint64{1}, QuorumSize: stale.QuorumSize, ConfigHash: stale.Hash()}})
	if err != nil {
		t.Fatalf("stale install returned error: %v", err)
	}
	if changed {
		t.Fatalf("expected stale lower-epoch install to be ignored")
	}
	if cfg == nil || cfg.ID != 2 {
		t.Fatalf("expected current config to remain at epoch 2, got %+v", cfg)
	}
	if _, ok := cfg.Validators[2]; !ok {
		t.Fatalf("expected newer validator to remain installed")
	}
	if _, ok := cfg.Validators[1]; ok {
		t.Fatalf("did not expect stale validator to be installed")
	}
	if got := len(mm.GetConfigHistory()); got != beforeHistory {
		t.Fatalf("expected stale install to avoid history append, got %d entries", got)
	}
}

func TestInstallValidatorSetFromTransitionsRejectsConflictingSameEpoch(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	kp2, _ := crypto.GenerateKeyPair()
	initial := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
	}
	mm := NewMembershipManager(initial)

	installed := types.NewValidatorSet(2, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
	})
	if _, _, changed, err := mm.InstallValidatorSetFromTransitions(installed, []*types.EpochTransition{{OldEpoch: 1, NewEpoch: 2, Added: []uint64{1}, QuorumSize: installed.QuorumSize, ConfigHash: installed.Hash()}}); err != nil || !changed {
		t.Fatalf("install validator set failed: changed=%v err=%v", changed, err)
	}
	beforeHistory := len(mm.GetConfigHistory())

	conflicting := types.NewValidatorSet(2, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		2: {ID: 2, PublicKey: kp2.PublicKey, Power: 1, IsActive: true},
	})
	if _, _, _, err := mm.InstallValidatorSetFromTransitions(conflicting, []*types.EpochTransition{{OldEpoch: 1, NewEpoch: 2, Added: []uint64{2}, QuorumSize: conflicting.QuorumSize, ConfigHash: conflicting.Hash()}}); err == nil {
		t.Fatalf("expected conflicting same-epoch install to fail")
	}
	cfg := mm.GetCurrentConfig()
	if cfg == nil || cfg.ID != 2 {
		t.Fatalf("expected current config to remain at epoch 2, got %+v", cfg)
	}
	if _, ok := cfg.Validators[1]; !ok {
		t.Fatalf("expected original epoch-2 validator to remain installed")
	}
	if _, ok := cfg.Validators[2]; ok {
		t.Fatalf("did not expect conflicting epoch-2 validator set to replace installed config")
	}
	if got := len(mm.GetConfigHistory()); got != beforeHistory {
		t.Fatalf("expected conflicting same-epoch install to avoid history append, got %d entries", got)
	}
}

func TestInstallValidatorSetFromTransitionsRejectsMismatchedTransitionMetadata(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	initial := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
	}
	mm := NewMembershipManager(initial)
	valSet := types.NewValidatorSet(2, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
	})
	beforeHistory := len(mm.GetConfigHistory())

	if _, _, _, err := mm.InstallValidatorSetFromTransitions(valSet, []*types.EpochTransition{{
		OldEpoch:         1,
		NewEpoch:         3,
		ActivationHeight: 7,
		ActivationRank:   11,
		Added:            []uint64{1},
		QuorumSize:       valSet.QuorumSize,
		ConfigHash:       valSet.Hash(),
	}}); err == nil {
		t.Fatalf("expected mismatched transition metadata to fail")
	}
	cfg := mm.GetCurrentConfig()
	if cfg == nil || cfg.ID != 1 {
		t.Fatalf("expected membership to remain at epoch 1, got %+v", cfg)
	}
	if got := len(mm.GetConfigHistory()); got != beforeHistory {
		t.Fatalf("expected mismatched transition metadata to avoid history append, got %d entries", got)
	}
}

func TestInstallValidatorSetFromTransitionsRejectsMismatchedValidatorDiffMetadata(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	initial := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
	}
	mm := NewMembershipManager(initial)
	valSet := types.NewValidatorSet(2, map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
	})
	beforeHistory := len(mm.GetConfigHistory())

	if _, _, _, err := mm.InstallValidatorSetFromTransitions(valSet, []*types.EpochTransition{{
		OldEpoch:         1,
		NewEpoch:         2,
		ActivationHeight: 7,
		ActivationRank:   11,
		Added:            []uint64{},
		QuorumSize:       valSet.QuorumSize,
		ConfigHash:       valSet.Hash(),
	}}); err == nil {
		t.Fatalf("expected mismatched validator diff metadata to fail")
	}
	cfg := mm.GetCurrentConfig()
	if cfg == nil || cfg.ID != 1 {
		t.Fatalf("expected membership to remain at epoch 1, got %+v", cfg)
	}
	if got := len(mm.GetConfigHistory()); got != beforeHistory {
		t.Fatalf("expected mismatched validator diff metadata to avoid history append, got %d entries", got)
	}
}

func TestApplyReconfigData_AutoLeaveRemovesValidator(t *testing.T) {
	kp0, _ := crypto.GenerateKeyPair()
	kp1, _ := crypto.GenerateKeyPair()
	initial := map[uint64]*types.Validator{
		0: {ID: 0, PublicKey: kp0.PublicKey, Power: 1, IsActive: true},
		1: {ID: 1, PublicKey: kp1.PublicKey, Power: 1, IsActive: true},
	}
	mm := NewMembershipManager(initial)

	autoLeave := &types.ReconfigData{
		Type:   types.ReconfigAutoLeave,
		NodeID: 1,
	}
	cfg, event, changed, err := mm.ApplyReconfigData(autoLeave)
	if err != nil || !changed {
		t.Fatalf("auto-leave failed: changed=%v err=%v", changed, err)
	}
	if _, ok := cfg.Validators[1]; ok {
		t.Fatalf("validator 1 should be removed by auto-leave")
	}
	if len(event.Removed) != 1 || event.Removed[0] != 1 {
		t.Fatalf("event should show node 1 removed, got %v", event.Removed)
	}

	// Idempotent: second auto-leave is no-op
	_, _, changed, err = mm.ApplyReconfigData(autoLeave)
	if err != nil || changed {
		t.Fatalf("duplicate auto-leave should be idempotent: changed=%v err=%v", changed, err)
	}
}
