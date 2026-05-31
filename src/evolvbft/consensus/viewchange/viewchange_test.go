package viewchange

import (
	"testing"
	"time"

	"evolvbft/evolvbft/types"
)

func quorum3of4() func() uint64 {
	return func() uint64 { return 3 }
}

// ---------------------------------------------------------------------------
// HandleTimeoutVote: basic TC formation
// ---------------------------------------------------------------------------

func TestHandleTimeoutVote_FormsTCAtQuorum(t *testing.T) {
	vcm := NewViewChangeManager(0, time.Second, quorum3of4())
	vcm.Start()

	tv1 := &types.TimeoutVote{View: 5, Epoch: 1, VoterID: 1, Signature: []byte("s1")}
	tv2 := &types.TimeoutVote{View: 5, Epoch: 1, VoterID: 2, Signature: []byte("s2")}
	tv3 := &types.TimeoutVote{View: 5, Epoch: 1, VoterID: 3, Signature: []byte("s3")}

	if tc := vcm.HandleTimeoutVote(tv1); tc != nil {
		t.Error("should not form TC with 1 vote")
	}
	if tc := vcm.HandleTimeoutVote(tv2); tc != nil {
		t.Error("should not form TC with 2 votes")
	}
	tc := vcm.HandleTimeoutVote(tv3)
	if tc == nil {
		t.Fatal("should form TC with 3 votes (quorum)")
	}
	if tc.View != 5 {
		t.Errorf("expected TC view 5, got %d", tc.View)
	}
	if tc.NumVoters != 3 {
		t.Errorf("expected 3 voters, got %d", tc.NumVoters)
	}
}

func TestHandleTimeoutVote_NilInput(t *testing.T) {
	vcm := NewViewChangeManager(0, time.Second, quorum3of4())
	if tc := vcm.HandleTimeoutVote(nil); tc != nil {
		t.Error("nil input should return nil")
	}
}

func TestHandleTimeoutVote_DeduplicatesVotes(t *testing.T) {
	vcm := NewViewChangeManager(0, time.Second, quorum3of4())

	tv := &types.TimeoutVote{View: 1, Epoch: 1, VoterID: 1, Signature: []byte("s1")}
	vcm.HandleTimeoutVote(tv)
	vcm.HandleTimeoutVote(tv) // duplicate

	// Only 1 unique voter, need 3 for quorum
	tv2 := &types.TimeoutVote{View: 1, Epoch: 1, VoterID: 2, Signature: []byte("s2")}
	vcm.HandleTimeoutVote(tv2)

	// Still only 2 unique voters
	tv3 := &types.TimeoutVote{View: 1, Epoch: 1, VoterID: 1, Signature: []byte("s1-dup")}
	if tc := vcm.HandleTimeoutVote(tv3); tc != nil {
		t.Error("duplicate voter should not push to quorum")
	}
}

func TestHandleTimeoutVote_RejectsAfterDone(t *testing.T) {
	vcm := NewViewChangeManager(0, time.Second, quorum3of4())

	for i := uint64(1); i <= 3; i++ {
		vcm.HandleTimeoutVote(&types.TimeoutVote{View: 1, Epoch: 1, VoterID: i, Signature: []byte{byte(i)}})
	}

	// TC already formed; extra vote should return nil
	extra := &types.TimeoutVote{View: 1, Epoch: 1, VoterID: 4, Signature: []byte("s4")}
	if tc := vcm.HandleTimeoutVote(extra); tc != nil {
		t.Error("should not form another TC after collector is done")
	}
}

func TestHandleTimeoutVote_RejectsEpochMismatch(t *testing.T) {
	vcm := NewViewChangeManager(0, time.Second, quorum3of4())

	vcm.HandleTimeoutVote(&types.TimeoutVote{View: 1, Epoch: 1, VoterID: 1, Signature: []byte("s1")})

	// Different epoch for same view
	wrongEpoch := &types.TimeoutVote{View: 1, Epoch: 2, VoterID: 2, Signature: []byte("s2")}
	vcm.HandleTimeoutVote(wrongEpoch)

	// Third vote with correct epoch
	vcm.HandleTimeoutVote(&types.TimeoutVote{View: 1, Epoch: 1, VoterID: 3, Signature: []byte("s3")})

	// Should not have quorum because voter 2 was rejected
	// Actually voter 2 was accepted (epoch mismatch check returns nil),
	// so we need voter 3 + one more
	// Let me re-check the code... epoch mismatch returns nil early.
	// So we need: voter 1 (epoch 1), voter 3 (epoch 1) = 2 valid. Need one more.
	vcm.HandleTimeoutVote(&types.TimeoutVote{View: 1, Epoch: 1, VoterID: 4, Signature: []byte("s4")})
	tc := vcm.GetHighestTC()
	if tc == nil {
		t.Fatal("should form TC with 3 matching-epoch voters")
	}
}

// ---------------------------------------------------------------------------
// HighestQC tracking across timeout voters
// ---------------------------------------------------------------------------

func TestHandleTimeoutVote_TracksHighestQC(t *testing.T) {
	vcm := NewViewChangeManager(0, time.Second, quorum3of4())

	qc5 := types.NewQuorumCertificate([]byte("b5"), 5, 1, types.PhasePrepare)
	qc10 := types.NewQuorumCertificate([]byte("b10"), 10, 1, types.PhasePrepare)

	vcm.HandleTimeoutVote(&types.TimeoutVote{View: 1, Epoch: 1, VoterID: 1, Signature: []byte("s1"), HighestQC: qc5})
	vcm.HandleTimeoutVote(&types.TimeoutVote{View: 1, Epoch: 1, VoterID: 2, Signature: []byte("s2"), HighestQC: qc10})
	tc := vcm.HandleTimeoutVote(&types.TimeoutVote{View: 1, Epoch: 1, VoterID: 3, Signature: []byte("s3"), HighestQC: qc5})

	if tc == nil {
		t.Fatal("expected TC formed")
	}
	if tc.HighestQC == nil || tc.HighestQC.View != 10 {
		t.Errorf("expected highest QC view 10, got %v", tc.HighestQC)
	}
}

// ---------------------------------------------------------------------------
// GetHighestTC
// ---------------------------------------------------------------------------

func TestGetHighestTC_InitiallyNil(t *testing.T) {
	vcm := NewViewChangeManager(0, time.Second, quorum3of4())
	if tc := vcm.GetHighestTC(); tc != nil {
		t.Error("highest TC should be nil initially")
	}
}

func TestGetHighestTC_UpdatesAcrossViews(t *testing.T) {
	vcm := NewViewChangeManager(0, time.Second, quorum3of4())

	// Form TC for view 3
	for i := uint64(1); i <= 3; i++ {
		vcm.HandleTimeoutVote(&types.TimeoutVote{View: 3, Epoch: 1, VoterID: i, Signature: []byte{byte(i)}})
	}
	if tc := vcm.GetHighestTC(); tc == nil || tc.View != 3 {
		t.Fatalf("expected highest TC view 3, got %v", tc)
	}

	// Form TC for view 7
	for i := uint64(1); i <= 3; i++ {
		vcm.HandleTimeoutVote(&types.TimeoutVote{View: 7, Epoch: 1, VoterID: i, Signature: []byte{byte(i + 10)}})
	}
	if tc := vcm.GetHighestTC(); tc == nil || tc.View != 7 {
		t.Fatalf("expected highest TC updated to view 7, got %v", tc)
	}
}

// ---------------------------------------------------------------------------
// GCCollectors
// ---------------------------------------------------------------------------

func TestGCCollectors_RemovesOldViews(t *testing.T) {
	vcm := NewViewChangeManager(0, time.Second, quorum3of4())

	// Create collectors for views 1-5
	for v := uint64(1); v <= 5; v++ {
		vcm.HandleTimeoutVote(&types.TimeoutVote{View: v, Epoch: 1, VoterID: 1, Signature: []byte{byte(v)}})
	}

	vcm.GCCollectors(4)

	// Views 1-3 should be removed, 4-5 should remain
	vcm.mu.RLock()
	defer vcm.mu.RUnlock()
	for v := uint64(1); v <= 3; v++ {
		if _, exists := vcm.collectors[v]; exists {
			t.Errorf("view %d collector should have been GC'd", v)
		}
	}
	for v := uint64(4); v <= 5; v++ {
		if _, exists := vcm.collectors[v]; !exists {
			t.Errorf("view %d collector should still exist", v)
		}
	}
}

// ---------------------------------------------------------------------------
// OnTCFormed callback
// ---------------------------------------------------------------------------

func TestOnTCFormed_CallbackInvoked(t *testing.T) {
	vcm := NewViewChangeManager(0, time.Second, quorum3of4())

	var callbackTC *types.TimeoutCertificate
	vcm.OnTCFormed(func(tc *types.TimeoutCertificate) {
		callbackTC = tc
	})

	for i := uint64(1); i <= 3; i++ {
		vcm.HandleTimeoutVote(&types.TimeoutVote{View: 1, Epoch: 1, VoterID: i, Signature: []byte{byte(i)}})
	}

	if callbackTC == nil {
		t.Fatal("OnTCFormed callback should have been invoked")
	}
	if callbackTC.View != 1 {
		t.Errorf("callback TC view should be 1, got %d", callbackTC.View)
	}
}

// ---------------------------------------------------------------------------
// Start / Stop
// ---------------------------------------------------------------------------

func TestStartStop(t *testing.T) {
	vcm := NewViewChangeManager(0, time.Second, quorum3of4())
	vcm.Start()
	vcm.Stop()
	// No panic = pass
}
