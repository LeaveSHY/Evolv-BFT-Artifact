package adaptive

import (
	"testing"

	pb "evolvbft/evolvbft/adaptive/proto"
)

func TestSFACGRPCPolicyName(t *testing.T) {
	// Unit test: verify policy name without needing a live server
	// Full integration test requires running sfac_grpc_server.py
	p := &SFACGRPCPolicy{addr: "127.0.0.1:50051"}
	if got := p.Name(); got != "sfac-grpc" {
		t.Errorf("Name() = %q, want %q", got, "sfac-grpc")
	}
}

func TestSFACGRPCPolicyConvertResponseConsumesFullTuple(t *testing.T) {
	p := &SFACGRPCPolicy{addr: "127.0.0.1:50051"}
	obs := Observation{
		CommitteeSize:             4,
		PacemakerTimeoutMs:        1000,
		MempoolMaxBatchTxs:        2048,
		MempoolProposalIntervalMs: 100,
		Agents: []AgentObservation{{
			InstanceID:                2,
			ValidatorCount:            7,
			CommitteeSize:             4,
			PacemakerTimeoutMs:        1000,
			MempoolMaxBatchTxs:        2048,
			MempoolProposalIntervalMs: 100,
		}},
		TrustSnapshots: []TrustSnapshot{{NodeID: 21}, {NodeID: 22}},
	}
	got := p.convertResponse(&pb.SFACResponse{Actions: []*pb.SFACAgentAction{{
		InstanceID: 2,
		Reconfig:   []int32{-1, 1},
		Rotate:     true,
		Params:     []float64{5, 1500, 512, 75},
	}}}, obs)

	if got.SubmitLeave {
		t.Fatal("leader rotation must not be translated into local validator leave")
	}
	if len(got.AgentActions) != 1 {
		t.Fatalf("expected one agent action, got %+v", got.AgentActions)
	}
	aa := got.AgentActions[0]
	if aa.InstanceID != 2 || !aa.RotateLeader || aa.CommitteeSize != 5 || aa.PacemakerTimeoutMs != 1500 || aa.MempoolMaxBatchTxs != 512 || aa.MempoolProposalIntervalMs != 75 {
		t.Fatalf("unexpected gRPC SFAC action: %+v", aa)
	}
	if len(aa.ReconfigEvictNodeIDs) != 1 || aa.ReconfigEvictNodeIDs[0] != 21 || len(aa.ReconfigAdmitNodeIDs) != 1 || aa.ReconfigAdmitNodeIDs[0] != 22 {
		t.Fatalf("unexpected gRPC SFAC reconfig mapping: %+v", aa)
	}
}

func TestSFACGRPCPolicyFallbackOnNoServer(t *testing.T) {
	// Test graceful fallback when server is not running
	p, err := NewSFACGRPCPolicy("127.0.0.1:59999", 100)
	if err != nil {
		t.Skipf("could not create gRPC policy: %v", err)
	}
	defer p.Close()

	obs := Observation{
		ThroughputTPS: 1000,
		LatencyP95Ms:  50,
	}
	action := p.Decide(obs)
	// Should return fallback action (not panic)
	if action.Reason == "" {
		t.Error("expected non-empty fallback reason")
	}
}
