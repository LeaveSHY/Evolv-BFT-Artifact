// Copyright 2024 Evolv-BFT Project
// Licensed under Apache License 2.0

package adaptive

import (
	"context"
	"fmt"
	"time"

	pb "evolvbft/evolvbft/adaptive/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// SFACGRPCPolicy wraps the Python SFAC trust manager as a gRPC policy.
// It replaces the HTTP bridge with a binary gRPC channel for lower latency.
type SFACGRPCPolicy struct {
	addr    string
	conn    *grpc.ClientConn
	client  pb.SFACServiceClient
	timeout time.Duration
}

// NewSFACGRPCPolicy creates a gRPC-backed SFAC policy.
// addr should be "host:port" (e.g., "127.0.0.1:50051").
func NewSFACGRPCPolicy(addr string, timeoutMs int) (*SFACGRPCPolicy, error) {
	if timeoutMs <= 0 {
		timeoutMs = 2000
	}
	timeout := time.Duration(timeoutMs) * time.Millisecond

	conn, err := grpc.NewClient(
		addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.CallContentSubtype("json")),
	)
	if err != nil {
		return nil, fmt.Errorf("sfac grpc dial %s: %w", addr, err)
	}

	return &SFACGRPCPolicy{
		addr:    addr,
		conn:    conn,
		client:  pb.NewSFACServiceClient(conn),
		timeout: timeout,
	}, nil
}

func (p *SFACGRPCPolicy) Name() string {
	return "sfac-grpc"
}

// Decide converts Observation → SFACRequest, calls gRPC, returns Action.
func (p *SFACGRPCPolicy) Decide(observation Observation) Action {
	fallback := observationFallbackAction(observation, "sfac-grpc-fallback")

	ctx, cancel := context.WithTimeout(context.Background(), p.timeout)
	defer cancel()

	req := p.buildRequest(observation)
	resp, err := p.client.Decide(ctx, req)
	if err != nil {
		fallback.Reason = fmt.Sprintf("sfac-grpc-error: %v", err)
		return fallback
	}

	return p.convertResponse(resp, observation)
}

// Feedback sends reward data for online training via gRPC.
func (p *SFACGRPCPolicy) Feedback(sample TrajectorySample) error {
	ctx, cancel := context.WithTimeout(context.Background(), p.timeout)
	defer cancel()

	pbSample := &pb.TrajectorySample{
		Reward:             sample.Reward,
		PerInstanceRewards: []float64{sample.TeamReward},
		RoleRewards:        sample.RoleRewards,
		Done:               false,
	}

	ack, err := p.client.Feedback(ctx, pbSample)
	if err != nil {
		return err
	}
	if !ack.Success {
		return fmt.Errorf("sfac feedback: %s", ack.Error)
	}
	return nil
}

// Close tears down the gRPC connection.
func (p *SFACGRPCPolicy) Close() error {
	if p.conn != nil {
		return p.conn.Close()
	}
	return nil
}

func (p *SFACGRPCPolicy) buildRequest(obs Observation) *pb.SFACRequest {
	req := &pb.SFACRequest{
		Epoch:        obs.Epoch,
		NumInstances: int32(len(obs.Agents)),
		GlobalState:  []float64{obs.ThroughputTPS, obs.LatencyP95Ms, obs.AdversaryScore, obs.ChurnRate},
	}
	for _, agent := range obs.Agents {
		inst := &pb.SFACInstanceRequest{
			InstanceID:     agent.InstanceID,
			ValidatorCount: int32(agent.ValidatorCount),
			FaultCount:     int32((agent.ValidatorCount - 1) / 3),
			Throughput:     obs.ThroughputTPS,
			Latency:        float64(obs.LatencyP95Ms),
			ViewChanges:    int32(estimateSFACViewChanges(obs.TrustSnapshots)),
		}
		for _, ts := range obs.TrustSnapshots {
			feat := &pb.SFACTrustFeature{
				AgentID:          ts.NodeID,
				TimeoutRate:      ts.TimeoutRate,
				EquivocationRate: ts.EquivocationRate,
				ViewChangeRate:   ts.ViewChangeRate,
				MeanLatency:      ts.MeanLatency,
				StdLatency:       ts.StdLatency,
			}
			inst.TrustFeatures = append(inst.TrustFeatures, feat)
		}
		req.Instances = append(req.Instances, inst)
	}
	return req
}

func (p *SFACGRPCPolicy) convertResponse(resp *pb.SFACResponse, obs Observation) Action {
	action := observationFallbackAction(obs, "sfac-grpc")
	action.Reason = "sfac-grpc-policy"

	for _, agentAction := range resp.Actions {
		aa := sfacActionFromTuple(agentAction.InstanceID, int32SliceToInts(agentAction.Reconfig), agentAction.Rotate, agentAction.Params, obs)
		action.AgentActions = append(action.AgentActions, aa)
	}
	return action
}

func int32SliceToInts(values []int32) []int {
	if len(values) == 0 {
		return nil
	}
	out := make([]int, 0, len(values))
	for _, value := range values {
		out = append(out, int(value))
	}
	return out
}
