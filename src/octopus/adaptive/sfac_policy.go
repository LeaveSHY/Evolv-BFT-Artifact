// Copyright 2024 Octopus Project
// Licensed under Apache License 2.0

package adaptive

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// SFACPolicy wraps the Python SFAC trust manager as an HTTP policy.
// It sends trust feature vectors and instance observations, and receives
// per-agent reconfiguration actions from the trained MARL policy.
//
// Protocol:
//
//	POST /sfac/decide
//	Request:  SFACRequest  (trust features + observations per instance)
//	Response: SFACResponse (per-agent actions: reconfig, rotate, param)
//
// The Python server wraps the trained SFAC PPO model from experiments/sfac_ppo.py.
type SFACPolicy struct {
	url       string
	client    *http.Client
	resilient *ResilientClient
}

// SFACRequest is sent to the Python SFAC server.
type SFACRequest struct {
	Epoch        uint64                `json:"epoch"`
	NumInstances int                   `json:"num_instances"`
	Instances    []SFACInstanceRequest `json:"instances"`
	GlobalState  []float64             `json:"global_state,omitempty"` // centralized critic state
}

// SFACInstanceRequest contains per-instance observation for SFAC.
type SFACInstanceRequest struct {
	InstanceID     uint64             `json:"instance_id"`
	ValidatorCount int                `json:"validator_count"`
	FaultCount     int                `json:"fault_count"`
	Throughput     float64            `json:"throughput"`
	Latency        float64            `json:"latency"`
	ViewChanges    int                `json:"view_changes"`
	TrustFeatures  []SFACTrustFeature `json:"trust_features"` // per-agent features (Eq. 5)
}

// SFACTrustFeature is the 5-dim trust feature vector (Eq. 5) for one agent.
type SFACTrustFeature struct {
	AgentID          uint64  `json:"agent_id"`
	TimeoutRate      float64 `json:"timeout_rate"`      // d_t^k / W
	EquivocationRate float64 `json:"equivocation_rate"` // e_t^k / W
	ViewChangeRate   float64 `json:"view_change_rate"`  // v_t^k / W
	MeanLatency      float64 `json:"mean_latency"`      // τ̄_t^k
	StdLatency       float64 `json:"std_latency"`       // σ_τ,t^k
}

// SFACResponse is returned by the Python SFAC server.
type SFACResponse struct {
	Actions []SFACAgentAction `json:"actions"`
	Value   float64           `json:"value,omitempty"` // critic value estimate
}

// SFACAgentAction is the SFAC output for one agent/instance.
type SFACAgentAction struct {
	InstanceID uint64    `json:"instance_id"`
	Reconfig   []int     `json:"reconfig"` // per-validator: -1=evict, 0=retain, +1=join
	Rotate     bool      `json:"rotate"`   // trigger leader rotation
	Params     []float64 `json:"params"`   // consensus parameter adjustments
}

// NewSFACPolicy creates a policy that delegates to the Python SFAC server.
func NewSFACPolicy(url string, timeoutMs int) *SFACPolicy {
	if timeoutMs <= 0 {
		timeoutMs = 2000 // SFAC inference is fast (<2ms) but allow network overhead
	}
	timeout := time.Duration(timeoutMs) * time.Millisecond
	return &SFACPolicy{
		url:       url,
		client:    &http.Client{Timeout: timeout},
		resilient: DefaultResilientClient(timeout),
	}
}

func (p *SFACPolicy) Name() string {
	return "sfac"
}

// Decide converts the adaptive controller Observation into an SFAC request,
// sends it to the Python server, and converts the response back to an Action.
func (p *SFACPolicy) Decide(observation Observation) Action {
	fallback := observationFallbackAction(observation, "sfac-fallback")

	req := p.buildSFACRequest(observation)
	body, err := json.Marshal(req)
	if err != nil {
		fallback.Reason = "sfac-marshal-error"
		return fallback
	}

	httpReq, err := http.NewRequest(http.MethodPost, p.url+"/sfac/decide", bytes.NewReader(body))
	if err != nil {
		fallback.Reason = "sfac-request-error"
		return fallback
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.GetBody = func() (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(body)), nil
	}

	resp, err := p.resilient.Do(httpReq)
	if err != nil {
		if _, ok := err.(*CircuitOpenError); ok {
			fallback.Reason = "sfac-circuit-open"
		} else {
			fallback.Reason = "sfac-connection-error"
		}
		return fallback
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		fallback.Reason = fmt.Sprintf("sfac-status-%d", resp.StatusCode)
		return fallback
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		fallback.Reason = "sfac-read-error"
		return fallback
	}

	var sfacResp SFACResponse
	if err := json.Unmarshal(respBody, &sfacResp); err != nil {
		fallback.Reason = "sfac-decode-error"
		return fallback
	}

	return p.convertSFACResponse(sfacResp, observation)
}

func (p *SFACPolicy) buildSFACRequest(obs Observation) SFACRequest {
	req := SFACRequest{
		Epoch:        obs.Epoch,
		NumInstances: len(obs.Agents),
		GlobalState:  []float64{obs.ThroughputTPS, obs.LatencyP95Ms, obs.AdversaryScore, obs.ChurnRate},
	}
	for _, agent := range obs.Agents {
		inst := SFACInstanceRequest{
			InstanceID:     agent.InstanceID,
			ValidatorCount: agent.ValidatorCount,
			FaultCount:     agent.FaultsEstimate,
			Throughput:     obs.ThroughputTPS,
			Latency:        float64(obs.LatencyP95Ms),
			ViewChanges:    estimateSFACViewChanges(obs.TrustSnapshots),
		}
		// Trust features from TrustSnapshots if available
		for _, ts := range obs.TrustSnapshots {
			feat := SFACTrustFeature{
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

func (p *SFACPolicy) convertSFACResponse(resp SFACResponse, obs Observation) Action {
	action := observationFallbackAction(obs, "sfac")
	action.Reason = "sfac-policy"

	for _, agentAction := range resp.Actions {
		aa := sfacActionFromTuple(agentAction.InstanceID, agentAction.Reconfig, agentAction.Rotate, agentAction.Params, obs)
		action.AgentActions = append(action.AgentActions, aa)
	}

	return action
}

func estimateSFACViewChanges(snapshots []TrustSnapshot) int {
	viewChanges := 0
	for _, snapshot := range snapshots {
		if snapshot.ViewChangeRate > 0 {
			viewChanges++
		}
	}
	return viewChanges
}

// sfacFeedbackPayload is the wire-format expected by the Python SFAC server
// at /sfac/feedback. It extracts the minimal fields needed for online training
// from the full TrajectorySample computed in the Go controller.
type sfacFeedbackPayload struct {
	PerInstanceRewards []float64          `json:"per_instance_rewards"`
	RoleRewards        map[string]float64 `json:"role_rewards,omitempty"`
	Done               bool               `json:"done"`
}

// Feedback sends reward data to the Python SFAC server for online training.
// Implements FeedbackPolicy interface.
func (p *SFACPolicy) Feedback(sample TrajectorySample) error {
	if p.url == "" {
		return nil
	}
	// Build the minimal payload matching the Python SFACFeedbackRequest schema.
	// Each Go controller manages one instance, so per_instance_rewards is a
	// single-element list containing this instance's team reward.
	payload := sfacFeedbackPayload{
		PerInstanceRewards: []float64{sample.TeamReward},
		RoleRewards:        sample.RoleRewards,
		Done:               false,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	req, err := http.NewRequest(http.MethodPost, p.url+"/sfac/feedback", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.GetBody = func() (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(body)), nil
	}
	resp, err := p.resilient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return fmt.Errorf("sfac feedback HTTP %d", resp.StatusCode)
	}
	return nil
}
