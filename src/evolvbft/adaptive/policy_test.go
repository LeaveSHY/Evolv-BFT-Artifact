package adaptive

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestSafeBaselinePolicyBacksOffOnHighLatency(t *testing.T) {
	policy := SafeBaselinePolicy{}
	obs := Observation{
		ValidatorCount:            32,
		LocalValidator:            true,
		CanParticipate:            true,
		PacemakerTimeoutMs:        1000,
		MempoolMaxBatchTxs:        2048,
		MempoolProposalIntervalMs: 100,
		LatencyP95Ms:              800,
		BacklogPending:            500,
		RejectTotal:               20,
	}

	got := policy.Decide(obs)

	if got.PacemakerTimeoutMs <= obs.PacemakerTimeoutMs {
		t.Fatalf("expected timeout increase, got %d", got.PacemakerTimeoutMs)
	}
	if got.MempoolMaxBatchTxs >= obs.MempoolMaxBatchTxs {
		t.Fatalf("expected batch reduction, got %d", got.MempoolMaxBatchTxs)
	}
}

func TestSafeBaselinePolicyRequestsJoinForNonValidator(t *testing.T) {
	policy := SafeBaselinePolicy{}
	obs := Observation{
		CurrentConfigID:      1,
		HighestKnownConfigID: 2,
		LocalValidator:       false,
		CanParticipate:       true,
	}

	got := policy.Decide(obs)
	if !got.SubmitJoin || got.HydraDiscoveryTarget != 3 {
		t.Fatalf("expected membership-aware join request, got %+v", got)
	}
}

func TestScriptedPolicyLoadsActionFromJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "policy.json")
	if err := os.WriteFile(path, []byte(`{
  "committee_size": 7,
  "pacemaker_timeout_ms": 1400,
  "mempool_max_batch_txs": 512,
  "mempool_proposal_interval_ms": 80
}`), 0o600); err != nil {
		t.Fatalf("write policy file: %v", err)
	}

	policy := NewScriptedPolicy(path)
	got := policy.Decide(Observation{ValidatorCount: 32})

	if got.CommitteeSize != 7 || got.PacemakerTimeoutMs != 1400 || got.MempoolMaxBatchTxs != 512 || got.MempoolProposalIntervalMs != 80 {
		t.Fatalf("unexpected scripted action: %+v", got)
	}
}

func TestHTTPPolicyPostsObservationAndDecodesAction(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("unexpected method: %s", r.Method)
		}
		var obs Observation
		if err := json.NewDecoder(r.Body).Decode(&obs); err != nil {
			t.Fatalf("decode observation: %v", err)
		}
		if obs.ValidatorCount != 12 {
			t.Fatalf("unexpected observation validator count: %d", obs.ValidatorCount)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(Action{
			CommitteeSize:             8,
			PacemakerTimeoutMs:        1600,
			MempoolMaxBatchTxs:        768,
			MempoolProposalIntervalMs: 90,
			Reason:                    "facmac-http",
		})
	}))
	defer server.Close()

	policy := NewHTTPPolicy(server.URL, 2_000)
	got := policy.Decide(Observation{ValidatorCount: 12})
	if got.CommitteeSize != 8 || got.PacemakerTimeoutMs != 1600 || got.MempoolMaxBatchTxs != 768 || got.MempoolProposalIntervalMs != 90 {
		t.Fatalf("unexpected http action: %+v", got)
	}
}

func TestHTTPPolicyPostsFullStageSchemaObservation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("unexpected method: %s", r.Method)
		}
		var raw map[string]any
		if err := json.NewDecoder(r.Body).Decode(&raw); err != nil {
			t.Fatalf("decode observation: %v", err)
		}
		for _, field := range []string{
			"timestamp",
			"node_id",
			"epoch",
			"validator_count",
			"current_config_id",
			"highest_known_config_id",
			"committee_size",
			"pacemaker_timeout_ms",
			"mempool_max_batch_txs",
			"mempool_proposal_interval_ms",
			"throughput_tps",
			"latency_p50_ms",
			"latency_p95_ms",
			"latency_p99_ms",
			"recovery_p95_ms",
			"backlog_pending",
			"backlog_missing",
			"reject_total",
			"connected_peers",
			"known_peers",
			"pending_joins",
			"pending_leaves",
			"lset_size",
			"can_participate",
			"local_validator",
			"global_confirmed_total",
			"global_confirmed_nil",
			"last_ordered_rank",
			"last_ordered_height",
			"last_ordered_lane_id",
			"last_ordered_config_id",
			"last_ordered_nil",
			"last_ordered_transition_count",
			"last_reconfig_epoch",
			"heterogeneity_score",
			"churn_rate",
			"adversary_score",
			"network_jitter_ms",
			"ai_load_score",
			"agents",
			"trust_snapshots",
		} {
			if _, ok := raw[field]; !ok {
				t.Fatalf("expected field %q in HTTP payload, got %v", field, raw)
			}
		}
		agents, ok := raw["agents"].([]any)
		if !ok || len(agents) != 1 {
			t.Fatalf("expected one agent observation, got %#v", raw["agents"])
		}
		agent, ok := agents[0].(map[string]any)
		if !ok {
			t.Fatalf("expected agent observation object, got %#v", agents[0])
		}
		if agent["instance_id"] != float64(1) || agent["validator_count"] != float64(12) || agent["pacemaker_timeout_ms"] != float64(1200) {
			t.Fatalf("unexpected agent observation payload: %#v", agent)
		}
		trustSnapshots, ok := raw["trust_snapshots"].([]any)
		if !ok || len(trustSnapshots) != 1 {
			t.Fatalf("expected one trust snapshot, got %#v", raw["trust_snapshots"])
		}
		trustSnapshot, ok := trustSnapshots[0].(map[string]any)
		if !ok {
			t.Fatalf("expected trust snapshot object, got %#v", trustSnapshots[0])
		}
		if trustSnapshot["node_id"] != float64(9) || trustSnapshot["sample_count"] != float64(10) {
			t.Fatalf("unexpected trust snapshot payload: %#v", trustSnapshot)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(Action{Reason: "facmac-http"})
	}))
	defer server.Close()

	policy := NewHTTPPolicy(server.URL, 2_000)
	got := policy.Decide(Observation{
		Timestamp:                 time.Unix(1_700_000_000, 0).UTC(),
		NodeID:                    7,
		Epoch:                     3,
		ValidatorCount:            12,
		CurrentConfigID:           3,
		HighestKnownConfigID:      4,
		CommitteeSize:             5,
		PacemakerTimeoutMs:        1200,
		MempoolMaxBatchTxs:        1024,
		MempoolProposalIntervalMs: 80,
		ThroughputTPS:             4321.5,
		LatencyP50Ms:              90,
		LatencyP95Ms:              140,
		LatencyP99Ms:              180,
		RecoveryP95Ms:             150,
		BacklogPending:            12,
		BacklogMissing:            1,
		RejectTotal:               2,
		ConnectedPeers:            6,
		KnownPeers:                8,
		PendingJoins:              1,
		PendingLeaves:             0,
		LSetSize:                  10,
		CanParticipate:            true,
		LocalValidator:            true,
		GlobalConfirmedTotal:      99,
		GlobalConfirmedNil:        3,
		LastOrderedRank:           120,
		LastOrderedHeight:         55,
		LastOrderedNil:            false,
		LastOrderedTransitionCount: 2,
		LastReconfigEpoch:         4,
		HeterogeneityScore:        0.6,
		ChurnRate:                 0.2,
		AdversaryScore:            0.1,
		NetworkJitterMs:           25,
		AILoadScore:               0.7,
		Agents: []AgentObservation{{
			InstanceID:                1,
			Epoch:                     3,
			ValidatorCount:            12,
			CommitteeSize:             5,
			PacemakerTimeoutMs:        1200,
			MempoolMaxBatchTxs:        1024,
			MempoolProposalIntervalMs: 80,
		}},
		TrustSnapshots: []TrustSnapshot{{
			NodeID:             9,
			SampleCount:        10,
			SuccessRate:        0.8,
			FailureProbability: 0.2,
		}},
	})
	if got.Reason != "facmac-http" {
		t.Fatalf("unexpected http action: %+v", got)
	}
}

func TestHTTPPolicyRejectsNonOKStatus(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "upstream unavailable", http.StatusBadGateway)
	}))
	defer server.Close()

	obs := Observation{
		CommitteeSize:             4,
		PacemakerTimeoutMs:        1000,
		MempoolMaxBatchTxs:        256,
		MempoolProposalIntervalMs: 50,
	}
	policy := NewHTTPPolicy(server.URL, 2_000)
	got := policy.Decide(obs)

	if got.CommitteeSize != obs.CommitteeSize || got.PacemakerTimeoutMs != obs.PacemakerTimeoutMs || got.MempoolMaxBatchTxs != obs.MempoolMaxBatchTxs || got.MempoolProposalIntervalMs != obs.MempoolProposalIntervalMs {
		t.Fatalf("expected non-2xx response to fall back to observation-derived action, got %+v", got)
	}
	if got.Reason != "http-status-error" {
		t.Fatalf("expected http-status-error reason, got %q", got.Reason)
	}
}

func TestHTTPPolicyRejectsNonJSONContentType(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		_, _ = w.Write([]byte(`not-json`))
	}))
	defer server.Close()

	obs := Observation{
		CommitteeSize:             4,
		PacemakerTimeoutMs:        1000,
		MempoolMaxBatchTxs:        256,
		MempoolProposalIntervalMs: 50,
	}
	policy := NewHTTPPolicy(server.URL, 2_000)
	got := policy.Decide(obs)

	if got.Reason != "http-content-type-error" {
		t.Fatalf("expected http-content-type-error reason, got %q", got.Reason)
	}
}

func TestHTTPPolicyRejectsMalformedContentType(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", `application/json; charset==utf-8`)
		_, _ = w.Write([]byte(`{"committee_size":8}`))
	}))
	defer server.Close()

	obs := Observation{
		CommitteeSize:             4,
		PacemakerTimeoutMs:        1000,
		MempoolMaxBatchTxs:        256,
		MempoolProposalIntervalMs: 50,
	}
	policy := NewHTTPPolicy(server.URL, 2_000)
	got := policy.Decide(obs)

	if got.Reason != "http-content-type-error" {
		t.Fatalf("expected malformed content type to be rejected, got %q", got.Reason)
	}
}

func TestHTTPPolicyRejectsTrailingJSONGarbage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"committee_size":8}garbage`))
	}))
	defer server.Close()

	obs := Observation{
		CommitteeSize:             4,
		PacemakerTimeoutMs:        1000,
		MempoolMaxBatchTxs:        256,
		MempoolProposalIntervalMs: 50,
	}
	policy := NewHTTPPolicy(server.URL, 2_000)
	got := policy.Decide(obs)

	if got.Reason != "http-decode-error" {
		t.Fatalf("expected http-decode-error reason, got %q", got.Reason)
	}
}

func TestHTTPPolicyRejectsEmptyBodyAndNoContent(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
	}{
		{name: "empty 200 body", statusCode: http.StatusOK},
		{name: "204 no content", statusCode: http.StatusNoContent},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tc.statusCode)
			}))
			defer server.Close()

			obs := Observation{
				CommitteeSize:             4,
				PacemakerTimeoutMs:        1000,
				MempoolMaxBatchTxs:        256,
				MempoolProposalIntervalMs: 50,
			}
			policy := NewHTTPPolicy(server.URL, 2_000)
			got := policy.Decide(obs)

			expectedReason := "http-decode-error"
			if tc.statusCode == http.StatusNoContent {
				expectedReason = "http-status-error"
			}
			if got.Reason != expectedReason {
				t.Fatalf("expected %s reason, got %q", expectedReason, got.Reason)
			}
		})
	}
}

func TestHTTPPolicyMergesPartialActionResponseWithObservationDefaults(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"reason":"partial-update","submit_join":true}`))
	}))
	defer server.Close()

	obs := Observation{
		CommitteeSize:             6,
		PacemakerTimeoutMs:        1400,
		MempoolMaxBatchTxs:        768,
		MempoolProposalIntervalMs: 90,
		CurrentConfigID:           2,
		HighestKnownConfigID:      3,
		LocalValidator:            false,
		CanParticipate:            true,
	}
	policy := NewHTTPPolicy(server.URL, 2_000)
	got := policy.Decide(obs)

	if got.CommitteeSize != obs.CommitteeSize || got.PacemakerTimeoutMs != obs.PacemakerTimeoutMs || got.MempoolMaxBatchTxs != obs.MempoolMaxBatchTxs || got.MempoolProposalIntervalMs != obs.MempoolProposalIntervalMs {
		t.Fatalf("expected partial response to inherit observation defaults, got %+v", got)
	}
	if !got.SubmitJoin {
		t.Fatalf("expected explicit boolean override to survive merge, got %+v", got)
	}
	if got.Reason != "partial-update" {
		t.Fatalf("expected explicit reason override, got %+v", got)
	}
}

func TestHTTPPolicyMergesPartialAgentActionsWithoutZeroingTopLevelFields(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"agent_actions":[{"instance_id":1,"pacemaker_timeout_ms":1250}],"reason":"agent-only"}`))
	}))
	defer server.Close()

	obs := Observation{
		CommitteeSize:             6,
		PacemakerTimeoutMs:        1400,
		MempoolMaxBatchTxs:        768,
		MempoolProposalIntervalMs: 90,
	}
	policy := NewHTTPPolicy(server.URL, 2_000)
	got := policy.Decide(obs)

	if got.CommitteeSize != obs.CommitteeSize || got.PacemakerTimeoutMs != obs.PacemakerTimeoutMs || got.MempoolMaxBatchTxs != obs.MempoolMaxBatchTxs || got.MempoolProposalIntervalMs != obs.MempoolProposalIntervalMs {
		t.Fatalf("expected agent-only response to preserve top-level defaults, got %+v", got)
	}
	if len(got.AgentActions) != 1 || got.AgentActions[0].InstanceID != 1 || got.AgentActions[0].PacemakerTimeoutMs != 1250 {
		t.Fatalf("expected merged agent action payload, got %+v", got)
	}
	if got.Reason != "agent-only" {
		t.Fatalf("expected explicit reason override, got %+v", got)
	}
}

func TestPolicyByNamePreservesFacmacHTTPAliasIdentity(t *testing.T) {
	policy := PolicyByName("facmac-http", "", "http://127.0.0.1:0")
	if policy == nil {
		t.Fatalf("expected facmac-http policy")
	}
	if policy.Name() != "facmac-http" {
		t.Fatalf("expected facmac-http alias to be preserved, got %q", policy.Name())
	}
}
