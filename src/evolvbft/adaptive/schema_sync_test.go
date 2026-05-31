package adaptive

import (
	"encoding/json"
	"reflect"
	"sort"
	"testing"
)

// TestSchemaFieldAlignment verifies that the JSON field names in Go structs
// match the expected Python field names. This catches drift between
// Go types.go and Python schemas.py.

func jsonFieldNames(v interface{}) []string {
	t := reflect.TypeOf(v)
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	var fields []string
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		tag := f.Tag.Get("json")
		if tag == "" || tag == "-" {
			continue
		}
		// Strip options like ",omitempty"
		name := tag
		for j := 0; j < len(tag); j++ {
			if tag[j] == ',' {
				name = tag[:j]
				break
			}
		}
		if name != "" && name != "-" {
			fields = append(fields, name)
		}
	}
	sort.Strings(fields)
	return fields
}

// expectedPythonObservationFields lists the fields from Python schemas.py Observation.
// Keep this in sync when adding fields to either side.
var expectedPythonObservationFields = []string{
	"adversary_score",
	"agents",
	"ai_load_score",
	"backlog_missing",
	"backlog_pending",
	"can_participate",
	"churn_rate",
	"committee_size",
	"connected_peers",
	"current_config_id",
	"epoch",
	"global_confirmed_nil",
	"global_confirmed_total",
	"heterogeneity_score",
	"highest_known_config_id",
	"known_peers",
	"last_ordered_config_id",
	"last_ordered_height",
	"last_ordered_lane_id",
	"last_ordered_nil",
	"last_ordered_rank",
	"last_ordered_transition_count",
	"last_reconfig_epoch",
	"latency_p50_ms",
	"latency_p95_ms",
	"latency_p99_ms",
	"local_validator",
	"lset_size",
	"mempool_max_batch_txs",
	"mempool_proposal_interval_ms",
	"network_jitter_ms",
	"node_id",
	"pacemaker_timeout_ms",
	"pending_joins",
	"pending_leaves",
	"recovery_p95_ms",
	"reject_total",
	"throughput_tps",
	"timestamp",
	"trust_snapshots",
	"validator_count",
}

var expectedPythonActionFields = []string{
	"agent_actions",
	"committee_size",
	"hydra_discovery_target",
	"mempool_max_batch_txs",
	"mempool_proposal_interval_ms",
	"pacemaker_timeout_ms",
	"reason",
	"submit_join",
	"submit_leave",
}

var expectedPythonAgentActionFields = []string{
	"committee_size",
	"instance_id",
	"mempool_max_batch_txs",
	"mempool_proposal_interval_ms",
	"pacemaker_timeout_ms",
	"param_vector",
	"reconfig",
	"reconfig_admit_node_ids",
	"reconfig_evict_node_ids",
	"rotate_leader",
}

var expectedPythonTrustSnapshotFields = []string{
	"claim_boundary",
	"equivocation_rate",
	"failure_probability",
	"mean_latency",
	"node_id",
	"sample_count",
	"std_latency",
	"success_rate",
	"timeout_rate",
	"view_change_rate",
}

func TestObservationFieldsMatchPython(t *testing.T) {
	goFields := jsonFieldNames(Observation{})
	sort.Strings(expectedPythonObservationFields)
	if !reflect.DeepEqual(goFields, expectedPythonObservationFields) {
		t.Fatalf("Observation field mismatch between Go and Python\nGo:     %v\nPython: %v",
			goFields, expectedPythonObservationFields)
	}
}

func TestActionFieldsMatchPython(t *testing.T) {
	goFields := jsonFieldNames(Action{})
	sort.Strings(expectedPythonActionFields)
	if !reflect.DeepEqual(goFields, expectedPythonActionFields) {
		t.Fatalf("Action field mismatch between Go and Python\nGo:     %v\nPython: %v",
			goFields, expectedPythonActionFields)
	}
}

func TestAgentActionFieldsMatchPython(t *testing.T) {
	goFields := jsonFieldNames(AgentAction{})
	sort.Strings(expectedPythonAgentActionFields)
	if !reflect.DeepEqual(goFields, expectedPythonAgentActionFields) {
		t.Fatalf("AgentAction field mismatch between Go and Python\nGo:     %v\nPython: %v",
			goFields, expectedPythonAgentActionFields)
	}
}

func TestTrustSnapshotFieldsMatchPython(t *testing.T) {
	goFields := jsonFieldNames(TrustSnapshot{})
	sort.Strings(expectedPythonTrustSnapshotFields)
	if !reflect.DeepEqual(goFields, expectedPythonTrustSnapshotFields) {
		t.Fatalf("TrustSnapshot field mismatch between Go and Python\nGo:     %v\nPython: %v",
			goFields, expectedPythonTrustSnapshotFields)
	}
}

// TestObservationRoundtripJSON verifies JSON serialization round-trips correctly.
func TestObservationRoundtripJSON(t *testing.T) {
	obs := Observation{
		NodeID:         42,
		Epoch:          10,
		ValidatorCount: 8,
		CommitteeSize:  6,
		ThroughputTPS:  1000.5,
		LatencyP95Ms:   45.2,
		Agents: []AgentObservation{
			{InstanceID: 0, ValidatorCount: 8, CommitteeSize: 6},
			{InstanceID: 1, ValidatorCount: 6, CommitteeSize: 4},
		},
		TrustSnapshots: []TrustSnapshot{
			{NodeID: 1, SampleCount: 100, SuccessRate: 0.95, FailureProbability: 0.05,
				TimeoutRate: 0.02, EquivocationRate: 0.01},
		},
	}

	data, err := json.Marshal(obs)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}

	var decoded Observation
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}

	if decoded.NodeID != obs.NodeID || decoded.ThroughputTPS != obs.ThroughputTPS {
		t.Fatalf("round-trip mismatch: %+v vs %+v", obs, decoded)
	}
	if len(decoded.Agents) != 2 || decoded.Agents[1].InstanceID != 1 {
		t.Fatalf("agents round-trip mismatch")
	}
	if len(decoded.TrustSnapshots) != 1 || decoded.TrustSnapshots[0].TimeoutRate != 0.02 {
		t.Fatalf("trust snapshots round-trip mismatch")
	}
}

// TestActionRoundtripJSON verifies Action JSON serialization.
func TestActionRoundtripJSON(t *testing.T) {
	action := Action{
		CommitteeSize:        6,
		PacemakerTimeoutMs:   500,
		MempoolMaxBatchTxs:   2048,
		SubmitJoin:           true,
		HydraDiscoveryTarget: 3,
		Reason:               "sfac-policy",
		AgentActions: []AgentAction{
			{InstanceID: 0, CommitteeSize: 4, PacemakerTimeoutMs: 500, Reconfig: []int{-1, 0, 1}, ReconfigEvictNodeIDs: []uint64{7}, ReconfigAdmitNodeIDs: []uint64{9}, RotateLeader: true, ParamVector: []float64{0.5}},
		},
	}

	data, err := json.Marshal(action)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}

	var decoded Action
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}

	if decoded.CommitteeSize != action.CommitteeSize || decoded.Reason != action.Reason {
		t.Fatalf("round-trip mismatch: %+v vs %+v", action, decoded)
	}
	if !decoded.SubmitJoin || decoded.HydraDiscoveryTarget != 3 {
		t.Fatalf("boolean/int round-trip mismatch")
	}
	if len(decoded.AgentActions) != 1 || !decoded.AgentActions[0].RotateLeader || len(decoded.AgentActions[0].ReconfigEvictNodeIDs) != 1 {
		t.Fatalf("agent action round-trip mismatch: %+v", decoded.AgentActions)
	}
}
