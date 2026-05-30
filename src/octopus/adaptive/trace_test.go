package adaptive

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestSchemaSnapshotExposesAdaptiveV1(t *testing.T) {
	snapshot := SchemaSnapshot()
	if snapshot["schema_version"] != SchemaVersion {
		t.Fatalf("unexpected schema version: %+v", snapshot)
	}
	traceFields, ok := snapshot["trace_fields"].([]string)
	if !ok || len(traceFields) == 0 {
		t.Fatalf("expected trace fields in schema snapshot, got %+v", snapshot)
	}
	stageFields, ok := snapshot["decision_action_stage_fields"].([]string)
	if !ok || len(stageFields) == 0 {
		t.Fatalf("expected decision action stage fields in schema snapshot, got %+v", snapshot)
	}
	trustSnapshotFields, ok := snapshot["trust_snapshot_fields"].([]string)
	if !ok || len(trustSnapshotFields) == 0 {
		t.Fatalf("expected trust snapshot fields in schema snapshot, got %+v", snapshot)
	}
	provenanceFields, ok := snapshot["trace_provenance_fields"].([]string)
	if !ok || len(provenanceFields) == 0 {
		t.Fatalf("expected trace provenance fields in schema snapshot, got %+v", snapshot)
	}

	requiredTrustSnapshot := map[string]struct{}{
		"node_id":             {},
		"sample_count":        {},
		"success_rate":        {},
		"failure_probability": {},
		"claim_boundary":      {},
	}
	for _, field := range trustSnapshotFields {
		delete(requiredTrustSnapshot, field)
	}
	if len(requiredTrustSnapshot) != 0 {
		t.Fatalf("missing trust snapshot fields from schema snapshot: %+v", requiredTrustSnapshot)
	}

	required := map[string]struct{}{
		"policy_name":    {},
		"policy_mode":    {},
		"schema_version": {},
		"truth_level":    {},
		"claim_boundary": {},
	}
	for _, field := range provenanceFields {
		delete(required, field)
	}
	if len(required) != 0 {
		t.Fatalf("missing provenance fields from schema snapshot: %+v", required)
	}
}

func TestJSONLTraceWriterAppendsTrajectorySample(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trace.jsonl")
	writer, err := NewJSONLTraceWriter(path)
	if err != nil {
		t.Fatalf("new trace writer: %v", err)
	}
	defer writer.Close()

	sample := TrajectorySample{
		Timestamp: time.Now(),
		Observation: Observation{
			NodeID:                     1,
			ValidatorCount:             4,
			GlobalConfirmedTotal:       8,
			GlobalConfirmedNil:         2,
			LastOrderedRank:            11,
			LastOrderedHeight:          5,
			LastOrderedTransitionCount: 1,
			LastReconfigEpoch:          3,
		},
		Candidate: DecisionActionStage{
			Action: Action{
				PacemakerTimeoutMs: 120,
			},
			Present: true,
			Reason:  "candidate",
		},
		Governed: DecisionActionStage{
			Action: Action{
				PacemakerTimeoutMs: 250,
			},
			Present:       true,
			Mutated:       true,
			Reason:        "governed",
			BlockedFields: []string{"submit_join"},
			Notes:         []string{"membership-frozen"},
		},
		Masked: DecisionActionStage{
			Action: Action{
				PacemakerTimeoutMs: 500,
			},
			Present: true,
			Mutated: true,
			Reason:  "masked",
		},
		Applied: DecisionActionStage{
			Action: Action{
				PacemakerTimeoutMs: 1500,
			},
			Present: true,
			Mutated: true,
			Reason:  "applied",
		},
		GovernanceDelta: true,
		GuardrailDelta:  true,
		Reward:          1.25,
		TeamReward:      1.25,
		RoleRewards: map[string]float64{
			"lane_tuner": 0.75,
		},
		SchemaVersion: SchemaVersion,
		PolicyName:    "safe-baseline",
		Provenance: TraceProvenance{
			PolicyName:    "safe-baseline",
			PolicyMode:    "safe-baseline",
			SchemaVersion: SchemaVersion,
			TruthLevel:    TraceTruthLevel,
			ClaimBoundary: TraceClaimBoundary,
		},
	}
	if err := writer.Write(sample); err != nil {
		t.Fatalf("write sample: %v", err)
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read trace file: %v", err)
	}
	var got TrajectorySample
	if err := json.Unmarshal(raw[:len(raw)-1], &got); err != nil {
		t.Fatalf("decode trace sample: %v", err)
	}
	if got.PolicyName != "safe-baseline" || got.Reward != 1.25 || got.TeamReward != 1.25 || got.SchemaVersion != SchemaVersion {
		t.Fatalf("unexpected trace sample: %+v", got)
	}
	if got.Observation.GlobalConfirmedTotal != 8 || got.Observation.LastOrderedRank != 11 || got.Observation.LastReconfigEpoch != 3 {
		t.Fatalf("expected ordered/reconfig observation metadata to round-trip, got %+v", got.Observation)
	}
	if !got.GovernanceDelta {
		t.Fatalf("expected governance delta in manual trace sample, got %+v", got)
	}
	if got.GovernanceDelta != true || got.Governed.Reason != "governed" || got.Masked.Reason != "masked" {
		t.Fatalf("expected governed/masked metadata to round-trip, got %+v", got)
	}
	if len(got.Governed.BlockedFields) != 1 || got.Governed.BlockedFields[0] != "submit_join" {
		t.Fatalf("expected blocked fields to round-trip, got %+v", got.Governed)
	}
	if len(got.Governed.Notes) != 1 || got.Governed.Notes[0] != "membership-frozen" {
		t.Fatalf("expected governance notes to round-trip, got %+v", got.Governed)
	}
	if !got.GuardrailDelta || got.Candidate.Action.PacemakerTimeoutMs != 120 || got.Applied.Action.PacemakerTimeoutMs != 1500 {
		t.Fatalf("expected candidate/applied action metadata to round-trip, got %+v", got)
	}
	if got.RoleRewards["lane_tuner"] != 0.75 {
		t.Fatalf("expected role reward to round-trip, got %+v", got.RoleRewards)
	}
	if got.Provenance.PolicyName != "safe-baseline" || got.Provenance.PolicyMode != "safe-baseline" {
		t.Fatalf("expected trace provenance policy metadata to round-trip, got %+v", got.Provenance)
	}
	if got.Provenance.SchemaVersion != SchemaVersion || got.Provenance.TruthLevel != TraceTruthLevel {
		t.Fatalf("expected trace provenance schema/truth metadata to round-trip, got %+v", got.Provenance)
	}
	if got.Provenance.ClaimBoundary != TraceClaimBoundary {
		t.Fatalf("expected trace provenance claim boundary to round-trip, got %+v", got.Provenance)
	}
}
