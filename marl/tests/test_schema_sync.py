"""Schema sync tests: verify Python dataclass fields match Go struct JSON fields.

This is the Python counterpart of Go's schema_sync_test.go. If a field is added
to Go types.go but not to Python schemas.py (or vice versa), these tests fail.

Run with: pytest marl/tests/test_schema_sync.py -v
"""
from __future__ import annotations

import json
from dataclasses import fields as dc_fields

import pytest

from marl.schemas import (
    Action,
    AgentAction,
    AgentObservation,
    Observation,
    TrustSnapshot,
)


def _field_names(cls) -> set[str]:
    """Extract field names from a dataclass."""
    return {f.name for f in dc_fields(cls)}


# ─── Expected Go JSON field names (from Go types.go) ────────────────────────
# Keep in sync with Go's schema_sync_test.go expectedPython* lists.

GO_OBSERVATION_FIELDS = {
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
}

GO_ACTION_FIELDS = {
    "committee_size",
    "pacemaker_timeout_ms",
    "mempool_max_batch_txs",
    "mempool_proposal_interval_ms",
    "submit_join",
    "submit_leave",
    "hydra_discovery_target",
    "reason",
    "agent_actions",
}

GO_AGENT_OBSERVATION_FIELDS = {
    "instance_id",
    "epoch",
    "validator_count",
    "committee_size",
    "pacemaker_timeout_ms",
    "mempool_max_batch_txs",
    "mempool_proposal_interval_ms",
}

GO_AGENT_ACTION_FIELDS = {
    "instance_id",
    "committee_size",
    "pacemaker_timeout_ms",
    "mempool_max_batch_txs",
    "mempool_proposal_interval_ms",
    "reconfig",
    "reconfig_evict_node_ids",
    "reconfig_admit_node_ids",
    "rotate_leader",
    "param_vector",
}

GO_TRUST_SNAPSHOT_FIELDS = {
    "node_id",
    "sample_count",
    "success_rate",
    "failure_probability",
    "claim_boundary",
    "timeout_rate",
    "equivocation_rate",
    "view_change_rate",
    "mean_latency",
    "std_latency",
}


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestSchemaSync:
    """Verify Python dataclass fields match Go struct JSON fields."""

    def test_observation_fields_match_go(self):
        py_fields = _field_names(Observation)
        missing_in_python = GO_OBSERVATION_FIELDS - py_fields
        extra_in_python = py_fields - GO_OBSERVATION_FIELDS
        assert not missing_in_python, f"Fields in Go but missing in Python Observation: {missing_in_python}"
        assert not extra_in_python, f"Fields in Python but missing in Go Observation: {extra_in_python}"

    def test_action_fields_match_go(self):
        py_fields = _field_names(Action)
        missing_in_python = GO_ACTION_FIELDS - py_fields
        extra_in_python = py_fields - GO_ACTION_FIELDS
        assert not missing_in_python, f"Fields in Go but missing in Python Action: {missing_in_python}"
        assert not extra_in_python, f"Fields in Python but missing in Go Action: {extra_in_python}"

    def test_agent_observation_fields_match_go(self):
        py_fields = _field_names(AgentObservation)
        missing_in_python = GO_AGENT_OBSERVATION_FIELDS - py_fields
        extra_in_python = py_fields - GO_AGENT_OBSERVATION_FIELDS
        assert not missing_in_python, f"Fields in Go but missing in Python AgentObservation: {missing_in_python}"
        assert not extra_in_python, f"Fields in Python but missing in Go AgentObservation: {extra_in_python}"

    def test_agent_action_fields_match_go(self):
        py_fields = _field_names(AgentAction)
        missing_in_python = GO_AGENT_ACTION_FIELDS - py_fields
        extra_in_python = py_fields - GO_AGENT_ACTION_FIELDS
        assert not missing_in_python, f"Fields in Go but missing in Python AgentAction: {missing_in_python}"
        assert not extra_in_python, f"Fields in Python but missing in Go AgentAction: {extra_in_python}"

    def test_trust_snapshot_fields_match_go(self):
        py_fields = _field_names(TrustSnapshot)
        missing_in_python = GO_TRUST_SNAPSHOT_FIELDS - py_fields
        extra_in_python = py_fields - GO_TRUST_SNAPSHOT_FIELDS
        assert not missing_in_python, f"Fields in Go but missing in Python TrustSnapshot: {missing_in_python}"
        assert not extra_in_python, f"Fields in Python but missing in Go TrustSnapshot: {extra_in_python}"


class TestSchemaRoundtrip:
    """Verify JSON round-trip between Go format and Python dataclasses."""

    def test_observation_roundtrip(self):
        """Simulate a Go-serialized Observation and parse it in Python."""
        go_json = {
            "timestamp": "2025-01-01T00:00:00Z",
            "node_id": 42,
            "epoch": 10,
            "validator_count": 8,
            "current_config_id": 5,
            "highest_known_config_id": 6,
            "committee_size": 6,
            "pacemaker_timeout_ms": 500,
            "mempool_max_batch_txs": 2048,
            "mempool_proposal_interval_ms": 50,
            "throughput_tps": 1000.5,
            "latency_p50_ms": 20.0,
            "latency_p95_ms": 45.2,
            "latency_p99_ms": 98.1,
            "recovery_p95_ms": 150.0,
            "backlog_pending": 10,
            "backlog_missing": 2,
            "reject_total": 0,
            "connected_peers": 7,
            "known_peers": 8,
            "pending_joins": 0,
            "pending_leaves": 0,
            "lset_size": 8,
            "can_participate": True,
            "local_validator": True,
            "global_confirmed_total": 500,
            "global_confirmed_nil": 3,
            "last_ordered_rank": 499,
            "last_ordered_height": 100,
            "last_ordered_lane_id": 2,
            "last_ordered_config_id": 5,
            "last_ordered_nil": False,
            "last_ordered_transition_count": 4,
            "last_reconfig_epoch": 8,
            "heterogeneity_score": 0.3,
            "churn_rate": 0.05,
            "adversary_score": 0.1,
            "network_jitter_ms": 5.0,
            "ai_load_score": 0.2,
            "agents": [
                {
                    "instance_id": 0,
                    "epoch": 10,
                    "validator_count": 8,
                    "committee_size": 6,
                    "pacemaker_timeout_ms": 500,
                    "mempool_max_batch_txs": 2048,
                    "mempool_proposal_interval_ms": 50,
                },
            ],
            "trust_snapshots": [
                {
                    "node_id": 1,
                    "sample_count": 100,
                    "success_rate": 0.95,
                    "failure_probability": 0.05,
                    "claim_boundary": "test",
                    "timeout_rate": 0.02,
                    "equivocation_rate": 0.01,
                    "view_change_rate": 0.005,
                    "mean_latency": 0.15,
                    "std_latency": 0.03,
                },
            ],
        }

        obs = Observation.from_dict(go_json)
        assert obs.node_id == 42
        assert obs.throughput_tps == 1000.5
        assert obs.global_confirmed_total == 500
        assert len(obs.agents) == 1
        assert obs.agents[0].instance_id == 0
        assert len(obs.trust_snapshots) == 1
        assert obs.trust_snapshots[0].timeout_rate == 0.02
        assert obs.trust_snapshots[0].equivocation_rate == 0.01

    def test_action_roundtrip(self):
        """Simulate a Python Action serialized to Go format."""
        action = Action(
            committee_size=6,
            pacemaker_timeout_ms=500,
            mempool_max_batch_txs=2048,
            mempool_proposal_interval_ms=50,
            submit_join=True,
            hydra_discovery_target=3,
            reason="sfac-policy",
            agent_actions=[
                AgentAction(instance_id=0, committee_size=4, pacemaker_timeout_ms=500),
            ],
        )
        d = action.to_dict()
        assert d["committee_size"] == 6
        assert d["submit_join"] is True
        assert d["hydra_discovery_target"] == 3
        assert len(d["agent_actions"]) == 1

        # Verify it parses back
        parsed = Action.from_dict(d)
        assert parsed.committee_size == 6
        assert parsed.submit_join is True

    def test_trust_snapshot_with_features(self):
        """Verify TrustSnapshot round-trips with trust feature vector."""
        data = {
            "node_id": 5,
            "sample_count": 50,
            "success_rate": 0.9,
            "failure_probability": 0.1,
            "claim_boundary": "",
            "timeout_rate": 0.04,
            "equivocation_rate": 0.02,
            "view_change_rate": 0.01,
            "mean_latency": 0.25,
            "std_latency": 0.05,
        }
        ts = TrustSnapshot.from_dict(data)
        assert ts.timeout_rate == 0.04
        assert ts.equivocation_rate == 0.02
        assert ts.mean_latency == 0.25
        assert ts.std_latency == 0.05

    def test_observation_rejects_unknown_field(self):
        """Verify _ensure_known_fields catches drift from the other direction."""
        with pytest.raises(ValueError, match="unknown field"):
            Observation.from_dict({"bogus_field_not_in_go": 123})
