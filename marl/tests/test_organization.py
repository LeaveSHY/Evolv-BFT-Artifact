import unittest

from marl.organization import MOISEOrganizationModel
from marl.schemas import Action, AgentObservation, Observation, SCHEMA_VERSION
from marl.service import PolicyService


class OrganizationTests(unittest.TestCase):
    def test_membership_role_masks_non_membership_fields(self):
        model = MOISEOrganizationModel()
        obs = Observation(
            current_config_id=1,
            highest_known_config_id=2,
            validator_count=4,
            committee_size=4,
            pacemaker_timeout_ms=1100,
            mempool_max_batch_txs=512,
            mempool_proposal_interval_ms=90,
            local_validator=False,
            can_participate=True,
            pending_joins=0,
        )
        raw = Action(
            committee_size=8,
            pacemaker_timeout_ms=1500,
            mempool_max_batch_txs=1024,
            mempool_proposal_interval_ms=80,
            submit_join=True,
            hydra_discovery_target=3,
        )
        action = model.sanitize(obs, raw)
        self.assertEqual(action.committee_size, 4)
        self.assertEqual(action.pacemaker_timeout_ms, 1100)
        self.assertEqual(action.mempool_max_batch_txs, 512)
        self.assertEqual(action.mempool_proposal_interval_ms, 90)
        self.assertTrue(action.submit_join)
        self.assertEqual(action.hydra_discovery_target, 3)

    def test_service_exposes_organization_snapshot(self):
        service = PolicyService()
        snapshot = service.organization_snapshot()
        self.assertIn("roles", snapshot)
        self.assertIn("membership_tuner", snapshot["roles"])
        self.assertEqual(snapshot["schema_version"], SCHEMA_VERSION)

    def test_service_exposes_schema_snapshot(self):
        service = PolicyService()
        snapshot = service.schema_snapshot()
        self.assertEqual(snapshot["schema_version"], SCHEMA_VERSION)
        self.assertIn("observation_fields", snapshot)
        self.assertIn("decision_fields", snapshot)

    def test_elevated_safety_freezes_membership_actions(self):
        model = MOISEOrganizationModel()
        obs = Observation(
            current_config_id=1,
            highest_known_config_id=2,
            validator_count=4,
            local_validator=False,
            can_participate=True,
            reject_total=1,
            adversary_score=0.4,
        )
        action = model.sanitize(obs, Action(submit_join=True, hydra_discovery_target=3))
        self.assertFalse(action.submit_join)
        self.assertEqual(action.hydra_discovery_target, 0)
        self.assertIn("membership-frozen", action.reason)

    def test_critical_safety_freezes_lane_actions_and_agent_actions(self):
        model = MOISEOrganizationModel()
        obs = Observation(
            validator_count=8,
            committee_size=5,
            pacemaker_timeout_ms=1300,
            mempool_max_batch_txs=768,
            mempool_proposal_interval_ms=85,
            local_validator=True,
            can_participate=True,
            backlog_missing=12,
            adversary_score=0.8,
            agents=[
                AgentObservation(
                    instance_id=0,
                    validator_count=8,
                    committee_size=4,
                    pacemaker_timeout_ms=1200,
                    mempool_max_batch_txs=512,
                    mempool_proposal_interval_ms=70,
                )
            ],
        )
        action = model.sanitize(
            obs,
            Action(
                committee_size=6,
                mempool_max_batch_txs=1024,
                mempool_proposal_interval_ms=80,
                pacemaker_timeout_ms=1500,
                agent_actions=[
                    {
                        "instance_id": 0,
                        "committee_size": 6,
                        "pacemaker_timeout_ms": 1600,
                        "mempool_max_batch_txs": 1024,
                        "mempool_proposal_interval_ms": 90,
                    }
                ],
            ),
        )
        self.assertEqual(action.committee_size, 5)
        self.assertEqual(action.mempool_max_batch_txs, 768)
        self.assertEqual(action.mempool_proposal_interval_ms, 85)
        self.assertEqual(len(action.agent_actions), 1)
        self.assertEqual(action.agent_actions[0].committee_size, 4)
        self.assertEqual(action.agent_actions[0].pacemaker_timeout_ms, 1600)
        self.assertEqual(action.agent_actions[0].mempool_max_batch_txs, 512)
        self.assertEqual(action.agent_actions[0].mempool_proposal_interval_ms, 70)
        self.assertEqual(action.pacemaker_timeout_ms, 1500)
        self.assertIn("lane-tuning-frozen", action.reason)

    def test_critical_safety_blocks_leave_when_node_cannot_participate(self):
        model = MOISEOrganizationModel()
        obs = Observation(
            current_config_id=1,
            highest_known_config_id=2,
            validator_count=5,
            committee_size=4,
            pacemaker_timeout_ms=1000,
            mempool_max_batch_txs=1024,
            mempool_proposal_interval_ms=80,
            local_validator=True,
            can_participate=False,
            backlog_missing=12,
            reject_total=15,
            adversary_score=0.9,
        )
        action = model.sanitize(obs, Action(submit_leave=True, hydra_discovery_target=3))
        self.assertFalse(action.submit_leave)
        self.assertEqual(action.hydra_discovery_target, 0)
        self.assertIn("membership-frozen", action.reason)
        self.assertIn("safety-escalation:critical", action.reason)

    def test_evaluate_exposes_explainability_metadata(self):
        model = MOISEOrganizationModel()
        decision = model.evaluate(
            Observation(
                current_config_id=1,
                highest_known_config_id=2,
                validator_count=4,
                local_validator=False,
                can_participate=True,
                reject_total=1,
                adversary_score=0.4,
                pending_joins=1,
            )
        )
        self.assertIn("reject_total", decision.evidence)
        self.assertIn("membership_tuner", decision.role_field_map)
        self.assertIn("submit_join", decision.role_field_map["membership_tuner"])

        snapshot = MOISEOrganizationModel().snapshot()
        self.assertIn("role_priority", snapshot)
        self.assertIn("freeze_rules", snapshot)
        self.assertIn("decision_fields", snapshot)
        self.assertIn("role_activation_rules", snapshot)
        self.assertIn("escalation_rules", snapshot)
        self.assertIn("safety_blocking_rules", snapshot)
        self.assertIn("blocked_field_reasoning", snapshot)
        self.assertIn("freeze_membership", snapshot["decision_fields"])
        self.assertIn("membership_tuner", snapshot["role_activation_rules"])
        self.assertIn("critical", snapshot["escalation_rules"])
        self.assertIn("cannot_participate", snapshot["safety_blocking_rules"])
        self.assertEqual(snapshot["blocked_field_reasoning"]["cannot-participate"], ["submit_leave"])
        self.assertEqual(snapshot["schema_version"], SCHEMA_VERSION)

    def test_evaluate_exposes_blocked_field_reasons(self):
        decision = MOISEOrganizationModel().evaluate(
            Observation(
                current_config_id=1,
                highest_known_config_id=2,
                validator_count=5,
                committee_size=4,
                pacemaker_timeout_ms=1000,
                mempool_max_batch_txs=1024,
                mempool_proposal_interval_ms=80,
                local_validator=True,
                can_participate=False,
                backlog_missing=12,
                reject_total=15,
                adversary_score=0.9,
            )
        )
        self.assertEqual(
            decision.blocked_field_reasons["submit_leave"],
            ["membership-frozen", "safety-escalation:critical", "cannot-participate"],
        )
        self.assertEqual(
            decision.blocked_field_reasons["committee_size"],
            ["lane-tuning-frozen", "safety-escalation:critical"],
        )


if __name__ == "__main__":
    unittest.main()
