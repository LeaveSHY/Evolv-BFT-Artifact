import unittest

from marl.dataset import ACTION_DIM, TrainingBatch
from marl.policy import SafeFACMACPolicy, project_action
from marl.trainer import AGENT_FEATURE_DIM, FEATURE_DIM, SafeFACMACModel, SafeFACMACTrainer
from marl.schemas import Action, AgentObservation, DecisionActionStage, Observation, TrajectorySample


class PolicyTests(unittest.TestCase):
    def test_trainer_fits_policy_and_inference_is_bounded(self):
        samples = [
            TrajectorySample(
                policy_name="safe-baseline",
                observation=Observation(
                    current_config_id=1,
                    highest_known_config_id=1,
                    validator_count=8,
                    committee_size=0,
                    pacemaker_timeout_ms=1000,
                    mempool_max_batch_txs=2048,
                    mempool_proposal_interval_ms=100,
                    throughput_tps=4000,
                    latency_p95_ms=400,
                    backlog_pending=120,
                    backlog_missing=4,
                    reject_total=3,
                    pending_joins=0,
                    pending_leaves=0,
                    lset_size=3,
                    can_participate=True,
                    local_validator=True,
                    heterogeneity_score=0.5,
                    churn_rate=0.2,
                    adversary_score=0.4,
                    network_jitter_ms=20,
                    ai_load_score=0.6,
                ),
                candidate=DecisionActionStage(
                    action=Action(
                        committee_size=4,
                        pacemaker_timeout_ms=1400,
                        mempool_max_batch_txs=1024,
                        mempool_proposal_interval_ms=120,
                        hydra_discovery_target=1,
                    ),
                    present=True,
                ),
                governed=DecisionActionStage(
                    action=Action(
                        committee_size=4,
                        pacemaker_timeout_ms=1400,
                        mempool_max_batch_txs=1024,
                        mempool_proposal_interval_ms=120,
                        hydra_discovery_target=1,
                    ),
                    present=True,
                ),
                applied=DecisionActionStage(
                    action=Action(
                        committee_size=4,
                        pacemaker_timeout_ms=1400,
                        mempool_max_batch_txs=1024,
                        mempool_proposal_interval_ms=120,
                        hydra_discovery_target=1,
                    ),
                    present=True,
                ),
                reward=-0.5,
            ),
            TrajectorySample(
                policy_name="safe-baseline",
                observation=Observation(
                    current_config_id=1,
                    highest_known_config_id=1,
                    validator_count=8,
                    committee_size=4,
                    pacemaker_timeout_ms=1200,
                    mempool_max_batch_txs=1024,
                    mempool_proposal_interval_ms=90,
                    throughput_tps=9000,
                    latency_p95_ms=80,
                    backlog_pending=8,
                    backlog_missing=0,
                    reject_total=0,
                    pending_joins=0,
                    pending_leaves=0,
                    lset_size=3,
                    can_participate=True,
                    local_validator=True,
                    heterogeneity_score=0.7,
                    churn_rate=0.1,
                    adversary_score=0.0,
                    network_jitter_ms=8,
                    ai_load_score=0.5,
                ),
                candidate=DecisionActionStage(
                    action=Action(
                        committee_size=4,
                        pacemaker_timeout_ms=1000,
                        mempool_max_batch_txs=1536,
                        mempool_proposal_interval_ms=70,
                    ),
                    present=True,
                ),
                governed=DecisionActionStage(
                    action=Action(
                        committee_size=4,
                        pacemaker_timeout_ms=1000,
                        mempool_max_batch_txs=1536,
                        mempool_proposal_interval_ms=70,
                    ),
                    present=True,
                ),
                applied=DecisionActionStage(
                    action=Action(
                        committee_size=4,
                        pacemaker_timeout_ms=1000,
                        mempool_max_batch_txs=1536,
                        mempool_proposal_interval_ms=70,
                    ),
                    present=True,
                ),
                reward=3.0,
            ),
        ]

        trainer = SafeFACMACTrainer()
        model = trainer.fit(samples)
        self.assertEqual(model.metadata["target_action"], "governed_stage_preferred")
        self.assertEqual(model.metadata["trainer"], "safe-facmac-ctde")
        self.assertEqual(model.metadata["trainer_family"], "CTDE-monotonic-mixing-with-safety-filter")
        self.assertTrue(model.metadata["ctde_architecture"])
        self.assertTrue(model.metadata["pre_argmax_safety_filter"])
        self.assertEqual(model.metadata["trainer_config"]["seed"], 7)
        self.assertEqual(model.metadata["trainer_config"]["ridge"], 1e-3)
        self.assertEqual(model.metadata["trainer_config"]["epochs"], 24)
        policy = SafeFACMACPolicy(model)

        action = policy.decide(
            Observation(
                current_config_id=1,
                highest_known_config_id=2,
                validator_count=8,
                committee_size=4,
                pacemaker_timeout_ms=1100,
                mempool_max_batch_txs=1200,
                mempool_proposal_interval_ms=80,
                throughput_tps=8000,
                latency_p95_ms=90,
                backlog_pending=12,
                backlog_missing=1,
                reject_total=0,
                pending_joins=0,
                pending_leaves=0,
                lset_size=3,
                can_participate=True,
                local_validator=False,
                heterogeneity_score=0.8,
                churn_rate=0.1,
                adversary_score=0.0,
                network_jitter_ms=10,
                ai_load_score=0.6,
            )
        )

        self.assertGreaterEqual(action.pacemaker_timeout_ms, 250)
        self.assertLessEqual(action.pacemaker_timeout_ms, 5000)
        self.assertGreaterEqual(action.mempool_max_batch_txs, 1)
        self.assertLessEqual(action.mempool_max_batch_txs, 8192)

    def test_policy_emits_per_agent_actions_when_agents_present(self):
        samples = [
            TrajectorySample(
                policy_name="safe-baseline",
                observation=Observation(
                    current_config_id=1,
                    highest_known_config_id=1,
                    validator_count=8,
                    throughput_tps=6000,
                    latency_p95_ms=120,
                    can_participate=True,
                    local_validator=True,
                    agents=[
                        AgentObservation(instance_id=0, validator_count=8, committee_size=4, pacemaker_timeout_ms=1200, mempool_max_batch_txs=512, mempool_proposal_interval_ms=80),
                        AgentObservation(instance_id=1, validator_count=8, committee_size=6, pacemaker_timeout_ms=1500, mempool_max_batch_txs=1024, mempool_proposal_interval_ms=60),
                    ],
                ),
                candidate=DecisionActionStage(
                    action=Action(
                        agent_actions=[
                            {"instance_id": 0, "committee_size": 4, "pacemaker_timeout_ms": 1100, "mempool_max_batch_txs": 768, "mempool_proposal_interval_ms": 70},
                            {"instance_id": 1, "committee_size": 6, "pacemaker_timeout_ms": 1400, "mempool_max_batch_txs": 1280, "mempool_proposal_interval_ms": 50},
                        ]
                    ),
                    present=True,
                ),
                governed=DecisionActionStage(
                    action=Action(
                        agent_actions=[
                            {"instance_id": 0, "committee_size": 4, "pacemaker_timeout_ms": 1100, "mempool_max_batch_txs": 768, "mempool_proposal_interval_ms": 70},
                            {"instance_id": 1, "committee_size": 6, "pacemaker_timeout_ms": 1400, "mempool_max_batch_txs": 1280, "mempool_proposal_interval_ms": 50},
                        ]
                    ),
                    present=True,
                ),
                applied=DecisionActionStage(
                    action=Action(
                        agent_actions=[
                            {"instance_id": 0, "committee_size": 4, "pacemaker_timeout_ms": 1100, "mempool_max_batch_txs": 768, "mempool_proposal_interval_ms": 70},
                            {"instance_id": 1, "committee_size": 6, "pacemaker_timeout_ms": 1400, "mempool_max_batch_txs": 1280, "mempool_proposal_interval_ms": 50},
                        ]
                    ),
                    present=True,
                ),
                reward=2.0,
            )
        ]
        trainer = SafeFACMACTrainer()
        model = trainer.fit(samples)
        policy = SafeFACMACPolicy(model)
        action = policy.decide(samples[0].observation)
        self.assertEqual(len(action.agent_actions), 2)
        self.assertEqual(action.agent_actions[0].instance_id, 0)
        self.assertEqual(action.agent_actions[1].instance_id, 1)

    def test_policy_can_emit_membership_join_action(self):
        samples = [
            TrajectorySample(
                policy_name="safe-baseline",
                observation=Observation(
                    current_config_id=1,
                    highest_known_config_id=2,
                    validator_count=4,
                    can_participate=True,
                    local_validator=False,
                    pending_joins=0,
                ),
                candidate=DecisionActionStage(
                    action=Action(
                        submit_join=True,
                        hydra_discovery_target=3,
                    ),
                    present=True,
                ),
                governed=DecisionActionStage(
                    action=Action(
                        submit_join=True,
                        hydra_discovery_target=3,
                    ),
                    present=True,
                ),
                applied=DecisionActionStage(
                    action=Action(
                        submit_join=True,
                        hydra_discovery_target=3,
                    ),
                    present=True,
                ),
                reward=1.5,
            )
        ]
        trainer = SafeFACMACTrainer()
        model = trainer.fit(samples)
        policy = SafeFACMACPolicy(model)
        action = policy.decide(samples[0].observation)
        self.assertTrue(action.submit_join)
        self.assertEqual(action.hydra_discovery_target, 3)

    def test_policy_prefers_role_actor_for_membership_role(self):
        trainer = SafeFACMACTrainer()
        model = trainer.fit(
            [
                TrajectorySample(
                    policy_name="safe-baseline",
                    observation=Observation(
                        current_config_id=1,
                        highest_known_config_id=2,
                        validator_count=4,
                        can_participate=True,
                        local_validator=False,
                        pending_joins=0,
                    ),
                    candidate=DecisionActionStage(
                        action=Action(submit_join=True, hydra_discovery_target=3),
                        present=True,
                    ),
                    governed=DecisionActionStage(
                        action=Action(submit_join=True, hydra_discovery_target=3),
                        present=True,
                    ),
                    applied=DecisionActionStage(
                        action=Action(submit_join=True, hydra_discovery_target=3),
                        present=True,
                    ),
                    reward=2.0,
                )
            ]
        )
        model.role_actor_bias["membership_tuner"] = [1.0, 0.0, 3.0]
        policy = SafeFACMACPolicy(model)
        action = policy.decide(
            Observation(
                current_config_id=1,
                highest_known_config_id=2,
                validator_count=4,
                can_participate=True,
                local_validator=False,
                pending_joins=0,
            )
        )
        self.assertTrue(action.submit_join)
        self.assertEqual(action.hydra_discovery_target, 3)

    def test_role_heads_compose_with_base_actor_by_field_family(self):
        model = SafeFACMACModel(
            actor_weights=[],
            actor_bias=[6.0, 900.0, 1400.0, 65.0, 0.0, 0.0, 0.0],
            role_actor_weights={
                "membership_tuner": [],
                "lane_tuner": [],
                "recovery_tuner": [],
            },
            role_actor_bias={
                "membership_tuner": [1.0, 0.0, 3.0],
                "lane_tuner": [7.0, 1800.0, 45.0],
                "recovery_tuner": [1500.0],
            },
            agent_actor_weights=[],
            agent_actor_bias=[],
            critic_weights=[],
            critic_bias=0.0,
            metadata={
                "role_head_coverage": {
                    "membership_tuner": {"active_samples": 1},
                    "lane_tuner": {"active_samples": 1},
                    "recovery_tuner": {"active_samples": 1},
                }
            },
        )
        policy = SafeFACMACPolicy(model)
        action = policy.decide(
            Observation(
                current_config_id=1,
                highest_known_config_id=2,
                validator_count=8,
                committee_size=4,
                pacemaker_timeout_ms=1100,
                mempool_max_batch_txs=1200,
                mempool_proposal_interval_ms=80,
                reject_total=1,
                can_participate=True,
                local_validator=True,
                pending_joins=0,
            )
        )
        self.assertEqual(action.committee_size, 7)
        self.assertEqual(action.pacemaker_timeout_ms, 1500)
        self.assertEqual(action.mempool_max_batch_txs, 1800)
        self.assertEqual(action.mempool_proposal_interval_ms, 45)
        self.assertFalse(action.submit_join)
        self.assertEqual(action.hydra_discovery_target, 0)

    def test_policy_exposes_role_override_attribution(self):
        model = SafeFACMACModel(
            actor_weights=[],
            actor_bias=[6.0, 900.0, 1400.0, 65.0, 0.0, 0.0, 0.0],
            role_actor_weights={
                "membership_tuner": [],
                "lane_tuner": [],
                "recovery_tuner": [],
            },
            role_actor_bias={
                "membership_tuner": [1.0, 0.0, 3.0],
                "lane_tuner": [7.0, 1800.0, 45.0],
                "recovery_tuner": [1500.0],
            },
            agent_actor_weights=[],
            agent_actor_bias=[],
            critic_weights=[],
            critic_bias=0.0,
            metadata={},
        )
        policy = SafeFACMACPolicy(model)
        observation = Observation(
            current_config_id=1,
            highest_known_config_id=2,
            validator_count=8,
            committee_size=4,
            pacemaker_timeout_ms=1100,
            mempool_max_batch_txs=1200,
            mempool_proposal_interval_ms=80,
            reject_total=1,
            can_participate=True,
            local_validator=True,
            pending_joins=0,
        )
        action, attribution = policy.propose_with_role_attribution(observation)
        self.assertEqual(action.to_dict(), policy.propose(observation).to_dict())
        self.assertEqual(set(attribution["override_roles"]), {"lane_tuner", "recovery_tuner", "membership_tuner"})
        self.assertEqual(
            set(attribution["by_role"]["lane_tuner"]["fields"]),
            {"committee_size", "mempool_max_batch_txs", "mempool_proposal_interval_ms"},
        )
        self.assertEqual(attribution["agent_count"], 0)
        self.assertEqual(attribution["agent_instances"], [])

    def test_policy_exposes_agent_override_context(self):
        model = SafeFACMACModel(
            actor_weights=[],
            actor_bias=[4.0, 1000.0, 1024.0, 80.0, 0.0, 0.0, 0.0],
            role_actor_weights={},
            role_actor_bias={},
            agent_actor_weights=[[1.0] * AGENT_FEATURE_DIM for _ in range(4)],
            agent_actor_bias=[4.0, 1000.0, 1024.0, 80.0],
            critic_weights=[],
            critic_bias=0.0,
            metadata={},
        )
        policy = SafeFACMACPolicy(model)
        observation = Observation(
            validator_count=8,
            local_validator=True,
            agents=[
                AgentObservation(instance_id=0, validator_count=8, committee_size=4, pacemaker_timeout_ms=1200, mempool_max_batch_txs=512, mempool_proposal_interval_ms=70),
                AgentObservation(instance_id=1, validator_count=8, committee_size=6, pacemaker_timeout_ms=1500, mempool_max_batch_txs=1024, mempool_proposal_interval_ms=60),
            ],
        )
        _, attribution = policy.propose_with_role_attribution(observation)
        self.assertEqual(attribution["agent_count"], 2)
        self.assertEqual(attribution["agent_instances"], [0, 1])
        trainer = SafeFACMACTrainer()
        model = trainer.fit(
            [
                TrajectorySample(
                    policy_name="safe-baseline",
                    observation=Observation(
                        current_config_id=1,
                        highest_known_config_id=2,
                        validator_count=4,
                        can_participate=True,
                        local_validator=False,
                    ),
                    candidate=DecisionActionStage(
                        action=Action(submit_join=False, hydra_discovery_target=0),
                        present=True,
                    ),
                    governed=DecisionActionStage(
                        action=Action(submit_join=False, hydra_discovery_target=0),
                        present=True,
                    ),
                    applied=DecisionActionStage(
                        action=Action(submit_join=False, hydra_discovery_target=0),
                        present=True,
                    ),
                    reward=0.1,
                    role_rewards={"membership_tuner": 0.0},
                ),
                TrajectorySample(
                    policy_name="safe-baseline",
                    observation=Observation(
                        current_config_id=1,
                        highest_known_config_id=2,
                        validator_count=4,
                        can_participate=True,
                        local_validator=False,
                    ),
                    candidate=DecisionActionStage(
                        action=Action(submit_join=True, hydra_discovery_target=3),
                        present=True,
                    ),
                    governed=DecisionActionStage(
                        action=Action(submit_join=True, hydra_discovery_target=3),
                        present=True,
                    ),
                    applied=DecisionActionStage(
                        action=Action(submit_join=True, hydra_discovery_target=3),
                        present=True,
                    ),
                    reward=0.1,
                    role_rewards={"membership_tuner": 3.0},
                ),
            ]
        )
        self.assertGreaterEqual(model.role_actor_bias["membership_tuner"][0], 0.5)
        self.assertGreaterEqual(model.role_actor_bias["membership_tuner"][2], 1.5)

    def test_trained_role_head_inference_handles_trainer_model_path(self):
        trainer = SafeFACMACTrainer()
        model = trainer.fit(
            [
                TrajectorySample(
                    policy_name="safe-baseline",
                    observation=Observation(
                        current_config_id=1,
                        highest_known_config_id=2,
                        validator_count=4,
                        can_participate=True,
                        local_validator=False,
                    ),
                    candidate=DecisionActionStage(
                        action=Action(submit_join=True, hydra_discovery_target=2),
                        present=True,
                    ),
                    governed=DecisionActionStage(
                        action=Action(submit_join=True, hydra_discovery_target=2),
                        present=True,
                    ),
                    applied=DecisionActionStage(
                        action=Action(submit_join=True, hydra_discovery_target=2),
                        present=True,
                    ),
                    reward=1.0,
                    role_rewards={"membership_tuner": 2.0},
                )
            ]
        )
        policy = SafeFACMACPolicy(model)
        action = policy.decide(
            Observation(
                current_config_id=1,
                highest_known_config_id=2,
                validator_count=4,
                can_participate=True,
                local_validator=False,
            )
        )
        self.assertTrue(action.submit_join)
        self.assertEqual(action.hydra_discovery_target, 2)

    def test_non_finite_role_rewards_are_sanitized(self):
        trainer = SafeFACMACTrainer()
        model = trainer.fit(
            [
                TrajectorySample(
                    policy_name="safe-baseline",
                    observation=Observation(
                        current_config_id=1,
                        highest_known_config_id=2,
                        validator_count=4,
                        can_participate=True,
                        local_validator=False,
                    ),
                    candidate=DecisionActionStage(
                        action=Action(submit_join=True, hydra_discovery_target=3),
                        present=True,
                    ),
                    governed=DecisionActionStage(
                        action=Action(submit_join=True, hydra_discovery_target=3),
                        present=True,
                    ),
                    applied=DecisionActionStage(
                        action=Action(submit_join=True, hydra_discovery_target=3),
                        present=True,
                    ),
                    reward=0.5,
                    role_rewards={"membership_tuner": float("inf")},
                )
            ]
        )
        self.assertLessEqual(model.role_actor_bias["membership_tuner"][0], 1.0)
        self.assertLessEqual(model.role_actor_bias["membership_tuner"][2], 3.0)

    def test_role_heads_train_only_on_role_active_samples_and_expose_metadata(self):
        trainer = SafeFACMACTrainer()
        model = trainer.fit(
            [
                TrajectorySample(
                    policy_name="safe-baseline",
                    observation=Observation(
                        current_config_id=1,
                        highest_known_config_id=1,
                        validator_count=4,
                        can_participate=True,
                        local_validator=True,
                    ),
                    candidate=DecisionActionStage(action=Action(submit_join=True, hydra_discovery_target=3), present=True),
                    governed=DecisionActionStage(action=Action(submit_join=True, hydra_discovery_target=3), present=True),
                    applied=DecisionActionStage(action=Action(submit_join=True, hydra_discovery_target=3), present=True),
                    reward=1.0,
                    role_rewards={"membership_tuner": 5.0},
                    team_reward=1.0,
                ),
                TrajectorySample(
                    policy_name="safe-baseline",
                    observation=Observation(
                        current_config_id=1,
                        highest_known_config_id=2,
                        validator_count=4,
                        can_participate=True,
                        local_validator=False,
                    ),
                    candidate=DecisionActionStage(action=Action(submit_join=False, hydra_discovery_target=0), present=True),
                    governed=DecisionActionStage(action=Action(submit_join=False, hydra_discovery_target=0), present=True),
                    applied=DecisionActionStage(action=Action(submit_join=False, hydra_discovery_target=0), present=True),
                    reward=0.1,
                    role_rewards={"membership_tuner": 0.0},
                    team_reward=0.2,
                ),
            ]
        )
        self.assertLess(model.role_actor_bias["membership_tuner"][0], 0.5)
        self.assertLess(model.role_actor_bias["membership_tuner"][2], 1.0)
        self.assertEqual(model.metadata["role_head_coverage"]["membership_tuner"]["active_samples"], 1)
        self.assertAlmostEqual(model.metadata["reward_summary"]["team_reward_mean"], 0.6)

    def test_missing_role_heads_do_not_override_base_policy(self):
        model = SafeFACMACModel(
            actor_weights=[],
            actor_bias=[6.0, 900.0, 1500.0, 75.0, 0.0, 0.0, 0.0],
            role_actor_weights={"membership_tuner": []},
            role_actor_bias={"membership_tuner": [1.0, 0.0, 3.0]},
            agent_actor_weights=[],
            agent_actor_bias=[],
            critic_weights=[],
            critic_bias=0.0,
            metadata={"role_head_coverage": {"membership_tuner": {"active_samples": 1}}},
        )
        policy = SafeFACMACPolicy(model)
        action = policy.decide(
            Observation(
                current_config_id=1,
                highest_known_config_id=1,
                validator_count=8,
                committee_size=6,
                pacemaker_timeout_ms=900,
                mempool_max_batch_txs=1500,
                mempool_proposal_interval_ms=75,
                can_participate=True,
                local_validator=True,
            )
        )
        self.assertEqual(action.committee_size, 6)
        self.assertEqual(action.mempool_max_batch_txs, 1500)
        self.assertEqual(action.mempool_proposal_interval_ms, 75)

    def test_role_heads_ignore_untrusted_metadata_and_still_apply(self):
        model = SafeFACMACModel(
            actor_weights=[],
            actor_bias=[6.0, 900.0, 1500.0, 75.0, 0.0, 0.0, 0.0],
            role_actor_weights={"lane_tuner": []},
            role_actor_bias={"lane_tuner": [7.0, 1800.0, 45.0]},
            agent_actor_weights=[],
            agent_actor_bias=[],
            critic_weights=[],
            critic_bias=0.0,
            metadata={"role_head_coverage": "malicious"},
        )
        policy = SafeFACMACPolicy(model)
        action = policy.decide(
            Observation(
                current_config_id=1,
                highest_known_config_id=1,
                validator_count=8,
                committee_size=6,
                pacemaker_timeout_ms=900,
                mempool_max_batch_txs=1500,
                mempool_proposal_interval_ms=75,
                can_participate=True,
                local_validator=True,
            )
        )
        self.assertEqual(action.committee_size, 7)
        self.assertEqual(action.mempool_max_batch_txs, 1800)
        self.assertEqual(action.mempool_proposal_interval_ms, 45)

    def test_non_finite_rewards_are_sanitized_in_metadata(self):
        trainer = SafeFACMACTrainer()
        model = trainer.fit(
            [
                TrajectorySample(
                    policy_name="safe-baseline",
                    observation=Observation(
                        current_config_id=1,
                        highest_known_config_id=2,
                        validator_count=4,
                        can_participate=True,
                        local_validator=False,
                    ),
                    candidate=DecisionActionStage(action=Action(submit_join=True, hydra_discovery_target=3), present=True),
                    governed=DecisionActionStage(action=Action(submit_join=True, hydra_discovery_target=3), present=True),
                    applied=DecisionActionStage(action=Action(submit_join=True, hydra_discovery_target=3), present=True),
                    reward=float("nan"),
                    team_reward=float("inf"),
                    role_rewards={"membership_tuner": 1.0},
                )
            ]
        )
        self.assertEqual(model.metadata["reward_summary"]["reward_mean"], 0.0)
        self.assertEqual(model.metadata["reward_summary"]["team_reward_mean"], 0.0)

    def test_extreme_role_overrides_preserve_membership_and_lane_safety_invariants(self):
        model = SafeFACMACModel(
            actor_weights=[],
            actor_bias=[1.0, 50.0, 50000.0, 1.0, 1.0, 1.0, 99.0],
            role_actor_weights={
                "membership_tuner": [],
                "lane_tuner": [],
                "recovery_tuner": [],
            },
            role_actor_bias={
                "membership_tuner": [1.0, 1.0, 99.0],
                "lane_tuner": [99.0, 50000.0, 1.0],
                "recovery_tuner": [50.0],
            },
            agent_actor_weights=[],
            agent_actor_bias=[],
            critic_weights=[],
            critic_bias=0.0,
            metadata={},
        )
        policy = SafeFACMACPolicy(model)
        action = policy.decide(
            Observation(
                current_config_id=1,
                highest_known_config_id=2,
                validator_count=3,
                committee_size=3,
                pacemaker_timeout_ms=900,
                mempool_max_batch_txs=1500,
                mempool_proposal_interval_ms=75,
                can_participate=False,
                local_validator=True,
                pending_joins=1,
                pending_leaves=1,
                backlog_missing=20,
                reject_total=12,
                adversary_score=0.9,
                churn_rate=0.6,
            )
        )
        self.assertFalse(action.submit_join)
        self.assertFalse(action.submit_leave)
        self.assertEqual(action.hydra_discovery_target, 0)
        self.assertEqual(action.committee_size, 4)
        self.assertEqual(action.mempool_max_batch_txs, 1500)
        self.assertEqual(action.mempool_proposal_interval_ms, 75)
        self.assertGreaterEqual(action.pacemaker_timeout_ms, 250)
        self.assertLessEqual(action.pacemaker_timeout_ms, 5000)

    def test_project_action_resolves_conflicting_membership_intents_safely(self):
        action = project_action(
            Observation(
                current_config_id=1,
                highest_known_config_id=1,
                validator_count=4,
                committee_size=4,
                pacemaker_timeout_ms=1000,
                mempool_max_batch_txs=1024,
                mempool_proposal_interval_ms=80,
                can_participate=True,
                local_validator=False,
            ),
            Action(
                committee_size=4,
                pacemaker_timeout_ms=1000,
                mempool_max_batch_txs=1024,
                mempool_proposal_interval_ms=80,
                submit_join=True,
                submit_leave=True,
                hydra_discovery_target=2,
            ),
        )
        self.assertTrue(action.submit_join)
        self.assertFalse(action.submit_leave)
        self.assertFalse(action.submit_join and action.submit_leave)

    def test_project_action_blocks_leave_at_validator_floor(self):
        action = project_action(
            Observation(
                current_config_id=1,
                highest_known_config_id=1,
                validator_count=3,
                committee_size=4,
                pacemaker_timeout_ms=1000,
                mempool_max_batch_txs=1024,
                mempool_proposal_interval_ms=80,
                can_participate=True,
                local_validator=True,
            ),
            Action(
                committee_size=4,
                pacemaker_timeout_ms=1000,
                mempool_max_batch_txs=1024,
                mempool_proposal_interval_ms=80,
                submit_leave=True,
            ),
        )
        self.assertFalse(action.submit_leave)

    def test_trainer_terminal_samples_stop_bootstrap(self):
        trainer = SafeFACMACTrainer()
        current_state = [1.0] + [0.0] * (FEATURE_DIM - 1)
        future_state = [0.0, 1.0] + [0.0] * (FEATURE_DIM - 2)
        zero_action = [0.0] * ACTION_DIM
        terminal_batch = TrainingBatch(
            features=[current_state, future_state],
            next_features=[future_state, future_state],
            actions=[zero_action, zero_action],
            candidate_actions=[zero_action, zero_action],
            governed_actions=[zero_action, zero_action],
            rewards=[0.0, 1.0],
            team_rewards=[0.0, 1.0],
            dones=[1.0, 1.0],
            governance_deltas=[0.0, 0.0],
            guardrail_deltas=[0.0, 0.0],
            agent_features=[],
            agent_actions=[],
        )
        bootstrapped_batch = TrainingBatch(
            features=[current_state, future_state],
            next_features=[future_state, future_state],
            actions=[zero_action, zero_action],
            candidate_actions=[zero_action, zero_action],
            governed_actions=[zero_action, zero_action],
            rewards=[0.0, 1.0],
            team_rewards=[0.0, 1.0],
            dones=[0.0, 1.0],
            governance_deltas=[0.0, 0.0],
            guardrail_deltas=[0.0, 0.0],
            agent_features=[],
            agent_actions=[],
        )
        terminal_model = trainer.fit_batch(terminal_batch)
        bootstrapped_model = trainer.fit_batch(bootstrapped_batch)
        self.assertGreater(float(bootstrapped_model.critic_bias), float(terminal_model.critic_bias))
        self.assertGreater(float(bootstrapped_model.critic_weights[0]), float(terminal_model.critic_weights[0]))


if __name__ == "__main__":
    unittest.main()
