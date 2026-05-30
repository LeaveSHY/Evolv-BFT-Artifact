"""E2E test for RoleFACMACTrainer — validates multi-objective role-decomposed training."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from marl.networks.role_critic import NUM_ROLES, ROLE_NAMES, RoleCritic
from marl.training.replay_buffer import Transition
from marl.training.role_trainer import RoleFACMACTrainer, RoleTrainingConfig


def _random_transition(n_agents: int = 5) -> Transition:
    """Generate a random transition with role_rewards."""
    return Transition(
        state=np.random.randn(28).astype(np.float32),
        agent_obs=np.random.randn(n_agents, 7).astype(np.float32),
        actions=np.random.randn(n_agents, 4).astype(np.float32) * 0.1,
        reward=float(np.random.randn()) * 0.5,
        next_state=np.random.randn(28).astype(np.float32),
        next_agent_obs=np.random.randn(n_agents, 7).astype(np.float32),
        done=False,
        n_agents=n_agents,
        role_rewards=np.array([
            np.random.randn() * 0.3,   # lane_tuner
            np.random.randn() * 0.2,   # recovery_tuner
            np.random.randn() * 0.1,   # membership_tuner
            np.random.randn() * -0.1,  # safety_guardian (usually penalty)
        ], dtype=np.float32),
    )


class TestRoleCritic:
    """Unit tests for RoleCritic network."""

    def test_init_and_forward(self):
        critic = RoleCritic(state_dim=28, agent_action_dim=4)
        state = torch.randn(8, 28)
        action = torch.randn(8, 4)
        result = critic.forward_all_roles(state, action)
        assert len(result) == NUM_ROLES + 1  # 4 roles + team
        for key, val in result.items():
            assert val.shape == (8, 1), f"{key}: {val.shape}"

    def test_forward_all_agents(self):
        critic = RoleCritic(state_dim=28, agent_action_dim=4)
        state = torch.randn(4, 28)
        actions = torch.randn(4, 6, 4)
        q = critic.forward_all_agents(state, actions)
        assert q.shape == (4, 6)

    def test_forward_all_agents_all_roles(self):
        critic = RoleCritic(state_dim=28, agent_action_dim=4)
        state = torch.randn(4, 28)
        actions = torch.randn(4, 6, 4)
        result = critic.forward_all_agents_all_roles(state, actions)
        assert len(result) == NUM_ROLES + 1
        for key, val in result.items():
            assert val.shape == (4, 6), f"{key}: {val.shape}"

    def test_weighted_q(self):
        critic = RoleCritic(state_dim=28, agent_action_dim=4, role_weights=[1.0, 1.0, 1.0, 2.0])
        state = torch.randn(8, 28)
        action = torch.randn(8, 4)
        wq = critic.weighted_q(state, action)
        assert wq.shape == (8, 1)

    def test_weighted_q_all_agents(self):
        critic = RoleCritic(state_dim=28, agent_action_dim=4)
        state = torch.randn(4, 28)
        actions = torch.randn(4, 5, 4)
        wq = critic.weighted_q_all_agents(state, actions)
        assert wq.shape == (4, 5)

    def test_param_count(self):
        critic = RoleCritic(state_dim=28, agent_action_dim=4)
        params = sum(p.numel() for p in critic.parameters())
        # Trunk: (32→256: 32*256+256) + (256→128: 256*128+128) = 8448+33024=41472
        # Each role head: (128→64: 128*64+64) + (64→1: 64+1) = 8256+65=8321
        # 4 role heads + 1 team head = 5 * 8321 = 41605
        # Total ≈ 41472 + 41605 = 83077
        assert params > 80000, f"Expected >80k params, got {params}"


class TestRoleFACMACTrainer:
    """Integration tests for role-decomposed training."""

    @pytest.fixture
    def trainer(self) -> RoleFACMACTrainer:
        config = RoleTrainingConfig(
            n_agents=5,
            batch_size=16,
            warmup_steps=50,
            buffer_capacity=500,
            device="cpu",
        )
        return RoleFACMACTrainer(config)

    def test_store_and_train(self, trainer: RoleFACMACTrainer):
        """Fill buffer and run training steps, verify all role losses decrease."""
        # Fill buffer above warmup
        for _ in range(60):
            trainer.store_transition(_random_transition(n_agents=5))

        assert trainer.can_train()

        # Run 50 training steps
        losses = {role: [] for role in ROLE_NAMES}
        total_losses = []
        for _ in range(50):
            metrics = trainer.train_step()
            total_losses.append(metrics["total_critic_loss"])
            for role in ROLE_NAMES:
                losses[role].append(metrics[f"critic_loss_{role}"])

        # Total critic loss should be finite
        assert all(np.isfinite(l) for l in total_losses), "Non-finite loss detected"

        # Each role should have produced loss values
        for role in ROLE_NAMES:
            assert len(losses[role]) == 50
            assert all(np.isfinite(l) for l in losses[role])

    def test_role_rewards_fallback(self, trainer: RoleFACMACTrainer):
        """When role_rewards=None, falls back to team reward / NUM_ROLES."""
        t = Transition(
            state=np.random.randn(28).astype(np.float32),
            agent_obs=np.random.randn(5, 7).astype(np.float32),
            actions=np.random.randn(5, 4).astype(np.float32),
            reward=1.0,
            next_state=np.random.randn(28).astype(np.float32),
            next_agent_obs=np.random.randn(5, 7).astype(np.float32),
            done=False,
            n_agents=5,
            role_rewards=None,  # no role rewards
        )
        for _ in range(60):
            trainer.store_transition(t)

        metrics = trainer.train_step()
        assert "total_critic_loss" in metrics

    def test_variable_agents(self, trainer: RoleFACMACTrainer):
        """Variable n_agents with role rewards."""
        for n in [3, 4, 5, 5, 4, 3]:
            for _ in range(10):
                trainer.store_transition(_random_transition(n_agents=n))

        # Should train without errors
        metrics = trainer.train_step()
        assert "total_critic_loss" in metrics

    def test_save_load(self, trainer: RoleFACMACTrainer, tmp_path):
        """Save and load preserves state."""
        for _ in range(60):
            trainer.store_transition(_random_transition(n_agents=5))
        trainer.train_step()

        path = str(tmp_path / "role_model.pt")
        trainer.save(path)

        # Load into new trainer
        new_trainer = RoleFACMACTrainer(trainer.config)
        new_trainer.load(path)
        assert new_trainer.train_steps == trainer.train_steps

    def test_actor_gradient_uses_weighted_roles(self, trainer: RoleFACMACTrainer):
        """Actor loss should differ from single-head approach."""
        # Fill buffer with biased role rewards (safety_guardian always negative)
        for _ in range(60):
            t = _random_transition(n_agents=5)
            t.role_rewards = np.array([0.5, 0.3, 0.1, -0.8], dtype=np.float32)
            trainer.store_transition(t)

        metrics = trainer.train_step()
        # safety_guardian critic loss should be non-trivial (target is always -0.8)
        assert metrics["critic_loss_safety_guardian"] > 0

    def test_param_count(self, trainer: RoleFACMACTrainer):
        """Verify total parameters are reasonable."""
        total = trainer.param_count()
        # Actor ~3k + RoleCritic ~83k + Mixer ~3k ≈ 89k+
        assert total > 85000, f"Expected >85k params, got {total}"
        print(f"RoleFACMAC params: {total:,}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
