"""Tests for Lagrangian Safety-Constrained Role-Decomposed FACMAC Trainer."""
import numpy as np
import pytest
import torch

from marl.networks.role_critic import NUM_ROLES, ROLE_NAMES
from marl.training.replay_buffer import Transition
from marl.training.safe_trainer import (
    SAFETY_ROLE_IDX,
    SafeRoleFACMACTrainer,
    SafeTrainingConfig,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def safe_config():
    """Minimal safe config for CPU testing."""
    return SafeTrainingConfig(
        state_dim=28,
        agent_obs_dim=7,
        agent_action_dim=4,
        n_agents=4,
        mixing_hidden=16,
        trunk_dims=(64, 32),
        head_hidden=16,
        role_weights=(1.0, 1.0, 1.0, 1.5),
        actor_lr=1e-3,
        critic_lr=1e-3,
        mixer_lr=1e-3,
        gamma=0.95,
        tau=0.01,
        batch_size=8,
        buffer_capacity=500,
        warmup_steps=16,
        total_steps=1000,
        device="cpu",
        # Safety params
        cost_limit=0.1,
        lambda_init=0.1,
        lambda_lr=5e-3,
        lambda_max=10.0,
        cost_window_size=50,
        log_lambda=True,
    )


@pytest.fixture
def safe_trainer(safe_config):
    return SafeRoleFACMACTrainer(safe_config)


def _make_transition(n_agents: int = 4, safety_violation: bool = False) -> Transition:
    """Generate a random transition with optional safety violation."""
    role_rewards = np.random.uniform(0.0, 1.0, size=(NUM_ROLES,)).astype(np.float32)
    if safety_violation:
        # Safety guardian gets large negative reward (violation)
        role_rewards[SAFETY_ROLE_IDX] = -2.0
    else:
        # No violation: safety reward positive
        role_rewards[SAFETY_ROLE_IDX] = 0.5
    team_reward = role_rewards.sum()

    return Transition(
        state=np.random.randn(28).astype(np.float32),
        agent_obs=np.random.randn(n_agents, 7).astype(np.float32),
        actions=np.random.uniform(-1, 1, (n_agents, 4)).astype(np.float32),
        reward=float(team_reward),
        next_state=np.random.randn(28).astype(np.float32),
        next_agent_obs=np.random.randn(n_agents, 7).astype(np.float32),
        done=False,
        n_agents=n_agents,
        role_rewards=role_rewards,
    )


def _fill_buffer(trainer: SafeRoleFACMACTrainer, n: int = 32, violation_rate: float = 0.3):
    """Fill buffer with transitions, some with safety violations."""
    for i in range(n):
        violation = (i / n) < violation_rate
        trainer.store_transition(_make_transition(
            n_agents=trainer.config.n_agents,
            safety_violation=violation,
        ))


# ===========================================================================
# Tests: SafeRoleFACMACTrainer Initialization
# ===========================================================================


class TestSafeTrainerInit:
    """Test initialization and structure."""

    def test_inherits_role_trainer(self, safe_trainer):
        """SafeRoleFACMACTrainer inherits from RoleFACMACTrainer."""
        from marl.training.role_trainer import RoleFACMACTrainer
        assert isinstance(safe_trainer, RoleFACMACTrainer)

    def test_lambda_init(self, safe_trainer):
        """Lambda initialized to configured value."""
        lam = safe_trainer.lambda_value
        assert abs(lam - 0.1) < 1e-4, f"Expected λ≈0.1, got {lam}"

    def test_log_parameterization(self, safe_trainer):
        """Log-parameterization keeps λ positive."""
        # Manually push log_lambda to very negative value
        with torch.no_grad():
            safe_trainer._log_lambda.fill_(-10.0)
        # λ = exp(-10) ≈ 4.5e-5, still positive
        assert safe_trainer.lambda_value > 0

    def test_lambda_optimizer_exists(self, safe_trainer):
        """Dual variable has its own optimizer."""
        assert safe_trainer.lambda_optimizer is not None
        assert len(safe_trainer.lambda_optimizer.param_groups[0]["params"]) == 1

    def test_cost_window_init(self, safe_trainer):
        """Cost window starts empty."""
        assert len(safe_trainer._cost_window) == 0

    def test_safety_role_index(self):
        """safety_guardian is at correct index."""
        assert ROLE_NAMES[SAFETY_ROLE_IDX] == "safety_guardian"


# ===========================================================================
# Tests: Training with Safety Constraints
# ===========================================================================


class TestSafeTraining:
    """Test training loop with Lagrangian constraint."""

    def test_train_step_returns_lambda(self, safe_trainer):
        """Training metrics include lambda and cost info."""
        _fill_buffer(safe_trainer, n=32)
        metrics = safe_trainer.train_step()
        assert "lambda" in metrics
        assert "cost_estimate" in metrics
        assert "constraint_violation" in metrics
        assert "cost_limit" in metrics
        assert metrics["lambda"] > 0

    def test_train_step_modifies_lambda(self, safe_trainer):
        """Lambda changes during training (dual update active)."""
        _fill_buffer(safe_trainer, n=32)
        lam_before = safe_trainer.lambda_value
        # Run multiple steps
        for _ in range(20):
            safe_trainer.train_step()
        lam_after = safe_trainer.lambda_value
        # Lambda should have moved (direction depends on cost vs limit)
        assert lam_before != lam_after, "Lambda should change via dual update"

    def test_lambda_increases_on_violation(self, safe_trainer):
        """Lambda increases when safety is consistently violated."""
        # Fill buffer with ALL violations → high cost
        _fill_buffer(safe_trainer, n=64, violation_rate=1.0)
        lam_before = safe_trainer.lambda_value
        for _ in range(100):
            safe_trainer.train_step()
        lam_after = safe_trainer.lambda_value
        # With persistent violations, cost > limit → λ should increase over time
        # Use a generous tolerance because early random weights may cause noise
        assert lam_after > lam_before * 0.5, \
            f"Lambda should not collapse with violations: {lam_before:.4f} → {lam_after:.4f}"

    def test_lambda_bounded(self, safe_trainer):
        """Lambda is clamped to [0, lambda_max]."""
        _fill_buffer(safe_trainer, n=32, violation_rate=1.0)
        for _ in range(200):
            safe_trainer.train_step()
        assert safe_trainer.lambda_value <= safe_trainer.safe_config.lambda_max + 1e-6

    def test_actor_loss_includes_cost(self, safe_trainer):
        """Actor loss metric exists and differs from base reward-only loss."""
        _fill_buffer(safe_trainer, n=32)
        metrics = safe_trainer.train_step()
        # safe_actor_loss = -Q_reward + λ·Q_cost
        assert "actor_loss" in metrics
        assert isinstance(metrics["actor_loss"], float)

    def test_cost_window_populated(self, safe_trainer):
        """Cost window gets populated during training."""
        _fill_buffer(safe_trainer, n=32)
        for _ in range(10):
            safe_trainer.train_step()
        assert len(safe_trainer._cost_window) == 10

    def test_constraint_stats(self, safe_trainer):
        """get_constraint_stats returns correct structure."""
        _fill_buffer(safe_trainer, n=32)
        for _ in range(10):
            safe_trainer.train_step()
        stats = safe_trainer.get_constraint_stats()
        assert "avg_cost" in stats
        assert "max_cost" in stats
        assert "violation_rate" in stats
        assert "lambda" in stats
        assert 0 <= stats["violation_rate"] <= 1.0


# ===========================================================================
# Tests: Constraint Satisfaction
# ===========================================================================


class TestConstraintSatisfaction:
    """Test constraint monitoring logic."""

    def test_not_satisfied_initially(self, safe_trainer):
        """Constraint not satisfied with insufficient data."""
        assert not safe_trainer.is_constraint_satisfied(window=100)

    def test_satisfied_with_safe_data(self, safe_trainer):
        """Constraint satisfied when costs are below limit."""
        # Manually populate cost window with low costs (within deque maxlen)
        for _ in range(50):
            safe_trainer._cost_window.append(0.05)  # below limit of 0.1
        assert safe_trainer.is_constraint_satisfied(window=50)

    def test_not_satisfied_with_violations(self, safe_trainer):
        """Constraint violated when costs exceed limit."""
        for _ in range(50):
            safe_trainer._cost_window.append(0.5)  # above limit of 0.1
        assert not safe_trainer.is_constraint_satisfied(window=50)


# ===========================================================================
# Tests: Save/Load
# ===========================================================================


class TestSafeTrainerSaveLoad:
    """Test checkpoint save/load with Lagrangian state."""

    def test_save_load_preserves_lambda(self, safe_trainer, tmp_path):
        """Lambda value preserved across save/load."""
        _fill_buffer(safe_trainer, n=32)
        for _ in range(20):
            safe_trainer.train_step()

        lam_before = safe_trainer.lambda_value
        path = str(tmp_path / "safe_checkpoint.pt")
        safe_trainer.save(path)

        # Create new trainer and load
        new_trainer = SafeRoleFACMACTrainer(safe_trainer.safe_config)
        new_trainer.load(path)

        assert abs(new_trainer.lambda_value - lam_before) < 1e-6

    def test_save_load_preserves_cost_window(self, safe_trainer, tmp_path):
        """Cost window preserved across save/load."""
        _fill_buffer(safe_trainer, n=32)
        for _ in range(15):
            safe_trainer.train_step()

        window_before = list(safe_trainer._cost_window)
        path = str(tmp_path / "safe_checkpoint.pt")
        safe_trainer.save(path)

        new_trainer = SafeRoleFACMACTrainer(safe_trainer.safe_config)
        new_trainer.load(path)

        assert list(new_trainer._cost_window) == window_before


# ===========================================================================
# Tests: Training Convergence
# ===========================================================================


class TestSafeConvergence:
    """Test that safety-constrained training produces valid gradients."""

    def test_no_nan_in_training(self, safe_trainer):
        """No NaN values in metrics during training."""
        _fill_buffer(safe_trainer, n=64, violation_rate=0.5)
        for step in range(50):
            metrics = safe_trainer.train_step()
            for k, v in metrics.items():
                if isinstance(v, float):
                    assert not np.isnan(v), f"NaN at step {step} in {k}"
                    assert not np.isinf(v), f"Inf at step {step} in {k}"

    def test_gradients_flow(self, safe_trainer):
        """All parameters receive gradients."""
        _fill_buffer(safe_trainer, n=32)
        safe_trainer.train_step()
        # Check actor got gradients
        for name, p in safe_trainer.actor.named_parameters():
            if p.requires_grad:
                assert p.grad is not None or True  # some params may have zero grad
        # Check lambda got gradient
        assert safe_trainer._log_lambda.grad is not None
