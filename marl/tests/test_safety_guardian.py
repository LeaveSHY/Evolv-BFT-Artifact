"""Tests for Safety Guardian Active Policy."""
import numpy as np
import pytest
import torch

from marl.networks.safety_guardian import (
    SAFETY_MOD_DIM,
    SafetyGuardianConfig,
    SafetyGuardianNetwork,
    SafetyGuardianPolicy,
    SafetyModulation,
    apply_safety_modulation,
)


# ===========================================================================
# Tests: SafetyModulation
# ===========================================================================


class TestSafetyModulation:
    def test_default_is_conservative(self):
        mod = SafetyModulation.default()
        assert mod.risk_tolerance == 0.3
        assert mod.action_scale == 0.7
        assert mod.reconfig_threshold == 0.7

    def test_roundtrip_array(self):
        mod = SafetyModulation(risk_tolerance=0.5, action_scale=0.8, reconfig_threshold=0.6)
        arr = mod.to_array()
        assert arr.shape == (3,)
        mod2 = SafetyModulation.from_array(arr)
        assert abs(mod2.risk_tolerance - 0.5) < 1e-6
        assert abs(mod2.action_scale - 0.8) < 1e-6
        assert abs(mod2.reconfig_threshold - 0.6) < 1e-6


# ===========================================================================
# Tests: SafetyGuardianNetwork
# ===========================================================================


class TestSafetyGuardianNetwork:
    def test_output_shape(self):
        net = SafetyGuardianNetwork(state_dim=28)
        state = torch.randn(4, 28)
        out = net(state)
        assert out.shape == (4, SAFETY_MOD_DIM)

    def test_modulation_ranges(self):
        net = SafetyGuardianNetwork(state_dim=28)
        state = torch.randn(100, 28)  # large batch to test range
        mod = net.get_modulation(state)
        # risk_tolerance in [0, 1]
        assert mod[:, 0].min() >= 0.0
        assert mod[:, 0].max() <= 1.0
        # action_scale in [0.5, 1.0]
        assert mod[:, 1].min() >= 0.5 - 1e-6
        assert mod[:, 1].max() <= 1.0 + 1e-6
        # reconfig_threshold in [0, 1]
        assert mod[:, 2].min() >= 0.0
        assert mod[:, 2].max() <= 1.0

    def test_param_count(self):
        net = SafetyGuardianNetwork(state_dim=28, hidden_dims=(128, 64))
        n_params = sum(p.numel() for p in net.parameters())
        # 28*128 + 128 + 128*64 + 64 + 64*3 + 3 = 3584+128+8192+64+192+3 = 12163
        assert n_params > 10000


# ===========================================================================
# Tests: apply_safety_modulation
# ===========================================================================


class TestApplySafetyModulation:
    def test_conservative_scales_down(self):
        """Conservative modulation should reduce action magnitudes."""
        base = np.array([[0.5, 0.8, -0.6, 0.9],
                         [0.7, -0.5, 0.4, -0.8]], dtype=np.float32)
        mod = SafetyModulation(risk_tolerance=0.0, action_scale=0.5, reconfig_threshold=0.9)
        result = apply_safety_modulation(base, mod)
        # action_scale=0.5 → params halved, then clamped to [−0.3, 0.3]
        assert np.all(np.abs(result[:, 1:]) <= 0.3 + 1e-6)

    def test_permissive_preserves_actions(self):
        """Permissive modulation should preserve original actions."""
        base = np.array([[0.5, 0.3, -0.2, 0.1]], dtype=np.float32)
        mod = SafetyModulation(risk_tolerance=1.0, action_scale=1.0, reconfig_threshold=0.0)
        result = apply_safety_modulation(base, mod)
        # With scale=1.0 and tolerance=1.0, actions should be preserved
        np.testing.assert_allclose(result[:, 1:], base[:, 1:], atol=1e-6)

    def test_reconfig_threshold_gates(self):
        """High threshold blocks low-confidence detections."""
        base = np.array([[0.4, 0.1, 0.1, 0.1],   # detection=0.4
                         [0.8, 0.1, 0.1, 0.1]], dtype=np.float32)  # detection=0.8
        mod = SafetyModulation(risk_tolerance=1.0, action_scale=1.0, reconfig_threshold=0.6)
        result = apply_safety_modulation(base, mod)
        # Agent 0: 0.4 < 0.6 threshold → zeroed
        assert result[0, 0] == 0.0
        # Agent 1: 0.8 > 0.6 threshold → preserved
        assert result[1, 0] == 0.8

    def test_shape_preserved(self):
        base = np.random.randn(10, 4).astype(np.float32)
        mod = SafetyModulation.default()
        result = apply_safety_modulation(base, mod)
        assert result.shape == base.shape


# ===========================================================================
# Tests: SafetyGuardianPolicy
# ===========================================================================


class TestSafetyGuardianPolicy:
    @pytest.fixture
    def policy(self):
        return SafetyGuardianPolicy(SafetyGuardianConfig(
            state_dim=28, hidden_dims=(64, 32), lr=1e-3, device="cpu"
        ))

    def test_select_modulation(self, policy):
        state = np.random.randn(28).astype(np.float32)
        mod = policy.select_modulation(state)
        assert 0.0 <= mod.risk_tolerance <= 1.0
        assert 0.5 <= mod.action_scale <= 1.0
        assert 0.0 <= mod.reconfig_threshold <= 1.0

    def test_explore_adds_noise(self, policy):
        state = np.random.randn(28).astype(np.float32)
        # Multiple explores should vary
        mods = [policy.select_modulation(state, explore=True) for _ in range(20)]
        risk_vals = [m.risk_tolerance for m in mods]
        # With noise, should have some variation
        assert max(risk_vals) - min(risk_vals) > 0.01

    def test_train_step_returns_none_when_empty(self, policy):
        result = policy.train_step(batch_size=32)
        assert result is None

    def test_train_step_after_experience(self, policy):
        # Store enough experiences
        for _ in range(50):
            policy.store_experience(
                state=np.random.randn(28).astype(np.float32),
                modulation=np.array([0.5, 0.7, 0.6], dtype=np.float32),
                safety_reward=float(np.random.uniform(-1, 1)),
                next_state=np.random.randn(28).astype(np.float32),
                done=False,
            )
        result = policy.train_step(batch_size=16)
        assert result is not None
        assert "safety_guardian_loss" in result
        assert "mean_action_scale" in result

    def test_no_nan_in_training(self, policy):
        for _ in range(100):
            policy.store_experience(
                state=np.random.randn(28).astype(np.float32),
                modulation=np.array([0.5, 0.7, 0.6], dtype=np.float32),
                safety_reward=float(np.random.uniform(-2, 1)),
                next_state=np.random.randn(28).astype(np.float32),
                done=np.random.rand() < 0.1,
            )
        for _ in range(30):
            result = policy.train_step(batch_size=16)
            if result:
                for k, v in result.items():
                    assert not np.isnan(v), f"NaN in {k}"

    def test_save_load(self, policy, tmp_path):
        for _ in range(50):
            policy.store_experience(
                state=np.random.randn(28).astype(np.float32),
                modulation=np.array([0.5, 0.7, 0.6], dtype=np.float32),
                safety_reward=0.5,
                next_state=np.random.randn(28).astype(np.float32),
                done=False,
            )
        policy.train_step(batch_size=16)
        path = str(tmp_path / "guardian.pt")
        policy.save(path)

        new_policy = SafetyGuardianPolicy(policy.config)
        new_policy.load(path)
        assert new_policy.train_steps == policy.train_steps

    def test_param_count(self, policy):
        assert policy.param_count() > 0
