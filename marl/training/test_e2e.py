"""End-to-end test for PyTorch FACMAC training loop."""
import sys
sys.path.insert(0, ".")

import numpy as np
import torch
import tempfile
import os

from marl.training import FACMACTrainer, TrainingConfig, Transition

def main():
    config = TrainingConfig(
        n_agents=4,
        batch_size=16,
        warmup_steps=50,
        buffer_capacity=1000,
        device="cpu",
    )
    trainer = FACMACTrainer(config)

    print(f"Device: {trainer.device}")
    print(f"Actor params: {sum(p.numel() for p in trainer.actor.parameters()):,}")
    print(f"Critic params: {sum(p.numel() for p in trainer.critic.parameters()):,}")
    print(f"Mixer params: {sum(p.numel() for p in trainer.mixer.parameters()):,}")

    # Fill buffer with random transitions
    rng = np.random.default_rng(42)
    for i in range(100):
        t = Transition(
            state=rng.standard_normal(28).astype(np.float32),
            agent_obs=rng.standard_normal((4, 7)).astype(np.float32),
            actions=rng.standard_normal((4, 4)).astype(np.float32),
            reward=float(rng.standard_normal()),
            next_state=rng.standard_normal(28).astype(np.float32),
            next_agent_obs=rng.standard_normal((4, 7)).astype(np.float32),
            done=bool(rng.random() < 0.1),
            n_agents=4,
        )
        trainer.store_transition(t)

    print(f"Buffer size: {len(trainer.buffer)}")
    print(f"Can train: {trainer.can_train()}")

    # Run 20 training steps
    for step in range(20):
        metrics = trainer.train_step()
        if step in (0, 9, 19):
            print(f"  Step {step}: critic_loss={metrics['critic_loss']:.4f}, "
                  f"actor_loss={metrics['actor_loss']:.4f}, "
                  f"mean_q={metrics['mean_q']:.4f}")

    # Test action selection (decentralized execution)
    obs = rng.standard_normal((4, 7)).astype(np.float32)
    actions = trainer.select_actions(obs, explore=False)
    assert actions.shape == (4, 4), f"Action shape: {actions.shape}"
    assert actions.min() >= -1.0 and actions.max() <= 1.0, "Actions out of [-1,1]"
    print(f"Actions (greedy): shape={actions.shape}, range=[{actions.min():.3f}, {actions.max():.3f}]")

    # Test with exploration noise
    actions_noisy = trainer.select_actions(obs, explore=True, noise_scale=0.2)
    assert actions_noisy.shape == (4, 4)
    print(f"Actions (explore): range=[{actions_noisy.min():.3f}, {actions_noisy.max():.3f}]")

    # Test save/load checkpoint
    tmp = tempfile.mktemp(suffix=".pt")
    trainer.save(tmp)
    size_kb = os.path.getsize(tmp) / 1024
    trainer2 = FACMACTrainer(config)
    trainer2.load(tmp)
    os.unlink(tmp)
    print(f"Save/Load: OK ({size_kb:.1f} KB, train_steps={trainer2.train_steps})")

    # Verify QMIX monotonicity preserved after training
    q_vals = torch.randn(16, 4)
    state = torch.randn(16, 28)
    assert trainer.mixer.monotonicity_check(q_vals.clone(), state.clone())
    print("QMIX monotonicity after training: VERIFIED")

    # Test variable n_agents (padding)
    config_large = TrainingConfig(n_agents=10, batch_size=8, warmup_steps=20, buffer_capacity=500, device="cpu")
    trainer_large = FACMACTrainer(config_large)
    for i in range(30):
        na = rng.integers(3, 8)  # variable number of agents
        t = Transition(
            state=rng.standard_normal(28).astype(np.float32),
            agent_obs=rng.standard_normal((na, 7)).astype(np.float32),
            actions=rng.standard_normal((na, 4)).astype(np.float32),
            reward=float(rng.standard_normal()),
            next_state=rng.standard_normal(28).astype(np.float32),
            next_agent_obs=rng.standard_normal((na, 7)).astype(np.float32),
            done=False,
            n_agents=na,
        )
        trainer_large.store_transition(t)
    metrics = trainer_large.train_step()
    print(f"Variable n_agents test: OK (critic_loss={metrics['critic_loss']:.4f})")

    print("\n=== ALL PHASE 2 TESTS PASSED ===")

if __name__ == "__main__":
    main()
