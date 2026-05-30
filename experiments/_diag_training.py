"""Diagnose whether actor/trust networks are actually training."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from run_role_ablation import SimConfig, RoamingBFTEnv, build_controller

cfg = SimConfig()
env = RoamingBFTEnv(cfg, seed=1)
ctrl = build_controller(seed=1, sim_cfg=cfg, overrides={
    "lambda_5": 5.0, "lambda_6": 5.0,
})

def actor_norm():
    return sum(p.data.norm().item() for p in ctrl.actor.parameters())

def trust_norm():
    return sum(p.data.norm().item() for p in ctrl.trust_estimator.parameters())

inst_sizes = np.full(cfg.m_instances, cfg.n_nodes // cfg.m_instances, dtype=np.int64)
T = 2000
last_act_means = []

print(f"BEFORE: actor_norm={actor_norm():.4f}  trust_norm={trust_norm():.4f}  buf={len(ctrl.buffer)}  train_steps={ctrl.train_step_count}")

for ep in range(T):
    raw, equiv, true_byz, _ = env.step()
    out = ctrl.decide(raw, inst_sizes, ep, T, equiv_signal=equiv)
    ctrl.store_transition(np.zeros(cfg.m_instances, dtype=np.float32), done=(ep==T-1))
    ctrl.train_step()
    if ep == 300 or ep == 1000 or ep == T-1:
        a = ctrl._curr_pre_mask_actions
        print(f"ep={ep:5d}: actor_norm={actor_norm():.4f}  trust_norm={trust_norm():.4f}  "
              f"buf={len(ctrl.buffer)}  train_steps={ctrl.train_step_count}  "
              f"pre_mask_act[1] mean={a[:,1].mean():.3f} std={a[:,1].std():.3f}")
