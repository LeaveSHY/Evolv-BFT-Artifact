"""Diagnostic: distribution of actor outputs and fault signal."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_role_ablation import build_controller, RoamingBFTEnv, SimConfig
import numpy as np

cfg = SimConfig(n_nodes=100, m_instances=4, f_fraction=0.3, signal_byz_active=0.65,
                noise_byz_dormant=0.05, noise_honest=0.02, signal_noise_std=0.15,
                roam_kappa_init=50, equiv_prob_active=0.25, equiv_prob_dormant=0.02)
env = RoamingBFTEnv(cfg, seed=42)
ctrl = build_controller(42, cfg, {})
ctrl.train_mode = True

acts, sigs, fprobs, true_byz_list = [], [], [], []
for ep in range(1000):
    raw, eq, true_byz, _ = env.step()
    out = ctrl.decide(raw, env.instance_sizes, ep, 1000, equiv_signal=eq)
    if ep >= 800:
        acts.append(ctrl._curr_actions.copy())
        sigs.append(raw.copy())
        fprobs.append(np.asarray(out["detection_probs"]).copy())
        true_byz_list.append(true_byz.copy())
    ctrl.store_transition(np.zeros(4, dtype=np.float32), done=(ep == 999))
    ctrl.train_step()

acts = np.stack(acts)
sigs = np.stack(sigs)
fprobs = np.stack(fprobs)
true_byz_arr = np.stack(true_byz_list)
print(f"action[0] (evict) range=[{acts[:,:,0].min():.3f}, {acts[:,:,0].max():.3f}]  mean={acts[:,:,0].mean():.3f}")
print(f"action[1] (rotate) range=[{acts[:,:,1].min():.3f}, {acts[:,:,1].max():.3f}]  mean={acts[:,:,1].mean():.3f}")
print(f"action[2] (param) range=[{acts[:,:,2].min():.3f}, {acts[:,:,2].max():.3f}]  mean={acts[:,:,2].mean():.3f}")
print(f"fault signal range=[{sigs.min():.3f}, {sigs.max():.3f}]  mean={sigs.mean():.3f}")
print(f"frac signal>0.5: {(sigs>0.5).mean():.3f}")
print(f"frac signal>0.4: {(sigs>0.4).mean():.3f}")
print(f"frac signal>0.3: {(sigs>0.3).mean():.3f}")
print(f"frac action[1]>0.5: {(acts[:,:,1]>0.5).mean():.3f}")
print(f"frac action[1]>0.3: {(acts[:,:,1]>0.3).mean():.3f}")
print(f"frac action[1]>0.1: {(acts[:,:,1]>0.1).mean():.3f}")
print(f"frac action[0]>0.5: {(acts[:,:,0]>0.5).mean():.3f}")
print(f"frac action[0]>0.3: {(acts[:,:,0]>0.3).mean():.3f}")

# correlation
sig_flat = sigs.flatten()
act1_flat = acts[:,:,1].flatten()
act0_flat = acts[:,:,0].flatten()
print(f"corr(signal, action[1]): {np.corrcoef(sig_flat, act1_flat)[0,1]:.3f}")
print(f"corr(signal, action[0]): {np.corrcoef(sig_flat, act0_flat)[0,1]:.3f}")

print()
print("=== fault_probs (post-sharpening) ===")
print(f"fault_probs range=[{fprobs.min():.3f}, {fprobs.max():.3f}]  mean={fprobs.mean():.3f}")
print(f"frac fault_probs>0.7 (theta_high): {(fprobs>0.7).mean():.3f}")
print(f"frac fault_probs>0.5: {(fprobs>0.5).mean():.3f}")
print(f"frac fault_probs>0.4: {(fprobs>0.4).mean():.3f}")
print(f"frac fault_probs>0.3: {(fprobs>0.3).mean():.3f}")
fp_flat = fprobs.flatten()
print(f"corr(fault_probs, action[1]): {np.corrcoef(fp_flat, act1_flat)[0,1]:.3f}")
print(f"corr(fault_probs, action[0]): {np.corrcoef(fp_flat, act0_flat)[0,1]:.3f}")

print()
print("=== true_byz vs fault_probs alignment ===")
tb_flat = true_byz_arr.flatten().astype(np.int64)
print(f"true_byz=1 count: {tb_flat.sum()}  total: {len(tb_flat)}  rate: {tb_flat.mean():.3f}")
print(f"mean fault_probs when true_byz=1: {fp_flat[tb_flat==1].mean():.3f}")
print(f"mean fault_probs when true_byz=0: {fp_flat[tb_flat==0].mean():.3f}")
print(f"mean signal when true_byz=1: {sig_flat[tb_flat==1].mean():.3f}")
print(f"mean signal when true_byz=0: {sig_flat[tb_flat==0].mean():.3f}")
print(f"mean action[1] when true_byz=1: {act1_flat[tb_flat==1].mean():.3f}")
print(f"mean action[1] when true_byz=0: {act1_flat[tb_flat==0].mean():.3f}")
