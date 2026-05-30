"""Fast validation of spike detection optimization.

This script simulates ONLY the detection + damage calculation layer
without training FACMAC from scratch. It reuses the exact signal model
from run_e2e_experiments.py and adds the spike detection path.

Purpose: quickly estimate the D(T) and detection_latency improvement
from the fast_detect_threshold optimization before committing to a
full multi-hour training run.
"""
import numpy as np
import json
from pathlib import Path

# Match E2EConfig defaults exactly
N_TOTAL = 100
M_INSTANCES = 4
F_BYZANTINE = 30
ROAM_SWITCH_INTERVAL = 50
T_EPOCHS = 500
EWMA_ALPHA = 0.10
NOISE_HONEST = 0.02
NOISE_BYZ_DORMANT = 0.05
SIGNAL_BYZ_ACTIVE = 0.60
SIGNAL_NOISE_STD = 0.15
EQUIVOCATION_PROB_ACTIVE = 0.25
EQUIVOCATION_PROB_DORMANT = 0.02
FAST_DETECT_THRESHOLD = 0.06
EWMA_INIT = 0.02
# CUSUM params
CUSUM_H = 0.25
CUSUM_DRIFT = 0.03
# Trust estimator detection threshold
TRUST_THRESHOLD = 0.356
# Equivocation threshold  
EQUIV_THRESHOLD = 0.075

SEEDS = [7, 13, 42, 97, 137]


def simulate_detection(seed, use_spike=True, use_ewma_init=True):
    """Simulate detection for a single seed.
    
    Models three detection channels:
    1. Trust estimator (EWMA + threshold) - models trained trust net behavior
    2. Equivocation fast-path
    3. Spike detection (NEW)
    """
    rng = np.random.default_rng(seed)
    
    # Assign nodes to instances and mark Byzantine
    is_byzantine = np.zeros(N_TOTAL, dtype=bool)
    is_byzantine[:F_BYZANTINE] = True
    instance_of = rng.integers(0, M_INSTANCES, size=N_TOTAL)
    
    # Count Byzantine per instance
    byz_per_inst = np.zeros(M_INSTANCES, dtype=int)
    for j in range(M_INSTANCES):
        byz_per_inst[j] = np.sum(is_byzantine & (instance_of == j))
    
    # EWMA state
    ewma = np.full(M_INSTANCES, EWMA_INIT if use_ewma_init else 0.0)
    ewma_equiv = np.zeros(M_INSTANCES)
    
    # CUSUM state (for comparison)
    cusum_pos = np.zeros(M_INSTANCES)
    cusum_neg = np.zeros(M_INSTANCES)
    
    # Metrics
    damage_octopus = 0
    damage_cusum = 0
    detection_latency_sum_oct = 0
    detection_latency_sum_cus = 0
    detection_events_oct = 0
    detection_events_cus = 0
    
    # Attack tracking
    attack_start_oct = {}
    attack_detected_oct = {}
    attack_start_cus = {}
    attack_detected_cus = {}
    
    target_instance = 0
    switch_counter = 0
    
    for t in range(T_EPOCHS):
        # Generate signals (matching run_e2e_experiments.py exactly)
        signals = np.zeros(N_TOTAL)
        equiv_signals = np.zeros(N_TOTAL)
        active_byz = np.zeros(N_TOTAL, dtype=bool)
        
        for nid in range(N_TOTAL):
            if is_byzantine[nid] and instance_of[nid] == target_instance:
                active_byz[nid] = True
                signals[nid] = np.clip(
                    SIGNAL_BYZ_ACTIVE + rng.normal(0, SIGNAL_NOISE_STD), 0, 1)
                equiv_signals[nid] = float(rng.random() < EQUIVOCATION_PROB_ACTIVE)
            elif is_byzantine[nid]:
                signals[nid] = np.clip(rng.normal(NOISE_BYZ_DORMANT, 0.06), 0, 1)
                equiv_signals[nid] = float(rng.random() < EQUIVOCATION_PROB_DORMANT)
            else:
                signals[nid] = np.clip(rng.normal(NOISE_HONEST, 0.04), 0, 1)
                equiv_signals[nid] = 0.0
        
        # Per-instance aggregation
        per_inst_signal = np.zeros(M_INSTANCES)
        per_inst_equiv = np.zeros(M_INSTANCES)
        per_inst_byz = np.zeros(M_INSTANCES, dtype=int)
        
        for j in range(M_INSTANCES):
            mask = instance_of == j
            per_inst_signal[j] = np.mean(signals[mask])
            per_inst_equiv[j] = np.mean(equiv_signals[mask])
            per_inst_byz[j] = np.sum(active_byz[mask])
        
        # === Octopus Detection (EWMA + equivocation + spike) ===
        # Update EWMA (asymmetric)
        alpha = np.where(per_inst_signal >= ewma, EWMA_ALPHA, 
                        min(EWMA_ALPHA * 2.5, 0.35))
        ewma = (1 - alpha) * ewma + alpha * per_inst_signal
        # Update equivocation EWMA
        alpha_e = np.where(per_inst_equiv >= ewma_equiv, EWMA_ALPHA,
                          min(EWMA_ALPHA * 2.5, 0.35))
        ewma_equiv = (1 - alpha_e) * ewma_equiv + alpha_e * per_inst_equiv
        
        # Detection channels:
        # 1. Trust estimator (model as EWMA > threshold, conservative approximation)
        #    In practice, the trained sigmoid is more accurate, but this models the
        #    EWMA-based detection behavior
        detected_oct = ewma > TRUST_THRESHOLD
        # 2. Equivocation fast-path
        detected_oct = detected_oct | (per_inst_equiv > EQUIV_THRESHOLD)
        # 3. Spike detection (NEW)
        if use_spike:
            detected_oct = detected_oct | (per_inst_signal > FAST_DETECT_THRESHOLD)
        
        # === CUSUM Detection ===
        cusum_pos = np.maximum(0, cusum_pos + per_inst_signal - CUSUM_DRIFT)
        cusum_neg = np.maximum(0, cusum_neg - per_inst_signal + CUSUM_DRIFT)
        scores = np.maximum(cusum_pos, cusum_neg)
        detected_cus = scores > CUSUM_H
        cusum_pos[detected_cus] = 0
        cusum_neg[detected_cus] = 0
        
        # === Damage calculation ===
        for j in range(M_INSTANCES):
            if per_inst_byz[j] > 0 and not detected_oct[j]:
                damage_octopus += per_inst_byz[j]
            if per_inst_byz[j] > 0 and not detected_cus[j]:
                damage_cusum += per_inst_byz[j]
        
        # === Detection latency tracking (Octopus) ===
        for j in range(M_INSTANCES):
            if per_inst_byz[j] > 0 and j not in attack_start_oct:
                attack_start_oct[j] = t
            if per_inst_byz[j] > 0 and detected_oct[j] and j not in attack_detected_oct:
                attack_detected_oct[j] = t
                if j in attack_start_oct:
                    detection_latency_sum_oct += (t - attack_start_oct[j])
                    detection_events_oct += 1
            if per_inst_byz[j] == 0 and j in attack_start_oct:
                del attack_start_oct[j]
                attack_detected_oct.pop(j, None)
        
        # === Detection latency tracking (CUSUM) ===
        for j in range(M_INSTANCES):
            if per_inst_byz[j] > 0 and j not in attack_start_cus:
                attack_start_cus[j] = t
            if per_inst_byz[j] > 0 and detected_cus[j] and j not in attack_detected_cus:
                attack_detected_cus[j] = t
                if j in attack_start_cus:
                    detection_latency_sum_cus += (t - attack_start_cus[j])
                    detection_events_cus += 1
            if per_inst_byz[j] == 0 and j in attack_start_cus:
                del attack_start_cus[j]
                attack_detected_cus.pop(j, None)
        
        # Advance roaming
        switch_counter += 1
        if switch_counter >= ROAM_SWITCH_INTERVAL:
            target_instance = (target_instance + 1) % M_INSTANCES
            switch_counter = 0
    
    lat_oct = (detection_latency_sum_oct / max(detection_events_oct, 1)
               if detection_events_oct > 0 else T_EPOCHS)
    lat_cus = (detection_latency_sum_cus / max(detection_events_cus, 1)
               if detection_events_cus > 0 else T_EPOCHS)
    
    return {
        "seed": seed,
        "byz_per_inst": byz_per_inst.tolist(),
        "octopus_D_T": int(damage_octopus),
        "octopus_latency": round(lat_oct, 2),
        "cusum_D_T": int(damage_cusum),
        "cusum_latency": round(lat_cus, 2),
    }


def main():
    print("=" * 70)
    print("FAST VALIDATION: Spike Detection Optimization Impact")
    print("=" * 70)
    print(f"Config: fast_detect_threshold={FAST_DETECT_THRESHOLD}, ewma_init={EWMA_INIT}")
    print(f"Seeds: {SEEDS}")
    print()
    
    # Run WITH spike detection (optimized)
    print("--- WITH spike detection + EWMA warm-start (OPTIMIZED) ---")
    results_opt = []
    for s in SEEDS:
        r = simulate_detection(s, use_spike=True, use_ewma_init=True)
        results_opt.append(r)
        print(f"  Seed {s:>3}: Octopus D(T)={r['octopus_D_T']:>5}, "
              f"latency={r['octopus_latency']:.2f} | "
              f"CUSUM D(T)={r['cusum_D_T']:>5}, latency={r['cusum_latency']:.2f} | "
              f"byz_dist={r['byz_per_inst']}")
    
    print()
    
    # Run WITHOUT spike detection (original behavior approximation)
    print("--- WITHOUT spike detection (ORIGINAL approximation) ---")
    results_orig = []
    for s in SEEDS:
        r = simulate_detection(s, use_spike=False, use_ewma_init=False)
        results_orig.append(r)
        print(f"  Seed {s:>3}: Octopus D(T)={r['octopus_D_T']:>5}, "
              f"latency={r['octopus_latency']:.2f} | "
              f"CUSUM D(T)={r['cusum_D_T']:>5}, latency={r['cusum_latency']:.2f}")
    
    print()
    print("--- SUMMARY ---")
    d_opt = [r["octopus_D_T"] for r in results_opt]
    d_orig = [r["octopus_D_T"] for r in results_orig]
    d_cusum = [r["cusum_D_T"] for r in results_opt]
    lat_opt = [r["octopus_latency"] for r in results_opt]
    lat_orig = [r["octopus_latency"] for r in results_orig]
    lat_cusum = [r["cusum_latency"] for r in results_opt]
    
    print(f"  Optimized Octopus:  D(T) mean={np.mean(d_opt):.1f}±{np.std(d_opt):.1f}, "
          f"latency={np.mean(lat_opt):.2f}±{np.std(lat_opt):.2f}")
    print(f"  Original Octopus:   D(T) mean={np.mean(d_orig):.1f}±{np.std(d_orig):.1f}, "
          f"latency={np.mean(lat_orig):.2f}±{np.std(lat_orig):.2f}")
    print(f"  CUSUM:              D(T) mean={np.mean(d_cusum):.1f}±{np.std(d_cusum):.1f}, "
          f"latency={np.mean(lat_cusum):.2f}±{np.std(lat_cusum):.2f}")
    
    print()
    print("  Improvement: D(T) reduced by {:.0f}% (opt vs orig)".format(
        (1 - np.mean(d_opt) / max(np.mean(d_orig), 1)) * 100))
    print("  Improvement: latency reduced by {:.0f}% (opt vs orig)".format(
        (1 - np.mean(lat_opt) / max(np.mean(lat_orig), 0.01)) * 100))
    
    # Save results
    output = {
        "optimized": results_opt,
        "original_approx": results_orig,
        "summary": {
            "opt_D_T_mean": round(float(np.mean(d_opt)), 1),
            "opt_D_T_std": round(float(np.std(d_opt)), 1),
            "opt_latency_mean": round(float(np.mean(lat_opt)), 2),
            "orig_D_T_mean": round(float(np.mean(d_orig)), 1),
            "cusum_D_T_mean": round(float(np.mean(d_cusum)), 1),
            "cusum_latency_mean": round(float(np.mean(lat_cusum)), 2),
        }
    }
    out_path = Path("experiments/results/spike_detection_validation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
