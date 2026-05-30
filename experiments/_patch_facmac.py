"""Patch sfac_facmac.py to add fast-path detection and EWMA warm-start."""
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
target = os.path.join(script_dir, "sfac_facmac.py")

with open(target, "r") as f:
    content = f.read()

# 1. Add fast_detect_threshold and ewma_init to FACMACConfig
old1 = """    # Detection threshold: slightly above BFT fault tolerance bound (1/3)
    # to reduce false-positive detections from estimation noise
    detection_threshold: float = 0.356"""
new1 = """    # Detection threshold: slightly above BFT fault tolerance bound (1/3)
    # to reduce false-positive detections from estimation noise
    detection_threshold: float = 0.356
    # Fast-path detection: raw signal spike threshold for instant detection
    # (bypasses EWMA smoothing delay for first-epoch attack response)
    fast_detect_threshold: float = 0.25
    # EWMA warm-start: initialize to honest noise floor to reduce convergence time
    ewma_init: float = 0.02"""
assert old1 in content, "ERROR: old1 not found"
content = content.replace(old1, new1)

# 2. EWMA initialization warm-start
old2 = """        # ── Trust estimation state (EWMA) ──
        self.ewma = np.zeros(cfg.m_instances)"""
new2 = """        # ── Trust estimation state (EWMA) ──
        # Warm-start to honest noise floor for faster convergence on attack onset
        self.ewma = np.full(cfg.m_instances, cfg.ewma_init)"""
assert old2 in content, "ERROR: old2 not found"
content = content.replace(old2, new2)

# 3. Reset method: EWMA warm-start
old3 = """        self.ewma[:] = 0"""
new3 = """        self.ewma[:] = self.cfg.ewma_init"""
assert old3 in content, "ERROR: old3 not found"
content = content.replace(old3, new3)

# 4. Add fast-path spike detection in decide method
old4 = """        # Fast detection via raw equivocation channel (\xa7III-B).
        # Equivocation (double-voting) is cryptographically verifiable and
        # never produced by honest nodes. When per-instance equivocation
        # exceeds the dormant noise floor (>= 2 equivocating nodes out of
        # ~25 per instance), this is strong Byzantine evidence that enables
        # detection before the EWMA trust estimate fully converges.
        if equiv_signal is not None:
            equivocation_fast = equiv_signal > 0.075
            detected = detected | equivocation_fast"""
new4 = """        # Fast detection via raw equivocation channel (\xa7III-B).
        # Equivocation (double-voting) is cryptographically verifiable and
        # never produced by honest nodes. When per-instance equivocation
        # exceeds the dormant noise floor (>= 2 equivocating nodes out of
        # ~25 per instance), this is strong Byzantine evidence that enables
        # detection before the EWMA trust estimate fully converges.
        if equiv_signal is not None:
            equivocation_fast = equiv_signal > 0.075
            detected = detected | equivocation_fast

        # Fast-path spike detection (\xa7III-D): raw signal exceeding the fast
        # threshold triggers immediate detection without waiting for EWMA
        # convergence. This provides CUSUM-like first-epoch responsiveness
        # while the EWMA-based trust estimation handles long-term management.
        # The threshold is set below BFT fault tolerance (1/3) to catch
        # attacks with high recall, since the safety filter gates any
        # false-positive reconfiguration actions downstream.
        spike_detected = raw_signal > self.cfg.fast_detect_threshold
        detected = detected | spike_detected"""
assert old4 in content, "ERROR: old4 not found"
content = content.replace(old4, new4)

with open(target, "w") as f:
    f.write(content)

print("SUCCESS: All 4 patches applied to sfac_facmac.py")
