"""Quick functional test of optimized FACMAC."""
import sys
sys.path.insert(0, '.')
from sfac_facmac import FACMACController, FACMACConfig
import numpy as np

# Test that the new params exist and init works
cfg = FACMACConfig(m_instances=4)
print(f'fast_detect_threshold = {cfg.fast_detect_threshold}')
print(f'ewma_init = {cfg.ewma_init}')

ctrl = FACMACController(cfg)
print(f'Initial EWMA = {ctrl.ewma}')  # should be [0.02, 0.02, 0.02, 0.02]

# Test decide works with raw signal above fast threshold
raw = np.array([0.20, 0.01, 0.02, 0.01])
sizes = np.array([25, 25, 25, 25])
result = ctrl.decide(raw, sizes, epoch=0, T_total=500)
print(f'Detected (raw=0.20 > 0.15): {result["detected_instances"]}')
# Instance 0 should be detected immediately via spike detection
assert result["detected_instances"][0] == True, "FAIL: should detect instance 0"

# Test with signal below threshold
ctrl.reset()
raw2 = np.array([0.05, 0.01, 0.02, 0.01])
result2 = ctrl.decide(raw2, sizes, epoch=0, T_total=500)
print(f'Detected (raw=0.05 < 0.15): {result2["detected_instances"]}')
# Instance 0 should NOT be detected via spike (may still trigger from trust)
# Just verify no crash

print('ALL TESTS PASSED')
