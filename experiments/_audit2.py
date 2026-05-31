import json, os
os.chdir(r'd:\Alex\Papers\NDSS 2027_SUBMISSION\figures\Evolv-BFT')
d=json.load(open('appendix_experiment_data.json'))

print('=== exp_d_detection_latency ===')
for k,v in list(d['exp_d_detection_latency'].items())[:3]:
    s = str(v)
    if len(s) > 200:
        s = s[:200] + '...'
    print(f'  {k}: {s}')
print('  keys count:', len(d['exp_d_detection_latency']))
print()

print('=== exp_a_gbc_overhead ===')
for k,v in d['exp_a_gbc_overhead'].items():
    print(f'  m={k}: {v}')
print()

print('=== exp_c_ctde_scaling ===')
for k,v in d['exp_c_ctde_scaling'].items():
    print(f'  {k}: {v}')
print()

os.chdir(r'd:\Alex\Papers\NDSS 2027_SUBMISSION\figures\v2x')
m=json.load(open('v2x_multiseed_aggregate.json'))
print('=== v2x_multiseed_aggregate top keys ===')
print(list(m.keys()))
print()
# pick first non-trivial subkey
for k in list(m.keys())[:2]:
    v = m[k]
    if isinstance(v, dict):
        print(f'{k}: keys={list(v.keys())[:6]}')
        for sk, sv in list(v.items())[:2]:
            print(f'  {sk}: {str(sv)[:200]}')
