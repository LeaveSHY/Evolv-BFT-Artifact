import json, statistics as S
files = [('postfixA8', 'results/role_ablation_postfixA8/role_ablation.json'),
         ('allorg', 'results/role_ablation_allorg/role_ablation.json')]
for label, f in files:
    d = json.load(open(f))
    for ab, summ in d['summary'].items():
        sofs=[r['sof'] for r in d['per_seed'] if r['ablation']==ab]
        fofs=[r['fof'] for r in d['per_seed'] if r['ablation']==ab]
        ofs=[r['of']  for r in d['per_seed'] if r['ablation']==ab]
        f1_m = summ['final_f1_mean']; f1_s = summ['final_f1_std']
        if len(sofs) > 1:
            print(f"{label:10s} {ab:12s} F1={f1_m:.4f}\u00b1{f1_s:.4f}  sof={S.mean(sofs):.3f}\u00b1{S.stdev(sofs):.3f}  fof={S.mean(fofs):.3f}\u00b1{S.stdev(fofs):.3f}  of={S.mean(ofs):.3f}\u00b1{S.stdev(ofs):.3f}")
