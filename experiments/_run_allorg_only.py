"""One-off harness: only run no_all_org ablation (others already in postfixA8)."""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'experiments')
import run_role_ablation as r
r.ABLATIONS = {
    'full': {'lambda_5': 5.0, 'lambda_6': 5.0},  # baseline for delta
    'no_all_org': {'lambda_5': 0.0, 'lambda_6': 0.0},
}
sys.argv = [
    'run_role_ablation.py',
    '--seeds', '7', '17', '27', '37', '47', '57', '67', '77',
    '--episodes', '6000',
    '--eval-episodes', '1500',
    '--output', 'results/role_ablation_allorg',
]
r.main()
