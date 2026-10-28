"""Independently check the saved default study and its documentation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
from src.simulation import read_csv


def near(actual, expected, label):
    if not math.isclose(float(actual),float(expected),rel_tol=1e-10,abs_tol=1e-8):
        raise AssertionError(f'{label}: {actual} != {expected}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest', action='store_true',
                        help='record SHA256 checksums after all other checks pass')
    args = parser.parse_args()
    config = json.loads(Path('config.json').read_text(encoding='utf-8'))
    expected = {
        'comparison': (config['means'],config['comparison_policies'],config['horizon']),
        'epsilon': (config['means'],[f'epsilon_{e}' for e in config['epsilons']],config['horizon']),
        'gap_easy': (config['gap_environments']['easy'],config['comparison_policies'],config['horizon']),
        'gap_difficult': (config['gap_environments']['difficult'],config['comparison_policies'],config['horizon']),
        'scaling': (config['means'],config['comparison_policies'],max(config['scaling_horizons']))}
    n = config['trials']
    all_raw = {}
    checkpoints_checked = 0
    for name,(means,policies,horizon) in expected.items():
        meta = json.loads(Path(f'results/{name}_metadata.json').read_text(encoding='utf-8'))
        assert meta['config']==config, f'{name}: default configuration mismatch'
        raw = read_csv(f'results/data/{name}_trials.csv')
        all_raw[name] = raw
        curves = read_csv(f'results/data/{name}_curves.csv')
        counts = read_csv(f'results/data/{name}_counts.csv')
        assert set(r['policy'] for r in raw)==set(policies)
        assert len(counts)==len(policies)*n*len(means)
        count_keys = {(r['policy'],int(r['trial']),int(r['arm'])) for r in counts}
        assert len(count_keys)==len(counts), f'{name}: duplicate count keys'
        assert len({(r['policy'],r['t'],r['metric']) for r in curves})==len(curves)
        gaps = max(means)-np.array(means)
        for policy in policies:
            selected = [r for r in raw if r['policy']==policy]
            times = sorted({int(r['t']) for r in selected})
            assert times[-1]==horizon
            assert len(selected)==n*len(times)
            for t in times:
                rows = [r for r in selected if int(r['t'])==t]
                assert {int(r['trial']) for r in rows}==set(range(n))
                for metric in ['reward','pseudo_regret','optimal_fraction','recent_optimal_fraction']:
                    values = np.array([float(r[metric]) for r in rows])
                    assert np.all(values>=0)
                    assert np.all(values<= (1 if 'fraction' in metric else t)+1e-8)
                    record = next(r for r in curves if r['policy']==policy and int(r['t'])==t and r['metric']==metric)
                    avg = math.fsum(values)/n
                    se = math.sqrt(math.fsum((float(x)-avg)**2 for x in values)/(n-1)/n)
                    for key, value in [('mean',avg),('se',se),('ci_low',avg-1.96*se),('ci_high',avg+1.96*se)]:
                        near(record[key],value,f'{name}/{policy}/{t}/{metric}/{key}')
                    checkpoints_checked += 1
            for j in range(n):
                trial = sorted([r for r in selected if int(r['trial'])==j],key=lambda r:int(r['t']))
                terminal = trial[-1]
                arm_rows = sorted([r for r in counts if r['policy']==policy and int(r['trial'])==j],key=lambda r:int(r['arm']))
                pulls = np.array([int(r['pulls']) for r in arm_rows])
                assert pulls.sum()==horizon and np.all(pulls>=0)
                if policy!='random' and horizon>=len(means):
                    assert np.all(pulls>=1)
                near(terminal['pseudo_regret'],pulls@gaps,f'{name}/{policy}/{j}: regret identity')
                near(terminal['optimal_fraction'],pulls[gaps==0].sum()/horizon,f'{name}: optimal fraction')
                for metric in ['reward','pseudo_regret']:
                    assert np.all(np.diff([float(r[metric]) for r in trial])>=-1e-8)
            if policy=='random':
                final = [float(r['pseudo_regret']) for r in selected if int(r['t'])==horizon]
                se = np.std(final,ddof=1)/math.sqrt(n)
                assert abs(np.mean(final)-horizon*gaps.mean()) <= 6*se+1e-8
        print(f'PASS {name}: {n} independent trials, counts, regrets, and checkpoint intervals')
    h = read_csv('results/data/hoeffding.csv')
    hc = config['hoeffding']
    expected_cells = {(mu,size,d) for mu in hc['means'] for size in hc['sample_sizes'] for d in hc['deltas']}
    assert {(float(r['mu']),int(r['n']),float(r['delta'])) for r in h}==expected_cells
    assert len(h)==len(expected_cells)
    for r in h:
        mu,size,delta = float(r['mu']),int(r['n']),float(r['delta'])
        m,events = int(r['replications']),int(r['events'])
        assert m==hc['replications'] and 0<=events<=m
        near(r['estimate'],events/m,'Hoeffding empirical probability')
        bound = min(1,2*math.exp(-2*size*delta**2))
        near(r['bound'],bound,'Hoeffding bound')
        # Independent direct binomial calculation (rather than log-gamma).
        exact = math.fsum(math.comb(size,k)*mu**k*(1-mu)**(size-k)
            for k in range(size+1) if abs(k/size-mu)>=delta-1e-12)
        near(r['exact_probability'],exact,'exact binomial probability')
        assert exact<=bound+1e-10
        assert abs(events/m-exact)<=6*math.sqrt(exact*(1-exact)/m)+6/m
        assert float(r['ci_low'])<=events/m<=float(r['ci_high'])
    print(f'PASS Hoeffding: all {len(h)} cells, exact binomial tails and Monte Carlo tolerances')
    for r in read_csv('results/data/paired_differences.csv'):
        raw = all_raw[r['experiment']]
        terminal = max(int(s['t']) for s in raw)
        groups = {p:{int(s['trial']):float(s['pseudo_regret']) for s in raw if s['policy']==p and int(s['t'])==terminal}
                  for p in ['ucb1','epsilon_0.1']}
        differences = np.array([groups['ucb1'][j]-groups['epsilon_0.1'][j] for j in range(n)])
        se = differences.std(ddof=1)/math.sqrt(n)
        near(r['mean'],differences.mean(),'paired mean')
        near(r['se'],se,'paired SE')
        near(r['ci_low'],differences.mean()-1.96*se,'paired low')
        near(r['ci_high'],differences.mean()+1.96*se,'paired high')
    for name in ['comparison','epsilon','reward_gaps','hoeffding','regret_scaling']:
        for suffix in ['png','pdf']:
            path = Path(f'results/figures/{name}.{suffix}')
            assert path.stat().st_size>1000
    for document in [Path('README.md'),Path('report/report.md')]:
        text = document.read_text(encoding='utf-8')
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if not target.startswith(('http://','https://','#')):
                assert (document.parent/target).is_file(), f'broken link: {document}: {target}'
    subprocess.run([sys.executable,'-m','tools.build_report','--check'],check=True)
    # Environment metadata deliberately varies across operating systems and
    # Python versions; only numerical CSV evidence must retain identical bytes.
    paths = sorted(Path('results/data').glob('*.csv'))
    manifest = {path.as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    manifest_path = Path('results/manifest.json')
    if args.write_manifest:
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    else:
        assert json.loads(manifest_path.read_text(encoding='utf-8'))==manifest, 'numerical evidence checksums changed'
    print(f'PASS {checkpoints_checked} checkpoint metric summaries, paired contrasts, 10 figures, links, documentation, and {len(paths)} evidence checksums')


if __name__ == '__main__':
    main()
