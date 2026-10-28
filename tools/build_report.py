"""Populate marked numerical sections directly from saved results."""
import argparse
import re
from pathlib import Path
import numpy as np
from src.simulation import read_csv, mean_ci, write_csv


def policy_label(name):
    return {'random':'Uniform random','ucb1':'UCB1'}.get(name, f"ε-greedy ({name.removeprefix('epsilon_')})")


def interval(values, digits=2):
    c = mean_ci(values)
    return f"{c['mean']:.{digits}f} [{c['ci_low']:.{digits}f}, {c['ci_high']:.{digits}f}]"


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |',
                       *['| '+' | '.join(map(str,row))+' |' for row in rows]])


def section(path, name, content, check):
    start, end = f'<!-- BEGIN {name} -->', f'<!-- END {name} -->'
    text = path.read_text(encoding='utf-8')
    updated, count = re.subn(re.escape(start)+r'.*?'+re.escape(end),
                            lambda _:start+'\n'+content+'\n'+end, text, flags=re.S)
    if count != 1:
        raise ValueError(f"expected one {name} section in {path}")
    if check:
        if updated != text:
            raise AssertionError(f"stale generated results in {path}; run python -m tools.build_report")
    else:
        path.write_text(updated,encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('results'))
    parser.add_argument('--check',action='store_true')
    args = parser.parse_args()
    data = args.results/'data'
    def raw(name, t=None):
        values = read_csv(data/f'{name}_trials.csv')
        t = t or max(int(r['t']) for r in values)
        return [r for r in values if int(r['t'])==t]
    def grouped(rows):
        return [(p,[r for r in rows if r['policy']==p]) for p in dict.fromkeys(r['policy'] for r in rows)]
    def metric(rows,key):
        return [float(r[key]) for r in rows]
    comparison = table(['Policy','Cumulative reward (95% CI)','Pseudo-regret (95% CI)','Optimal frequency, last 1,000 (95% CI)'],
        [[policy_label(p), interval(metric(r,'reward')), interval(metric(r,'pseudo_regret')),
          interval(metric(r,'recent_optimal_fraction'),4)] for p,r in grouped(raw('comparison'))])
    epsilon_rows = []
    for p, rows in grouped(raw('epsilon')):
        values = metric(rows,'pseudo_regret')
        epsilon_rows.append([p.removeprefix('epsilon_'),interval(values),f'{np.median(values):.2f}',
                             f'{np.quantile(values,.95):.2f}',interval(metric(rows,'recent_optimal_fraction'),4)])
    epsilon = table(['ε','Pseudo-regret (95% CI)','Median regret','95th percentile regret','Last-1,000 optimal frequency (95% CI)'],epsilon_rows)
    gap = table(['Environment','Policy','Pseudo-regret (95% CI)','Last-1,000 optimal frequency (95% CI)'],
        [[env,policy_label(p),interval(metric(r,'pseudo_regret')),interval(metric(r,'recent_optimal_fraction'),4)]
          for env in ['easy','difficult'] for p,r in grouped(raw('gap_'+env))])
    scaling_rows = []
    for t in sorted({int(r['t']) for r in read_csv(data/'scaling_trials.csv')}):
        policies = dict(grouped(raw('scaling',t)))
        scaling_rows.append([str(t),*[interval(metric(policies[p],'pseudo_regret')) for p in ['random','epsilon_0.1','ucb1']]])
    scaling = table(['T','Random pseudo-regret (95% CI)','ε = 0.1 pseudo-regret (95% CI)','UCB1 pseudo-regret (95% CI)'],scaling_rows)
    h = read_csv(data/'hoeffding.csv')
    selected = [r for r in h if r['mu']=='0.5' and r['delta']=='0.1']
    hoeffding = table(['n','Events / replications','Empirical probability (Wilson 95% CI)','Exact binomial probability','Hoeffding bound, capped at 1'],
        [[r['n'],f"{r['events']} / {r['replications']}",f"{float(r['estimate']):.6f} [{float(r['ci_low']):.6f}, {float(r['ci_high']):.6f}]",
          f"{float(r['exact_probability']):.6g}",f"{float(r['bound']):.6g}"] for r in selected])
    paired = []
    for name in ['comparison','gap_easy','gap_difficult','scaling']:
        groups = dict(grouped(raw(name)))
        left = {int(r['trial']):float(r['pseudo_regret']) for r in groups['ucb1']}
        right = {int(r['trial']):float(r['pseudo_regret']) for r in groups['epsilon_0.1']}
        delta = [left[j]-right[j] for j in sorted(left)]
        ci = mean_ci(delta)
        paired.append(dict(experiment=name,contrast='ucb1_minus_epsilon_0.1',trials=len(delta),**ci))
    paired_table = table(['Experiment','UCB1 − ε = 0.1 regret, paired 95% CI'],
                         [[r['experiment'],f"{r['mean']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}]"] for r in paired])
    if not args.check:
        write_csv(data/'paired_differences.csv',paired)
    for name,content in [('comparison',comparison),('epsilon',epsilon),('gaps',gap),('hoeffding',hoeffding),('scaling',scaling),('paired',paired_table)]:
        section(Path('report/report.md'),name,content,args.check)
    section(Path('README.md'),'comparison',comparison,args.check)
    print('Documentation numerical sections match saved trial data.' if args.check else 'Documentation numerical sections regenerated.')


if __name__ == '__main__':
    main()
