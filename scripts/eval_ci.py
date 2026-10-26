"""Bootstrap confidence intervals + IQM over per-episode evaluation results.

Implements the reporting layer of Agarwal et al. (2021, "Deep RL at the Edge of
the Statistical Precipice", rliable) with numpy only:
  - stratified bootstrap 95% CI of the mean goal difference (and reward)
  - interquartile mean (IQM), the rliable default robust aggregate
  - pairwise P(X > Y) = bootstrap probability that policy X beats policy Y

Reads the `<algo>_episodes.json` files written by scripts/eval.py in each
results directory; writes `<dir>/ci_summary.json`.

Usage:
  python3 scripts/eval_ci.py evaluation_results_3m evaluation_results_ablation ...
"""

import glob
import json
import os
import sys

import numpy as np

N_BOOT = 10_000
SEED = 0


def load_episodes(path):
    """-> dict algo -> np.array of per-episode goal differences."""
    out = {}
    for fp in sorted(glob.glob(os.path.join(path, '*_episodes.json'))):
        algo = os.path.basename(fp)[:-len('_episodes.json')]
        d = json.load(open(fp))
        gd = np.array(d['goals_for'], dtype=float) - np.array(d['goals_against'], dtype=float)
        out[algo] = {'gd': gd, 'reward': np.array(d['rewards'], dtype=float), 'file': fp}
    return out


def load_shards(path):
    """Fallback: pool the per-arm arrays of `shard_*.json` (one entry per seed).

    Each entry is that seed's *total* goal difference over its evaluation
    episodes, so n = number of seeds, not episodes.
    """
    files = sorted(glob.glob(os.path.join(path, 'shard_*.json')))
    if not files:
        return {}
    pooled = {}
    for fp in files:
        for algo, r in json.load(open(fp)).get('results', {}).items():
            p = pooled.setdefault(algo, {'gd': [], 'reward': [], 'file': path})
            p['gd'] += list(r.get('goal_differences', []))
            p['reward'] += list(r.get('rewards', []))
    return {a: {k: np.array(v, dtype=float) for k, v in d.items() if k != 'file'} |
               {'file': d['file']} for a, d in pooled.items() if d['gd']}


def iqm(x, axis=-1):
    """Interquartile mean; supports an axis so it can be a bootstrap aggregate."""
    x = np.sort(x, axis=axis)
    n = x.shape[axis]
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    if hi <= lo:
        return x.mean(axis=axis)
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(lo, hi)
    return x[tuple(sl)].mean(axis=axis)


def bootstrap_ci(x, agg=np.mean):
    rng = np.random.RandomState(SEED)
    idx = rng.randint(0, len(x), size=(N_BOOT, len(x)))
    stats = agg(x[idx], axis=1)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def p_improvement(x, y):
    """P(mean(x) > mean(y)) under independent stratified bootstraps."""
    rng = np.random.RandomState(SEED + 1)
    bx = x[rng.randint(0, len(x), size=(N_BOOT, len(x)))].mean(axis=1)
    by = y[rng.randint(0, len(y), size=(N_BOOT, len(y)))].mean(axis=1)
    return float((bx > by).mean())


def summarise(data, metric='gd'):
    rows = {}
    for algo, d in sorted(data.items()):
        x = d[metric]
        lo, hi = bootstrap_ci(x)
        rows[algo] = {
            'n': len(x), 'mean': float(x.mean()), 'sd': float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            'median': float(np.median(x)), 'iqm': iqm(x),
            'ci95_boot_lo': lo, 'ci95_boot_hi': hi,
            'iqm_ci95_boot_lo': bootstrap_ci(x, iqm)[0],
            'iqm_ci95_boot_hi': bootstrap_ci(x, iqm)[1],
        }
    return rows


def main():
    dirs = sys.argv[1:] or ['.']
    for path in dirs:
        data = load_episodes(path) or load_shards(path)
        if not data:
            print('%s: no *_episodes.json / shard_*.json' % path)
            continue
        summary = {'dir': path, 'n_boot': N_BOOT, 'metrics': {}}
        for metric in ('gd', 'reward'):
            rows = summarise(data, metric)
            summary['metrics'][metric] = rows
            print('\n== %s  [%s]' % (path, metric))
            print('%-22s %3s %9s %8s %9s %8s %-19s %9s' % (
                'algo', 'n', 'mean', 'sd', 'IQM', 'median', 'mean 95% boot CI', 'IQM CI'))
            for algo, r in rows.items():
                print('%-22s %3d %9.2f %8.2f %9.2f %8.2f [%7.2f,%7.2f] [%6.2f,%6.2f]' % (
                    algo, r['n'], r['mean'], r['sd'], r['iqm'], r['median'],
                    r['ci95_boot_lo'], r['ci95_boot_hi'],
                    r['iqm_ci95_boot_lo'], r['iqm_ci95_boot_hi']))

        algos = sorted(data)
        summary['p_improvement'] = {}
        small = '  (n<=5: read the sign only, not the level)'
        print('\npairwise P(row > col), goal difference%s' % (
            small if any(len(data[a]['gd']) <= 5 for a in algos) else ''))
        print('%-22s %s' % ('algo', ' '.join('%9s' % a[:9] for a in algos)))
        for a in algos:
            line = []
            for b in algos:
                p = p_improvement(data[a]['gd'], data[b]['gd'])
                summary['p_improvement'].setdefault(a, {})[b] = p
                line.append('%9.2f' % p)
            print('%-22s %s' % (a, ' '.join(line)))

        out = os.path.join(path, 'ci_summary.json')
        with open(out, 'w') as fh:
            json.dump(summary, fh, indent=1)
        print('\nwrote %s' % out)


if __name__ == '__main__':
    main()
