"""Render LaTeX baseline-comparison table from evaluation_results_baseline/stat_test_results.json."""
import json, sys

ALG_LABEL = {'hmarl': 'HMARL (usulan)', 'random': 'Random Policy', 'ippo': 'IPPO', 'mappo': 'MAPPO', 'shppo': 'SHPPO'}


def fmt(v, d=2):
    if v is None:
        return '---'
    try:
        v = float(v)
    except (TypeError, ValueError):
        return '---'
    if v != v:  # NaN
        return '---'
    return ('%%.%df' % d) % v


ROW_LABEL = {'rci': 'RCI (kategorikal)', 'rci_strict': 'RCI (ketat)', 'fai_mean': 'FAI',
             'ppr': 'PPR', 'psr': 'PSR', 'compactness_mean': 'Compactness',
             'positional_entropy': 'Positional entropy', 'rci_nomove': 'RCI (nomove)'}


def main(path, rows=('rci', 'rci_strict', 'fai_mean', 'ppr', 'psr', 'compactness_mean', 'positional_entropy')):
    d = json.load(open(path, encoding='utf-8'))
    res = d['results']
    algs = [a for a in ('hmarl', 'random', 'ippo', 'mappo', 'shppo') if a in res]
    print('%% meta:', json.dumps(d.get('meta', {}))[:200])
    print('%% config:', json.dumps(d.get('config', {})))
    print('\\begin{table}[H]')
    print('\\caption{Perbandingan HMARL dengan kebijakan pembanding}')
    print('\\label{tab:baseline_comparison}')
    print('\\centering\\small')
    print('\\begin{tabular}{l%s}' % ('c' * len(algs)))
    print('\\toprule')
    print('\\textbf{Metrik} & ' + ' & '.join('\\textbf{%s}' % ALG_LABEL.get(a, a) for a in algs) + ' \\\\')
    print('\\midrule')
    for key, name in (('win_rate_mean', 'Win rate (\\%)'), ('goal_difference_mean', 'Goal difference')):
        cells = []
        for a in algs:
            m = res[a].get(key)
            s = res[a].get(key.replace('_mean', '_std'))
            cells.append('%s \\pm %s' % (fmt(m), fmt(s)))
        print('%s & %s \\\\' % (name, ' & '.join(cells)))
    for key in rows:
        if key in ('win_rate', 'goal_difference'):
            continue
        cells = []
        for a in algs:
            vals = res[a].get(key)
            if isinstance(vals, list) and vals:
                cells.append(fmt(sum(vals) / len(vals), 3))
            else:
                cells.append('---')
        print('%s & %s \\\\' % (ROW_LABEL.get(key, key), ' & '.join(cells)))
    print('\\bottomrule')
    print('\\end{tabular}')
    print('\\end{table}')
    for t in d.get('statistical_tests', []):
        print('%% STAT', json.dumps(t))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'evaluation_results_baseline/stat_test_results.json')
