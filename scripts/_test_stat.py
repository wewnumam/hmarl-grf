import sys, os
sys.path.insert(0, '/gfootball')
sys.path.insert(0, '/gfootball/scripts')
from stat_test import run_statistical_test

# degenerate: combined identical (the --quick failure)
r = run_statistical_test([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 'random')
assert r['p_value'] == 1.0 and not r['significant'], r
print('degenerate:', r['test'])

# complete separation, both constant but different (must NOT short-circuit)
r = run_statistical_test([10.0, 10.0, 10.0], [0.0, 0.0, 0.0], 'random')
print('separated:', r.get('test'), 'p=', r.get('p_value'), 'sig=', r.get('significant'))
assert 'p_value' in r and r['p_value'] < 0.05, r

# normal varied samples
r = run_statistical_test([5.0, 8.0, 3.0, 9.0, 6.0], [1.0, 2.0, 0.0, 3.0, 1.0], 'shppo')
print('varied:', r.get('test'), 'p=', round(r['p_value'], 4))
assert r['significant']

# single seed still skipped
r = run_statistical_test([1.0], [0.0], 'x')
assert 'skipped' in r['test'], r
print('skip:', r['test'])
print('ALL OK')
