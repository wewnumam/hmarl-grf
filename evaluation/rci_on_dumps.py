"""Compute RCI (strict/cat/nomove/streak) on GRF episode .dump files.

Feeds each frame's raw observation + actual actions through the thesis's
rule-based hierarchy (HighLevel -> MidLevel -> ExpertPolicyEval) to obtain
the ideal action a*, then applies hmarl.rci. Written for comparing a flat
baseline (e.g. ppo_flock) against the HMARL research RCI framework.

Run (in Docker):
    python evaluation/rci_on_dumps.py dumps/episode_done_*.dump [--out dumps/rci_flock_eval.json]
"""
import argparse
import glob
import json
import os
import pickle
import random
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hmarl.expert import ExpertPolicy, ExpertPolicyEval
from hmarl.env import get_action_category
from hmarl.policy import HighLevelPolicy, MidLevelPolicy
from hmarl.rci import (compute_rci_category, compute_rci_nomove,
                       compute_rci_strict, compute_rci_streak)

# GRF 'default' action-set: name -> index (mirrors hmarl.expert ACT_* table)
ACTION_NAME_TO_IDX = {
    'idle': 0, 'left': 1, 'top_left': 2, 'top': 3, 'top_right': 4,
    'right': 5, 'bottom_right': 6, 'bottom': 7, 'bottom_left': 8,
    'long_pass': 9, 'high_pass': 10, 'short_pass': 11, 'shot': 12,
    'sprint': 13, 'release_direction': 14, 'release_sprint': 15,
    'sliding': 16, 'dribble': 17, 'release_dribble': 18,
}


def action_index(a) -> int:
    name = getattr(a, '_name', None) or getattr(a, 'name', None)
    if name is None:
        raise ValueError('unknown action object: %r' % (a,))
    if name not in ACTION_NAME_TO_IDX:
        raise ValueError('unknown action name: %r' % (name,))
    return ACTION_NAME_TO_IDX[name]


def load_frames(path: str) -> List[Dict]:
    """GRF .dump = concatenated per-frame pickles (one per timestep)."""
    frames = []
    with open(path, 'rb') as f:
        while True:
            try:
                frames.append(pickle.load(f, encoding='latin1'))
            except EOFError:
                break
    return frames


def analyze_dump(path: str, expert, high, mid, max_frames=None) -> Dict:
    frames = load_frames(path)
    if max_frames:
        frames = frames[:max_frames]

    actual, ideal = [], []
    for fr in frames:
        obs = fr['observation']
        acts = [action_index(a) for a in fr['debug']['action']]
        macro = high.decide(obs)
        sub_goals = [mid.decide(obs, i, macro) for i in range(len(acts))]
        ideal_acts = [
            expert.get_ideal_action(obs, i, sub_goals[i], macro)
            for i in range(len(acts))
        ]
        actual.append(acts)
        ideal.append(ideal_acts)

    strict, strict_pa = compute_rci_strict(actual, ideal)
    cat, cat_pa = compute_rci_category(actual, ideal)
    nomove, _ = compute_rci_nomove(actual, ideal)
    streak = compute_rci_streak(actual, ideal, min_streak=10)

    # uniform-random reference (same expert actions, random actual)
    rng = random.Random(0)
    n_act = len(ACTION_NAME_TO_IDX)
    rand_act = [[rng.randrange(n_act) for _ in range(len(actual[0]))]
                for _ in range(len(actual))]
    r_strict, _ = compute_rci_strict(rand_act, ideal)
    r_cat, _ = compute_rci_category(rand_act, ideal)

    # action distribution of actual actions
    dist = np.zeros(n_act, dtype=int)
    for step in actual:
        for a in step:
            dist[a] += 1

    score = frames[-1]['observation'].get('score', [0, 0])
    return {
        'dump': os.path.basename(path),
        'frames': len(frames),
        'score': list(score),
        'rci_strict': round(strict, 4),
        'rci_cat': round(cat, 4),
        'rci_nomove': round(nomove, 4),
        'rci_streak': round(streak['rci_streak'], 4),
        'mean_streak_length': round(streak['mean_streak_length'], 2),
        'role_switch_rate': round(streak['role_switch_rate'], 4),
        'random_ref_strict': round(r_strict, 4),
        'random_ref_cat': round(r_cat, 4),
        'rci_strict_per_agent': [round(v, 4) for v in strict_pa],
        'rci_cat_per_agent': [round(v, 4) for v in cat_pa],
        'action_dist': {k: int(v) for k, v in
                        zip(ACTION_NAME_TO_IDX.keys(), dist)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dumps', nargs='+', help='dump file path(s) or globs')
    parser.add_argument('--out', default='dumps/rci_on_dumps.json')
    parser.add_argument('--expert', choices=('eval', 'train'), default='eval',
                        help='eval = ExpertPolicyEval (thesis evaluation expert, '
                             'anti-circularity); train = ExpertPolicy default '
                             '(matches HMARL training_log rci reference)')
    args = parser.parse_args()

    paths = []
    for p in args.dumps:
        paths.extend(sorted(glob.glob(p)) or [p])
    if not paths:
        print('no dump files matched')
        sys.exit(1)

    expert = ExpertPolicyEval() if args.expert == 'eval' else ExpertPolicy()
    high = HighLevelPolicy()
    mid = MidLevelPolicy()

    results = []
    for p in paths:
        print('analyzing %s ...' % p, flush=True)
        r = analyze_dump(p, expert, high, mid)
        results.append(r)
        print('  frames=%d score=%s strict=%.3f cat=%.3f nomove=%.3f '
              'streak=%.3f (random ref: strict=%.3f cat=%.3f)'
              % (r['frames'], r['score'], r['rci_strict'], r['rci_cat'],
                 r['rci_nomove'], r['rci_streak'], r['random_ref_strict'],
                 r['random_ref_cat']), flush=True)

    agg = {k: round(float(np.mean([r[k] for r in results])), 4)
           for k in ('rci_strict', 'rci_cat', 'rci_nomove', 'rci_streak',
                     'role_switch_rate', 'random_ref_strict',
                     'random_ref_cat')}
    payload = {'script': 'evaluation/rci_on_dumps.py',
               'expert': ('ExpertPolicyEval (d_tackle=0.06, d_safe=0.18, d_shoot=0.25)'
                          if args.expert == 'eval' else
                          'ExpertPolicy/train (d_tackle=0.05, d_safe=0.15, d_shoot=0.30)'),
               'dumps': results, 'aggregate': agg}
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2)
    print('\naggregate:', json.dumps(agg, indent=2))
    print('written -> %s' % args.out)


if __name__ == '__main__':
    main()