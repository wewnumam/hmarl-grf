"""Action-category distribution of the expert reference on one dump (confound diag)."""
import json
import os
import pickle
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hmarl.env import get_action_category
from hmarl.expert import ExpertPolicy, ExpertPolicyEval
from hmarl.policy import HighLevelPolicy, MidLevelPolicy

DUMP = sys.argv[1] if len(sys.argv) > 1 else 'dumps/episode_done_20261010-031412501865.dump'

frames = []
with open(DUMP, 'rb') as f:
    while True:
        try:
            frames.append(pickle.load(f, encoding='latin1'))
        except EOFError:
            break

high, mid = HighLevelPolicy(), MidLevelPolicy()
for tag, expert in (('eval', ExpertPolicyEval()), ('train', ExpertPolicy())):
    act_c, cat_c = Counter(), Counter()
    for fr in frames:
        obs = fr['observation']
        n = len(fr['debug']['action'])
        macro = high.decide(obs)
        subs = [mid.decide(obs, i, macro) for i in range(n)]
        for i in range(n):
            a = expert.get_ideal_action(obs, i, subs[i], macro)
            act_c[a] += 1
            cat_c[get_action_category(a)] += 1
    tot = sum(act_c.values())
    print('expert=%s total=%d' % (tag, tot))
    print('  categories:', {k: round(100.0 * v / tot, 2) for k, v in cat_c.most_common()})
    names = ['idle','left','top_left','top','top_right','right','bottom_right',
             'bottom','bottom_left','long_pass','high_pass','short_pass','shot',
             'sprint','release_direction','release_sprint','sliding','dribble',
             'release_dribble']
    print('  top actions:', [(names[k], round(100.0*v/tot,2)) for k, v in act_c.most_common(8)])