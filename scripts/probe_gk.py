"""Probe GK: seberapa sering GK expert meninggalkan box, dan posisi GK saat kebobolan.

GRF koordinat: x -1 (garis gawang sendiri) .. +1 (gawang lawan).
Penalty box: kedalaman 15m dari garis gawang, lebar 40m dari tengah.
Pitch 105x68m -> 1m = 2/105 di x, 2/68 di y.
  box_x_line  = -1 + 15*(2/105)  = -0.7143
  box_y_half  = (40/2)*(2/68)    = 0.294

ukur per episode:
  - % langkah GK x > box_x_line (di luar box)
  - distribusi posisi GK (mean/p90/max x)
  - posisi GK saat kebobolan (ga naik) + jarak ke goal center
  - trigger save (dist_to_ball < 0.1) berapa kali & di posisi mana

jalankan: python /gfootball/scripts/probe_gk.py --episodes 3
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from hmarl.utils import EPISODE_MAX_STEPS
from hmarl.env import (
    create_raw_env, extract_game_state, get_player_position,
    get_distance, ROLE_GK, get_player_role,
)
from hmarl.policy import HierarchicalController
from hmarl.expert import ExpertPolicyAllAgents, ExpertPolicy

BOX_X_LINE = -1.0 + 15.0 * (2.0 / 105.0)   # -0.7143
BOX_Y_HALF = 20.0 * (2.0 / 68.0)           # 0.2941
GK_ID = 0


def run_episode(env, controller, expert, ep_idx):
    reset = env.reset()
    obs_raw = reset[0] if isinstance(reset, tuple) else reset
    gs = extract_game_state(obs_raw)

    xs, ys = [], []
    out_box = 0
    in_box_lateral = 0   # x di box tapi y di luar lebar box
    ga_events = []       # posisi GK saat kebobolan
    loose_events = 0
    prev_ga = 0
    steps = 0
    done = False
    while not done and steps < EPISODE_MAX_STEPS:
        macro = controller.get_macro_strategy(gs)
        sub_goals = controller.get_sub_goals(gs, macro)
        ideal = expert.get_ideal_actions(gs, sub_goals, macro)

        pos = get_player_position(gs, 'left', GK_ID)
        xs.append(pos[0])
        ys.append(pos[1])
        if pos[0] > BOX_X_LINE:
            out_box += 1
        if pos[0] <= BOX_X_LINE and abs(pos[1]) > BOX_Y_HALF:
            in_box_lateral += 1

        score = gs.get('score', [0, 0])
        ga_now = score[1] if isinstance(score, (list, tuple)) else 0

        step_result = env.step(ideal)
        gs = extract_game_state(step_result[0])
        done = step_result[2] if len(step_result) == 4 else (step_result[2] or step_result[3])

        score2 = gs.get('score', [0, 0])
        ga2 = score2[1] if isinstance(score2, (list, tuple)) else 0
        if ga2 > prev_ga:
            pos_after = get_player_position(gs, 'left', GK_ID)
            ball = gs.get('ball', [0, 0, 0])
            loose_events += 0
            ga_events.append({
                'gk_pos': [round(pos_after[0], 3), round(pos_after[1], 3)],
                'gk_out_box': bool(pos_after[0] > BOX_X_LINE),
                'dist_to_goal': round(get_distance(pos_after, [-1.0, 0.0]), 3),
                'ball_pos': [round(ball[0], 3), round(ball[1], 3)],
                'ball_in_our_box': bool(ball[0] < BOX_X_LINE),
            })
            prev_ga = ga2
        steps += 1

    xs_a = np.array(xs)
    ys_a = np.array(ys)
    return {
        'episode': ep_idx,
        'steps': steps,
        'final_score': list(gs.get('score', [0, 0])),
        'gk_out_box_pct': round(100.0 * out_box / max(steps, 1), 2),
        'gk_out_box_lateral_pct': round(100.0 * in_box_lateral / max(steps, 1), 2),
        'gk_x_mean': round(float(xs_a.mean()), 3),
        'gk_x_p90': round(float(np.percentile(xs_a, 90)), 3),
        'gk_x_max': round(float(xs_a.max()), 3),
        'gk_y_absmean': round(float(np.abs(ys_a).mean()), 3),
        'goals_conceded': prev_ga,
        'ga_events': ga_events,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--output', default='evaluation_results/probe_gk.json')
    args = ap.parse_args()

    env = create_raw_env()
    controller = HierarchicalController()
    expert = ExpertPolicyAllAgents()

    eps = []
    for ep in range(args.episodes):
        r = run_episode(env, controller, expert, ep)
        eps.append(r)
        print(f"ep{ep}: score={r['final_score']} GA={r['goals_conceded']} "
              f"GK_out_box={r['gk_out_box_pct']}% lateral_out={r['gk_out_box_lateral_pct']}% "
              f"x(mean/p90/max)={r['gk_x_mean']}/{r['gk_x_p90']}/{r['gk_x_max']} "
              f"|y|_mean={r['gk_y_absmean']}", flush=True)
        for e in r['ga_events']:
            print(f"    GA: GK@{e['gk_pos']} out_box={e['gk_out_box']} "
                  f"ball@{e['ball_pos']} ball_in_our_box={e['ball_in_our_box']}", flush=True)

    all_ga = [e for r in eps for e in r['ga_events']]
    n_ga = len(all_ga)
    n_out = sum(1 for e in all_ga if e['gk_out_box'])
    summary = {
        'episodes': args.episodes,
        'total_goals_conceded': n_ga,
        'ga_with_gk_out_of_box': n_out,
        'ga_with_gk_out_pct': round(100.0 * n_out / max(n_ga, 1), 1),
        'mean_gk_out_box_pct': round(float(np.mean([r['gk_out_box_pct'] for r in eps])), 2),
        'box_definition': {'x_line': round(BOX_X_LINE, 4), 'y_half': round(BOX_Y_HALF, 4)},
        'per_episode': eps,
    }
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=1)
    print(f"\n=== SUMMARY ===")
    print(f"GA total={n_ga}, saat GK out of box={n_out} ({summary['ga_with_gk_out_pct']}%)")
    print(f"GK out-of-box rata-rata {summary['mean_gk_out_box_pct']}% langkah")
    print(f"saved -> {args.output}")
    env.close()


if __name__ == '__main__':
    main()