"""Probe: mengapa model tidak pernah shoot / gol.

Menggabungkan sub-goal scheduler + expert oracle (tanpa kebijakan PPO),
mengukur:
  - seberapa sering ideal action = ACT_SHOT
  - apakah tim pernah membawa bola ke final third (x > 0.5)
  - di mana kepemilikan bola berakhir (progression failure vs shot-gate vs imitation failure)

Jalankan di container:
  python /gfootball/scripts/probe_shot_path.py --episodes 5

Output: stdout ringkas + evaluation_results/probe_shot_path.json
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hmarl.env import (
    NUM_AGENTS, create_raw_env, extract_game_state,
    get_player_position, get_distance, get_player_role,
    ROLE_LM, ROLE_RM, ROLE_CF,
)
from hmarl.utils import EPISODE_MAX_STEPS
from hmarl.policy import HierarchicalController
from hmarl.expert import (
    ExpertPolicyAllAgents, D_SHOOT,
    ACT_SHOT, ACT_SHORT_PASS, ACT_LONG_PASS, ACT_HIGH_PASS, ACT_SPRINT,
)

ATTACKER_ROLES = (ROLE_LM, ROLE_RM, ROLE_CF)
ATTACKER_IDS = (8, 9, 10)  # 4-3-3: LM, RM, CF (index == player_idx)

PASS_IDS = (ACT_LONG_PASS, ACT_HIGH_PASS, ACT_SHORT_PASS)


def run_episode(env, controller, expert, ep_idx):
    reset = env.reset()
    obs_raw = reset[0] if isinstance(reset, tuple) else reset
    gs = extract_game_state(obs_raw)

    ideal_counter = Counter()
    actual_counter = Counter()
    n_steps = 0

    # possession split
    poss = Counter()  # 'us' | 'opp' | 'loose'

    # progression
    max_ball_x_us = -1.0      # max ball x while WE own it
    max_ball_x_any = -1.0
    steps_final_third_us = 0  # kita pegang bola saat ball_x > 0.5
    steps_final_third_att_carrier = 0

    # shot gate analysis (attacker dengan bola)
    att_carrier_steps = 0
    shot_gate_fail_reason = Counter()
    shot_ideal_count = 0

    # dist to goal saat attacker pegang bola
    dists_when_att_carrier = []

    done = False
    step = 0
    while not done and step < EPISODE_MAX_STEPS:
        macro = controller.get_macro_strategy(gs)
        sub_goals = controller.get_sub_goals(gs, macro)
        ideal = expert.get_ideal_actions(gs, sub_goals, macro)

        ball = gs.get('ball', [0, 0, 0])
        ball_x = ball[0]
        owned_team = gs.get('ball_owned_team', -1)
        owned_player = gs.get('ball_owned_player', -1)
        max_ball_x_any = max(max_ball_x_any, ball_x)

        if owned_team == 0:
            poss['us'] += 1
            max_ball_x_us = max(max_ball_x_us, ball_x)
            if ball_x > 0.5:
                steps_final_third_us += 1
        elif owned_team == 1:
            poss['opp'] += 1
        else:
            poss['loose'] += 1

        for a in ideal:
            ideal_counter[a] += 1
        if ACT_SHOT in ideal:
            shot_ideal_count += 1

        # attacker dengan bola: kenapa tidak shot?
        if owned_team == 0 and owned_player in ATTACKER_IDS:
            att_carrier_steps += 1
            pos = get_player_position(gs, 'left', owned_player)
            d_goal = get_distance(pos, [1.0, 0.0])
            dists_when_att_carrier.append(round(d_goal, 3))
            if ball_x > 0.5:
                steps_final_third_att_carrier += 1
            if ideal[owned_player] != ACT_SHOT:
                # catat syarat yang gagal
                if not (d_goal < max(D_SHOOT, 0.6) and pos[0] > 0.5):
                    shot_gate_fail_reason['gate: x<=0.5 atau d_goal>=0.6'] += 1
                else:
                    shot_gate_fail_reason['LAIN: gate terpenuhi tapi bukan shot'] += 1

        step_result = env.step(ideal)
        obs_new = step_result[0]
        done = step_result[2] if len(step_result) == 4 else (step_result[2] or step_result[3])
        gs = extract_game_state(obs_new)

        n_steps += 1
        step += 1

    score = gs.get('score', [0, 0])
    return {
        'episode': ep_idx,
        'steps': n_steps,
        'score': list(score),
        'ideal_action_dist': dict(ideal_counter),
        'ideal_shots': ideal_counter.get(ACT_SHOT, 0),
        'ideal_passes': sum(ideal_counter.get(a, 0) for a in PASS_IDS),
        'possession_steps': dict(poss),
        'max_ball_x_with_possession': round(max_ball_x_us, 3),
        'max_ball_x_overall': round(max_ball_x_any, 3),
        'steps_final_third_with_possession': steps_final_third_us,
        'steps_attacker_carrier': att_carrier_steps,
        'steps_attacker_carrier_in_final_third': steps_final_third_att_carrier,
        'attacker_carrier_goal_dist_p50': (
            sorted(dists_when_att_carrier)[len(dists_when_att_carrier) // 2]
            if dists_when_att_carrier else None),
        'attacker_carrier_goal_dist_min': min(dists_when_att_carrier) if dists_when_att_carrier else None,
        'shot_gate_fail_reason': dict(shot_gate_fail_reason),
        'steps_ideal_contains_shot': shot_ideal_count,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=5)
    ap.add_argument('--output', default='evaluation_results/probe_shot_path.json')
    args = ap.parse_args()

    env = create_raw_env()
    controller = HierarchicalController()
    expert = ExpertPolicyAllAgents()

    episodes = []
    for ep in range(args.episodes):
        r = run_episode(env, controller, expert, ep)
        episodes.append(r)
        print(f"ep{ep}: score={r['score']} steps={r['steps']} "
              f"poss={r['possession_steps']} maxX_us={r['max_ball_x_with_possession']} "
              f"shots(ideal)={r['ideal_shots']} final_third_steps={r['steps_final_third_with_possession']} "
              f"attCarrier={r['steps_attacker_carrier']} attCarrierInFinal3rd={r['steps_attacker_carrier_in_final_third']}",
              flush=True)

    # agregat
    agg_ideal = Counter()
    tot = Counter()
    for r in episodes:
        agg_ideal.update({int(k): v for k, v in r['ideal_action_dist'].items()})
        tot.update({k: v for k, v in r['possession_steps'].items()})
        for k, v in r['shot_gate_fail_reason'].items():
            tot[('gate', k)] += v

    from hmarl.env import ACTION_NAMES
    summary = {
        'episodes': args.episodes,
        'total_ideal_action_dist': {ACTION_NAMES[i]: agg_ideal.get(i, 0)
                                    for i in sorted(set(agg_ideal))},
        'total_ideal_shots': agg_ideal.get(ACT_SHOT, 0),
        'total_ideal_passes': sum(agg_ideal.get(a, 0) for a in PASS_IDS),
        'possession_steps': {k: v for k, v in tot.items() if not isinstance(k, tuple)},
        'shot_gate_fail_reason': {k[1]: v for k, v in tot.items() if isinstance(k, tuple)},
        'per_episode': episodes,
    }

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=1)

    print("\n=== AGGREGATE ===")
    print("ideal action dist:", summary['total_ideal_action_dist'])
    print("ideal SHOT steps:", summary['total_ideal_shots'])
    print("possession:", summary['possession_steps'])
    print("shot gate failures:", summary['shot_gate_fail_reason'])
    print(f"saved -> {args.output}")
    env.close()


if __name__ == '__main__':
    main()