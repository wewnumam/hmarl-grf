"""Kriteria ground truth kebijakan ahli — BAB IV, kriteria (a)(b)(c).

Menjalankan kebijakan ahli secara mandiri (tanpa learning) terhadap bot GRF
tingkat kesulitan 1.0 (11_vs_11_stochastic) dan mengukur:

  (a) progressive passes per episode                 (target > 30)
  (b) shots on target per episode                    (target > 50)
  (c) porsi waktu menguasai bola di sepertiga menyerang (target > 5%)

Dijalankan untuk kedua referensi expert yang dipakai tesis:
  * ExpertPolicyAllAgents — expert komponen shaping reward
  * ExpertPolicyEval      — expert acuan RCI (threshold diperturbasi)

Output: evaluation_results/expert_ground_truth.json + ringkasan stdout.

Jalankan di container:
  python /gfootball/scripts/validate_expert_criteria.py --episodes 5
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hmarl.env import create_raw_env, extract_game_state
from hmarl.expert import ACT_SHOT, ExpertPolicyAllAgents, ExpertPolicyEval
from hmarl.metrics import progressive_pass_ratio
from hmarl.policy import HierarchicalController
from hmarl.utils import EPISODE_MAX_STEPS, set_seed

ATTACKING_THIRD_X = 0.5  # ball_x > 0.5 == sepertiga lapangan menyerang


def run_episode(env, controller, expert):
    reset = env.reset()
    obs = reset[0] if isinstance(reset, tuple) else reset
    gs = extract_game_state(obs)

    states, actions_log = [], []
    shot_steps = shot_owner_steps = third_steps = 0
    info_keys = None
    done, step = False, 0
    while not done and step < EPISODE_MAX_STEPS:
        macro = controller.get_macro_strategy(gs)
        sub_goals = controller.get_sub_goals(gs, macro)
        actions = expert.get_ideal_actions(gs, sub_goals, macro)

        states.append(gs)
        actions_log.append(list(actions))

        owner = gs.get('ball_owned_player', -1)
        if ACT_SHOT in actions:
            shot_steps += 1
            # GRF exposes no "shots on target" counter, so the closest honest
            # proxy is a shot decision taken by the player actually on the ball.
            if gs.get('ball_owned_team', -1) == 0 and 0 <= owner < len(actions) \
                    and actions[owner] == ACT_SHOT:
                shot_owner_steps += 1
        if gs.get('ball_owned_team', -1) == 0 and gs.get('ball', [0, 0, 0])[0] > ATTACKING_THIRD_X:
            third_steps += 1

        res = env.step(actions)
        obs = res[0]
        if info_keys is None and len(res) == 4:
            info_keys = sorted(res[3].keys())
        done = res[2] if len(res) == 4 else (res[2] or res[3])
        gs = extract_game_state(obs)
        step += 1

    _, progressive, _ = progressive_pass_ratio(actions_log, states)
    score = gs.get('score', [0, 0])
    return {
        'steps': step,
        'score': [int(score[0]), int(score[1])],
        'progressive_passes': progressive,
        'shot_steps': shot_steps,
        'shot_owner_steps': shot_owner_steps,
        'attacking_third_steps': third_steps,
        'info_keys': info_keys,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=5)
    ap.add_argument('--scenario', default='11_vs_11_stochastic')
    ap.add_argument('--output', default='evaluation_results/expert_ground_truth.json')
    args = ap.parse_args()

    env = create_raw_env(env_name=args.scenario)
    controller = HierarchicalController()
    # Same convention as scripts/eval.py: swap the inner policy to switch expert.
    variants = {
        'all_agents': ExpertPolicyAllAgents(),
        'eval': ExpertPolicyAllAgents(),
    }
    variants['eval'].policy = ExpertPolicyEval()

    out = {'episodes': args.episodes, 'scenario': args.scenario, 'criteria': {
        'progressive_passes_per_episode': '> 30',
        'shot_steps_per_episode': '> 50',
        'attacking_third_share_pct': '> 5',
    }, 'variants': {}}

    for name, expert in variants.items():
        set_seed(42)
        eps = []
        for ep in range(args.episodes):
            r = run_episode(env, controller, expert)
            eps.append(r)
            print(f"[{name}] ep{ep}: {r['score'][0]}-{r['score'][1]} "
                  f"steps={r['steps']} prog_pass={r['progressive_passes']} "
                  f"shot_steps={r['shot_steps']} third_steps={r['attacking_third_steps']}",
                  flush=True)

        n = len(eps)
        gf = sum(e['score'][0] for e in eps)
        ga = sum(e['score'][1] for e in eps)
        wins = sum(1 for e in eps if e['score'][0] > e['score'][1])
        mean = lambda k: sum(e[k] for e in eps) / n
        third_share = 100.0 * sum(e['attacking_third_steps'] for e in eps) / sum(e['steps'] for e in eps)
        out['variants'][name] = {
            'per_episode': eps,
            'progressive_passes_per_episode': round(mean('progressive_passes'), 2),
            'shot_steps_per_episode': round(mean('shot_steps'), 2),
            'shot_by_ball_owner_per_episode': round(mean('shot_owner_steps'), 2),
            'attacking_third_share_pct': round(third_share, 2),
            'win_rate_pct': round(100.0 * wins / n, 1),
            'goals_for_per_episode': round(gf / n, 2),
            'goals_against_per_episode': round(ga / n, 2),
        }
        v = out['variants'][name]
        print(f"[{name}] progressive_passes/ep={v['progressive_passes_per_episode']} "
              f"shot_steps/ep={v['shot_steps_per_episode']} "
              f"shot_by_owner/ep={v['shot_by_ball_owner_per_episode']} "
              f"third_share={v['attacking_third_share_pct']}% WR={v['win_rate_pct']}% "
              f"GF/ep={v['goals_for_per_episode']} GA/ep={v['goals_against_per_episode']}",
              flush=True)
        print(f"[{name}] info keys: {eps[0].get('info_keys')}", flush=True)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"saved -> {args.output}", flush=True)
    env.close()


if __name__ == '__main__':
    main()
