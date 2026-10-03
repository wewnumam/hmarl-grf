"""Expert Policy Threshold Calibration.

Empirically determines d_tackle, d_safe, d_shoot from GRF environment data.

Method:
1. Run N episodes of GRF Academy scenarios (11v11)
2. Record per-timestep distances:
   - min_opp_dist: nearest opponent distance → d_tackle
   - space_radius: radius to nearest opponent for dribble decisions → d_safe
   - goal_dist: distance from ball carrier to goal → d_shoot
3. Analyze distributions and pick thresholds as percentiles
4. Cross-validate: run expert policy with calibrated thresholds, measure win rate

Usage (inside Docker):
    python scripts/calibrate_expert.py --episodes 100 --scenario 11_vs_11_stochastic
    python scripts/calibrate_expert.py --episodes 50 --scenario academy_empty_goal --quick
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hmarl.env import (
    create_raw_env, extract_game_state, NUM_AGENTS,
    get_ball_position, get_player_position, get_distance,
    ball_owned_by_us, get_player_role,
    ROLE_GK, ROLE_CB, ROLE_LB, ROLE_RB, ROLE_DM, ROLE_CM,
    ROLE_LM, ROLE_RM, ROLE_AM, ROLE_CF,
)
from hmarl.policy import HierarchicalController
from hmarl.expert import ExpertPolicyAllAgents, ExpertPolicy
from hmarl.utils import set_seed


def collect_distance_distributions(
    n_episodes: int = 100,
    scenario: str = "11_vs_11_stochastic",
    seed: int = 42,
) -> Dict[str, List[float]]:
    """Run episodes and collect distance distributions for threshold calibration.

    Returns dict with lists of distances:
      - min_opp_dist_all: all nearest-opponent distances (all timesteps, all agents)
      - min_opp_dist_with_ball: nearest-opponent when agent has ball (for dribble decision)
      - space_radius: open space radius around ball carrier
      - goal_dist_ball_carrier: distance from ball carrier to goal
      - sliding_eligible_dist: distances where sliding is a reasonable action
    """
    set_seed(seed)
    env = create_raw_env(render=False)
    controller = HierarchicalController()

    min_opp_dist_all = []
    min_opp_dist_with_ball = []
    space_radius = []
    goal_dist_ball_carrier = []
    sliding_eligible_dist = []

    for ep in range(n_episodes):
        reset_result = env.reset()
        obs_raw = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        game_state = extract_game_state(obs_raw)
        done, step = False, 0

        while not done and step < 3000:
            for i in range(NUM_AGENTS):
                player_pos = get_player_position(game_state, 'left', i)
                ball_pos = get_ball_position(game_state)

                # Min opponent distance
                min_opp = float('inf')
                for j in range(NUM_AGENTS):
                    opp_pos = get_player_position(game_state, 'right', j)
                    d = get_distance(player_pos, opp_pos)
                    if d < min_opp:
                        min_opp = d
                min_opp_dist_all.append(min_opp)

                has_ball = (ball_owned_by_us(game_state, team_id=0) and
                           game_state.get('ball_owned_player', -1) == i)

                if has_ball:
                    min_opp_dist_with_ball.append(min_opp)

                    # Space radius: distance to 2nd nearest opponent (open space proxy)
                    opp_dists = []
                    for j in range(NUM_AGENTS):
                        opp_pos = get_player_position(game_state, 'right', j)
                        opp_dists.append(get_distance(player_pos, opp_pos))
                    opp_dists.sort()
                    second_nearest = opp_dists[1] if len(opp_dists) > 1 else opp_dists[0]
                    space_radius.append(second_nearest)

                    # Distance to goal
                    goal_pos = [1.0, 0.0]
                    goal_dist_ball_carrier.append(get_distance(player_pos, goal_pos))

                # Sliding eligible: agent is defender, opponent has ball nearby, in own half
                role = get_player_role(game_state, 'left', i)
                if role in (ROLE_GK, ROLE_CB, ROLE_LB, ROLE_RB, ROLE_DM):
                    if not ball_owned_by_us(game_state, team_id=0) and ball_pos[0] < 0.0:
                        sliding_eligible_dist.append(min_opp)

            # Use random actions for data collection (we only need distances)
            joint_actions = [np.random.randint(0, 19) for _ in range(NUM_AGENTS)]
            step_result = env.step(joint_actions)
            if len(step_result) == 5:
                obs_raw, _, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                obs_raw, _, done, _ = step_result

            game_state = extract_game_state(obs_raw)
            step += 1

        if (ep + 1) % 20 == 0:
            print(f"  Collected {ep + 1}/{n_episodes} episodes")

    env.close()

    return {
        'min_opp_dist_all': min_opp_dist_all,
        'min_opp_dist_with_ball': min_opp_dist_with_ball,
        'space_radius': space_radius,
        'goal_dist_ball_carrier': goal_dist_ball_carrier,
        'sliding_eligible_dist': sliding_eligible_dist,
    }


def analyze_distributions(dists: Dict[str, List[float]]) -> Dict:
    """Analyze distributions and propose thresholds.

    Threshold logic:
      d_tackle: 90th percentile of sliding_eligible_dist
                 (sliding only when opponent very close)
      d_safe:   10th percentile of space_radius when has ball
                 (dribble only when reasonable space around)
      d_shoot:  10th percentile of goal_dist_ball_carrier
                 (shoot when close to goal, not from far away)
    """
    results = {}

    for key, values in dists.items():
        if not values:
            results[key] = {'count': 0}
            continue
        arr = np.array(values)
        results[key] = {
            'count': len(values),
            'mean': round(float(np.mean(arr)), 4),
            'std': round(float(np.std(arr)), 4),
            'min': round(float(np.min(arr)), 4),
            'max': round(float(np.max(arr)), 4),
            'p5': round(float(np.percentile(arr, 5)), 4),
            'p10': round(float(np.percentile(arr, 10)), 4),
            'p25': round(float(np.percentile(arr, 25)), 4),
            'p50': round(float(np.percentile(arr, 50)), 4),
            'p75': round(float(np.percentile(arr, 75)), 4),
            'p90': round(float(np.percentile(arr, 90)), 4),
            'p95': round(float(np.percentile(arr, 95)), 4),
        }

    # Propose thresholds
    sliding = dists.get('sliding_eligible_dist', [])
    space = dists.get('space_radius', [])
    goal = dists.get('goal_dist_ball_carrier', [])

    proposed = {}
    if sliding:
        arr = np.array(sliding)
        proposed['d_tackle'] = {
            'value': round(float(np.percentile(arr, 90)), 4),
            'basis': '90th percentile of sliding-eligible distances',
            'description': 'Sliding triggered only when opponent within this distance',
        }
    if space:
        arr = np.array(space)
        proposed['d_safe'] = {
            'value': round(float(np.percentile(arr, 10)), 4),
            'basis': '10th percentile of open space radius (2nd nearest opponent)',
            'description': 'Dribble allowed only when at least this much space',
        }
    if goal:
        arr = np.array(goal)
        proposed['d_shoot'] = {
            'value': round(float(np.percentile(arr, 10)), 4),
            'basis': '10th percentile of ball-carrier to goal distance',
            'description': 'Shot triggered only when this close to goal',
        }

    return {
        'distributions': results,
        'proposed_thresholds': proposed,
    }


def validate_thresholds(
    proposed: Dict,
    n_episodes: int = 50,
    seed: int = 42,
) -> Dict:
    """Validate proposed thresholds by running expert policy and measuring win rate.

    Also compare with old hardcoded thresholds (0.05, 0.15, 0.30).
    """
    try:
        from hmarl.utils import set_seed as _set_seed

        results = {}

        # Test multiple threshold sets
        threshold_sets = {
            'calibrated': {
                'd_tackle': proposed.get('d_tackle', {}).get('value', 0.05),
                'd_safe': proposed.get('d_safe', {}).get('value', 0.15),
                'd_shoot': proposed.get('d_shoot', {}).get('value', 0.30),
            },
            'old_hardcoded': {
                'd_tackle': 0.05,
                'd_safe': 0.15,
                'd_shoot': 0.30,
            },
        }

        for name, thresholds in threshold_sets.items():
            _set_seed(seed)
            env = create_raw_env(render=False)
            controller = HierarchicalController()
            expert = ExpertPolicy(
                d_tackle=thresholds['d_tackle'],
                d_safe=thresholds['d_safe'],
                d_shoot=thresholds['d_shoot'],
            )

            wins, draws, losses = 0, 0, 0
            goals_for, goals_against = 0, 0

            for ep in range(n_episodes):
                reset_result = env.reset()
                obs_raw = reset_result[0] if isinstance(reset_result, tuple) else reset_result
                game_state = extract_game_state(obs_raw)
                done, step = False, 0
                ep_gf, ep_ga = 0, 0

                while not done and step < 3000:
                    macro = controller.get_macro_strategy(game_state)
                    sub_goals = controller.get_sub_goals(game_state, macro)
                    ideal_actions = [
                        expert.get_ideal_action(game_state, i, sub_goals[i], macro)
                        for i in range(NUM_AGENTS)
                    ]

                    step_result = env.step(ideal_actions)
                    if len(step_result) == 5:
                        obs_raw, _, terminated, truncated, _ = step_result
                        done = terminated or truncated
                    else:
                        obs_raw, _, done, _ = step_result

                    new_gs = extract_game_state(obs_raw)
                    score = new_gs.get('score', [0, 0])
                    if score[0] > ep_gf:
                        ep_gf = score[0]
                    if score[1] > ep_ga:
                        ep_ga = score[1]

                    game_state = new_gs
                    step += 1

                goals_for += ep_gf
                goals_against += ep_ga
                if ep_gf > ep_ga:
                    wins += 1
                elif ep_gf == ep_ga:
                    draws += 1
                else:
                    losses += 1

            env.close()

            results[name] = {
                'thresholds': thresholds,
                'n_episodes': n_episodes,
                'win_rate': round(wins / n_episodes * 100, 2),
                'draw_rate': round(draws / n_episodes * 100, 2),
                'loss_rate': round(losses / n_episodes * 100, 2),
                'avg_goals_for': round(goals_for / n_episodes, 2),
                'avg_goals_against': round(goals_against / n_episodes, 2),
            }

        return results

    except ImportError as e:
        return {'status': 'skipped', 'reason': str(e)}


def main():
    parser = argparse.ArgumentParser(description="Calibrate Expert Policy Thresholds")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Episodes for data collection")
    parser.add_argument("--validate-episodes", type=int, default=50,
                        help="Episodes for validation")
    parser.add_argument("--scenario", type=str, default="11_vs_11_stochastic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="evaluation_results/calibration.json")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer episodes")
    args = parser.parse_args()

    if args.quick:
        args.episodes = 20
        args.validate_episodes = 10

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("=" * 60)
    print("  EXPERT POLICY THRESHOLD CALIBRATION")
    print("=" * 60)
    print(f"  Scenario: {args.scenario}")
    print(f"  Collection episodes: {args.episodes}")
    print(f"  Validation episodes: {args.validate_episodes}")
    print()

    # Step 1: Collect distributions
    print("Step 1: Collecting distance distributions...")
    dists = collect_distance_distributions(
        n_episodes=args.episodes,
        scenario=args.scenario,
        seed=args.seed,
    )

    # Step 2: Analyze
    print("\nStep 2: Analyzing distributions...")
    analysis = analyze_distributions(dists)

    print("\n  Distribution summaries:")
    for key, stats in analysis['distributions'].items():
        if stats.get('count', 0) > 0:
            print(f"    {key:35s} n={stats['count']:>8,}  "
                  f"mean={stats['mean']:.4f}  "
                  f"p10={stats['p10']:.4f}  p50={stats['p50']:.4f}  p90={stats['p90']:.4f}")

    print("\n  Proposed thresholds:")
    for param, info in analysis['proposed_thresholds'].items():
        print(f"    {param} = {info['value']:.4f}")
        print(f"      Basis: {info['basis']}")

    # Step 3: Validate
    print(f"\nStep 3: Validating thresholds ({args.validate_episodes} episodes each)...")
    validation = validate_thresholds(
        analysis['proposed_thresholds'],
        n_episodes=args.validate_episodes,
        seed=args.seed,
    )

    print("\n  Validation results:")
    for name, res in validation.items():
        if isinstance(res, dict) and 'win_rate' in res:
            print(f"    {name:20s} WR={res['win_rate']:5.1f}%  "
                  f"GF={res['avg_goals_for']:.1f}  GA={res['avg_goals_against']:.1f}  "
                  f"thresholds={res['thresholds']}")

    # Save
    report = {
        'config': {
            'episodes': args.episodes,
            'validate_episodes': args.validate_episodes,
            'scenario': args.scenario,
            'seed': args.seed,
        },
        'analysis': analysis,
        'validation': validation,
    }

    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Report saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
