"""Expert Policy Diagnostic Script.

Runs expert policy against GRF bot and collects detailed behavioral data
to understand WHY the expert is not competitive (win rate, goals, etc.).

Collects per-episode:
- Action distribution per role (11 players × 19 actions)
- Ball possession time per player
- Pass attempts / completions / progressive passes
- Shots taken, shots on target
- Goals for / against, conceded position
- Ball position heatmap (x, y grid occupancy)
- Time spent in each pitch zone
- Sub-goal activation frequency
- Macro strategy activation frequency

Usage (inside Docker):
    python scripts/diagnose_expert.py --episodes 5 --output expert_diagnostics.json
    python scripts/diagnose_expert.py --episodes 3 --quick
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hmarl.env import (
    create_raw_env, extract_game_state, NUM_AGENTS, ACTION_SPACE_SIZE,
    get_ball_position, get_player_position, get_player_role,
    get_distance, ball_owned_by_us,
    ROLE_NAMES, ACTION_NAMES,
    ROLE_GK, ROLE_CB, ROLE_LB, ROLE_RB, ROLE_DM, ROLE_CM,
    ROLE_LM, ROLE_RM, ROLE_AM, ROLE_CF,
)
from hmarl.policy import (
    HierarchicalController,
    STRATEGY_HIGH_PRESSING, STRATEGY_COUNTER_ATTACK, STRATEGY_POSSESSION,
    SUBGOAL_ZONAL_MARKING, SUBGOAL_BUILD_UP, SUBGOAL_WING_ATTACK,
    SUBGOAL_MAN_MARKING, SUBGOAL_CLEARANCE,
)
from hmarl.expert import ExpertPolicy
from hmarl.utils import set_seed

# Tactical unit grouping
DEFENSE_ROLES = {ROLE_GK, ROLE_CB, ROLE_LB, ROLE_RB}
MIDFIELD_ROLES = {ROLE_DM, ROLE_CM, ROLE_AM}
ATTACK_ROLES = {ROLE_LM, ROLE_RM, ROLE_CF}

STRATEGY_NAMES = {
    STRATEGY_HIGH_PRESSING: 'High Pressing',
    STRATEGY_COUNTER_ATTACK: 'Counter Attack',
    STRATEGY_POSSESSION: 'Possession Play',
}

SUBGOAL_NAMES = {
    SUBGOAL_ZONAL_MARKING: 'Zonal Marking',
    SUBGOAL_BUILD_UP: 'Build-up',
    SUBGOAL_WING_ATTACK: 'Wing Attack',
    SUBGOAL_MAN_MARKING: 'Man Marking',
    SUBGOAL_CLEARANCE: 'Clearance',
}


def classify_tactical_unit(role: int) -> str:
    """Classify role into tactical unit."""
    if role in DEFENSE_ROLES:
        return 'defense'
    elif role in MIDFIELD_ROLES:
        return 'midfield'
    elif role in ATTACK_ROLES:
        return 'attack'
    return 'other'


def get_pitch_zone(x: float, y: float) -> str:
    """Classify position into pitch zone (3×3 grid)."""
    x_zone = 'defensive' if x < -0.33 else ('midfield' if x < 0.33 else 'attacking')
    y_zone = 'left' if y < -0.14 else ('center' if y < 0.14 else 'right')
    return f'{x_zone}_{y_zone}'


def diagnose_episode(
    env,
    controller: HierarchicalController,
    expert: ExpertPolicy,
    episode_idx: int,
    max_steps: int = 3000,
) -> Dict:
    """Run one diagnostic episode and collect detailed behavioral data."""
    reset_result = env.reset()
    obs_raw = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    game_state = extract_game_state(obs_raw)

    # Per-role action counters: role_id -> action_id -> count
    role_action_counts = defaultdict(lambda: defaultdict(int))
    # Ball possession: player_idx -> steps with ball
    possession_steps = [0] * NUM_AGENTS
    # Tactical unit action counts
    unit_action_counts = defaultdict(lambda: defaultdict(int))
    # Pass tracking
    pass_attempts = 0
    pass_completions = 0
    pass_progressive = 0
    last_possessor = -1
    last_possessor_pos = None
    # Shot tracking
    shots_taken = 0
    shots_on_target = 0
    # Goals
    goals_for = 0
    goals_against = 0
    conceded_positions = []
    # Ball position tracking
    ball_positions = []
    # Strategy/sub-goal activation
    strategy_counts = defaultdict(int)
    subgoal_counts = defaultdict(int)
    subgoal_per_role = defaultdict(lambda: defaultdict(int))
    # Zone occupancy per unit
    unit_zone_time = defaultdict(lambda: defaultdict(int))
    # Position tracking per role
    role_positions = defaultdict(list)

    done, step = False, 0
    prev_game_state = None

    while not done and step < max_steps:
        # High-level
        macro = controller.get_macro_strategy(game_state)
        strategy_counts[macro] += 1

        # Mid-level
        sub_goals = controller.get_sub_goals(game_state, macro)
        for i, sg in enumerate(sub_goals):
            subgoal_counts[sg] += 1
            role = get_player_role(game_state, 'left', i)
            subgoal_per_role[role][sg] += 1

        # Expert ideal actions
        ideal_actions = [
            expert.get_ideal_action(game_state, i, sub_goals[i], macro)
            for i in range(NUM_AGENTS)
        ]

        # Record per-agent data
        ball_pos = get_ball_position(game_state)
        ball_positions.append(ball_pos[:2])

        for i in range(NUM_AGENTS):
            role = get_player_role(game_state, 'left', i)
            unit = classify_tactical_unit(role)
            pos = get_player_position(game_state, 'left', i)

            # Action distribution
            role_action_counts[role][ideal_actions[i]] += 1
            unit_action_counts[unit][ideal_actions[i]] += 1

            # Zone occupancy
            zone = get_pitch_zone(pos[0], pos[1])
            unit_zone_time[unit][zone] += 1

            # Position tracking
            role_positions[role].append(pos[:2])

            # Possession
            if ball_owned_by_us(game_state, team_id=0) and \
               game_state.get('ball_owned_player', -1) == i:
                possession_steps[i] += 1
                if last_possessor != i:
                    # Check if this is a pass completion
                    if last_possessor >= 0 and prev_game_state is not None:
                        prev_team = prev_game_state.get('ball_owned_team', -1)
                        if prev_team == 0:
                            pass_completions += 1
                            # Progressive check
                            if last_possessor_pos is not None and pos[0] > last_possessor_pos[0]:
                                pass_progressive += 1
                    last_possessor = i
                    last_possessor_pos = pos[:2]
                else:
                    last_possessor_pos = pos[:2]

            # Shot detection
            if ideal_actions[i] == 12:  # ACT_SHOT
                shots_taken += 1
                dist_to_goal = get_distance(pos, [1.0, 0.0])
                if dist_to_goal < 0.5:
                    shots_on_target += 1

            # Pass attempt detection
            if ideal_actions[i] in (9, 10, 11):  # long_pass, high_pass, short_pass
                pass_attempts += 1

        # Execute
        step_result = env.step(ideal_actions)
        if len(step_result) == 5:
            obs_raw, _, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            obs_raw, _, done, _ = step_result

        new_gs = extract_game_state(obs_raw)
        score = new_gs.get('score', [0, 0])

        # Detect goals
        if score[0] > goals_for:
            goals_for = score[0]
        if score[1] > goals_against:
            goals_against = score[1]
            conceded_positions.append(get_ball_position(new_gs)[:2])

        prev_game_state = game_state
        game_state = new_gs
        step += 1

    # Compute aggregates
    total_possession = sum(possession_steps)
    possession_pct = [round(p / max(step, 1) * 100, 1) for p in possession_steps]

    # Per-role summaries
    role_summaries = {}
    for role, action_counts in role_action_counts.items():
        total = sum(action_counts.values())
        if total == 0:
            continue
        # Dominant actions
        top_actions = sorted(action_counts.items(), key=lambda x: -x[1])[:5]
        role_summaries[ROLE_NAMES.get(role, f'role_{role}')] = {
            'total_steps': total,
            'top_actions': [
                {'action': ACTION_NAMES[a], 'count': c, 'pct': round(c / total * 100, 1)}
                for a, c in top_actions
            ],
        }

    # Per-unit summaries
    unit_summaries = {}
    for unit, action_counts in unit_action_counts.items():
        total = sum(action_counts.values())
        if total == 0:
            continue
        top_actions = sorted(action_counts.items(), key=lambda x: -x[1])[:3]
        unit_summaries[unit] = {
            'total_steps': total,
            'top_actions': [
                {'action': ACTION_NAMES[a], 'count': c, 'pct': round(c / total * 100, 1)}
                for a, c in top_actions
            ],
        }

    # Ball position heatmap (3×3 grid)
    heatmap = defaultdict(int)
    for bx, by in ball_positions:
        zone = get_pitch_zone(bx, by)
        heatmap[zone] += 1

    return {
        'episode': episode_idx,
        'steps': step,
        'goals_for': goals_for,
        'goals_against': goals_against,
        'conceded_positions': conceded_positions,
        'posession_pct_per_player': possession_pct,
        'possession_total_steps': total_possession,
        'pass_attempts': pass_attempts,
        'pass_completions': pass_completions,
        'pass_progressive': pass_progressive,
        'pass_completion_rate': round(pass_completions / max(pass_attempts, 1) * 100, 1),
        'shots_taken': shots_taken,
        'shots_on_target': shots_on_target,
        'shot_accuracy': round(shots_on_target / max(shots_taken, 1) * 100, 1),
        'strategy_distribution': {STRATEGY_NAMES[k]: v for k, v in strategy_counts.items()},
        'subgoal_distribution': {SUBGOAL_NAMES[k]: v for k, v in subgoal_counts.items()},
        'role_summaries': role_summaries,
        'unit_summaries': unit_summaries,
        'ball_heatmap': dict(heatmap),
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose Expert Policy Behavior")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes to run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="evaluation_results/expert_diagnostics.json")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 3 episodes")
    parser.add_argument("--max-steps", type=int, default=3000)
    args = parser.parse_args()

    if args.quick:
        args.episodes = 3

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    print("=" * 60)
    print("  EXPERT POLICY DIAGNOSTIC")
    print("=" * 60)
    print(f"  Episodes: {args.episodes}")
    print(f"  Seed: {args.seed}")
    print()

    set_seed(args.seed)
    env = create_raw_env(env_name="11_vs_11_stochastic", num_agents=NUM_AGENTS, render=False)
    controller = HierarchicalController()
    expert = ExpertPolicy(d_tackle=0.05, d_safe=0.15, d_shoot=0.30)

    all_episodes = []
    total_gf = total_ga = 0
    total_pass_att = total_pass_comp = total_pass_prog = 0
    total_shots = total_shots_ot = 0

    t0 = time.time()
    for ep in range(args.episodes):
        diag = diagnose_episode(env, controller, expert, ep, args.max_steps)
        all_episodes.append(diag)
        total_gf += diag['goals_for']
        total_ga += diag['goals_against']
        total_pass_att += diag['pass_attempts']
        total_pass_comp += diag['pass_completions']
        total_pass_prog += diag['pass_progressive']
        total_shots += diag['shots_taken']
        total_shots_ot += diag['shots_on_target']

        result = 'W' if diag['goals_for'] > diag['goals_against'] else \
                 'D' if diag['goals_for'] == diag['goals_against'] else 'L'
        print(f"  Ep {ep}: {result} {diag['goals_for']}-{diag['goals_against']} "
              f"| Pass: {diag['pass_completions']}/{diag['pass_attempts']} "
              f"| Shots: {diag['shots_taken']} (OT: {diag['shots_on_target']}) "
              f"| Steps: {diag['steps']}", flush=True)

    env.close()
    elapsed = time.time() - t0

    # Aggregate
    n = args.episodes
    wins = sum(1 for e in all_episodes if e['goals_for'] > e['goals_against'])
    draws = sum(1 for e in all_episodes if e['goals_for'] == e['goals_against'])
    losses = n - wins - draws

    # Aggregate role summaries across episodes
    agg_role_actions = defaultdict(lambda: defaultdict(int))
    agg_unit_actions = defaultdict(lambda: defaultdict(int))
    agg_heatmap = defaultdict(int)
    agg_strategy = defaultdict(int)
    agg_subgoal = defaultdict(int)

    for ep_data in all_episodes:
        for role_str, summary in ep_data['role_summaries'].items():
            for ta in summary['top_actions']:
                agg_role_actions[role_str][ta['action']] += ta['count']
        for unit, summary in ep_data['unit_summaries'].items():
            for ta in summary['top_actions']:
                agg_unit_actions[unit][ta['action']] += ta['count']
        for zone, count in ep_data['ball_heatmap'].items():
            agg_heatmap[zone] += count
        for strat, count in ep_data['strategy_distribution'].items():
            agg_strategy[strat] += count
        for sg, count in ep_data['subgoal_distribution'].items():
            agg_subgoal[sg] += count

    # Print summary
    print()
    print("=" * 60)
    print("  AGGREGATE DIAGNOSTICS")
    print("=" * 60)
    print(f"  W/D/L: {wins}/{draws}/{losses} | WR: {100*wins/n:.0f}%")
    print(f"  GF/ep: {total_gf/n:.2f} | GA/ep: {total_ga/n:.2f}")
    print(f"  Pass: {total_pass_comp}/{total_pass_att} completed ({100*total_pass_comp/max(total_pass_att,1):.1f}%), "
          f"{total_pass_prog} progressive")
    print(f"  Shots: {total_shots} (on target: {total_shots_ot}, "
          f"accuracy: {100*total_shots_ot/max(total_shots,1):.1f}%)")
    print(f"  Time: {elapsed:.0f}s")
    print()

    print("  --- Strategy Distribution ---")
    for strat, count in sorted(agg_strategy.items(), key=lambda x: -x[1]):
        print(f"    {strat}: {count} ({100*count/max(sum(agg_strategy.values()),1):.1f}%)")

    print()
    print("  --- Sub-Goal Distribution ---")
    for sg, count in sorted(agg_subgoal.items(), key=lambda x: -x[1]):
        print(f"    {sg}: {count} ({100*count/max(sum(agg_subgoal.values()),1):.1f}%)")

    print()
    print("  --- Ball Heatmap (3×3 grid) ---")
    zones = ['defensive_left', 'defensive_center', 'defensive_right',
             'midfield_left', 'midfield_center', 'midfield_right',
             'attacking_left', 'attacking_center', 'attacking_right']
    total_ball = max(sum(agg_heatmap.values()), 1)
    for zone in zones:
        count = agg_heatmap.get(zone, 0)
        bar = '█' * int(count / total_ball * 40)
        print(f"    {zone:20s}: {count:5d} ({100*count/total_ball:5.1f}%) {bar}")

    print()
    print("  --- Role Action Summary (top 3 per role) ---")
    for role_str in ['GK', 'CB', 'LB', 'RB', 'DM', 'CM', 'LM', 'RM', 'AM', 'CF']:
        if role_str not in agg_role_actions:
            continue
        actions = agg_role_actions[role_str]
        total = max(sum(actions.values()), 1)
        top3 = sorted(actions.items(), key=lambda x: -x[1])[:3]
        top3_str = ', '.join(f'{a}:{c}({100*c/total:.0f}%)' for a, c in top3)
        print(f"    {role_str:3s}: {top3_str}")

    print()
    print("  --- Unit Action Summary ---")
    for unit in ['defense', 'midfield', 'attack']:
        if unit not in agg_unit_actions:
            continue
        actions = agg_unit_actions[unit]
        total = max(sum(actions.values()), 1)
        top3 = sorted(actions.items(), key=lambda x: -x[1])[:3]
        top3_str = ', '.join(f'{a}:{c}({100*c/total:.0f}%)' for a, c in top3)
        print(f"    {unit:10s}: {top3_str}")

    # Save JSON
    report = {
        'config': {
            'episodes': args.episodes,
            'seed': args.seed,
            'max_steps': args.max_steps,
            'thresholds': {'d_tackle': 0.05, 'd_safe': 0.15, 'd_shoot': 0.30},
        },
        'aggregate': {
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'win_rate': round(100 * wins / n, 1),
            'goals_for_per_ep': round(total_gf / n, 2),
            'goals_against_per_ep': round(total_ga / n, 2),
            'pass_attempts_total': total_pass_att,
            'pass_completions_total': total_pass_comp,
            'pass_progressive_total': total_pass_prog,
            'pass_completion_rate': round(100 * total_pass_comp / max(total_pass_att, 1), 1),
            'shots_total': total_shots,
            'shots_on_target_total': total_shots_ot,
        },
        'episodes': all_episodes,
    }

    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Report saved: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
