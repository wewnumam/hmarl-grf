"""HMARL Evaluation Script.

Evaluates trained models across all metrics defined in the thesis:
- Performance: Win Rate, Goal Difference, Cumulative Reward
- Coordination: PSR, PPR, Positional Entropy, Team Compactness, FAI
- Role Coherence: RCI_strict, RCI_cat

Scenarios (from thesis BAB_4):
1. Training convergence (learning curve)
2. Competitive performance comparison (vs IPPO, random)
3. Tactical coordination and RCI evaluation
"""

import argparse
import json
import os
import traceback
import sys
import time
from typing import Dict, List, Optional

# Ensure project root is on path so `import hmarl` works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from hmarl.utils import (
    OBS_DIM, HIDDEN_DIM, HEAD_DIM, ACTION_SPACE_SIZE, EPISODE_MAX_STEPS,
    extract_obs_vector as _utils_obs,
    load_hmarl_checkpoint as _load_ckpt,
    ProgressTracker,
)

from hmarl.env import create_raw_env, NUM_AGENTS, extract_game_state
from hmarl.policy import (
    HierarchicalController, HierarchicalActorCritic,
    SubGoalEmbedding, SUBGOAL_EMBED_DIM,
)
from hmarl.expert import ExpertPolicyAllAgents
from hmarl.reward import PassTracker, RCITracker, compute_hierarchical_reward
from hmarl.metrics import _convex_hull_area_numpy
from hmarl.metrics import (
    compute_all_metrics, print_metrics,
    compute_win_rate, compute_goal_difference,
    team_compactness,
    pass_network_matrix,
    inter_agent_distance_per_line,
    convex_hull_area,
    action_transition_matrix,
    ball_progression_rate,
)
from hmarl.rci import compute_rci
from evaluation.visualizations import (
    plot_role_heatmap,
    plot_formation_snapshot,
    plot_action_distribution,
    plot_macro_strategy_timeline,
    plot_compactness_over_time,
    plot_comparative_bars,
    plot_metric_correlation,
    plot_pass_network,
    plot_iad_per_line,
    plot_convex_hull,
    plot_action_transitions,
    plot_bpr_distribution,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_hmarl_model(checkpoint_path: str):
    """Load trained HMARL model."""
    policy, subgoal_emb, meta = _load_ckpt(checkpoint_path, device=str(DEVICE))
    print(f"Model loaded from {checkpoint_path}")
    print(f"  Trained for {meta.get('global_step', '?')} steps")
    if 'git_hash' in meta:
        print(f"  Git: {meta['git_hash']}")
    if 'hyperparams' in meta:
        hp = meta['hyperparams']
        print(f"  LR: {hp.get('learning_rate', '?')}, Gamma: {hp.get('gamma', '?')}")
    return policy, subgoal_emb


def _extract_dict_from_simple(obs):
    flat = obs[0]
    ball_x, ball_y, ball_z = float(flat[0]), float(flat[1]), float(flat[2])
    ball_owned_team = int(round(float(flat[9])))
    ball_owned_player = int(round(float(flat[10])))
    left_roles = []; left_team = []; idx = 11
    for p in range(11):
        idx += 1; x, y = float(flat[idx]), float(flat[idx+1])
        idx += 3; idx += 3
        role = int(round(float(flat[idx]))); idx += 1
        left_team.append([x, y]); left_roles.append(role)
    return {'ball': [ball_x, ball_y, ball_z], 'ball_owned_team': ball_owned_team,
            'ball_owned_player': ball_owned_player, 'left_team': left_team, 'left_team_roles': left_roles}


def get_obs_vector(obs_or_state, player_idx):
    """Extract 115-dim observation vector from raw dict."""
    # GRF raw obs is list[dict], take first element
    if isinstance(obs_or_state, list) and len(obs_or_state) > 0 and isinstance(obs_or_state[0], dict):
        game_state = obs_or_state[0]
    elif isinstance(obs_or_state, dict):
        game_state = obs_or_state
    else:
        game_state = {}
    return _utils_obs(game_state, player_idx, OBS_DIM)


def evaluate_hmarl(
    policy: HierarchicalActorCritic,
    subgoal_embedding: SubGoalEmbedding,
    num_episodes: int = 100,
    render: bool = False,
    output_dir: str = "evaluation_results",
) -> Dict:
    """Run full HMARL evaluation.

    Evaluates all three scenarios:
    1. Per-episode performance metrics
    2. Coordination metrics
    3. RCI across all episodes
    """
    os.makedirs(output_dir, exist_ok=True)
    DEVICE = next(policy.parameters()).device

    env = create_raw_env(render=render)
    controller = HierarchicalController()
    expert = ExpertPolicyAllAgents()

    all_match_results = []
    all_goals_for = []
    all_goals_against = []
    all_rewards = []
    ep_reward_components = []  # per-episode reward component sums
    ep_episode_rc = {'game_reward': 0.0, 'r_high': 0.0, 'r_mid': 0.0, 'r_low': 0.0}

    # Aggregate actions/states for RCI across all episodes
    all_actual_actions_flat = []
    all_ideal_actions_flat = []
    all_game_states_flat = []

    # Per-timestep data for visualizations
    all_strategies = []
    all_sub_goals = []
    all_compactness_ts = []
    all_convex_hull_ts = []
    all_roles = None

    print(f"\nEvaluating HMARL over {num_episodes} episodes...")
    print(f"{'='*60}")
    progress = ProgressTracker(num_episodes, label="Eval HMARL", print_every=10)

    for ep in range(num_episodes):
        reset_result = env.reset()
        obs_raw = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        game_state = extract_game_state(obs_raw)

        pass_tracker = PassTracker()
        rci_tracker = RCITracker(NUM_AGENTS)
        episode_reward = 0.0
        ep_episode_rc = {'game_reward': 0.0, 'r_high': 0.0, 'r_mid': 0.0, 'r_low': 0.0}
        episode_actual_actions = []
        episode_ideal_actions = []
        episode_game_states = []

        prev_game_state = None
        done = False
        step = 0

        while not done and step < EPISODE_MAX_STEPS:
            macro = controller.get_macro_strategy(game_state)
            sub_goals = controller.get_sub_goals(game_state, macro)
            ideal_actions = expert.get_ideal_actions(game_state, sub_goals, macro)

            # Track strategies and sub-goals for visualization
            all_strategies.append(macro)
            all_sub_goals.append(list(sub_goals))
            if all_roles is None:
                roles_raw = game_state.get('left_team_roles', list(range(11)))
                all_roles = [int(r) for r in roles_raw[:11]]

            # Track compactness per timestep
            if ep == 0:  # Only first episode for per-timestep plots
                tc_key = 'left_team'
                tc_positions = game_state.get(tc_key, [])
                if len(tc_positions) >= 11:
                    pos_arr = np.array(tc_positions[:11])
                    scaled = np.column_stack([
                        (pos_arr[:, 0] + 1.0) * 60.0,
                        (pos_arr[:, 1] + 0.42) * (80.0 / 0.84),
                    ])
                    centroid = scaled.mean(axis=0)
                    rho = float(np.sqrt(np.mean(np.sum((scaled - centroid)**2, axis=1))))
                    all_compactness_ts.append(rho)
                    # Convex hull area per timestep
                    all_convex_hull_ts.append(_convex_hull_area_numpy(scaled))

            joint_actions = []
            for i in range(NUM_AGENTS):
                obs_vec = get_obs_vector(obs_raw, i)
                with torch.no_grad():
                    sg_embed_val = subgoal_embedding(
                        torch.LongTensor([sub_goals[i]]).to(DEVICE)
                    ).cpu().numpy().flatten()

                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs_vec).unsqueeze(0).to(DEVICE)
                    sg_t = torch.FloatTensor(sg_embed_val).unsqueeze(0).to(DEVICE)
                    logits, _ = policy(obs_t, sg_t)
                    action = logits.argmax(dim=-1).item()
                joint_actions.append(action)

            step_result = env.step(joint_actions)
            if len(step_result) == 5:
                obs_raw_new, game_reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs_raw_new, game_reward, done, info = step_result
            team_reward = float(np.sum(game_reward))
            if isinstance(obs_raw_new, np.ndarray) and obs_raw_new.ndim == 2:
                new_game_state = _extract_dict_from_simple(obs_raw_new)
            else:
                new_game_state = extract_game_state(obs_raw_new)

            pass_tracker.update(new_game_state, prev_game_state)
            total_reward, reward_breakdown = compute_hierarchical_reward(
                team_reward, new_game_state, joint_actions, ideal_actions,
                pass_tracker, rci_tracker,
            )

            episode_reward += total_reward
            for k in ep_episode_rc:
                ep_episode_rc[k] += reward_breakdown.get(k, 0.0)
            episode_actual_actions.append(joint_actions)
            episode_ideal_actions.append(ideal_actions)
            episode_game_states.append(new_game_state)

            prev_game_state = game_state
            game_state = new_game_state
            obs_raw = obs_raw_new
            step += 1

        # Determine match result
        score = game_state.get('score', [0, 0])
        if isinstance(score, (list, tuple)) and len(score) >= 2:
            gf, ga = score[0], score[1]
        else:
            gf, ga = 0, 0

        result = 'win' if gf > ga else ('loss' if gf < ga else 'draw')

        all_match_results.append(result)
        all_goals_for.append(gf)
        all_goals_against.append(ga)
        all_rewards.append(episode_reward)
        ep_reward_components.append({k: float(v) for k, v in ep_episode_rc.items()})

        all_actual_actions_flat.extend(episode_actual_actions)
        all_ideal_actions_flat.extend(episode_ideal_actions)
        all_game_states_flat.extend(episode_game_states)

        avg_reward = np.mean(all_rewards)
        wr = compute_win_rate(all_match_results)
        progress.update(extra=f"WR: {wr:.1f}% AvgR: {avg_reward:.1f}")
    progress.done()

    # Compute aggregate metrics
    metrics = compute_all_metrics(
        actual_actions=all_actual_actions_flat,
        game_states=all_game_states_flat,
        ideal_actions=all_ideal_actions_flat,
        match_result='win' if compute_win_rate(all_match_results) > 50 else 'loss',
        goals_for=sum(all_goals_for),
        goals_against=sum(all_goals_against),
        cumulative_reward=sum(all_rewards),
    )

    # Override per-episode aggregation
    metrics['win_rate'] = compute_win_rate(all_match_results)
    metrics['goal_difference'] = compute_goal_difference(all_goals_for, all_goals_against)
    metrics['goals_for'] = sum(all_goals_for)
    metrics['goals_against'] = sum(all_goals_against)
    metrics['cumulative_reward'] = sum(all_rewards)
    metrics['num_episodes'] = num_episodes

    # Print results
    print_metrics(metrics, "HMARL")

    # Save results
    results_path = os.path.join(output_dir, "hmarl_results.json")
    serializable = {k: v for k, v in metrics.items()
                    if not isinstance(v, (list, np.ndarray)) or
                    (isinstance(v, list) and all(isinstance(x, (int, float)) for x in v))}
    # Convert numpy types
    for k, v in serializable.items():
        if isinstance(v, (np.floating, np.integer)):
            serializable[k] = float(v)
        elif isinstance(v, list):
            serializable[k] = [float(x) for x in v]

    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {results_path}")

    # Per-episode breakdown
    episode_data = {
        'match_results': all_match_results,
        'goals_for': all_goals_for,
        'goals_against': all_goals_against,
        'rewards': [float(r) for r in all_rewards],
    }
    ep_path = os.path.join(output_dir, "hmarl_episodes.json")
    with open(ep_path, 'w') as f:
        json.dump(episode_data, f, indent=2)

    # Save per-episode reward components for plot 11
    rc_path = os.path.join(output_dir, "reward_components.json")
    rc_data = {
        'game_reward': [d['game_reward'] for d in ep_reward_components],
        'r_high': [d['r_high'] for d in ep_reward_components],
        'r_mid': [d['r_mid'] for d in ep_reward_components],
        'r_low': [d['r_low'] for d in ep_reward_components],
    }
    with open(rc_path, 'w') as f:
        json.dump(rc_data, f)

    # Save per-timestep strategies and sub_goals for plot 10
    strat_path = os.path.join(output_dir, "strategies.json")
    with open(strat_path, 'w') as f:
        json.dump(all_strategies, f)
    sg_path = os.path.join(output_dir, "sub_goals.json")
    with open(sg_path, 'w') as f:
        json.dump(all_sub_goals, f)

    # Save per-timestep convex hull areas for plot 15
    if all_convex_hull_ts:
        ch_path = os.path.join(output_dir, "convex_hull_ts.json")
        with open(ch_path, 'w') as f:
            json.dump(all_convex_hull_ts, f)

    # --- Generate visualizations ---
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    try:
        print(f"\nGenerating visualizations...")
        # Role heatmap
        if all_game_states_flat:
            plot_role_heatmap(
                all_game_states_flat[:3000],  # Cap at ~1 episode for speed
                os.path.join(plots_dir, "04_role_heatmap.png"),
            )
            # Formation snapshots (start, mid, late)
            for ts, label in [(0, 'start'), (min(1500, len(all_game_states_flat)//2), 'mid'),
                              (min(2999, len(all_game_states_flat)-1), 'late')]:
                plot_formation_snapshot(
                    all_game_states_flat,
                    os.path.join(plots_dir, f"05_formation_{label}.png"),
                    timestep=ts,
                    title=f"Formation Snapshot ({label})",
                )
        # Action distribution
        if all_actual_actions_flat and all_roles:
            plot_action_distribution(
                all_actual_actions_flat[:3000],
                all_roles,
                os.path.join(plots_dir, "06_action_distribution.png"),
            )
        # Strategy timeline
        if all_strategies:
            plot_macro_strategy_timeline(
                all_strategies[:3000],
                os.path.join(plots_dir, "07_strategy_timeline.png"),
            )
        # Compactness over time
        if all_compactness_ts:
            plot_compactness_over_time(
                {'HMARL': all_compactness_ts},
                os.path.join(plots_dir, "08_compactness.png"),
            )
        print(f"Visualizations saved to {plots_dir}")

        # --- Extended metrics and visualizations ---
        # Pass network
        pn = pass_network_matrix(all_game_states_flat, all_actual_actions_flat)
        if pn.sum() > 0:
            plot_pass_network(
                pn,
                os.path.join(plots_dir, '13_pass_network.png'),
                roles=all_roles,
            )

        # Inter-agent distance per line
        iad = inter_agent_distance_per_line(all_game_states_flat)
        metrics['defence_midfield_gap'] = iad['defence_midfield_gap']
        metrics['midfield_attack_gap'] = iad['midfield_attack_gap']
        metrics['overall_spread'] = iad['overall_spread']

        # Convex hull area
        ch_mean, ch_std, ch_min, ch_max = convex_hull_area(all_game_states_flat)
        metrics['convex_hull_mean'] = ch_mean
        metrics['convex_hull_std'] = ch_std
        metrics['convex_hull_min'] = ch_min
        metrics['convex_hull_max'] = ch_max

        # Action transition matrix
        _, trans_probs = action_transition_matrix(all_actual_actions_flat)
        plot_action_transitions(
            trans_probs,
            os.path.join(plots_dir, '16_action_transitions.png'),
            title='Action Transition Probabilities (HMARL)',
        )

        # Ball progression rate
        bpr_mean, bpr_std = ball_progression_rate(all_game_states_flat)
        metrics['bpr_mean'] = bpr_mean
        metrics['bpr_std'] = bpr_std

        # Update JSON with extended metrics
        serializable_ext = {}
        for k in ('defence_midfield_gap', 'midfield_attack_gap', 'overall_spread',
                   'convex_hull_mean', 'convex_hull_std', 'convex_hull_min', 'convex_hull_max',
                   'bpr_mean', 'bpr_std'):
            v = metrics.get(k, 0)
            if isinstance(v, (np.floating, np.integer)):
                v = float(v)
            serializable_ext[k] = v
        with open(results_path, 'r') as f:
            existing = json.load(f)
        existing.update(serializable_ext)
        with open(results_path, 'w') as f:
            json.dump(existing, f, indent=2)
        print(f"Extended metrics saved to {results_path}")
    except Exception as e:
        print(f"Warning: Extended visualization generation failed: {e}")
        traceback.print_exc()

    env.close()
    return metrics


def evaluate_random_baseline(
    num_episodes: int = 100,
    render: bool = False,
    output_dir: str = "evaluation_results",
) -> Dict:
    """Evaluate random action baseline."""
    import random

    os.makedirs(output_dir, exist_ok=True)
    env = create_raw_env(render=render)
    expert = ExpertPolicyAllAgents()
    controller = HierarchicalController()

    all_match_results = []
    all_goals_for = []
    all_goals_against = []
    all_rewards = []
    all_actual_actions_flat = []
    all_ideal_actions_flat = []
    all_game_states_flat = []

    print(f"\nEvaluating Random Baseline over {num_episodes} episodes...")
    progress_r = ProgressTracker(num_episodes, label="Eval Random", print_every=10)

    for ep in range(num_episodes):
        reset_result = env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs_raw, info = reset_result
        else:
            obs_raw = reset_result; info = {}
        if isinstance(obs_raw, np.ndarray) and obs_raw.ndim == 2:
            game_state = _extract_dict_from_simple(obs_raw)
        else:
            game_state = extract_game_state(obs_raw)

        pass_tracker = PassTracker()
        rci_tracker = RCITracker(NUM_AGENTS)
        episode_reward = 0.0
        episode_actions = []
        episode_ideals = []
        episode_states = []

        prev_gs = None
        done = False
        step = 0

        while not done and step < EPISODE_MAX_STEPS:
            macro = controller.get_macro_strategy(game_state)
            sub_goals = controller.get_sub_goals(game_state, macro)
            ideal_actions = expert.get_ideal_actions(game_state, sub_goals, macro)

            # Random actions
            joint_actions = [random.randint(0, 18) for _ in range(11)]

            step_result = env.step(joint_actions)
            if len(step_result) == 5:
                obs_raw, game_reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs_raw, game_reward, done, info = step_result
            team_reward = float(np.sum(game_reward))
            if isinstance(obs_raw, np.ndarray) and obs_raw.ndim == 2:
                new_gs = _extract_dict_from_simple(obs_raw)
            else:
                new_gs = extract_game_state(obs_raw)

            pass_tracker.update(new_gs, prev_gs)
            total_reward, reward_breakdown = compute_hierarchical_reward(
                team_reward, new_gs, joint_actions, ideal_actions,
                pass_tracker, rci_tracker,
            )

            episode_reward += total_reward
            episode_actions.append(joint_actions)
            episode_ideals.append(ideal_actions)
            episode_states.append(new_gs)

            prev_gs = game_state
            game_state = new_gs
            step += 1

        score = game_state.get('score', [0, 0])
        gf, ga = score[0], score[1] if isinstance(score, (list, tuple)) and len(score) >= 2 else (0, 0)
        result = 'win' if gf > ga else ('loss' if gf < ga else 'draw')

        all_match_results.append(result)
        all_goals_for.append(gf)
        all_goals_against.append(ga)
        all_rewards.append(episode_reward)
        all_actual_actions_flat.extend(episode_actions)
        all_ideal_actions_flat.extend(episode_ideals)
        all_game_states_flat.extend(episode_states)
        progress_r.update(extra=f"({gf}-{ga})")
    progress_r.done()

    metrics = compute_all_metrics(
        actual_actions=all_actual_actions_flat,
        game_states=all_game_states_flat,
        ideal_actions=all_ideal_actions_flat,
        match_result='win' if compute_win_rate(all_match_results) > 50 else 'loss',
        goals_for=sum(all_goals_for),
        goals_against=sum(all_goals_against),
        cumulative_reward=sum(all_rewards),
    )
    metrics['win_rate'] = compute_win_rate(all_match_results)
    metrics['goal_difference'] = compute_goal_difference(all_goals_for, all_goals_against)
    metrics['goals_for'] = sum(all_goals_for)
    metrics['goals_against'] = sum(all_goals_against)
    metrics['cumulative_reward'] = sum(all_rewards)
    metrics['num_episodes'] = num_episodes

    print_metrics(metrics, "Random Baseline")

    results_path = os.path.join(output_dir, "random_results.json")
    serializable = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            serializable[k] = float(v)
        elif isinstance(v, (int, float, str)):
            serializable[k] = v
        elif isinstance(v, list):
            serializable[k] = [float(x) for x in v]
    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    env.close()
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HMARL on GRF 11v11")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of evaluation episodes")
    parser.add_argument("--output-dir", type=str, default="evaluation_results")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--baseline", choices=["random", "all"], default=None,
                        help="Also run baseline evaluation")

    args = parser.parse_args()

    # Load and evaluate HMARL
    policy, subgoal_emb = load_hmarl_model(args.checkpoint)
    hmarl_metrics = evaluate_hmarl(
        policy, subgoal_emb,
        num_episodes=args.episodes,
        render=args.render,
        output_dir=args.output_dir,
    )

    # Run baselines if requested
    random_metrics = None
    if args.baseline in ("random", "all"):
        random_metrics = evaluate_random_baseline(
            num_episodes=args.episodes,
            render=args.render,
            output_dir=args.output_dir,
        )

    # Print comparison summary (only when baseline was evaluated)
    if random_metrics:
        print(f"\n{'='*60}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Metric':<25} {'HMARL':>12} {'Random':>12}")
        print(f"  {'-'*49}")
        print(f"  {'Win Rate (%)':<25} {hmarl_metrics['win_rate']:>11.1f}% {random_metrics.get('win_rate', 0):>11.1f}%")
        print(f"  {'Goal Difference':<25} {hmarl_metrics['goal_difference']:>12} {random_metrics.get('goal_difference', 0):>12}")
        print(f"  {'RCI_strict':<25} {hmarl_metrics['rci_strict']:>12.4f} {random_metrics.get('rci_strict', 0):>12.4f}")
        print(f"  {'RCI_cat':<25} {hmarl_metrics['rci_cat']:>12.4f} {random_metrics.get('rci_cat', 0):>12.4f}")
        print(f"  {'PSR (%)':<25} {hmarl_metrics['psr']:>11.2f}% {random_metrics.get('psr', 0):>11.2f}%")
        print(f"  {'PPR (%)':<25} {hmarl_metrics['ppr']:>11.2f}% {random_metrics.get('ppr', 0):>11.2f}%")
        print(f"  {'Positional Entropy':<25} {hmarl_metrics['positional_entropy']:>12.4f} {random_metrics.get('positional_entropy', 0):>12.4f}")
        print(f"  {'Compactness':<25} {hmarl_metrics['compactness_mean']:>12.4f} {random_metrics.get('compactness_mean', 0):>12.4f}")
        print(f"  {'FAI':<25} {hmarl_metrics['fai_mean']:>12.4f} {random_metrics.get('fai_mean', 0):>12.4f}")
        print(f"  {'Def-Mid Gap':<25} {hmarl_metrics.get('defence_midfield_gap', 0):>12.4f} {random_metrics.get('defence_midfield_gap', 0):>12.4f}")
        print(f"  {'Mid-Att Gap':<25} {hmarl_metrics.get('midfield_attack_gap', 0):>12.4f} {random_metrics.get('midfield_attack_gap', 0):>12.4f}")
        print(f"  {'Convex Hull':<25} {hmarl_metrics.get('convex_hull_mean', 0):>12.4f} {random_metrics.get('convex_hull_mean', 0):>12.4f}")
        print(f"  {'BPR Mean':<25} {hmarl_metrics.get('bpr_mean', 0):>12.4f} {random_metrics.get('bpr_mean', 0):>12.4f}")
        print(f"{'='*60}")
