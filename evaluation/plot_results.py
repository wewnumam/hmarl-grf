"""Generate all thesis visualizations from saved evaluation data.

Usage:
    python evaluation/plot_results.py --results-dir evaluation_results --output-dir evaluation_results/plots
    python evaluation/plot_results.py --demo  # Generate demo plots with synthetic data
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from evaluation.visualizations import (
    plot_learning_curve,
    plot_rci_evolution,
    plot_comparative_bars,
    plot_role_heatmap,
    plot_formation_snapshot,
    plot_action_distribution,
    plot_macro_strategy_timeline,
    plot_compactness_over_time,
    plot_ablation_study,
    plot_tactic_transitions,
    plot_reward_breakdown,
    plot_metric_correlation,
)


def load_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def generate_plots_from_results(
    results_dir: str,
    output_dir: str,
    training_dir: str = None,
):
    """Generate plots from saved evaluation JSON files."""
    os.makedirs(output_dir, exist_ok=True)

    # --- Load evaluation results ---
    hmarl_results = load_json(os.path.join(results_dir, 'hmarl_results.json'))
    random_results = load_json(os.path.join(results_dir, 'random_results.json'))
    episodes_data = load_json(os.path.join(results_dir, 'hmarl_episodes.json'))

    # --- Load training logs (if available) ---
    if training_dir:
        training_log = load_json(os.path.join(training_dir, 'training_log.json'))
    else:
        # Auto-detect: check results_dir, dumps/, ../dumps/, ../../dumps/
        training_log = {}
        for candidate in [
            os.path.join(results_dir, 'training_log.json'),
            os.path.join(results_dir, '..', 'dumps', 'training_log.json'),
            os.path.join(results_dir, '..', 'training_log.json'),
        ]:
            training_log = load_json(candidate)
            if training_log:
                break

    models_data = {}
    if hmarl_results:
        models_data['HMARL'] = hmarl_results
    if random_results:
        models_data['Random'] = random_results

    # --- Load per-model comparison data (IPPO, SHPPO, MAPPO results if available) ---
    for name in ['ippo', 'shppo', 'mappo']:
        path = os.path.join(results_dir, f'{name}_results.json')
        data = load_json(path)
        if data:
            models_data[name.upper()] = data

    # --- 1. Learning Curve ---
    if training_log and 'episode_rewards' in training_log:
        rewards_data = {'HMARL': training_log['episode_rewards']}

        # Also load per-model training logs
        for name in ['ippo', 'shppo', 'mappo']:
            path = os.path.join(results_dir, f'{name}_training_log.json')
            data = load_json(path)
            if data and 'episode_rewards' in data:
                rewards_data[name.upper()] = data['episode_rewards']

        if rewards_data:
            plot_learning_curve(
                rewards_data,
                os.path.join(output_dir, '01_learning_curve.png'),
                window=50,
            )

    # --- 2. RCI Evolution ---
    if training_log and 'episode_rci_cat' in training_log:
        rci_data = {
            'HMARL': {
                'rci_cat': training_log.get('episode_rci_cat', []),
                'rci_strict': training_log.get('episode_rci_strict', []),
            }
        }
        for name in ['ippo', 'shppo', 'mappo']:
            path = os.path.join(results_dir, f'{name}_training_log.json')
            data = load_json(path)
            if data and 'episode_rci_cat' in data:
                rci_data[name.upper()] = {
                    'rci_cat': data.get('episode_rci_cat', []),
                    'rci_strict': data.get('episode_rci_strict', []),
                }

        plot_rci_evolution(
            rci_data,
            os.path.join(output_dir, '02_rci_evolution.png'),
            window=50,
        )

    # --- 3. Comparative Bar Chart ---
    if len(models_data) >= 2:
        plot_comparative_bars(
            models_data,
            os.path.join(output_dir, '03_comparative_metrics.png'),
        )

    # --- Plot compactness comparison ---
    compactness_data = {}
    for model_name, m_data in models_data.items():
        if 'compactness_ts' in m_data:
            compactness_data[model_name] = m_data['compactness_ts']
        elif 'compactness_mean' in m_data:
            # Use single value as flat line
            compactness_data[model_name] = [m_data['compactness_mean']] * 100

    if compactness_data:
        plot_compactness_over_time(
            compactness_data,
            os.path.join(output_dir, '08_compactness.png'),
        )

    # --- 9. Ablation Study ---
    ablation_data = load_json(os.path.join(results_dir, 'ablation_results.json'))
    if not ablation_data:
        # Try parent dir
        ablation_data = load_json(os.path.join(results_dir, '..', 'ablation_results.json'))
    if ablation_data:
        # Extract win_rate per config from nested summary
        summary = ablation_data.get('summary', ablation_data)
        ablation_flat = {}
        for name, metrics in summary.items():
            if isinstance(metrics, dict):
                ablation_flat[name] = metrics.get('win_rate', 0.0)
            else:
                ablation_flat[name] = metrics  # already scalar
        if ablation_flat:
            plot_ablation_study(
                ablation_flat,
                os.path.join(output_dir, '09_ablation.png'),
            )

    # --- 10. Tactic Transitions ---
    strategies = load_json(os.path.join(results_dir, 'strategies.json'))
    sub_goals = load_json(os.path.join(results_dir, 'sub_goals.json'))
    if strategies and sub_goals:
        plot_tactic_transitions(
            strategies,
            sub_goals,
            os.path.join(output_dir, '10_tactic_transitions.png'),
        )

    # --- 11. Reward Breakdown ---
    reward_components = load_json(os.path.join(results_dir, 'reward_components.json'))
    if reward_components and any(reward_components.values()):
        plot_reward_breakdown(
            reward_components,
            os.path.join(output_dir, '11_reward_breakdown.png'),
        )

    # --- 14. Inter-Agent Distance Per Line ---
    iad_data = {}
    for model_name, m_data in models_data.items():
        if all(k in m_data for k in ('defence_midfield_gap', 'midfield_attack_gap', 'overall_spread')):
            iad_data[model_name] = {
                'defence_midfield_gap': m_data['defence_midfield_gap'],
                'midfield_attack_gap': m_data['midfield_attack_gap'],
                'overall_spread': m_data['overall_spread'],
            }
    if iad_data:
        # Plot grouped bars for IAD
        from evaluation.visualizations import plot_iad_per_line as _plot_iad
        # Use first model's data (typically HMARL)
        first_model = list(iad_data.keys())[0]
        _plot_iad(
            iad_data[first_model],
            os.path.join(output_dir, '14_iad_per_line.png'),
            title=f'Inter-Agent Distance by Tactical Line ({first_model})',
        )

    # --- 15. Convex Hull Over Time ---
    convex_hull_ts = load_json(os.path.join(results_dir, 'convex_hull_ts.json'))
    if convex_hull_ts:
        plot_compactness_over_time(
            {'HMARL': convex_hull_ts},
            os.path.join(output_dir, '15_convex_hull_ts.png'),
            title='Convex Hull Area Over Time',
        )

    # --- 12. Correlation Heatmap ---
    if len(models_data) >= 2:
        # Core metrics for correlation — exclude auxiliary stats (std, min, max)
        CORE_CORR_METRICS = {
            'wr', 'goal_difference', 'goals_for', 'goals_against',
            'cumulative_reward', 'psr', 'ppr', 'positional_entropy',
            'compactness_mean', 'fai_mean',
            'rci_strict', 'rci_cat',
            'defence_midfield_gap', 'midfield_attack_gap', 'overall_spread',
            'convex_hull_mean', 'bpr_mean',
        }
        all_metric_data = {}
        for model_name, m_data in models_data.items():
            for k, v in m_data.items():
                if isinstance(v, (int, float)) and k in CORE_CORR_METRICS:
                    if k not in all_metric_data:
                        all_metric_data[k] = []
                    all_metric_data[k].append(v)

        if len(all_metric_data) >= 4:
            plot_metric_correlation(
                all_metric_data,
                os.path.join(output_dir, '12_correlation_heatmap.png'),
            )

    print(f'\nPlots generated in: {output_dir}')


def generate_demo_plots(output_dir: str):
    """Generate demo plots with synthetic data for visual verification."""
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(42)

    # --- 1. Learning Curve (demo) ---
    n_episodes = 500
    base_rewards = np.linspace(-80, 20, n_episodes)
    noise = np.random.normal(0, 15, n_episodes)

    hmarl_rewards = (base_rewards + noise + np.sin(np.linspace(0, 8, n_episodes)) * 5).tolist()
    ippo_rewards = (base_rewards * 0.6 - 20 + noise * 1.5).tolist()
    shppo_rewards = (base_rewards * 0.8 - 10 + noise * 1.2).tolist()
    mappo_rewards = (base_rewards * 0.7 - 15 + noise * 1.3).tolist()

    plot_learning_curve(
        {'HMARL': hmarl_rewards, 'IPPO': ippo_rewards, 'SHPPO': shppo_rewards, 'MAPPO': mappo_rewards},
        os.path.join(output_dir, '01_learning_curve.png'),
        window=30,
    )

    # --- 2. RCI Evolution (demo) ---
    rci_cat_hmarl = (np.linspace(0.05, 0.72, n_episodes) + np.random.normal(0, 0.03, n_episodes)).clip(0, 1).tolist()
    rci_strict_hmarl = (np.linspace(0.01, 0.35, n_episodes) + np.random.normal(0, 0.02, n_episodes)).clip(0, 1).tolist()
    rci_cat_ippo = (np.linspace(0.04, 0.45, n_episodes) + np.random.normal(0, 0.04, n_episodes)).clip(0, 1).tolist()
    rci_strict_ippo = (np.linspace(0.01, 0.20, n_episodes) + np.random.normal(0, 0.03, n_episodes)).clip(0, 1).tolist()
    rci_cat_mappo = (np.linspace(0.04, 0.52, n_episodes) + np.random.normal(0, 0.035, n_episodes)).clip(0, 1).tolist()
    rci_strict_mappo = (np.linspace(0.01, 0.25, n_episodes) + np.random.normal(0, 0.025, n_episodes)).clip(0, 1).tolist()

    plot_rci_evolution(
        {
            'HMARL': {'rci_cat': rci_cat_hmarl, 'rci_strict': rci_strict_hmarl},
            'IPPO': {'rci_cat': rci_cat_ippo, 'rci_strict': rci_strict_ippo},
            'MAPPO': {'rci_cat': rci_cat_mappo, 'rci_strict': rci_strict_mappo},
        },
        os.path.join(output_dir, '02_rci_evolution.png'),
        window=30,
    )

    # --- 3. Comparative Bar Chart (demo) ---
    plot_comparative_bars(
        {
            'HMARL': {
                'wr': 72.0, 'gd': 28, 'cumulative_reward': 1250.0,
                'psr': 68.5, 'ppr': 55.2, 'positional_entropy': 1.82,
                'compactness_mean': 18.3, 'fai_mean': 0.87,
                'rci_strict': 0.34, 'rci_cat': 0.71,
            },
            'IPPO': {
                'wr': 45.0, 'gd': 8, 'cumulative_reward': 420.0,
                'psr': 42.1, 'ppr': 32.8, 'positional_entropy': 2.45,
                'compactness_mean': 24.1, 'fai_mean': 0.72,
                'rci_strict': 0.18, 'rci_cat': 0.45,
            },
            'SHPPO': {
                'wr': 58.0, 'gd': 16, 'cumulative_reward': 780.0,
                'psr': 55.3, 'ppr': 43.1, 'positional_entropy': 2.10,
                'compactness_mean': 21.0, 'fai_mean': 0.79,
                'rci_strict': 0.25, 'rci_cat': 0.58,
            },
            'MAPPO': {
                'wr': 52.0, 'gd': 12, 'cumulative_reward': 620.0,
                'psr': 48.7, 'ppr': 38.5, 'positional_entropy': 2.25,
                'compactness_mean': 22.5, 'fai_mean': 0.75,
                'rci_strict': 0.22, 'rci_cat': 0.52,
            },
        },
        os.path.join(output_dir, '03_comparative_metrics.png'),
    )

    # --- 4. Role Heatmap (demo with synthetic positions) ---
    n_frames = 500
    # Simulate 11 players in 4-3-3
    base_positions = [
        [-0.95, 0.0],    # GK
        [-0.5, -0.15],   # CB
        [-0.5, 0.15],    # CB
        [-0.3, -0.35],   # LB
        [-0.3, 0.35],    # RB
        [-0.1, 0.0],     # DM
        [0.1, -0.2],     # CM
        [0.1, 0.2],      # CM
        [0.3, -0.35],    # LM
        [0.3, 0.35],     # RM
        [0.5, 0.0],      # CF
    ]

    demo_game_states = []
    for t in range(n_frames):
        ball_x = np.random.uniform(-1, 1)
        positions = []
        for bp in base_positions:
            x = bp[0] + np.random.normal(0, 0.08) + ball_x * 0.15
            y = bp[1] + np.random.normal(0, 0.06)
            positions.append([np.clip(x, -1, 1), np.clip(y, -0.42, 0.42)])
        demo_game_states.append({
            'left_team': positions,
            'left_team_roles': [0, 1, 1, 2, 3, 4, 5, 5, 6, 7, 9],
            'ball': [ball_x, np.random.uniform(-0.42, 0.42), 0.1],
        })

    plot_role_heatmap(
        demo_game_states,
        os.path.join(output_dir, '04_role_heatmap.png'),
    )

    # --- 5. Formation Snapshot (demo) ---
    plot_formation_snapshot(
        demo_game_states,
        os.path.join(output_dir, '05_formation_mid.png'),
        timestep=n_frames // 2,
        title='Formation Snapshot (mid-match)',
    )

    # --- 6. Action Distribution (demo) ---
    n_steps = 3000
    demo_actions = []
    for _ in range(n_steps):
        step_actions = []
        for i in range(11):
            # Weighted by role
            if i == 0:  # GK
                step_actions.append(np.random.choice([0, 11, 16], p=[0.5, 0.3, 0.2]))
            elif i in (1, 2, 3, 4):  # Defense
                step_actions.append(np.random.choice([0, 3, 5, 11, 16, 17], p=[0.15, 0.2, 0.15, 0.2, 0.1, 0.2]))
            elif i in (5, 6, 7):  # Midfield
                step_actions.append(np.random.choice([0, 3, 5, 11, 12, 13, 17], p=[0.1, 0.15, 0.15, 0.25, 0.05, 0.15, 0.15]))
            else:  # Attack
                step_actions.append(np.random.choice([0, 3, 5, 11, 12, 13, 17], p=[0.05, 0.15, 0.1, 0.2, 0.15, 0.2, 0.15]))
        demo_actions.append(step_actions)

    plot_action_distribution(
        demo_actions,
        [0, 1, 1, 2, 3, 4, 5, 5, 6, 7, 9],
        os.path.join(output_dir, '06_action_distribution.png'),
    )

    # --- 7. Macro Strategy Timeline (demo) ---
    demo_strategies = []
    for t in range(3000):
        if t < 800:
            demo_strategies.append(0)  # High pressing
        elif t < 1500:
            demo_strategies.append(2)  # Possession
        elif t < 2200:
            demo_strategies.append(1)  # Counter
        else:
            demo_strategies.append(0)  # High pressing

    plot_macro_strategy_timeline(
        demo_strategies,
        os.path.join(output_dir, '07_strategy_timeline.png'),
    )

    # --- 8. Compactness Over Time (demo) ---
    tc_hmarl = (20 + np.random.normal(0, 2, 3000) + np.sin(np.linspace(0, 20, 3000)) * 3).tolist()
    tc_ippo = (25 + np.random.normal(0, 3, 3000) + np.sin(np.linspace(0, 15, 3000)) * 5).tolist()

    plot_compactness_over_time(
        {'HMARL': tc_hmarl, 'IPPO': tc_ippo},
        os.path.join(output_dir, '08_compactness.png'),
    )

    # --- 9. Ablation Study (demo) ---
    plot_ablation_study(
        {
            'Full HMARL': 0.712,
            'No High-Level (rule)': 0.584,
            'No Mid-Level (rule)': 0.643,
            'No RCI reward ($\\alpha_L=0$)': 0.621,
            'IPPO (no hierarchy)': 0.448,
        },
        os.path.join(output_dir, '09_ablation.png'),
    )

    # --- 10. Tactic Transitions (demo) ---
    demo_subgoals = []
    for t in range(3000):
        step_sgs = []
        for i in range(11):
            if demo_strategies[t] == 0:  # High pressing
                step_sgs.append(np.random.choice([0, 2, 3], p=[0.4, 0.35, 0.25]))
            elif demo_strategies[t] == 1:  # Counter
                step_sgs.append(np.random.choice([2, 4], p=[0.6, 0.4]))
            else:  # Possession
                step_sgs.append(np.random.choice([1, 0], p=[0.7, 0.3]))
        demo_subgoals.append(step_sgs)

    plot_tactic_transitions(
        demo_strategies,
        demo_subgoals,
        os.path.join(output_dir, '10_tactic_transitions.png'),
    )

    # --- 11. Reward Breakdown (demo) ---
    n_eps = 300
    game_r = (np.random.choice([-1, 0, 1], size=n_eps, p=[0.3, 0.5, 0.2]) * 1.0).tolist()
    r_high = (np.ones(n_eps) * 0.01 * np.linspace(0.6, 0.9, n_eps) + np.random.normal(0, 0.001, n_eps)).tolist()
    r_mid = (np.ones(n_eps) * 0.01 * np.linspace(0.3, 0.6, n_eps) + np.random.normal(0, 0.002, n_eps)).tolist()
    r_low = (np.ones(n_eps) * 0.01 * np.linspace(0.1, 0.7, n_eps) + np.random.normal(0, 0.001, n_eps)).tolist()

    plot_reward_breakdown(
        {'game_reward': game_r, 'r_high': r_high, 'r_mid': r_mid, 'r_low': r_low},
        os.path.join(output_dir, '11_reward_breakdown.png'),
    )

    # --- 12. Correlation Heatmap (demo) ---
    n_pts = 20
    plot_metric_correlation(
        {
            'rci_cat': np.random.uniform(0.4, 0.8, n_pts).tolist(),
            'rci_strict': np.random.uniform(0.15, 0.45, n_pts).tolist(),
            'fai_mean': np.random.uniform(0.7, 0.95, n_pts).tolist(),
            'positional_entropy': np.random.uniform(1.5, 2.5, n_pts).tolist(),
            'compactness_mean': np.random.uniform(15, 25, n_pts).tolist(),
            'psr': np.random.uniform(40, 75, n_pts).tolist(),
            'ppr': np.random.uniform(30, 60, n_pts).tolist(),
            'wr': np.random.uniform(40, 80, n_pts).tolist(),
        },
        os.path.join(output_dir, '12_correlation_heatmap.png'),
    )

    print(f'\nAll demo plots saved to: {output_dir}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate thesis visualizations')
    parser.add_argument('--results-dir', type=str, default='evaluation_results',
                       help='Directory with evaluation JSON results')
    parser.add_argument('--output-dir', type=str, default='evaluation_results/plots',
                       help='Directory to save plots')
    parser.add_argument('--training-dir', type=str, default=None,
                       help='Directory with training logs')
    parser.add_argument('--demo', action='store_true',
                       help='Generate demo plots with synthetic data')

    args = parser.parse_args()

    if args.demo:
        generate_demo_plots(args.output_dir)
    else:
        generate_plots_from_results(
            args.results_dir,
            args.output_dir,
            args.training_dir,
        )
