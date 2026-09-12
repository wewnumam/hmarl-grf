"""RCI Validity Evaluation Script.

Evaluates Role Coherence Index (RCI) validity through three approaches:
1. Construct Validity  — Pearson correlation between RCI and existing metrics
2. Discrimination Validity — One-sided t-test: HMARL > IPPO > Random
3. Internal Consistency — Coefficient of Variation (CV) across seeds

Also includes:
4. Sensitivity Analysis — RCI stability under expert policy threshold variation

Usage (inside Docker):
    python scripts/validate_rci.py --seeds 3 --timesteps 3000000 --eval-episodes 100
    python scripts/validate_rci.py --seeds 3 --timesteps 100000 --eval-episodes 10 --quick
    python scripts/validate_rci.py --load-results evaluation_results/stat_test_results.json
"""

import argparse
import json
import os
import sys
import traceback
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not installed. Statistical tests require scipy.")
    print("  Install with: pip install scipy")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RCI_ALPHA = 0.05  # significance level


# ===========================================================================
# 1. CONSTRUCT VALIDITY — Pearson Correlation
# ===========================================================================

def construct_validity(
    per_episode_metrics: Dict[str, List[float]],
) -> Dict:
    """Compute Pearson correlation between RCI and other coordination metrics.

    Expected correlations (per thesis):
        RCI_cat  ↔  FAI            →  positive
        RCI_cat  ↔  Entropy (H)    →  negative
        RCI_cat  ↔  Compactness    →  negative (tighter formation = more coherent)
        RCI_strict ↔ FAI           →  positive

    Args:
        per_episode_metrics: dict of metric_name -> list of per-episode values.
            All lists must have the same length (same number of episodes).

    Returns:
        dict with per-pair {r, p_value, significant, direction_expected, matched}
    """
    if not HAS_SCIPY:
        return {'error': 'scipy not installed', 'pairs': {}}

    # Pairs to test with expected direction
    pairs_spec = [
        ('rci_cat', 'fai_mean', 'positive'),
        ('rci_strict', 'fai_mean', 'positive'),
        ('rci_cat', 'positional_entropy', 'negative'),
        ('rci_strict', 'positional_entropy', 'negative'),
        ('rci_cat', 'compactness_mean', 'negative'),
        ('rci_strict', 'compactness_mean', 'negative'),
        ('rci_cat', 'psr', 'positive'),
        ('rci_cat', 'ppr', 'positive'),
        ('rci_cat', 'win_rate', 'positive'),
    ]

    results = {}
    for metric_a, metric_b, expected_dir in pairs_spec:
        if metric_a not in per_episode_metrics or metric_b not in per_episode_metrics:
            continue
        arr_a = np.array([v for v in per_episode_metrics[metric_a] if isinstance(v, (int, float)) and not np.isnan(v)])
        arr_b = np.array([v for v in per_episode_metrics[metric_b] if isinstance(v, (int, float)) and not np.isnan(v)])
        n = min(len(arr_a), len(arr_b))
        if n < 3:
            results[f'{metric_a}↔{metric_b}'] = {
                'status': 'skipped',
                'reason': f'insufficient data points (n={n}, need ≥3)',
            }
            continue

        r, p = sp_stats.pearsonr(arr_a[:n], arr_b[:n])
        significant = p < RCI_ALPHA
        if expected_dir == 'positive':
            matched = significant and r > 0
        else:
            matched = significant and r < 0

        results[f'{metric_a}↔{metric_b}'] = {
            'pearson_r': round(float(r), 4),
            'p_value': round(float(p), 6),
            'significant': significant,
            'expected_direction': expected_dir,
            'matched': matched,
            'n': n,
            'interpretation': (
                'PASS' if matched
                else f'FAIL — expected {expected_dir} significant correlation'
            ),
        }

    return results


# ===========================================================================
# 2. DISCRIMINATION VALIDITY — One-Sided t-test
# ===========================================================================

def discrimination_validity(
    hmarl_rci_values: List[float],
    baseline_rci_values: List[float],
    baseline_name: str,
    alternative: str = 'greater',
) -> Dict:
    """One-sided t-test: HMARL RCI > baseline RCI.

    H0: μ_hmarl ≤ μ_baseline
    H1: μ_hmarl > μ_baseline

    Args:
        hmarl_rci_values: list of RCI_cat values per seed (HMARL)
        baseline_rci_values: list of RCI_cat values per seed
        baseline_name: name of baseline
        alternative: 'greater' for HMARL > baseline

    Returns:
        dict with test results
    """
    if not HAS_SCIPY:
        return {'error': 'scipy not installed', 'baseline': baseline_name}

    result = {
        'baseline': baseline_name,
        'test': 'One-sided t-test (HMARL > baseline)',
        'alpha': RCI_ALPHA,
        'hmarl_mean': round(float(np.mean(hmarl_rci_values)), 4),
        'hmarl_std': round(float(np.std(hmarl_rci_values, ddof=1)), 4)
                     if len(hmarl_rci_values) > 1 else 0.0,
        'baseline_mean': round(float(np.mean(baseline_rci_values)), 4),
        'baseline_std': round(float(np.std(baseline_rci_values, ddof=1)), 4)
                        if len(baseline_rci_values) > 1 else 0.0,
        'n_hmarl': len(hmarl_rci_values),
        'n_baseline': len(baseline_rci_values),
    }

    if len(hmarl_rci_values) < 2 or len(baseline_rci_values) < 2:
        result['status'] = 'skipped'
        result['reason'] = 'need ≥2 values per group'
        return result

    t_stat, p_two = sp_stats.ttest_ind(
        hmarl_rci_values, baseline_rci_values, equal_var=False,
    )
    # One-sided: if t_stat > 0 (HMARL higher), p_one = p_two / 2
    # If t_stat < 0, p_one = 1 - p_two / 2 (opposite direction)
    if t_stat > 0:
        p_one = p_two / 2
    else:
        p_one = 1.0 - p_two / 2

    result['t_statistic'] = round(float(t_stat), 4)
    result['p_value_one_sided'] = round(float(p_one), 6)
    result['significant'] = p_one < RCI_ALPHA
    result['interpretation'] = (
        'PASS — HMARL RCI significantly higher'
        if result['significant']
        else 'FAIL — no significant difference'
    )
    return result


# ===========================================================================
# 3. INTERNAL CONSISTENCY — CV Across Seeds
# ===========================================================================

def internal_consistency(
    seed_values: Dict[str, List[float]],
    cv_threshold: float = 15.0,
) -> Dict:
    """Compute Coefficient of Variation across seeds for each model.

    CV = (std / mean) * 100
    CV < 15% indicates acceptable internal consistency.

    Args:
        seed_values: dict of model_name -> list of RCI values per seed
        cv_threshold: acceptable CV percentage (default 15%)

    Returns:
        dict per model with CV, pass/fail, and per-variant results
    """
    results = {}
    for model_name, values in seed_values.items():
        if len(values) < 2:
            results[model_name] = {
                'status': 'skipped',
                'reason': f'need ≥2 seeds, got {len(values)}',
            }
            continue

        arr = np.array(values)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1))
        cv = (std_val / mean_val * 100) if mean_val != 0 else float('inf')

        results[model_name] = {
            'mean': round(mean_val, 4),
            'std': round(std_val, 4),
            'cv_percent': round(cv, 2),
            'cv_threshold': cv_threshold,
            'n_seeds': len(values),
            'consistent': cv < cv_threshold,
            'interpretation': (
                f'PASS — CV={cv:.1f}% < {cv_threshold}%'
                if cv < cv_threshold
                else f'FAIL — CV={cv:.1f}% ≥ {cv_threshold}%'
            ),
        }

    return results


# ===========================================================================
# 4. SENSITIVITY ANALYSIS — Expert Policy Threshold Variation
# ===========================================================================

def sensitivity_analysis(
    base_rci_cat: float,
    base_rci_strict: float,
    n_episodes: int = 5,
    seed: int = 42,
    base_thresholds: Dict[str, float] = None,
    variations: List[Dict[str, float]] = None,
) -> Dict:
    """Sensitivity of RCI to expert policy threshold variations.

    Varies each threshold by ±20% from default and measures RCI change.
    For each variation, runs n_episodes and computes RCI.

    This requires GRF environment — only runs inside Docker.

    Args:
        base_rci_cat: baseline RCI_cat (for reporting reference)
        base_rci_strict: baseline RCI_strict (for reporting reference)
        n_episodes: episodes per variation
        seed: random seed
        base_thresholds: default {d_tackle, d_safe, d_shoot}
        variations: list of threshold dicts to test

    Returns:
        dict with per-variation RCI deltas
    """
    if base_thresholds is None:
        base_thresholds = {'d_tackle': 0.05, 'd_safe': 0.15, 'd_shoot': 0.30}

    if variations is None:
        # ±20% of each threshold, one at a time
        variations = []
        for key, val in base_thresholds.items():
            for factor in [0.8, 1.2]:
                var = dict(base_thresholds)
                var[key] = round(val * factor, 4)
                var['_label'] = f'{key}={var[key]:.4f} ({factor:.0%})'
                variations.append(var)

    # Attempt to run with GRF (may fail outside Docker)
    try:
        import torch
        from hmarl.env import create_raw_env, extract_game_state, NUM_AGENTS, ACTION_SPACE_SIZE
        from hmarl.expert import ExpertPolicyAllAgents, ExpertPolicy, D_TACKLE, D_SAFE, D_SHOOT
        from hmarl.policy import HierarchicalActorCritic, HierarchicalController, SubGoalEmbedding, SUBGOAL_EMBED_DIM
        from hmarl.rci import compute_rci
        from hmarl.utils import set_seed, extract_obs_vector, HIDDEN_DIM, HEAD_DIM, OBS_DIM, EPISODE_MAX_STEPS

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load trained HMARL model
        ckpt_path = "dumps/hmarl_model.pt"
        if not os.path.exists(ckpt_path):
            return {
                'status': 'skipped',
                'reason': f'model checkpoint not found: {ckpt_path}',
                'note': 'Train model first or provide --model-path',
            }

        policy = HierarchicalActorCritic(
            obs_dim=OBS_DIM, subgoal_embed_dim=SUBGOAL_EMBED_DIM,
            hidden_dim=HIDDEN_DIM, head_dim=HEAD_DIM, action_dim=ACTION_SPACE_SIZE,
        ).to(DEVICE)
        subgoal_emb = SubGoalEmbedding().to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        policy.load_state_dict(ckpt['policy_state'])
        subgoal_emb.load_state_dict(ckpt['subgoal_embedding_state'])
        policy.eval()

        controller = HierarchicalController()
        results = {'base_thresholds': base_thresholds, 'base_rci_cat': base_rci_cat}

        for var in variations:
            label = var.get('_label', str(var))
            expert = ExpertPolicy(
                d_tackle=var['d_tackle'],
                d_safe=var['d_safe'],
                d_shoot=var['d_shoot'],
            )
            all_actual = []
            all_ideal = []

            set_seed(seed)
            env = create_raw_env(render=False)

            for ep in range(n_episodes):
                reset_result = env.reset()
                obs_raw = reset_result[0] if isinstance(reset_result, tuple) else reset_result
                game_state = extract_game_state(obs_raw)
                done = False
                step = 0

                while not done and step < EPISODE_MAX_STEPS:
                    macro = controller.get_macro_strategy(game_state)
                    sub_goals = controller.get_sub_goals(game_state, macro)
                    ideal_actions = [
                        expert.get_ideal_action(game_state, i, sub_goals[i], macro)
                        for i in range(NUM_AGENTS)
                    ]
                    joint_actions = []
                    for i in range(NUM_AGENTS):
                        obs_vec = extract_obs_vector(game_state, i)
                        with torch.no_grad():
                            sg_emb = subgoal_emb(
                                torch.LongTensor([sub_goals[i]]).to(DEVICE)
                            ).cpu().numpy().flatten()
                            obs_t = torch.FloatTensor(obs_vec).unsqueeze(0).to(DEVICE)
                            sg_t = torch.FloatTensor(sg_emb).unsqueeze(0).to(DEVICE)
                            logits, _ = policy(obs_t, sg_t)
                            joint_actions.append(logits.argmax(dim=-1).item())

                    step_result = env.step(joint_actions)
                    if len(step_result) == 5:
                        obs_raw, _, terminated, truncated, _ = step_result
                        done = terminated or truncated
                    else:
                        obs_raw, _, done, _ = step_result

                    all_actual.append(joint_actions)
                    all_ideal.append(ideal_actions)
                    game_state = extract_game_state(obs_raw)
                    step += 1

            env.close()

            rci_res = compute_rci(all_actual, all_ideal)
            results[label] = {
                'rci_cat': round(rci_res['rci_cat'], 4),
                'rci_strict': round(rci_res['rci_strict'], 4),
                'delta_rci_cat': round(rci_res['rci_cat'] - base_rci_cat, 4),
                'delta_rci_strict': round(rci_res['rci_strict'] - base_rci_strict, 4),
                'thresholds': {k: v for k, v in var.items() if not k.startswith('_')},
            }

        return results

    except ImportError as e:
        return {
            'status': 'skipped',
            'reason': f'Missing dependency: {e}',
            'note': 'Sensitivity analysis requires GRF environment (run inside Docker)',
        }


# ===========================================================================
# ORCHESTRATION
# ===========================================================================

def run_full_validation(
    per_seed_metrics: Dict[str, Dict[str, List[float]]],
    model_order: List[str] = None,
) -> Dict:
    """Run all three validity checks on collected per-seed data.

    Args:
        per_seed_metrics: {
            model_name: {
                'rci_cat': [seed1, seed2, ...],
                'rci_strict': [seed1, seed2, ...],
                'fai_mean': [seed1, seed2, ...],
                ...per-episode metrics aggregated per seed...
            }
        }
        model_order: ordered list for discrimination tests

    Returns:
        Full validation report dict
    """
    if model_order is None:
        model_order = ['hmarl', 'ippo', 'shppo', 'mappo', 'random']

    report = {}

    # --- 1. Construct Validity ---
    # Aggregate all episodes across seeds into per-episode lists
    print("\n" + "=" * 60)
    print("  1. CONSTRUCT VALIDITY (Pearson Correlation)")
    print("=" * 60)

    # We need per-episode data for correlation, not per-seed aggregates.
    # The per_seed_metrics has aggregated values per seed.
    # For proper correlation, we use seed-level data points (each seed = one observation).
    agg_metrics = {}
    available_models = [m for m in model_order if m in per_seed_metrics]
    for model in available_models:
        for metric, values in per_seed_metrics[model].items():
            if isinstance(values, list) and len(values) > 0:
                if metric not in agg_metrics:
                    agg_metrics[metric] = []
                agg_metrics[metric].extend(values)

    report['construct_validity'] = construct_validity(agg_metrics)

    print("\n  Pearson correlation between RCI and existing metrics:")
    print(f"  {'Pair':<35} {'r':>8} {'p':>10} {'Sig':>5} {'Dir':>6} {'Result'}")
    print(f"  {'-'*95}")
    for pair, res in report['construct_validity'].items():
        if isinstance(res, dict) and 'pearson_r' in res:
            print(
                f"  {pair:<35} {res['pearson_r']:>8.4f} {res['p_value']:>10.4f} "
                f"{'Y' if res['significant'] else 'N':>5} "
                f"{res['expected_direction']:>6}  {res['interpretation']}"
            )
        else:
            print(f"  {pair:<35} {res.get('reason', res.get('status', 'N/A'))}")

    # --- 2. Discrimination Validity ---
    print("\n" + "=" * 60)
    print("  2. DISCRIMINATION VALIDITY (One-Sided t-test)")
    print("=" * 60)

    report['discrimination_validity'] = {}
    if 'hmarl' in per_seed_metrics and 'rci_cat' in per_seed_metrics.get('hmarl', {}):
        hmarl_rci = per_seed_metrics['hmarl']['rci_cat']
        for model in available_models:
            if model == 'hmarl':
                continue
            baseline_rci = per_seed_metrics.get(model, {}).get('rci_cat', [])
            if baseline_rci:
                test_result = discrimination_validity(hmarl_rci, baseline_rci, model)
                report['discrimination_validity'][f'hmarl_vs_{model}'] = test_result
                print(f"\n  HMARL vs {model.upper()}:")
                print(f"    HMARL:  {test_result['hmarl_mean']:.4f} ± {test_result['hmarl_std']:.4f} (n={test_result['n_hmarl']})")
                print(f"    {model:8s}: {test_result['baseline_mean']:.4f} ± {test_result['baseline_std']:.4f} (n={test_result['n_baseline']})")
                if 't_statistic' in test_result:
                    print(f"    t={test_result['t_statistic']:.4f}, p={test_result['p_value_one_sided']:.6f}")
                    print(f"    → {test_result['interpretation']}")
                else:
                    print(f"    Skipped: {test_result.get('reason', test_result.get('status', 'unknown'))}")

    # --- 3. Internal Consistency ---
    print("\n" + "=" * 60)
    print("  3. INTERNAL CONSISTENCY (CV across seeds)")
    print("=" * 60)

    consistency_data = {}
    for model in available_models:
        if 'rci_cat' in per_seed_metrics.get(model, {}):
            consistency_data[model] = per_seed_metrics[model]['rci_cat']

    report['internal_consistency'] = internal_consistency(consistency_data)

    print(f"\n  {'Model':<12} {'Mean':>8} {'Std':>8} {'CV%':>8} {'Threshold':>10} {'Result'}")
    print(f"  {'-'*65}")
    for model, res in report['internal_consistency'].items():
        if isinstance(res, dict) and 'cv_percent' in res:
            print(
                f"  {model:<12} {res['mean']:>8.4f} {res['std']:>8.4f} "
                f"{res['cv_percent']:>7.1f}% {res['cv_threshold']:>9.1f}%  "
                f"{res['interpretation']}"
            )
        else:
            print(f"  {model:<12} {res.get('reason', res.get('status', 'N/A'))}")

    return report


def run_sensitivity_from_checkpoint(model_path: str, output_path: str, n_episodes: int, seed: int):
    """Run sensitivity analysis as standalone step."""
    print("\n" + "=" * 60)
    print("  4. SENSITIVITY ANALYSIS (Expert Policy Threshold)")
    print("=" * 60)

    # First compute base RCI
    try:
        import torch
        from hmarl.env import create_raw_env, extract_game_state, NUM_AGENTS, ACTION_SPACE_SIZE
        from hmarl.expert import ExpertPolicyAllAgents
        from hmarl.policy import HierarchicalActorCritic, HierarchicalController, SubGoalEmbedding, SUBGOAL_EMBED_DIM
        from hmarl.rci import compute_rci
        from hmarl.utils import set_seed, extract_obs_vector

        HIDDEN_DIM, HEAD_DIM, OBS_DIM = 256, 128, 115
        EPISODE_MAX_STEPS = 3000
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        set_seed(seed)
        policy = HierarchicalActorCritic(
            obs_dim=OBS_DIM, subgoal_embed_dim=SUBGOAL_EMBED_DIM,
            hidden_dim=HIDDEN_DIM, head_dim=HEAD_DIM, action_dim=ACTION_SPACE_SIZE,
        ).to(DEVICE)
        subgoal_emb = SubGoalEmbedding().to(DEVICE)
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        policy.load_state_dict(ckpt['policy_state'])
        subgoal_emb.load_state_dict(ckpt['subgoal_embedding_state'])
        policy.eval()

        env = create_raw_env(render=False)
        controller = HierarchicalController()
        expert_default = ExpertPolicyAllAgents()

        base_actual, base_ideal = [], []
        for ep in range(n_episodes):
            reset_result = env.reset()
            obs_raw = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            game_state = extract_game_state(obs_raw)
            done, step = False, 0
            while not done and step < EPISODE_MAX_STEPS:
                macro = controller.get_macro_strategy(game_state)
                sub_goals = controller.get_sub_goals(game_state, macro)
                ideal = expert_default.get_ideal_actions(game_state, sub_goals, macro)
                joint = []
                for i in range(NUM_AGENTS):
                    obs_vec = extract_obs_vector(game_state, i)
                    with torch.no_grad():
                        sg = subgoal_emb(torch.LongTensor([sub_goals[i]]).to(DEVICE)).cpu().numpy().flatten()
                        obs_t = torch.FloatTensor(obs_vec).unsqueeze(0).to(DEVICE)
                        sg_t = torch.FloatTensor(sg).unsqueeze(0).to(DEVICE)
                        logits, _ = policy(obs_t, sg_t)
                        joint.append(logits.argmax(dim=-1).item())
                step_result = env.step(joint)
                if len(step_result) == 5:
                    obs_raw, _, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    obs_raw, _, done, _ = step_result
                base_actual.append(joint)
                base_ideal.append(ideal)
                game_state = extract_game_state(obs_raw)
                step += 1
        env.close()

        base_rci = compute_rci(base_actual, base_ideal)
        print(f"\n  Base RCI_cat={base_rci['rci_cat']:.4f}, RCI_strict={base_rci['rci_strict']:.4f}")

        results = sensitivity_analysis(
            base_rci_cat=base_rci['rci_cat'],
            base_rci_strict=base_rci['rci_strict'],
            n_episodes=n_episodes,
            seed=seed,
        )

    except ImportError as e:
        print(f"  SKIPPED: {e}")
        results = {'status': 'skipped', 'reason': str(e)}

    print(f"\n  {'Variation':<40} {'RCI_cat':>8} {'Δ':>8} {'RCI_str':>8} {'Δ':>8}")
    print(f"  {'-'*80}")
    for label, res in results.items():
        if isinstance(res, dict) and 'rci_cat' in res:
            print(
                f"  {label:<40} {res['rci_cat']:>8.4f} {res['delta_rci_cat']:>+8.4f} "
                f"{res['rci_strict']:>8.4f} {res['delta_rci_strict']:>+8.4f}"
            )
        elif isinstance(res, dict) and 'status' in res:
            print(f"  {label:<40} {res.get('reason', res['status'])}")

    return results


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="RCI Validity Evaluation")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of random seeds per model")
    parser.add_argument("--timesteps", type=int, default=3_000_000,
                        help="Training timesteps per seed")
    parser.add_argument("--eval-episodes", type=int, default=100,
                        help="Evaluation episodes per seed")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--algorithms", nargs="+", default=["hmarl", "random"],
                        help="Algorithms to evaluate")
    parser.add_argument("--output", type=str,
                        default="evaluation_results/rci_validity.json")
    parser.add_argument("--load-results", type=str, default=None,
                        help="Load pre-computed stat_test results instead of running")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Also run sensitivity analysis")
    parser.add_argument("--model-path", type=str, default="dumps/hmarl_model.pt",
                        help="Path to trained HMARL checkpoint for sensitivity")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer steps/episodes")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.quick:
        args.timesteps = 100_000
        args.eval_episodes = 10
        args.seeds = min(args.seeds, 3)

    if args.load_results and os.path.exists(args.load_results):
        # Load existing results and convert to validation format
        print(f"Loading results from {args.load_results}")
        with open(args.load_results) as f:
            data = json.load(f)
        # Convert format — the stat_test output has per-seed metrics
        # but validation needs per-episode metric lists
        # We'll use the seed-level aggregates as approximate per-seed values
        print("NOTE: Loaded results use seed-level aggregates.")
        print("      For full per-episode correlation, run without --load-results.")
        per_seed = {}
        for alg, res in data.get('results', {}).items():
            per_seed[alg] = {}
            for key in ['rci', 'win_rates', 'rewards']:
                if key in res:
                    per_seed[alg][key] = res[key]
            # Map common keys
            if 'rci' in res:
                per_seed[alg]['rci_cat'] = res['rci']
            if 'win_rates' in res:
                per_seed[alg]['win_rate'] = res['win_rates']
    else:
        # Run evaluation and collect per-seed metrics
        print("Running evaluation to collect per-seed metrics...")
        print(f"  Seeds: {args.seeds} | Timesteps: {args.timesteps:,}")
        print(f"  Episodes: {args.eval_episodes} | Algorithms: {args.algorithms}")

        # Import evaluation functions
        from stat_test import (
            train_hmarl, train_flat, evaluate_hmarl_from_checkpoint,
            evaluate_flat, evaluate_random, set_seed,
        )

        per_seed = {}
        for alg in args.algorithms:
            print(f"\n  Evaluating: {alg.upper()}")
            seed_metrics = {
                'rci_cat': [], 'rci_strict': [],
                'fai_mean': [], 'positional_entropy': [],
                'compactness_mean': [], 'psr': [], 'ppr': [],
                'win_rate': [],
            }
            for s in range(args.seeds):
                seed = args.base_seed + s
                try:
                    if alg == 'hmarl':
                        ckpt = train_hmarl(args.timesteps, seed,
                                           f"valid_temp/{alg}_{s}")
                        metrics = evaluate_hmarl_from_checkpoint(
                            ckpt, args.eval_episodes, seed + 1000,
                        )
                    elif alg == 'random':
                        metrics = evaluate_random(args.eval_episodes, seed)
                    else:
                        train_flat(alg, args.timesteps, seed,
                                   f"valid_temp/{alg}_{s}")
                        metrics = evaluate_flat(alg, args.eval_episodes,
                                               seed + 1000)

                    for key in seed_metrics:
                        if key in metrics:
                            seed_metrics[key].append(metrics[key])
                        elif key == 'win_rate':
                            seed_metrics[key].append(metrics.get('win_rate', 0))
                        else:
                            seed_metrics[key].append(0.0)

                    rci_val = metrics.get('rci_cat')
                    rci_str = f"{rci_val:.4f}" if isinstance(rci_val, (int, float)) and not np.isnan(rci_val) else "N/A"
                    wr_val = metrics.get('win_rate', 0)
                    wr_str = f"{wr_val:.1f}%" if isinstance(wr_val, (int, float)) else "N/A"
                    print(f"    Seed {s+1}: RCI_cat={rci_str}, WR={wr_str}")

                except Exception as e:
                    traceback.print_exc()
                    print(f"    Seed {s+1} FAILED: {e}")

            per_seed[alg] = seed_metrics

    # Run validation
    report = run_full_validation(per_seed)

    # Run sensitivity analysis if requested
    if args.sensitivity:
        report['sensitivity_analysis'] = run_sensitivity_from_checkpoint(
            args.model_path, args.output, args.eval_episodes, args.base_seed,
        )

    # Save
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n{'='*60}")
    print(f"  Full validation report saved to: {args.output}")
    print(f"{'='*60}")

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    cv_results = report.get('internal_consistency', {})
    for model, res in cv_results.items():
        if isinstance(res, dict) and 'consistent' in res:
            status = "✅" if res['consistent'] else "❌"
            print(f"  {status} {model}: CV={res['cv_percent']:.1f}%")

    disc_results = report.get('discrimination_validity', {})
    for pair, res in disc_results.items():
        if isinstance(res, dict) and 'significant' in res:
            status = "✅" if res['significant'] else "❌"
            print(f"  {status} {pair}: p={res['p_value_one_sided']:.4f}")

    construct_results = report.get('construct_validity', {})
    for pair, res in construct_results.items():
        if isinstance(res, dict) and 'matched' in res:
            status = "✅" if res['matched'] else "❌"
            print(f"  {status} {pair}: r={res['pearson_r']:.3f}, p={res['p_value']:.4f}")


if __name__ == "__main__":
    main()
