"""Multi-Seed Statistical Significance Testing.

Runs each algorithm (HMARL, IPPO, SHPPO, MAPPO, Random) over N seeds,
computes mean ± std, and runs Mann-Whitney U test for HMARL vs each baseline.

Usage:
    python scripts/stat_test.py --seeds 5 --timesteps 3000000 --eval-episodes 100
    python scripts/stat_test.py --seeds 3 --timesteps 100000 --quick
"""

import argparse
import json
import os
import sys
import time
import traceback
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from hmarl.env import create_raw_env, NUM_AGENTS, extract_game_state
from hmarl.policy import HierarchicalActorCritic, HierarchicalController, SubGoalEmbedding, SUBGOAL_EMBED_DIM
from hmarl.expert import ExpertPolicyAllAgents, ExpertPolicyEval
from hmarl.reward import PassTracker, RCITracker, BallProgressionTracker, compute_hierarchical_reward
from hmarl.metrics import compute_win_rate, compute_goal_difference, compute_all_metrics, print_metrics
from hmarl.rci import compute_rci
from hmarl.utils import (
    set_seed, extract_obs_vector, OBS_DIM, HIDDEN_DIM, HEAD_DIM,
    ACTION_SPACE_SIZE, EPISODE_MAX_STEPS, ProgressTracker,
    cleanup_temp_dirs, load_ckpt,
)
from hmarl.run_logging import run_metadata

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not installed. Statistical tests will be skipped.")
    print("  Install with: pip install scipy")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_hmarl(timesteps: int, seed: int, output_dir: str) -> str:
    """Train HMARL for one seed. Returns checkpoint path."""
    import train as train_mod
    set_seed(seed)
    trainer = train_mod.HMARLTrainer(
        total_timesteps=timesteps,
        log_dir=output_dir,
        model_dir=os.path.join(output_dir, "checkpoints"),
    )
    trainer.train()
    return os.path.join(trainer.model_dir, "hmarl_model.pt")


def train_flat(algorithm: str, timesteps: int, seed: int, output_dir: str) -> str:
    """Train a flat baseline for one seed. Returns checkpoint path."""
    set_seed(seed)

    if algorithm == "ippo":
        from hmarl.ippo import IPPOTrainer
        trainer = IPPOTrainer(total_timesteps=timesteps, log_dir=output_dir)
    elif algorithm == "shppo":
        from hmarl.shppo import SHPPOTrainer
        trainer = SHPPOTrainer(total_timesteps=timesteps, log_dir=output_dir)
    elif algorithm == "mappo":
        from hmarl.mappo import MAPPOTrainer
        trainer = MAPPOTrainer(total_timesteps=timesteps, log_dir=output_dir)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    trainer.train()
    if algorithm == "mappo":
        return os.path.join(output_dir, "mappo_checkpoint.pt")
    return os.path.join(output_dir, f"{algorithm}_model.pt")


def evaluate_hmarl_from_checkpoint(ckpt_path: str, num_episodes: int, seed: int) -> Dict:
    """Evaluate HMARL model from checkpoint."""
    set_seed(seed)
    policy = HierarchicalActorCritic(
        obs_dim=OBS_DIM, subgoal_embed_dim=SUBGOAL_EMBED_DIM,
        hidden_dim=HIDDEN_DIM, head_dim=HEAD_DIM, action_dim=ACTION_SPACE_SIZE,
    ).to(DEVICE)
    subgoal_emb = SubGoalEmbedding().to(DEVICE)
    ckpt = load_ckpt(ckpt_path, map_location='cpu')
    policy.load_state_dict(ckpt['policy_state'])
    subgoal_emb.load_state_dict(ckpt['subgoal_embedding_state'])
    policy.eval()
    subgoal_emb.eval()

    env = create_raw_env(render=False)
    controller = HierarchicalController()
    expert = ExpertPolicyAllAgents()
    # Anti-circularity: RCI reference = detuned expert (same as scripts/eval.py)
    expert.policy = ExpertPolicyEval()

    match_results = []
    all_rewards = []
    all_game_rewards = []  # game-only (GRF) reward per episode — comparable across algorithms
    all_actual = []
    all_ideal = []
    all_states = []
    all_goals_for = []
    all_goals_against = []

    for _ in range(num_episodes):
        reset_result = env.reset()
        obs_raw = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        game_state = extract_game_state(obs_raw)
        pass_tracker = PassTracker()
        rci_tracker = RCITracker(NUM_AGENTS)
        ball_prog_tracker = BallProgressionTracker()
        ep_reward = 0.0
        ep_game_reward = 0.0
        ep_actual = []
        ep_ideal = []
        ep_states = []
        prev_gs = None
        done = False
        step = 0

        while not done and step < EPISODE_MAX_STEPS:
            macro = controller.get_macro_strategy(game_state)
            sub_goals = controller.get_sub_goals(game_state, macro)
            ideal = expert.get_ideal_actions(game_state, sub_goals, macro)

            joint_actions = []
            for i in range(NUM_AGENTS):
                obs_vec = extract_obs_vector(game_state, i)
                with torch.no_grad():
                    sg_emb = subgoal_emb(torch.LongTensor([sub_goals[i]]).to(DEVICE)).cpu().numpy().flatten()
                    obs_t = torch.FloatTensor(obs_vec).unsqueeze(0).to(DEVICE)
                    sg_t = torch.FloatTensor(sg_emb).unsqueeze(0).to(DEVICE)
                    logits, _ = policy(obs_t, sg_t)
                    joint_actions.append(logits.argmax(dim=-1).item())

            step_result = env.step(joint_actions)
            if len(step_result) == 5:
                obs_raw, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs_raw, reward, done, info = step_result
            team_reward = float(np.sum(reward))
            ep_game_reward += team_reward
            new_gs = extract_game_state(obs_raw)
            pass_tracker.update(new_gs, prev_gs)
            total_r, _ = compute_hierarchical_reward(team_reward, new_gs, joint_actions, ideal, pass_tracker, rci_tracker, ball_progression_tracker=ball_prog_tracker)
            ep_reward += total_r
            ep_actual.append(joint_actions)
            ep_ideal.append(ideal)
            ep_states.append(new_gs)
            prev_gs = game_state
            game_state = new_gs
            step += 1

        score = game_state.get('score', [0, 0])
        gf, ga = (score[0], score[1]) if isinstance(score, (list, tuple)) and len(score) >= 2 else (0, 0)
        match_results.append('win' if gf > ga else ('loss' if gf < ga else 'draw'))
        all_rewards.append(ep_reward)
        all_game_rewards.append(ep_game_reward)
        all_actual.extend(ep_actual)
        all_ideal.extend(ep_ideal)
        all_states.extend(ep_states)
        all_goals_for.append(gf)
        all_goals_against.append(ga)

    env.close()

    metrics = compute_all_metrics(all_actual, all_states, all_ideal,
                                  goals_for=sum(all_goals_for), goals_against=sum(all_goals_against),
                                  cumulative_reward=sum(all_rewards))
    metrics['win_rate'] = compute_win_rate(match_results)
    metrics['goal_difference'] = compute_goal_difference(all_goals_for, all_goals_against)
    metrics['avg_game_reward'] = float(np.mean(all_game_rewards))
    return metrics


def evaluate_flat(algorithm: str, num_episodes: int, seed: int, model_path: str = None) -> Dict:
    """Evaluate flat baseline using SB3 or random policy."""
    set_seed(seed)
    env = create_raw_env(render=False)

    if model_path is None:
        model_path = f"dumps/{algorithm}_model.pt"
    if not os.path.exists(model_path):
        print(f"  WARNING: No model found at {model_path}. Running random.")
        algorithm = "random"

    if algorithm == "random":
        env.close()
        return evaluate_random(num_episodes, seed)

    # Load model ONCE before episode loop
    if algorithm == "ippo":
        from hmarl.ippo import FlatActorCritic
        model = FlatActorCritic().to(DEVICE)
    elif algorithm == "shppo":
        from hmarl.shppo import HeterogeneousActorCritic
        model = HeterogeneousActorCritic().to(DEVICE)
    elif algorithm == "mappo":
        from hmarl.mappo import SharedActorCritic
        model = SharedActorCritic().to(DEVICE)
    else:
        env.close()
        return {}

    ckpt = load_ckpt(model_path, map_location='cpu')
    model.load_state_dict(ckpt['policy_state'])
    model.eval()

    controller = HierarchicalController()
    expert = ExpertPolicyAllAgents()
    # Anti-circularity: RCI reference = detuned expert (same as scripts/eval.py)
    expert.policy = ExpertPolicyEval()
    all_match_results = []
    all_rewards = []
    all_game_rewards = []  # game-only (GRF) reward per episode
    all_actual = []
    all_ideal = []
    all_states = []
    all_goals_for = []
    all_goals_against = []

    ep_prog = ProgressTracker(num_episodes, "Eval Episodes")
    for _ in range(num_episodes):
        reset_result = env.reset()
        obs_raw = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        game_state = extract_game_state(obs_raw)
        pass_tracker = PassTracker()
        rci_tracker = RCITracker(NUM_AGENTS)
        ball_prog_tracker = BallProgressionTracker()
        done = False
        ep_reward = 0.0
        ep_game_reward = 0.0
        ep_actual = []
        ep_ideal = []
        ep_states = []
        prev_gs = None
        step = 0

        while not done and step < EPISODE_MAX_STEPS:
            macro = controller.get_macro_strategy(game_state)
            sub_goals = controller.get_sub_goals(game_state, macro)
            ideal = expert.get_ideal_actions(game_state, sub_goals, macro)

            joint_actions = []
            for i in range(NUM_AGENTS):
                obs_vec = extract_obs_vector(game_state, i)
                obs_t = torch.FloatTensor(obs_vec).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    if algorithm == "mappo":
                        logits = model.policy_head(model.actor(obs_t))
                    else:
                        logits, _ = model(obs_t)
                    joint_actions.append(logits.argmax(dim=-1).item())

            step_result = env.step(joint_actions)
            if len(step_result) == 5:
                obs_raw, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs_raw, reward, done, info = step_result
            team_reward = float(np.sum(reward))
            ep_game_reward += team_reward
            new_gs = extract_game_state(obs_raw)
            pass_tracker.update(new_gs, prev_gs)
            total_r, _ = compute_hierarchical_reward(team_reward, new_gs, joint_actions, ideal, pass_tracker, rci_tracker, ball_progression_tracker=ball_prog_tracker)
            ep_reward += total_r
            ep_actual.append(joint_actions)
            ep_ideal.append(ideal)
            ep_states.append(new_gs)
            prev_gs = game_state
            game_state = new_gs
            step += 1

        game_state = extract_game_state(obs_raw)
        score = game_state.get('score', [0, 0])
        gf, ga = (score[0], score[1]) if isinstance(score, (list, tuple)) and len(score) >= 2 else (0, 0)
        all_match_results.append('win' if gf > ga else ('loss' if gf < ga else 'draw'))
        all_rewards.append(ep_reward)
        all_game_rewards.append(ep_game_reward)
        all_actual.extend(ep_actual)
        all_ideal.extend(ep_ideal)
        all_states.extend(ep_states)
        all_goals_for.append(gf)
        all_goals_against.append(ga)
        ep_prog.update()

    ep_prog.done()
    env.close()

    metrics = compute_all_metrics(all_actual, all_states, all_ideal,
                                  goals_for=sum(all_goals_for), goals_against=sum(all_goals_against),
                                  cumulative_reward=sum(all_rewards))
    metrics['win_rate'] = compute_win_rate(all_match_results)
    metrics['goal_difference'] = compute_goal_difference(all_goals_for, all_goals_against)
    metrics['avg_game_reward'] = float(np.mean(all_game_rewards))
    return metrics




def evaluate_random(num_episodes: int, seed: int) -> Dict:
    """Evaluate random baseline (same metric pipeline as trained agents)."""
    import random
    set_seed(seed)
    env = create_raw_env(render=False)
    controller = HierarchicalController()
    expert = ExpertPolicyAllAgents()
    # Anti-circularity: RCI reference = detuned expert (same as scripts/eval.py)
    expert.policy = ExpertPolicyEval()

    match_results = []
    all_game_rewards = []
    all_actual = []
    all_ideal = []
    all_states = []
    all_goals_for = []
    all_goals_against = []

    for ep in range(num_episodes):
        obs = env.reset()
        obs_raw = obs[0] if isinstance(obs, tuple) else obs
        game_state = extract_game_state(obs_raw)
        done = False
        ep_game_reward = 0.0
        step = 0

        while not done and step < EPISODE_MAX_STEPS:
            macro = controller.get_macro_strategy(game_state)
            sub_goals = controller.get_sub_goals(game_state, macro)
            ideal = expert.get_ideal_actions(game_state, sub_goals, macro)
            joint_actions = [random.randint(0, 18) for _ in range(11)]
            step_result = env.step(joint_actions)
            if len(step_result) == 5:
                obs_raw, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs_raw, reward, done, info = step_result
            ep_game_reward += float(np.sum(reward))
            game_state = extract_game_state(obs_raw)
            all_actual.append(joint_actions)
            all_ideal.append(ideal)
            all_states.append(game_state)
            step += 1

        score = game_state.get('score', [0, 0])
        gf, ga = (score[0], score[1]) if isinstance(score, (list, tuple)) and len(score) >= 2 else (0, 0)
        match_results.append('win' if gf > ga else ('loss' if gf < ga else 'draw'))
        all_goals_for.append(gf)
        all_goals_against.append(ga)
        all_game_rewards.append(ep_game_reward)

    env.close()
    metrics = compute_all_metrics(all_actual, all_states, all_ideal,
                                  goals_for=sum(all_goals_for),
                                  goals_against=sum(all_goals_against),
                                  cumulative_reward=sum(all_game_rewards))
    metrics['win_rate'] = compute_win_rate(match_results)
    metrics['goal_difference'] = compute_goal_difference(all_goals_for, all_goals_against)
    metrics['avg_game_reward'] = float(np.mean(all_game_rewards))
    return metrics


def run_statistical_test(hmarl_values: List[float], baseline_values: List[float],
                         baseline_name: str) -> Dict:
    """Run Mann-Whitney U test comparing HMARL to a baseline."""
    result = {'baseline': baseline_name, 'n_seeds': len(hmarl_values)}

    if not HAS_SCIPY:
        result['test'] = 'skipped (scipy not installed)'
        return result

    if len(hmarl_values) < 2 or len(baseline_values) < 2:
        result['test'] = 'skipped (need >= 2 seeds)'
        return result

    if len(set(hmarl_values + baseline_values)) == 1:
        result['test'] = 'degenerate (all values identical across both groups — no variance to test)'
        result['p_value'] = 1.0
        result['significant'] = False
        return result

    try:
        u_stat, p_value = stats.mannwhitneyu(
            hmarl_values, baseline_values, alternative='greater'
        )
        result['test'] = 'Mann-Whitney U (one-sided: HMARL > baseline)'
        result['u_statistic'] = float(u_stat)
        result['p_value'] = float(p_value)
        result['significant'] = bool(p_value < 0.05)
    except ValueError as e:
        result['test'] = f'error: {e}'

    return result


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Statistical Testing")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--timesteps", type=int, default=3_000_000, help="Training timesteps")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Evaluation episodes per seed")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--algorithms", nargs="+", default=["hmarl", "random"],
                        help="Algorithms to test")
    parser.add_argument("--output", type=str, default="evaluation_results/stat_test_results.json")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 100k steps, 10 eval episodes")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; evaluate --model-path per eval seed "
                             "(seeds vary the environment rollouts, not training)")
    parser.add_argument("--model-path", type=str, default="checkpoints/hmarl_model.pt",
                        help="Checkpoint for --eval-only (hmarl)")
    args = parser.parse_args()

    if args.quick:
        args.timesteps = 100_000
        args.eval_episodes = 10
        args.seeds = min(args.seeds, 3)

    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    all_results = {}
    hmarl_win_rates = []
    hmarl_gds = []
    main_start = time.time()

    for alg in args.algorithms:
        print(f"\n{'='*60}")
        print(f"  Algorithm: {alg.upper()} | Seeds: {args.seeds} | Timesteps: {args.timesteps:,}")
        print(f"{'='*60}")

        seed_win_rates = []
        seed_rewards = []
        seed_rci = []
        seed_gd = []
        # Keys consumed by validate_rci.py --load-results (construct validity pairs)
        SEED_KEYS = ['rci_strict', 'rci_nomove', 'fai_mean', 'positional_entropy',
                     'compactness_mean', 'psr', 'ppr']
        seed_extra = {k: [] for k in SEED_KEYS}

        seed_prog = ProgressTracker(args.seeds, f"Seeds ({alg.upper()})")
        for s in range(args.seeds):
            seed = args.base_seed + s
            seed_prog.update(extra=f"seed={seed}")

            try:
                if alg == "hmarl":
                    if args.eval_only:
                        if not os.path.exists(args.model_path):
                            raise FileNotFoundError(
                                f"--eval-only needs an existing checkpoint: {args.model_path}")
                        metrics = evaluate_hmarl_from_checkpoint(
                            args.model_path, args.eval_episodes, seed + 1000)
                    else:
                        ckpt = train_hmarl(args.timesteps, seed, f"stat_temp/{alg}_{s}")
                        metrics = evaluate_hmarl_from_checkpoint(ckpt, args.eval_episodes, seed + 1000)
                elif alg == "random":
                    metrics = evaluate_random(args.eval_episodes, seed)
                elif args.eval_only:
                    print(f"    WARNING: --eval-only not supported for {alg} (no checkpoint); skipping")
                    continue
                else:
                    ckpt = train_flat(alg, args.timesteps, seed, f"stat_temp/{alg}_{s}")
                    metrics = evaluate_flat(alg, args.eval_episodes, seed + 1000, model_path=ckpt)

                seed_win_rates.append(metrics.get('win_rate', 0))
                seed_gd.append(metrics.get('goal_difference', 0))
                # Uniform metric across algorithms: GRF game-only reward per episode.
                # (Old bug: hmarl reported shaped cumulative_reward vs random's
                # avg_game_reward — incomparable scales.)
                seed_rewards.append(metrics.get('avg_game_reward', 0))
                seed_rci.append(metrics.get('rci_cat', 0) if 'rci_cat' in metrics else 0)
                for k in SEED_KEYS:
                    v = metrics.get(k)
                    if isinstance(v, (int, float)):
                        seed_extra[k].append(float(v))

                print(f"\n    WR: {metrics.get('win_rate', 0):.1f}% | "
                      f"GD: {metrics.get('goal_difference', 'N/A')} | "
                      f"RCI: {metrics.get('rci_cat', 'N/A')}")

            except Exception:
                traceback.print_exc()

        seed_prog.done()

        if seed_win_rates:
            all_results[alg] = {
                'win_rates': seed_win_rates,
                'win_rate_mean': float(np.mean(seed_win_rates)),
                'win_rate_std': float(np.std(seed_win_rates)),
                'goal_differences': seed_gd,
                'goal_difference_mean': float(np.mean(seed_gd)),
                'goal_difference_std': float(np.std(seed_gd)),
                'rewards': seed_rewards,
                'rci': seed_rci,
            }
            # Per-seed coordination metrics for validate_rci.py --load-results
            for k, vals in seed_extra.items():
                if vals:
                    all_results[alg][k] = vals

            if alg == "hmarl":
                hmarl_win_rates = seed_win_rates
                hmarl_gds = seed_gd

            # Incremental flush: a multi-hour run that is interrupted (host
            # reboot, OOM, manual stop) otherwise loses every finished
            # algorithm, since the JSON is only written at the very end.
            with open(args.output, 'w') as f:
                json.dump({'meta': run_metadata(script="scripts/stat_test.py",
                                                train_time_s=round(time.time() - main_start, 2),
                                                algorithms=args.algorithms),
                           'config': {'seeds': args.seeds, 'timesteps': args.timesteps,
                                      'eval_episodes': args.eval_episodes,
                                      'base_seed': args.base_seed, 'partial': True,
                                      'algorithms_done': list(all_results.keys())},
                           'results': all_results, 'statistical_tests': []}, f, indent=2)
                print(f"    [partial results flushed: {list(all_results.keys())}]")

    # Statistical tests
    print(f"\n{'='*60}")
    print(f"  STATISTICAL TESTS (HMARL vs baselines, Mann-Whitney U on goal difference)")
    print(f"{'='*60}")

    stat_tests = []
    if hmarl_gds:
        hmarl_res = all_results.get('hmarl', {})
        for alg, res in all_results.items():
            if alg == "hmarl":
                continue
            baseline_gds = res.get('goal_differences', [])
            test_result = run_statistical_test(hmarl_gds, baseline_gds, alg)
            stat_tests.append(test_result)
            print(f"\n  HMARL vs {alg}:")
            # NOTE (fixed 2026-10-13): these two lines used to print `res`
            # (the baseline) for BOTH rows; HMARL row now uses hmarl_res.
            print(f"    HMARL WR: {hmarl_res.get('win_rate_mean', 0):.1f}% ± "
                  f"{hmarl_res.get('win_rate_std', 0):.1f}% | "
                  f"GD: {np.mean(hmarl_gds):.1f} ± {np.std(hmarl_gds):.1f}")
            print(f"    {alg} WR: {res.get('win_rate_mean', 0):.1f}% ± "
                  f"{res.get('win_rate_std', 0):.1f}% | GD: {np.mean(baseline_gds):.1f} ± {np.std(baseline_gds):.1f}")
            if 'p_value' in test_result:
                sig = "YES" if test_result['significant'] else "NO"
                print(f"    p-value: {test_result['p_value']:.4f} (significant: {sig})")
            else:
                print(f"    Test: {test_result.get('test', 'N/A')}")

    # Save results
    output = {
        'meta': run_metadata(
            script="scripts/stat_test.py",
            train_time_s=round(time.time() - main_start, 2),
            algorithms=args.algorithms,
        ),
        'config': {
            'seeds': args.seeds,
            'timesteps': args.timesteps,
            'eval_episodes': args.eval_episodes,
            'base_seed': args.base_seed,
            'eval_only': args.eval_only,
            'model_path': args.model_path if args.eval_only else None,
            'note': ('eval_only: seeds vary environment rollouts on ONE checkpoint, '
                     'not independent training runs' if args.eval_only else
                     'seeds = independent training runs'),
        },
        'results': all_results,
        'statistical_tests': stat_tests,
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")

    cleanup_temp_dirs()


if __name__ == "__main__":
    main()
