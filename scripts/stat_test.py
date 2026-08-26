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
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from hmarl.env import create_raw_env, NUM_AGENTS, extract_game_state, ACTION_SPACE_SIZE
from hmarl.policy import HierarchicalActorCritic, HierarchicalController, SubGoalEmbedding, SUBGOAL_EMBED_DIM
from hmarl.expert import ExpertPolicyAllAgents
from hmarl.reward import PassTracker, RCITracker, compute_hierarchical_reward
from hmarl.metrics import compute_win_rate, compute_goal_difference, compute_all_metrics, print_metrics
from hmarl.rci import compute_rci

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not installed. Statistical tests will be skipped.")
    print("  Install with: pip install scipy")


HIDDEN_DIM = 256
HEAD_DIM = 128
OBS_DIM = 115
EPISODE_MAX_STEPS = 3000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_obs_vector(game_state, player_idx):
    """Extract 115-dim observation vector."""
    features = []
    ball = game_state.get('ball', [0, 0, 0])
    features.extend(ball)
    features.extend(game_state.get('ball_direction', [0, 0, 0]))
    features.extend(game_state.get('ball_rotation', [0, 0, 0]))
    features.append(float(game_state.get('ball_owned_team', -1)))
    features.append(float(game_state.get('ball_owned_player', -1)))
    for i in range(11):
        pos = game_state.get('left_team', [[0, 0]] * 11)[i]
        d = game_state.get('left_team_direction', [[0, 0]] * 11)[i] if 'left_team_direction' in game_state else [0, 0]
        t = game_state.get('left_team_tired_factor', [0.0] * 11)[i] if 'left_team_tired_factor' in game_state else 0.0
        y = game_state.get('left_team_yellow_card', [0] * 11)[i] if 'left_team_yellow_card' in game_state else 0
        r = game_state.get('left_team_roles', [5] * 11)[i] if 'left_team_roles' in game_state else 5
        features.append(1.0 if i == player_idx else 0.0)
        features.extend(pos)
        features.extend(d)
        features.append(t)
        features.append(float(y))
        features.append(float(r))
    features = features[:OBS_DIM]
    while len(features) < OBS_DIM:
        features.append(0.0)
    return np.array(features, dtype=np.float32)


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
    return output_dir


def evaluate_hmarl_from_checkpoint(ckpt_path: str, num_episodes: int, seed: int) -> Dict:
    """Evaluate HMARL model from checkpoint."""
    set_seed(seed)
    policy = HierarchicalActorCritic(
        obs_dim=OBS_DIM, subgoal_embed_dim=SUBGOAL_EMBED_DIM,
        hidden_dim=HIDDEN_DIM, head_dim=HEAD_DIM, action_dim=ACTION_SPACE_SIZE,
    ).to(DEVICE)
    subgoal_emb = SubGoalEmbedding().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    policy.load_state_dict(ckpt['policy_state'])
    subgoal_emb.load_state_dict(ckpt['subgoal_embedding_state'])
    policy.eval()
    subgoal_emb.eval()

    env = create_raw_env(render=False)
    controller = HierarchicalController()
    expert = ExpertPolicyAllAgents()

    match_results = []
    all_rewards = []
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
        ep_reward = 0.0
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
            new_gs = extract_game_state(obs_raw)
            pass_tracker.update(new_gs, prev_gs)
            total_r, _ = compute_hierarchical_reward(team_reward, new_gs, joint_actions, ideal, pass_tracker, rci_tracker)
            ep_reward += total_r
            ep_actual.append(joint_actions)
            ep_ideal.append(ideal)
            ep_states.append(new_gs)
            prev_gs = game_state
            game_state = new_gs
            step += 1

        score = info.get('score', [0, 0])
        gf, ga = (score[0], score[1]) if isinstance(score, (list, tuple)) and len(score) >= 2 else (0, 0)
        match_results.append('win' if gf > ga else ('loss' if gf < ga else 'draw'))
        all_rewards.append(ep_reward)
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
    return metrics


def evaluate_flat(algorithm: str, num_episodes: int, seed: int) -> Dict:
    """Evaluate flat baseline using SB3 or random policy."""
    set_seed(seed)
    env = create_raw_env(render=False)
    controller = HierarchicalController()
    expert = ExpertPolicyAllAgents()

    # Load trained model
    model_path = f"dumps/{algorithm}_model.pt"
    if not os.path.exists(model_path):
        print(f"  WARNING: No model found at {model_path}. Running random.")
        algorithm = "random"

    if algorithm == "random":
        return evaluate_random(num_episodes, seed)

    # Load model and evaluate
    set_seed(seed)
    all_match_results = []
    all_rewards = []

    for _ in range(num_episodes):
        reset_result = env.reset()
        obs_raw = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        game_state = extract_game_state(obs_raw)
        done = False
        ep_reward = 0.0
        step = 0

        while not done and step < EPISODE_MAX_STEPS:
            joint_actions = []
            for i in range(NUM_AGENTS):
                obs_vec = extract_obs_vector(game_state, i)
                obs_t = torch.FloatTensor(obs_vec).unsqueeze(0).to(DEVICE)

                if algorithm == "ippo":
                    from hmarl.ippo import FlatActorCritic
                    model = FlatActorCritic().to(DEVICE)
                elif algorithm == "shppo":
                    from hmarl.shppo import SharedActorCritic
                    model = SharedActorCritic().to(DEVICE)
                elif algorithm == "mappo":
                    from hmarl.mappo import DecentralizedActor
                    model = DecentralizedActor().to(DEVICE)
                else:
                    break

                if i == 0:
                    ckpt = torch.load(model_path, map_location='cpu')
                    if algorithm == "mappo":
                        model.load_state_dict(ckpt['actor_state'])
                    else:
                        model.load_state_dict(ckpt['policy_state'])
                    model.eval()

                with torch.no_grad():
                    logits, _ = model(obs_t) if algorithm != "mappo" else (model(obs_t), None)
                    joint_actions.append(logits.argmax(dim=-1).item())

            step_result = env.step(joint_actions)
            if len(step_result) == 5:
                obs_raw, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs_raw, reward, done, info = step_result
            ep_reward += float(np.sum(reward))
            game_state = extract_game_state(obs_raw)
            step += 1

        score = info.get('score', [0, 0])
        gf, ga = (score[0], score[1]) if isinstance(score, (list, tuple)) and len(score) >= 2 else (0, 0)
        all_match_results.append('win' if gf > ga else ('loss' if gf < ga else 'draw'))
        all_rewards.append(ep_reward)

    env.close()
    return {'win_rate': compute_win_rate(all_match_results), 'avg_reward': float(np.mean(all_rewards))}


def evaluate_random(num_episodes: int, seed: int) -> Dict:
    """Evaluate random baseline."""
    import random
    set_seed(seed)
    env = create_raw_env(render=False)

    match_results = []
    all_rewards = []

    for _ in range(num_episodes):
        obs = env.reset()
        obs_raw = obs[0] if isinstance(obs, tuple) else obs
        done = False
        ep_reward = 0.0
        step = 0

        while not done and step < EPISODE_MAX_STEPS:
            joint_actions = [random.randint(0, 18) for _ in range(11)]
            step_result = env.step(joint_actions)
            if len(step_result) == 5:
                obs_raw, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs_raw, reward, done, info = step_result
            ep_reward += float(np.sum(reward))
            step += 1

        score = info.get('score', [0, 0])
        gf, ga = (score[0], score[1]) if isinstance(score, (list, tuple)) and len(score) >= 2 else (0, 0)
        all_match_results.append('win' if gf > ga else ('loss' if gf < ga else 'draw'))
        all_rewards.append(ep_reward)

    env.close()
    return {'win_rate': compute_win_rate(all_match_results), 'avg_reward': float(np.mean(all_rewards))}


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

    try:
        u_stat, p_value = stats.mannwhitneyu(
            hmarl_values, baseline_values, alternative='greater'
        )
        result['test'] = 'Mann-Whitney U (one-sided: HMARL > baseline)'
        result['u_statistic'] = float(u_stat)
        result['p_value'] = float(p_value)
        result['significant'] = p_value < 0.05
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
    args = parser.parse_args()

    if args.quick:
        args.timesteps = 100_000
        args.eval_episodes = 10
        args.seeds = min(args.seeds, 3)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    all_results = {}
    hmarl_win_rates = []

    for alg in args.algorithms:
        print(f"\n{'='*60}")
        print(f"  Algorithm: {alg.upper()} | Seeds: {args.seeds} | Timesteps: {args.timesteps:,}")
        print(f"{'='*60}")

        seed_win_rates = []
        seed_rewards = []
        seed_rci = []

        for s in range(args.seeds):
            seed = args.base_seed + s
            print(f"\n  Seed {s+1}/{args.seeds} (seed={seed})")

            try:
                if alg == "hmarl":
                    ckpt = train_hmarl(args.timesteps, seed, f"stat_temp/{alg}_{s}")
                    metrics = evaluate_hmarl_from_checkpoint(ckpt, args.eval_episodes, seed + 1000)
                elif alg == "random":
                    metrics = evaluate_random(args.eval_episodes, seed)
                else:
                    train_flat(alg, args.timesteps, seed, f"stat_temp/{alg}_{s}")
                    metrics = evaluate_flat(alg, args.eval_episodes, seed + 1000)

                seed_win_rates.append(metrics.get('win_rate', 0))
                seed_rewards.append(metrics.get('avg_reward', 0) if 'avg_reward' in metrics else 0)
                seed_rci.append(metrics.get('rci_cat', 0) if 'rci_cat' in metrics else 0)

                print(f"    WR: {metrics.get('win_rate', 0):.1f}% | "
                      f"GD: {metrics.get('goal_difference', 'N/A')} | "
                      f"RCI: {metrics.get('rci_cat', 'N/A')}")

            except Exception as e:
                print(f"    FAILED: {e}")

        if seed_win_rates:
            all_results[alg] = {
                'win_rates': seed_win_rates,
                'win_rate_mean': float(np.mean(seed_win_rates)),
                'win_rate_std': float(np.std(seed_win_rates)),
                'rewards': seed_rewards,
                'rci': seed_rci,
            }

            if alg == "hmarl":
                hmarl_win_rates = seed_win_rates

    # Statistical tests
    print(f"\n{'='*60}")
    print(f"  STATISTICAL TESTS (HMARL vs baselines)")
    print(f"{'='*60}")

    stat_tests = []
    if hmarl_win_rates:
        for alg, res in all_results.items():
            if alg == "hmarl":
                continue
            baseline_wrs = res.get('win_rates', [])
            test_result = run_statistical_test(hmarl_win_rates, baseline_wrs, alg)
            stat_tests.append(test_result)
            print(f"\n  HMARL vs {alg}:")
            print(f"    HMARL WR: {np.mean(hmarl_win_rates):.1f}% ± {np.std(hmarl_win_rates):.1f}%")
            print(f"    {alg} WR: {np.mean(baseline_wrs):.1f}% ± {np.std(baseline_wrs):.1f}%")
            if 'p_value' in test_result:
                sig = "YES" if test_result['significant'] else "NO"
                print(f"    p-value: {test_result['p_value']:.4f} (significant: {sig})")
            else:
                print(f"    Test: {test_result.get('test', 'N/A')}")

    # Save results
    output = {
        'config': {
            'seeds': args.seeds,
            'timesteps': args.timesteps,
            'eval_episodes': args.eval_episodes,
            'base_seed': args.base_seed,
        },
        'results': all_results,
        'statistical_tests': stat_tests,
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
