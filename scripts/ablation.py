"""Ablation Study for HMARL.

Tests the contribution of each component by systematically removing/altering:
1. Full HMARL (baseline) — all components active
2. No FAI reward (α_H = 0) — formation adherence removed
3. No PPR reward (α_M = 0) — progressive pass removed
4. No RCI reward (α_L = 0) — role coherence removed
5. No reward shaping (all α = 0) — game reward only
6. No hierarchical structure (flat policy, no sub-goal conditioning)
7. Random expert (random ideal actions instead of rule-based)

Usage:
    python scripts/ablation.py --timesteps 300000 --eval-episodes 50
    python scripts/ablation.py --timesteps 100000 --eval-episodes 20 --quick
"""

import argparse
import json
import os
import sys
import time
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
import traceback
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from hmarl.env import (
    create_raw_env, NUM_AGENTS,
    extract_game_state, get_action_category,
)
from hmarl.policy import (
    HierarchicalActorCritic, HierarchicalController,
    SubGoalEmbedding, SUBGOAL_EMBED_DIM,
)
from hmarl.expert import ExpertPolicyAllAgents
from hmarl.reward import (
    PassTracker, RCITracker, BallProgressionTracker, compute_hierarchical_reward,
    compute_fai, ALPHA_HIGH, ALPHA_MID, ALPHA_LOW,
)
from hmarl.metrics import compute_win_rate, compute_goal_difference, compute_all_metrics, print_metrics
from hmarl.rci import compute_rci
from hmarl.utils import (
    set_seed, extract_obs_vector, OBS_DIM, HIDDEN_DIM, HEAD_DIM,
    ACTION_SPACE_SIZE, EPISODE_MAX_STEPS, ProgressTracker, cleanup_own_temp,
    load_ckpt,
)
from hmarl.run_logging import run_metadata

# Reuse the single-source-of-truth trainer + evaluator from stat_test: ablation
# arms only swap hooks (reward fn / sub-goal conditioning / expert), never the
# PPO hyper-parameters, so no stale copy of the trainer can drift.
from stat_test import (
    train_hmarl, evaluate_hmarl_from_checkpoint,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Custom Reward for Ablation
# ---------------------------------------------------------------------------
import hmarl.reward as reward_mod

_orig_compute = reward_mod.compute_hierarchical_reward


def make_custom_reward(
    alpha_h: float,
    alpha_m: float,
    alpha_l: float,
    alpha_prog: float,
    alpha_gk: float,
    game_scale: float,
    enable_fai: bool = True,
    enable_ppr: bool = True,
    enable_rci: bool = True,
    enable_gk_penalty: bool = True,
) -> callable:
    """Create a reward function with specified components enabled/disabled and custom alphas."""
    from hmarl.reward import compute_fai, PassTracker, RCITracker, BallProgressionTracker, get_player_position, get_ball_position, GK_BOX_X, GK_BOX_Y
    def custom_reward(game_reward, game_state, actual_actions, ideal_actions,
                      pass_tracker, rci_tracker, num_agents=NUM_AGENTS,
                      ball_progression_tracker=None,
                      enable_fai_=None, enable_ppr_=None, enable_rci_=None,
                      **kwargs):
        # The trainer forwards its own enable_* flags; the arm's flags live in
        # this closure and deliberately win, so the passed values are ignored.
        # FAI
        rho_fa = compute_fai(game_state, num_agents) if enable_fai else 0.0

        # PPR
        ppr = pass_tracker.get_ppr() if enable_ppr else 0.0

        # RCI
        if enable_rci:
            rci_tracker.update(actual_actions, ideal_actions)
            rci_per_agent = rci_tracker.get_rci_per_agent()
            avg_rci = float(np.mean(rci_per_agent))
        else:
            avg_rci = 0.0

        # Ball progression
        r_progression = 0.0
        if ball_progression_tracker is not None:
            r_progression = ball_progression_tracker.update(game_state)

        # GK penalty
        r_gk = 0.0
        if enable_gk_penalty:
            gk_pos = get_player_position(game_state, 'left', 0)
            overshoot_x = max(0.0, gk_pos[0] - GK_BOX_X)
            overshoot_y = max(0.0, abs(gk_pos[1]) - GK_BOX_Y)
            overshoot = overshoot_x + overshoot_y
            if overshoot > 0:
                r_gk = -alpha_gk * overshoot

        # Combined
        r_high = alpha_h * rho_fa
        r_mid = alpha_m * ppr
        r_low = alpha_l * avg_rci
        r_prog = alpha_prog * r_progression

        total = game_scale * game_reward + r_high + r_mid + r_low + r_prog + r_gk

        breakdown = {
            'game_reward': game_scale * game_reward,
            'fai': rho_fa, 'ppr': ppr, 'rci_avg': avg_rci,
            'r_high': r_high, 'r_mid': r_mid, 'r_low': r_low,
            'r_progression': r_prog, 'r_gk_penalty': r_gk,
            'total': total
        }
        return total, breakdown
    return custom_reward


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def load_and_modify_model(
    base_checkpoint_path: str,
    config_name: str,
    mod_reward_kwargs: Dict[str, Any],
    flat_policy: bool = False,
    seed: int = 42,
) -> str:
    """Load base model and modify its reward function for ablation evaluation."""
    set_seed(seed)

    policy = HierarchicalActorCritic(
        obs_dim=OBS_DIM, subgoal_embed_dim=SUBGOAL_EMBED_DIM,
        hidden_dim=HIDDEN_DIM, head_dim=HEAD_DIM, action_dim=ACTION_SPACE_SIZE,
    ).to(DEVICE)
    subgoal_embedding = SubGoalEmbedding().to(DEVICE)
    ckpt = load_ckpt(base_checkpoint_path, map_location='cpu')
    policy.load_state_dict(ckpt['policy_state'])
    subgoal_embedding.load_state_dict(ckpt['subgoal_embedding_state'])
    policy.eval()
    subgoal_embedding.eval()

    # No need to save/load reward function parameters from checkpoint,
    # they are passed directly to `compute_hierarchical_reward` during eval.

    # Save the modified policy/subgoal_embedding to a temporary checkpoint
    # for evaluation. (No changes needed to policy itself for reward ablations)
    temp_ckpt_dir = f"ablation_temp/{config_name}"
    os.makedirs(temp_ckpt_dir, exist_ok=True)
    temp_ckpt_path = os.path.join(temp_ckpt_dir, "model.pt")
    torch.save({
        'policy_state': policy.state_dict(),
        'subgoal_embedding_state': subgoal_embedding.state_dict(),
    }, temp_ckpt_path)

    return temp_ckpt_path


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_ablation_config(ckpt_path: str, num_episodes: int, seed: int,
                             mod_reward_kwargs: Dict[str, Any],
                             flat_policy: bool = False,
                             random_expert: bool = False) -> Dict:
    """Evaluate an ablation config."""
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
    expert.policy = ExpertPolicyEval() # Anti-circularity for RCI evaluation

    reward_fn = make_custom_reward(**mod_reward_kwargs)

    match_results = []
    all_rewards = []
    all_game_rewards = []
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
        prev_gs = None
        done = False
        step = 0

        while not done and step < EPISODE_MAX_STEPS:
            macro = controller.get_macro_strategy(game_state)
            sub_goals = controller.get_sub_goals(game_state, macro)

            if random_expert:
                ideal_actions = [np.random.randint(0, 18) for _ in range(NUM_AGENTS)]
            else:
                ideal_actions = expert.get_ideal_actions(game_state, sub_goals, macro)

            joint_actions = []
            for i in range(NUM_AGENTS):
                obs_vec = extract_obs_vector(game_state, i)
                if flat_policy:
                    sg_embed = np.zeros(SUBGOAL_EMBED_DIM, dtype=np.float32)
                else:
                    with torch.no_grad():
                        sg_embed = subgoal_emb(
                            torch.LongTensor([sub_goals[i]]).to(DEVICE)
                        ).cpu().numpy().flatten()
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs_vec).unsqueeze(0).to(DEVICE)
                    sg_t = torch.FloatTensor(sg_embed).unsqueeze(0).to(DEVICE)
                    logits, _ = policy(obs_t, sg_t)
                    joint_actions.append(logits.argmax(dim=-1).item())

            step_result = env.step(joint_actions)
            if len(step_result) == 5:
                obs_raw, game_reward_raw, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs_raw, game_reward_raw, done, info = step_result
            
            team_game_reward_raw = float(np.sum(game_reward_raw))
            new_game_state = extract_game_state(obs_raw)

            pass_tracker.update(new_game_state, prev_gs)
            total_r, _ = reward_fn(
                team_game_reward_raw, new_game_state, joint_actions, ideal_actions,
                pass_tracker, rci_tracker,
                ball_progression_tracker=ball_prog_tracker,
            )
            ep_reward += total_r
            ep_game_reward += team_game_reward_raw

            prev_gs = game_state
            game_state = new_game_state
            step += 1

        score = game_state.get('score', [0, 0])
        gf, ga = (score[0], score[1]) if isinstance(score, (list, tuple)) and len(score) >= 2 else (0, 0)
        match_results.append('win' if gf > ga else ('loss' if gf < ga else 'draw'))
        all_rewards.append(ep_reward)
        all_game_rewards.append(ep_game_reward)
        all_goals_for.append(gf)
        all_goals_against.append(ga)

    env.close()

    return {
        'win_rate': compute_win_rate(match_results),
        'avg_reward': float(np.mean(all_rewards)),
        'avg_game_reward': float(np.mean(all_game_rewards)),
        'goal_difference': compute_goal_difference(all_goals_for, all_goals_against),
    }


# ---------------------------------------------------------------------------
# Ablation Configurations
# ---------------------------------------------------------------------------
ABLATION_CONFIGS = {
    'full_hmarl': {
        'description': 'Full HMARL (baseline)',
        'mod_reward_kwargs': {
            'alpha_h': reward_mod.ALPHA_HIGH,
            'alpha_m': reward_mod.ALPHA_MID,
            'alpha_l': reward_mod.ALPHA_LOW,
            'alpha_prog': reward_mod.ALPHA_PROG,
            'alpha_gk': reward_mod.ALPHA_GK,
            'game_scale': reward_mod.GAME_SCALE,
            'enable_fai': True, 'enable_ppr': True, 'enable_rci': True, 'enable_gk_penalty': True
        },
        'flat_policy': False,
        'random_expert': False,
    },
    'no_fai': {
        'description': 'Without FAI reward (α_H = 0)',
        'mod_reward_kwargs': {
            'alpha_h': 0.0,
            'alpha_m': reward_mod.ALPHA_MID,
            'alpha_l': reward_mod.ALPHA_LOW,
            'alpha_prog': reward_mod.ALPHA_PROG,
            'alpha_gk': reward_mod.ALPHA_GK,
            'game_scale': reward_mod.GAME_SCALE,
            'enable_fai': False, 'enable_ppr': True, 'enable_rci': True, 'enable_gk_penalty': True
        },
        'flat_policy': False,
        'random_expert': False,
    },
    'no_ppr': {
        'description': 'Without PPR reward (α_M = 0)',
        'mod_reward_kwargs': {
            'alpha_h': reward_mod.ALPHA_HIGH,
            'alpha_m': 0.0,
            'alpha_l': reward_mod.ALPHA_LOW,
            'alpha_prog': reward_mod.ALPHA_PROG,
            'alpha_gk': reward_mod.ALPHA_GK,
            'game_scale': reward_mod.GAME_SCALE,
            'enable_fai': True, 'enable_ppr': False, 'enable_rci': True, 'enable_gk_penalty': True
        },
        'flat_policy': False,
        'random_expert': False,
    },
    'no_rci': {
        'description': 'Without RCI reward (α_L = 0)',
        'mod_reward_kwargs': {
            'alpha_h': reward_mod.ALPHA_HIGH,
            'alpha_m': reward_mod.ALPHA_MID,
            'alpha_l': 0.0,
            'alpha_prog': reward_mod.ALPHA_PROG,
            'alpha_gk': reward_mod.ALPHA_GK,
            'game_scale': reward_mod.GAME_SCALE,
            'enable_fai': True, 'enable_ppr': True, 'enable_rci': False, 'enable_gk_penalty': True
        },
        'flat_policy': False,
        'random_expert': False,
    },
    'no_reward_shaping': {
        'description': 'No reward shaping (game reward only)',
        'mod_reward_kwargs': {
            'alpha_h': 0.0,
            'alpha_m': 0.0,
            'alpha_l': 0.0,
            'alpha_prog': 0.0,
            'alpha_gk': 0.0,
            'game_scale': reward_mod.GAME_SCALE,
            'enable_fai': False, 'enable_ppr': False, 'enable_rci': False, 'enable_gk_penalty': False
        },
        'flat_policy': False,
        'random_expert': False,
    },
    'no_gk_penalty': {
        'description': 'Without GK positional penalty (α_GK = 0)',
        'mod_reward_kwargs': {
            'alpha_h': reward_mod.ALPHA_HIGH,
            'alpha_m': reward_mod.ALPHA_MID,
            'alpha_l': reward_mod.ALPHA_LOW,
            'alpha_prog': reward_mod.ALPHA_PROG,
            'alpha_gk': 0.0,
            'game_scale': reward_mod.GAME_SCALE,
            'enable_fai': True, 'enable_ppr': True, 'enable_rci': True, 'enable_gk_penalty': False
        },
        'flat_policy': False,
        'random_expert': False,
    },
    'flat_policy': {
        'description': 'Flat policy (no sub-goal conditioning)',
        'mod_reward_kwargs': { # Shaping still active but policy cannot use sub-goals
            'alpha_h': reward_mod.ALPHA_HIGH,
            'alpha_m': reward_mod.ALPHA_MID,
            'alpha_l': reward_mod.ALPHA_LOW,
            'alpha_prog': reward_mod.ALPHA_PROG,
            'alpha_gk': reward_mod.ALPHA_GK,
            'game_scale': reward_mod.GAME_SCALE,
            'enable_fai': True, 'enable_ppr': True, 'enable_rci': True, 'enable_gk_penalty': True
        },
        'flat_policy': True,
        'random_expert': False,
    },
    'random_expert': {
        'description': 'Random expert (random ideal actions)',
        'mod_reward_kwargs': { # Shaping still active but expert is random
            'alpha_h': reward_mod.ALPHA_HIGH,
            'alpha_m': reward_mod.ALPHA_MID,
            'alpha_l': reward_mod.ALPHA_LOW,
            'alpha_prog': reward_mod.ALPHA_PROG,
            'alpha_gk': reward_mod.ALPHA_GK,
            'game_scale': reward_mod.GAME_SCALE,
            'enable_fai': True, 'enable_ppr': True, 'enable_rci': True, 'enable_gk_penalty': True
        },
        'flat_policy': False,
        'random_expert': True,
    },
}


def main():
    parser = argparse.ArgumentParser(description="HMARL Ablation Study")
    parser.add_argument("--timesteps", type=int, default=300_000, help="Training timesteps per config")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Evaluation episodes per config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="Shift the seed (parallel shards)")
    parser.add_argument("--temp-root", type=str, default="ablation_temp",
                        help="Per-shard temp dir (parallel shards must not share one)")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Specific configs to run (default: all)")
    parser.add_argument("--output", type=str, default="evaluation_results/ablation_results.json")
    parser.add_argument("--merge-inputs", nargs="+", default=None,
                        help="Merge shard JSONs into --output instead of training")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 100k steps, 20 eval episodes")
    args = parser.parse_args()

    if args.merge_inputs:
        merge_shards(args.merge_inputs, args.output)
        return

    if args.quick:
        args.timesteps = 100_000
        args.eval_episodes = 20

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(args.temp_root, exist_ok=True)

    configs_to_run = args.configs if args.configs else list(ABLATION_CONFIGS.keys())
    results = {}

    print(f"Ablation Study | Configs: {len(configs_to_run)} | "
          f"Steps: {args.timesteps:,} | Eval: {args.eval_episodes} episodes")
    print(f"{'='*60}")

    total_start = time.time()

    progress = ProgressTracker(total=len(configs_to_run), label="Ablation configs")
    for config_name in configs_to_run:
        if config_name not in ABLATION_CONFIGS:
            print(f"  WARNING: Unknown config '{config_name}', skipping.")
            progress.update(1, extra=config_name)
            continue

        try:
            config = ABLATION_CONFIGS[config_name]
            print(f"\n--- {config_name}: {config['description']} ---")

            # Same trainer as the main runs: only the ablation hooks differ, so
            # gamma / entropy / GAE / clipping stay identical across arms.
            train_fn = make_custom_reward(**config['mod_reward_kwargs'])
            ckpt = train_hmarl(
                timesteps=args.timesteps,
                seed=args.seed + args.seed_offset,
                output_dir=os.path.join(args.temp_root, config_name),
                reward_fn=train_fn,
                flat_policy=config['flat_policy'],
                random_expert=config['random_expert'],
            )

            eval_result = evaluate_hmarl_from_checkpoint(
                ckpt_path=ckpt,
                num_episodes=args.eval_episodes,
                seed=args.seed + args.seed_offset + 1000,
                flat_policy=config['flat_policy'],
                random_expert=config['random_expert'],
            )

            results[config_name] = {
                'description': config['description'],
                'wins': [eval_result['win_rate']],
                'rewards': [eval_result['cumulative_reward']],
                'game_rewards': [eval_result['avg_game_reward']],
                'goal_differences': [eval_result['goal_difference']],
                'rci': [eval_result['rci_cat']],
                'fai': [eval_result['fai_mean']],
                'checkpoints': [ckpt],
            }

            print(f"  Result: WR={eval_result['win_rate']:.1f}% | "
                  f"Reward={eval_result['cumulative_reward']:.2f} | "
                  f"GD={eval_result['goal_difference']} | "
                  f"RCI={eval_result['rci_cat']:.4f}")
        except Exception:
            traceback.print_exc()
            print(f"  FAILED: {config_name}, skipping.")
        progress.update(1, extra=config_name)

    progress.done()

    # Summary table
    print(f"\n{'='*72}")
    print(f"  ABLATION SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Config':<25} {'Win Rate':>10} {'Avg Reward':>12} {'Goal Diff':>10}")
    print(f"  {'-'*57}")

    full_hmarl_wr = results.get('full_hmarl', {}).get('wins', [0])[0]

    for name, res in results.items():
        wr = res.get('wins', [0])[0]
        rw = res.get('rewards', [0])[0]
        gd = res.get('goal_differences', [0])[0]
        delta = wr - full_hmarl_wr if name != 'full_hmarl' else 0
        delta_str = f" ({delta:+.1f}%)" if name != 'full_hmarl' else ""
        print(f"  {name:<25} {wr:>9.1f}% {rw:>12.2f} {gd:>10}{delta_str}")

    print(f"{'='*72}")

    # Save
    output = {
        'meta': run_metadata(
            script="scripts/ablation.py",
            train_time_s=round(time.time() - total_start, 2),
            configs_run=configs_to_run,
        ),
        'config': {
            'timesteps': args.timesteps,
            'eval_episodes': args.eval_episodes,
            'base_seed': args.seed,
            'seed_offset': args.seed_offset,
            'seeds': 1,
            'note': 'seeds = independent training runs (pooled from shards)',
        },
        'results': results,
        'summary': {
            name: {
                'win_rate': res['wins'][0],
                'avg_reward': res['rewards'][0],
                'goal_difference': res['goal_differences'][0],
            }
            for name, res in results.items()
        },
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print(f"Total time: {time.time()-total_start:.0f}s")

    # Only our own temp root: parallel shards each own one (see --temp-root).
    cleanup_own_temp(args.temp_root)


if __name__ == "__main__":
    main()
