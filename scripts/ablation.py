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
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from hmarl.env import (
    create_raw_env, NUM_AGENTS, ACTION_SPACE_SIZE,
    extract_game_state, get_action_category,
)
from hmarl.policy import (
    HierarchicalActorCritic, HierarchicalController,
    SubGoalEmbedding, SUBGOAL_EMBED_DIM,
)
from hmarl.expert import ExpertPolicyAllAgents
from hmarl.reward import (
    PassTracker, RCITracker, compute_hierarchical_reward,
    compute_fai, ALPHA_HIGH, ALPHA_MID, ALPHA_LOW,
)
from hmarl.metrics import compute_win_rate, compute_goal_difference, compute_all_metrics, print_metrics
from hmarl.rci import compute_rci

HIDDEN_DIM = 256
HEAD_DIM = 128
OBS_DIM = 115
EPISODE_MAX_STEPS = 3000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual.seed(seed)
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


# ---------------------------------------------------------------------------
# Custom Reward for Ablation
# ---------------------------------------------------------------------------
import hmarl.reward as reward_mod

_orig_compute = reward_mod.compute_hierarchical_reward


def make_custom_reward(alpha_h=ALPHA_HIGH, alpha_m=ALPHA_MID, alpha_l=ALPHA_LOW,
                       use_fai=True, use_ppr=True, use_rci=True):
    """Create a reward function with specified components enabled/disabled."""
    def custom_reward(game_reward, game_state, actual_actions, ideal_actions,
                      pass_tracker, rci_tracker, num_agents=11):
        rho_fa = compute_fai(game_state, num_agents) if use_fai else 0.0
        ppr = pass_tracker.get_ppr() if use_ppr else 0.0
        rci_tracker.update(actual_actions, ideal_actions)
        rci_per_agent = rci_tracker.get_rci_per_agent()
        avg_rci = float(np.mean(rci_per_agent)) if use_rci else 0.0

        r_high = alpha_h * rho_fa
        r_mid = alpha_m * ppr
        r_low = alpha_l * avg_rci
        total = game_reward + r_high + r_mid + r_low

        breakdown = {'game_reward': game_reward, 'fai': rho_fa, 'ppr': ppr,
                     'rci_avg': avg_rci, 'r_high': r_high, 'r_mid': r_mid,
                     'r_low': r_low, 'total': total}
        return total, breakdown
    return custom_reward


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_hmarl_config(
    config_name: str,
    timesteps: int,
    seed: int,
    reward_fn=None,
    flat_policy: bool = False,
    random_expert: bool = False,
) -> Dict:
    """Train HMARL with a specific ablation configuration."""
    set_seed(seed)

    env = create_raw_env(render=False)
    controller = HierarchicalController()
    expert = ExpertPolicyAllAgents()

    if reward_fn is None:
        reward_fn = _orig_compute

    subgoal_embedding = SubGoalEmbedding().to(DEVICE)

    if flat_policy:
        # Flat policy: no sub-goal conditioning, just obs -> action
        policy = HierarchicalActorCritic(
            obs_dim=OBS_DIM, subgoal_embed_dim=SUBGOAL_EMBED_DIM,
            hidden_dim=HIDDEN_DIM, head_dim=HEAD_DIM, action_dim=ACTION_SPACE_SIZE,
        ).to(DEVICE)
    else:
        policy = HierarchicalActorCritic(
            obs_dim=OBS_DIM, subgoal_embed_dim=SUBGOAL_EMBED_DIM,
            hidden_dim=HIDDEN_DIM, head_dim=HEAD_DIM, action_dim=ACTION_SPACE_SIZE,
        ).to(DEVICE)

    params = list(policy.parameters()) + list(subgoal_embedding.parameters())
    optimizer = optim.Adam(params, lr=3e-4, eps=1e-5)

    # Rollout buffer
    observations, subgoal_embeds, actions_list = [], [], []
    log_probs, rewards_buf, values, dones = [], [], [], []

    global_step = 0
    episode_count = 0
    global_step_limit = timesteps

    print(f"  Training {config_name} | Device: {DEVICE} | Steps: {timesteps:,}")
    start_time = time.time()

    while global_step < global_step_limit:
        reset_result = env.reset()
        obs_raw = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        game_state = extract_game_state(obs_raw)
        pass_tracker = PassTracker()
        rci_tracker = RCITracker(NUM_AGENTS)

        ep_reward = 0.0
        observations.clear()
        subgoal_embeds.clear()
        actions_list.clear()
        log_probs.clear()
        rewards_buf.clear()
        values.clear()
        dones.clear()

        prev_game_state = None
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
            step_log_probs = []
            step_values = []

            for i in range(NUM_AGENTS):
                obs_vec = extract_obs_vector(game_state, i)
                if flat_policy:
                    sg_embed = np.zeros(SUBGOAL_EMBED_DIM, dtype=np.float32)
                else:
                    with torch.no_grad():
                        sg_embed = subgoal_embedding(
                            torch.LongTensor([sub_goals[i]]).to(DEVICE)
                        ).cpu().numpy().flatten()

                obs_t = torch.FloatTensor(obs_vec).unsqueeze(0).to(DEVICE)
                sg_t = torch.FloatTensor(sg_embed).unsqueeze(0).to(DEVICE)
                action, log_prob, _, value = policy.get_action_and_value(obs_t, sg_t)
                a = action.item()
                joint_actions.append(a)
                step_log_probs.append(log_prob.item())
                step_values.append(value.item())
                observations.append(obs_vec)
                subgoal_embeds.append(sg_embed)
                actions_list.append(a)
                log_probs.append(log_prob.item())
                rewards_buf.append(0.0)
                values.append(value.item())
                dones.append(0.0)

            step_result = env.step(joint_actions)
            if len(step_result) == 5:
                obs_raw, game_reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs_raw, game_reward, done, info = step_result
            team_reward = float(np.sum(game_reward))
            new_game_state = extract_game_state(obs_raw)

            pass_tracker.update(new_game_state, prev_game_state)
            total_reward, _ = reward_fn(
                team_reward, new_game_state, joint_actions, ideal_actions,
                pass_tracker, rci_tracker,
            )

            # Fill rewards for all agents in this timestep
            agent_start = len(rewards_buf) - NUM_AGENTS
            for j in range(NUM_AGENTS):
                idx = agent_start + j
                if idx < len(rewards_buf):
                    rewards_buf[idx] = total_reward / NUM_AGENTS
                    if done:
                        dones[idx] = 1.0

            ep_reward += total_reward
            prev_game_state = game_state
            game_state = new_game_state
            step += 1
            global_step += 1

        # PPO update
        if len(observations) > 0:
            with torch.no_grad():
                last_obs = torch.FloatTensor(observations[-1]).unsqueeze(0).to(DEVICE)
                last_sg = torch.FloatTensor(subgoal_embeds[-1]).unsqueeze(0).to(DEVICE)
                last_value = policy.get_value(last_obs, last_sg).item()

            # GAE
            T = len(rewards_buf)
            advantages = [0.0] * T
            returns_arr = [0.0] * T
            gae = 0.0
            for t in reversed(range(T)):
                next_val = values[t + 1] if t < T - 1 else last_value
                delta = rewards_buf[t] + 0.99 * next_val * (1 - dones[t]) - values[t]
                gae = delta + 0.99 * 0.95 * (1 - dones[t]) * gae
                advantages[t] = gae
                returns_arr[t] = gae + values[t]

            # Mini-batch updates
            indices = np.random.permutation(T)
            for start in range(0, T, 64):
                end = min(start + 64, T)
                idx = indices[start:end]

                obs_b = torch.FloatTensor(np.array([observations[i] for i in idx])).to(DEVICE)
                sg_b = torch.FloatTensor(np.array([subgoal_embeds[i] for i in idx])).to(DEVICE)
                act_b = torch.LongTensor([actions_list[i] for i in idx]).to(DEVICE)
                old_lp = torch.FloatTensor([log_probs[i] for i in idx]).to(DEVICE)
                adv_b = torch.FloatTensor([advantages[i] for i in idx]).to(DEVICE)
                ret_b = torch.FloatTensor([returns_arr[i] for i in idx]).to(DEVICE)

                adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

                _, new_lp, entropy, new_val = policy.get_action_and_value(obs_b, sg_b, act_b)
                ratio = torch.exp(new_lp - old_lp)
                s1 = ratio * adv_b
                s2 = torch.clamp(ratio, 0.8, 1.2) * adv_b
                pg_loss = -torch.min(s1, s2).mean()
                v_loss = nn.MSELoss()(new_val, ret_b)
                loss = pg_loss + 0.5 * v_loss - 0.01 * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

        episode_count += 1
        if episode_count % 200 == 0:
            elapsed = time.time() - start_time
            print(f"    Ep {episode_count:5d} | Step {global_step:7d}/{timesteps:,} | "
                  f"Reward: {ep_reward:7.2f} | {global_step/max(elapsed,1):.0f} steps/s")

    env.close()

    # Save checkpoint
    ckpt_dir = f"ablation_temp/{config_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "model.pt")
    torch.save({
        'policy_state': policy.state_dict(),
        'subgoal_embedding_state': subgoal_embedding.state_dict(),
    }, ckpt_path)

    print(f"  {config_name} done. {episode_count} episodes, {global_step} steps. "
          f"({time.time()-start_time:.0f}s)")

    return {'checkpoint': ckpt_path, 'episodes': episode_count, 'steps': global_step}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_ablation_config(ckpt_path: str, num_episodes: int, seed: int,
                             flat_policy: bool = False, random_expert: bool = False) -> Dict:
    """Evaluate an ablation config."""
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
    all_goals_for = []
    all_goals_against = []

    for _ in range(num_episodes):
        reset_result = env.reset()
        obs_raw = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        game_state = extract_game_state(obs_raw)
        ep_reward = 0.0
        done = False
        step = 0

        while not done and step < EPISODE_MAX_STEPS:
            macro = controller.get_macro_strategy(game_state)
            sub_goals = controller.get_sub_goals(game_state, macro)

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
                obs_raw, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs_raw, reward, done, info = step_result
            ep_reward += float(np.sum(reward))
            game_state = extract_game_state(obs_raw)
            step += 1

        score = info.get('score', [0, 0])
        gf, ga = (score[0], score[1]) if isinstance(score, (list, tuple)) and len(score) >= 2 else (0, 0)
        match_results.append('win' if gf > ga else ('loss' if gf < ga else 'draw'))
        all_rewards.append(ep_reward)
        all_goals_for.append(gf)
        all_goals_against.append(ga)

    env.close()

    return {
        'win_rate': compute_win_rate(match_results),
        'avg_reward': float(np.mean(all_rewards)),
        'goal_difference': compute_goal_difference(all_goals_for, all_goals_against),
    }


# ---------------------------------------------------------------------------
# Ablation Configurations
# ---------------------------------------------------------------------------
ABLATION_CONFIGS = {
    'full_hmarl': {
        'description': 'Full HMARL (baseline)',
        'reward_fn': make_custom_reward(use_fai=True, use_ppr=True, use_rci=True),
        'flat_policy': False,
        'random_expert': False,
    },
    'no_fai': {
        'description': 'Without FAI reward (α_H = 0)',
        'reward_fn': make_custom_reward(use_fai=False, use_ppr=True, use_rci=True),
        'flat_policy': False,
        'random_expert': False,
    },
    'no_ppr': {
        'description': 'Without PPR reward (α_M = 0)',
        'reward_fn': make_custom_reward(use_fai=True, use_ppr=False, use_rci=True),
        'flat_policy': False,
        'random_expert': False,
    },
    'no_rci': {
        'description': 'Without RCI reward (α_L = 0)',
        'reward_fn': make_custom_reward(use_fai=True, use_ppr=True, use_rci=False),
        'flat_policy': False,
        'random_expert': False,
    },
    'no_reward_shaping': {
        'description': 'No reward shaping (game reward only)',
        'reward_fn': make_custom_reward(use_fai=False, use_ppr=False, use_rci=False),
        'flat_policy': False,
        'random_expert': False,
    },
    'flat_policy': {
        'description': 'Flat policy (no sub-goal conditioning)',
        'reward_fn': make_custom_reward(use_fai=True, use_ppr=True, use_rci=True),
        'flat_policy': True,
        'random_expert': False,
    },
    'random_expert': {
        'description': 'Random expert (random ideal actions)',
        'reward_fn': make_custom_reward(use_fai=True, use_ppr=True, use_rci=True),
        'flat_policy': False,
        'random_expert': True,
    },
}


def main():
    parser = argparse.ArgumentParser(description="HMARL Ablation Study")
    parser.add_argument("--timesteps", type=int, default=300_000, help="Training timesteps per config")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Evaluation episodes per config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Specific configs to run (default: all)")
    parser.add_argument("--output", type=str, default="evaluation_results/ablation_results.json")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 100k steps, 20 eval episodes")
    args = parser.parse_args()

    if args.quick:
        args.timesteps = 100_000
        args.eval_episodes = 20

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs("ablation_temp", exist_ok=True)

    configs_to_run = args.configs if args.configs else list(ABLATION_CONFIGS.keys())
    results = {}

    print(f"Ablation Study | Configs: {len(configs_to_run)} | "
          f"Steps: {args.timesteps:,} | Eval: {args.eval_episodes} episodes")
    print(f"{'='*60}")

    total_start = time.time()

    for config_name in configs_to_run:
        if config_name not in ABLATION_CONFIGS:
            print(f"  WARNING: Unknown config '{config_name}', skipping.")
            continue

        config = ABLATION_CONFIGS[config_name]
        print(f"\n--- {config_name}: {config['description']} ---")

        train_result = train_hmarl_config(
            config_name=config_name,
            timesteps=args.timesteps,
            seed=args.seed,
            reward_fn=config['reward_fn'],
            flat_policy=config['flat_policy'],
            random_expert=config['random_expert'],
        )

        eval_result = evaluate_ablation_config(
            ckpt_path=train_result['checkpoint'],
            num_episodes=args.eval_episodes,
            seed=args.seed + 1000,
            flat_policy=config['flat_policy'],
            random_expert=config['random_expert'],
        )

        results[config_name] = {
            'description': config['description'],
            'train': train_result,
            'eval': eval_result,
        }

        print(f"  Result: WR={eval_result['win_rate']:.1f}% | "
              f"Reward={eval_result['avg_reward']:.2f} | "
              f"GD={eval_result['goal_difference']}")

    # Summary table
    print(f"\n{'='*72}")
    print(f"  ABLATION SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Config':<25} {'Win Rate':>10} {'Avg Reward':>12} {'Goal Diff':>10}")
    print(f"  {'-'*57}")

    full_hmarl_wr = results.get('full_hmarl', {}).get('eval', {}).get('win_rate', 0)

    for name, res in results.items():
        ev = res.get('eval', {})
        wr = ev.get('win_rate', 0)
        rw = ev.get('avg_reward', 0)
        gd = ev.get('goal_difference', 0)
        delta = wr - full_hmarl_wr if name != 'full_hmarl' else 0
        delta_str = f" ({delta:+.1f}%)" if name != 'full_hmarl' else ""
        print(f"  {name:<25} {wr:>9.1f}% {rw:>12.2f} {gd:>10}{delta_str}")

    print(f"{'='*72}")

    # Save
    output = {
        'config': {
            'timesteps': args.timesteps,
            'eval_episodes': args.eval_episodes,
            'seed': args.seed,
        },
        'results': results,
        'summary': {
            name: {
                'win_rate': res['eval']['win_rate'],
                'avg_reward': res['eval']['avg_reward'],
                'goal_difference': res['eval']['goal_difference'],
            }
            for name, res in results.items()
        },
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print(f"Total time: {time.time()-total_start:.0f}s")


if __name__ == "__main__":
    main()
