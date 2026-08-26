"""Shared PPO (SHPPO) Baseline.

All 11 agents share a single actor-critic network. Unlike IPPO which tracks
per-agent transitions independently, SHPPO pools ALL agents' experiences
into one unified buffer, computes a single GAE over the pooled data, and
performs one centralized PPO update.

Key difference from IPPO:
  IPPO:  per-agent transition tracking, reward split evenly to each agent
  SHPPO: all agent transitions pooled into one buffer, team-level GAE

Note: Baselines use game reward only (no FAI/PPR/RCI reward shaping).
This ensures a fair comparison — HMARL's improvement over baselines
is attributed to the hierarchical architecture, not reward shaping.

Reference: Song et al. (2024) - An Empirical Study on Google Research Football
"""

import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from hmarl.env import create_raw_env, NUM_AGENTS, extract_game_state

OBS_DIM = 115
HIDDEN_DIM = 256
ACTION_SPACE_SIZE = 19
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MINIBATCH_SIZE = 64
NUM_EPOCHS = 4
TOTAL_TIMESTEPS = 3_000_000
EPISODE_MAX_STEPS = 3000
LEARNING_RATE = 3e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SharedActorCritic(nn.Module):
    """Shared Actor-Critic (same as IPPO's FlatActorCritic).

    All agents use the same network. Difference from IPPO is in training:
    SHPPO pools all agents' data into one buffer for unified update.
    """

    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = ACTION_SPACE_SIZE,
                 hidden: int = HIDDEN_DIM):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs):
        x = self.shared(obs)
        return self.policy_head(x), self.value_head(x)

    def get_action_and_value(self, obs, action=None):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)


class SHPPORolloutBuffer:
    """Unified rollout buffer — pools ALL agents' transitions.

    Unlike IPPO which tracks per-agent indices separately, SHPPO treats
    all (agent, timestep) pairs as independent samples in one flat buffer.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.advantages = []
        self.returns = []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, last_value, gamma=GAMMA, lam=GAE_LAMBDA):
        """Compute GAE over the pooled buffer (all agents)."""
        T = len(self.rewards)
        self.advantages = [0.0] * T
        self.returns = [0.0] * T
        gae = 0.0
        for t in reversed(range(T)):
            next_val = self.values[t + 1] if t < T - 1 else last_value
            delta = (self.rewards[t] + gamma * next_val * (1 - self.dones[t])
                     - self.values[t])
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

    def get_batches(self, bs=MINIBATCH_SIZE):
        """Yield random mini-batches from pooled data."""
        T = len(self.obs)
        indices = np.random.permutation(T)
        for start in range(0, T, bs):
            end = min(start + bs, T)
            idx = indices[start:end]
            yield {
                'obs': torch.FloatTensor(np.array([self.obs[i] for i in idx])).to(DEVICE),
                'actions': torch.LongTensor([self.actions[i] for i in idx]).to(DEVICE),
                'log_probs_old': torch.FloatTensor([self.log_probs[i] for i in idx]).to(DEVICE),
                'advantages': torch.FloatTensor([self.advantages[i] for i in idx]).to(DEVICE),
                'returns': torch.FloatTensor([self.returns[i] for i in idx]).to(DEVICE),
            }


def extract_obs_vector(game_state, player_idx):
    """Extract 115-dim observation vector for a player from raw dict."""
    features = []
    ball = game_state.get('ball', [0, 0, 0])
    features.extend(ball)
    features.extend(game_state.get('ball_direction', [0, 0, 0]))
    features.extend(game_state.get('ball_rotation', [0, 0, 0]))
    features.append(float(game_state.get('ball_owned_team', -1)))
    features.append(float(game_state.get('ball_owned_player', -1)))

    for i in range(11):
        pos = game_state.get('left_team', [[0, 0]] * 11)[i]
        direction = (game_state.get('left_team_direction', [[0, 0]] * 11)[i]
                     if 'left_team_direction' in game_state else [0, 0])
        tired = (game_state.get('left_team_tired_factor', [0.0] * 11)[i]
                 if 'left_team_tired_factor' in game_state else 0.0)
        yellow = (game_state.get('left_team_yellow_card', [0] * 11)[i]
                  if 'left_team_yellow_card' in game_state else 0)
        role = (game_state.get('left_team_roles', [5] * 11)[i]
                if 'left_team_roles' in game_state else 5)
        features.append(1.0 if i == player_idx else 0.0)
        features.extend(pos)
        features.extend(direction)
        features.append(tired)
        features.append(float(yellow))
        features.append(float(role))

    features = features[:OBS_DIM]
    while len(features) < OBS_DIM:
        features.append(0.0)
    return np.array(features, dtype=np.float32)


class SHPPOTrainer:
    """Shared PPO: one shared policy, pooled experience update.

    Training loop:
    1. Collect transitions from ALL agents into one unified buffer
    2. Compute GAE over the entire pooled buffer
    3. PPO update on pooled data (all agents contribute to one gradient)
    """

    def __init__(self, total_timesteps=TOTAL_TIMESTEPS, log_dir="dumps", render=False):
        self.total_timesteps = total_timesteps
        self.log_dir = log_dir
        self.render = render

        self.env = create_raw_env(render=render)
        self.policy = SharedActorCritic().to(DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE, eps=1e-5)
        self.buffer = SHPPORolloutBuffer()

        os.makedirs(log_dir, exist_ok=True)
        self.global_step = 0
        self.episode_count = 0

    def train(self):
        print(f"SHPPO Training | Device: {DEVICE} | Timesteps: {self.total_timesteps:,}")
        start = time.time()

        while self.global_step < self.total_timesteps:
            reset_result = self.env.reset()
            obs_raw = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            game_state = extract_game_state(obs_raw)
            self.buffer.reset()

            episode_reward = 0.0
            done = False
            step = 0

            while not done and step < EPISODE_MAX_STEPS:
                # Collect ALL agents' transitions into one buffer per timestep
                joint_actions = []
                last_values = []

                for i in range(NUM_AGENTS):
                    obs_vec = extract_obs_vector(game_state, i)
                    obs_t = torch.FloatTensor(obs_vec).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        action, log_prob, _, value = self.policy.get_action_and_value(obs_t)

                    a = action.item()
                    lp = log_prob.item()
                    v = value.item()
                    joint_actions.append(a)
                    last_values.append(v)

                    # Store in unified buffer (will get team reward later)
                    self.buffer.add(obs_vec, a, lp, 0.0, v, 0.0)

                # Step environment
                step_result = self.env.step(joint_actions)
                if len(step_result) == 5:
                    obs_raw_new, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs_raw_new, reward, done, info = step_result
                team_reward = float(np.sum(reward))

                # Fill rewards: team_reward for ALL agents in this timestep
                # (key SHPPO behavior — unified reward signal, not split)
                for i in range(NUM_AGENTS):
                    idx = len(self.buffer.rewards) - NUM_AGENTS + i
                    self.buffer.rewards[idx] = team_reward
                    self.buffer.dones[idx] = float(done)

                obs_raw = obs_raw_new
                game_state = extract_game_state(obs_raw)
                episode_reward += team_reward
                step += 1
                self.global_step += NUM_AGENTS

            # PPO update over pooled data
            with torch.no_grad():
                last_obs = torch.FloatTensor(
                    extract_obs_vector(game_state, 0)
                ).unsqueeze(0).to(DEVICE)
                _, _, _, last_val = self.policy.get_action_and_value(last_obs)
                last_value = last_val.item()

            self.buffer.compute_gae(last_value)
            self._ppo_update()

            self.episode_count += 1
            if self.episode_count % 500 == 0:
                elapsed = time.time() - start
                print(
                    f"Ep {self.episode_count:6d} | Step {self.global_step:8d} | "
                    f"Reward: {episode_reward:7.2f} | "
                    f"Steps/s: {self.global_step/max(elapsed,1):.1f}"
                )
                # Mid-training evaluation + save
                self._save()
                eval_stats = self._quick_eval(num_episodes=10)
                print(
                    f"  [EVAL] WR: {eval_stats['win_rate']:.1f}% | "
                    f"Avg Reward: {eval_stats['avg_reward']:.2f} | "
                    f"GD: {eval_stats['goal_diff']}"
                )

        self._save()
        print(f"\nSHPPO Training complete. ({time.time()-start:.1f}s)")

    def _ppo_update(self):
        """PPO update over the pooled buffer (all agents' data)."""
        self.policy.train()
        for _ in range(NUM_EPOCHS):
            for batch in self.buffer.get_batches():
                obs = batch['obs']
                actions = batch['actions']
                old_lp = batch['log_probs_old']
                adv = batch['advantages']
                returns = batch['returns']

                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                _, new_lp, entropy, values = self.policy.get_action_and_value(obs, actions)

                ratio = torch.exp(new_lp - old_lp)
                s1 = ratio * adv
                s2 = torch.clamp(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE) * adv
                pg_loss = -torch.min(s1, s2).mean()
                v_loss = nn.MSELoss()(values, returns)
                loss = pg_loss + VF_COEF * v_loss - ENT_COEF * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

        self.buffer.reset()

    def _save(self):
        path = os.path.join(self.log_dir, "shppo_model.pt")
        torch.save({
            'policy_state': self.policy.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'episode_count': self.episode_count,
        }, path)
        print(f"Model saved: {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=DEVICE)
        self.policy.load_state_dict(ckpt['policy_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.global_step = ckpt['global_step']
        self.episode_count = ckpt['episode_count']

    def _quick_eval(self, num_episodes: int = 10) -> Dict:
        """Run quick evaluation during training."""
        self.policy.eval()
        eval_env = create_raw_env(render=False)
        wins, total_goals_for, total_goals_against, total_reward = 0, 0, 0, 0.0

        for _ in range(num_episodes):
            reset_result = eval_env.reset()
            obs_raw = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            game_state = extract_game_state(obs_raw)
            ep_reward = 0.0
            done = False
            step = 0

            while not done and step < EPISODE_MAX_STEPS:
                joint_actions = []
                for i in range(NUM_AGENTS):
                    obs_vec = extract_obs_vector(game_state, i)
                    obs_t = torch.FloatTensor(obs_vec).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        logits, _ = self.policy(obs_t)
                        joint_actions.append(logits.argmax(dim=-1).item())

                step_result = eval_env.step(joint_actions)
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
            if gf > ga: wins += 1
            total_goals_for += gf
            total_goals_against += ga
            total_reward += ep_reward

        eval_env.close()
        self.policy.train()
        return {
            'win_rate': wins / num_episodes * 100.0,
            'avg_reward': total_reward / num_episodes,
            'goal_diff': total_goals_for - total_goals_against,
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train SHPPO baseline")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--log-dir", type=str, default="dumps")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint (.pt file)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)
    trainer = SHPPOTrainer(
        total_timesteps=args.timesteps,
        log_dir=args.log_dir,
        render=args.render,
    )
    if args.resume:
        trainer.load(args.resume)
        print(f"Resumed from {args.resume} at step {trainer.global_step}")
    trainer.train()
