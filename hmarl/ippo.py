"""Independent PPO (IPPO) Baseline.

Each of the 11 agents is trained independently with its own PPO policy.
No hierarchical structure - flat policy from observation to action.
Used as comparison baseline per thesis BAB_4.

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

from hmarl.utils import set_seed, extract_obs_vector, quick_eval
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


class FlatActorCritic(nn.Module):
    """Flat Actor-Critic (no hierarchy). MLP with shared layers."""

    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = ACTION_SPACE_SIZE, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
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


class IPPORolloutBuffer:
    """Per-agent rollout buffer for IPPO."""

    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def reset(self):
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, last_value, gamma=GAMMA, lam=GAE_LAMBDA):
        T = len(self.rewards)
        self.advantages = [0.0] * T
        self.returns = [0.0] * T
        gae = 0.0
        for t in reversed(range(T)):
            next_val = self.values[t + 1] if t < T - 1 else last_value
            delta = self.rewards[t] + gamma * next_val * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

    def get_batches(self, bs=MINIBATCH_SIZE):
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


class IPPOTrainer:
    """Independent PPO: trains one shared policy for all agents (flat)."""

    def __init__(self, total_timesteps=TOTAL_TIMESTEPS, log_dir="dumps", render=False):
        self.total_timesteps = total_timesteps
        self.log_dir = log_dir
        self.render = render

        self.env = create_raw_env(render=render)
        self.policy = FlatActorCritic().to(DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE, eps=1e-5)
        self.buffer = IPPORolloutBuffer()

        os.makedirs(log_dir, exist_ok=True)
        self.global_step = 0
        self.episode_count = 0

    def train(self):
        print(f"IPPO Training | Device: {DEVICE} | Timesteps: {self.total_timesteps:,}")
        start = time.time()

        while self.global_step < self.total_timesteps:
            reset_result = self.env.reset()
            obs_raw = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            game_state = extract_game_state(obs_raw)
            self.buffer.reset()

            episode_reward = 0.0
            done = False
            step = 0

            # Collect transitions for all agents
            agent_transitions = {i: [] for i in range(NUM_AGENTS)}

            while not done and step < EPISODE_MAX_STEPS:
                joint_actions = []
                for i in range(NUM_AGENTS):
                    obs_vec = extract_obs_vector(game_state, i)
                    obs_t = torch.FloatTensor(obs_vec).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        action, log_prob, _, value = self.policy.get_action_and_value(obs_t)

                    a = action.item()
                    lp = log_prob.item()
                    v = value.item()
                    joint_actions.append(a)

                    agent_transitions[i].append((obs_vec, a, lp, v))

                step_result = self.env.step(joint_actions)
                if len(step_result) == 5:
                    obs_raw_new, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs_raw_new, reward, done, info = step_result
                team_reward = float(np.sum(reward))
                obs_raw = obs_raw_new
                game_state = extract_game_state(obs_raw)

                for i in range(NUM_AGENTS):
                    obs_vec, a, lp, v = agent_transitions[i][-1]
                    self.buffer.add(
                        obs_vec, a, lp,
                        team_reward / NUM_AGENTS,
                        v, float(done),
                    )

                episode_reward += team_reward
                step += 1
                self.global_step += 1  # count env steps, not per-agent

            # PPO update
            with torch.no_grad():
                last_obs = torch.FloatTensor(extract_obs_vector(game_state, 0)).unsqueeze(0).to(DEVICE)
                _, _, _, last_val = self.policy.get_action_and_value(last_obs)
                last_value = last_val.item()

            self.buffer.compute_gae(last_value)
            self._ppo_update()

            self.episode_count += 1
            if self.episode_count % 500 == 0:
                elapsed = time.time() - start
                print(
                    f"Ep {self.episode_count:6d} | Step {self.global_step:8d}/{self.total_timesteps:,} | "
                    f"Reward: {episode_reward:7.2f} | "
                    f"Steps/s: {self.global_step/max(elapsed,1):.1f}"
                )

            # Mid-training evaluation + save every 500 episodes
            if self.episode_count % 500 == 0:
                self._save()
                eval_stats = self._quick_eval(num_episodes=10)
                print(
                    f"  [EVAL] WR: {eval_stats['win_rate']:.1f}% | "
                    f"Avg Reward: {eval_stats['avg_reward']:.2f} | "
                    f"GD: {eval_stats['goal_diff']}"
                )

        self._save()
        print(f"\nIPPO Training complete. ({time.time()-start:.1f}s)")

    def _ppo_update(self):
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
        path = os.path.join(self.log_dir, "ippo_model.pt")
        torch.save({
            'policy_state': self.policy.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'episode_count': self.episode_count,
        }, path)
        print(f"Model saved: {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        self.policy.load_state_dict(ckpt['policy_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.global_step = ckpt['global_step']
        self.episode_count = ckpt['episode_count']

    def _quick_eval(self, num_episodes: int = 10) -> Dict:
        """Run quick evaluation during training."""
        return quick_eval(self.policy, num_episodes)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train IPPO baseline")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--log-dir", type=str, default="dumps")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint (.pt file)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)
    trainer = IPPOTrainer(
        total_timesteps=args.timesteps,
        log_dir=args.log_dir,
        render=args.render,
    )
    if args.resume:
        trainer.load(args.resume)
        print(f"Resumed from {args.resume} at step {trainer.global_step}")
    trainer.train()
