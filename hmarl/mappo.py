"""Multi-Agent PPO (MAPPO) Baseline.

Decentralized actors + centralized critic.

Architecture:
  Actor (shared across agents): 115-dim local obs -> action
  Critic (centralized):         1265-dim (11 × 115) concatenated obs -> value

Key difference from IPPO/SHPPO:
  IPPO/SHPPO: critic sees only one agent's local observation
  MAPPO:      critic sees ALL agents' observations (global state)

Reference: Yu et al. (2022) - The Surprising Effectiveness of PPO in
Cooperative Multi-Agent Games

Note: Baselines use game reward only (no FAI/PPR/RCI reward shaping).
This ensures a fair comparison — HMARL's improvement over baselines
is attributed to the hierarchical architecture, not reward shaping.
"""

import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from hmarl.utils import set_seed, extract_obs_vector, build_centralized_obs, quick_eval
from hmarl.env import create_raw_env, NUM_AGENTS, extract_game_state

OBS_DIM = 115
HIDDEN_DIM = 256
ACTION_SPACE_SIZE = 19
CENTRALIZED_OBS_DIM = OBS_DIM * NUM_AGENTS  # 115 * 11 = 1265
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MINIBATCH_SIZE = 64
NUM_EPOCHS = 4
TOTAL_TIMESTEPS = 3_000_000
LOG_FREQ = 50              # Print progress every N episodes
EPISODE_MAX_STEPS = 3000
LEARNING_RATE = 3e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DecentralizedActor(nn.Module):
    """Per-agent actor network (shared weights across agents).

    Input: 115-dim local observation
    Output: action logits (19 actions)
    """

    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = ACTION_SPACE_SIZE,
                 hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)

    def get_action_and_value(self, obs, action=None):
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()


class CentralizedCritic(nn.Module):
    """Centralized critic — sees concatenated observations of ALL agents.

    Input: 1265-dim = concat(obs_0, obs_1, ..., obs_10)
    Output: state value V(s)

    Uses 2 hidden layers (same depth as actor and HMARL's low-level policy)
    for fair parameter count comparison.
    """
    def __init__(self, obs_dim: int = CENTRALIZED_OBS_DIM, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, centralized_obs):
        return self.net(centralized_obs).squeeze(-1)


class MAPPORolloutBuffer:
    """Rollout buffer storing both local obs (for actor) and centralized obs
    (for critic).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.obs = []              # per-agent local obs
        self.centralized_obs = []  # full team obs at each agent's timestep
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.advantages = []
        self.returns = []

    def add(self, obs, centralized_obs, action, log_prob, value, reward=0.0, done=0.0):
        self.obs.append(obs)
        self.centralized_obs.append(centralized_obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, last_value, gamma=GAMMA, lam=GAE_LAMBDA):
        """Compute GAE using centralized values."""
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
        """Yield random mini-batches."""
        T = len(self.obs)
        indices = np.random.permutation(T)
        for start in range(0, T, bs):
            end = min(start + bs, T)
            idx = indices[start:end]
            yield {
                'obs': torch.FloatTensor(
                    np.array([self.obs[i] for i in idx])
                ).to(DEVICE),
                'centralized_obs': torch.FloatTensor(
                    np.array([self.centralized_obs[i] for i in idx])
                ).to(DEVICE),
                'actions': torch.LongTensor([self.actions[i] for i in idx]).to(DEVICE),
                'log_probs_old': torch.FloatTensor(
                    [self.log_probs[i] for i in idx]
                ).to(DEVICE),
                'advantages': torch.FloatTensor(
                    [self.advantages[i] for i in idx]
                ).to(DEVICE),
                'returns': torch.FloatTensor(
                    [self.returns[i] for i in idx]
                ).to(DEVICE),
            }



class MAPPOTrainer:
    """MAPPO: decentralized actors + centralized critic.

    Training loop:
    1. Collect per-agent transitions with centralized value estimates
    2. Compute GAE using centralized critic's values
    3. Joint update: actor loss (per-agent local obs) + critic loss (centralized obs)
    """

    def __init__(self, total_timesteps=TOTAL_TIMESTEPS, log_dir="dumps", render=False):
        self.total_timesteps = total_timesteps
        self.log_dir = log_dir
        self.render = render

        self.env = create_raw_env(render=render)

        # Decentralized actor (shared weights, per-agent obs)
        self.actor = DecentralizedActor().to(DEVICE)
        # Centralized critic (sees all agents' obs concatenated)
        self.critic = CentralizedCritic().to(DEVICE)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE, eps=1e-5)

        self.buffer = MAPPORolloutBuffer()

        os.makedirs(log_dir, exist_ok=True)
        self.global_step = 0
        self.episode_count = 0

    def train(self):
        print(f"MAPPO Training | Device: {DEVICE} | Timesteps: {self.total_timesteps:,}")
        print(f"  Actor: 115-dim obs -> action | Critic: {CENTRALIZED_OBS_DIM}-dim centralized obs -> value")
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
                # Build all agents' local observations
                all_obs = [extract_obs_vector(game_state, i) for i in range(NUM_AGENTS)]
                centralized_obs = build_centralized_obs(all_obs)

                # Get centralized value estimate
                central_obs_t = torch.FloatTensor(centralized_obs).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    central_value = self.critic(central_obs_t).item()

                # Collect per-agent actions using decentralized actor
                joint_actions = []
                for i in range(NUM_AGENTS):
                    obs_t = torch.FloatTensor(all_obs[i]).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        action, log_prob, _ = self.actor.get_action_and_value(obs_t)

                    a = action.item()
                    lp = log_prob.item()
                    joint_actions.append(a)

                    # Store: local obs (for actor), centralized obs (for critic)
                    self.buffer.add(
                        obs=all_obs[i],
                        centralized_obs=centralized_obs,
                        action=a,
                        log_prob=lp,
                        value=central_value,  # all agents share same centralized value
                    )

                # Step environment
                step_result = self.env.step(joint_actions)
                if len(step_result) == 5:
                    obs_raw_new, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs_raw_new, reward, done, info = step_result
                team_reward = float(np.sum(reward))

                # Fill rewards for all agents in this timestep
                for i in range(NUM_AGENTS):
                    idx = len(self.buffer.rewards) - NUM_AGENTS + i
                    self.buffer.rewards[idx] = team_reward
                    self.buffer.dones[idx] = float(done)

                obs_raw = obs_raw_new
                game_state = extract_game_state(obs_raw)
                episode_reward += team_reward
                step += 1
                self.global_step += 1  # count env steps, not per-agent

            # Compute last centralized value for GAE
            with torch.no_grad():
                last_central = torch.FloatTensor(
                    build_centralized_obs(
                        [extract_obs_vector(game_state, i) for i in range(NUM_AGENTS)]
                    )
                ).unsqueeze(0).to(DEVICE)
                last_value = self.critic(last_central).item()

            self.buffer.compute_gae(last_value)
            self._update()

            self.episode_count += 1
            if self.episode_count % LOG_FREQ == 0:
                elapsed = time.time() - start
                steps_per_sec = self.global_step / max(elapsed, 1)
                pct = 100.0 * self.global_step / self.total_timesteps
                remaining = (self.total_timesteps - self.global_step) / max(steps_per_sec, 1)
                hrs, rem = divmod(int(remaining), 3600)
                mins, secs = divmod(rem, 60)
                print(
                    f"[{pct:5.1f}%] Ep {self.episode_count:6d} | "
                    f"Step {self.global_step:8d}/{self.total_timesteps:,} | "
                    f"Rwd {episode_reward:7.2f} | "
                    f"{steps_per_sec:.1f} steps/s | ETA {hrs:02d}:{mins:02d}:{secs:02d}"
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
        print(f"\nMAPPO Training complete. ({time.time()-start:.1f}s)")

    def _update(self):
        """Joint actor + critic PPO update."""
        self.actor.train()
        self.critic.train()

        for _ in range(NUM_EPOCHS):
            for batch in self.buffer.get_batches():
                obs = batch['obs']
                central_obs = batch['centralized_obs']
                actions = batch['actions']
                old_lp = batch['log_probs_old']
                adv = batch['advantages']
                returns = batch['returns']

                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Actor update (decentralized — uses local obs)
                _, new_lp, entropy = self.actor.get_action_and_value(obs, actions)
                ratio = torch.exp(new_lp - old_lp)
                s1 = ratio * adv
                s2 = torch.clamp(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE) * adv
                actor_loss = -torch.min(s1, s2).mean() - ENT_COEF * entropy.mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                # Critic update (centralized — uses full team obs)
                values = self.critic(central_obs)
                critic_loss = nn.MSELoss()(values, returns)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

        self.buffer.reset()

    def _save(self):
        path = os.path.join(self.log_dir, "mappo_model.pt")
        torch.save({
            'actor_state': self.actor.state_dict(),
            'critic_state': self.critic.state_dict(),
            'actor_optimizer_state': self.actor_optimizer.state_dict(),
            'critic_optimizer_state': self.critic_optimizer.state_dict(),
            'global_step': self.global_step,
            'episode_count': self.episode_count,
        }, path)
        print(f"Model saved: {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        self.actor.load_state_dict(ckpt['actor_state'])
        self.critic.load_state_dict(ckpt['critic_state'])
        self.actor_optimizer.load_state_dict(ckpt['actor_optimizer_state'])
        self.critic_optimizer.load_state_dict(ckpt['critic_optimizer_state'])
        self.global_step = ckpt['global_step']
        self.episode_count = ckpt['episode_count']

    def _quick_eval(self, num_episodes: int = 10) -> Dict:
        """Run quick evaluation during training."""
        return quick_eval(self.actor, num_episodes)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train MAPPO baseline")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--log-dir", type=str, default="dumps")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint (.pt file)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)
    trainer = MAPPOTrainer(
        total_timesteps=args.timesteps,
        log_dir=args.log_dir,
        render=args.render,
    )
    if args.resume:
        trainer.load(args.resume)
        print(f"Resumed from {args.resume} at step {trainer.global_step}")
    trainer.train()
