"""MAPPO Baseline — Multi-Agent PPO with Shared Policy + Centralized Critic.

Key difference from IPPO:
- IPPO: each agent trains independently (or with shared weights, no central info)
- MAPPO: all agents share one policy, centralized critic uses global state

Reference: Yu et al. (2022) "The surprising effectiveness of PPO in cooperative, multi-agent games"
           Song et al. (2024) "An Empirical Study on Google Research Football Multi-agent Scenarios"
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
GAMMA = 1.0             # Song et al. (2024): gamma=1 for 11v11
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MINIBATCH_SIZE = 64
NUM_EPOCHS = 4
TOTAL_TIMESTEPS = 3_000_000
LOG_FREQ = 50
EPISODE_MAX_STEPS = 3000
LEARNING_RATE = 3e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SharedActorCritic(nn.Module):
    """Shared policy for all agents (MAPPO). Centralized critic uses global state."""

    def __init__(self, obs_dim: int = OBS_DIM, state_dim: int = None,
                 action_dim: int = ACTION_SPACE_SIZE, hidden: int = HIDDEN_DIM):
        super().__init__()
        if state_dim is None:
            state_dim = obs_dim * NUM_AGENTS  # concatenation of all agent obs

        # Actor: shared policy (same weights for all agents)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
        )

        # Critic: centralized — sees ALL agents' observations (global state)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def get_action_and_value(self, obs, state=None, action=None):
        """obs: (B, obs_dim), state: (B, state_dim)."""
        logits = self.policy_head(self.actor(obs))
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        value = self.critic(state).squeeze(-1)
        return action, dist.log_prob(action), dist.entropy(), value


class MAPPORolloutBuffer:
    """Rollout buffer with global state for centralized critic."""

    def __init__(self):
        self.obs = []
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def reset(self):
        self.obs.clear(); self.states.clear(); self.actions.clear()
        self.log_probs.clear(); self.rewards.clear(); self.values.clear()
        self.dones.clear()

    def add(self, obs, state, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.states.append(state)
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
                'state': torch.FloatTensor(np.array([self.states[i] for i in idx])).to(DEVICE),
                'actions': torch.LongTensor([self.actions[i] for i in idx]).to(DEVICE),
                'log_probs_old': torch.FloatTensor([self.log_probs[i] for i in idx]).to(DEVICE),
                'advantages': torch.FloatTensor([self.advantages[i] for i in idx]).to(DEVICE),
                'returns': torch.FloatTensor([self.returns[i] for i in idx]).to(DEVICE),
            }


class MAPPOTrainer:
    """MAPPO: shared policy + centralized critic for all 11 agents."""

    def __init__(self, total_timesteps=TOTAL_TIMESTEPS, log_dir="dumps", render=False):
        self.total_timesteps = total_timesteps
        self.log_dir = log_dir
        self.render = render
        self.env = create_raw_env(render=render)
        self.policy = SharedActorCritic().to(DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE, eps=1e-5)
        self.buffer = MAPPORolloutBuffer()
        os.makedirs(log_dir, exist_ok=True)
        self.global_step = 0
        self.episode_count = 0

    def _get_global_state(self, game_state: Dict) -> np.ndarray:
        """Concatenate all agent observations into a global state vector."""
        all_obs = []
        for i in range(NUM_AGENTS):
            all_obs.append(extract_obs_vector(game_state, i, OBS_DIM))
        return np.concatenate(all_obs).astype(np.float32)

    def train(self):
        print(f"MAPPO Training | Device: {DEVICE} | Timesteps: {self.total_timesteps:,}")
        start = time.time()

        while self.global_step < self.total_timesteps:
            reset_result = self.env.reset()
            obs_raw = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            game_state = extract_game_state(obs_raw)
            self.buffer.reset()

            episode_reward = 0.0
            obs_list = [extract_obs_vector(game_state, i, OBS_DIM) for i in range(NUM_AGENTS)]

            with torch.no_grad():
                obs_tensor = torch.FloatTensor(np.array(obs_list)).to(DEVICE)
                global_state = self._get_global_state(game_state)
                state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(DEVICE)
                # Get value from centralized critic for all agents
                all_values = self.policy.critic(state_tensor).squeeze(-1).item()

            for step in range(EPISODE_MAX_STEPS):
                obs_tensor = torch.FloatTensor(np.array(obs_list)).to(DEVICE)
                global_state = self._get_global_state(game_state)
                state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    logits = self.policy.policy_head(self.policy.actor(obs_tensor))
                    dist = torch.distributions.Categorical(logits=logits)
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions)
                    values = self.policy.critic(state_tensor).squeeze(-1).expand(NUM_AGENTS)

                action_list = actions.cpu().numpy().tolist()
                reset_result2 = self.env.step(action_list)
                if isinstance(reset_result2, tuple):
                    obs_raw2, reward, done, info = reset_result2[:4]
                else:
                    obs_raw2 = reset_result2
                    reward, done, info = 0.0, False, {}

                # Reward: average across agents (shared reward)
                if isinstance(reward, (list, np.ndarray)):
                    r = float(np.mean(reward))
                else:
                    r = float(reward)
                episode_reward += r

                # Store per-agent transitions (MAPPO: shared reward, per-agent log_prob)
                for i in range(NUM_AGENTS):
                    self.buffer.add(
                        obs_list[i], global_state,
                        actions[i].item(), log_probs[i].item(),
                        r, values[i].item(), float(done)
                    )
                    self.global_step += 1

                if done or step == EPISODE_MAX_STEPS - 1:
                    break

                game_state = extract_game_state(obs_raw2)
                obs_list = [extract_obs_vector(game_state, i, OBS_DIM) for i in range(NUM_AGENTS)]

            # Compute GAE with last value
            with torch.no_grad():
                last_obs = torch.FloatTensor(np.array(obs_list)).to(DEVICE)
                last_state = torch.FloatTensor(self._get_global_state(game_state)).unsqueeze(0).to(DEVICE)
                last_value = self.policy.critic(last_state).squeeze(-1).item()
            self.buffer.compute_gae(last_value)

            # PPO update
            total_pg_loss, total_v_loss, total_ent = 0.0, 0.0, 0.0
            n_updates = 0
            for batch in self.buffer.get_batches():
                logits, values = self.policy(batch['obs'], batch['state'])
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch['actions'])
                entropy = dist.entropy().mean()

                ratio = (new_log_probs - batch['log_probs_old']).exp()
                surr1 = ratio * batch['advantages']
                surr2 = torch.clamp(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE) * batch['advantages']
                pg_loss = -torch.min(surr1, surr2).mean()
                v_loss = nn.functional.mse_loss(values, batch['returns'])
                loss = pg_loss + VF_COEF * v_loss - ENT_COEF * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_ent += entropy.item()
                n_updates += 1

            self.episode_count += 1
            if self.episode_count % LOG_FREQ == 0:
                elapsed = time.time() - start
                avg_pg = total_pg_loss / max(n_updates, 1)
                avg_v = total_v_loss / max(n_updates, 1)
                avg_e = total_ent / max(n_updates, 1)
                print(f"[MAPPO] Ep {self.episode_count:5d} | "
                      f"Steps {self.global_step:8,d} | "
                      f"Reward {episode_reward:7.3f} | "
                      f"PG {avg_pg:.4f} V {avg_v:.4f} Ent {avg_e:.4f} | "
                      f"Time {elapsed:.0f}s")

        print(f"MAPPO training complete. Total steps: {self.global_step:,}")
        torch.save({
            'policy_state': self.policy.state_dict(),
            'global_step': self.global_step,
            'episode_count': self.episode_count,
        }, os.path.join(self.log_dir, 'mappo_checkpoint.pt'))

    def close(self):
        self.env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MAPPO Baseline")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--log-dir", type=str, default="dumps")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    trainer = MAPPOTrainer(total_timesteps=args.timesteps, log_dir=args.log_dir, render=args.render)
    try:
        trainer.train()
    finally:
        trainer.close()
