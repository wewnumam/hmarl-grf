"""Simplified PPO trainer for GRF academy scenarios.

Purpose: test whether PPO agents can score goals in easy scenarios
(academy_single_goal_versus_lazy, academy_empty_goal, etc.) WITHOUT
the full HMARL hierarchy (no FAI/RCI/PPR shaping, no expert, no
hierarchical controller). Pure game reward + optional goal bonus.

Usage:
    python scripts/train_academy.py --scenario academy_single_goal_versus_lazy \
        --timesteps 100000 --log-dir dumps_academy

This answers: "can our PPO network learn to score at all?"
If yes → HMARL reward shaping is the bottleneck, not the network.
If no  → PPO hyperparams or obs encoding is broken.
"""
import argparse
import json
import os
import signal
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hmarl.env import create_raw_env, extract_game_state, ACTION_SPACE_SIZE
from hmarl.run_logging import (
    run_metadata, save_run_log, hyperparams_snapshot, obs_snapshot, OBS_HMARL_115,
)
from hmarl.utils import (
    extract_obs_vector as _extract_obs_util,
    set_seed as _set_seed_util,
    OBS_DIM,
)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MINIBATCH_SIZE = 64
NUM_EPOCHS = 4
GOAL_BONUS = 10.0   # extra reward per goal scored

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Academy scenarios: short episodes (~100-200 steps)
SCENARIO_INFO = {
    "academy_single_goal_versus_lazy": {"agents": 4, "max_steps": 500},
    "academy_empty_goal": {"agents": 4, "max_steps": 500},
    "academy_empty_goal_close": {"agents": 1, "max_steps": 500},
    "academy_run_to_score": {"agents": 1, "max_steps": 500},
    "academy_run_to_score_with_keeper": {"agents": 1, "max_steps": 500},
    "academy_pass_and_shoot_with_keeper": {"agents": 2, "max_steps": 500},
    "academy_3_vs_1_with_keeper": {"agents": 3, "max_steps": 500},
}


# ---------------------------------------------------------------------------
# Simple Actor-Critic (no sub-goal embedding)
# ---------------------------------------------------------------------------
class SimpleActorCritic(nn.Module):
    """Minimal actor-critic for academy scenarios."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

    def get_action_and_value(self, x):
        logits, value = self(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)

    def get_value(self, x):
        h = self.shared(x)
        return self.critic(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout Buffer
# ---------------------------------------------------------------------------
class RolloutBuffer:
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

    def add(self, obs, action, log_prob, value, reward, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_gae(self, last_value, gamma=GAMMA, lam=GAE_LAMBDA):
        T = len(self.rewards)
        self.advantages = [0.0] * T
        self.returns = [0.0] * T
        gae = 0.0
        for t in reversed(range(T)):
            next_val = last_value if t == T - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_val * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

    def get_batches(self, batch_size=MINIBATCH_SIZE):
        T = len(self.obs)
        indices = np.random.permutation(T)
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            bi = indices[start:end]
            yield {
                'obs': torch.FloatTensor(np.array([self.obs[i] for i in bi])).to(DEVICE),
                'actions': torch.LongTensor([self.actions[i] for i in bi]).to(DEVICE),
                'log_probs_old': torch.FloatTensor([self.log_probs[i] for i in bi]).to(DEVICE),
                'advantages': torch.FloatTensor([self.advantages[i] for i in bi]).to(DEVICE),
                'returns': torch.FloatTensor([self.returns[i] for i in bi]).to(DEVICE),
            }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class AcademyTrainer:
    def __init__(self, scenario, total_timesteps, log_dir, render=False, seed=42, goal_bonus=10.0):
        info = SCENARIO_INFO.get(scenario, {"agents": 4, "max_steps": 500})
        self.num_agents = info["agents"]
        self.max_steps = info["max_steps"]
        self.scenario = scenario
        self.total_timesteps = total_timesteps
        self.log_dir = log_dir
        self.goal_bonus = goal_bonus

        os.makedirs(log_dir, exist_ok=True)

        _set_seed_util(seed)

        self.env = create_raw_env(
            env_name=scenario,
            num_agents=self.num_agents,
            render=render,
            write_dumps=False,
        )

        self.policy = SimpleActorCritic(OBS_DIM, ACTION_SPACE_SIZE).to(DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.buffer = RolloutBuffer()

        self.global_step = 0
        self.episode_count = 0
        self.log = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_gf': [],
            'episode_ga': [],
            'episode_game_reward': [],
        }

    def _get_obs(self, game_state, player_idx):
        return _extract_obs_util(game_state, player_idx, OBS_DIM)

    def _run_episode(self):
        reset_result = self.env.reset()
        obs_raw = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        game_state = extract_game_state(obs_raw)

        episode_reward = 0.0
        episode_game_reward = 0.0
        episode_steps = 0
        prev_gf = 0

        done = False
        while not done and episode_steps < self.max_steps:
            # Collect actions for all agents
            joint_actions = []
            agent_data = []

            for i in range(self.num_agents):
                obs_vec = self._get_obs(game_state, i)
                obs_t = torch.FloatTensor(obs_vec).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    action, log_prob, _, value = self.policy.get_action_and_value(obs_t)
                    action = action.item()
                    log_prob = log_prob.item()
                    value = value.item()

                joint_actions.append(action)
                agent_data.append((obs_vec, action, log_prob, value))

            # Step environment
            step_result = self.env.step(joint_actions)
            if len(step_result) == 5:
                obs_raw, game_reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs_raw, game_reward, done, info = step_result

            new_game_state = extract_game_state(obs_raw)

            # Reward = raw game reward (summed over controlled agents) + goal bonus
            _score = new_game_state.get('score', [0, 0])
            _gf = _score[0] if isinstance(_score, (list, tuple)) else 0
            goals_scored = _gf - prev_gf
            game_reward = float(np.sum(game_reward))  # multi-agent: per-agent array
            reward = game_reward + self.goal_bonus * max(0, goals_scored)
            prev_gf = _gf

            episode_game_reward += game_reward
            episode_reward += reward

            # Store per-agent transitions (shared team reward)
            per_agent_reward = reward / self.num_agents
            for obs_vec, act, lp, val in agent_data:
                self.buffer.add(obs_vec, act, lp, val, per_agent_reward, done)

            episode_steps += 1
            self.global_step += self.num_agents
            game_state = new_game_state

        # PPO update
        if len(self.buffer.obs) > 0:
            with torch.no_grad():
                last_obs = torch.FloatTensor(self.buffer.obs[-1]).unsqueeze(0).to(DEVICE)
                last_value = self.policy.get_value(last_obs).item()
            self.buffer.compute_gae(last_value)
            self._ppo_update()

        # Stats
        _final = game_state.get('score', [0, 0])
        gf = _final[0] if isinstance(_final, (list, tuple)) else 0
        ga = _final[1] if isinstance(_final, (list, tuple)) else 0

        self.episode_count += 1
        self.log['episode_rewards'].append(episode_reward)
        self.log['episode_lengths'].append(episode_steps)
        self.log['episode_gf'].append(gf)
        self.log['episode_ga'].append(ga)
        self.log['episode_game_reward'].append(episode_game_reward)

        return {
            'reward': episode_reward,
            'game_reward': episode_game_reward,
            'length': episode_steps,
            'gf': gf, 'ga': ga,
        }

    def _ppo_update(self):
        for _ in range(NUM_EPOCHS):
            for batch in self.buffer.get_batches():
                logits, values = self.policy(batch['obs'])
                dist = torch.distributions.Categorical(logits=logits)

                log_probs = dist.log_prob(batch['actions'])
                entropy = dist.entropy().mean()

                ratio = (log_probs - batch['log_probs_old']).exp()
                adv = batch['advantages']
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                clip_adv = torch.clamp(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE) * adv
                policy_loss = -torch.min(ratio * adv, clip_adv).mean()

                value_loss = 0.5 * (values.squeeze(-1) - batch['returns']).pow(2).mean()

                loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

        self.buffer.reset()

    def _build_run_log(self, train_time_s=None):
        """Run log: shared metadata + hyperparams + per-episode iteration results."""
        payload = run_metadata(
            script="scripts/train_academy.py",
            algo="ppo",
            scenario=self.scenario,
            num_agents=self.num_agents,
            total_timesteps=self.total_timesteps,
            timesteps_executed=self.global_step,
            episodes=len(self.log['episode_rewards']),
            train_time_s=round(train_time_s, 2) if train_time_s is not None else None,
            goal_bonus=self.goal_bonus,
            hyperparams=hyperparams_snapshot({
                "learning_rate": LEARNING_RATE, "gamma": GAMMA, "gae_lambda": GAE_LAMBDA,
                "clip_range": CLIP_RANGE, "ent_coef": ENT_COEF, "vf_coef": VF_COEF,
                "minibatch_size": MINIBATCH_SIZE, "num_epochs": NUM_EPOCHS,
                "goal_bonus": self.goal_bonus,
            }),
            observation=obs_snapshot(
                "raw game_state -> hmarl.utils.extract_obs_vector", [OBS_DIM], OBS_HMARL_115,
            ),
        )
        payload.update(self.log)
        return payload

    def train(self):
        print(f"Academy Trainer: {self.scenario} | agents={self.num_agents} | "
              f"timesteps={self.total_timesteps} | GOAL_BONUS={self.goal_bonus}")
        print(f"Device: {DEVICE}")

        start_time = time.time()
        pbar = None
        if HAS_TQDM:
            pbar = tqdm(total=self.total_timesteps, desc="Training", unit="step")

        while self.global_step < self.total_timesteps:
            stats = self._run_episode()

            if pbar:
                pbar.update(stats['length'] * self.num_agents)
                pbar.set_postfix({
                    'Ep': self.episode_count,
                    'R': f"{stats['reward']:.1f}",
                    'GF': stats['gf'],
                    'GA': stats['ga'],
                })

            if self.episode_count % 20 == 0:
                recent = self.log['episode_gf'][-20:]
                avg_gf = sum(recent) / len(recent)
                recent_ga = self.log['episode_ga'][-20:]
                avg_ga = sum(recent_ga) / len(recent_ga)
                recent_r = self.log['episode_rewards'][-20:]
                avg_r = sum(recent_r) / len(recent_r)
                print(f"  Ep {self.episode_count:4d} | Step {self.global_step:7d} | "
                      f"Avg GF={avg_gf:.1f} GA={avg_ga:.1f} R={avg_r:.1f}")

        if pbar:
            pbar.close()

        # Save log (shared metadata + hyperparams + per-episode iteration results)
        save_run_log(
            os.path.join(self.log_dir, 'training_log.json'),
            self._build_run_log(train_time_s=time.time() - start_time),
        )

        # Save model
        model_path = os.path.join(self.log_dir, 'academy_model.pt')
        torch.save(self.policy.state_dict(), model_path)

        elapsed = time.time() - start_time
        n = len(self.log['episode_gf'])
        total_gf = sum(self.log['episode_gf'])
        total_ga = sum(self.log['episode_ga'])
        print(f"\n{'='*60}")
        print(f"Training complete: {n} episodes in {elapsed:.0f}s")
        print(f"Total GF={total_gf} GA={total_ga}")
        print(f"Scoring rate: {total_gf/n:.2f} goals/episode")
        q = max(1, n // 4)  # short runs: avoid 0-episode quartiles
        print(f"First 25% avg GF: {sum(self.log['episode_gf'][:q])/q:.2f}")
        print(f"Last  25% avg GF: {sum(self.log['episode_gf'][-q:])/q:.2f}")
        print(f"Saved: {log_path}, {model_path}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Academy PPO Trainer (no HMARL)")
    parser.add_argument("--scenario", type=str, default="academy_single_goal_versus_lazy",
                        help="GRF academy scenario name")
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="Total training timesteps")
    parser.add_argument("--log-dir", type=str, default="dumps_academy",
                        help="Log/output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--goal-bonus", type=float, default=GOAL_BONUS,
                        help="Extra reward per goal scored (default: 10.0)")
    args = parser.parse_args()

    trainer = AcademyTrainer(
        scenario=args.scenario,
        total_timesteps=args.timesteps,
        log_dir=args.log_dir,
        render=args.render,
        seed=args.seed,
        goal_bonus=args.goal_bonus,
    )

    # Graceful shutdown
    def _shutdown(signum, frame):
        print(f"\n[SHUTDOWN] Saving...")
        save_run_log(
            os.path.join(args.log_dir, 'training_log.json'),
            trainer._build_run_log(),
        )
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    trainer.train()


if __name__ == "__main__":
    main()
