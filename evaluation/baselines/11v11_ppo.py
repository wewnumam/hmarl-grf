"""Flat PPO baseline on GRF 11_vs_11_stochastic (SB3, old-gym API).

Run-logging schema matches evaluation/baselines/academy.py:
per-episode results + run metadata -> dumps/{algo}_{env}_training_log.json.
"""
import os
import sys
import time

import gfootball.env as football_env
import numpy as np
import gym  # old gym: SB3 1.3.0 isinstance-checks spaces against these classes
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from typing import Any, Dict, List, Tuple

# Ensure project root is on path so `import hmarl` works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hmarl.run_logging import (
    run_metadata, save_run_log, sb3_hyperparams, obs_snapshot, OBS_SIMPLE115V2,
)

# --- Configuration Constants ---
ENV_NAME = "11_vs_11_stochastic"
NUM_AGENTS = 11
ACTION_SPACE_SIZE = 19
LOG_DIR = "dumps"
ALGO = "ppo"


class EpisodeRewardCallback(BaseCallback):
    """Collect one total reward value for every completed training episode."""

    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []
        self._current_reward = 0.0
        self._current_length = 0

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]

        self._current_reward += float(rewards[0])
        self._current_length += 1
        if dones[0]:
            self.episode_rewards.append(self._current_reward)
            self.episode_lengths.append(self._current_length)
            self._current_reward = 0.0
            self._current_length = 0

        return True


class FootballGymEnv(gym.Env):
    """
    A Gym-compatible wrapper for the Google Research Football environment.
    Adapted for older gym/Stable Baselines 3 API (Python 3.6 compatibility).
    """
    def __init__(self, env_name: str = ENV_NAME, num_agents: int = NUM_AGENTS, render: bool = False):
        super().__init__()
        self.num_agents = num_agents
        
        # Create GRF environment
        self.env = football_env.create_environment(
            env_name=env_name,
            representation="simple115v2",
            number_of_left_players_agent_controls=num_agents,
            stacked=False,
            logdir=LOG_DIR,
            write_full_episode_dumps=False,
            render=render
        )
        
        # Action space: 11 agents
        self.action_space = gym.spaces.MultiDiscrete([ACTION_SPACE_SIZE] * num_agents)
        
        # Observation space: (11, 115)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(num_agents, 115), 
            dtype=np.float32
        )

    def reset(self) -> np.ndarray:
        """Resets the environment to an initial state."""
        obs = self.env.reset()

        # Ensure we only return the observation, not a tuple
        if isinstance(obs, tuple):
            obs = obs[0]

        return np.array(obs, dtype=np.float32)

    def step(self, actions: Any) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Executes one timestep in the environment."""
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()
            
        result = self.env.step(actions)
        
        # Safely unpack based on what GRF returns, converting to 4-tuple API
        if len(result) == 4:
            obs, reward, done, info = result
        elif len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            raise ValueError(f"Unexpected return from step: {len(result)} items")
        
        obs = np.array(obs, dtype=np.float32)
        team_reward = float(np.sum(reward))
        
        # Return exactly 4 items expected by older Stable Baselines 3
        return obs, team_reward, done, info

    def render(self):
        """Renders the environment."""
        self.env.render()

    def close(self):
        """Closes the environment."""
        self.env.close()

class SoccerMatchPPO:
    """
    Model class that manages the soccer environment and PPO training.
    Follows structure similar to the A2C implementation.
    """
    def __init__(self, env_name: str = ENV_NAME, num_agents: int = NUM_AGENTS, render: bool = False):
        self.env_name = env_name
        self.num_agents = num_agents
        self.render = render
        self.env = FootballGymEnv(env_name, num_agents, render)
        
        # Initialize PPO model following the style of baseline3_ppo.py
        print(f"Initializing PPO on {env_name}...")
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            verbose=1,
            learning_rate=3e-4, # Standard PPO learning rate
            n_steps=2048,       # More steps per update for PPO stability
            batch_size=64,
            n_epochs=10,
            gamma=1.0,           # Song et al. (2024): gamma=1 for 11v11
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01
        )

    def train(self, total_timesteps: int = 25000):
        """Trains the PPO model."""
        print(f"Starting training for {total_timesteps} steps...")
        reward_callback = EpisodeRewardCallback()
        train_start = time.time()
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=reward_callback,
        )
        train_time_s = time.time() - train_start

        model_path = "ppo_11v11_model"
        self.model.save(model_path)
        print(f"Training finished in {train_time_s:.1f}s. Model saved to {model_path}.zip")

        # Run log: shared metadata + per-episode iteration results
        payload = run_metadata(
            script="evaluation/baselines/11v11_ppo.py",
            algo=ALGO,
            env_name=self.env_name,
            num_agents=self.num_agents,
            train_time_s=round(train_time_s, 2),
            total_timesteps=total_timesteps,
            timesteps_executed=int(self.model.num_timesteps),
            episodes=len(reward_callback.episode_rewards),
            hyperparams=sb3_hyperparams(self.model),
            observation=obs_snapshot("simple115v2", [self.num_agents, 115], OBS_SIMPLE115V2),
            episode_rewards=reward_callback.episode_rewards,
            episode_lengths=reward_callback.episode_lengths,
        )
        save_run_log(os.path.join(LOG_DIR, f"{ALGO}_{ENV_NAME}_training_log.json"), payload)

    def run(self, max_steps: int = 3000):
        """Evaluates the trained model in the environment."""
        print("Starting match evaluation...")
        obs = self.env.reset()

        try:
            for step in range(max_steps):
                # Predict action using the trained model
                action, _states = self.model.predict(obs, deterministic=True)

                obs, reward, done, info = self.env.step(action)

                if reward != 0:
                    print(f"Step {step:4d} | Team Reward: {reward}")

                if done:
                    print(f"Match ended after {step} steps.")
                    obs = self.env.reset()
                    break
        except KeyboardInterrupt:
            print("\nEvaluation interrupted by user.")
        finally:
            self.env.close()

if __name__ == "__main__":
    # Create and run the PPO Match
    match = SoccerMatchPPO(render=False)
    
    # Run training (25,000 steps as in baseline3_ppo.py)
    match.train(total_timesteps=3000)
    
    # Run a demonstration
    match.run(max_steps=3000)
