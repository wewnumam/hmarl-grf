import argparse
import glob
import json
import os
import sys
import time
import gfootball.env as football_env

# Ensure project root is on path so `import hmarl` works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import gym  # old gym: SB3 1.3.0 isinstance-checks spaces against these classes
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback
from typing import Any, Dict, List, Optional, Tuple
from hmarl.run_logging import (
    run_metadata, save_run_log, sb3_hyperparams, obs_snapshot, OBS_SIMPLE115V2,
)

# --- Configuration Constants ---
DEFAULT_ENV_NAME = "academy_empty_goal_close"
DEFAULT_NUM_AGENTS = 1
ACTION_SPACE_SIZE = 19
LOG_DIR = "dumps"

parser = argparse.ArgumentParser(description="Train and evaluate PPO or A2C baselines in Google Research Football.")
parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "a2c"], help="Algorithm: ppo or a2c.")
parser.add_argument("--env-name", type=str, default=DEFAULT_ENV_NAME, help="GRF environment name to use.")
parser.add_argument("--num-agents", type=int, default=DEFAULT_NUM_AGENTS, help="Number of controlled agents.")
parser.add_argument("--render", action="store_true", help="Render the environment during evaluation.")
parser.add_argument("--train-steps", type=int, default=2048 * 50, help="Total training timesteps.")
parser.add_argument("--eval-steps", type=int, default=3000, help="Maximum evaluation steps.")

args = parser.parse_args()
ENV_NAME = args.env_name
NUM_AGENTS = args.num_agents

# Combined comparison figures: one line per algorithm, rebuilt from all JSON logs
TRAINING_REWARD_PLOT_PATH = f"dumps/{ENV_NAME}_training_mean_episode_reward.png"
TRAINING_LENGTH_PLOT_PATH = f"dumps/{ENV_NAME}_training_mean_episode_length.png"
ALGO_COLORS = {"ppo": "tab:green", "a2c": "tab:orange"}


def plot_training_comparison():
    """Overlay every algorithm's training curve (from saved JSON logs) on one figure."""
    plt.style.use("seaborn-darkgrid")
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['cmr10', 'Computer Modern Roman', 'DejaVu Serif']
    
    for metric, ylabel, plot_path in (
        ("episode_rewards", "Mean episode reward", TRAINING_REWARD_PLOT_PATH),
        ("episode_lengths", "Mean episode length (steps)", TRAINING_LENGTH_PLOT_PATH),
    ):
        plt.figure(figsize=(4, 4))
        plotted = False
        for json_path in sorted(glob.glob(f"{LOG_DIR}/*_{ENV_NAME}_training_log.json")):
            with open(json_path) as f:
                log = json.load(f)
            data = log.get(metric, [])
            if not data:
                continue
            algo = log.get("algo", os.path.basename(json_path).split("_")[0])
            episodes = np.arange(1, len(data) + 1)
            mean_values = np.cumsum(data) / episodes
            plt.plot(
                episodes,
                mean_values,
                label=algo.upper(),
                color=ALGO_COLORS.get(algo, None),
            )
            plotted = True

        if not plotted:
            plt.close()
            continue
        
        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.title(f"{ENV_NAME} ($n$={NUM_AGENTS})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Comparison plot saved to {plot_path}")


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
    def __init__(
        self,
        env_name: str = ENV_NAME,
        num_agents: int = NUM_AGENTS,
        render: bool = False,
        write_full_episode_dumps: bool = False,
    ):
        super().__init__()
        self.num_agents = num_agents
        
        # Create GRF environment
        self.env = football_env.create_environment(
            env_name=env_name,
            representation="simple115v2",
            number_of_left_players_agent_controls=num_agents,
            stacked=False,
            logdir=LOG_DIR,
            write_full_episode_dumps=write_full_episode_dumps,
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

    def _format_observation(self, obs: Any) -> np.ndarray:
        """Convert a single-agent GRF observation to the declared shape."""
        if isinstance(obs, tuple):
            obs = obs[0]

        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape == (115,) and self.num_agents == 1:
            obs = obs.reshape(1, 115)

        if obs.shape != self.observation_space.shape:
            raise ValueError(
                f"Unexpected observation shape {obs.shape}; "
                f"expected {self.observation_space.shape}"
            )
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Dict[str, Any] = None):
        """Reset and return observation only (SB3 1.3.0 old-gym API)."""
        if seed is not None:
            try:
                self.env.reset(seed=seed)
            except TypeError:
                pass  # GRF build here does not accept seed kwarg

        result = self.env.reset()
        if isinstance(result, tuple):
            result = result[0]
        return self._format_observation(result)

    def step(self, actions: Any) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one timestep; SB3 1.3.0 expects the 4-tuple old-gym API."""
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()

        result = self.env.step(actions)

        if len(result) == 4:
            obs, reward, done, info = result
        elif len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = bool(terminated) or bool(truncated)
        else:
            raise ValueError(f"Unexpected return from step: {len(result)} items")

        obs = self._format_observation(obs)
        team_reward = float(np.sum(reward))

        return obs, team_reward, bool(done), info

    def render(self):
        """Renders the environment."""
        self.env.render()

    def close(self):
        """Closes the environment."""
        self.env.close()

class SoccerMatch:
    """
    Model class that manages the soccer environment and PPO/A2C training.
    Follows structure similar to the A2C implementation.
    """
    def __init__(self, env_name: str = ENV_NAME, num_agents: int = NUM_AGENTS, render: bool = False, algo: str = "ppo"):
        self.env_name = env_name
        self.num_agents = num_agents
        self.render = render
        self.algo = algo.lower()
        self.env = FootballGymEnv(
            env_name,
            num_agents,
            render,
            write_full_episode_dumps=False,
        )

        # Initialize model following the style of baseline3_ppo.py
        print(f"Initializing {self.algo.upper()} on {env_name}...")
        if self.algo == "a2c":
            # A2C: on-policy, updates every n_steps; no clipping/n_epochs.
            # ent_coef/vf_coef kept equal to PPO's for comparability.
            self.model = A2C(
                policy="MlpPolicy",
                env=self.env,
                verbose=1,
                learning_rate=7e-4,
                n_steps=5,
                gamma=0.99,
                ent_coef=0.01,
                vf_coef=0.5,
            )
        else:
            self.model = PPO(
                policy="MlpPolicy",
                env=self.env,
                verbose=1,
                learning_rate=3e-4, # Standard PPO learning rate
                n_steps=2048,       # More steps per update for PPO stability
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01
            )

    def train(self, total_timesteps: int = 25000):
        """Trains the model (PPO or A2C)."""
        print(f"Starting training for {total_timesteps} steps...")
        reward_callback = EpisodeRewardCallback()
        train_start = time.time()
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=reward_callback,
        )
        train_time_s = time.time() - train_start

        # Save per-episode iteration results + run metadata as JSON (replot later without retraining)
        log_path = os.path.join(LOG_DIR, f"{self.algo}_{ENV_NAME}_training_log.json")
        payload = run_metadata(
            script="evaluation/baselines/academy.py",
            algo=self.algo,
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
        save_run_log(log_path, payload)

        # Rebuild combined comparison figure from all saved algorithm logs
        plot_training_comparison()
        
        # ponytail: was "ppo_11v11_model" — clobbered the 11v11 baseline artifact
        model_path = f"{self.algo}_{ENV_NAME}_model"
        self.model.save(model_path)
        print(f"Training finished in {train_time_s:.1f}s. Model saved to {model_path}.zip")

    def run(self, max_steps: int = 3000):
        """Evaluates the trained model in the environment."""
        print("Starting match evaluation...")
        eval_env = FootballGymEnv(
            self.env_name,
            self.num_agents,
            self.render,
            write_full_episode_dumps=True,
        )
        obs = eval_env.reset()
        cumulative_rewards = []
        cumulative_reward = 0.0

        try:
            for step in range(max_steps):
                # Predict action using the trained model
                action, _states = self.model.predict(obs, deterministic=True)

                obs, reward, done, info = eval_env.step(action)
                cumulative_reward += reward
                cumulative_rewards.append(cumulative_reward)

                if reward != 0:
                    print(f"Step {step:4d} | Team Reward: {reward}")

                if done:
                    print(f"Match ended after {step} steps.")
                    obs = self.env.reset()
                    break
        except KeyboardInterrupt:
            print("\nEvaluation interrupted by user.")
        finally:
            eval_env.close()
            self.env.close()

if __name__ == "__main__":
    match = SoccerMatch(
        env_name=ENV_NAME,
        num_agents=NUM_AGENTS,
        render=args.render,
        algo=args.algo,
    )

    match.train(total_timesteps=args.train_steps)
    match.run(max_steps=args.eval_steps)