import argparse
import gfootball.env as football_env
import numpy as np
import gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from typing import Any, Dict, List, Tuple

# --- Configuration Constants ---
DEFAULT_ENV_NAME = "academy_empty_goal_close"
DEFAULT_NUM_AGENTS = 1
ACTION_SPACE_SIZE = 19
LOG_DIR = "dumps"
TRAINING_REWARD_PLOT_PATH = "dumps/training_mean_episode_reward.png"
TRAINING_LENGTH_PLOT_PATH = "dumps/training_mean_episode_length.png"

parser = argparse.ArgumentParser(description="Train and evaluate a PPO baseline in Google Research Football.")
parser.add_argument("--env-name", type=str, default=DEFAULT_ENV_NAME, help="GRF environment name to use.")
parser.add_argument("--num-agents", type=int, default=DEFAULT_NUM_AGENTS, help="Number of controlled agents.")
parser.add_argument("--render", action="store_true", help="Render the environment during evaluation.")
parser.add_argument("--train-steps", type=int, default=2048 * 50, help="Total training timesteps.")
parser.add_argument("--eval-steps", type=int, default=3000, help="Maximum evaluation steps.")

args = parser.parse_args()
ENV_NAME = args.env_name
NUM_AGENTS = args.num_agents


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

    def reset(self) -> np.ndarray:
        """Resets the environment to an initial state."""
        return self._format_observation(self.env.reset())

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
        
        obs = self._format_observation(obs)
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
        self.env = FootballGymEnv(
            env_name,
            num_agents,
            render,
            write_full_episode_dumps=False,
        )
        
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
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01
        )

    def train(self, total_timesteps: int = 25000):
        """Trains the PPO model."""
        print(f"Starting training for {total_timesteps} steps...")
        reward_callback = EpisodeRewardCallback()
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=reward_callback,
        )

        if reward_callback.episode_rewards:
            episode_numbers = np.arange(1, len(reward_callback.episode_rewards) + 1)
            mean_episode_rewards = (
                np.cumsum(reward_callback.episode_rewards) / episode_numbers
            )
            mean_episode_lengths = (
                np.cumsum(reward_callback.episode_lengths) / episode_numbers
            )

            plt.figure(figsize=(10, 5))
            plt.plot(
                episode_numbers,
                mean_episode_rewards,
                label="Mean episode reward",
                color="tab:green",
            )
            plt.xlabel("Episode")
            plt.ylabel("Mean episode reward")
            plt.title("Training Mean Episode Reward")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(TRAINING_REWARD_PLOT_PATH, dpi=150)
            plt.close()
            print(f"Training reward plot saved to {TRAINING_REWARD_PLOT_PATH}")

            plt.figure(figsize=(10, 5))
            plt.plot(
                episode_numbers,
                mean_episode_lengths,
                label="Mean episode length",
                color="tab:orange",
            )
            plt.xlabel("Episode")
            plt.ylabel("Mean episode length (steps)")
            plt.title("Training Mean Episode Length")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(TRAINING_LENGTH_PLOT_PATH, dpi=150)
            plt.close()
            print(f"Training length plot saved to {TRAINING_LENGTH_PLOT_PATH}")
        
        model_path = "ppo_11v11_model"
        self.model.save(model_path)
        print(f"Training finished. Model saved to {model_path}.zip")

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
    match = SoccerMatchPPO(
        env_name=ENV_NAME,
        num_agents=NUM_AGENTS,
        render=args.render,
    )

    match.train(total_timesteps=args.train_steps)
    match.run(max_steps=args.eval_steps)