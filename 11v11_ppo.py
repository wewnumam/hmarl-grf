import gfootball.env as football_env
import numpy as np
import gym
from stable_baselines3 import PPO
from typing import Any, Dict, List, Tuple

# --- Configuration Constants ---
ENV_NAME = "11_vs_11_stochastic"
NUM_AGENTS = 11
ACTION_SPACE_SIZE = 19
LOG_DIR = "dumps"

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
            write_full_episode_dumps=True,
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
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01
        )

    def train(self, total_timesteps: int = 25000):
        """Trains the PPO model."""
        print(f"Starting training for {total_timesteps} steps...")
        self.model.learn(total_timesteps=total_timesteps)
        
        model_path = "ppo_11v11_model"
        self.model.save(model_path)
        print(f"Training finished. Model saved to {model_path}.zip")

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
