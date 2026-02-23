import gfootball.env as football_env
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from typing import Any, Dict, List, Tuple

# --- Configuration Constants ---
ENV_NAME = "11_vs_11_stochastic"
NUM_AGENTS = 11
ACTION_SPACE_SIZE = 19
LOG_DIR = "dumps"

class FootballGymEnv(gym.Env):
    """
    A Gymnasium-compatible wrapper for the Google Research Football environment.
    This wrapper adapts the multi-agent GRF environment to a single-policy interface 
    suitable for Stable Baselines 3, enabling training with PPO.
    """
    def __init__(self, env_name: str = ENV_NAME, num_agents: int = NUM_AGENTS, render: bool = False):
        super().__init__()
        self.num_agents = num_agents
        
        # Create GRF environment
        env = football_env.create_environment(
            env_name=env_name,
            representation="simple115v2",
            number_of_left_players_agent_controls=num_agents,
            stacked=False,
            logdir=LOG_DIR,
            write_full_episode_dumps=True,
            render=render
        )
        self.env = self._patch_grf_env(env)
        
        # Action space: 11 agents
        self.action_space = gym.spaces.MultiDiscrete([ACTION_SPACE_SIZE] * num_agents)
        
        # Observation space: (11, 115)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(num_agents, 115), 
            dtype=np.float32
        )

    def _patch_grf_env(self, env: Any) -> Any:
        """Recursively patches GRF environment hierarchy for Gymnasium compatibility."""
        import types

        def patch_single_env(e):
            if hasattr(e, '_is_patched'):
                return
            
            orig_reset = e.reset
            orig_step = e.step

            def reset_wrapper(self_env, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
                try:
                    result = orig_reset(*args, **kwargs)
                except (TypeError, ValueError):
                    result = orig_reset()
                
                if isinstance(result, tuple) and len(result) == 2:
                    return result
                return result, {}

            def step_wrapper(self_env, *args, **kwargs) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
                result = orig_step(*args, **kwargs)
                if isinstance(result, tuple) and len(result) == 4:
                    obs, reward, done, info = result
                    return obs, reward, done, False, info
                return result

            e.reset = types.MethodType(reset_wrapper, e)
            e.step = types.MethodType(step_wrapper, e)
            e._is_patched = True

        curr = env
        while curr is not None:
            patch_single_env(curr)
            if hasattr(curr, 'env'):
                curr = curr.env
            else:
                break
        return env

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to an initial state."""
        super().reset(seed=seed)
        obs, info = self.env.reset()
        return np.array(obs, dtype=np.float32), info

    def step(self, actions: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes one timestep in the environment."""
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()
            
        obs, reward, terminated, truncated, info = self.env.step(actions)
        
        obs = np.array(obs, dtype=np.float32)
        team_reward = float(np.sum(reward))
        
        return obs, team_reward, terminated, truncated, info

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
        obs, _ = self.env.reset()
        
        try:
            for step in range(max_steps):
                # Predict action using the trained model
                action, _states = self.model.predict(obs, deterministic=True)
                
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                if reward != 0:
                    print(f"Step {step:4d} | Team Reward: {reward}")
                
                if terminated or truncated:
                    print(f"Match ended after {step} steps.")
                    obs, _ = self.env.reset()
                    break
        except KeyboardInterrupt:
            print("\nEvaluation interrupted by user.")
        finally:
            self.env.close()

if __name__ == "__main__":
    # Create and run the PPO Match
    match = SoccerMatchPPO(render=False)
    
    # Run training (25,000 steps as in baseline3_ppo.py)
    match.train(total_timesteps=12000)
    
    # Run a demonstration
    match.run(max_steps=3000)
