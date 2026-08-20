import random
import types
from typing import Any, Dict, List, Tuple

import gfootball.env as football_env
import numpy as np

# --- Configuration Constants ---
ENV_NAME = "11_vs_11_stochastic"
NUM_AGENTS = 11
ACTION_SPACE_SIZE = 19  # 0 to 18
LOG_DIR = "dumps"


class SoccerAgent:
    """Represents an individual agent (player) in the soccer simulation."""

    def __init__(self, agent_id: int):
        self.agent_id = agent_id

    def decide_action(self, observation: Any) -> int:
        """
        Decides on an action based on the current observation.
        Currently implements a random policy.
        """
        return random.randint(0, ACTION_SPACE_SIZE - 1)


class SoccerMatch:
    """
    Model class that manages the soccer environment and agents.
    Follows Agent-Based Modeling (ABM) principles.
    """

    def __init__(self, env_name: str = ENV_NAME, num_agents: int = NUM_AGENTS, render: bool = True):
        self.should_render = render
        self.num_agents = num_agents
        self.env = self._create_and_patch_env(env_name, num_agents, self.should_render)
        self.agents = [SoccerAgent(i) for i in range(num_agents)]
        self.current_obs = None
        self.step_count = 0

    def _create_and_patch_env(self, env_name: str, num_agents: int, render: bool) -> Any:
        """Initializes and patches the football environment."""
        env = football_env.create_environment(
            env_name=env_name,
            representation="raw",
            render=render,
            number_of_left_players_agent_controls=num_agents,
            write_full_episode_dumps=True,
            logdir=LOG_DIR,
        )
        return self._patch_grf_env(env)

    def _patch_grf_env(self, env: Any) -> Any:
        """Patches GRF environment for Gymnasium compatibility."""
        orig_reset = env.reset
        orig_step = env.step

        def reset_wrapper(self_env, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
            try:
                result = orig_reset(*args, **kwargs)
            except TypeError:
                result = orig_reset()
            if isinstance(result, tuple):
                return result
            return result, {}

        def step_wrapper(self_env, *args, **kwargs) -> Tuple[Any, float, bool, Dict[str, Any]]:
            result = orig_step(*args, **kwargs)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
                return obs, reward, done, info
            return result

        env.reset = types.MethodType(reset_wrapper, env)
        env.step = types.MethodType(step_wrapper, env)
        return env

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        """Resets the match to the initial state."""
        self.current_obs, info = self.env.reset()
        self.step_count = 0
        return self.current_obs, info

    def step(self) -> Tuple[Any, List[float], bool, Dict[str, Any]]:
        """Executes a single step in the simulation."""
        self.step_count += 1
        
        # Each agent decides its own action based on the observation
        actions = [agent.decide_action(self.current_obs) for agent in self.agents]
        
        self.current_obs, rewards, done, info = self.env.step(actions)
        return self.current_obs, rewards, done, info

    def render(self):
        """Renders the current state of the match."""
        self.env.render()

    def close(self):
        """Closes the environment."""
        self.env.close()

    def run(self, max_steps: int = 1000):
        """Runs the complete match simulation."""
        obs, info = self.reset()
        done = False

        print(f"Starting match: {ENV_NAME} with {self.num_agents} agents.")
        try:
            while not done and self.step_count < max_steps:
                obs, rewards, done, info = self.step()
                if self.should_render:
                    self.render()

                if any(r != 0 for r in rewards):
                    print(f"Step {self.step_count:4d} | Rewards: {rewards}")

        except KeyboardInterrupt:
            print("\nMatch interrupted by user.")
        finally:
            self.close()
            print(f"Match ended after {self.step_count} steps.")


if __name__ == "__main__":
    match = SoccerMatch(render=False)
    match.run(max_steps=3000)
