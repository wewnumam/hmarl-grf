
import mesa
from mesa.space import ContinuousSpace
import numpy as np
import gfootball.env as football_env
from gfootball.env import football_action_set
from hmarl import HierarchicalController
import types

def patch_grf_env(env):
    """Patch GRF env (unwrapped) to handle Gymnasium-style API calls safely."""
    # We target the unwrapped env because Gym wrappers might be crashing 
    # if the inner env doesn't return (obs, info) or 5-tuple step.
    target_env = env.unwrapped 
    orig_reset = target_env.reset
    orig_step = target_env.step

    def reset_wrapper(self, *args, **kwargs):
        # Accept all Gymnasium args (seed, options, etc.)
        try:
            result = orig_reset(*args, **kwargs)
        except TypeError:
            result = orig_reset()
        if isinstance(result, tuple):
            return result
        return result, {}

    def step_wrapper(self, *args, **kwargs):
        result = orig_step(*args, **kwargs)
        # Convert 4-tuple (old gym) to 5-tuple (new gym/gymnasium)
        if len(result) == 4:
            obs, reward, done, info = result
            return obs, reward, done, False, info
        return result

    target_env.reset = types.MethodType(reset_wrapper, target_env)
    target_env.step = types.MethodType(step_wrapper, target_env)
    return env

class FootballAgent(mesa.Agent):
    """
    An agent representing a football player in the GRF environment.
    """
    def __init__(self, unique_id, model, team_id, player_idx):
        super().__init__(model)
        self.unique_id = unique_id
        self.team_id = team_id # 0 for left, 1 for right
        self.player_idx = player_idx
        self.pos = (0, 0)
        self.action = 0 
        self.is_controlled = False

    def update_state(self, position):
        self.pos = position

    def step(self):
        # Actions are decided centrally by the Model's Controller for the team in this MARL setup
        pass

class FootballModel(mesa.Model):
    """
    A Mesa model that wraps the Google Research Football environment.
    """
    def __init__(self, env_name="11_vs_11_stochastic", width=100, height=64, training_mode=False):
        super().__init__()
        self.env_name = env_name
        self.grid = ContinuousSpace(width, height, True)
        
        # Initialize GRF Environment
        # Use 'simple115' representation for PPO input (vector)
        self.env = football_env.create_environment(
            env_name=self.env_name,
            representation="simple115v2", 
            rewards="scoring,checkpoints",
            number_of_left_players_agent_controls=1,  # Start with single agent control for simplicity or use MultiAgentWrapper
            render=False 
        )
        self.env = patch_grf_env(self.env)
        
        # Determine observation size
        self.reset_env()
        obs_dim = self.last_obs.shape[0] if len(self.last_obs.shape) == 1 else self.last_obs.shape[-1]
        action_dim = 19 # Standard GRF actions
        
        # Initialize HMARL Controller
        self.controller = HierarchicalController(obs_dim, action_dim)
        
        # Create Agents
        # Left Team (We control player 0 for now as 'active' agent in single-agent mode for simplicity)
        # If multi-agent, we need 11 controllers or shared one.
        for i in range(11):
            agent = FootballAgent(i, self, team_id=0, player_idx=i)
            if i == 0: agent.is_controlled = True
            self.agents.add(agent)
            
        # Right Team
        for i in range(11):
            agent = FootballAgent(i+11, self, team_id=1, player_idx=i)
            self.agents.add(agent)
            
        self.reward_history = []
        self.current_episode_reward = 0
        self.steps = 0
        
        # Training buffers
        self.training_mode = training_mode

    def reset_env(self):
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
             self.last_obs, _ = reset_result
        else:
             self.last_obs = reset_result
             
        # In simple115, obs is a flat vector.
        # In simple115, obs is a flat vector.
        # If multi-agent, it's a list. With number_of_left_players_agent_controls=1, it is likely a single array (check shape).
        if len(self.last_obs.shape) > 1 and self.last_obs.shape[0] == 1:
            self.last_obs = self.last_obs[0] # Unwrap
            
        self.current_episode_reward = 0
        self.steps = 0
        # self.update_agents(self.last_obs) # Hard to map 115 vector to positions easily without 'raw'. 
        # For visualization, we might need a separate 'raw' observer or trusted engine access (GRF is tricky).
        # We will assume 'raw' is not easily available if we chose 'simple115'. 
        # Actually we can get both if we wrap, but let's stick to PPO.
        
    def step(self):
        """Advance the model by one step."""
        obs = self.last_obs
        
        # Get Action from HMARL Controller
        # For the controlled agent
        action = self.controller.get_action(obs)
        
        # Step Environment
        step_result = self.env.step([action])
        if len(step_result) == 5:
             next_obs, reward, terminated, truncated, info = step_result
             done = terminated or truncated
        else:
             next_obs, reward, done, info = step_result
        
        if len(next_obs.shape) > 1 and next_obs.shape[0] == 1:
            next_obs = next_obs[0]

        # Store for PPO update
        self.controller.store_transition(reward, done)
        
        self.last_obs = next_obs
        self.current_episode_reward += reward
        self.steps += 1
        
        if done:
            self.reward_history.append(self.current_episode_reward)
            self.reset_env()
            
        # Trigger training periodically if in training mode
        if self.training_mode and done and len(self.controller.low_level.obs) > 100:
             self.controller.low_level.train(next_value_est=0)

        return done

    def train_batch(self, episodes=10):
        for _ in range(episodes):
            done = False
            self.reset_env()
            while not done:
                done = self.step()

