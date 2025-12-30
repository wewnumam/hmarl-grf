
import mesa
from mesa.space import ContinuousSpace
import numpy as np
import gfootball.env as football_env
from gfootball.env import football_action_set

class FootballAgent(mesa.Agent):
    """
    Au agent representing a football player in the GRF environment.
    """
    def __init__(self, unique_id, model, team_id, player_idx):
        super().__init__(model) # Mesa 3.x Agent init takes model, unique_id is optional/handled differently or property
        self.unique_id = unique_id
        self.team_id = team_id # 0 for left, 1 for right
        self.player_idx = player_idx
        self.pos = (0, 0) # Virtual position (grid or continuous)
        self.action = 0 # Default action (idle)
        self.is_controlled = False # If this agent is controlled by our policy

    def update_state(self, position):
        """Update the agent's knowledge of its position."""
        self.pos = position

    def step(self):
        """
        Decide on an action. 
        For now, this is a random or rule-based placeholder.
        In a full MARL setup, this would query a policy network.
        """
        if self.is_controlled:
            # Example: Simple random action or heuristic
            # In a real MARL, we would pass state to a Q-network here
            self.action = self.model.random.choice(
                [football_action_set.action_idle, 
                 football_action_set.action_left, 
                 football_action_set.action_right,
                 football_action_set.action_top,
                 football_action_set.action_bottom,
                 football_action_set.action_short_pass,
                 football_action_set.action_shot]
            )
        else:
            # Non-controlled agents (opponents or teammates handled by built-in AI) do nothing here
            # The environment converts their 'idle' effectively to built-in AI if we don't control them
            self.action = football_action_set.action_idle

class FootballModel(mesa.Model):
    """
    A Mesa model that wraps the Google Research Football environment.
    """
    def __init__(self, env_name="11_vs_11_stochastic", width=100, height=64):
        super().__init__() # Initialize Mesa Model
        self.env_name = env_name
        self.grid = ContinuousSpace(width, height, True)
        # self.schedule removed in Mesa 3.x, use self.agents
        
        # Initialize GRF Environment
        self.env = football_env.create_environment(
            env_name=self.env_name,
            representation="raw",
            # We control all 11 players on the left team for MARL
            number_of_left_players_agent_controls=11, 
            render=False 
        )
        
        # Create Agents
        # Left Team (ID 0-10)
        for i in range(11):
            agent = FootballAgent(i, self, team_id=0, player_idx=i)
            agent.is_controlled = True
            self.agents.add(agent)
            
        # Right Team (ID 11-21) - Managed by built-in AI usually, but we track them
        for i in range(11):
            agent = FootballAgent(i+11, self, team_id=1, player_idx=i)
            self.agents.add(agent)

        self.last_obs = None
        self.reset_env()

    def reset_env(self):
        self.last_obs = self.env.reset()
        self.update_agents(self.last_obs)

    def update_agents(self, obs):
        """Synchronize Mesa agents with GRF observation."""
        # GRF 'raw' observation structure depends on config.
        # With multi-agent, it might be a list of observations.
        # However, 'raw' positions are shared.
        
        if isinstance(obs, list):
            obs_data = obs[0]
        else:
            obs_data = obs

        # Left team positions
        left_team_pos = obs_data['left_team']
        # We assume agents are ordered in self.agents as added (0-10, 11-21)
        # But explicitly getting by ID is safer if AgentSet order is not guaranteed.
        # For simplicity, we iterate known IDs.
        
        for i in range(11):
            # GRF coords: [-1,1] x [-0.42, 0.42]. Map to Mesa grid (e.g., 100x64)
            # Or just keep raw coords safely.
            pos = left_team_pos[i]
            # Find agent by ID? Or rely on order. 
            # Mesa 3 AgentSet might not be ordered.
            # let's assume we can get them by ID or access list.
            # self.agents is iterable.
            # optimizing:
            pass

        # To avoid heavy lookup, let's just map ID to agent in a local dict if needed, 
        # or iterate all agents and check ID.
        # But we know IDs 0-10 are left, 11-21 are right.
        
        agents_by_id = {agent.unique_id: agent for agent in self.agents}

        for i in range(11):
            if i in agents_by_id:
                agents_by_id[i].update_state(left_team_pos[i])

        right_team_pos = obs_data['right_team']
        for i in range(11):
            aid = i + 11
            if aid in agents_by_id:
                agents_by_id[aid].update_state(right_team_pos[i])
            
        self.ball_pos = obs_data['ball'][:2] # x,y,z -> take x,y

    def step(self):
        """Advance the model by one step."""
        
        # 1. Agents decide actions
        # New Mesa 3.0 API: use shuffle_do or just do
        self.agents.shuffle_do("step")
        
        # 2. Collect actions from controlled agents
        # We need an array of actions for the environment
        # We need to ensure correct order for GRF actions (index 0 to 10 for left team)
        
        # We can construct action list by sorting agents 0-10
        agents_map = {a.unique_id: a for a in self.agents}
        actions = []
        for i in range(11):
            if i in agents_map:
                actions.append(agents_map[i].action)
            else:
                actions.append(0) # Should not happen
            
        # 3. Step GRF Environment
        obs, reward, done, info = self.env.step(actions)
        
        # 4. Update Agents with new state
        self.update_agents(obs)
        self.last_obs = obs
        
        return done

