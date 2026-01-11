import numpy as np
import gfootball.env as football_env
import pickle
import os
import types

# === Configuration ===
ENV_NAME = "11_vs_11_stochastic"
# Using 1 controlled player for Q-learning feasibility (tabular methods struggle with high-dim joint state spaces)
NUM_CONTROLLED_PLAYERS = 1 
GRID_SIZE_X = 10
GRID_SIZE_Y = 8

ACTION_SET = [
    0, # idle
    1, # left
    2, # top_left
    3, # top
    4, # top_right
    5, # right
    6, # bottom_right
    7, # bottom
    8, # bottom_left
    9, # long_pass
    10, # high_pass
    11, # short_pass
    12, # shot
    13, # sprint
    14, # release_direction
    15, # release_sprint
    16, # sliding
    17, # dribble
    18, # stop_dribble
]
# We might want to restrict actions for faster learning, but let's keep full set or subset.
# For tabular, fewer actions = faster convergence. Let's use a subset for movement + pass/shoot.
SIMPLE_ACTIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13] # movement + short_pass + shot + sprint

# === PATCH: GRF Compatibility for Gymnasium / NumPy 2.x ===
def patch_grf_env(env):
    """Patch GRF env to handle Gymnasium-style API calls safely."""
    # Use unwrapped to bypass broken Gym compatibility wrappers
    base_env = env.unwrapped
    
    def reset_wrapper(self, *args, **kwargs):
        # GRF base reset doesn't accept seed/options usually
        result = base_env.reset()
        return result, {}

    def step_wrapper(self, *args, **kwargs):
        # GRF base step returns 4 values: obs, reward, done, info
        obs, reward, done, info = base_env.step(*args, **kwargs)
        return obs, reward, done, info

    env.reset = types.MethodType(reset_wrapper, env)
    env.step = types.MethodType(step_wrapper, env)
    return env

class StateDiscretizer:
    def __init__(self, grid_x=10, grid_y=8):
        self.grid_x = grid_x
        self.grid_y = grid_y
        # Field bounds roughly [-1, 1] in X, [-0.42, 0.42] in Y
        self.x_bins = np.linspace(-1, 1, grid_x + 1)
        self.y_bins = np.linspace(-0.42, 0.42, grid_y + 1)

    def discretize(self, obs):
        """
        Convert continuous observation to discrete state index.
        State = (Player Grid Pos, Ball Grid Pos, Ball Ownership)
        """
        # obs is a list for multi-agent, but we control 1, so it might be list of 1 or just dict?
        # GRF returns list of obs, one per controlled player.
        if isinstance(obs, list):
            obs = obs[0]
            
        # 1. Player Position (Active Player)
        # 'left_team' contains positions of all left players.
        # 'active' index usually tracked by environment, but in 'raw' representation for 1 agent, 
        # the agent controls the player marked as 'active' or valid index.
        # However, for simplicity, we assume we control the player closest to ball or just the designated one.
        # In single agent mode, obs['left_team'] has 11 entries, but we effectively control the one with 'sticky_actions' focus?
        # Actually gfootball wrapper handles "active" player logic.
        # But 'raw' obs gives absolute positions.
        # Let's use the position of the player marked as 'ball_owned_player' if we own it,
        # OR just the position of the designated player agent is controlling.
        # For simplicity in this demo: Use the position of player 0 (if we control player 0) or 
        # finding the active one is tricky in 'raw' without extra info.
        # Let's assume we simply discretize the BALL position and OUR GOAL direction?
        # Better: Discretize the player's own position and the ball's position.
        
        # We need to know WHICH player we are. In '11_vs_11_stochastic' with 1 agent, 
        # the agent usually controls the player close to ball.
        # The 'sticky_actions' might tell us, but let's just cheat and check who we probably are.
        # Actually, let's just discretize BALL position and RELATIVE distance to Goal.
        
        ball_pos = obs['ball'][:2]
        ball_x_idx = np.digitize(ball_pos[0], self.x_bins) - 1
        ball_y_idx = np.digitize(ball_pos[1], self.y_bins) - 1
        ball_x_idx = np.clip(ball_x_idx, 0, self.grid_x - 1)
        ball_y_idx = np.clip(ball_y_idx, 0, self.grid_y - 1)
        
        # Ownership
        # 0 = left team, 1 = right team, -1 = no one
        ball_owned = obs['ball_owned_team']
        if ball_owned == -1: owner = 0 # loose
        elif ball_owned == 0: owner = 1 # us
        else: owner = 2 # them
        
        # Tuple state
        return (ball_x_idx, ball_y_idx, owner)

    def get_state_size(self):
        return self.grid_x * self.grid_y * 3

    def state_to_index(self, state):
        # Flatten (x, y, owner) -> int
        bx, by, own = state
        return (bx * self.grid_y + by) * 3 + own

class QLearningAgent:
    def __init__(self, action_size, state_size, learning_rate=0.1, discount_rate=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.action_size = action_size
        self.state_size = state_size
        self.lr = learning_rate
        self.gamma = discount_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table
        self.q_table = np.zeros((state_size, action_size))

    def get_action(self, state_idx, force_greedy=False):
        # Greedy Search / Epsilon Greedy
        if not force_greedy and np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size) # Explore
        return np.argmax(self.q_table[state_idx]) # Exploit

    def update(self, state_idx, action_idx, reward, next_state_idx):
        # Q(s,a) = Q(s,a) + lr * [R + gamma * max Q(s',a') - Q(s,a)]
        target = reward + self.gamma * np.max(self.q_table[next_state_idx])
        self.q_table[state_idx, action_idx] += self.lr * (target - self.q_table[state_idx, action_idx])

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename="q_table.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {filename}")

    def load(self, filename="q_table.pkl"):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Q-table loaded from {filename}")
        else:
            print("No saved Q-table found, starting fresh.")

import matplotlib.pyplot as plt

# ... (Previous imports and config)

def plot_results(rewards, agent, grid_x, grid_y):
    # 1. Plot Reward Curve
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title("Training Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig("reward_curve.png")
    print("Saved reward_curve.png")
    plt.close()

    # 2. Plot Q-Table Heatmaps (Max Q-value per state)
    # State structure: (bx * grid_y + by) * 3 + owner
    # Reshape: (grid_x, grid_y, 3)
    
    # Calculate max Q-value for each state
    q_max = np.max(agent.q_table, axis=1) # Shape: (state_size,)
    
    # Reshape to (GridX, GridY, 3)
    # Warning: Reshape logic must match state_to_index: (bx * grid_y + by) * 3 + own
    # This means 'own' is the fastest varying index, then 'by', then 'bx'.
    # reshape((grid_x, grid_y, 3)) works if indices are filled in that order.
    try:
        heatmap_data = q_max.reshape((grid_x, grid_y, 3))
    except ValueError:
        print("Error reshaping Q-table for heatmap.")
        return

    owner_labels = ["Loose Ball", "Our Possession", "Opponent Possession"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i in range(3):
        im = axes[i].imshow(heatmap_data[:, :, i].T, origin='lower', cmap='viridis', aspect='auto')
        axes[i].set_title(f"Value Map: {owner_labels[i]}")
        axes[i].set_xlabel("Ball X (Grid)")
        axes[i].set_ylabel("Ball Y (Grid)")
        fig.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig("q_table_heatmap.png")
    print("Saved q_table_heatmap.png")
    plt.close()

def train_agent(episodes=100):
    env = football_env.create_environment(
        env_name=ENV_NAME,
        representation="raw",
        stacked=False,
        logdir='/tmp/football',
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        render=False, # Disable render for faster training
        number_of_left_players_agent_controls=NUM_CONTROLLED_PLAYERS
    )
    env = patch_grf_env(env)
    
    discretizer = StateDiscretizer(GRID_SIZE_X, GRID_SIZE_Y)
    agent = QLearningAgent(
        action_size=len(SIMPLE_ACTIONS),
        state_size=discretizer.get_state_size()
    )
    
    # Try to load existing
    agent.load()
    
    reward_history = []
    
    for ep in range(episodes):
        obs, info = env.reset()
        state = discretizer.discretize(obs)
        state_idx = discretizer.state_to_index(state)
        
        total_reward = 0
        done = False
        step = 0
        
        while not done:
            # Get action
            action_idx = agent.get_action(state_idx)
            real_action = SIMPLE_ACTIONS[action_idx]
            
            # Step
            next_obs, reward, done, info = env.step([real_action])
            
            # Process Next State
            next_state = discretizer.discretize(next_obs)
            next_state_idx = discretizer.state_to_index(next_state)
            
            # Update Agent
            agent.update(state_idx, action_idx, reward, next_state_idx)
            
            state_idx = next_state_idx
            total_reward += reward
            step += 1
            
            if done:
                break
        
        agent.decay_epsilon()
        reward_history.append(total_reward)
        print(f"Episode {ep+1}/{episodes} | Steps: {step} | Reward: {total_reward} | Epsilon: {agent.epsilon:.4f}")
        
    agent.save()
    env.close()
    
    # Plot results
    plot_results(reward_history, agent, GRID_SIZE_X, GRID_SIZE_Y)

if __name__ == "__main__":
    train_agent(episodes=10)
