
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np

class PPONetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=64):
        super(PPONetwork, self).__init__()
        # Shared Feature Extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        
        # Actor Head (Policy)
        self.actor = nn.Linear(hidden_size, output_dim)
        
        # Critic Head (Value)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_action_and_value(self, x, action=None):
        logits, value = self.forward(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value

class PPOAgent:
    def __init__(self, observation_space_dim, action_space_dim, lr=3e-4, gamma=0.99, clip_coef=0.2, ent_coef=0.01, vf_coef=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = PPONetwork(observation_space_dim, action_space_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.gamma = gamma
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        
        # Storage for one rollout
        self.obs = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def select_action(self, obs):
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        with torch.no_grad():
            action, logprob, _, value = self.network.get_action_and_value(obs_tensor)
            # Store probs for visualization
            logits, _ = self.network(obs_tensor)
            self.last_probs = torch.softmax(logits, dim=-1).cpu().numpy()
            
        return action.item(), logprob.item(), value.item()

    def store(self, obs, action, logprob, reward, done, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def calculate_advantages(self, next_value):
        # Generalized Advantage Estimation (GAE) usually, but simple MC or n-step for simplicity here
        # Let's do a simple return calculation for the prototype unless GAE is strictly needed.
        # But GAE is standard for PPO.
        
        returns = []
        advantages = [] # We need advantages for PPO
        
        # Bootstrap value
        next_val = next_value
        gae = 0
        lam = 0.95 # GAE lambda
        
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * next_val * (1 - self.dones[step]) - self.values[step]
            gae = delta + self.gamma * lam * (1 - self.dones[step]) * gae
            advantages.insert(0, gae)
            next_val = self.values[step]
            returns.insert(0, gae + self.values[step])
            
        return np.array(returns), np.array(advantages)

    def train(self, next_value_est):
        obs_tensor = torch.FloatTensor(np.array(self.obs)).to(self.device)
        actions_tensor = torch.LongTensor(np.array(self.actions)).to(self.device)
        logprobs_tensor = torch.FloatTensor(np.array(self.logprobs)).to(self.device)
        values_tensor = torch.FloatTensor(np.array(self.values)).to(self.device)
        
        returns, advantages = self.calculate_advantages(next_value_est)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Optimization epochs (e.g. 4 or 10)
        for _ in range(4):
            new_logits, new_values = self.network(obs_tensor)
            new_probs = Categorical(logits=new_logits)
            new_logprobs = new_probs.log_prob(actions_tensor)
            entropy = new_probs.entropy()
            
            logratio = new_logprobs - logprobs_tensor
            ratio = logratio.exp()
            
            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            
            v_loss = 0.5 * ((new_values.view(-1) - returns) ** 2).mean()
            
            loss = pg_loss - self.ent_coef * entropy.mean() + self.vf_coef * v_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
        
        self.clear_memory()
        return loss.item()

    def clear_memory(self):
        self.obs = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []


class HierarchicalController:
    """
    Manages High, Mid, and Low level policies.
    """
    def __init__(self, obs_dim, action_dim):
        # 1. High Level: Strategy (e.g. 0=Attack, 1=Defend)
        self.high_level = PPOAgent(obs_dim, 2, lr=1e-4)
        
        # 2. Mid Level: Sub-goals (e.g. Regions 0-3)
        # Input: Obs + High_Action (appended)
        self.mid_level = PPOAgent(obs_dim + 1, 4, lr=2e-4)
        
        # 3. Low Level: Atomic Actions (from GRF action set)
        # Input: Obs + Mid_Action (appended)
        self.low_level = PPOAgent(obs_dim + 1, action_dim, lr=3e-4) # action_dim approx 19 for GRF
        
        # Timers / Counters for hierarchy frequency
        self.high_interval = 20
        self.mid_interval = 5
        self.step_count = 0
        
        self.current_high_action = 0
        self.current_mid_action = 0
        
        self.last_obs = None
        
        # Temp storage for the current step's low-level decision
        self.last_aug_obs_low = None
        self.last_action = None
        self.last_logprob = None
        self.last_val = None

    def get_action(self, obs, training=True):
        self.step_count += 1
        
        # Handle observation preprocessing if valid
        if not isinstance(obs, (list, np.ndarray)):
             obs = np.zeros(10) # Placeholder
             
        obs = np.array(obs, dtype=np.float32).flatten()
        self.last_obs = obs
        
        # High Level Update
        if self.step_count % self.high_interval == 1:
            action, logprob, val = self.high_level.select_action(obs)
            self.current_high_action = action
            
        # Mid Level Update
        if self.step_count % self.mid_interval == 1:
            # Augment obs with high level command
            aug_obs = np.append(obs, self.current_high_action)
            action, logprob, val = self.mid_level.select_action(aug_obs)
            self.current_mid_action = action
            
        # Low Level Action
        aug_obs_low = np.append(obs, self.current_mid_action)
        action, logprob, val = self.low_level.select_action(aug_obs_low)
        
        # Save context for storage
        self.last_aug_obs_low = aug_obs_low
        self.last_action = action
        self.last_logprob = logprob
        self.last_val = val
        
        return action
        
    def store_transition(self, reward, done):
        """Store the transition in the low-level agent (and others if implemented)."""
        if self.last_aug_obs_low is not None:
            self.low_level.store(
                self.last_aug_obs_low, 
                self.last_action, 
                self.last_logprob, 
                self.last_reward_shaping(reward), # Hook for potential shaping
                done, 
                self.last_val
            )
            
    def last_reward_shaping(self, reward):
        # Passthrough for now
        return reward
