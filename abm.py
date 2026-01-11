from mesa import Model
from mesa.agent import Agent
import gfootball.env as football_env
from policy import QLearningPolicy


class FootballAgent(Agent):
    def __init__(self, model, player_id, policy):
        super().__init__(model)
        self.player_id = player_id
        self.policy = policy
        self.prev_state = None
        self.prev_action = None

    def step(self):
        obs = self.model.observations[self.player_id]
        state = self.policy.encode_state(obs)
        action = self.policy.act(state)
        self.prev_state = state
        self.prev_action = action
        self.model.actions[self.player_id] = action

    def learn(self):
        r = self.model.rewards[self.player_id]
        obs_next = self.model.observations[self.player_id]
        s_next = self.policy.encode_state(obs_next)
        self.policy.learn(self.prev_state, self.prev_action, r, s_next)


class FootballModel(Model):
    def __init__(self, max_steps=3000):
        super().__init__()
        self.env = football_env.create_environment(
            env_name="11_vs_11_easy_stochastic",
            representation="raw",
            number_of_left_players_agent_controls=11,
            number_of_right_players_agent_controls=0,
            rewards="scoring",
            render=False
        )

        self.n_actions = self.env.action_space.nvec[0]
        self.policy = QLearningPolicy(self.n_actions)

        self.players = [FootballAgent(self, i, self.policy) for i in range(11)]

        self.max_steps = max_steps
        self.step_count = 0

        self.episode_rewards = []
        self.current_reward = 0.0

        self.reset_env()

    def reset_env(self):
        self.observations = self.env.reset()
        self.rewards = [0.0] * 11
        self.actions = {}
        self.step_count = 0
        self.current_reward = 0.0

    def step(self):
        self.step_count += 1

        for p in self.players:
            p.step()

        acts = [self.actions[i] for i in range(11)]
        obs, rewards, done, _ = self.env.step(acts)

        self.observations = obs
        self.rewards = rewards
        self.current_reward += sum(rewards)

        for p in self.players:
            p.learn()

        if done or self.step_count >= self.max_steps:
            self.episode_rewards.append(self.current_reward)
            self.reset_env()