import numpy as np
import random


class QLearningPolicy:
    def __init__(self, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.q = {}
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def encode_state(self, obs):
        bx, by, _ = obs["ball"]
        return (round(bx, 1), round(by, 1))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        self.q.setdefault(state, np.zeros(self.n_actions))
        return int(np.argmax(self.q[state]))

    def learn(self, s, a, r, s_next):
        self.q.setdefault(s, np.zeros(self.n_actions))
        self.q.setdefault(s_next, np.zeros(self.n_actions))
        td = r + self.gamma * np.max(self.q[s_next])
        self.q[s][a] += self.alpha * (td - self.q[s][a])