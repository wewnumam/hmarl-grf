import os
import numpy as np
import pickle
from abm import FootballModel

# --- resolve project root safely ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

model = FootballModel()

episodes = 10
steps_per_episode = 3000

for ep in range(episodes):
    for _ in range(steps_per_episode):
        model.step()
    print(f"Episode {ep} reward: {model.episode_rewards[-1]:.2f}")

# --- save results ---
np.save(
    os.path.join(RESULTS_DIR, "rewards.npy"),
    np.array(model.episode_rewards)
)

with open(os.path.join(RESULTS_DIR, "qtable.pkl"), "wb") as f:
    pickle.dump(model.policy.q, f)
