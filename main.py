import numpy as np
import gfootball.env as football_env
import types

# === PATCH: GRF Compatibility for Gymnasium / NumPy 2.x ===
def patch_grf_env(env):
    """Patch GRF env to handle Gymnasium-style API calls safely."""
    orig_reset = env.reset
    orig_step = env.step

    def reset_wrapper(self, *args, **kwargs):
        # Accept all Gymnasium args (seed, options, etc.)
        try:
            result = orig_reset(*args, **kwargs)
        except TypeError:
            # fallback if old signature causes error
            result = orig_reset()
        if isinstance(result, tuple):
            return result
        return result, {}

    def step_wrapper(self, *args, **kwargs):
        result = orig_step(*args, **kwargs)
        if len(result) == 5:  # Gymnasium format
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            return obs, reward, done, info
        else:  # GRF / old Gym format
            return result

    env.reset = types.MethodType(reset_wrapper, env)
    env.step = types.MethodType(step_wrapper, env)
    return env
# ==========================================================


# === Create patched environment ===
env = football_env.create_environment(
    env_name="11_vs_11_stochastic",
    representation="raw",
    render=True,
    number_of_left_players_agent_controls=11
)
env = patch_grf_env(env)

# === Formation logic etc. ===
formation = [
    (0.05, 0.50),  # GK
    (0.15, 0.35), (0.15, 0.40), (0.15, 0.60), (0.15, 0.65),
    (0.35, 0.20), (0.35, 0.40), (0.35, 0.60), (0.35, 0.80),
    (0.55, 0.40), (0.55, 0.60)
]

GK_INDEX = 0
BACK4_INDICES = [1, 2, 3, 4]
FPS = 30
PASS_DELAY = 3 * FPS
possession_timer = np.zeros(11, dtype=int)

def move_towards(player_pos, target_pos):
    dx, dy = target_pos[0] - player_pos[0], target_pos[1] - player_pos[1]
    if abs(dx) < 0.02 and abs(dy) < 0.02: return 0
    if dx > 0 and abs(dy) < 0.1: return 5
    if dx < 0 and abs(dy) < 0.1: return 1
    if dy > 0 and abs(dx) < 0.1: return 7
    if dy < 0 and abs(dx) < 0.1: return 3
    if dx > 0 and dy > 0: return 6
    if dx > 0 and dy < 0: return 4
    if dx < 0 and dy > 0: return 8
    if dx < 0 and dy < 0: return 2
    return 0

# === Main loop ===
obs, info = env.reset()
done, step = False, 0

while not done:
    step += 1
    obs_dict = obs[0] if isinstance(obs, list) else obs

    player_positions = np.array(obs_dict['left_team'])
    norm_positions = (player_positions + 1) / 2.0
    ball_pos = (np.array(obs_dict['ball'][:2]) + 1) / 2.0

    left_has_ball = obs_dict['ball_owned_team'] == 0
    ball_owner = obs_dict['ball_owned_player'] if left_has_ball else None

    actions = []
    for i in range(11):
        pos = norm_positions[i]

        if left_has_ball and ball_owner == i:
            possession_timer[i] += 1
            if possession_timer[i] >= PASS_DELAY:
                act = 9  # short pass
                possession_timer[i] = 0
            else:
                act = 9  # keep ball
        elif left_has_ball:
            act = 9
        else:
            if i in BACK4_INDICES or i == GK_INDEX:
                target = formation[i]
                act = move_towards(pos, target)
            else:
                act = 9
            possession_timer[i] = 0

        actions.append(act)

    obs, reward, done, info = env.step(actions)
    env.render()

    if any(r != 0 for r in reward):
        print(f"Step {step} | Reward: {reward}")

env.close()
