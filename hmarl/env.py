"""Base GRF environment wrapper with state extraction for hierarchical MARL."""

import types
from typing import Any, Dict, List, Tuple

import gfootball.env as football_env
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ENV_NAME = "11_vs_11_stochastic"
NUM_AGENTS = 11
ACTION_SPACE_SIZE = 19
LOG_DIR = "dumps"

# Observation feature indices in simple115v2 per player
PLAYER_FEATURES = 10  # active, x, y, dx, dy, tired, yellow, red, role, ?
BALL_FEATURES = 11    # x,y,z,dx,dy,dz,rx,ry,rz,owned_team,owned_player
LEFT_START = 0        # left team features start at index 0 (after ball)
# Actually in simple115v2: ball(11) + left(11*10) + right(11*10) + game_state(8)
# But the actual layout is different. Let's just use the raw features directly.

# GRF observation keys for raw representation
RAW_OBS_KEYS = [
    'ball', 'ball_direction', 'ball_rotation',
    'ball_owned_team', 'ball_owned_player',
    'left_team', 'left_team_direction', 'left_team_tired_factor',
    'left_team_yellow_card', 'left_team_active', 'left_team_roles',
    'right_team', 'right_team_direction', 'right_team_tired_factor',
    'right_team_yellow_card', 'right_team_active', 'right_team_roles',
    'active', 'designated', 'sticky_actions',
    'score', 'steps_left', 'game_mode',
]

# Role constants
ROLE_GK = 0
ROLE_CB = 1
ROLE_LB = 2  # LB in GRF = Left Back
ROLE_RB = 3  # RB in GRF = Right Back
ROLE_DM = 4
ROLE_CM = 5
ROLE_LM = 6  # Left Wing
ROLE_RM = 7  # Right Wing
ROLE_AM = 8
ROLE_CF = 9

ROLE_NAMES = {
    ROLE_GK: 'GK', ROLE_CB: 'CB', ROLE_LB: 'LB', ROLE_RB: 'RB',
    ROLE_DM: 'DM', ROLE_CM: 'CM', ROLE_LM: 'LM', ROLE_RM: 'RM',
    ROLE_AM: 'AM', ROLE_CF: 'CF',
}

# 4-3-3 formation mapping: role_id -> player_index
FORMATION_433 = {
    0: ROLE_GK, 1: ROLE_CB, 2: ROLE_CB, 3: ROLE_LB, 4: ROLE_RB,
    5: ROLE_DM, 6: ROLE_CM, 7: ROLE_CM, 8: ROLE_LM, 9: ROLE_RM, 10: ROLE_CF,
}

# Tactical unit grouping by role
TACTICAL_UNITS = {
    'defense': [ROLE_GK, ROLE_CB, ROLE_LB, ROLE_RB],
    'midfield': [ROLE_DM, ROLE_CM, ROLE_AM],
    'attack': [ROLE_LM, ROLE_RM, ROLE_CF],
}

# GRF action names (index -> name)
ACTION_NAMES = [
    'idle',
    'left', 'top_left', 'top', 'top_right',
    'right', 'bottom_right', 'bottom', 'bottom_left',
    'long_pass', 'high_pass', 'short_pass', 'shot',
    'sprint', 'release_direction', 'release_sprint',
    'sliding', 'dribble', 'release_dribble',
]

# Action category mapping for RCI category-based matching
ACTION_CATEGORIES = {
    'passing': {9, 10, 11},       # long_pass, high_pass, short_pass
    'shooting': {12},              # shot
    'movement': {1, 2, 3, 4, 5, 6, 7, 8, 13},  # 8 directions + sprint
    'ball_control': {17},          # dribble
    'defensive': {0, 14, 15, 16, 18},  # idle, release_direction, release_sprint, sliding, release_dribble
}


def get_action_category(action: int) -> str:
    """Map GRF action index to tactical category."""
    for cat, actions in ACTION_CATEGORIES.items():
        if action in actions:
            return cat
    return 'defensive'


# ---------------------------------------------------------------------------
# Environment patching
# ---------------------------------------------------------------------------

def _patch_grf_env(env: Any) -> Any:
    """Patch GRF environment for modern gym compatibility."""
    orig_reset = env.reset
    orig_step = env.step

    def reset_wrapper(self_env, *args, **kwargs):
        try:
            result = orig_reset(*args, **kwargs)
        except TypeError:
            result = orig_reset()
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, {}

    def step_wrapper(self_env, *args, **kwargs):
        result = orig_step(*args, **kwargs)
        if len(result) == 4:
            obs, reward, done, info = result
            return obs, reward, done, False, info
        return result

    env.reset = types.MethodType(reset_wrapper, env)
    env.step = types.MethodType(step_wrapper, env)

    # Recursively patch inner environments
    if hasattr(env, 'env'):
        _patch_grf_env(env.env)

    return env


# ---------------------------------------------------------------------------
# Raw observation parser
# ---------------------------------------------------------------------------

def create_raw_env(
    env_name: str = ENV_NAME,
    num_agents: int = NUM_AGENTS,
    log_dir: str = LOG_DIR,
    render: bool = False,
) -> Any:
    """Create GRF environment with raw representation.
    Returns list[dict] obs — one dict per controlled agent (all identical).
    No patching needed: raw obs works directly with old gym API."""
    return football_env.create_environment(
        env_name=env_name,
        representation="raw",
        number_of_left_players_agent_controls=num_agents,
        stacked=False,
        logdir=log_dir,
        write_full_episode_dumps=True,
        render=render,
    )


def create_simple_env(
    env_name: str = ENV_NAME,
    num_agents: int = NUM_AGENTS,
    log_dir: str = LOG_DIR,
    render: bool = False,
) -> Any:
    """Create GRF environment with simple115v2 representation.
    No patching needed — simple115v2 returns proper numpy arrays."""
    return football_env.create_environment(
        env_name=env_name,
        representation="simple115v2",
        number_of_left_players_agent_controls=num_agents,
        stacked=False,
        logdir=log_dir,
        write_full_episode_dumps=True,
        render=render,
    )


# ---------------------------------------------------------------------------
# State extraction helpers
# ---------------------------------------------------------------------------

def extract_game_state(obs: Any) -> Dict[str, Any]:
    """Extract structured game state from GRF raw observation.

    Works with both raw dict and simple115v2 array observations.
    For raw obs: returns the dict directly.
    For simple115v2: parses the 115-feature vector.
    """
    if isinstance(obs, dict):
        return obs

    # GRF raw obs: list of dicts (one per controlled agent, all identical)
    if isinstance(obs, list) and len(obs) > 0:
        if isinstance(obs[0], dict):
            return obs[0]

    # Parse simple115v2 (shape: (11, 115) for 11 controlled agents)
    # In simple115v2, each agent sees the same global state
    # The obs is (num_agents, 115) but all agents get same global features
    if isinstance(obs, np.ndarray) and obs.ndim == 2:
        # Use the first agent's observation (all share same global state)
        flat = obs[0]
        return _parse_simple115v2(flat)

    return {}


def _parse_simple115v2(flat: np.ndarray) -> Dict[str, Any]:
    """Parse the 115-feature simple115v2 vector into structured state."""
    state = {}
    idx = 0

    # Ball: x, y, z, dx, dy, dz, rx, ry, rz
    state['ball'] = flat[idx:idx+3].tolist()
    idx += 3
    state['ball_direction'] = flat[idx:idx+3].tolist()
    idx += 3
    state['ball_rotation'] = flat[idx:idx+3].tolist()
    idx += 3
    # ball_owned_team, ball_owned_player
    state['ball_owned_team'] = int(round(flat[idx]))
    idx += 1
    state['ball_owned_player'] = int(round(flat[idx]))
    idx += 1

    # Left team: 11 players × 10 features
    left_team = []
    left_roles = []
    for p in range(11):
        active = bool(round(flat[idx]))
        idx += 1
        x, y = flat[idx], flat[idx+1]
        idx += 2
        dx, dy = flat[idx], flat[idx+1]
        idx += 2
        tired = flat[idx]
        idx += 1
        yellow = int(round(flat[idx]))
        idx += 1
        red = int(round(flat[idx]))
        idx += 1
        role = int(round(flat[idx]))
        idx += 1
        left_team.append([x, y])
        left_roles.append(role)

    state['left_team'] = left_team
    state['left_team_roles'] = left_roles

    # Right team: 11 players × 10 features
    right_team = []
    for p in range(11):
        idx += 1  # active
        x, y = flat[idx], flat[idx+1]
        idx += 2
        idx += 2  # direction
        idx += 1  # tired
        idx += 1  # yellow
        idx += 1  # red
        idx += 1  # role
        right_team.append([x, y])

    state['right_team'] = right_team

    # Remaining features: sticky_actions (10), game_mode (7)
    # The exact layout varies; skip for now if we can't parse
    try:
        state['steps_left'] = int(round(flat[-2])) if len(flat) > 113 else 3000
        state['score'] = [0, 0]  # Not directly in simple115v2
        state['game_mode'] = 0
    except (IndexError, ValueError):
        state['steps_left'] = 3000
        state['score'] = [0, 0]
        state['game_mode'] = 0

    return state


def get_ball_position(game_state: Dict) -> List[float]:
    """Get ball [x, y] position."""
    ball = game_state.get('ball', [0, 0, 0])
    return [ball[0], ball[1]] if len(ball) >= 2 else [0, 0]


def get_ball_x(game_state: Dict) -> float:
    """Get ball x-coordinate (left=-1, right=1)."""
    return game_state.get('ball', [0, 0, 0])[0]


def ball_owned_by_us(game_state: Dict, team_id: int = 0) -> bool:
    """Check if ball is owned by our team."""
    return game_state.get('ball_owned_team', -1) == team_id


def get_player_position(game_state: Dict, team: str, player_idx: int) -> List[float]:
    """Get [x, y] position of a specific player."""
    key = f'{team}_team'
    positions = game_state.get(key, [])
    if player_idx < len(positions):
        return positions[player_idx]
    return [0, 0]


def get_player_role(game_state: Dict, team: str, player_idx: int) -> int:
    """Get role ID of a specific player."""
    # GRF raw obs uses 'left_team_roles', 'right_team_roles'
    for key in (f'{team}_team_roles', f'{team}_roles'):
        roles = game_state.get(key, [])
        if player_idx < len(roles):
            return roles[player_idx]
    return ROLE_CM  # default


def get_distance(pos1: List[float], pos2: List[float]) -> float:
    """Euclidean distance between two [x, y] positions."""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
