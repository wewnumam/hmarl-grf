"""Reward shaping components for HMARL training.

Implements:
- Formation Adherence Index (FAI) - r_high
- Progressive Pass Ratio (PPR) - r_mid
- Role Coherence Index (RCI) - r_low (component, not full metric)
- Ball Progression Reward - r_progression (dense forward-progress signal)

Combined reward: R_t = r_game + α_H·ρ_fa(t) + α_M·PPR(t) + (α_L/N)·Σ RCI_i(t) + α_P·progression(t)
"""

from typing import Dict, List, Tuple, Optional

import numpy as np

from hmarl.env import (
    ROLE_GK, ROLE_CB, ROLE_LB, ROLE_RB, ROLE_DM, ROLE_CM,
    ROLE_LM, ROLE_RM, ROLE_AM, ROLE_CF,
    get_ball_position, get_player_position, get_distance,
    get_action_category, ACTION_CATEGORIES,
)
from hmarl.policy import (
    STRATEGY_HIGH_PRESSING, STRATEGY_COUNTER_ATTACK, STRATEGY_POSSESSION,
    SUBGOAL_ZONAL_MARKING, SUBGOAL_BUILD_UP, SUBGOAL_WING_ATTACK,
    SUBGOAL_MAN_MARKING, SUBGOAL_CLEARANCE,
    HighLevelPolicy, MidLevelPolicy, NUM_SUBGOALS,
)

# Reward coefficients (from thesis, scaled up from 0.01 to be meaningful)
# Original 0.01 produced ~0.01 total shaping per step vs -5 to -10 game penalty
ALPHA_HIGH = 0.1    # FAI contribution
ALPHA_MID = 0.1     # PPR contribution
ALPHA_LOW = 0.05    # RCI contribution
ALPHA_PROG = 0.2    # Ball progression dense reward

# Maximum possible deviation (diagonal of the pitch)
D_MAX = 2.24  # sqrt(2^2 + 1.2^2) ≈ 2.33, use 2.24 as practical max

# Ball progression: x-delta thresholds
PROG_STEP_REWARD = 0.5      # Reward per significant forward ball movement
PROG_STEP_THRESHOLD = 0.02  # Min x-delta to count as progression


def compute_formation_targets(
    game_state: Dict,
    ball_pos: List[float],
) -> List[List[float]]:
    """Compute dynamic formation target positions based on ball position.

    Shifts the formation template toward the ball side to create
    realistic tactical positioning.
    """
    ball_x = ball_pos[0]
    ball_y = ball_pos[1]

    # 4-3-3 formation template (role_id -> [x, y])
    # Aligns with env.py FORMATION_433 role assignments
    template = [
        [-0.92,  0.00],  # 0: GK
        [-0.62, -0.12],  # 1: CB
        [-0.62,  0.12],  # 2: CB
        [-0.58, -0.32],  # 3: LB
        [-0.58,  0.32],  # 4: RB
        [-0.30,  0.00],  # 5: DM
        [-0.22, -0.16],  # 6: CM
        [-0.22,  0.16],  # 7: CM
        [ 0.25, -0.32],  # 8: LM
        [ 0.25,  0.32],  # 9: RM
        [ 0.38,  0.00],  # 10: CF
    ]

    # Shift entire formation forward/backward based on ball position
    shift_x = ball_x * 0.3  # Move formation up to 30% toward ball
    shift_y = ball_y * 0.15  # Slight lateral shift

    targets = []
    for i, (tx, ty) in enumerate(template):
        new_x = np.clip(tx + shift_x, -1.0, 1.0)
        new_y = np.clip(ty + shift_y, -0.42, 0.42)
        targets.append([new_x, new_y])

    return targets


# ---------------------------------------------------------------------------
# Formation Adherence Index (FAI) - ρ_fa
# ---------------------------------------------------------------------------
def compute_fai(
    game_state: Dict,
    num_agents: int = 11,
) -> float:
    """Compute Formation Adherence Index.

    ρ_fa(t) = 1 - (1/(N·D_max)) · Σ d_{i,t}

    where d_{i,t} is Euclidean distance from player i's actual position
    to its target position in the formation.
    """
    ball_pos = get_ball_position(game_state)
    targets = compute_formation_targets(game_state, ball_pos)

    total_deviation = 0.0
    for i in range(num_agents):
        actual = get_player_position(game_state, 'left', i)
        target = targets[i]
        d_i = get_distance(actual, target)
        total_deviation += d_i

    rho_fa = 1.0 - (total_deviation / (num_agents * D_MAX))
    return float(np.clip(rho_fa, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Ball Progression Reward (dense signal for moving ball forward)
# ---------------------------------------------------------------------------
class BallProgressionTracker:
    """Track ball x-position across timesteps for progression reward.

    Provides dense reward signal: reward when ball moves forward (positive Δx).
    This is critical because game reward is extremely sparse (only on goals).
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self._prev_ball_x = None
        self.total_progression = 0.0

    def update(self, game_state: Dict) -> float:
        """Compute ball progression reward for current step.

        Returns positive reward when ball moves toward opponent goal (x increases).
        Returns small negative reward when ball moves backward (x decreases).
        """
        ball_pos = get_ball_position(game_state)
        ball_x = ball_pos[0]

        if self._prev_ball_x is None:
            self._prev_ball_x = ball_x
            return 0.0

        delta_x = ball_x - self._prev_ball_x
        self._prev_ball_x = ball_x

        # Progressive reward: scale by magnitude, cap extremes
        if delta_x > PROG_STEP_THRESHOLD:
            # Ball moving forward: positive reward
            reward = min(PROG_STEP_REWARD * (delta_x / PROG_STEP_THRESHOLD), 1.0)
            self.total_progression += reward
            return reward
        elif delta_x < -PROG_STEP_THRESHOLD:
            # Ball moving backward: small penalty
            reward = max(-0.2 * (abs(delta_x) / PROG_STEP_THRESHOLD), -0.5)
            self.total_progression += reward
            return reward
        return 0.0


# ---------------------------------------------------------------------------
# Progressive Pass Ratio (PPR) - PPR(t)
# ---------------------------------------------------------------------------
class PassTracker:
    """Track pass attempts and success for PPR computation.

    PPR = P_progressive / P_success

    A pass is progressive if ball moves forward (Δx > 0 toward opponent goal).
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.pass_attempts = 0
        self.successful_passes = 0
        self.progressive_passes = 0
        self._last_possessor = -1
        self._last_possessor_pos = None
        self._pass_initiated = False
        self._pass_start_pos = None

    def update(self, game_state: Dict, prev_game_state: Dict = None):
        """Update pass tracking with current game state.

        Detects:
        1. Pass initiated: ball changes from player A to player B on same team
        2. Pass progressive: ball x increased during the pass
        """
        ball_owned_team = game_state.get('ball_owned_team', -1)
        ball_owned_player = game_state.get('ball_owned_player', -1)

        if prev_game_state is not None:
            prev_team = prev_game_state.get('ball_owned_team', -1)
            prev_player = prev_game_state.get('ball_owned_player', -1)

            if prev_team == 0 and ball_owned_team == 0 and prev_player != ball_owned_player:
                # Pass completed within our team
                self.pass_attempts += 1
                self.successful_passes += 1

                # Check if progressive
                if self._pass_start_pos is not None:
                    start_x = self._pass_start_pos[0]
                    end_pos = get_player_position(game_state, 'left', ball_owned_player)
                    if end_pos[0] > start_x:
                        self.progressive_passes += 1

            elif prev_team == 0 and ball_owned_team != 0:
                # Pass failed (lost possession)
                self.pass_attempts += 1

        # Track pass initiation position
        if ball_owned_team == 0 and ball_owned_player >= 0:
            if self._last_possessor != ball_owned_player:
                self._pass_start_pos = get_player_position(game_state, 'left', ball_owned_player)
            self._last_possessor = ball_owned_player
            self._last_possessor_pos = get_player_position(game_state, 'left', ball_owned_player)

    def get_ppr(self) -> float:
        """Get current Progressive Pass Ratio."""
        if self.successful_passes == 0:
            return 0.0
        return self.progressive_passes / self.successful_passes

    def get_stats(self) -> Dict[str, int]:
        """Get pass statistics."""
        return {
            'attempts': self.pass_attempts,
            'successful': self.successful_passes,
            'progressive': self.progressive_passes,
        }


# ---------------------------------------------------------------------------
# Role Coherence Index (RCI) - reward component
# ---------------------------------------------------------------------------
class RCITracker:
    """Track per-agent role coherence for reward shaping.

    Computes running average of f_cat(action, ideal_action) for each agent.
    """
    def __init__(self, num_agents: int = 11):
        self.num_agents = num_agents
        self.reset()

    def reset(self):
        self.scores = np.zeros(self.num_agents)
        self.counts = np.zeros(self.num_agents)

    def update(self, actual_actions: List[int], ideal_actions: List[int]):
        """Update RCI scores with new actions.

        Args:
            actual_actions: actions taken by each agent
            ideal_actions: ideal actions from expert policy
        """
        for i in range(self.num_agents):
            if i < len(actual_actions) and i < len(ideal_actions):
                cat_actual = get_action_category(actual_actions[i])
                cat_ideal = get_action_category(ideal_actions[i])
                match = 1.0 if cat_actual == cat_ideal else 0.0
                self.scores[i] += match
                self.counts[i] += 1

    def get_rci_per_agent(self) -> List[float]:
        """Get RCI for each agent."""
        rci = np.zeros(self.num_agents)
        for i in range(self.num_agents):
            if self.counts[i] > 0:
                rci[i] = self.scores[i] / self.counts[i]
        return rci.tolist()

    def get_team_rci(self) -> float:
        """Get average RCI across all agents."""
        per_agent = self.get_rci_per_agent()
        return float(np.mean(per_agent))


# ---------------------------------------------------------------------------
# Combined reward computation
# ---------------------------------------------------------------------------
def compute_hierarchical_reward(
    game_reward: float,
    game_state: Dict,
    actual_actions: List[int],
    ideal_actions: List[int],
    pass_tracker: PassTracker,
    rci_tracker: RCITracker,
    num_agents: int = 11,
    ball_progression_tracker: Optional[BallProgressionTracker] = None,
) -> Tuple[float, Dict[str, float]]:
    """Compute combined hierarchical reward.

    R_t = r_game + α_H·ρ_fa(t) + α_M·PPR(t) + (α_L/N)·Σ RCI_i(t) + α_P·progression(t)

    Returns:
        total_reward, breakdown_dict
    """
    # FAI (formation adherence)
    rho_fa = compute_fai(game_state, num_agents)

    # PPR (progressive pass ratio)
    ppr = pass_tracker.get_ppr()

    # RCI per agent
    rci_tracker.update(actual_actions, ideal_actions)
    rci_per_agent = rci_tracker.get_rci_per_agent()
    avg_rci = float(np.mean(rci_per_agent))

    # Ball progression (dense signal)
    r_progression = 0.0
    if ball_progression_tracker is not None:
        r_progression = ball_progression_tracker.update(game_state)

    # Combined reward
    r_high = ALPHA_HIGH * rho_fa
    r_mid = ALPHA_MID * ppr
    r_low = ALPHA_LOW * avg_rci
    r_prog = ALPHA_PROG * r_progression

    total = game_reward + r_high + r_mid + r_low + r_prog

    breakdown = {
        'game_reward': game_reward,
        'fai': rho_fa,
        'ppr': ppr,
        'rci_avg': avg_rci,
        'r_high': r_high,
        'r_mid': r_mid,
        'r_low': r_low,
        'r_progression': r_prog,
        'total': total,
    }

    return total, breakdown
