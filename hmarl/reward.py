"""Reward shaping components for HMARL training.

Implements:
- Formation Adherence Index (FAI) - r_high
- Progressive Pass Ratio (PPR) - r_mid
- Role Coherence Index (RCI) - r_low (component, not full metric)
- Ball Progression Reward - r_progression (dense forward-progress signal)
- GK Positional Penalty - r_gk (penalise goalkeeper leaving the box)

Combined reward: R_t = r_game + α_H·ρ_fa(t) + α_M·PPR(t) + (α_L/N)·Σ RCI_i(t) + α_P·progression(t) + gk_penalty(t)
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

# Reward coefficients (rescaled 2026-10-13)
# Previous values made shaping ~99% of episode reward (game signal ≈ 1.5%).
# New balance: game ≈ 71%, shaping ≈ 29%.  r_mid and r_low are non-trivial
# (29% and 22% of shaping respectively).  See analysis in commit message.
ALPHA_HIGH = 0.001   # FAI contribution (was 0.05)
ALPHA_MID = 0.002    # PPR contribution (was 0.05)
ALPHA_LOW = 0.001    # RCI contribution (was 0.02)
# Ball progression dense reward (v2, 2026-10-13): potential-based, see
# BallProgressionTracker docstring for why v1 produced no gradient.
ALPHA_PROG = 0.05    # (was 0.5)

# Scale raw GRF game reward (±1 per goal) so it dominates shaping.
# 3 goals against → -15/ep vs shaping ≈ +6/ep.
GAME_SCALE = 5.0

# Game reward modifiers (scale from +1/-1 to meaningful values)
GAME_WIN = +10.0
GAME_DRAW = 0.0
GAME_LOSE = -15.0

# Concede penalty: per goal against during episode
CONCEDE_PENALTY = -2.0

# Maximum possible deviation (diagonal of the pitch)
D_MAX = 2.24  # sqrt(2^2 + 1.2^2) ≈ 2.33, use 2.24 as practical max

# Ball progression (potential-based v2): per-step clip
PROG_CLIP = 0.1

# GK positional penalty: penalise goalkeeper leaving the penalty area.
# GRF penalty area ≈ x ∈ [-1.0, -0.7], y ∈ [-0.26, 0.26].
# Penalty is linear in overshoot per axis (summed) so the agent learns
# a soft rectangular boundary.
GK_BOX_X = -0.7
GK_BOX_Y = 0.26
ALPHA_GK = 0.02   # per-step penalty weight (was 0.3)


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
    """Track ball possession-progress for dense shaping (v2, potential-based).

    Potential φ(s) = ball_x · [we own the ball]; reward r_t = clip(φ(s_{t+1}) − φ(s_t)).

    Why v1 failed (2026-10-13 diagnosis): v1 rewarded Δx only past a 0.02
    threshold; typical per-step Δx ≈ 0.005, so r_prog fired ~never
    (episode_r_progression ≈ 0.004/episode) and gave PPO zero dense gradient.
    v1 also ignored possession: shoving the ball forward with no control paid
    the same as a real attack. v2 is dense every step and gates on
    possession, so turnovers in advanced areas are penalised via the φ drop.
    As a potential-difference term it does not bias optimal policy
    (Ng et al. 1999).
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self._prev_phi = None
        self.total_progression = 0.0

    def update(self, game_state: Dict) -> float:
        """Compute potential-based progression reward for current step."""
        ball_x = get_ball_position(game_state)[0]
        owned = 1.0 if game_state.get('ball_owned_team', -1) == 0 else 0.0
        phi = ball_x * owned

        if self._prev_phi is None:
            self._prev_phi = phi
            return 0.0

        delta = phi - self._prev_phi
        self._prev_phi = phi
        reward = float(np.clip(delta, -PROG_CLIP, PROG_CLIP))
        self.total_progression += reward
        return reward


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
        self._prev_owned_team = -1
        self._prev_owned_player = -1

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

                # Check if progressive using pass initiator position
                if self._pass_start_pos is not None:
                    start_x = self._pass_start_pos[0]
                    end_pos = get_player_position(game_state, 'left', ball_owned_player)
                    if end_pos[0] > start_x:
                        self.progressive_passes += 1

            elif prev_team == 0 and ball_owned_team != 0:
                # Pass failed (lost possession) — reset pass tracking
                self.pass_attempts += 1
                self._pass_start_pos = None
                self._last_possessor = -1

            elif prev_team == 0 and ball_owned_team == 0 and prev_player == ball_owned_player:
                # Same possessor, no pass — just update position
                pass
            elif ball_owned_team != 0 or ball_owned_player < 0:
                # Loose ball or opponent possession — reset
                self._pass_start_pos = None
                self._last_possessor = -1

        # Track pass initiation position
        if ball_owned_team == 0 and ball_owned_player >= 0:
            if self._last_possessor != ball_owned_player:
                # New possessor: set pass start to PREVIOUS possessor's position
                if self._last_possessor >= 0 and self._last_possessor_pos is not None:
                    self._pass_start_pos = self._last_possessor_pos
                else:
                    self._pass_start_pos = get_player_position(game_state, 'left', ball_owned_player)
            self._last_possessor = ball_owned_player
            self._last_possessor_pos = get_player_position(game_state, 'left', ball_owned_player)
        elif ball_owned_team != 0 or ball_owned_player < 0:
            self._last_possessor = -1
            self._last_possessor_pos = None
            self._pass_start_pos = None

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
            'ppr': self.get_ppr(),
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
    enable_fai: bool = True,
    enable_ppr: bool = True,
    enable_rci: bool = True,
    enable_gk_penalty: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """Compute combined hierarchical reward.

    R_t = r_game + α_H·ρ_fa(t) + α_M·PPR(t) + (α_L/N)·Σ RCI_i(t)
          + α_P·progression(t) + gk_penalty(t)

    Ablation flags: enable_fai, enable_ppr, enable_rci, enable_gk_penalty
    (default True).  Set to False to ablate individual reward components.

    Returns:
        total_reward, breakdown_dict
    """
    # FAI (formation adherence)
    rho_fa = compute_fai(game_state, num_agents) if enable_fai else 0.0

    # PPR (progressive pass ratio)
    ppr = pass_tracker.get_ppr() if enable_ppr else 0.0

    # RCI per agent
    if enable_rci:
        rci_tracker.update(actual_actions, ideal_actions)
        rci_per_agent = rci_tracker.get_rci_per_agent()
        avg_rci = float(np.mean(rci_per_agent))
    else:
        avg_rci = 0.0

    # Ball progression (dense signal)
    r_progression = 0.0
    if ball_progression_tracker is not None:
        r_progression = ball_progression_tracker.update(game_state)

    # GK positional penalty: linear penalty when GK (player 0) leaves the box
    r_gk = 0.0
    if enable_gk_penalty:
        gk_pos = get_player_position(game_state, 'left', 0)
        overshoot_x = max(0.0, gk_pos[0] - GK_BOX_X)          # too far forward
        overshoot_y = max(0.0, abs(gk_pos[1]) - GK_BOX_Y)      # too far sideways
        overshoot = overshoot_x + overshoot_y
        if overshoot > 0:
            r_gk = -ALPHA_GK * overshoot

    # Combined reward
    r_high = ALPHA_HIGH * rho_fa
    r_mid = ALPHA_MID * ppr
    r_low = ALPHA_LOW * avg_rci
    r_prog = ALPHA_PROG * r_progression

    total = GAME_SCALE * game_reward + r_high + r_mid + r_low + r_prog + r_gk

    breakdown = {
        'game_reward': GAME_SCALE * game_reward,
        'fai': rho_fa,
        'ppr': ppr,
        'rci_avg': avg_rci,
        'r_high': r_high,
        'r_mid': r_mid,
        'r_low': r_low,
        'r_progression': r_prog,
        'r_gk_penalty': r_gk,
        'total': total,
    }

    return total, breakdown
