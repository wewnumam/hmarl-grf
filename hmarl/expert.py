"""Expert Policy: rule-based reference for RCI computation.

Expert Policy (π*_Ri) provides the ideal action for each (role, observation, sub-goal).
Used as oracle reference to compute RCI metric.

Parameters (from thesis):
- d_tackle = 0.05  (sliding activation distance)
- d_safe   = 0.15  (safe dribble threshold)
- d_shoot  = 0.30  (optimal shooting distance from penalty point)
"""

from typing import Dict, List, Optional

from hmarl.env import (
    ROLE_GK, ROLE_CB, ROLE_LB, ROLE_RB, ROLE_DM, ROLE_CM,
    ROLE_LM, ROLE_RM, ROLE_AM, ROLE_CF,
    get_ball_position, get_player_position, get_distance, get_player_role,
    ball_owned_by_us,
)
from hmarl.policy import (
    STRATEGY_HIGH_PRESSING, STRATEGY_COUNTER_ATTACK, STRATEGY_POSSESSION,
    SUBGOAL_ZONAL_MARKING, SUBGOAL_BUILD_UP, SUBGOAL_WING_ATTACK,
    SUBGOAL_MAN_MARKING, SUBGOAL_CLEARANCE,
)

# Thresholds
D_TACKLE = 0.05
D_SAFE = 0.15
D_SHOOT = 0.30

# GRF action indices
ACT_IDLE = 0
ACT_LEFT = 1
ACT_TOP_LEFT = 2
ACT_TOP = 3
ACT_TOP_RIGHT = 4
ACT_RIGHT = 5
ACT_BOTTOM_RIGHT = 6
ACT_BOTTOM = 7
ACT_BOTTOM_LEFT = 8
ACT_LONG_PASS = 9
ACT_HIGH_PASS = 10
ACT_SHORT_PASS = 11
ACT_SHOT = 12
ACT_SPRINT = 13
ACT_RELEASE_DIR = 14
ACT_RELEASE_SPRINT = 15
ACT_SLIDING = 16
ACT_DRIBBLE = 17
ACT_RELEASE_DRIBBLE = 18


def _move_toward(from_pos: List[float], to_pos: List[float]) -> int:
    """Return movement action index to move from one position toward another."""
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]

    # Determine primary direction (GRF: 'top' action moves -y, 'bottom' +y)
    if abs(dx) > abs(dy) * 1.5:
        return ACT_RIGHT if dx > 0 else ACT_LEFT
    elif abs(dy) > abs(dx) * 1.5:
        return ACT_BOTTOM if dy > 0 else ACT_TOP
    else:
        if dx > 0 and dy > 0:
            return ACT_BOTTOM_RIGHT
        elif dx > 0 and dy < 0:
            return ACT_TOP_RIGHT
        elif dx < 0 and dy > 0:
            return ACT_BOTTOM_LEFT
        else:
            return ACT_BOTTOM_LEFT


def _get_best_pass_target(game_state: Dict, player_idx: int, max_open_dist: float = 0.05) -> Optional[int]:
    """Find best pass target: open teammate, prioritizing forward positions.

    Args:
        max_open_dist: max distance to nearest opponent for teammate to be "open".
            Lower = stricter (default 0.05 for GRF scale).
    """
    player_pos = get_player_position(game_state, 'left', player_idx)

    best_idx = None
    best_score = -float('inf')

    for i in range(11):
        if i == player_idx:
            continue
        teammate_pos = get_player_position(game_state, 'left', i)

        # Check if teammate is open (no opponent very close)
        is_open = True
        for j in range(11):
            opp_pos = get_player_position(game_state, 'right', j)
            if get_distance(teammate_pos, opp_pos) < max_open_dist:
                is_open = False
                break

        if not is_open:
            continue

        # Score: forward progression weighted 3x, distance penalized
        forward_bonus = (teammate_pos[0] - player_pos[0]) * 3.0
        dist_penalty = get_distance(player_pos, teammate_pos) * 0.5
        score = forward_bonus - dist_penalty

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


def _face_direction(game_state: Dict, player_idx: int, target_pos: List[float]) -> int:
    """Return movement action to face toward target (GRF pass goes toward facing dir)."""
    player_pos = get_player_position(game_state, 'left', player_idx)
    dx = target_pos[0] - player_pos[0]
    dy = target_pos[1] - player_pos[1]

    # Determine primary direction
    if abs(dx) > abs(dy) * 1.5:
        return ACT_RIGHT if dx > 0 else ACT_LEFT
    elif abs(dy) > abs(dx) * 1.5:
        return ACT_BOTTOM if dy > 0 else ACT_TOP
    else:
        if dx > 0 and dy > 0:
            return ACT_BOTTOM_RIGHT
        elif dx > 0 and dy < 0:
            return ACT_TOP_RIGHT
        elif dx < 0 and dy > 0:
            return ACT_BOTTOM_LEFT
        else:
            return ACT_TOP_LEFT


class ExpertPolicy:
    """Rule-based expert policy for RCI reference.

    Returns the ideal action (a*) for each agent given their role,
    observation state, and active sub-goal.
    """

    def __init__(
        self,
        d_tackle: float = D_TACKLE,
        d_safe: float = D_SAFE,
        d_shoot: float = D_SHOOT,
    ):
        self.d_tackle = d_tackle
        self.d_safe = d_safe
        self.d_shoot = d_shoot
        # Track facing state for face-then-pass logic
        self._facing_target = None  # (player_idx, target_idx) or None
        self._facing_timestep = 0

    def get_ideal_action(
        self,
        game_state: Dict,
        player_idx: int,
        sub_goal: int,
        macro_strategy: int = STRATEGY_POSSESSION,
    ) -> int:
        """Get ideal action for player based on role and conditions.

        Implements the expert policy table from thesis (Tabel 7).
        """
        role = get_player_role(game_state, 'left', player_idx)
        player_pos = get_player_position(game_state, 'left', player_idx)
        ball_pos = get_ball_position(game_state)
        ball_x = ball_pos[0]
        dist_to_ball = get_distance(player_pos, ball_pos)

        # Check nearest opponent distance
        min_opp_dist = float('inf')
        for j in range(11):
            opp_pos = get_player_position(game_state, 'right', j)
            d = get_distance(player_pos, opp_pos)
            if d < min_opp_dist:
                min_opp_dist = d

        has_ball = ball_owned_by_us(game_state, team_id=0) and \
                   game_state.get('ball_owned_player', -1) == player_idx

        if role == ROLE_GK:
            return self._gk_action(game_state, player_pos, ball_pos, has_ball, dist_to_ball)
        elif role in (ROLE_CB, ROLE_LB, ROLE_RB):
            return self._defender_action(
                game_state, player_idx, player_pos, ball_pos, role,
                has_ball, dist_to_ball, min_opp_dist, sub_goal, macro_strategy,
            )
        elif role in (ROLE_DM, ROLE_CM, ROLE_AM):
            return self._midfielder_action(
                game_state, player_idx, player_pos, ball_pos, role,
                has_ball, dist_to_ball, min_opp_dist, sub_goal, macro_strategy,
            )
        elif role in (ROLE_LM, ROLE_RM, ROLE_CF):
            return self._attacker_action(
                game_state, player_idx, player_pos, ball_pos, role,
                has_ball, dist_to_ball, min_opp_dist, sub_goal, macro_strategy,
            )
        return ACT_IDLE

    def _gk_action(
        self, game_state: Dict, player_pos: List[float],
        ball_pos: List[float], has_ball: bool, dist_to_ball: float,
    ) -> int:
        """Goalkeeper: track ball, stay between ball and own goal."""
        if has_ball:
            target = _get_best_pass_target(game_state, 0)
            if target is not None:
                return ACT_SHORT_PASS
            return ACT_HIGH_PASS

        # Ball very close: direct save attempt
        if dist_to_ball < 0.1:
            return _move_toward(player_pos, ball_pos)

        # Hold the shot line: position between ball and goal center,
        # 0.06 in front of the goal line, y clamped to reachable range
        gx, gy = -1.0, 0.0
        dx = ball_pos[0] - gx
        dy = ball_pos[1] - gy
        n = (dx * dx + dy * dy) ** 0.5 or 1.0
        target = [gx + 0.06 * dx / n, max(-0.2, min(0.2, 0.06 * dy / n))]
        return _move_toward(player_pos, target)


    def _defender_action(
        self, game_state: Dict, player_idx: int,
        player_pos: List[float], ball_pos: List[float], role: int,
        has_ball: bool, dist_to_ball: float, min_opp_dist: float,
        sub_goal: int, macro_strategy: int,
    ) -> int:
        """Defender actions based on sub-goal and conditions."""
        if has_ball:
            # nearest opponent position for escape direction
            opp_pos, min_d = None, float('inf')
            for j in range(11):
                p = get_player_position(game_state, 'right', j)
                d = get_distance(player_pos, p)
                if d < min_d:
                    min_d, opp_pos = d, p

            # heavy pressure: escape away from opponent, biased forward
            if min_d < 0.12:
                esc = [player_pos[0] * 2 - opp_pos[0] + 0.15,
                       player_pos[1] * 2 - opp_pos[1]]
                return _move_toward(player_pos, esc)

            # find best pass target (strict open check)
            pass_target = _get_best_pass_target(game_state, player_idx, max_open_dist=0.03)

            # safe window: face forward first, then pass
            if min_d < 0.25:
                # If not facing forward (need to turn), move RIGHT first
                if self._facing_target != ('def', player_idx, 'pass'):
                    self._facing_target = ('def', player_idx, 'pass')
                    self._facing_timestep = 0
                    return ACT_RIGHT  # face forward

                self._facing_timestep += 1
                # After facing forward for 1 step, try to pass
                if self._facing_timestep >= 1:
                    if pass_target is not None:
                        return ACT_SHORT_PASS
                    else:
                        return _move_toward(player_pos, [player_pos[0] + 0.25, player_pos[1] * 0.5])

                return ACT_RIGHT  # keep facing forward

            # reset facing state when not passing
            self._facing_target = None
            self._facing_timestep = 0

            # open space: carry forward
            return _move_toward(player_pos, [player_pos[0] + 0.3, player_pos[1] * 0.5])

        # reset facing state when not having ball
        self._facing_target = None
        self._facing_timestep = 0

        if sub_goal in (SUBGOAL_ZONAL_MARKING, SUBGOAL_MAN_MARKING):
            # Swarm press: every defender closes the ball in own half
            if ball_pos[0] < 0.1 and dist_to_ball < 0.3:
                if dist_to_ball < 0.12:
                    return ACT_SLIDING
                return _move_toward(player_pos, ball_pos)
            # Position between ball and own goal
            target = [
                (ball_pos[0] + (-1.0)) / 2.0,  # midpoint
                ball_pos[1] * 0.8,
            ]
            # Sliding if opponent very close and in defensive zone
            if min_opp_dist < self.d_tackle and ball_pos[0] < 0.0:
                return ACT_SLIDING
            return _move_toward(player_pos, target)

        if sub_goal == SUBGOAL_BUILD_UP:
            # Move to support build-up
            target = [player_pos[0] - 0.05, player_pos[1]]
            return _move_toward(player_pos, target)

        if sub_goal == SUBGOAL_CLEARANCE:
            # Close down the ball in defensive zone
            if dist_to_ball < 0.12:
                return ACT_SLIDING
            return _move_toward(player_pos, ball_pos)

        return _move_toward(player_pos, ball_pos)

    def _midfielder_action(
        self, game_state: Dict, player_idx: int,
        player_pos: List[float], ball_pos: List[float], role: int,
        has_ball: bool, dist_to_ball: float, min_opp_dist: float,
        sub_goal: int, macro_strategy: int,
    ) -> int:
        """Midfielder actions based on sub-goal and conditions."""
        if has_ball:
            # nearest opponent position for escape direction
            opp_pos, min_d = None, float('inf')
            for j in range(11):
                p = get_player_position(game_state, 'right', j)
                d = get_distance(player_pos, p)
                if d < min_d:
                    min_d, opp_pos = d, p

            # heavy pressure: escape away from opponent, biased forward
            if min_d < 0.12:
                esc = [player_pos[0] * 2 - opp_pos[0] + 0.15,
                       player_pos[1] * 2 - opp_pos[1]]
                return _move_toward(player_pos, esc)

            # find best pass target (strict open check)
            pass_target = _get_best_pass_target(game_state, player_idx, max_open_dist=0.03)

            # through-ball: if CF is making forward run and we're in midfield, use high pass
            cf_pos = get_player_position(game_state, 'left', 10)  # CF is index 10
            cf_forward = cf_pos[0] > player_pos[0] + 0.15  # CF ahead by 0.15+
            in_midfield = player_pos[0] > -0.2  # in opposition half or approaching

            if cf_forward and in_midfield and min_d < 0.35:
                return ACT_HIGH_PASS  # through-ball to CF

            # safe window: face forward first, then pass
            if min_d < 0.25:
                # If not facing forward (need to turn), move RIGHT first
                # GRF: RIGHT = forward toward opponent goal
                if self._facing_target != ('mid', player_idx, 'pass'):
                    self._facing_target = ('mid', player_idx, 'pass')
                    self._facing_timestep = 0
                    return ACT_RIGHT  # face forward

                self._facing_timestep += 1
                # After facing forward for 1 step, try to pass
                if self._facing_timestep >= 1:
                    if pass_target is not None:
                        return ACT_SHORT_PASS
                    else:
                        return _move_toward(player_pos, [player_pos[0] + 0.25, player_pos[1] * 0.7])

                return ACT_RIGHT  # keep facing forward

            # reset facing state when not passing
            self._facing_target = None
            self._facing_timestep = 0

            # open space: advance toward attack
            return _move_toward(player_pos, [player_pos[0] + 0.3, player_pos[1]])

        # reset facing state when not having ball
        self._facing_target = None
        self._facing_timestep = 0

        if sub_goal == SUBGOAL_BUILD_UP:
            # Position to receive and distribute
            target = [ball_pos[0] - 0.05, ball_pos[1] * 0.5]
            return _move_toward(player_pos, target)

        if sub_goal == SUBGOAL_CLEARANCE:
            if dist_to_ball < 0.1:
                # Check for open teammate before blind clearance
                pass_target = _get_best_pass_target(game_state, player_idx, max_open_dist=0.03)
                if pass_target is not None:
                    return ACT_SHORT_PASS  # controlled pass
                return ACT_LONG_PASS  # no open target: desperate clearance
            return _move_toward(player_pos, ball_pos)

        if sub_goal == SUBGOAL_WING_ATTACK:
            # Move toward wing
            target = [ball_pos[0] + 0.1, 0.3 if player_pos[1] > 0 else -0.3]
            return _move_toward(player_pos, target)

        return _move_toward(player_pos, ball_pos)

    def _attacker_action(
        self, game_state: Dict, player_idx: int,
        player_pos: List[float], ball_pos: List[float], role: int,
        has_ball: bool, dist_to_ball: float, min_opp_dist: float,
        sub_goal: int, macro_strategy: int,
    ) -> int:
        """Attacker actions based on sub-goal and conditions."""
        if has_ball:
            goal_pos = [1.0, 0.0]
            dist_to_goal = get_distance(player_pos, goal_pos)

            # Shoot if in attacking zone with makeable distance
            if dist_to_goal < max(self.d_shoot, 0.6) and player_pos[0] > 0.5:
                return ACT_SHOT
            # Also shoot from very close, centered
            if dist_to_goal < 0.15 and abs(player_pos[1]) < 0.1:
                return ACT_SHOT

            # nearest opponent position for escape direction
            opp_pos, min_d = None, float('inf')
            for j in range(11):
                p = get_player_position(game_state, 'right', j)
                d = get_distance(player_pos, p)
                if d < min_d:
                    min_d, opp_pos = d, p

            # heavy pressure: escape away from opponent, biased forward
            if min_d < 0.12:
                esc = [player_pos[0] * 2 - opp_pos[0] + 0.15,
                       player_pos[1] * 2 - opp_pos[1]]
                return _move_toward(player_pos, esc)

            # find best pass target (strict open check)
            pass_target = _get_best_pass_target(game_state, player_idx, max_open_dist=0.03)

            # safe window: face forward first, then pass
            if min_d < 0.25:
                # If not facing forward (need to turn), move RIGHT first
                if self._facing_target != ('att', player_idx, 'pass'):
                    self._facing_target = ('att', player_idx, 'pass')
                    self._facing_timestep = 0
                    return ACT_RIGHT  # face forward

                self._facing_timestep += 1
                # After facing forward for 1 step, try to pass
                if self._facing_timestep >= 1:
                    if pass_target is not None:
                        return ACT_SHORT_PASS
                    else:
                        return _move_toward(player_pos, [player_pos[0] + 0.25, player_pos[1] * 0.7])

                return ACT_RIGHT  # keep facing forward

            # reset facing state when not passing
            self._facing_target = None
            self._facing_timestep = 0

            # open space: drive at goal
            return _move_toward(player_pos, goal_pos)

        # reset facing state when not having ball
        self._facing_target = None
        self._facing_timestep = 0

        if sub_goal == SUBGOAL_WING_ATTACK:
            if role in (ROLE_LM, ROLE_RM):
                wing_y = 0.35 if role == ROLE_LM else -0.35
                target_x = max(ball_pos[0] + 0.2, 0.5)
                target = [target_x, wing_y]
                if dist_to_ball < 0.2:
                    return ACT_SPRINT
                return _move_toward(player_pos, target)
            # CF: position ahead of ball between defenders
            target = [max(ball_pos[0] + 0.1, 0.4), ball_pos[1] * 0.5]
            return _move_toward(player_pos, target)

        if sub_goal == SUBGOAL_BUILD_UP:
            # CF: make forward run to receive through-ball
            if role == ROLE_CF:
                # Run into space ahead of ball
                target = [min(ball_pos[0] + 0.4, 0.8), ball_pos[1] * 0.3]
                return _move_toward(player_pos, target)
            target = [ball_pos[0] + 0.1, ball_pos[1]]
            return _move_toward(player_pos, target)

        return _move_toward(player_pos, ball_pos)


class ExpertPolicyEval(ExpertPolicy):
    """Detuned expert policy for RCI evaluation ONLY — never used in training.

    Uses perturbed thresholds to break circularity:
      - Training reward uses ExpertPolicyAllAgents (d_tackle=0.05, d_safe=0.15, d_shoot=0.30)
      - RCI evaluation uses this (d_tackle=0.06, d_safe=0.18, d_shoot=0.25)

    This ensures RCI measures generalization to a different reference,
    not just memorization of the training expert.
    """

    def __init__(self):
        super().__init__(
            d_tackle=0.06,   # +20% from training
            d_safe=0.18,     # +20% from training
            d_shoot=0.25,    # -17% from training
        )


class ExpertPolicyAllAgents:
    """Convenience wrapper: get ideal actions for all 11 agents."""

    def __init__(self, **kwargs):
        self.policy = ExpertPolicy(**kwargs)

    def get_ideal_actions(
        self,
        game_state: Dict,
        sub_goals: List[int],
        macro_strategy: int = STRATEGY_POSSESSION,
    ) -> List[int]:
        """Get ideal action for each of the 11 agents."""
        return [
            self.policy.get_ideal_action(game_state, i, sub_goals[i], macro_strategy)
            for i in range(len(sub_goals))
        ]
