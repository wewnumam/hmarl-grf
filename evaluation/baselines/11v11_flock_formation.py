import random
import types
from typing import Any, Dict, List, Optional, Tuple

import gfootball.env as football_env
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration Constants ---
ENV_NAME = "11_vs_11_easy_stochastic"
NUM_AGENTS = 11
ACTION_SPACE_SIZE = 19  # 0 to 18
LOG_DIR = "dumps"
DEFAULT_ROLE_ORDER = (
    "GK", "LCB", "RCB", "LB", "RB", "CDM", "LCM", "RCM", "LW", "CF", "RW"
)
ENGINE_ROLE_ORDER = (
    "GK", "LB", "LCB", "RCB", "RB", "LCM", "CDM", "RCM", "LW", "CF", "RW"
)
ENGINE_ROLE_IDS = (0, 7, 9, 2, 1, 1, 3, 5, 5, 5, 6)

# --- GRF Action Mapping (dx, dy) -> discrete action ---
MOVES = {
    (-1, -1): 2,  # Up-Left
    (-1,  1): 8,  # Down-Left
    (-1,  0): 1,  # Left
    ( 1, -1): 4,  # Up-Right
    ( 1,  1): 6,  # Down-Right
    ( 1,  0): 5,  # Right
    ( 0, -1): 3,  # Up
    ( 0,  1): 7,  # Down
    ( 0,  0): 0,  # Idle
}
# GRF action ids (see hmarl/env.py ACTION_NAMES)
IDLE_ACTION = 0
PASS_ACTION = 11        # short_pass
LONG_PASS_ACTION = 9    # long_pass
HIGH_PASS_ACTION = 10   # high_pass
SHOT_ACTION = 12        # shot
SLIDING_ACTION = 16     # sliding tackle

# --- 4-3-3 Formation Anchors (normalized GRF coords: x∈[-1,1], y∈[-0.42,0.42]) ---
FORMATION_433 = {
    "GK":  (-0.92,  0.00),
    "LB":  (-0.50,  0.30),
    "LCB": (-0.55,  0.10),
    "RCB": (-0.55, -0.10),
    "RB":  (-0.50, -0.30),
    "CDM": (-0.20,  0.00),
    "LCM": (-0.15,  0.15),
    "RCM": (-0.15, -0.15),
    "LW":  ( 0.20,  0.32),
    "CF":  ( 0.30,  0.00),
    "RW":  ( 0.20, -0.32),
}

# Flocking edges: adjacent teammates within each unit
LINE_NEIGHBORS = {
    "LB":  ("LCB",),
    "LCB": ("LB", "RCB"),
    "RCB": ("LCB", "RB"),
    "RB":  ("RCB",),
    "LCM": ("CDM", "LW"),
    "CDM": ("LCM", "RCM"),
    "RCM": ("CDM", "RW"),
    "LW":  ("LCM", "CF"),
    "CF":  ("LW", "RW"),
    "RW":  ("CF", "RCM"),
}

# Unit membership for line-alignment term
LINES = {
    "LB": "Def", "LCB": "Def", "RCB": "Def", "RB": "Def",
    "LCM": "Mid", "CDM": "Mid", "RCM": "Mid",
    "LW": "Att", "CF": "Att", "RW": "Att",
}

# Ideal flocking distance per edge, derived from formation anchors
D_IDEAL = {
    (r, n): float(np.linalg.norm(np.array(FORMATION_433[r]) - np.array(FORMATION_433[n])))
    for r, nbs in LINE_NEIGHBORS.items() for n in nbs
}

# Composite-formation hyperparameters
W_POS = 1.0
D_MIN = 0.05
FORM_SHIFT = 0.5

# --- Flock-formation controller parameters ---
W_FLOCK = 0.5
W_ALIGN = 0.3
W_SPACING = 0.2
W_PENALTY = 2.0
ATTACK_SHIFT = 0.15   # attacking line pushes forward
DEFEND_SHIFT = 0.15   # defensive line drops deeper
BOX_X = 0.8           # opponent penalty-box entry threshold
OUR_BOX_X = -0.8      # own penalty-box edge; GK comes off line inside this
TACKLE_RANGE = 0.05
SHOT_PROB = 0.7
TACKLE_PROB = 0.5
# Passing
SHORT_PASS_RANGE = 0.35   # beyond this a pass becomes long_pass
PASS_CLEARANCE = 0.12     # min distance from pass lane to nearest opponent
PRESSURE_DIST = 0.15      # opponent closer than this -> pass/escape, not dribble
FACING_COS = 0.707        # cos(45°): pass goes where the player faces
GK_HOME_TOL = 0.05        # GK returns to shot line when further than this
PASS_PROB = 0.6           # probability of choosing pass over dribble when lane open
PASS_DIR_BIAS = 0.5       # pass direction preference: 0.5=straight ahead, 0/1=±90°

ACTION_DIRS = {action: delta for delta, action in MOVES.items()}
PASS_ACTIONS = {LONG_PASS_ACTION, HIGH_PASS_ACTION, PASS_ACTION}


def get_role_order(observation: Any) -> Tuple[str, ...]:
    """Convert the engine's current left-team role array to player roles."""
    role_values = tuple(int(role_id) for role_id in observation[0]["left_team_roles"])
    if role_values == ENGINE_ROLE_IDS:
        return ENGINE_ROLE_ORDER
    return DEFAULT_ROLE_ORDER


def goalkeeper_home(observation: Any) -> np.ndarray:
    """GK default position: goal line depth, y tracks ball clamped to goal width."""
    ball = np.asarray(observation[0]["ball"][:2], dtype=float)
    return np.array([-0.94, float(np.clip(ball[1], -0.15, 0.15))])


class SoccerAgent:
    """Individual agent (player). Movement follows a flock-formation controller."""

    def __init__(self, agent_id: int, role_order: Tuple[str, ...] = DEFAULT_ROLE_ORDER):
        self.agent_id = agent_id
        self.role_order = role_order
        self.role = role_order[agent_id]
        self.facing = np.array([1.0, 0.0])  # ponytail: engine-facing approximation = last movement dir

    def update_role_order(self, role_order: Tuple[str, ...]) -> None:
        self.role_order = role_order
        self.role = role_order[self.agent_id]

    # ------------------------------------------------------------------ State
    def position(self, observation: Any) -> np.ndarray:
        state = observation[self.agent_id]
        return np.asarray(state["left_team"][self.agent_id][:2], dtype=float)

    def ball_position(self, observation: Any) -> np.ndarray:
        return np.asarray(observation[0]["ball"][:2], dtype=float)

    def distance_to_ball(self, observation: Any) -> float:
        return float(np.linalg.norm(self.position(observation) - self.ball_position(observation)))

    # ------------------------------------------------- Ball-carrier decision tree
    def is_inside_box(self, observation: Any) -> bool:
        return bool(self.position(observation)[0] > BOX_X)

    def nearest_opponent(self, observation: Any) -> float:
        pos = self.position(observation)
        right = np.asarray(observation[0]["right_team"], dtype=float)
        return float(np.min(np.linalg.norm(right - pos, axis=1)))

    def _lane_clearance(self, observation: Any, a: np.ndarray, b: np.ndarray) -> float:
        """Min distance from opponents to segment a→b (0 if projection falls outside)."""
        right = np.asarray(observation[0]["right_team"], dtype=float)
        ab = b - a
        denom = float(ab @ ab)
        if denom < 1e-12:
            return 0.0
        clearance = np.inf
        for opp in right:
            t = float(np.clip((opp - a) @ ab / denom, 0.0, 1.0))
            clearance = min(clearance, float(np.linalg.norm(a + t * ab - opp)))
        return clearance

    def _move_action(self, delta: np.ndarray) -> int:
        return MOVES.get((int(np.sign(delta[0])), int(np.sign(delta[1]))), IDLE_ACTION)

    def _face_toward(self, observation: Any, target: np.ndarray) -> int:
        """Turn/face the target first (long_pass and shot go where the player faces)."""
        dir_delta = target - self.position(observation)
        if float(dir_delta @ self.facing) / (np.linalg.norm(dir_delta) + 1e-9) >= FACING_COS:
            return IDLE_ACTION
        return self._move_action(dir_delta)

    def _open_teammate(self, observation: Any) -> Optional[np.ndarray]:
        """Best forward pass target: clear lane, closest to preferred pass direction."""
        pos = self.position(observation)
        left = np.asarray(observation[0]["left_team"], dtype=float)
        # preferred direction: PASS_DIR_BIAS 0.5 -> straight ahead (+x), 0/1 -> flanks
        pref = np.array([1.0, (PASS_DIR_BIAS - 0.5) * 2.0])
        pref /= np.linalg.norm(pref)
        candidates = []
        for j, mate in enumerate(left):
            if j == self.agent_id or mate[0] <= pos[0]:
                continue
            if self._lane_clearance(observation, pos, mate) < PASS_CLEARANCE:
                continue
            candidates.append(mate[:2])
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda m: float((m - pos) @ pref / (np.linalg.norm(m - pos) + 1e-9)),
        )

    def attempt_shot(self, observation: Any) -> int:
        """Shot goes where the player faces: face the goal, then shoot."""
        face = self._face_toward(observation, np.array([1.0, 0.0]))
        if face != IDLE_ACTION:
            return face
        if random.random() < SHOT_PROB:
            return SHOT_ACTION
        return self.dribble_towards_goal(observation)

    def pass_ball(self, observation: Any) -> int:
        """Gate on PASS_PROB; face the target (passes travel facing dir), then pass."""
        if random.random() >= PASS_PROB:
            return self.dribble_towards_goal(observation)
        target = self._open_teammate(observation)
        if target is None:
            return self.dribble_towards_goal(observation)
        face = self._face_toward(observation, target)
        if face != IDLE_ACTION:
            return face
        dist = float(np.linalg.norm(target - self.position(observation)))
        return LONG_PASS_ACTION if dist > SHORT_PASS_RANGE else PASS_ACTION

    def dribble_towards_goal(self, observation: Any) -> int:
        pos = self.position(observation)
        goal = np.array([1.0, 0.0])
        delta = goal - pos
        return MOVES.get((int(np.sign(delta[0])), int(np.sign(delta[1]))), IDLE_ACTION)

    def escape_pressure(self, observation: Any) -> int:
        """Move away from nearest opponent, forward-biased (expert ladder step 2)."""
        pos = self.position(observation)
        right = np.asarray(observation[0]["right_team"], dtype=float)
        nearest = right[np.argmin(np.linalg.norm(right - pos, axis=1))]
        escape = pos - nearest
        escape[0] = max(escape[0], 0.05)
        return self._move_action(escape)

    # ------------------------------------------------- Off-ball defensive actions
    def attempt_tackle(self, observation: Any) -> int:
        if random.random() < TACKLE_PROB:
            return SLIDING_ACTION
        return self.press_ball(observation)

    def press_ball(self, observation: Any) -> int:
        delta = self.ball_position(observation) - self.position(observation)
        return MOVES.get((int(np.sign(delta[0])), int(np.sign(delta[1]))), IDLE_ACTION)

    def goalkeeper_action(self, observation: Any) -> int:
        """Off possession: hold the shot line; only come off it for a ball in own box."""
        if self.ball_position(observation)[0] <= OUR_BOX_X:
            return self.press_ball(observation)
        home = goalkeeper_home(observation)
        dist = float(np.linalg.norm(self.position(observation) - home))
        if dist <= GK_HOME_TOL:
            return IDLE_ACTION
        return self._move_action(home - self.position(observation))

    # ------------------------------------------------- Flock-formation controller
    def maintain_formation(
        self,
        observation: Any,
        target_pos: np.ndarray,
        flock_weight: float = W_FLOCK,
        alignment_weight: float = W_ALIGN,
        spacing_weight: float = W_SPACING,
    ) -> int:
        force = self.formation_force(observation, target_pos, flock_weight, alignment_weight, spacing_weight)
        return MOVES.get((int(np.sign(force[0])), int(np.sign(force[1]))), IDLE_ACTION)

    def formation_force(
        self,
        observation: Any,
        target_pos: np.ndarray,
        flock_weight: float = W_FLOCK,
        alignment_weight: float = W_ALIGN,
        spacing_weight: float = W_SPACING,
    ) -> np.ndarray:
        """Composite force: position + flock + alignment + spacing + bounds penalty."""
        state = observation[self.agent_id]
        pos = self.position(observation)
        left = np.asarray(state["left_team"], dtype=float)
        role_ids = {r: i for i, r in enumerate(self.role_order)}

        anchor = np.clip(np.asarray(target_pos, dtype=float), [-1.0, -0.42], [1.0, 0.42])
        force = W_POS * (anchor - pos)

        for nb in LINE_NEIGHBORS.get(self.role, ()):
            nb_pos = left[role_ids[nb]][:2]
            delta = pos - nb_pos
            dist = np.linalg.norm(delta)
            if dist > 1e-8:
                force += flock_weight * (dist - D_IDEAL[(self.role, nb)]) * delta / dist

        line = LINES.get(self.role)
        if line is not None:
            mean_x = np.mean([left[role_ids[r]][0] for r, ln in LINES.items() if ln == line])
            force[0] += alignment_weight * (mean_x - pos[0])

        for j in range(len(left)):
            if j == self.agent_id:
                continue
            delta = pos - left[j][:2]
            dist = np.linalg.norm(delta)
            if 1e-8 < dist < D_MIN:
                force -= spacing_weight * (D_MIN - dist) * delta / dist

        for axis, limit in ((0, 1.0), (1, 0.42)):
            excess = abs(pos[axis]) - limit
            if excess > 0:
                force[axis] -= W_PENALTY * excess * np.sign(pos[axis])

        return force

    def formation_target(self, observation: Any, attacking: bool) -> np.ndarray:
        """4-3-3 anchor for this role, shifted for phase of play and ball position."""
        role = self.role
        if role == "GK":
            return goalkeeper_home(observation)
        base = np.array(FORMATION_433[role], dtype=float)
        shift = ATTACK_SHIFT if attacking else -DEFEND_SHIFT
        base = base + np.array([shift, 0.0])
        if not attacking:
            base[1] *= 0.8  # compact defensive width
        ball = self.ball_position(observation)
        scale = FORM_SHIFT if attacking else FORM_SHIFT * 0.5
        return np.clip(base + ball * scale, [-1.0, -0.42], [1.0, 0.42])

    def _off_possession_target(self, observation: Any, attacking: bool = False) -> np.ndarray:
        """Effective target point for plotting: current position + formation force."""
        target = self.formation_target(observation, attacking)
        return self.position(observation) + self.formation_force(observation, target)


class SoccerMatch:
    """
    Team-level flock-formation controller driving the soccer environment.
    Follows the possession/defense pseudocode each step.
    """

    def __init__(self, env_name: str = ENV_NAME, num_agents: int = NUM_AGENTS, render: bool = True, dump: bool = False):
        self.should_render = render
        self.dump = dump
        self.num_agents = num_agents
        self.env = self._create_and_patch_env(env_name, num_agents, self.should_render, self.dump)
        self.agents = [SoccerAgent(i) for i in range(num_agents)]
        self.role_order = DEFAULT_ROLE_ORDER
        self.current_obs = None
        self.step_count = 0

    def _create_and_patch_env(self, env_name: str, num_agents: int, render: bool, dump: bool = False) -> Any:
        env = football_env.create_environment(
            env_name=env_name,
            representation="raw",
            render=render,
            number_of_left_players_agent_controls=num_agents,
            write_full_episode_dumps=dump,
            logdir=LOG_DIR,
        )
        return self._patch_grf_env(env)

    def _patch_grf_env(self, env: Any) -> Any:
        orig_reset = env.reset
        orig_step = env.step

        def reset_wrapper(self_env, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
            try:
                result = orig_reset(*args, **kwargs)
            except TypeError:
                result = orig_reset()
            if isinstance(result, tuple):
                return result
            return result, {}

        def step_wrapper(self_env, *args, **kwargs) -> Tuple[Any, float, bool, Dict[str, Any]]:
            result = orig_step(*args, **kwargs)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
                return obs, reward, done, info
            return result

        env.reset = types.MethodType(reset_wrapper, env)
        env.step = types.MethodType(step_wrapper, env)
        return env

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        self.current_obs, info = self.env.reset()
        self.step_count = 0
        self.role_order = get_role_order(self.current_obs)
        for agent in self.agents:
            agent.update_role_order(self.role_order)
        return self.current_obs, info

    # ------------------------------------------------------------- Team decisions
    def get_ball_carrier(self, observation: Any) -> Optional[int]:
        state = observation[0]
        if state["ball_owned_team"] == 0 and state["ball_owned_player"] >= 0:
            return int(state["ball_owned_player"])
        return None

    def get_closest_defender(self, observation: Any) -> int:
        outfield = [a for a in self.agents if a.role != "GK"]
        return min(outfield, key=lambda a: a.distance_to_ball(observation)).agent_id

    def _decide_team_actions(self, observation: Any) -> List[int]:
        """Implements the possession/defense pseudocode for one step."""
        actions = [IDLE_ACTION] * self.num_agents
        carrier = self.get_ball_carrier(observation)

        if carrier is not None:
            # --- Team has possession ---
            for agent in self.agents:
                if agent.agent_id == carrier and agent.role == "GK":
                    # GK distribution: escape pressure or pass upfield
                    if agent.nearest_opponent(observation) <= PRESSURE_DIST:
                        actions[agent.agent_id] = agent.escape_pressure(observation)
                    else:
                        actions[agent.agent_id] = agent.pass_ball(observation)
                elif agent.agent_id == carrier:
                    # Ball Carrier Decision Tree (expert ladder): shot -> escape -> pass -> dribble
                    if agent.is_inside_box(observation):
                        actions[agent.agent_id] = agent.attempt_shot(observation)
                    elif agent.nearest_opponent(observation) <= PRESSURE_DIST:
                        actions[agent.agent_id] = agent.escape_pressure(observation)
                    elif agent._open_teammate(observation) is None:
                        # every forward lane blocked -> dribble to make one
                        actions[agent.agent_id] = agent.dribble_towards_goal(observation)
                    else:
                        actions[agent.agent_id] = agent.pass_ball(observation)
                elif agent.role == "GK":
                    actions[agent.agent_id] = agent.goalkeeper_action(observation)
                else:
                    # Off-ball Attacking Movement
                    target = agent.formation_target(observation, attacking=True)
                    actions[agent.agent_id] = agent.maintain_formation(
                        observation, target, W_FLOCK, W_ALIGN, W_SPACING
                    )
        else:
            # --- Defense / Out of Possession ---
            closest = self.get_closest_defender(observation)
            for agent in self.agents:
                if agent.role == "GK":
                    actions[agent.agent_id] = agent.goalkeeper_action(observation)
                elif agent.agent_id == closest:
                    # Pressure & Tackle Decision Tree
                    if agent.distance_to_ball(observation) <= TACKLE_RANGE:
                        actions[agent.agent_id] = agent.attempt_tackle(observation)
                    else:
                        actions[agent.agent_id] = agent.press_ball(observation)
                else:
                    # Off-ball Defensive Movement
                    target = agent.formation_target(observation, attacking=False)
                    actions[agent.agent_id] = agent.maintain_formation(
                        observation, target, W_FLOCK, W_ALIGN, W_SPACING
                    )

        # Record facing from executed movement (pass/shot travel facing direction)
        for agent in self.agents:
            delta = ACTION_DIRS.get(actions[agent.agent_id])
            if delta and (delta[0] or delta[1]):
                agent.facing = np.array(delta, dtype=float) / np.linalg.norm(delta)
        return actions

    # -------------------------------------------------------------------- Plot
    def plot_positions(self, observation: Any, step: int) -> None:
        state = observation[0]
        role_order = get_role_order(observation)
        left_positions = np.asarray(state["left_team"], dtype=float)
        right_positions = np.asarray(state["right_team"], dtype=float)
        ball_position = np.asarray(state["ball"][:2], dtype=float)
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.set_facecolor("#2f7d4a")
        fig.patch.set_facecolor("white")
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-0.42, 0.42)
        ax.axvline(0.0, color="white", linewidth=1)
        ax.plot([-1, -1, 1, 1, -1], [-0.42, 0.42, 0.42, -0.42, -0.42],
                color="white", linewidth=1)
        circle = plt.Circle((0, 0), 0.105, fill=False, color="white")
        ax.add_patch(circle)

        ax.scatter(
            ball_position[0], ball_position[1],
            c="#fdd835", edgecolors="black", s=140, marker="*",
            label="Ball", zorder=4,
        )

        for positions, color, team_name, marker in (
            (left_positions, "#1976d2", "Left team", "o"),
            (right_positions, "#d32f2f", "Right team", "s"),
        ):
            ax.scatter(
                positions[:, 0], positions[:, 1],
                c=color, edgecolors="white", s=100, marker=marker,
                label=team_name, zorder=3,
            )
            for role_name, position in zip(role_order, positions):
                ax.annotate(
                    f'{role_name}',
                    (position[0], position[1]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=8, color="black",
                    bbox={"facecolor": "white", "alpha": 0.75, "pad": 1},
                )

        attacking = self.get_ball_carrier(observation) is not None
        for agent in self.agents:
            agent.update_role_order(role_order)
            target_position = agent._off_possession_target(observation, attacking=attacking)
            current_pos = agent.position(observation)
            ax.plot(
                [current_pos[0], target_position[0]],
                [current_pos[1], target_position[1]],
                color="#f5d76e", linestyle=":", linewidth=1.2, alpha=0.9, zorder=2,
            )
            ax.scatter(
                target_position[0], target_position[1],
                c="#ffeb3b", edgecolors="black", s=55, marker="x",
                label=f"{agent.role} target" if agent.agent_id == 0 else None,
                zorder=5,
            )

        phase = "Attacking" if attacking else "Defensive"
        ax.set_title(f"Flock Formation ({phase}) - Step {step}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="upper right", ncol=2)
        ax.grid(color="white", alpha=0.2)
        fig.tight_layout()
        output_path = f"{LOG_DIR}/positions_step_{step}.png"
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        print(f"Saved position plot: {output_path}")

    # -------------------------------------------------------------------- Loop
    def step(self) -> Tuple[Any, List[float], bool, Dict[str, Any]]:
        self.step_count += 1
        self.role_order = get_role_order(self.current_obs)
        for agent in self.agents:
            agent.update_role_order(self.role_order)

        actions = self._decide_team_actions(self.current_obs)
        self.current_obs, rewards, done, info = self.env.step(actions)
        if self.step_count % 100 == 0:
            self.plot_positions(self.current_obs, self.step_count)
        return self.current_obs, rewards, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def run(self, max_steps: int = 1000):
        obs, info = self.reset()
        done = False

        print(f"Starting match: {ENV_NAME} with {self.num_agents} agents (flock formation).")
        try:
            while not done and self.step_count < max_steps:
                obs, rewards, done, info = self.step()
                if self.should_render:
                    self.render()

                if any(r != 0 for r in rewards):
                    print(f"Step {self.step_count:4d} | Rewards: {rewards}")

        except KeyboardInterrupt:
            print("\nMatch interrupted by user.")
        finally:
            self.close()
            print(f"Match ended after {self.step_count} steps.")


if __name__ == "__main__":
    match = SoccerMatch(render=False, dump=True)
    match.run(max_steps=2048)
