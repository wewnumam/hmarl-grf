import random
import types
from typing import Any, Dict, List, Tuple

import gfootball.env as football_env
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration Constants ---
ENV_NAME = "11_vs_11_stochastic"
NUM_AGENTS = 11
ACTION_SPACE_SIZE = 19  # 0 to 18
LOG_DIR = "dumps"
DEFAULT_ROLE_ORDER = (
    "GK", "LCB", "RCB", "LB", "RB", "CDM", "LCM", "RCM", "LW", "CF", "RW"
)
ENGINE_ROLE_ORDER = (
    # "GK", "RW", "CF", "LB", "LCB", "RCB", "RB", "LCM", "CDM", "RCM", "LW"
    "GK", "LB", "LCB", "RCB", "RB", "LCM", "CDM", "RCM", "LW", "CF", "RW"
)
ENGINE_ROLE_IDS = (0, 7, 9, 2, 1, 1, 3, 5, 5, 5, 6)


def get_role_order(observation: Any) -> Tuple[str, ...]:
    """Convert the engine's current left-team role array to player roles."""
    role_values = tuple(int(role_id) for role_id in observation[0]["left_team_roles"])
    if role_values == ENGINE_ROLE_IDS:
        return ENGINE_ROLE_ORDER
    return DEFAULT_ROLE_ORDER


class SoccerAgent:
    """Represents an individual agent (player) in the soccer simulation."""

    # The attached role rules specify gaps but not their numerical values.
    DEFENDER_GAP = np.array([0.1, 0.0])
    MIDFIELDER_GAP = np.array([0.1, 0.0])
    MIDFIELD_SIDE_GAP = 0.1

    def __init__(self, agent_id: int, role_order: Tuple[str, ...] = DEFAULT_ROLE_ORDER):
        self.agent_id = agent_id
        self.role_order = role_order
        self.role = role_order[agent_id]

    def update_role_order(self, role_order: Tuple[str, ...]) -> None:
        """Update the role assigned to this player for the current step."""
        previous_role = self.role
        self.role_order = role_order
        self.role = role_order[self.agent_id]
        if self.role != previous_role:
            print(
                f"Agent {self.agent_id:<2} role changed: "
                f"{previous_role} -> {self.role}"
            )

    def _off_possession_target(self, observation: Any) -> np.ndarray:
        """Return the role reference point used when this player lacks the ball."""
        state = observation[self.agent_id]
        left_team = np.asarray(state["left_team"], dtype=float)
        right_team = np.asarray(state["right_team"], dtype=float)
        ball_position = np.asarray(state["ball"][:2], dtype=float)
        role_ids = {role: role_id for role_id, role in enumerate(self.role_order)}

        lcb = left_team[role_ids["LCB"]][:2]
        rcb = left_team[role_ids["RCB"]][:2]
        cdm = left_team[role_ids["CDM"]][:2]
        opponent_cf = right_team[role_ids["CF"]][:2]
        opponent_rcb = right_team[role_ids["RCB"]][:2]
        opponent_lcb = right_team[role_ids["LCB"]][:2]
        role = self.role

        if role == "LCB":
            return opponent_cf + self.DEFENDER_GAP
        if role == "RCB":
            return opponent_cf - self.DEFENDER_GAP
        if role == "LB":
            return np.array([lcb[0], right_team[role_ids["RW"]][1]])
        if role == "RB":
            return np.array([rcb[0], right_team[role_ids["LW"]][1]])
        if role == "CDM":
            return (ball_position + opponent_cf) / 2.0
        if role == "LCM":
            return cdm + self.MIDFIELDER_GAP + np.array(
                [0.0, -self.MIDFIELD_SIDE_GAP]
            )
        if role == "RCM":
            return cdm + self.MIDFIELDER_GAP + np.array(
                [0.0, self.MIDFIELD_SIDE_GAP]
            )
        if role == "LW":
            return np.array([left_team[role_ids["CF"]][0], -0.42])
        if role == "CF":
            return (opponent_rcb + opponent_lcb) / 2.0
        if role == "RW":
            return np.array([left_team[role_ids["CF"]][0], 0.42])

        return np.array([-0.92, 0.0])

    def decide_action(self, observation: Any) -> int:
        """
        Role-based policy for the 4-3-3 formation.

        - Moves toward the opponent's goal while in possession.
        - Uses role-specific reference points while off possession.
        """

        # Action mapping
        moves = {
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

        role = self.role

        # Goalkeeper stays in position
        if role == "GK":
            return 0

        # Extract positions
        state = observation[self.agent_id]
        role_ids = {current_role: role_id for role_id, current_role in enumerate(self.role_order)}
        agent_position = np.array(state["left_team"][role_ids[role]][:2])

        # Check ball possession
        is_ball_owned_by_agent = (
            state["ball_owned_player"] == self.agent_id
        )

        print(
            f"Agent {self.agent_id:<2} | "
            f"Role: {role:<3} | "
            f"Engine role: {state['left_team_roles'][self.agent_id]:<2} | "
            f"Ball Owned: {is_ball_owned_by_agent} | "
        )

        # =====================================================
        # 1. Ball possession: move toward opponent's goal
        # =====================================================

        if state["ball_owned_team"] == 0 and is_ball_owned_by_agent:
                return 5

        # Off possession: execute the role-specific positioning rule.
        target_position = self._off_possession_target(observation)
        dx = int(np.sign(target_position[0] - agent_position[0]))
        dy = int(np.sign(target_position[1] - agent_position[1]))

        return moves.get((dx, dy), 0)


class SoccerMatch:
    """
    Model class that manages the soccer environment and agents.
    Follows Agent-Based Modeling (ABM) principles.
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
        """Initializes and patches the football environment."""
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
        """Patches GRF environment for Gymnasium compatibility."""
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
        """Resets the match to the initial state."""
        self.current_obs, info = self.env.reset()
        self.step_count = 0
        self.role_order = get_role_order(self.current_obs)
        for agent in self.agents:
            agent.update_role_order(self.role_order)
        return self.current_obs, info

    def plot_positions(self, observation: Any, step: int) -> None:
        """Save team positions, ball position, role connections, and each agent target."""
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

        role_ids = {role: role_id for role_id, role in enumerate(role_order)}
        left = {role: left_positions[role_ids[role]][:2] for role in role_order}
        right = {role: right_positions[role_ids[role]][:2] for role in role_order}

        def draw_connection(
            start: np.ndarray,
            end: np.ndarray,
            label: str,
            color: str,
            linestyle: str = "-",
        ) -> None:
            ax.plot(
                [start[0], end[0]], [start[1], end[1]],
                color=color, linestyle=linestyle, linewidth=1.4,
                alpha=0.85, label=label, zorder=1,
            )

        # Defensive and attacking reference connections.
        draw_connection(left["LB"], left["LCB"], "LB-LCB", "#90caf9")
        draw_connection(left["LB"], right["RW"], "LB-right RW", "#64b5f6", "--")
        draw_connection(left["RB"], left["RCB"], "RB-RCB", "#90caf9")
        draw_connection(left["RB"], right["LW"], "RB-right LW", "#64b5f6", "--")
        draw_connection(left["RCB"], right["CF"], "RCB-right CF", "#ef9a9a", "--")
        draw_connection(left["LCB"], right["CF"], "LCB-right CF", "#ef9a9a", "--")
        draw_connection(left["RW"], left["CF"], "RW-CF", "#ffcc80")
        draw_connection(left["LW"], left["CF"], "LW-CF", "#ffcc80")
        draw_connection(ball_position, right["CF"], "ball-right CF", "#fff176", ":")
        draw_connection(right["LCB"], right["RCB"], "right LCB-right RCB", "#ce93d8")

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
                    f'{observation[0]["left_team_roles"][role_order.index(role_name)]}: {role_name}',
                    (position[0], position[1]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=8, color="black",
                    bbox={"facecolor": "white", "alpha": 0.75, "pad": 1},
                )

        # Draw the role-based target position for each controlled left-team agent.
        for agent in self.agents:
            agent.update_role_order(role_order)
            target_position = agent._off_possession_target(observation)
            current_role_position = np.asarray(
                observation[agent.agent_id]["left_team"][role_ids[agent.role]][:2],
                dtype=float,
            )
            ax.plot(
                [current_role_position[0], target_position[0]],
                [current_role_position[1], target_position[1]],
                color="#f5d76e",
                linestyle=":",
                linewidth=1.2,
                alpha=0.9,
                zorder=2,
            )
            ax.scatter(
                target_position[0], target_position[1],
                c="#ffeb3b",
                edgecolors="black",
                s=55,
                marker="x",
                label=f"{agent.role} target" if agent.agent_id == 0 else None,
                zorder=5,
            )
            ax.annotate(
                f"{agent.role} target",
                (target_position[0], target_position[1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                color="black",
                bbox={"facecolor": "#fff7d6", "alpha": 0.8, "pad": 1},
            )

        ax.set_title(f"Team Positions - Step {step}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="upper right", ncol=2)
        ax.grid(color="white", alpha=0.2)
        fig.tight_layout()
        output_path = f"{LOG_DIR}/positions_step_{step}.png"
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        print(f"Saved starting-position plot: {output_path}")

    def step(self) -> Tuple[Any, List[float], bool, Dict[str, Any]]:
        """Executes a single step in the simulation."""
        self.step_count += 1
        
        # Each agent decides its own action based on the observation
        self.role_order = get_role_order(self.current_obs)
        for agent in self.agents:
            agent.update_role_order(self.role_order)

        print(f"Step {self.step_count:4d} | Agents deciding actions...")
        print(f"Left team roles: {self.current_obs[0]['left_team_roles']}")
        print(f"Right team roles: {self.current_obs[0]['right_team_roles']}")
        actions = [agent.decide_action(self.current_obs) for agent in self.agents]
        
        self.current_obs, rewards, done, info = self.env.step(actions)
        if self.step_count % 100 == 0:
            self.plot_positions(self.current_obs, self.step_count)
        return self.current_obs, rewards, done, info

    def render(self):
        """Renders the current state of the match."""
        self.env.render()

    def close(self):
        """Closes the environment."""
        self.env.close()

    def run(self, max_steps: int = 1000):
        """Runs the complete match simulation."""
        obs, info = self.reset()
        done = False

        print(f"Starting match: {ENV_NAME} with {self.num_agents} agents.")
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
    match.run(max_steps=3000)
