"""Small-scale hierarchical PPO for GRF academy scenarios.

Two learned levels (both PPO, per user choice):
- High-level PPO: picks a subgoal every ``period`` low-level steps from a
  compact team-state observation. Reward = raw game reward over the period
  + subgoal completion bonus + subgoal switch penalty.
- Low-level PPO: shared across controlled agents, joint action space,
  observation = per-agent 115-dim vector + subgoal one-hot. Reward = raw
  game reward + dense subgoal-progress shaping.

Training alternates: low-level learn (subgoals sampled from current high
policy) -> high-level learn (low policy frozen, deterministic) -> repeat.

Scenarios (small scale):
- academy_run_to_score: 1 attacker vs keeper, subgoals {dribble, shoot}
- academy_pass_and_shoot_with_keeper: 2 attackers + keeper + GK,
  subgoals {dribble, shoot, pass}

ponytail: alternating block-coordinate training, not joint; nearest-2
opponents in high obs; PASS bonus detects any controlled-player possession
change. Upgrade path: joint fine-tuning phase / full game-state high obs.

Run inside Docker only (gfootball + SB3 1.3.0 + old gym, Python 3.6):
    docker exec gfootball-dev bash -c \\
        "cd /gfootball && python evaluation/baselines/academy_hierarchical.py \\
         --env-name academy_pass_and_shoot_with_keeper"
Smoke test (tiny budget, ~2 min):
    ... academy_hierarchical.py --smoke
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import gym  # old gym: SB3 1.3.0 isinstance-checks spaces against these classes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from hmarl.env import create_raw_env, extract_game_state
from hmarl.utils import ACTION_SPACE_SIZE, OBS_DIM, extract_obs_vector, set_seed

# --- Scenario registry (keys are CLI names; GRF full names below) ---
SCENARIOS = {
    "run_to_score": {
        "grf_name": "academy_run_to_score",
        "num_agents": 1,
        "subgoals": ["dribble", "shoot"],
    },
    "pass_and_shoot_with_keeper": {
        "grf_name": "academy_pass_and_shoot_with_keeper",
        "num_agents": 2,
        "subgoals": ["dribble", "shoot", "pass"],
    },
}

EPISODE_CAP = 20480
BONUS_SHOOT_GOAL = 0.5
BONUS_PASS_COMPLETE = 0.5
BONUS_DRIBBLE_DX = 0.1     # min ball x-progress to earn dribble bonus
BONUS_DRIBBLE_SCALE = 0.3  # max dribble bonus
LOG_DIR = "dumps"


# ---------------------------------------------------------------------------
# Observation builders
# ---------------------------------------------------------------------------
def high_obs_dim(num_agents: int) -> int:
    """ball(3) + possession(2) + agents(4n) + nearest opponents(4) + steps_left(1)."""
    return 10 + 4 * num_agents


def build_high_obs(game_state, num_agents: int) -> np.ndarray:
    """Compact team-level observation for the high-level PPO."""
    ball = game_state.get("ball", [0.0, 0.0, 0.0])
    feats = [
        float(ball[0]), float(ball[1]), float(ball[2]),
        float(game_state.get("ball_owned_team", -1)),
        float(game_state.get("ball_owned_player", -1)),
    ]
    left = game_state.get("left_team", [[0.0, 0.0]] * 11)
    left_dir = game_state.get("left_team_direction", [[0.0, 0.0]] * 11)
    for i in range(num_agents):
        feats.extend([float(left[i][0]), float(left[i][1]),
                      float(left_dir[i][0]), float(left_dir[i][1])])
    right = game_state.get("right_team", [[0.0, 0.0]] * 11)
    nearest = sorted(
        right,
        key=lambda p: (p[0] - ball[0]) ** 2 + (p[1] - ball[1]) ** 2,
    )[:2]
    for k in range(2):
        if k < len(nearest):
            feats.extend([float(nearest[k][0]), float(nearest[k][1])])
        else:
            feats.extend([0.0, 0.0])
    feats.append(float(game_state.get("steps_left", 0)) / 3001.0)
    obs = np.array(feats, dtype=np.float32)
    assert obs.shape == (high_obs_dim(num_agents),), obs.shape
    return obs


def _extract_low_obs(game_state, player_idx: int) -> np.ndarray:
    """extract_obs_vector with team arrays padded to 11 (academy has <11)."""
    padded = dict(game_state)
    for key, fill in (("left_team", [0.0, 0.0]), ("left_team_direction", [0.0, 0.0]),
                      ("left_team_tired_factor", 0.0), ("left_team_yellow_card", 0),
                      ("left_team_roles", 5)):
        lst = list(game_state.get(key, []))
        padded[key] = lst + [fill] * (11 - len(lst))
    return extract_obs_vector(padded, player_idx)


def build_low_obs(game_state, num_agents: int, subgoal_id: int, num_subgoals: int) -> np.ndarray:
    """Joint low-level obs: per-agent [115-dim vector | subgoal one-hot]."""
    onehot = np.zeros(num_subgoals, dtype=np.float32)
    onehot[subgoal_id] = 1.0
    parts = [
        np.concatenate([_extract_low_obs(game_state, i), onehot])
        for i in range(num_agents)
    ]
    return np.concatenate(parts).astype(np.float32)


def _ball_teammate_dist(game_state, player_idx: int) -> float:
    ball = game_state.get("ball", [0.0, 0.0])
    left = game_state.get("left_team", [[0.0, 0.0]] * 11)
    p = left[player_idx]
    return float(np.hypot(ball[0] - p[0], ball[1] - p[1]))


# ---------------------------------------------------------------------------
# SB3 callback (same pattern as academy.py)
# ---------------------------------------------------------------------------
class EpisodeRewardCallback(BaseCallback):
    """Collect one total reward value per completed episode."""

    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []
        self._current_reward = 0.0
        self._current_length = 0

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        self._current_reward += float(rewards[0])
        self._current_length += 1
        if dones[0]:
            self.episode_rewards.append(self._current_reward)
            self.episode_lengths.append(self._current_length)
            self._current_reward = 0.0
            self._current_length = 0
        return True


# ---------------------------------------------------------------------------
# Environments (old gym API: reset -> obs, step -> 4-tuple; SB3 1.3.0)
# ---------------------------------------------------------------------------
class LowLevelGymEnv(gym.Env):
    """GRF-backed env for low-level PPO: one game step per step().

    Subgoal changes every ``period`` steps via ``subgoal_provider(game_state)``
    (None -> random). Dense shaping reward uses the subgoal in effect.
    """

    def __init__(self, grf_name: str, num_agents: int, subgoal_names,
                 period: int, shaping_alpha: float, render: bool = False):
        super().__init__()
        self.num_agents = num_agents
        self.subgoal_names = list(subgoal_names)
        self.num_subgoals = len(self.subgoal_names)
        self.period = period
        self.alpha = shaping_alpha
        self.grf = create_raw_env(
            env_name=grf_name, num_agents=num_agents, render=render,
            log_dir=LOG_DIR,
        )
        self.action_space = gym.spaces.MultiDiscrete([ACTION_SPACE_SIZE] * num_agents)
        low_dim = num_agents * (OBS_DIM + self.num_subgoals)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(low_dim,), dtype=np.float32,
        )
        self.subgoal_provider = None  # callable(game_state) -> subgoal id
        self.cur_subgoal = 0
        self._steps_in_period = 0
        self._steps_in_episode = 0
        self.game_state = {}

    @staticmethod
    def _unpack(result):
        if len(result) == 4:
            obs, reward, done, info = result
        else:
            obs, reward, terminated, truncated, info = result
            done = bool(terminated) or bool(truncated)
        return obs, float(np.sum(reward)), bool(done), info

    def reset(self):
        result = self.grf.reset()
        obs_raw = result[0] if isinstance(result, tuple) else result
        self.game_state = extract_game_state(obs_raw)
        self._steps_in_period = 0
        self._steps_in_episode = 0
        self.cur_subgoal = 0
        return build_low_obs(self.game_state, self.num_agents,
                             self.cur_subgoal, self.num_subgoals)

    def step(self, actions):
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()
        gs0 = self.game_state
        ball_x0 = float(gs0.get("ball", [0.0, 0.0, 0.0])[0])
        owner0 = int(gs0.get("ball_owned_player", -1))

        result = self.grf.step(actions)
        obs_raw, r_game, done, info = self._unpack(result)
        self.game_state = extract_game_state(obs_raw)

        # Dense subgoal-progress shaping (subgoal in effect for this action)
        ball_x1 = float(self.game_state.get("ball", [0.0, 0.0, 0.0])[0])
        name = self.subgoal_names[self.cur_subgoal]
        if name == "pass":
            teammates = [j for j in range(self.num_agents) if j != owner0]
            if not teammates:
                teammates = list(range(self.num_agents))
            d0 = _ball_teammate_dist(gs0, teammates[0])
            d1 = _ball_teammate_dist(self.game_state, teammates[0])
            shaped = self.alpha * (d0 - d1)
        else:  # dribble / shoot: forward ball progress
            shaped = self.alpha * (ball_x1 - ball_x0)
        reward = r_game + shaped

        self._steps_in_episode += 1
        if self._steps_in_episode >= EPISODE_CAP:
            done = True

        # Subgoal switch AFTER reward so action/reward stay consistent
        self._steps_in_period += 1
        if self._steps_in_period >= self.period:
            self._steps_in_period = 0
            if self.subgoal_provider is not None:
                self.cur_subgoal = int(self.subgoal_provider(self.game_state))

        low_obs = build_low_obs(self.game_state, self.num_agents,
                                self.cur_subgoal, self.num_subgoals)
        return low_obs, reward, done, info


class HighLevelGymEnv(gym.Env):
    """Drives the shared GRF env one subgoal-period per step.

    Each step runs up to ``period`` game steps with the current low-level
    PPO (deterministic, forced subgoal). Reward = sum of raw game rewards
    + subgoal completion bonus + switch penalty.
    """

    def __init__(self, low_env: LowLevelGymEnv, low_model, period: int,
                 switch_penalty: float = 0.05):
        super().__init__()
        self.low_env = low_env
        self.low_model = low_model
        self.period = period
        self.switch_penalty = switch_penalty
        self.action_space = gym.spaces.Discrete(low_env.num_subgoals)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(high_obs_dim(low_env.num_agents),), dtype=np.float32,
        )
        self.prev_subgoal = None
        self.subgoal_counts = np.zeros(low_env.num_subgoals, dtype=np.int64)
        self.periods_played = 0
        self.switches = 0

    def reset(self):
        self.low_env.reset()
        self.prev_subgoal = None
        return build_high_obs(self.low_env.game_state, self.low_env.num_agents)

    def step(self, action):
        subgoal = int(action)
        env = self.low_env
        gs0 = env.game_state
        ball_x0 = float(gs0.get("ball", [0.0, 0.0, 0.0])[0])
        score0 = gs0.get("score", [0, 0])

        possession = []  # ball_owned_player values while our team owns ball
        total_r = 0.0
        done = False
        steps = 0
        while steps < self.period and not done:
            low_obs = build_low_obs(env.game_state, env.num_agents,
                                    subgoal, env.num_subgoals)
            joint_action, _ = self.low_model.predict(low_obs, deterministic=True)
            if isinstance(joint_action, np.ndarray):
                joint_action = joint_action.tolist()
            obs_raw, r_game, done, info = env._unpack(env.grf.step(joint_action))
            env.game_state = extract_game_state(obs_raw)
            total_r += r_game
            if env.game_state.get("ball_owned_team", -1) == 0:
                possession.append(int(env.game_state.get("ball_owned_player", -1)))
            env._steps_in_episode += 1
            if env._steps_in_episode >= EPISODE_CAP:
                done = True
            steps += 1

        gs1 = env.game_state
        name = env.subgoal_names[subgoal]
        bonus = 0.0
        score1 = gs1.get("score", [0, 0])
        if name == "shoot":
            gf0 = score0[0] if isinstance(score0, (list, tuple)) else 0
            gf1 = score1[0] if isinstance(score1, (list, tuple)) else 0
            if gf1 > gf0:
                bonus += BONUS_SHOOT_GOAL
        elif name == "pass":
            for a, b in zip(possession, possession[1:]):
                if a != b and a < env.num_agents and b < env.num_agents:
                    bonus += BONUS_PASS_COMPLETE
                    break
        elif name == "dribble":
            dx = float(gs1.get("ball", [0.0, 0.0, 0.0])[0]) - ball_x0
            if dx >= BONUS_DRIBBLE_DX:
                bonus += BONUS_DRIBBLE_SCALE * min(dx / BONUS_DRIBBLE_SCALE, 1.0)

        switch_pen = 0.0
        if self.prev_subgoal is not None and subgoal != self.prev_subgoal:
            switch_pen = -self.switch_penalty
            self.switches += 1
        self.prev_subgoal = subgoal
        self.subgoal_counts[subgoal] += 1
        self.periods_played += 1

        high_obs = build_high_obs(gs1, env.num_agents)
        reward = total_r + bonus + switch_pen
        info = {"bonus": bonus, "game_reward": total_r, "subgoal": subgoal}
        return high_obs, reward, done, info


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
def _mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


class HierarchicalTrainer:
    """Alternating block-coordinate training of high + low PPO."""

    def __init__(self, args):
        scenario = SCENARIOS[args.env_name]
        self.env_name = args.env_name
        self.grf_name = scenario["grf_name"]
        self.num_agents = scenario["num_agents"]
        self.subgoal_names = scenario["subgoals"]

        print(f"Initializing hierarchical PPO on {self.grf_name} "
              f"({self.num_agents} agent(s), subgoals={self.subgoal_names})")
        self.low_env = LowLevelGymEnv(
            self.grf_name, self.num_agents, self.subgoal_names,
            period=args.period, shaping_alpha=args.shaping_alpha,
            render=args.render,
        )
        self.low_model = PPO(
            "MlpPolicy", self.low_env, verbose=1,
            learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
            gamma=0.99,          # dense shaping: short horizon, discount < 1
            gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        )
        self.high_env = HighLevelGymEnv(
            self.low_env, self.low_model,
            period=args.period, switch_penalty=args.switch_penalty,
        )
        self.high_model = PPO(
            "MlpPolicy", self.high_env, verbose=1,
            learning_rate=3e-4, n_steps=128, batch_size=64, n_epochs=4,
            gamma=1.0,           # period-level: full credit, per skill guidance
            gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        )

    def _sample_high_subgoal(self, game_state):
        """Subgoal provider for the low-level phase: high policy, exploring."""
        high_obs = build_high_obs(game_state, self.num_agents)
        action, _ = self.high_model.predict(high_obs, deterministic=False)
        return int(action)

    def train(self, args):
        low_cb = EpisodeRewardCallback()
        high_cb = EpisodeRewardCallback()
        history = {"low_mean": [], "high_mean": [], "switch_rates": [], "usage": []}

        for cycle in range(args.cycles):
            print(f"\n=== Cycle {cycle + 1}/{args.cycles} ===")
            self.low_env.subgoal_provider = self._sample_high_subgoal
            self.low_model.learn(
                total_timesteps=args.low_steps,
                reset_num_timesteps=(cycle == 0),
                callback=low_cb,
            )
            self.high_env.subgoal_counts[:] = 0
            periods_before = self.high_env.periods_played
            switches_before = self.high_env.switches
            self.low_env.subgoal_provider = None
            self.high_model.learn(
                total_timesteps=args.high_steps,
                reset_num_timesteps=(cycle == 0),
                callback=high_cb,
            )
            periods = self.high_env.periods_played - periods_before
            switches = self.high_env.switches - switches_before
            usage = self.high_env.subgoal_counts.copy()
            switch_rate = switches / max(periods, 1)

            print(f"[cycle {cycle+1}] low_mean={_mean(low_cb.episode_rewards):.3f} "
                  f"high_mean={_mean(high_cb.episode_rewards):.3f} "
                  f"usage={dict(zip(self.subgoal_names, usage.tolist()))} "
                  f"switch_rate={switch_rate:.3f}")
            history["low_mean"].append(_mean(low_cb.episode_rewards))
            history["high_mean"].append(_mean(high_cb.episode_rewards))
            history["switch_rates"].append(switch_rate)
            history["usage"].append(usage.tolist())
            self._save(cycle)

        self._plot(history, low_cb, high_cb)
        return history

    def _save(self, cycle):
        # ponytail: overwrite per cycle = keep only latest (crash-recovery
        # convention, same as hmarl checkpoints); numbered saves when
        # ablations need multiple snapshots.
        self.low_model.save(f"hier_{self.env_name}_low_model")
        self.high_model.save(f"hier_{self.env_name}_high_model")
        print(f"Models saved (cycle {cycle + 1}).")

    def _plot(self, history, low_cb, high_cb):
        cycles = np.arange(1, len(history["low_mean"]) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(cycles, history["low_mean"], marker="o", color="tab:blue")
        axes[0].set_title("Low-level mean episode reward")
        axes[0].set_xlabel("Cycle")
        axes[1].plot(cycles, history["high_mean"], marker="o", color="tab:green")
        axes[1].set_title("High-level mean episode reward")
        axes[1].set_xlabel("Cycle")
        axes[2].plot(cycles, history["switch_rates"], marker="o", color="tab:red")
        axes[2].set_title("Subgoal switch rate")
        axes[2].set_xlabel("Cycle")
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.tight_layout()
        out = f"{LOG_DIR}/hier_{self.env_name}_training.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Training plot saved to {out}")

        usage = np.array(history["usage"][-1], dtype=float) if history["usage"] else None
        if usage is not None and usage.sum() > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(self.subgoal_names, usage / usage.sum(), color="tab:purple")
            ax.set_title("Final-cycle subgoal usage")
            ax.set_ylabel("Fraction of periods")
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            out = f"{LOG_DIR}/hier_{self.env_name}_subgoals.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"Subgoal usage plot saved to {out}")

    def run(self, args):
        """Deterministic evaluation of both levels."""
        print("Starting hierarchical evaluation...")
        eval_env = create_raw_env(
            env_name=self.grf_name, num_agents=self.num_agents,
            render=args.render, write_dumps=True, log_dir=LOG_DIR,
        )
        ep_rewards, sequences, goals = [], [], 0
        for ep in range(args.eval_episodes):
            result = eval_env.reset()
            gs = extract_game_state(result[0] if isinstance(result, tuple) else result)
            ep_r, done, steps_in_ep = 0.0, False, 0
            period_left, subgoal, prev_sg, switches = 0, 0, None, 0
            seq = []
            while not done and steps_in_ep < EPISODE_CAP:
                if period_left == 0:
                    high_obs = build_high_obs(gs, self.num_agents)
                    subgoal = int(self.high_model.predict(high_obs, deterministic=True)[0])
                    if prev_sg is not None and subgoal != prev_sg:
                        switches += 1
                    prev_sg = subgoal
                    seq.append(subgoal)
                    period_left = args.period
                low_obs = build_low_obs(gs, self.num_agents, subgoal,
                                        len(self.subgoal_names))
                joint, _ = self.low_model.predict(low_obs, deterministic=True)
                if isinstance(joint, np.ndarray):
                    joint = joint.tolist()
                res = eval_env.step(joint)
                if len(res) == 4:
                    obs_raw, r, done, info = res
                else:
                    obs_raw, r, term, trunc, info = res
                    done = bool(term) or bool(trunc)
                gs = extract_game_state(obs_raw)
                ep_r += float(np.sum(r))
                period_left -= 1
                steps_in_ep += 1
            score = gs.get("score", [0, 0])
            gf = int(score[0]) if isinstance(score, (list, tuple)) else 0
            goals += gf
            ep_rewards.append(ep_r)
            sequences.append(seq)
            print(f"Ep {ep+1}: reward={ep_r:.2f} goals_for={gf} "
                  f"periods={len(seq)} switches={switches} "
                  f"seq={[self.subgoal_names[s] for s in seq]}")
        eval_env.close()

        usage = np.zeros(len(self.subgoal_names), dtype=np.int64)
        for seq in sequences:
            for s in seq:
                usage[s] += 1
        summary = {
            "env": self.grf_name,
            "episodes": args.eval_episodes,
            "avg_reward": _mean(ep_rewards),
            "total_goals_for": goals,
            "subgoal_usage": {n: int(u) for n, u in zip(self.subgoal_names, usage.tolist())},
            "avg_periods_per_episode": _mean([len(s) for s in sequences]),
        }
        print(json.dumps(summary, indent=2))
        out = f"{LOG_DIR}/hier_{self.env_name}_eval.json"
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Eval summary saved to {out}")
        return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Small-scale hierarchical PPO on GRF academy scenarios.")
parser.add_argument("--env-name", type=str, default="pass_and_shoot_with_keeper",
                    help=f"Scenario key: {list(SCENARIOS)} (academy_ prefix optional).")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--period", type=int, default=16,
                    help="Low-level steps per high-level decision.")
parser.add_argument("--shaping-alpha", type=float, default=0.5,
                    help="Dense subgoal-progress shaping weight.")
parser.add_argument("--switch-penalty", type=float, default=0.05,
                    help="High-level penalty per subgoal switch.")
parser.add_argument("--cycles", type=int, default=6,
                    help="Alternating low/high training cycles.")
parser.add_argument("--low-steps", type=int, default=20000,
                    help="Low-level PPO timesteps (game steps) per cycle.")
parser.add_argument("--high-steps", type=int, default=800,
                    help="High-level PPO steps (periods) per cycle.")
parser.add_argument("--eval-episodes", type=int, default=10)
parser.add_argument("--render", action="store_true")
parser.add_argument("--smoke", action="store_true",
                    help="Tiny budget end-to-end check (~2 min).")


def _normalize_env_name(name: str) -> str:
    if name.startswith("academy_"):
        name = name[len("academy_"):]
    if name not in SCENARIOS:
        raise SystemExit(f"Unknown env '{name}'. Choose from: {list(SCENARIOS)}")
    return name


if __name__ == "__main__":
    args = parser.parse_args()
    args.env_name = _normalize_env_name(args.env_name)
    if args.smoke:
        args.cycles = 1
        args.low_steps = 20480
        args.high_steps = 640
        args.eval_episodes = 10

    set_seed(args.seed)
    trainer = HierarchicalTrainer(args)

    if args.smoke:
        gs = trainer.low_env.reset()
        raw = gs  # reset already returned low obs; inspect game state directly
        gs_state = trainer.low_env.game_state
        print("[smoke] ball_owned_player:", gs_state.get("ball_owned_player"),
              "| active:", gs_state.get("active"),
              "| left_team[:2]:", gs_state.get("left_team", [])[:2],
              "| low obs dim:", gs.shape)

    trainer.train(args)
    trainer.run(args)
