"""PPO-tuned flock formation baseline on GRF (SB3, old-gym API).

Reuses 11v11_flock_formation's team controller. PPO's action is a 5-dim
parameter vector chosen from the observation each timestep:
    [w_flock, w_align, w_spacing, shot_prob, tackle_prob]
which are injected into the flock controller before actions are decided.
Reward = team game reward (sum over players).

Run-logging schema matches evaluation/baselines/11v11_ppo.py:
per-episode results + run metadata + learned parameters -> dumps/.

Run (in Docker):
    python evaluation/baselines/ppo_flock_formation.py [total_timesteps]
"""
import glob
import importlib
import os
import sys
import time
from typing import Any, Dict, Tuple

import gym  # old gym: SB3 1.3.0 isinstance-checks spaces against these classes
import gfootball.env as football_env
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Sibling baseline module (name starts with a digit -> importlib only)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ff = importlib.import_module("11v11_flock_formation")

# Ensure project root is on path so `import hmarl` works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hmarl.run_logging import (
    run_metadata, save_run_log, sb3_hyperparams, obs_snapshot, OBS_HMARL_115,
)

ENV_NAME = ff.ENV_NAME
NUM_AGENTS = ff.NUM_AGENTS
ACTION_SPACE_SIZE = ff.ACTION_SPACE_SIZE
LOG_DIR = ff.LOG_DIR
ALGO = "ppo_flock"
MODEL_PATH = "ppo_flock_formation_model"

PARAM_NAMES = (
    "w_flock", "w_align", "w_spacing",
    "shot_prob", "tackle_prob", "pass_prob", "pass_dir",
)
WEIGHT_SCALE = 2.0  # action[0:3] in [0,1] -> weight in [0, 2]

# Pass verification reward: after a pass, check ball ownership PASS_WINDOW steps later
PASS_WINDOW = 10
PASS_REWARD = 1.0      # left team still owns the ball -> reward
PASS_PENALTY = -1.0    # right team owns it (pass failed) -> penalty

SHOT_ON_TARGET_WINDOW = 5  # shot counts on target if goal or right possession within N steps
CHECKPOINTS = (0.25, 0.5, 0.75)  # partial training dumps at quarter iterations
MAX_DUMPS = 10  # keep at most N recent .dump files (README Episode Dumps convention)


def _score_of(obs: Any) -> list:
    """Score [left, right] from GRF raw observation (defaults [0,0])."""
    s = obs[0].get("score", [0, 0])
    return [int(s[0]), int(s[1])]


class EpisodeMetricsTracker:
    """Per-episode football metrics for one episode: GA, GF, shots, passes, possession.

    Definitions (ponytail: GRF raw exposes no shot events):
    - shot = carrier action == shot (12)
    - shot on target = shot followed within SHOT_ON_TARGET_WINDOW steps by a goal
      OR right-team possession (keeper collected). Upgrade path: parse full episode
      dumps (write_full_episode_dumps) for engine-side shot events.
    - pass = carrier action in {9,10,11}
    - pass success = within PASS_WINDOW steps, left team owns and passer changed
    """

    def __init__(self):
        self.episodes = []
        self._pending = []   # [{ep, passer, due}]
        self._tick = 0
        self.ep = None
        self._reset_episode()

    def _reset_episode(self):
        self.ep = {"gf": 0, "ga": 0, "shots": 0, "shots_on_target": 0,
                   "passes": 0, "passes_success": 0, "possession_steps": 0,
                   "steps": 0, "goals_timeline": []}

    def start_episode(self):
        if self.ep["steps"] > 0:
            self.episodes.append(self.ep)
        self._pending = []
        self._reset_episode()

    def on_step(self, pre_obs: Any, post_obs: Any, actions: list, prev_score: list):
        ep = self.ep
        self._tick += 1
        ep["steps"] += 1
        score = _score_of(post_obs)
        pre = pre_obs[0]
        post = post_obs[0]

        # Goals / GA
        d_gf, d_ga = score[0] - prev_score[0], score[1] - prev_score[1]
        if d_gf > 0:
            ep["gf"] += d_gf
            ep["goals_timeline"].append({"step": ep["steps"], "side": "gf"})
            # any pending shot resolved by this goal is on target
            for p in [p for p in self._pending if p.get("shot")]:
                p["ep"]["shots_on_target"] += 1
                self._pending.remove(p)
        if d_ga > 0:
            ep["ga"] += d_ga
            ep["goals_timeline"].append({"step": ep["steps"], "side": "ga"})

        # Possession
        if post.get("ball_owned_team") == 0:
            ep["possession_steps"] += 1

        # Carrier actions: pass / shot (pre-step ownership)
        if pre.get("ball_owned_team") == 0 and pre.get("ball_owned_player", -1) >= 0:
            carrier = int(pre["ball_owned_player"])
            if carrier < len(actions):
                act = actions[carrier]
                if act in ff.PASS_ACTIONS:
                    ep["passes"] += 1
                    self._pending.append(
                        {"ep": ep, "passer": carrier, "due": self._tick + PASS_WINDOW}
                    )
                if act == ff.SHOT_ACTION:
                    ep["shots"] += 1
                    on_target = d_gf > 0
                    if not on_target:
                        # pending shot-on-target check within window
                        self._pending.append(
                            {"ep": ep, "shot": True, "due": self._tick + SHOT_ON_TARGET_WINDOW}
                        )

        # Resolve pending pass completions / shot checks due this step
        still = []
        for p in self._pending:
            if p["due"] > self._tick:
                still.append(p)
                continue
            if p.get("shot"):
                # shot on target: goal already counted above, or keeper collected
                if post.get("ball_owned_team") == 1:
                    p["ep"]["shots_on_target"] += 1
            else:
                if (post.get("ball_owned_team") == 0
                        and post.get("ball_owned_player", -1) >= 0
                        and int(post["ball_owned_player"]) != p["passer"]):
                    p["ep"]["passes_success"] += 1
        self._pending = still

    def finish_episode(self):
        if self.ep["steps"] > 0:
            self.episodes.append(self.ep)
            self._pending = []
            self._reset_episode()

    @property
    def current(self) -> dict:
        return self.ep

    @staticmethod
    def summary(episodes: list) -> dict:
        """Aggregate football metrics over finished episodes."""
        n = len(episodes)
        if n == 0:
            return {"games": 0}
        tot = lambda k: sum(e[k] for e in episodes)
        gf, ga = tot("gf"), tot("ga")
        shots, shots_ot = tot("shots"), tot("shots_on_target")
        passes, passes_ok = tot("passes"), tot("passes_success")
        poss, steps = tot("possession_steps"), tot("steps")
        return {
            "games": n,
            "wins": sum(1 for e in episodes if e["gf"] > e["ga"]),
            "draws": sum(1 for e in episodes if e["gf"] == e["ga"]),
            "losses": sum(1 for e in episodes if e["gf"] < e["ga"]),
            "gf_total": gf, "ga_total": ga, "gd_total": gf - ga,
            "gf_per_game": round(gf / n, 3), "ga_per_game": round(ga / n, 3),
            "shots": shots, "shots_on_target": shots_ot,
            "shot_accuracy_pct": round(100.0 * shots_ot / shots, 2) if shots else 0.0,
            "passes": passes, "passes_success": passes_ok,
            "pass_success_pct": round(100.0 * passes_ok / passes, 2) if passes else 0.0,
            "possession_pct": round(100.0 * poss / steps, 2) if steps else 0.0,
            "total_steps": steps,
        }

    def summary_current(self) -> dict:
        """Aggregate including the in-flight episode (for partial dumps)."""
        eps = self.episodes + ([self.ep] if self.ep["steps"] > 0 else [])
        return self.summary(eps)


def feature_vector(obs: Any) -> np.ndarray:
    """Raw-observation features fed to the parameter network (49 floats)."""
    s = obs[0]
    ball = np.asarray(s["ball"], dtype=np.float32)                      # 3
    left = np.asarray(s["left_team"], dtype=np.float32).ravel()         # 22
    right = np.asarray(s["right_team"], dtype=np.float32).ravel()       # 22
    owned = np.array(
        [np.float32(s["ball_owned_team"]), np.float32(s["ball_owned_player"]) / 11.0],
        dtype=np.float32,
    )                                                                    # 2
    return np.concatenate([ball, left, right, owned])                    # 49


def decode_params(action: np.ndarray) -> np.ndarray:
    """Map raw action in [0,1]^7 to actual controller parameters."""
    a = np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)
    return np.array(
        [a[0] * WEIGHT_SCALE, a[1] * WEIGHT_SCALE, a[2] * WEIGHT_SCALE,
         a[3], a[4], a[5], a[6]],
        dtype=np.float32,
    )


class EpisodeRewardCallback(BaseCallback):
    """Collect one total reward value for every completed training episode.

    When given a payload_fn, also dumps a partial training-log JSON at each
    quarter of total_timesteps (checkpoint dumps include football metrics of
    episodes finished so far, plus the in-flight episode).
    """

    def __init__(self, total_timesteps: int = 0, payload_fn=None, name: str = ""):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []
        self._current_reward = 0.0
        self._current_length = 0
        self.total_timesteps = max(1, total_timesteps)
        self.payload_fn = payload_fn
        self.name = name
        self._hit = set()

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

        # Quarter-iteration checkpoint dumps
        if self.payload_fn is not None:
            frac = self.num_timesteps / self.total_timesteps
            for cp in CHECKPOINTS:
                if cp not in self._hit and frac >= cp:
                    self._hit.add(cp)
                    pct = int(round(cp * 100))
                    payload = self.payload_fn(checkpoint=f"{pct}%")
                    save_run_log(
                        os.path.join(LOG_DIR, f"{ALGO}_{self.name}_checkpoint_{pct}pct.json"),
                        payload,
                    )
                    # arm a GRF .dump for the next episode at this quarter mark
                    if hasattr(self, "env") and hasattr(self.env, "arm_dump"):
                        self.env.arm_dump()

        return True


class FlockParamEnv(gym.Env):
    """GRF wrapped so PPO outputs flock parameters, not player actions.

    The rule-based controller in 11v11_flock_formation picks the 11 player
    actions; PPO tunes its formation weights and decision probabilities.
    """
    N_PARAMS = 7

    def __init__(self, env_name: str = ENV_NAME, num_agents: int = NUM_AGENTS, render: bool = False):
        super().__init__()
        self.num_agents = num_agents
        raw = football_env.create_environment(
            env_name=env_name,
            representation="raw",
            number_of_left_players_agent_controls=num_agents,
            render=render,
            logdir=LOG_DIR,
            write_full_episode_dumps=False,
        )
        # Reuse the flock baseline's SoccerMatch without its env construction
        self.match = ff.SoccerMatch.__new__(ff.SoccerMatch)
        self.match.env = ff.SoccerMatch._patch_grf_env(None, raw)
        self.match.num_agents = num_agents
        self.match.agents = [ff.SoccerAgent(i) for i in range(num_agents)]
        self.match.role_order = ff.DEFAULT_ROLE_ORDER
        self.match.current_obs = None
        self.match.step_count = 0
        # pending pass-verification checks: step_number -> None
        self._pending_passes = []
        # football metrics across training/eval episodes
        self.metrics = EpisodeMetricsTracker()
        self._last_score = [0, 0]
        # .dump lifecycle: arm at checkpoints, dump next episode, auto-clean
        self.env_name = env_name
        self.render = render
        self._dump_armed = False   # dump the next episode (one-shot, set at checkpoints)
        self._dumping = False      # current env writes .dump
        self._dump_all = False     # persistent (evaluation): dump every episode

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(49,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.N_PARAMS,), dtype=np.float32
        )

    def _recreate_env(self, write_dumps: bool) -> None:
        """Rebuild the GRF env with write_full_episode_dumps toggled (README
        Episode Dumps convention: selective dumping instead of every episode)."""
        try:
            self.match.env.close()
        except Exception:
            pass
        raw = football_env.create_environment(
            env_name=self.env_name,
            representation="raw",
            number_of_left_players_agent_controls=self.num_agents,
            render=self.render,
            logdir=LOG_DIR,
            write_full_episode_dumps=write_dumps,
        )
        self.match.env = ff.SoccerMatch._patch_grf_env(None, raw)
        self._dumping = write_dumps

    def arm_dump(self) -> None:
        """Request a .dump for the next episode (called at quarter checkpoints)."""
        if not self._dumping and not self._dump_armed:
            self._dump_armed = True

    def set_dumping(self, on: bool) -> None:
        """Toggle persistent .dump writing (evaluation dumps every episode)."""
        self._dump_all = bool(on)
        if on:
            self._dump_armed = False
        if on != self._dumping:
            self._recreate_env(on)

    @staticmethod
    def cleanup_dumps(max_dumps: int = MAX_DUMPS) -> int:
        """Keep only the max_dumps most recent .dump files; returns removed count."""
        files = sorted(glob.glob(os.path.join(LOG_DIR, "*.dump")), key=os.path.getmtime)
        removed = 0
        for f in files[: max(0, len(files) - max_dumps)]:
            try:
                os.remove(f)
                removed += 1
            except OSError:
                pass
        return removed

    @staticmethod
    def apply_params(action: np.ndarray) -> np.ndarray:
        """Inject PPO's parameters into the shared flock controller constants."""
        p = decode_params(action)
        ff.W_FLOCK = float(p[0])
        ff.W_ALIGN = float(p[1])
        ff.W_SPACING = float(p[2])
        ff.SHOT_PROB = float(p[3])
        ff.TACKLE_PROB = float(p[4])
        ff.PASS_PROB = float(p[5])
        ff.PASS_DIR_BIAS = float(p[6])
        return p

    def _sync_roles(self, obs: Any) -> None:
        self.match.role_order = ff.get_role_order(obs)
        for agent in self.match.agents:
            agent.update_role_order(self.match.role_order)

    def reset(self) -> np.ndarray:
        # carry finished episode metrics (checkpoint dumps need in-flight episodes)
        if self.match.current_obs is not None and self.match.step_count > 0:
            self.metrics.finish_episode()
        # one-episode .dump finished -> revert to no-dumping (before arm check,
        # so a just-finished dump episode doesn't cancel a newly armed one)
        if self._dumping and not self._dump_all and self.match.step_count > 0:
            self._recreate_env(False)
            self.cleanup_dumps()
        # .dump for the next episode: arm at checkpoint, apply here (episode boundary)
        if self._dump_armed and not self._dumping:
            self._recreate_env(True)
            self._dump_armed = False
        obs, _info = self.match.env.reset()
        self.match.current_obs = obs
        self.match.step_count = 0
        self._pending_passes = []
        self._last_score = _score_of(obs)
        self._sync_roles(obs)
        return feature_vector(obs)

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.apply_params(action)
        obs = self.match.current_obs
        self.match.step_count += 1
        self._sync_roles(obs)
        player_actions = self.match._decide_team_actions(obs)

        # Schedule pass verification: check ownership PASS_WINDOW steps later
        if any(a in ff.PASS_ACTIONS for a in player_actions):
            self._pending_passes.append(self.match.step_count + PASS_WINDOW)

        pre_obs = obs
        prev_score = list(self._last_score)
        obs, reward, done, info = self.match.env.step(player_actions)
        self.match.current_obs = obs
        team_reward = float(np.sum(reward))

        # Football metrics for this step
        self.metrics.on_step(pre_obs, obs, player_actions, prev_score)
        self._last_score = _score_of(obs)

        # Resolve pass checks due this step: left still owns -> reward, right -> penalty
        pass_credit = 0.0
        due = [t for t in self._pending_passes if t <= self.match.step_count]
        if due:
            owned = int(obs[0]["ball_owned_team"])
            if owned == 0:
                pass_credit = PASS_REWARD * len(due)
            elif owned == 1:
                pass_credit = PASS_PENALTY * len(due)
            self._pending_passes = [t for t in self._pending_passes if t > self.match.step_count]

        info = dict(info) if info else {}
        info["params"] = decode_params(action)
        info["pass_credit"] = pass_credit
        info["pending_passes"] = len(self._pending_passes)
        return feature_vector(obs), team_reward + pass_credit, bool(done), info

    def render(self):
        self.match.env.render()

    def close(self):
        self.match.env.close()


class PPOFlockMatch:
    """Trains PPO to select flock parameters, then evaluates them."""

    def __init__(self, env_name: str = ENV_NAME, render: bool = False):
        self.env_name = env_name
        self.env = FlockParamEnv(env_name, NUM_AGENTS, render)
        print(f"Initializing PPO flock-parameter tuner on {env_name}...")
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            verbose=1,
            learning_rate=3e-4,  # matches 11v11_ppo.py
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=1.0,           # Song et al. (2024): gamma=1 for 11v11
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
        )

    def learned_parameters(self) -> Dict[str, float]:
        """Deterministic policy output (mean action) on a fresh observation."""
        obs = self.env.reset()
        action, _states = self.model.predict(obs, deterministic=True)
        params = decode_params(action)
        return {name: round(float(v), 4) for name, v in zip(PARAM_NAMES, params)}

    @staticmethod
    def current_params() -> Dict[str, float]:
        """Live controller parameters (no env reset — safe inside training callback)."""
        vals = (ff.W_FLOCK, ff.W_ALIGN, ff.W_SPACING,
                ff.SHOT_PROB, ff.TACKLE_PROB, ff.PASS_PROB, ff.PASS_DIR_BIAS)
        return {name: round(float(v), 4) for name, v in zip(PARAM_NAMES, vals)}

    def _training_payload(self, reward_cb: EpisodeRewardCallback,
                          total_timesteps: int, train_time_s: float = 0.0,
                          checkpoint: str = "final") -> Dict[str, Any]:
        """Shared payload for final training log and quarter checkpoints."""
        metrics = self.env.metrics
        return run_metadata(
            script="evaluation/baselines/ppo_flock_formation.py",
            algo=ALGO,
            env_name=self.env_name,
            num_agents=NUM_AGENTS,
            checkpoint=checkpoint,
            train_time_s=round(train_time_s, 2),
            total_timesteps=total_timesteps,
            timesteps_executed=int(self.model.num_timesteps),
            episodes=len(reward_cb.episode_rewards),
            hyperparams=sb3_hyperparams(self.model),
            learned_parameters=(self.current_params() if checkpoint != "final"
                                else self.learned_parameters()),
            parameter_names=list(PARAM_NAMES),
            parameter_scale={"weights": [0.0, 2.0 * WEIGHT_SCALE], "probs": [0.0, 1.0]},
            pass_verification={"window": PASS_WINDOW,
                               "reward_left_retains": PASS_REWARD,
                               "penalty_right_owns": PASS_PENALTY},
            observation=obs_snapshot("raw", [49], OBS_HMARL_115),
            episode_dumps={"at_checkpoints": [f"{int(c * 100)}%" for c in CHECKPOINTS],
                           "max_kept": MAX_DUMPS, "logdir": LOG_DIR},
            episode_rewards=reward_cb.episode_rewards,
            episode_lengths=reward_cb.episode_lengths,
            football_metrics=metrics.summary_current(),
            episodes_detail=metrics.episodes + ([metrics.current]
                                                if metrics.current["steps"] > 0 else []),
        )

    def train(self, total_timesteps: int = 25000) -> Dict[str, Any]:
        print(f"Starting training for {total_timesteps} steps...")
        # quarter-iteration checkpoint dumps (current_params: no env reset mid-training)
        reward_callback = EpisodeRewardCallback(
            total_timesteps=total_timesteps,
            payload_fn=lambda checkpoint: self._training_payload(
                reward_callback, total_timesteps, checkpoint=checkpoint),
            name=self.env_name,
        )
        train_start = time.time()
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=reward_callback,
        )
        train_time_s = time.time() - train_start

        self.model.save(MODEL_PATH)
        print(f"Training finished in {train_time_s:.1f}s. Model saved to {MODEL_PATH}.zip")
        payload = self._training_payload(
            reward_callback, total_timesteps, train_time_s, checkpoint="final")
        print(f"Learned parameters: {payload['learned_parameters']}")
        print(f"Training football metrics: {payload['football_metrics']}")
        save_run_log(
            os.path.join(LOG_DIR, f"{ALGO}_{self.env_name}_training_log.json"), payload
        )
        return payload["learned_parameters"]

    def run(self, max_steps: int = 3000, episodes: int = 10) -> Dict[str, Any]:
        """Evaluate the trained parameter policy; dump .dump + evaluation log."""
        print(f"Starting evaluation: up to {episodes} episodes / {max_steps} steps...")
        self.env.metrics = EpisodeMetricsTracker()
        # evaluation: every episode written as .dump (~12MB each)
        self.env.set_dumping(True)
        total_reward = 0.0
        steps_done = 0
        obs = self.env.reset()
        try:
            for step in range(max_steps):
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                steps_done += 1
                if step % 200 == 0:
                    print(
                        f"Step {step:4d} | params: {decode_params(action)} | "
                        f"reward: {reward:.3f}"
                    )
                if done:
                    m = self.env.metrics.current
                    print(f"Episode done at step {step}: GF {m['gf']} - GA {m['ga']}")
                    if len(self.env.metrics.episodes) >= episodes:
                        break
                    obs = self.env.reset()
        except KeyboardInterrupt:
            print("\nEvaluation interrupted by user.")
        finally:
            self.env.close()

        metrics = self.env.metrics
        summary = metrics.summary_current()
        payload = run_metadata(
            script="evaluation/baselines/ppo_flock_formation.py",
            algo=ALGO,
            env_name=self.env_name,
            num_agents=NUM_AGENTS,
            phase="evaluation",
            eval_max_steps=max_steps,
            eval_episodes_requested=episodes,
            eval_steps_executed=steps_done,
            eval_total_reward=round(total_reward, 3),
            learned_parameters=self.current_params(),
            parameter_names=list(PARAM_NAMES),
            pass_verification={"window": PASS_WINDOW,
                               "reward_left_retains": PASS_REWARD,
                               "penalty_right_owns": PASS_PENALTY},
            observation=obs_snapshot("raw", [49], OBS_HMARL_115),
            episode_dumps={"enabled": True, "every_episode": True,
                           "max_kept": MAX_DUMPS, "logdir": LOG_DIR},
            football_metrics=summary,
            episodes_detail=metrics.episodes + ([metrics.current]
                                                if metrics.current["steps"] > 0 else []),
        )
        self.env.cleanup_dumps()  # bound storage: keep MAX_DUMPS most recent
        print(f"Evaluation football metrics: {summary}")
        save_run_log(
            os.path.join(LOG_DIR, f"{ALGO}_{self.env_name}_evaluation_log.json"), payload
        )
        return summary


if __name__ == "__main__":
    total_timesteps = int(sys.argv[1]) if len(sys.argv) > 1 else 25000
    match = PPOFlockMatch(render=False)
    match.train(total_timesteps=total_timesteps)
    match.run(max_steps=25000)