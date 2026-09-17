"""Shared utilities for HMARL-GRF.

Consolidates duplicated helpers across scripts:
- seed management
- observation vector extraction
- checkpoint loading with reproducibility metadata
- progress/ETA display
- temp directory cleanup
"""

import hashlib
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants (single source of truth)
# ---------------------------------------------------------------------------
OBS_DIM = 115
HIDDEN_DIM = 256
HEAD_DIM = 128
ACTION_SPACE_SIZE = 19
EPISODE_MAX_STEPS = 3000


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Observation vector
# ---------------------------------------------------------------------------
def extract_obs_vector(game_state: Dict, player_idx: int,
                       obs_dim: int = OBS_DIM) -> np.ndarray:
    """Extract fixed-length observation vector from GRF game state dict."""
    features = []
    ball = game_state.get('ball', [0, 0, 0])
    features.extend(ball)
    features.extend(game_state.get('ball_direction', [0, 0, 0]))
    features.extend(game_state.get('ball_rotation', [0, 0, 0]))
    features.append(float(game_state.get('ball_owned_team', -1)))
    features.append(float(game_state.get('ball_owned_player', -1)))

    for i in range(11):
        pos = game_state.get('left_team', [[0, 0]] * 11)[i]
        d = game_state.get('left_team_direction', [[0, 0]] * 11)[i] \
            if 'left_team_direction' in game_state else [0, 0]
        t = game_state.get('left_team_tired_factor', [0.0] * 11)[i] \
            if 'left_team_tired_factor' in game_state else 0.0
        y = game_state.get('left_team_yellow_card', [0] * 11)[i] \
            if 'left_team_yellow_card' in game_state else 0
        r = game_state.get('left_team_roles', [5] * 11)[i] \
            if 'left_team_roles' in game_state else 5
        features.append(1.0 if i == player_idx else 0.0)
        features.extend(pos)
        features.extend(d)
        features.append(t)
        features.append(float(y))
        features.append(float(r))

    features = features[:obs_dim]
    while len(features) < obs_dim:
        features.append(0.0)
    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def load_hmarl_checkpoint(
    ckpt_path: str,
    device: Optional[str] = None,
) -> Tuple:
    """Load HMARL policy + subgoal_embedding from checkpoint.

    Returns:
        (policy, subgoal_embedding, metadata_dict)
    """
    import torch
    from hmarl.policy import (
        HierarchicalActorCritic, SubGoalEmbedding, SUBGOAL_EMBED_DIM,
    )

    if device is not None:
        dev = torch.device(device)
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = HierarchicalActorCritic(
        obs_dim=OBS_DIM, subgoal_embed_dim=SUBGOAL_EMBED_DIM,
        hidden_dim=HIDDEN_DIM, head_dim=HEAD_DIM, action_dim=ACTION_SPACE_SIZE,
    ).to(dev)
    subgoal_emb = SubGoalEmbedding().to(dev)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    policy.load_state_dict(ckpt['policy_state'])
    subgoal_emb.load_state_dict(ckpt['subgoal_embedding_state'])
    policy.eval()
    subgoal_emb.eval()

    metadata = {k: v for k, v in ckpt.items()
                if k not in ('policy_state', 'subgoal_embedding_state', 'optimizer_state')}

    return policy, subgoal_emb, metadata


def save_checkpoint(
    path: str,
    policy_state: dict,
    subgoal_embedding_state: dict,
    optimizer_state: Optional[dict] = None,
    extra: Optional[Dict] = None,
):
    """Save checkpoint with reproducibility metadata."""
    import torch
    payload = {
        'policy_state': policy_state,
        'subgoal_embedding_state': subgoal_embedding_state,
    }
    if optimizer_state is not None:
        payload['optimizer_state'] = optimizer_state

    # Metadata
    payload['git_hash'] = get_git_hash()
    payload['python_version'] = sys.version
    payload['save_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    try:
        import torch
        payload['torch_version'] = torch.__version__
    except Exception:
        pass

    if extra:
        payload.update(extra)

    torch.save(payload, path)


def get_git_hash() -> str:
    """Best-effort git hash of current repo."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or 'unknown'
    except Exception:
        return 'unknown'


# ---------------------------------------------------------------------------
# Progress / ETA
# ---------------------------------------------------------------------------
class ProgressTracker:
    """Lightweight progress + ETA display for loops."""

    def __init__(self, total: int, label: str = "Progress",
                 print_every: int = 1):
        self.total = max(total, 1)
        self.label = label
        self.print_every = print_every
        self.start_time = time.time()
        self.current = 0

    def update(self, n: int = 1, extra: str = ""):
        self.current += n
        if self.current % self.print_every != 0 and self.current != self.total:
            return
        elapsed = time.time() - self.start_time
        rate = self.current / max(elapsed, 0.01)
        remaining = (self.total - self.current) / max(rate, 0.001)
        eta_str = self._format_time(remaining)
        pct = self.current / self.total * 100
        suffix = f" | {extra}" if extra else ""
        print(
            f"\r  {self.label}: {self.current}/{self.total} "
            f"({pct:.0f}%) | {rate:.1f}/s | ETA: {eta_str}{suffix}   ",
            end="", flush=True,
        )

    def done(self):
        elapsed = time.time() - self.start_time
        print(f"\r  {self.label}: {self.total}/{self.total} "
              f"(100%) | elapsed: {self._format_time(elapsed)}" + " " * 20)

    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        if minutes < 60:
            return f"{minutes}m{secs:02d}s"
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours}h{mins:02d}m"


# ---------------------------------------------------------------------------
# Temp cleanup
# ---------------------------------------------------------------------------
def cleanup_temp_dirs(base_dirs=None):
    """Remove temporary directories created by scripts."""
    if base_dirs is None:
        base_dirs = ['stat_temp', 'ablation_temp', 'sweep_temp', 'valid_temp']
    for d in base_dirs:
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            print(f"  Cleaned up: {d}/")


# ---------------------------------------------------------------------------
# Centralized obs (MAPPO / SHPPO shared)
# ---------------------------------------------------------------------------
def build_centralized_obs(obs_list) -> np.ndarray:
    """Concatenate all agents' observations into one centralized obs vector.

    Args:
        obs_list: list of N (obs_dim,) arrays, one per agent
    Returns:
        (N * obs_dim,) float32 array
    """
    return np.concatenate(obs_list).astype(np.float32)


# ---------------------------------------------------------------------------
# Quick eval (shared across flat baselines)
# ---------------------------------------------------------------------------
def quick_eval(
    model,
    num_episodes: int = 10,
    extra_modules=None,
) -> Dict:
    """Run quick evaluation during training.

    Args:
        model: nn.Module with forward(obs) -> (logits, value)
        num_episodes: episodes to evaluate
        extra_modules: list of nn.Module to put in eval/train mode
                       alongside model (e.g. [inference_net])
    """
    import torch
    from hmarl.env import create_raw_env, NUM_AGENTS, extract_game_state

    DEVICE = next(model.parameters()).device
    model.eval()
    if extra_modules:
        for m in extra_modules:
            m.eval()

    eval_env = create_raw_env(render=False)
    wins, total_gf, total_ga, total_r = 0, 0, 0, 0.0

    for _ in range(num_episodes):
        reset_result = eval_env.reset()
        obs_raw = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        game_state = extract_game_state(obs_raw)
        ep_r = 0.0
        done = False
        step = 0

        while not done and step < EPISODE_MAX_STEPS:
            joint_actions = []
            for i in range(NUM_AGENTS):
                obs_vec = extract_obs_vector(game_state, i)
                obs_t = torch.FloatTensor(obs_vec).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    action = model.get_action_and_value(obs_t)[0]
                    joint_actions.append(action.item())

            step_result = eval_env.step(joint_actions)
            if len(step_result) == 5:
                obs_raw, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs_raw, reward, done, info = step_result
            ep_r += float(np.sum(reward))
            game_state = extract_game_state(obs_raw)
            step += 1

        score = game_state.get('score', [0, 0])
        gf, ga = (score[0], score[1]) if isinstance(score, (list, tuple)) and len(score) >= 2 else (0, 0)
        if gf > ga:
            wins += 1
        total_gf += gf
        total_ga += ga
        total_r += ep_r

    eval_env.close()
    model.train()
    if extra_modules:
        for m in extra_modules:
            m.train()

    return {
        'win_rate': wins / num_episodes * 100.0,
        'avg_reward': total_r / num_episodes,
        'goal_diff': total_gf - total_ga,
    }
