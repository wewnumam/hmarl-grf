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

    ckpt = torch.load(ckpt_path, map_location='cpu')
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
