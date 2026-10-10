"""Uniform run-metadata JSON logging for training/eval/analysis scripts.

Every run log produced via save_run_log() carries the same metadata block:
    script, timestamp_utc, device, versions
plus whatever the caller adds (algo, env_name, hyperparams, results, ...).

Usage:
    from hmarl.run_logging import (
        run_metadata, save_run_log, hyperparams_snapshot, obs_snapshot,
        OBS_SIMPLE115V2, OBS_HMARL_115,
    )
    payload = run_metadata(script="train.py", algo="hmarl", train_time_s=elapsed,
                           hyperparams=hyperparams_snapshot({...}),
                           **results)
    save_run_log(os.path.join(log_dir, "training_log.json"), payload)
"""
import json
import os
import platform
from datetime import datetime

import numpy as np
import torch

# --- Observation descriptions (source-verified) -------------------------------
# GRF built-in simple115v2 (gfootball/env/wrappers.py Simple115StateWrapper)
OBS_SIMPLE115V2 = (
    "GRF simple115v2 vector per controlled agent (115 float32): "
    "88 = team positions & directions (11 players x 2 teams x xy; missing players backfilled -1), "
    "3 ball position (x,y,z), 3 ball direction, 3 ball-ownership one-hot (none/left/right), "
    "11 active-player one-hot, 7 game-mode one-hot; fixed team indices (v2)"
)
# hmarl.utils.extract_obs_vector (custom vector from raw game_state)
OBS_HMARL_115 = (
    "hmarl extract_obs_vector (115 float32, zero-padded/truncated): "
    "11 ball block (position xyz, direction xyz, rotation xyz, owned_team, owned_player), "
    "88 = per left-team player x11 (active-flag, xy position, xy direction, tired factor, "
    "yellow card, role); right team not included"
)


def device_snapshot(device=None):
    """Detailed compute-device description."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_avail = torch.cuda.is_available()
    return {
        "torch_device": str(device),
        "cuda_available": cuda_avail,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda_avail else None,
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
    }


def versions_snapshot():
    """Software versions relevant to reproducibility."""
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    try:
        import stable_baselines3
        versions["stable_baselines3"] = stable_baselines3.__version__
    except ImportError:
        pass
    try:
        import gym
        versions["gym"] = gym.__version__
    except ImportError:
        pass
    return versions


def hyperparams_snapshot(hp):
    """JSON-serializable snapshot of a hyperparameter dict.

    SB3 wraps floats in constant_fn(progress_remaining) callables; unwrap by calling.
    """
    out = {}
    for k, v in (hp or {}).items():
        if callable(v):
            try:
                v = v(1.0)
            except TypeError:
                v = str(v)
        if isinstance(v, (int, float, str, bool, type(None))):
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(
            isinstance(x, (int, float, str, bool, type(None))) for x in v
        ):
            out[k] = list(v)
        elif isinstance(v, dict):
            out[k] = hyperparams_snapshot(v)
        else:
            out[k] = str(v)
    return out


def obs_snapshot(representation, shape, description, stacked=False):
    """Observation-space description for run logs."""
    return {
        "representation": representation,
        "shape": list(shape),
        "dtype": "float32",
        "stacked": stacked,
        "description": description,
    }


def run_metadata(script, device=None, **extra):
    """Standard metadata block: script, timestamp, device, versions (+extras)."""
    meta = {
        "script": script,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "device": device_snapshot(device),
        "versions": versions_snapshot(),
    }
    meta.update(extra)
    return meta


def save_run_log(path, payload):
    """Write a run-log JSON; returns the path."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Run log saved to {path}")
    return path


# Attributes SB3 PPO/A2C expose as training hyperparameters
SB3_HP_KEYS = [
    "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma",
    "gae_lambda", "clip_range", "clip_range_vf", "ent_coef", "vf_coef",
    "max_grad_norm", "rms_prop_eps", "normalize_advantage",
]


def sb3_hyperparams(model, policy="MlpPolicy"):
    """JSON-serializable hyperparameter snapshot of an SB3 model instance."""
    hp = {"policy": policy}
    for k in SB3_HP_KEYS:
        if hasattr(model, k):
            hp[k] = getattr(model, k)
    return hyperparams_snapshot(hp)
