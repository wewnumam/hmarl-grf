#!/usr/bin/env python
"""Smoke test (rekomendasi 5, 2026-10-15 training_log 500-episode analysis).

Trains a short run with the reworked reward (v3 ungated progression,
forward-pass r_mid) + GAE lambda 0.99, then checks the signals that were
dead in the full run, BEFORE committing to another 3M-step run.

Pass criteria (means over the run):
  1. mean update_adv_std > 0                — GAE advantage non-degenerate
  2. mean episode_r_mid > 5e-4              — forward-pass path fires (was 0.000)
  3. mean episode_r_progression_abs > 1e-5  — dense progression fires each step
     (signed episode_r_progression is ~0 by construction — zero-sum term —
     so the density check uses the mean absolute value)
  4. PPO update ran every episode           — update diagnostics populated

Usage (inside the container, from repo root):
    python scripts/smoke_test.py [--episodes 12]
Exit code 0 = all pass.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser(description="Smoke test for reward/gradient health")
    ap.add_argument("--episodes", type=int, default=12,
                    help="Episodes to train (~3000 steps each, ~4-5 min/episode CPU)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    log_dir = os.path.join(ROOT, "dumps_smoke")
    model_dir = os.path.join(ROOT, "checkpoints_smoke")

    cmd = [
        sys.executable, os.path.join(HERE, "train.py"),
        "--timesteps", str(args.episodes * 3000),
        "--log-dir", log_dir,
        "--model-dir", model_dir,
        "--dump-freq", "0",
        "--seed", str(args.seed),
    ]
    print(f"[smoke] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)

    with open(os.path.join(log_dir, "training_log.json")) as f:
        log = json.load(f)

    def mean(key):
        vals = log.get(key) or []
        return sum(vals) / len(vals) if vals else None

    m_adv = mean("update_adv_std")
    m_mid = mean("episode_r_mid")
    m_prog = mean("episode_r_progression_abs")
    n_updates = len(log.get("update_policy_loss", []))

    results = [
        ("mean update_adv_std > 0", m_adv is not None and m_adv > 0, m_adv),
        ("mean episode_r_mid > 5e-4", m_mid is not None and m_mid > 5e-4, m_mid),
        ("mean episode_r_progression_abs > 1e-5",
         m_prog is not None and m_prog > 1e-5, m_prog),
        (f"PPO update every episode ({n_updates}/{args.episodes})",
         n_updates >= args.episodes - 1, n_updates),
    ]

    ok = True
    for name, passed, actual in results:
        print(f"[smoke] {'PASS' if passed else 'FAIL'}  {name}  (actual={actual!r})")
        ok = ok and passed

    print(f"[smoke] {'ALL PASS' if ok else 'FAILED'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
