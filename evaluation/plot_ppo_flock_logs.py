"""Plot training + evaluation logs of ppo_flock_formation.

Reads:
    dumps/ppo_flock_11_vs_11_easy_stochastic_training_log.json
    dumps/ppo_flock_11_vs_11_easy_stochastic_evaluation_log.json
    dumps/ppo_flock_*_checkpoint_*pct.json   (optional, parameter evolution)

Writes:
    dumps/ppo_flock_training_eval_plots.png

Run (host Python, no Docker needed):
    python evaluation/plot_ppo_flock_logs.py [--train PATH] [--eval PATH] [--out PATH]
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUMPS = os.path.join(BASE, "dumps")
DEFAULT_TRAIN = os.path.join(DUMPS, "ppo_flock_11_vs_11_easy_stochastic_training_log.json")
DEFAULT_EVAL = os.path.join(DUMPS, "ppo_flock_11_vs_11_easy_stochastic_evaluation_log.json")
DEFAULT_OUT = os.path.join(DUMPS, "ppo_flock_training_eval_plots.png")


def rolling(vals, w=5):
    """Simple rolling mean (edge-padded)."""
    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return v
    out = np.empty_like(v)
    for i in range(len(v)):
        lo = max(0, i - w // 2)
        hi = min(len(v), i + w // 2 + 1)
        out[i] = v[lo:hi].mean()
    return out


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default=DEFAULT_TRAIN)
    ap.add_argument("--eval", default=DEFAULT_EVAL)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    for p in (args.train, args.eval):
        if not os.path.exists(p):
            sys.exit(f"missing log: {p}")

    tr = load(args.train)
    ev = load(args.eval)
    m_tr, m_ev = tr["football_metrics"], ev["football_metrics"]
    ep_r = tr["episode_rewards"]
    ed = tr["episodes_detail"]

    # parameter evolution: checkpoints + final + eval current params
    param_series = []  # (label, params)
    for cp in sorted(glob.glob(os.path.join(DUMPS, "ppo_flock_*_checkpoint_*pct.json"))):
        c = load(cp)
        label = c.get("checkpoint", os.path.basename(cp))
        param_series.append((label, c["learned_parameters"]))
    param_series.append((tr.get("checkpoint", "final"), tr["learned_parameters"]))
    if "learned_parameters" in ev:
        param_series.append(("eval", ev["learned_parameters"]))
    pnames = tr["parameter_names"]
    plabels = [p.replace("_", " ") for p in pnames]
    xs = list(range(len(param_series)))
    xticklabels = [s[0] for s in param_series]


    plt.style.use("seaborn-darkgrid")
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['cmr10', 'Computer Modern Roman', 'DejaVu Serif']
    
    fig, axes = plt.subplots(3, 2, figsize=(13, 14))
    fig.suptitle("PPO Flock Formation — 11_vs_11_easy_stochastic", fontsize=14)

    # (a) episode reward
    ax = axes[0, 0]
    ax.plot(ep_r, ".", alpha=0.4, color="gray", label="episode")
    if len(ep_r) >= 5:
        ax.plot(rolling(ep_r), "-", color="black", lw=1.5, label="rolling mean (5)")
    ax.axhline(0, color="red", lw=0.8, ls="--")
    ax.set(title="(a) Episode reward (training)", xlabel="episode",
           ylabel="reward")
    ax.legend(fontsize=8)

    # (b) GF / GA per game
    ax = axes[0, 1]
    gf = [e["gf"] for e in ed]
    ga = [e["ga"] for e in ed]
    ax.plot(gf, ".", alpha=0.4, color="black", label="GF")
    ax.plot(ga, ".", alpha=0.4, color="gray", label="GA")
    if len(ed) >= 5:
        ax.plot(rolling(gf), "-", color="black", lw=1.5, label="GF rolling")
        ax.plot(rolling(ga), "--", color="dimgray", lw=1.5, label="GA rolling")
    ax.set(title="(b) Goals for / against per game (training)", xlabel="episode",
           ylabel="goals")
    ax.legend(fontsize=8)

    # (c) learned parameter evolution
    ax = axes[1, 0]
    for i, name in enumerate(pnames):
        vals = [s[1].get(name, float("nan")) for s in param_series]
        ax.plot(xs, vals, marker="o", lw=1.5, label=plabels[i])
    ax.set(title="(c) Learned parameters across checkpoints", xticks=xs,
           xticklabels=xticklabels, ylabel="value", ylim=(-0.1, 2.1))
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=7, ncol=2)

    # (d) football metrics: training vs evaluation
    ax = axes[1, 1]
    keys = ["gf_per_game", "ga_per_game", "possession_pct", "pass_success_pct",
            "shot_accuracy_pct"]
    nice = ["GF/game", "GA/game", "Poss %", "Pass succ %", "Shot acc %"]
    tvals = [m_tr.get(k, 0) for k in keys]
    evals = [m_ev.get(k, 0) for k in keys]
    ix = np.arange(len(keys))
    b1 = ax.bar(ix - 0.18, tvals, 0.36, label=f"training ({m_tr['games']} game"
                f"{'s' if m_tr['games'] != 1 else ''})")
    b2 = ax.bar(ix + 0.18, evals, 0.36, label=f"evaluation ({m_ev['games']} game"
                f"{'s' if m_ev['games'] != 1 else ''})")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=7)
    ax.set(title="(d) Football metrics: training vs evaluation", xticks=ix,
           xticklabels=nice, ylabel="value")
    ax.legend(fontsize=8)

    # (e) shots & passes: volume vs success (training)
    ax = axes[2, 0]
    cats = ["shots", "shots_on\n_target", "passes", "passes\n_success"]
    vals = [m_tr.get("shots", 0), m_tr.get("shots_on_target", 0),
            m_tr.get("passes", 0), m_tr.get("passes_success", 0)]
    bars = ax.bar(cats, vals, color=["white", "gray", "white", "gray"],
                  edgecolor="black")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom",
                fontsize=9)
    ax.set(title="(e) Training event counts (total 204.8k steps)", ylabel="count")

    # (f) evaluation: score timeline + episode stats
    ax = axes[2, 1]
    det = ev.get("episodes_detail", [])
    gf_ev = [e["gf"] for e in det]
    ga_ev = [e["ga"] for e in det]
    steps = [e["steps"] for e in det]
    ax.barh(range(len(det)), gf_ev, color="black", label="GF")
    ax.barh(range(len(det)), ga_ev, left=gf_ev, color="white", edgecolor="black",
            hatch="//", label="GA")
    for i, e in enumerate(det):
        ax.text(e["gf"] + e["ga"] + 0.05, i,
                f"steps={e['steps']}, poss={e['possession_steps'] / max(1, e['steps']) * 100:.0f}%",
                va="center", fontsize=8)
    ax.set(title=f"(f) Evaluation per episode — total reward {ev.get('eval_total_reward')}",
           xlabel="goals", yticks=range(len(det)),
           yticklabels=[f"ep {i}" for i in range(len(det))])
    ax.legend(fontsize=8, loc="upper right")
    if ev.get("eval_episodes_requested", 0) > len(det):
        ax.set_title(
            f"(f) Evaluation per episode — reward {ev.get('eval_total_reward')}\n"
            f"only {len(det)}/{ev['eval_episodes_requested']} episodes "
            f"finished (max_steps={ev.get('eval_max_steps')} hit)",
            fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"saved: {args.out}")

    # ---- text summary for the terminal ----
    print("\n=== TRAINING ===")
    print(f"games={m_tr['games']}  W/D/L={m_tr['wins']}/{m_tr['draws']}/{m_tr['losses']}"
          f"  GF/gm={m_tr['gf_per_game']}  GA/gm={m_tr['ga_per_game']}")
    print(f"shots={m_tr['shots']} on-target={m_tr['shots_on_target']}"
          f" ({m_tr['shot_accuracy_pct']}%)  passes={m_tr['passes']}"
          f" success={m_tr['passes_success']} ({m_tr['pass_success_pct']}%)"
          f"  possession={m_tr['possession_pct']}%")
    r = np.asarray(ep_r, dtype=float)
    if len(r) >= 10:
        print(f"reward first10={r[:10].mean():.1f}  last10={r[-10:].mean():.1f}")
    print(f"params(final): {tr['learned_parameters']}")
    print("\n=== EVALUATION ===")
    print(f"games={m_ev['games']}  GF={m_ev['gf_total']} GA={m_ev['ga_total']}"
          f"  poss={m_ev['possession_pct']}%  reward={ev.get('eval_total_reward')}")
    print(f"params(eval): {ev['learned_parameters']}")


if __name__ == "__main__":
    main()