#!/usr/bin/env python
"""Generate grayscale figures for BAB_6 from authoritative n=4 JSON results.

Outputs to ../../Figures/ (repo root of the thesis):
  fig_baseline_dist.png   - 2 panels: goal difference & RCI_cat per algorithm
  fig_ablation_bar.png    - 8-arm ablation, game reward (sorted, +- std)
  fig_validity_scatter.png- 2 panels: RCI vs FAI (n=16), RCI vs GD (ablation)

Run from anywhere: python scripts/make_thesis_plots.py
"""
import json
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.abspath(os.path.join(ROOT, "..", "..", "Figures"))

BASE = os.path.join(ROOT, "evaluation_results_multiseed")
ABL = os.path.join(ROOT, "evaluation_results_ablation")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "black",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

ALGOS = [("hmarl", "HMARL"), ("ippo", "IPPO"), ("mappo", "MAPPO"), ("random", "Random")]
MARKERS = {"hmarl": "o", "ippo": "s", "mappo": "^", "random": "D"}
PRETTY = {
    "random_expert": "Expert acak",
    "full_hmarl": "Full HMARL",
    "no_fai": "Tanpa FAI",
    "no_reward_shaping": "Tanpa shaping",
    "no_gk_penalty": "Tanpa penalti GK",
    "no_ppr": "Tanpa PPR",
    "no_rci": "Tanpa RCI",
    "flat_policy": "Flat policy",
}


def dot_plot(ax, series, labels, ylabel, title):
    """Per-seed dot plot: 4 open circles + median bar + min-max range line.

    A boxplot is unusable at n=4: Q1/Q3 are interpolated, so the min and max
    land far outside the box and read as stray outliers.
    """
    for i, vals in enumerate(series, start=1):
        sv = sorted(vals)
        med = st.median(vals)
        ax.plot([i, i], [sv[0], sv[-1]], color="0.55", lw=0.8, zorder=1)
        ax.plot([i - 0.19, i + 0.19], [med, med], color="black", lw=1.5, zorder=3)
        xs = [i + (j - 1.5) * 0.075 for j in range(len(vals))]
        ax.scatter(xs, vals, s=13, facecolors="none", edgecolors="black",
                   linewidths=0.7, zorder=4)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.45, len(labels) + 0.55)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8, pad=6)
    ax.grid(axis="y", ls=":", lw=0.4, color="0.75")
    ax.set_axisbelow(True)


def main():
    base = json.load(open(os.path.join(BASE, "stat_test_results.json")))["results"]
    abl = json.load(open(os.path.join(ABL, "ablation_results.json")))["results"]
    val = json.load(open(os.path.join(BASE, "rci_validity.json")))  # noqa: F841

    os.makedirs(OUT, exist_ok=True)

    # --- Figure 1: baseline distributions -----------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    dot_plot(axes[0], [base[a]["goal_differences"] for a, _ in ALGOS],
             [n for _, n in ALGOS], "Goal difference",
             "(a) Goal difference per seed (n=4)")
    dot_plot(axes[1], [base[a]["rci"] for a, _ in ALGOS],
             [n for _, n in ALGOS], r"RCI$_{cat}$",
             r"(b) RCI$_{cat}$ per seed (n=4)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_baseline_dist.png"))
    plt.close(fig)

    # --- Figure 2: ablation bar ---------------------------------------------
    rows = []
    for name, a in abl.items():
        rows.append((name, st.mean(a["game_rewards"]), st.stdev(a["game_rewards"]),
                     st.mean(a["goal_differences"])))
    rows.sort(key=lambda r: r[1])
    labels = [PRETTY.get(r[0], r[0].replace("_", " ")) for r in rows]
    means = [r[1] for r in rows]
    errs = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    y = range(len(rows))
    ax.barh(list(y), means, height=0.62, color="0.85",
            edgecolor="black", linewidth=0.6, zorder=2)
    for i, (m, e) in enumerate(zip(means, errs)):
        ax.errorbar([m], [i], xerr=[e], fmt="none", ecolor="black", elinewidth=1.0,
                    capsize=3.5, capthick=1.0, zorder=4)
        ax.annotate("%.1f" % m, (m - e, i), xytext=(-4, 0),
                    textcoords="offset points", va="center", ha="right",
                    fontsize=6.5)
    lo = min(m - e for m, e in zip(means, errs))
    hi = max(m + e for m, e in zip(means, errs))
    ax.set_xlim(min(lo - 18, -200), max(0.0, hi) + 10)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Game reward per episode (mean $\\pm$ std, n=4)")
    ax.grid(axis="x", ls=":", lw=0.4, color="0.75")
    ax.set_axisbelow(True)
    ax.axvline(0, color="black", lw=0.7, zorder=4)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_ablation_bar.png"))
    plt.close(fig)

    # --- Figure 3: validity scatter -----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))

    rci, fai = [], []
    for a, label in ALGOS:
        xa = base[a]["rci"]
        ya = base[a]["fai_mean"]
        rci += list(xa)
        fai += list(ya)
        axes[0].scatter(xa, ya, s=17, facecolors="none", edgecolors="black",
                        linewidths=0.7, marker=MARKERS[a], label=label, zorder=3)
    axes[0].set_xlabel(r"RCI$_{cat}$")
    axes[0].set_ylabel("FAI")
    r_pooled = _pearson(rci, fai)
    axes[0].set_title("(a) RCI$_{cat}$ vs FAI, pooled (n=16), r=%.3f" % r_pooled,
                      fontsize=8, pad=6)
    axes[0].legend(fontsize=6, loc="lower left", framealpha=0.9, handletextpad=0.3,
                   borderpad=0.3, labelspacing=0.25)
    axes[0].grid(ls=":", lw=0.4, color="0.75")
    axes[0].set_axisbelow(True)

    order = sorted(abl.items(), key=lambda kv: st.mean(kv[1]["rci"]))
    xr = [st.mean(a["rci"]) for _, a in order]
    yr = [st.mean(a["goal_differences"]) for _, a in order]
    nr = [n for n, _ in order]
    axes[1].scatter(xr, yr, s=18, facecolors="none", edgecolors="black", linewidths=0.7)
    for k, (xn, yn) in enumerate(zip(xr, yr), start=1):
        axes[1].annotate(str(k), (xn, yn), fontsize=5.5, fontweight="bold",
                         xytext=(0, 0), textcoords="offset points",
                         ha="center", va="center", color="white",
                         bbox=dict(boxstyle="circle,pad=0.18", fc="0.25", ec="none"))
    legend = "\n".join("%d = %s" % (k, PRETTY.get(n, n.replace("_", " ")))
                       for k, n in enumerate(nr, start=1))
    axes[1].text(1.03, 0.5, legend, transform=axes[1].transAxes, fontsize=5.4,
                 va="center", ha="left", linespacing=1.5)
    axes[1].set_xlabel(r"RCI$_{cat}$ (arm mean)")
    axes[1].set_ylabel("Goal difference (arm mean)")
    axes[1].set_title("(b) Ablation arms (n=8), rho=-0.762", fontsize=8, pad=6)
    axes[1].grid(ls=":", lw=0.4, color="0.75")
    axes[1].set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_validity_scatter.png"))
    plt.close(fig)

    print("wrote to", OUT)
    for f in ("fig_baseline_dist.png", "fig_ablation_bar.png", "fig_validity_scatter.png"):
        p = os.path.join(OUT, f)
        print("  %-28s %8d bytes" % (f, os.path.getsize(p)))


def _pearson(x, y):
    mx, my = st.mean(x), st.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5
    return num / den if den else float("nan")


if __name__ == "__main__":
    main()
