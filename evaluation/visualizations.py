"""Visualization module for HMARL thesis evaluation.

Generates all plots required for thesis chapters:
1. Learning curve (cumulative reward vs episode)
2. RCI evolution over training
3. Comparative bar chart — all metrics
4. Role assignment heatmap
5. Formation snapshot
6. Action distribution by role
7. Macro strategy duration
8. Team compactness over time
9. Ablation study results
10. Tactic transition Sankey
11. Reward component breakdown
12. Metric correlation heatmap

All plots use matplotlib only (no seaborn dependency).
Academic style: clean lines, grayscale-compatible, 300 DPI.
"""

import os
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------
# Wong (2011) colorblind-safe palette
COLORS = {
    'hmarl': '#0072B2',    # Blue (primary — proposed method)
    'shppo': '#D55E00',    # Vermillion (strongest baseline)
    'mappo': '#CC79A7',    # Reddish purple
    'ippo': '#009E73',     # Bluish green
    'random': '#999999',   # Gray (neutral, weakest baseline)
    'accent': '#E69F00',   # Orange (for highlights)
    'secondary': '#56B4E9', # Sky blue
    'tertiary': '#F0E442', # Yellow (for accents only)
}

# Stable color mapping for methods — consistent across ALL figures
METHOD_COLORS = {
    'HMARL': COLORS['hmarl'],
    'SHPPO': COLORS['shppo'],
    'MAPPO': COLORS['mappo'],
    'IPPO': COLORS['ippo'],
    'Random': COLORS['random'],
    'Full HMARL': COLORS['hmarl'],
}

# Stable line styles for method distinction (training curves)
METHOD_LINESTYLES = {
    'HMARL': '-',
    'SHPPO': '--',
    'MAPPO': ':',
    'IPPO': '-.',
    'Random': (0, (1, 3)),  # dotted
}

# Stable markers for methods
METHOD_MARKERS = {
    'HMARL': 'o',
    'SHPPO': 's',
    'MAPPO': 'D',
    'IPPO': '^',
    'Random': 'v',
}

# Pattern fills for black-and-white printing
PATTERNS = {
    'hmarl': '',
    'ippo': '//',
    'shppo': '\\\\',
    'random': '..',
}

ROLE_NAMES = {
    0: 'GK', 1: 'CB', 2: 'LB', 3: 'RB', 4: 'DM',
    5: 'CM', 6: 'LM', 7: 'RM', 8: 'AM', 9: 'CF', 10: 'CF',
}

ROLE_COLORS = {
    'GK': '#e74c3c',   # Red
    'CB': '#3498db',   # Blue
    'LB': '#2ecc71',   # Green
    'RB': '#2ecc71',   # Green
    'DM': '#f39c12',   # Orange
    'CM': '#9b59b6',   # Purple
    'LM': '#1abc9c',   # Teal
    'RM': '#1abc9c',   # Teal
    'AM': '#e67e22',   # Dark orange
    'CF': '#e74c3c',   # Red
}

ACTION_CATEGORIES = {
    'Passing': {9, 10, 11},
    'Shooting': {12},
    'Movement': {1, 2, 3, 4, 5, 6, 7, 8, 13},
    'Ball\nControl': {17},
    'Defensive': {0, 14, 15, 16, 18},
}

MACRO_STRATEGIES = {0: 'High\nPressing', 1: 'Counter\nAttack', 2: 'Possession\nPlay'}


def _apply_style():
    """Apply clean academic plotting style for thesis figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        'lines.markeredgewidth': 0.5,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#cccccc',
        'mathtext.fontset': 'cm',
        'text.usetex': False,
    })


def _get_method_style(model_name: str) -> dict:
    """Return consistent color, linestyle, and marker for a model name."""
    # Normalize name
    name_upper = model_name.upper()
    name_map = {
        'HMARL': 'HMARL', 'IPPO': 'IPPO', 'SHppo': 'SHPPO',
        'MAPPO': 'MAPPO', 'RANDOM': 'Random', 'RANDOM BASELINE': 'Random',
        'FULL HMARL': 'Full HMARL', 'FULL_HMARL': 'Full HMARL',
    }
    key = name_map.get(name_upper, model_name)
    color = METHOD_COLORS.get(key, COLORS.get(model_name.lower(), '#333333'))
    ls = METHOD_LINESTYLES.get(key, '-')
    marker = METHOD_MARKERS.get(key, 'o')
    return {'color': color, 'linestyle': ls, 'marker': marker}


def _rolling_mean(data: List[float], window: int = 10) -> np.ndarray:
    """Compute rolling mean with given window size."""
    arr = np.array(data, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='valid')


def _rolling_std(data: List[float], window: int = 10) -> np.ndarray:
    """Compute rolling standard deviation."""
    arr = np.array(data, dtype=float)
    if len(arr) < window:
        return np.zeros_like(arr)
    result = np.zeros(len(arr) - window + 1)
    for i in range(len(result)):
        result[i] = np.std(arr[i:i + window])
    return result


# ---------------------------------------------------------------------------
# 1. Learning Curve
# ---------------------------------------------------------------------------
def plot_learning_curve(
    data: Dict[str, List[float]],
    output_path: str,
    window: int = 50,
    title: str = 'Learning Curve',
):
    """Plot cumulative reward vs episode for multiple models.

    Args:
        data: dict mapping model_name -> list of episode rewards
        output_path: path to save the figure
        window: rolling average window size
        title: plot title (ignored for thesis figures — use caption instead)
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for model, rewards in data.items():
        style = _get_method_style(model)
        episodes = np.arange(1, len(rewards) + 1)

        if len(rewards) >= window:
            smoothed = _rolling_mean(rewards, window)
            std = _rolling_std(rewards, window)
            x = episodes[window - 1:]
            ax.plot(x, smoothed, color=style['color'], linestyle=style['linestyle'],
                    label=model, linewidth=1.5)
            ax.fill_between(x, smoothed - std, smoothed + std,
                           alpha=0.12, color=style['color'])
        else:
            ax.plot(episodes, rewards, color=style['color'],
                    linestyle=style['linestyle'], label=model, linewidth=1.5)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    ax.legend(frameon=True, framealpha=0.9, edgecolor='#cccccc',
              loc='best')
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f'Saved: {output_path}')


# ---------------------------------------------------------------------------
# 2. RCI Evolution Over Training
# ---------------------------------------------------------------------------
def plot_rci_evolution(
    data: Dict[str, Dict[str, List[float]]],
    output_path: str,
    window: int = 50,
    title: str = 'RCI Evolution Over Training',
):
    """Plot RCI_cat and RCI_strict over episodes for multiple models.

    Uses subfigure labels (a), (b) for thesis integration.
    """
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for model, rci_data in data.items():
        style = _get_method_style(model)

        for ax, variant, label in [
            (ax1, 'rci_cat', r'RCI$_{cat}$'),
            (ax2, 'rci_strict', r'RCI$_{strict}$'),
        ]:
            values = rci_data.get(variant, [])
            if not values:
                continue
            episodes = np.arange(1, len(values) + 1)
            if len(values) >= window:
                smoothed = _rolling_mean(values, window)
                std = _rolling_std(values, window)
                x = episodes[window - 1:]
                ax.plot(x, smoothed, color=style['color'],
                        linestyle=style['linestyle'], label=model, linewidth=1.5)
                ax.fill_between(x, smoothed - std, smoothed + std,
                               alpha=0.12, color=style['color'])
            else:
                ax.plot(episodes, values, color=style['color'],
                        linestyle=style['linestyle'], label=model, linewidth=1.5)

    for ax, label in [(ax1, '(a)'), (ax2, '(b)')]:
        ax.set_xlabel('Episode')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(frameon=True, framealpha=0.9, edgecolor='#cccccc')
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
        ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=11,
                fontweight='bold', va='top')

    ax1.set_ylabel('RCI Value')
    ax1.set_title(r'Category-based ($RCI_{cat}$)')
    ax2.set_title(r'Strict ($RCI_{strict}$)')

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f'Saved: {output_path}')


# ---------------------------------------------------------------------------
# 3. Comparative Bar Chart — All Metrics
# ---------------------------------------------------------------------------
def plot_comparative_bars(
    data: Dict[str, Dict[str, float]],
    output_path: str,
    title: str = 'Comparative Metrics Evaluation',
):
    """Two-panel grouped bar chart for metrics across models.

    Panel (a): Performance metrics (WR, GD, Cumulative Reward) — separate scales
    Panel (b): Coordination metrics (PSR, PPR, Entropy, Compactness, FAI, RCI) — normalized to [0,1]
    """
    _apply_style()

    models = list(data.keys())
    n_models = len(models)

    # Panel (a): Performance metrics — keep original scales
    perf_metrics = ['wr', 'gd', 'cumulative_reward']
    perf_display = {'wr': 'Win Rate (%)', 'gd': 'Goal Diff.', 'cumulative_reward': 'Cum. Reward'}

    # Panel (b): Coordination metrics — all roughly in [0, 1] range
    coord_metrics = ['psr', 'ppr', 'positional_entropy', 'compactness_mean', 'fai_mean', 'rci_strict', 'rci_cat']
    coord_display = {
        'psr': 'PSR', 'ppr': 'PPR', 'positional_entropy': 'Entropy',
        'compactness_mean': 'Compactness', 'fai_mean': 'FAI',
        'rci_strict': r'RCI$_{strict}$', 'rci_cat': r'RCI$_{cat}$',
    }

    # Normalize coordination metrics to [0, 1] for fair comparison
    # Each metric is normalized to its observed range across models
    coord_data_norm = {}
    for m in coord_metrics:
        vals = [data[mdl].get(m, 0) for mdl in models]
        vmin, vmax = min(vals), max(vals)
        rng = vmax - vmin if vmax != vmin else 1.0
        coord_data_norm[m] = [(v - vmin) / rng for v in vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel (a): Performance
    x_perf = np.arange(len(perf_metrics))
    width = 0.7 / n_models
    for idx, model in enumerate(models):
        style = _get_method_style(model)
        values = [data[model].get(m, 0) for m in perf_metrics]
        offset = (idx - n_models / 2 + 0.5) * width
        ax1.bar(x_perf + offset, values, width, label=model,
                color=style['color'], edgecolor='white', linewidth=0.5)

    ax1.set_xticks(x_perf)
    ax1.set_xticklabels([perf_display[m] for m in perf_metrics])
    ax1.set_ylabel('Value')
    ax1.legend(frameon=True, framealpha=0.9, edgecolor='#cccccc')
    ax1.text(0.02, 0.98, '(a) Performance', transform=ax1.transAxes,
             fontsize=11, fontweight='bold', va='top')

    # Panel (b): Coordination (normalized)
    x_coord = np.arange(len(coord_metrics))
    for idx, model in enumerate(models):
        style = _get_method_style(model)
        values = [coord_data_norm[m][idx] for m in coord_metrics]
        offset = (idx - n_models / 2 + 0.5) * width
        ax2.bar(x_coord + offset, values, width, label=model,
                color=style['color'], edgecolor='white', linewidth=0.5)

    ax2.set_xticks(x_coord)
    ax2.set_xticklabels([coord_display[m] for m in coord_metrics], rotation=30, ha='right')
    ax2.set_ylabel('Normalized Value [0, 1]')
    ax2.legend(frameon=True, framealpha=0.9, edgecolor='#cccccc')
    ax2.set_ylim(0, 1.15)
    ax2.text(0.02, 0.98, '(b) Coordination', transform=ax2.transAxes,
             fontsize=11, fontweight='bold', va='top')

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f'Saved: {output_path}')


# ---------------------------------------------------------------------------
# 4. Role Assignment Heatmap
# ---------------------------------------------------------------------------
def plot_role_heatmap(
    game_states: List[Dict],
    output_path: str,
    team_side: str = 'left',
    grid_shape: Tuple[int, int] = (12, 8),
    title: str = 'Player Position Heatmap',
):
    """2D histogram on football pitch showing position frequency per role.

    Args:
        game_states: list of game state dicts (each with 'left_team' positions)
        output_path: path to save
        team_side: 'left' or 'right'
        grid_shape: (nx, ny) grid for the heatmap
        title: plot title
    """
    _apply_style()
    nx, ny = grid_shape
    pitch_length = 120.0
    pitch_width = 80.0

    key = f'{team_side}_team'
    roles_key = f'{team_side}_team_roles'

    # Collect all positions
    all_positions = []
    for gs in game_states:
        positions = gs.get(key, [])
        if len(positions) >= 11:
            all_positions.append(positions[:11])

    if not all_positions:
        print(f'No data for heatmap ({team_side} team)')
        return

    positions_arr = np.array(all_positions)  # (T, 11, 2)

    # Scale to pitch coordinates
    x_scaled = (positions_arr[:, :, 0] + 1.0) * 60.0
    y_scaled = (positions_arr[:, :, 1] + 0.42) * (80.0 / 0.84)

    # Get roles from first frame
    roles = game_states[0].get(roles_key, list(range(11)))[:11]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    # Create pitch background helper
    def draw_pitch(ax):
        # Pitch outline
        ax.add_patch(Rectangle((0, 0), pitch_length, pitch_width,
                              fill=False, edgecolor='gray', linewidth=0.8))
        # Center line
        ax.axvline(x=pitch_length/2, color='gray', linewidth=0.5, alpha=0.5)
        # Center circle
        circle = plt.Circle((pitch_length/2, pitch_width/2), 9.15,
                           fill=False, edgecolor='gray', linewidth=0.5, alpha=0.5)
        ax.add_patch(circle)
        ax.set_xlim(0, pitch_length)
        ax.set_ylim(0, pitch_width)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(11):
        if i >= len(axes):
            break
        ax = axes[i]
        draw_pitch(ax)

        role_id = roles[i] if i < len(roles) else i
        role_name = ROLE_NAMES.get(role_id, f'P{i}')

        px = x_scaled[:, i]
        py = y_scaled[:, i]

        # 2D histogram
        h, xedges, yedges = np.histogram2d(
            px, py, bins=[nx, ny],
            range=[[0, pitch_length], [0, pitch_width]],
        )

        # Plot heatmap
        im = ax.pcolormesh(xedges, yedges, h.T, cmap='hot', alpha=0.8)

        ax.set_title(f'{role_name} (n={len(px)})', fontsize=10, fontweight='bold')

    # Add subplot label (a) in top-left of first axes
    axes[0].text(0.02, 0.98, '(a)', transform=axes[0].transAxes,
                 fontsize=11, fontweight='bold', va='top')

    # Remove extra axes
    for j in range(11, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f'Saved: {output_path}')


# ---------------------------------------------------------------------------
# 5. Formation Snapshot
# ---------------------------------------------------------------------------
def plot_formation_snapshot(
    game_states: List[Dict],
    output_path: str,
    timestep: int = 0,
    team_side: str = 'left',
    show_target: bool = True,
    title: str = 'Formation Snapshot',
):
    """Scatter plot of agent positions at a specific timestep on the field.

    Args:
        game_states: list of game state dicts
        output_path: path to save
        timestep: which frame to plot
        team_side: 'left' or 'right'
        show_target: whether to show ideal formation positions
        title: plot title
    """
    _apply_style()
    pitch_length = 120.0
    pitch_width = 80.0

    if timestep >= len(game_states):
        timestep = len(game_states) - 1

    gs = game_states[timestep]
    key = f'{team_side}_team'
    roles_key = f'{team_side}_team_roles'

    positions = gs.get(key, [])[:11]
    roles = gs.get(roles_key, list(range(11)))[:11]

    # Scale to pitch
    x = np.array([p[0] for p in positions]) * 60.0 + 60.0
    y = np.array([p[1] for p in positions]) * (80.0 / 0.84) + 33.6

    # Formation template targets (scaled to pitch)
    targets = np.array([
        [5.0, 40.0],    # GK
        [30.0, 28.0],   # CB
        [30.0, 52.0],   # CB
        [42.0, 12.0],   # LB
        [42.0, 68.0],   # RB
        [54.0, 40.0],   # DM
        [66.0, 24.0],   # CM
        [66.0, 56.0],   # CM
        [78.0, 12.0],   # LM
        [78.0, 68.0],   # RM
        [90.0, 40.0],   # CF
    ])

    fig, ax = plt.subplots(figsize=(10, 7))

    # Draw pitch
    ax.add_patch(Rectangle((0, 0), pitch_length, pitch_width,
                          fill=False, edgecolor='gray', linewidth=1))
    ax.axvline(x=pitch_length/2, color='gray', linewidth=0.5, alpha=0.5)
    circle = plt.Circle((pitch_length/2, pitch_width/2), 9.15,
                       fill=False, edgecolor='gray', linewidth=0.5, alpha=0.5)
    ax.add_patch(circle)
    # Penalty areas
    ax.add_patch(Rectangle((0, 16.5), 16.5, 47.0, fill=False, edgecolor='gray', linewidth=0.5))
    ax.add_patch(Rectangle((pitch_length - 16.5, 16.5), 16.5, 47.0,
                          fill=False, edgecolor='gray', linewidth=0.5))

    # Plot actual positions
    for i in range(min(11, len(positions))):
        role_id = roles[i] if i < len(roles) else i
        role_name = ROLE_NAMES.get(role_id, f'P{i}')
        color = ROLE_COLORS.get(role_name, '#333333')

        ax.scatter(x[i], y[i], c=color, s=500, zorder=3,
                  edgecolors='white', linewidth=1.5)
        ax.annotate(role_name, (x[i], y[i]), fontsize=12, fontweight='bold',
                   color='white', ha='center', va='center', zorder=4)

    # Plot target formation
    if show_target:
        ax.scatter(targets[:, 0], targets[:, 1], c='none',
                  s=500, edgecolors='gray', linewidth=1, linestyle='--',
                  zorder=2, alpha=0.5, label='Target Formation')

    ax.set_xlim(-2, pitch_length + 2)
    ax.set_ylim(-2, pitch_width + 2)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{title} (t={timestep})', fontsize=12)
    ax.legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f'Saved: {output_path}')


# ---------------------------------------------------------------------------
# 6. Action Distribution by Role
# ---------------------------------------------------------------------------
def plot_action_distribution(
    actual_actions: List[List[int]],
    roles: List[int],
    output_path: str,
    title: str = 'Action Distribution by Role',
):
    """Stacked bar chart: action category frequency per role.

    Args:
        actual_actions: list of (num_agents,) action lists per timestep
        roles: list of role IDs for each agent (len=11)
        output_path: path to save
        title: plot title
    """
    _apply_style()

    if not actual_actions:
        print('No action data for distribution plot')
        return

    num_agents = len(actual_actions[0])
    categories = list(ACTION_CATEGORIES.keys())
    n_cats = len(categories)

    # Count actions per agent per category
    counts = np.zeros((num_agents, n_cats))
    for actions in actual_actions:
        for agent_idx in range(min(num_agents, len(actions))):
            action = actions[agent_idx]
            for cat_idx, (cat_name, cat_actions) in enumerate(ACTION_CATEGORIES.items()):
                if action in cat_actions:
                    counts[agent_idx, cat_idx] += 1
                    break

    # Normalize to percentages
    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    percentages = counts / totals * 100

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(num_agents)
    bottom = np.zeros(num_agents)
    cat_colors = [COLORS['hmarl'], '#E69F00', COLORS['secondary'], COLORS['accent'], '#999999']

    for cat_idx in range(n_cats):
        ax.bar(x, percentages[:, cat_idx], bottom=bottom,
              label=categories[cat_idx], color=cat_colors[cat_idx],
              edgecolor='white', linewidth=0.5, width=0.7)
        bottom += percentages[:, cat_idx]

    # X-axis labels with role names
    labels = [ROLE_NAMES.get(roles[i] if i < len(roles) else i, f'P{i}')
             for i in range(num_agents)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Action Frequency (%)')
    ax.set_xlabel('Player Role')
    ax.set_title(title)
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=8)
    ax.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f'Saved: {output_path}')


# ---------------------------------------------------------------------------
# 7. Macro Strategy Duration (Timeline)
# ---------------------------------------------------------------------------
def plot_macro_strategy_timeline(
    strategies,
    output_path: str,
    title: str = 'Macro Strategy Timeline',
    window: int = 100,
):
    r"""Timeline visualization of macro strategy activation over match duration.

    Accepts a single episode (List[int]) or multiple episodes
    (List[List[int]]).  Multi-episode: stacked area of dominant strategy
    proportion per timestep, averaged across episodes with rolling window.

    Args:
        strategies: single episode list of strategy indices, OR list of
                    per-episode strategy lists (each inner list = one episode).
        output_path: path to save
        title: plot title
        window: rolling window for multi-episode averaging (default 100)
    """
    _apply_style()
    if not strategies:
        print('No strategy data for timeline plot')
        return

    strategy_labels = ['High Pressing', 'Counter Attack', 'Possession Play']
    strategy_colors = [COLORS['hmarl'], COLORS['secondary'], COLORS['accent']]
    num_strategies = len(strategy_labels)

    # --- Detect single vs multi episode ---
    is_multi = (
        isinstance(strategies[0], (list, np.ndarray))
        and len(strategies) > 1
    )

    if is_multi:
        # Pad episodes to equal length, then build (N_episodes, T) array
        max_len = max(len(ep) for ep in strategies)
        arr = np.full((len(strategies), max_len), -1, dtype=float)
        for i, ep in enumerate(strategies):
            arr[i, :len(ep)] = np.array(ep, dtype=float)

        T = max_len
        if T < window:
            window = max(1, T // 4)
        x = np.arange(window - 1, T)
        proportions = np.zeros((num_strategies, len(x)))
        for s in range(num_strategies):
            indicator = (arr == s).astype(float)
            # Average across episodes first, then rolling mean on 1D
            mean_indicator = indicator.mean(axis=0)  # (T,)
            proportions[s] = _rolling_mean(mean_indicator, window)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.stackplot(x, proportions * 100, labels=strategy_labels,
                     colors=strategy_colors, alpha=0.85)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Strategy Dominance (%)')
        ax.set_title(title + '  (n=%d eps, w=%d)' % (len(strategies), window))
        ax.legend(loc='upper right', fontsize=8, frameon=True, framealpha=0.9)
    else:
        # Single episode --- original color-coded timeline
        strategies = np.array(strategies)
        fig, ax = plt.subplots(figsize=(12, 2.5))
        cmap = plt.cm.colors.ListedColormap(strategy_colors)
        ax.imshow(strategies.reshape(1, -1), aspect='auto', cmap=cmap,
                 interpolation='nearest', extent=[0, len(strategies), 0, 1])
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=c, label=l)
            for c, l in zip(strategy_colors, strategy_labels)
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
                 frameon=True, framealpha=0.9)
        ax.set_yticks([])
        ax.set_title(title)

    ax.set_xlabel('Timestep')
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print('Saved: %s' % output_path)


# ---------------------------------------------------------------------------
# 8. Team Compactness Over Time
# ---------------------------------------------------------------------------
def plot_compactness_over_time(
    data: Dict[str, Any],
    output_path: str,
    window: int = 100,
    title: str = 'Team Compactness Over Time',
):
    r"""Line plot: team compactness rho_tc(t) across match duration.

    Accepts per-model data as either:
      - List[float]        (single episode, backward-compatible)
      - List[List[float]]  (multiple episodes: mean +/- std band)

    Args:
        data: dict mapping model_name -> compactness values
              (single list OR list of per-episode lists)
        output_path: path to save
        window: smoothing window
        title: plot title
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 4.5))

    for model, values in data.items():
        style = _get_method_style(model)
        color = style['color']

        # --- Detect single vs multi episode ---
        is_multi = (
            isinstance(values, (list, np.ndarray))
            and len(values) > 0
            and isinstance(values[0], (list, np.ndarray))
        )

        if is_multi:
            # Pad episodes to equal length
            max_len = max(len(ep) for ep in values)
            arr = np.full((len(values), max_len), np.nan, dtype=float)
            for i, ep in enumerate(values):
                arr[i, :len(ep)] = np.array(ep, dtype=float)

            mean_ts = np.nanmean(arr, axis=0)
            std_ts  = np.nanstd(arr, axis=0)

            timesteps = np.arange(max_len)
            if max_len >= window:
                mean_sm = _rolling_mean(mean_ts, window)
                std_sm  = _rolling_mean(std_ts, window)
                x = timesteps[window - 1:]
                ax.fill_between(x, mean_sm - std_sm, mean_sm + std_sm,
                                color=color, alpha=0.15)
                ax.plot(x, mean_sm, color=color, label=model, linewidth=1.2)
            else:
                ax.fill_between(timesteps, mean_ts - std_ts, mean_ts + std_ts,
                                color=color, alpha=0.15)
                ax.plot(timesteps, mean_ts, color=color, label=model,
                        linewidth=1.2)
        else:
            # Single episode (original behavior)
            timesteps = np.arange(len(values))
            if len(values) >= window:
                smoothed = _rolling_mean(values, window)
                x = timesteps[window - 1:]
                ax.plot(x, smoothed, color=color, label=model, linewidth=1.2)
            else:
                ax.plot(timesteps, values, color=color, label=model,
                        linewidth=1.2)

    ax.set_xlabel('Timestep')
    ax.set_ylabel(r'Team Compactness ($\rho_{tc}$)')
    ax.set_title(title)
    ax.legend(frameon=True, framealpha=0.9, edgecolor='#cccccc')

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print('Saved: %s' % output_path)


# ---------------------------------------------------------------------------
# 9. Ablation Study Results
# ---------------------------------------------------------------------------
def plot_ablation_study(
    data: Dict[str, float],
    output_path: str,
    title: str = 'Ablation Study — RCI Contribution by Hierarchy Level',
):
    """Horizontal bar chart: metric value with each component disabled.

    Bars sorted by value (descending) for immediate comparison.
    Full HMARL shown in primary color; ablated configs in lighter shades.
    """
    _apply_style()

    # Sort by value (descending)
    sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)
    configs = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    n = len(configs)

    # Color: Full HMARL gets primary, others get progressively lighter
    bar_colors = []
    for cfg in configs:
        if 'Full' in cfg or 'HMARL' in cfg.upper():
            bar_colors.append(COLORS['hmarl'])
        else:
            bar_colors.append('#90B4CE')  # Light blue for ablated

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.barh(range(n), values, color=bar_colors, edgecolor='white',
                   linewidth=0.5, height=0.6)

    ax.set_yticks(range(n))
    ax.set_yticklabels(configs)
    ax.set_xlabel(r'RCI$_{cat}$')
    ax.invert_yaxis()  # Highest value on top

    # Reference line for full model
    full_val = values[0]  # Already sorted descending
    ax.axvline(x=full_val, color=COLORS['hmarl'], linestyle='--',
               alpha=0.3, linewidth=1)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', va='center', fontsize=9)

    ax.set_xlim(0, max(values) * 1.15 if values else 1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f'Saved: {output_path}')


# ---------------------------------------------------------------------------
# 10. Tactic Transition Sankey (simplified as stacked area)
# ---------------------------------------------------------------------------
def plot_tactic_transitions(
    macro_strategies,
    sub_goals_per_agent,
    output_path: str,
    title: str = 'Tactic Transition Overview',
    window: int = 200,
):
    r"""Stacked area of sub-goal frequency over time.

    Accepts single episode (flat lists) or multiple episodes (list of lists).
    Multi-episode: frequency matrix computed per episode then averaged.

    Args:
        macro_strategies: single-episode list of strategy indices per timestep,
                          OR list of per-episode strategy lists.
        sub_goals_per_agent: single-episode list of sub-goal lists per timestep,
                             OR list of per-episode sub-goal lists.
        output_path: path to save
        title: plot title
        window: aggregation window in timesteps (default 200)
    """
    _apply_style()
    if not sub_goals_per_agent:
        print('No sub-goal data for tactic transitions')
        return

    num_subgoals = 5
    subgoal_names = ['Zonal\nMarking', 'Build-up', 'Wing\nAttack',
                     'Man\nMarking', 'Clearance']
    colors = [COLORS['hmarl'], COLORS['secondary'], COLORS['accent'], '#CC79A7', COLORS['accent']]

    # --- Detect single vs multi episode ---
    first = sub_goals_per_agent[0]
    is_multi = (
        isinstance(first, (list, np.ndarray))
        and len(first) > 0
        and isinstance(first[0], (list, np.ndarray))
    )

    if is_multi:
        # Pad episodes to equal length
        max_len = max(len(ep) for ep in sub_goals_per_agent)
        n_eps = len(sub_goals_per_agent)
        n_windows = max(1, max_len // window)

        # Accumulate frequency matrices across episodes
        freq_matrix = np.zeros((n_windows, num_subgoals))
        for ep in sub_goals_per_agent:
            ep_arr = ep
            for w in range(n_windows):
                start = w * window
                end = min(start + window, len(ep_arr))
                for t in range(start, end):
                    for sg in ep_arr[t]:
                        if 0 <= sg < num_subgoals:
                            freq_matrix[w, sg] += 1

        # Average across episodes, then normalize rows
        freq_matrix = freq_matrix / n_eps
        row_sums = freq_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        freq_matrix = freq_matrix / row_sums * 100

        fig, ax = plt.subplots(figsize=(10, 4.5))
        x = np.arange(n_windows) * window
        ax.stackplot(x, freq_matrix.T, labels=subgoal_names,
                     colors=colors, alpha=0.85)
        ax.set_title(title + '  (n=%d eps, w=%d)' % (n_eps, window))
    else:
        # Single episode --- original behavior
        sga_flat = sub_goals_per_agent

        n_windows = max(1, len(sga_flat) // window)
        freq_matrix = np.zeros((n_windows, num_subgoals))
        for w in range(n_windows):
            start = w * window
            end = min(start + window, len(sga_flat))
            for t in range(start, end):
                for sg in sga_flat[t]:
                    if 0 <= sg < num_subgoals:
                        freq_matrix[w, sg] += 1

        row_sums = freq_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        freq_matrix = freq_matrix / row_sums * 100

        fig, ax = plt.subplots(figsize=(10, 4.5))
        x = np.arange(n_windows) * window
        ax.stackplot(x, freq_matrix.T, labels=subgoal_names,
                     colors=colors, alpha=0.85)
        ax.set_title(title)

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Sub-Goal Frequency (%)')
    ax.legend(loc='upper right', fontsize=8, frameon=True, framealpha=0.9)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print('Saved: %s' % output_path)


# ---------------------------------------------------------------------------
# 11. Reward Component Breakdown
# ---------------------------------------------------------------------------
def plot_reward_breakdown(
    data: Dict[str, List[float]],
    output_path: str,
    window: int = 100,
    title: str = 'Reward Component Breakdown Over Training',
):
    """Stacked area chart: contribution of each reward component over training.

    Args:
        data: dict with keys 'game_reward', 'r_high', 'r_mid', 'r_low'
              each mapping to a list of per-episode values
        output_path: path to save
        title: plot title
    """
    _apply_style()

    components = ['game_reward', 'r_high', 'r_mid', 'r_low']
    labels = ['Game Reward ($r_{game}$)', 'FAI ($r_{high}$)',
              'PPR ($r_{mid}$)', 'RCI ($r_{low}$)']
    colors = [COLORS['hmarl'], COLORS['secondary'], COLORS['accent'], '#CC79A7']

    fig, ax = plt.subplots(figsize=(10, 5))

    # Smooth each component
    all_data = []
    min_len = float('inf')
    for comp in components:
        vals = data.get(comp, [])
        if vals:
            if len(vals) >= window:
                smoothed = _rolling_mean(vals, window)
            else:
                smoothed = np.array(vals)
            all_data.append(smoothed)
            min_len = min(min_len, len(smoothed))
        else:
            all_data.append(np.array([]))

    if min_len == float('inf') or min_len == 0:
        print('No reward data for breakdown plot')
        plt.close(fig)
        return

    # Truncate to min_len
    all_data = [d[:min_len] for d in all_data]
    x = np.arange(min_len)

    ax.stackplot(x, *all_data, labels=labels, colors=colors, alpha=0.8)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward Contribution (cumulative per episode)')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8, frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f'Saved: {output_path}')


# ---------------------------------------------------------------------------
# 12. Metric Correlation Heatmap
# ---------------------------------------------------------------------------
def plot_metric_correlation(
    data: Dict[str, List[float]],
    output_path: str,
    title: str = 'Metric Correlation Matrix',
):
    """Pearson correlation matrix between coordination metrics.

    Args:
        data: dict mapping metric_name -> list of values (same length)
              e.g. {'rci_cat': [...], 'fai_mean': [...], 'psr': [...], ...}
        output_path: path to save
        title: plot title
    """
    _apply_style()

    metrics = list(data.keys())
    n = len(metrics)

    # Build matrix
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            arr_i = np.array(data[metrics[i]])
            arr_j = np.array(data[metrics[j]])
            min_len = min(len(arr_i), len(arr_j))
            if min_len < 2:
                matrix[i, j] = 0.0
            else:
                corr = np.corrcoef(arr_i[:min_len], arr_j[:min_len])[0, 1]
                matrix[i, j] = corr if not np.isnan(corr) else 0.0

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # Display names
    display = {
        'rci_cat': r'RCI$_{cat}$',
        'rci_strict': r'RCI$_{strict}$',
        'fai_mean': 'FAI',
        'positional_entropy': 'Entropy',
        'compactness_mean': 'Compactness',
        'psr': 'PSR',
        'ppr': 'PPR',
        'wr': 'WR',
        'cumulative_reward': 'Cum. Reward',
        'goal_difference': 'Goal Diff.',
        'goals_for': 'Goals For',
        'goals_against': 'Goals Against',
        'defence_midfield_gap': 'Def-Mid Gap',
        'midfield_attack_gap': 'Mid-Att Gap',
        'overall_spread': 'Spread',
        'convex_hull_mean': 'Convex Hull',
        'bpr_mean': 'BPR',
    }
    labels = [display.get(m, m) for m in metrics]

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate with r values
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = 'white' if abs(val) > 0.6 else 'black'
            if i == j:
                ax.text(j, i, '1.00', ha='center', va='center',
                       fontsize=8, color=color)
            else:
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       fontsize=8, color=color)

    # No figure title — thesis caption serves this role
    fig.colorbar(im, ax=ax, shrink=0.8, label='Pearson r')

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f'Saved: {output_path}')


# ---------------------------------------------------------------------------
# Generate all plots from saved evaluation results
# ---------------------------------------------------------------------------
def generate_all_plots(
    results_dir: str = 'evaluation_results',
    output_dir: str = 'evaluation_results/plots',
    game_states: Optional[List[Dict]] = None,
    actions: Optional[List[List[int]]] = None,
    roles: Optional[List[int]] = None,
    strategies: Optional[List[int]] = None,
    sub_goals: Optional[List[List[int]]] = None,
    reward_breakdown: Optional[Dict[str, List[float]]] = None,
    ablation_data: Optional[Dict[str, float]] = None,
    compactness_data: Optional[Dict[str, List[float]]] = None,
    model_comparisons: Optional[Dict[str, Dict[str, float]]] = None,
):
    """Generate all visualization plots from saved data.

    Args:
        results_dir: directory containing hmarl_results.json, hmarl_episodes.json
        output_dir: directory to save plots
        game_states: list of game state dicts (for heatmaps, snapshots)
        actions: list of action lists (for action distribution, tactics)
        roles: list of role IDs (for action distribution)
        strategies: list of macro strategy indices (for timeline)
        sub_goals: list of sub-goal lists (for tactic transitions)
        reward_breakdown: dict of reward component lists (for breakdown plot)
        ablation_data: dict mapping config -> RCI value (for ablation)
        compactness_data: dict mapping model -> compactness list (for compactness)
        model_comparisons: dict mapping model -> metrics dict (for bar chart)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load saved results
    hmarl_path = os.path.join(results_dir, 'hmarl_results.json')
    episodes_path = os.path.join(results_dir, 'hmarl_episodes.json')

    hmarl_metrics = {}
    episode_data = {}

    if os.path.exists(hmarl_path):
        with open(hmarl_path) as f:
            hmarl_metrics = json.load(f)
        print(f'Loaded: {hmarl_path}')

    if os.path.exists(episodes_path):
        with open(episodes_path) as f:
            episode_data = json.load(f)
        print(f'Loaded: {episodes_path}')

    # --- Plot 3: Comparative bar chart ---
    if model_comparisons:
        plot_comparative_bars(
            model_comparisons,
            os.path.join(output_dir, '03_comparative_metrics.png'),
        )

    # --- Plot 4: Role heatmap ---
    if game_states:
        plot_role_heatmap(
            game_states,
            os.path.join(output_dir, '04_role_heatmap.png'),
        )

    # --- Plot 5: Formation snapshot ---
    if game_states:
        # Plot at start, middle, and late timesteps
        for ts, label in [(0, 'start'), (len(game_states)//2, 'mid'), (-1, 'late')]:
            actual_ts = ts if ts >= 0 else len(game_states) + ts
            plot_formation_snapshot(
                game_states,
                os.path.join(output_dir, f'05_formation_{label}.png'),
                timestep=actual_ts,
                title=f'Formation Snapshot ({label})',
            )

    # --- Plot 6: Action distribution ---
    if actions and roles:
        plot_action_distribution(
            actions, roles,
            os.path.join(output_dir, '06_action_distribution.png'),
        )

    # --- Plot 7: Macro strategy timeline ---
    if strategies:
        plot_macro_strategy_timeline(
            strategies,
            os.path.join(output_dir, '07_strategy_timeline.png'),
        )

    # --- Plot 8: Compactness over time ---
    if compactness_data:
        plot_compactness_over_time(
            compactness_data,
            os.path.join(output_dir, '08_compactness.png'),
        )

    # --- Plot 9: Ablation study ---
    if ablation_data:
        plot_ablation_study(
            ablation_data,
            os.path.join(output_dir, '09_ablation.png'),
        )

    # --- Plot 10: Tactic transitions ---
    if strategies and sub_goals:
        plot_tactic_transitions(
            strategies, sub_goals,
            os.path.join(output_dir, '10_tactic_transitions.png'),
        )

    # --- Plot 11: Reward breakdown ---
    if reward_breakdown:
        plot_reward_breakdown(
            reward_breakdown,
            os.path.join(output_dir, '11_reward_breakdown.png'),
        )

    # --- Plot 12: Correlation heatmap ---
    if model_comparisons:
        # Core metrics for correlation — exclude auxiliary stats
        CORE_CORR_METRICS = {
            'wr', 'goal_difference', 'goals_for', 'goals_against',
            'cumulative_reward', 'psr', 'ppr', 'positional_entropy',
            'compactness_mean', 'fai_mean',
            'rci_strict', 'rci_cat',
            'defence_midfield_gap', 'midfield_attack_gap', 'overall_spread',
            'convex_hull_mean', 'bpr_mean',
        }
        all_metric_data = {}
        for model, metrics_dict in model_comparisons.items():
            for k, v in metrics_dict.items():
                if isinstance(v, (int, float)) and k in CORE_CORR_METRICS:
                    if k not in all_metric_data:
                        all_metric_data[k] = []
                    all_metric_data[k].append(v)

        if len(all_metric_data) >= 3:
            plot_metric_correlation(
                all_metric_data,
                os.path.join(output_dir, '12_correlation_heatmap.png'),
            )

    print(f'\nAll plots saved to: {output_dir}')


# ---------------------------------------------------------------------------
# 13. Pass Network Graph
# ---------------------------------------------------------------------------
def plot_pass_network(
    pass_matrix: np.ndarray,
    output_path: str,
    roles: Optional[List[int]] = None,
    title: str = 'Pass Network',
):
    """Weighted directed graph of pass connections between players.

    Args:
        pass_matrix: (11, 11) pass count matrix from pass_network_matrix()
        output_path: path to save figure
        roles: list of 11 role IDs for labeling; defaults to 0-10
        title: plot title
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    n = pass_matrix.shape[0]
    # Fixed positions: GK left, defenders, midfielders, attackers right
    # Layout: 4-2-4 on a half-pitch
    positions = {
        0: (0.10, 0.50),   # GK
        1: (0.30, 0.70),   # CB
        2: (0.30, 0.25),   # LB
        3: (0.30, 0.75),   # RB
        4: (0.50, 0.35),   # DM
        5: (0.50, 0.65),   # CM
        6: (0.70, 0.15),   # LM
        7: (0.70, 0.40),   # RM
        8: (0.70, 0.60),   # AM
        9: (0.70, 0.85),   # CF
        10: (0.85, 0.50),  # CF2
    }

    role_names = roles if roles else list(range(n))
    labels = [ROLE_NAMES.get(role_names[i], str(i)) for i in range(n)]

    # Draw edges with width proportional to pass count
    max_pass = max(pass_matrix.max(), 1)
    for i in range(n):
        for j in range(n):
            if i == j or pass_matrix[i, j] == 0:
                continue
            width = 0.5 + 3.0 * (pass_matrix[i, j] / max_pass)
            xi, yi = positions[i]
            xj, yj = positions[j]
            ax.annotate(
                '', xy=(xj, yj), xytext=(xi, yi),
                arrowprops=dict(
                    arrowstyle='->', lw=width,
                    color='#666666', alpha=0.5,
                    connectionstyle='arc3,rad=0.1',
                ),
            )

    # Draw nodes
    for i in range(n):
        x, y = positions[i]
        total_pass = pass_matrix[i].sum()
        size = 300 + 700 * (total_pass / max(max_pass, 1))
        role_label = ROLE_NAMES.get(role_names[i], '?')
        color = ROLE_COLORS.get(role_label, '#888888')
        ax.scatter(x, y, s=size, c=color, edgecolors='#333333',
                   linewidths=0.8, zorder=3)
        ax.annotate(labels[i], (x, y), textcoords='offset points',
                    xytext=(0, -14), ha='center', fontsize=8, fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f'Saved: {output_path}')


# ---------------------------------------------------------------------------
# 14. Inter-Agent Distance Per Line
# ---------------------------------------------------------------------------
def plot_iad_per_line(
    data: Dict[str, float],
    output_path: str,
    title: str = 'Inter-Agent Distance by Tactical Line',
):
    """Bar chart of mean gaps between defence/midfield/attack lines.

    Args:
        data: dict with keys 'defence_midfield_gap', 'midfield_attack_gap',
              'overall_spread'
        output_path: path to save figure
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    labels = ['Defence ↔ Midfield', 'Midfield ↔ Attack', 'Overall Spread']
    values = [
        data.get('defence_midfield_gap', 0),
        data.get('midfield_attack_gap', 0),
        data.get('overall_spread', 0),
    ]
    colors = [COLORS['hmarl'], COLORS['accent'], COLORS['secondary']]

    bars = ax.bar(labels, values, color=colors, edgecolor='#333333', width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Distance (pitch units)')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f'Saved: {output_path}')


# ---------------------------------------------------------------------------
# 15. Convex Hull Area Distribution
# ---------------------------------------------------------------------------
def plot_convex_hull(
    game_states: List[Dict],
    output_path: str,
    team_side: str = 'left',
    title: str = 'Convex Hull Area Over Time',
):
    """Plot per-timestep convex hull area as a time series.

    Args:
        game_states: list of game state dicts
        output_path: path to save figure
        team_side: 'left' or 'right'
        title: plot title
    """
    from hmarl.metrics import _convex_hull_area_numpy

    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 4))

    key = f'{team_side}_team'
    areas = []
    for gs in game_states:
        positions = gs.get(key, [])
        if len(positions) < 11:
            areas.append(0.0)
            continue
        outfield_pos = np.array(positions[1:11])
        x_s = (outfield_pos[:, 0] + 1.0) * 60.0
        y_s = (outfield_pos[:, 1] + 0.42) * (80.0 / 0.84)
        pts = np.column_stack([x_s, y_s])
        areas.append(_convex_hull_area_numpy(pts))

    if not areas:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
    else:
        ax.plot(range(len(areas)), areas, color=COLORS['secondary'],
                linewidth=0.8, alpha=0.8)
        if len(areas) >= 20:
            smoothed = np.convolve(areas, np.ones(20) / 20, mode='valid')
            ax.plot(range(19, len(areas)), smoothed,
                    color=COLORS['accent'], linewidth=2, label='Rolling mean (20)')
            ax.legend()

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Convex Hull Area')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f'Saved: {output_path}')


# ---------------------------------------------------------------------------
# 16. Action Transition Probabilities Heatmap
# ---------------------------------------------------------------------------
def plot_action_transitions(
    trans_probs: np.ndarray,
    output_path: str,
    title: str = 'Action Transition Probabilities',
    max_actions: int = 19,
):
    """Heatmap of action-to-action transition probabilities.

    Args:
        trans_probs: (num_actions, num_actions) probability matrix
        output_path: path to save figure
        title: plot title
        max_actions: limit axes to this many actions for readability
    """
    _apply_style()
    n = min(trans_probs.shape[0], max_actions)
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(trans_probs[:n, :n], cmap='viridis', aspect='equal')
    ax.set_xlabel('Next Action')
    ax.set_ylabel('Current Action')
    ax.set_title(title)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('P(next | current)')
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f'Saved: {output_path}')


# ---------------------------------------------------------------------------
# 17. Ball Progression Rate Distribution
# ---------------------------------------------------------------------------
def plot_bpr_distribution(
    game_states: List[Dict],
    output_path: str,
    title: str = 'Ball Progression Rate Distribution',
):
    """Histogram of per-timestep ball x-displacement when team owns the ball.

    Args:
        game_states: list of game state dicts
        output_path: path to save figure
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    deltas = []
    for t in range(len(game_states) - 1):
        ball_x = game_states[t].get('ball', [0, 0, 0])[0]
        ball_x_next = game_states[t + 1].get('ball', [0, 0, 0])[0]
        if game_states[t].get('ball_owned_team', -1) == 0:
            deltas.append((ball_x_next - ball_x) * 60.0)

    if deltas:
        ax.hist(deltas, bins=50, color=COLORS['secondary'], edgecolor='white', alpha=0.8)
        mean_d = np.mean(deltas)
        ax.axvline(mean_d, color=COLORS['accent'], linestyle='--', linewidth=1.5,
                    label=f'Mean: {mean_d:.2f}')
        ax.legend()
    ax.set_xlabel('Ball x-displacement (pitch units)')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f'Saved: {output_path}')
