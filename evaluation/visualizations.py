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
# Academic color palette (grayscale-friendly, colorblind-safe)
COLORS = {
    'hmarl': '#2c3e50',       # Dark blue-gray
    'ippo': '#7f8c8d',        # Medium gray
    'shppo': '#95a5a6',       # Light gray
    'random': '#bdc3c7',      # Pale gray
    'accent': '#e74c3c',      # Red accent
    'secondary': '#3498db',   # Blue accent
    'tertiary': '#2ecc71',    # Green accent
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
    """Apply clean academic plotting style."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
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
    })


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
              e.g. {'HMARL': [...], 'IPPO': [...], 'SHPPO': [...]}
        output_path: path to save the figure
        window: rolling average window size
        title: plot title
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    color_keys = list(COLORS.keys())
    pattern_keys = list(PATTERNS.keys())

    for idx, (model, rewards) in enumerate(data.items()):
        key = color_keys[idx % len(color_keys)]
        color = COLORS[key]
        episodes = np.arange(1, len(rewards) + 1)

        if len(rewards) >= window:
            smoothed = _rolling_mean(rewards, window)
            std = _rolling_std(rewards, window)
            x = episodes[window - 1:]
            ax.plot(x, smoothed, color=color, label=model, linewidth=1.5)
            ax.fill_between(x, smoothed - std, smoothed + std,
                           alpha=0.15, color=color)
        else:
            ax.plot(episodes, rewards, color=color, label=model, linewidth=1.5)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title(title)
    ax.legend(frameon=True, framealpha=0.9, edgecolor='gray')
    fig.tight_layout()
    fig.savefig(output_path)
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

    Args:
        data: dict mapping model_name -> {'rci_cat': [...], 'rci_strict': [...]}
        output_path: path to save
        window: rolling average window
        title: plot title
    """
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    color_keys = list(COLORS.keys())

    for idx, (model, rci_data) in enumerate(data.items()):
        color = COLORS[color_keys[idx % len(color_keys)]]

        for ax, variant, label in [
            (ax1, 'rci_cat', 'RCI$_{cat}$'),
            (ax2, 'rci_strict', 'RCI$_{strict}$'),
        ]:
            values = rci_data.get(variant, [])
            if not values:
                continue
            episodes = np.arange(1, len(values) + 1)
            if len(values) >= window:
                smoothed = _rolling_mean(values, window)
                x = episodes[window - 1:]
                ax.plot(x, smoothed, color=color, label=model, linewidth=1.5)
            else:
                ax.plot(episodes, values, color=color, label=model, linewidth=1.5)

        ax1.set_ylabel('RCI Value')
        ax1.set_xlabel('Episode')
        ax2.set_xlabel('Episode')
        ax1.set_title('Category-based ($RCI_{cat}$)')
        ax2.set_title('Strict ($RCI_{strict}$)')
        ax1.set_ylim(-0.05, 1.05)

    for ax in (ax1, ax2):
        ax.legend(frameon=True, framealpha=0.9, edgecolor='gray')
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
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
    """Grouped bar chart for all metrics across models.

    Args:
        data: dict mapping model_name -> {metric_name: value}
              e.g. {'HMARL': {'wr': 80, 'rci_cat': 0.75, ...}, 'IPPO': {...}}
        output_path: path to save
        title: plot title
    """
    _apply_style()

    # Define metric groups
    perf_metrics = ['wr', 'gd', 'cumulative_reward']
    coord_metrics = ['psr', 'ppr', 'positional_entropy', 'compactness_mean', 'fai_mean']
    rci_metrics = ['rci_strict', 'rci_cat']

    # Normalize display names
    display_names = {
        'wr': 'WR (%)',
        'gd': 'GD',
        'cumulative_reward': 'Cum. Reward',
        'psr': 'PSR (%)',
        'ppr': 'PPR (%)',
        'positional_entropy': 'Entropy (H)',
        'compactness_mean': 'Compactness',
        'fai_mean': 'FAI',
        'rci_strict': 'RCI$_{strict}$',
        'rci_cat': 'RCI$_{cat}$',
    }

    all_metrics = perf_metrics + coord_metrics + rci_metrics
    models = list(data.keys())
    n_models = len(models)
    n_metrics = len(all_metrics)

    fig, ax = plt.subplots(figsize=(14, 5))

    x = np.arange(n_metrics)
    width = 0.8 / n_models

    color_keys = list(COLORS.keys())
    for idx, model in enumerate(models):
        values = []
        for m in all_metrics:
            values.append(data[model].get(m, 0))
        offset = (idx - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model,
                      color=COLORS[color_keys[idx % len(color_keys)]],
                      edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([display_names.get(m, m) for m in all_metrics],
                       rotation=30, ha='right')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend(frameon=True, framealpha=0.9, edgecolor='gray',
              ncol=min(n_models, 3))

    # Add vertical separators between metric groups
    for sep in [len(perf_metrics) - 0.5, len(perf_metrics) + len(coord_metrics) - 0.5]:
        ax.axvline(x=sep, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)

    fig.tight_layout()
    fig.savefig(output_path)
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
        im = ax.pcolormesh(xedges, yedges, h.T, cmap='YlOrRd', alpha=0.8)

        ax.set_title(f'{role_name} (n={len(px)})', fontsize=10, fontweight='bold')

    # Remove extra axes
    for j in range(11, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path)
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

        ax.scatter(x[i], y[i], c=color, s=180, zorder=3,
                  edgecolors='white', linewidth=1.5)
        ax.annotate(role_name, (x[i], y[i]), fontsize=8, fontweight='bold',
                   color='white', ha='center', va='center', zorder=4)

    # Plot target formation
    if show_target:
        ax.scatter(targets[:, 0], targets[:, 1], c='none',
                  s=200, edgecolors='gray', linewidth=1, linestyle='--',
                  zorder=2, alpha=0.5, label='Target Formation')

    ax.set_xlim(-2, pitch_length + 2)
    ax.set_ylim(-2, pitch_width + 2)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{title} (t={timestep})', fontsize=12)
    ax.legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path)
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
    cat_colors = ['#2c3e50', '#e74c3c', '#3498db', '#f39c12', '#95a5a6']

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
    ax.set_ylim(0, 105)

    fig.tight_layout()
    fig.savefig(output_path)
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
    strategy_colors = ['#2c3e50', '#3498db', '#2ecc71']
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
    fig.savefig(output_path)
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

    color_keys = list(COLORS.keys())
    for idx, (model, values) in enumerate(data.items()):
        color = COLORS[color_keys[idx % len(color_keys)]]

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
    ax.set_ylabel('Team Compactness (rho_tc)')
    ax.set_title(title)
    ax.legend(frameon=True, framealpha=0.9, edgecolor='gray')

    fig.tight_layout()
    fig.savefig(output_path)
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
    """Bar chart: RCI value with each hierarchy level disabled.

    Args:
        data: dict mapping ablation_config -> RCI_cat value
              e.g. {'Full HMARL': 0.75, 'No High-Level': 0.60, ...}
        output_path: path to save
        title: plot title
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    configs = list(data.keys())
    values = list(data.values())
    n = len(configs)

    # Color gradient
    colors = plt.cm.Blues(np.linspace(0.9, 0.3, n))

    bars = ax.barh(range(n), values, color=colors, edgecolor='white',
                   linewidth=0.5, height=0.6)

    ax.set_yticks(range(n))
    ax.set_yticklabels(configs)
    ax.set_xlabel('RCI$_{cat}$')
    ax.set_title(title)
    ax.set_xlim(0, max(values) * 1.15 if values else 1.0)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', va='center', fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path)
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
    colors = ['#2c3e50', '#3498db', '#2ecc71', '#e74c3c', '#f39c12']

    # --- Detect single vs multi episode ---
    # Single: sub_goals_per_agent[0] is a list of ints (one timestep's sub-goals)
    # Multi:  sub_goals_per_agent[0] is a list of lists (one episode's timesteps)
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
    fig.savefig(output_path)
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
    colors = ['#2c3e50', '#3498db', '#2ecc71', '#e74c3c']

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
    ax.set_ylabel('Reward Contribution')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8, frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path)
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
        'rci_cat': 'RCI$_{cat}$',
        'rci_strict': 'RCI$_{strict}$',
        'fai_mean': 'FAI',
        'positional_entropy': 'Entropy (H)',
        'compactness_mean': 'Compactness',
        'psr': 'PSR',
        'ppr': 'PPR',
        'wr': 'WR',
        'cumulative_reward': 'Cum. Reward',
    }
    labels = [display.get(m, m) for m in metrics]

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate values
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   fontsize=8, color=color)

    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, label='Pearson r')

    fig.tight_layout()
    fig.savefig(output_path)
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
        # Aggregate per-episode data from all models
        all_metric_data = {}
        for model, metrics_dict in model_comparisons.items():
            for k, v in metrics_dict.items():
                if isinstance(v, (int, float)):
                    if k not in all_metric_data:
                        all_metric_data[k] = []
                    all_metric_data[k].append(v)

        if len(all_metric_data) >= 3:
            plot_metric_correlation(
                all_metric_data,
                os.path.join(output_dir, '12_correlation_heatmap.png'),
            )

    print(f'\nAll plots saved to: {output_dir}')
