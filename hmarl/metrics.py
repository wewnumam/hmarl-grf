"""Evaluation metrics for HMARL.

Combines all metrics from the thesis:
- Performance: Win Rate, Goal Difference, Cumulative Reward
- Coordination: PSR, PPR, Positional Entropy, Team Compactness, FAI, RCI
"""

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from hmarl.env import get_action_category, ACTION_CATEGORIES


# ---------------------------------------------------------------------------
# Performance Metrics
# ---------------------------------------------------------------------------
def compute_win_rate(results: List[str]) -> float:
    """Compute Win Rate.

    WR = N_win / N_match × 100%

    Args:
        results: list of 'win', 'loss', 'draw' strings
    """
    if not results:
        return 0.0
    wins = sum(1 for r in results if r == 'win')
    return (wins / len(results)) * 100.0


def compute_goal_difference(
    goals_for: List[int],
    goals_against: List[int],
) -> int:
    """Compute cumulative Goal Difference.

    GD = Σ (G_for,m - G_against,m)
    """
    return sum(gf - ga for gf, ga in zip(goals_for, goals_against))


def compute_cumulative_reward(rewards: List[float]) -> float:
    """Compute cumulative reward (sum of all rewards in an episode)."""
    return sum(rewards)


# ---------------------------------------------------------------------------
# Coordination Metrics
# ---------------------------------------------------------------------------
def pass_success_ratio(
    actual_actions: List[List[int]],
    game_states: List[Dict],
    team_id: int = 0,
    success_window: int = 5,
) -> Tuple[int, int, float]:
    """Compute Pass Success Ratio.

    PSR = P_success / P_total × 100%

    Args:
        actual_actions: list of (num_agents,) action lists per timestep
        game_states: list of game state dicts per timestep
        team_id: 0 for left, 1 for right
        success_window: max timesteps to check for pass completion
    """
    PASS_ACTIONS = {9, 10, 11}  # long_pass, high_pass, short_pass
    total_attempts = 0
    successful = 0

    for idx in range(len(game_states) - 1):
        gs = game_states[idx]
        owned_team = gs.get('ball_owned_team', -1)
        owned_player = gs.get('ball_owned_player', -1)

        if owned_team != team_id or owned_player is None or owned_player < 0:
            continue

        actions = actual_actions[idx] if idx < len(actual_actions) else []
        if owned_player >= len(actions):
            continue
        if actions[owned_player] not in PASS_ACTIONS:
            continue

        total_attempts += 1

        # Check future frames for pass completion
        for future_idx in range(idx + 1, min(idx + 1 + success_window, len(game_states))):
            future_gs = game_states[future_idx]
            future_team = future_gs.get('ball_owned_team', -1)
            future_player = future_gs.get('ball_owned_player', -1)

            if (future_team == team_id and future_player is not None
                    and future_player >= 0 and future_player != owned_player):
                successful += 1
                break

            if future_team != team_id:
                break  # Lost possession

    ratio = (successful / total_attempts * 100.0) if total_attempts > 0 else 0.0
    return total_attempts, successful, ratio


def progressive_pass_ratio(
    actual_actions: List[List[int]],
    game_states: List[Dict],
    team_id: int = 0,
) -> Tuple[int, int, float]:
    """Compute Progressive Pass Ratio.

    PPR = P_progressive / P_success × 100%

    A pass is progressive if Δx > 0 (ball moves toward opponent goal).
    """
    PASS_ACTIONS = {9, 10, 11}

    successful = 0
    progressive = 0

    for idx in range(len(game_states) - 1):
        gs = game_states[idx]
        owned_team = gs.get('ball_owned_team', -1)
        owned_player = gs.get('ball_owned_player', -1)

        if owned_team != team_id or owned_player is None or owned_player < 0:
            continue

        actions = actual_actions[idx] if idx < len(actual_actions) else []
        if owned_player >= len(actions):
            continue
        if actions[owned_player] not in PASS_ACTIONS:
            continue

        # Get passer position
        passer_pos = gs.get('left_team', [None] * 11)[owned_player]
        if passer_pos is None:
            continue

        # Check for successful pass to another player
        for future_idx in range(idx + 1, min(idx + 6, len(game_states))):
            future_gs = game_states[future_idx]
            future_team = future_gs.get('ball_owned_team', -1)
            future_player = future_gs.get('ball_owned_player', -1)

            if (future_team == team_id and future_player is not None
                    and future_player >= 0 and future_player != owned_player):
                successful += 1
                receiver_pos = future_gs.get('left_team', [None] * 11)[future_player]
                if receiver_pos is not None and receiver_pos[0] > passer_pos[0]:
                    progressive += 1
                break

            if future_team != team_id:
                break

    ratio = (progressive / successful * 100.0) if successful > 0 else 0.0
    return successful, progressive, ratio


def positional_entropy(
    game_states: List[Dict],
    team_side: str = 'left',
    grid_shape: Tuple[int, int] = (6, 4),
    pitch_length: float = 120.0,
    pitch_width: float = 80.0,
) -> Tuple[np.ndarray, float]:
    """Compute Positional Entropy.

    H_i = -Σ p_{i,j} log2(p_{i,j})
    H_team = (1/N) Σ H_i
    """
    nx, ny = grid_shape
    num_zones = nx * ny

    if not game_states:
        return np.zeros(11), 0.0

    # Collect all player positions
    key = f'{team_side}_team'
    positions_list = []
    for gs in game_states:
        team_positions = gs.get(key, [])
        if len(team_positions) >= 11:
            positions_list.append(team_positions[:11])

    if not positions_list:
        return np.zeros(11), 0.0

    positions = np.array(positions_list)  # (T, 11, 2)

    # Scale to pitch coordinates
    positions_scaled = np.stack([
        _scale_to_pitch(positions[t]) for t in range(len(positions))
    ])

    # Build zone histograms
    counts = np.zeros((11, num_zones), dtype=float)
    x_edges = np.linspace(0.0, pitch_length, nx + 1)
    y_edges = np.linspace(0.0, pitch_width, ny + 1)

    for t in range(len(positions_scaled)):
        for p in range(11):
            x, y = positions_scaled[t, p]
            xi = int(np.clip(np.searchsorted(x_edges, x, side='right') - 1, 0, nx - 1))
            yi = int(np.clip(np.searchsorted(y_edges, y, side='right') - 1, 0, ny - 1))
            zone = yi * nx + xi
            counts[p, zone] += 1

    # Compute entropy per player
    entropies = np.zeros(11)
    for p in range(11):
        total = counts[p].sum()
        if total <= 0:
            continue
        probs = counts[p] / total
        mask = probs > 0
        entropies[p] = -np.sum(probs[mask] * np.log2(probs[mask]))

    team_entropy = float(np.mean(entropies))
    return entropies, team_entropy


def _scale_to_pitch(coords):
    """Scale GRF coords to pitch coordinates."""
    if isinstance(coords, list):
        coords = np.array(coords)
    x = (coords[:, 0] + 1.0) * 60.0
    y = (coords[:, 1] + 0.42) * (80.0 / 0.84)
    return np.column_stack((x, y))


def team_compactness(
    game_states: List[Dict],
    team_side: str = 'left',
) -> Tuple[float, float, float, float]:
    """Compute Team Compactness.

    ρ_tc(t) = sqrt((1/N) Σ (x_i - x̄)² + (y_i - ȳ)²)
    """
    key = f'{team_side}_team'
    densities = []

    for gs in game_states:
        positions = gs.get(key, [])
        if len(positions) < 11:
            continue

        pos_arr = np.array(positions[:11])
        # Scale to pitch coords
        scaled = np.column_stack([
            (pos_arr[:, 0] + 1.0) * 60.0,
            (pos_arr[:, 1] + 0.42) * (80.0 / 0.84),
        ])

        centroid = scaled.mean(axis=0)
        deviations = scaled - centroid
        squared_distances = np.sum(deviations ** 2, axis=1)
        rho = math.sqrt(np.mean(squared_distances))
        densities.append(rho)

    if not densities:
        return 0.0, 0.0, 0.0, 0.0

    densities = np.array(densities)
    return float(densities.mean()), float(densities.std()), float(densities.min()), float(densities.max())


def formation_adherence_index(
    game_states: List[Dict],
    team_side: str = 'left',
) -> Tuple[float, float]:
    """Compute FAI over multiple timesteps.

    Returns (mean_fai, std_fai)
    """
    D_MAX = 2.24
    fai_values = []

    for gs in game_states:
        ball = gs.get('ball', [0, 0, 0])
        ball_x, ball_y = ball[0], ball[1]

        template = [
            [-0.95, 0.0], [-0.5, -0.15], [-0.5, 0.15],
            [-0.3, -0.35], [-0.3, 0.35],
            [-0.1, 0.0], [0.1, -0.2], [0.1, 0.2],
            [0.3, -0.35], [0.3, 0.35], [0.5, 0.0],
        ]

        shift_x = ball_x * 0.3
        shift_y = ball_y * 0.15

        positions = gs.get(f'{team_side}_team', [])
        if len(positions) < 11:
            continue

        total_dev = 0.0
        for i in range(11):
            tx = np.clip(template[i][0] + shift_x, -1.0, 1.0)
            ty = np.clip(template[i][1] + shift_y, -0.42, 0.42)
            actual = positions[i]
            d = np.sqrt((actual[0] - tx)**2 + (actual[1] - ty)**2)
            total_dev += d

        rho_fa = 1.0 - (total_dev / (11 * D_MAX))
        fai_values.append(float(np.clip(rho_fa, 0.0, 1.0)))

    if not fai_values:
        return 0.0, 0.0

    return float(np.mean(fai_values)), float(np.std(fai_values))


# ---------------------------------------------------------------------------
# Extended Coordination Metrics (Pass Network, IAD, Convex Hull, Transitions, BPR)
# ---------------------------------------------------------------------------
PASS_ACTION_IDS = {9, 10, 11}


def pass_network_matrix(
    game_states: List[Dict],
    actual_actions: List[List[int]],
    team_id: int = 0,
) -> np.ndarray:
    """Build 11×11 pass adjacency matrix: entry (i,j) = pass count i→j.

    Returns:
        np.ndarray shape (11, 11) with pass counts.
    """
    counts = np.zeros((11, 11), dtype=int)
    for t in range(len(game_states) - 1):
        gs = game_states[t]
        if gs.get('ball_owned_team', -1) != team_id:
            continue
        passer = gs.get('ball_owned_player', -1)
        if passer is None or passer < 0 or passer >= 11:
            continue
        if t >= len(actual_actions) or passer >= len(actual_actions[t]):
            continue
        if actual_actions[t][passer] not in PASS_ACTION_IDS:
            continue
        # Check next frame for successful pass
        gs_next = game_states[t + 1]
        if gs_next.get('ball_owned_team', -1) != team_id:
            continue
        receiver = gs_next.get('ball_owned_player', -1)
        if receiver is None or receiver < 0 or receiver >= 11:
            continue
        if receiver == passer:
            continue
        counts[passer, receiver] += 1
    return counts


def inter_agent_distance_per_line(
    game_states: List[Dict],
    team_side: str = 'left',
) -> Dict[str, float]:
    """Mean distance between tactical lines over the episode.

    Lines: defense (GK+CB+LB+RB), midfield (DM+CM), attack (LM+RM+AM+CF).
    Returns dict with:
        defence_midfield_gap  – mean gap between defense line and midfield line
        midfield_attack_gap   – mean gap between midfield line and attack line
        overall_spread        – mean max distance between any two outfield players
    """
    key = f'{team_side}_team'
    key_roles = f'{team_side}_team_roles'

    DEFENSE = {0, 1, 2, 3}   # GK, CB, LB, RB
    MIDFIELD = {4, 5}         # DM, CM
    ATTACK = {6, 7, 8, 9}     # LM, RM, AM, CF

    def_mid_gaps = []
    mid_att_gaps = []
    spreads = []

    for gs in game_states:
        positions = gs.get(key, [])
        if len(positions) < 11:
            continue
        roles_raw = gs.get(key_roles, list(range(11)))
        roles = roles_raw[:11]
        pos_arr = np.array(positions[:11])
        # Scale to pitch coordinates
        x_scaled = (pos_arr[:, 0] + 1.0) * 60.0

        def_idx = [i for i, r in enumerate(roles) if r in DEFENSE]
        mid_idx = [i for i, r in enumerate(roles) if r in MIDFIELD]
        att_idx = [i for i, r in enumerate(roles) if r in ATTACK]

        def_x = np.mean(x_scaled[def_idx]) if def_idx else None
        mid_x = np.mean(x_scaled[mid_idx]) if mid_idx else None
        att_x = np.mean(x_scaled[att_idx]) if att_idx else None

        if def_x is not None and mid_x is not None:
            def_mid_gaps.append(abs(mid_x - def_x))
        if mid_x is not None and att_x is not None:
            mid_att_gaps.append(abs(att_x - mid_x))

        # Overall spread (max pairwise distance, outfield only)
        outfield_idx = [i for i in range(11) if roles[i] != 0]
        if len(outfield_idx) >= 2:
            outfield = x_scaled[outfield_idx]
            spreads.append(float(np.max(outfield) - np.min(outfield)))

    return {
        'defence_midfield_gap': float(np.mean(def_mid_gaps)) if def_mid_gaps else 0.0,
        'midfield_attack_gap': float(np.mean(mid_att_gaps)) if mid_att_gaps else 0.0,
        'overall_spread': float(np.mean(spreads)) if spreads else 0.0,
    }


def _convex_hull_area_numpy(points: np.ndarray) -> float:
    """Compute convex hull area using Graham scan (numpy-only, no scipy).

    Args:
        points: (N, 2) array of 2D coordinates.
    Returns:
        Area of the convex hull.
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 3:
        return 0.0

    # Sort by x then y
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    # Cross product for turn direction
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenate lower and upper, removing endpoints of each (duplicated)
    hull = lower[:-1] + upper[:-1]
    hull = np.array(hull)

    if len(hull) < 3:
        return 0.0

    # Shoelace formula
    x, y = hull[:, 0], hull[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def convex_hull_area(
    game_states: List[Dict],
    team_side: str = 'left',
) -> Tuple[float, float, float, float]:
    """Convex hull area of outfield players (excluding GK).

    Returns (mean, std, min, max).
    """
    key = f'{team_side}_team'
    areas = []
    for gs in game_states:
        positions = gs.get(key, [])
        if len(positions) < 11:
            continue
        # Skip GK (index 0), take outfield players
        outfield_pos = np.array(positions[1:11])
        x_scaled = (outfield_pos[:, 0] + 1.0) * 60.0
        y_scaled = (outfield_pos[:, 1] + 0.42) * (80.0 / 0.84)
        points = np.column_stack([x_scaled, y_scaled])
        area = _convex_hull_area_numpy(points)
        if area > 0:
            areas.append(area)
    if not areas:
        return 0.0, 0.0, 0.0, 0.0
    areas = np.array(areas)
    return float(areas.mean()), float(areas.std()), float(areas.min()), float(areas.max())


def action_transition_matrix(
    actual_actions: List[List[int]],
    num_actions: int = 19,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute action transition probability matrix.

    For each timestep where the ball owner is active, record the transition
    from the ball owner's action at t to the ball owner's action at t+1.

    Returns:
        (counts_matrix, prob_matrix) – raw counts and row-normalized probabilities.
    """
    counts = np.zeros((num_actions, num_actions), dtype=int)
    for t in range(len(actual_actions) - 1):
        for agent_idx in range(min(len(actual_actions[t]), len(actual_actions[t + 1]))):
            a_t = actual_actions[t][agent_idx]
            a_t1 = actual_actions[t + 1][agent_idx]
            if 0 <= a_t < num_actions and 0 <= a_t1 < num_actions:
                counts[a_t, a_t1] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    probs = counts / row_sums
    return counts, probs


def ball_progression_rate(
    game_states: List[Dict],
    team_id: int = 0,
) -> Tuple[float, float]:
    """Mean and std of ball x-displacement per timestep (pitch coordinates).

    Positive value = ball moving toward opponent goal.
    Returns (mean, std).
    """
    deltas = []
    for t in range(len(game_states) - 1):
        gs = game_states[t]
        gs_next = game_states[t + 1]
        ball_x = gs.get('ball', [0, 0, 0])[0]
        ball_x_next = gs_next.get('ball', [0, 0, 0])[0]
        # Only count when this team owns the ball
        if gs.get('ball_owned_team', -1) == team_id:
            dx = (ball_x_next - ball_x) * 60.0  # scale to pitch coordinates
            deltas.append(dx)
    if not deltas:
        return 0.0, 0.0
    arr = np.array(deltas)
    return float(arr.mean()), float(arr.std())


def compute_all_metrics(
    actual_actions: List[List[int]],
    game_states: List[Dict],
    ideal_actions: List[List[int]],
    match_result: str = 'draw',
    goals_for: int = 0,
    goals_against: int = 0,
    cumulative_reward: float = 0.0,
) -> Dict[str, Any]:
    """Compute all evaluation metrics.

    Returns a comprehensive metrics dictionary.
    """
    from hmarl.rci import compute_rci

    # Performance
    wr = 100.0 if match_result == 'win' else 0.0
    gd = goals_for - goals_against

    # Coordination
    psr_attempts, psr_success, psr = pass_success_ratio(actual_actions, game_states)
    ppr_success, ppr_prog, ppr = progressive_pass_ratio(actual_actions, game_states)
    _, h_team = positional_entropy(game_states)
    tc_mean, tc_std, tc_min, tc_max = team_compactness(game_states)
    fai_mean, fai_std = formation_adherence_index(game_states)

    # Extended coordination
    iad = inter_agent_distance_per_line(game_states)
    ch_mean, ch_std, ch_min, ch_max = convex_hull_area(game_states)
    _, bpr_std = ball_progression_rate(game_states)

    # RCI
    rci_results = compute_rci(actual_actions, ideal_actions)

    return {
        # Performance
        'win_rate': wr,
        'goal_difference': gd,
        'goals_for': goals_for,
        'goals_against': goals_against,
        'cumulative_reward': cumulative_reward,
        # Coordination
        'psr': psr,
        'psr_attempts': psr_attempts,
        'psr_success': psr_success,
        'ppr': ppr,
        'ppr_success': ppr_success,
        'ppr_progressive': ppr_prog,
        'positional_entropy': h_team,
        'compactness_mean': tc_mean,
        'compactness_std': tc_std,
        'compactness_min': tc_min,
        'compactness_max': tc_max,
        'fai_mean': fai_mean,
        'fai_std': fai_std,
        # Extended coordination
        'defence_midfield_gap': iad['defence_midfield_gap'],
        'midfield_attack_gap': iad['midfield_attack_gap'],
        'overall_spread': iad['overall_spread'],
        'convex_hull_mean': ch_mean,
        'convex_hull_std': ch_std,
        'convex_hull_min': ch_min,
        'convex_hull_max': ch_max,
        'bpr_std': bpr_std,
        # RCI
        'rci_strict': rci_results['rci_strict'],
        'rci_cat': rci_results['rci_cat'],
        'rci_strict_per_agent': rci_results['rci_strict_per_agent'],
        'rci_cat_per_agent': rci_results['rci_cat_per_agent'],
    }


def print_metrics(metrics: Dict[str, Any], model_name: str = "Model"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"  Evaluation Results: {model_name}")
    print(f"{'='*60}")

    print(f"\n  --- Performance Metrics ---")
    print(f"  Win Rate:              {metrics['win_rate']:.1f}%")
    print(f"  Goal Difference:       {metrics['goal_difference']}")
    print(f"  Goals For / Against:   {metrics['goals_for']} / {metrics['goals_against']}")
    print(f"  Cumulative Reward:     {metrics['cumulative_reward']:.2f}")

    print(f"\n  --- Coordination Metrics ---")
    print(f"  PSR:                   {metrics['psr']:.2f}% ({metrics['psr_success']}/{metrics['psr_attempts']})")
    print(f"  PPR:                   {metrics['ppr']:.2f}% ({metrics['ppr_progressive']}/{metrics['ppr_success']})")
    print(f"  Positional Entropy:    {metrics['positional_entropy']:.4f} bits")
    print(f"  Team Compactness:      {metrics['compactness_mean']:.4f} ± {metrics['compactness_std']:.4f}")
    print(f"  FAI:                   {metrics['fai_mean']:.4f} ± {metrics['fai_std']:.4f}")

    print(f"\n  --- Extended Coordination ---")
    print(f"  Def-Mid Gap:           {metrics.get('defence_midfield_gap', 0):.4f}")
    print(f"  Mid-Att Gap:           {metrics.get('midfield_attack_gap', 0):.4f}")
    print(f"  Overall Spread:        {metrics.get('overall_spread', 0):.4f}")
    print(f"  Convex Hull:           {metrics.get('convex_hull_mean', 0):.4f} ± {metrics.get('convex_hull_std', 0):.4f}")
    print(f"  BPR Std:               {metrics.get('bpr_std', 0):.4f}")

    print(f"\n  --- Role Coherence Index ---")
    print(f"  RCI_strict:            {metrics['rci_strict']:.4f}")
    print(f"  RCI_cat:               {metrics['rci_cat']:.4f}")

    if metrics.get('rci_strict_per_agent'):
        print(f"\n  Per-agent RCI_strict:")
        for i, rci in enumerate(metrics['rci_strict_per_agent']):
            print(f"    Agent {i:2d}: {rci:.4f}")

    print(f"{'='*60}\n")
