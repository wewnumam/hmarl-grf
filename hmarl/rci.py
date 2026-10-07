"""Role Coherence Index (RCI) evaluation metric.

RCI = (1/NT) · Σ_i Σ_t f_match(a_{i,t}, a*)

Two variants (per-timestep):
- RCI_strict: f_strict = I(a == a*)  (exact match)
- RCI_cat: f_cat = I(C(a) == C(a*))  (category match)

One temporal variant:
- RCI_streak: streak-aware, rewards consecutive matching timesteps

Action categories:
  passing:   {short_pass, long_pass, high_pass}
  shooting:  {shot}
  movement:  {8 directions, sprint}
  ball_control: {dribble}
  defensive: {idle, release_direction, release_sprint, sliding, release_dribble}

Circularity note (thesis §5.3):
  RCI used as evaluation metric must use a DIFFERENT expert policy
  than the one used for training reward shaping. Use ExpertPolicyEval
  for evaluation, ExpertPolicyAllAgents for training reward.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from hmarl.env import get_action_category


def f_strict(actual: int, ideal: int) -> float:
    """Strict matching: 1 if exact match, 0 otherwise."""
    return 1.0 if actual == ideal else 0.0


def f_category(actual: int, ideal: int) -> float:
    """Category-based matching: 1 if same tactical category, 0 otherwise."""
    cat_actual = get_action_category(actual)
    cat_ideal = get_action_category(ideal)
    return 1.0 if cat_actual == cat_ideal else 0.0


def f_category_nomove(actual: int, ideal: int) -> float:
    """Category matching EXCLUDING movement confound.

    Movement dominates both sides (8 of 19 actions) — a movement-vs-movement
    "match" carries no role information. Only non-movement categories
    (passing, shooting, ball_control, defensive) earn credit.

    Motivation (confound test, 2026-10-06): rci_cat correlated ~+0.9 with
    positional_entropy and compactness, both expected NEGATIVE. Hypothesis:
    that correlation is an artefact of movement volume, not role coherence.
    """
    cat_actual = get_action_category(actual)
    cat_ideal = get_action_category(ideal)
    if cat_actual == 'movement':
        return 0.0
    return 1.0 if cat_actual == cat_ideal else 0.0


# ---------------------------------------------------------------------------
# Per-timestep RCI (original)
# ---------------------------------------------------------------------------

def compute_rci_strict(
    actual_actions: List[List[int]],
    ideal_actions: List[List[int]],
) -> Tuple[float, List[float]]:
    """Compute RCI_strict over an episode.

    Args:
        actual_actions: list of (num_agents,) action lists per timestep
        ideal_actions: list of (num_agents,) action lists per timestep

    Returns:
        (rci_overall, rci_per_agent)
    """
    if not actual_actions or not ideal_actions:
        return 0.0, []

    num_agents = len(actual_actions[0])
    T = len(actual_actions)

    per_agent_scores = np.zeros(num_agents)

    for t in range(min(T, len(ideal_actions))):
        for i in range(num_agents):
            per_agent_scores[i] += f_strict(
                actual_actions[t][i],
                ideal_actions[t][i],
            )

    rci_per_agent = (per_agent_scores / T).tolist()
    rci_overall = float(np.mean(rci_per_agent))

    return rci_overall, rci_per_agent


def compute_rci_category(
    actual_actions: List[List[int]],
    ideal_actions: List[List[int]],
) -> Tuple[float, List[float]]:
    """Compute RCI_cat over an episode.

    Args:
        actual_actions: list of (num_agents,) action lists per timestep
        ideal_actions: list of (num_agents,) action lists per timestep

    Returns:
        (rci_overall, rci_per_agent)
    """
    return _compute_rci_generic(actual_actions, ideal_actions, f_category)


def compute_rci_nomove(
    actual_actions: List[List[int]],
    ideal_actions: List[List[int]],
) -> Tuple[float, List[float]]:
    """Compute RCI_nomove over an episode (movement matches earn zero).

    Diagnostic variant to test the movement-volume confound.
    Lower absolute value than rci_cat is expected — the movement term
    previously contributed a large inflated baseline.
    """
    return _compute_rci_generic(actual_actions, ideal_actions, f_category_nomove)


def _compute_rci_generic(
    actual_actions: List[List[int]],
    ideal_actions: List[List[int]],
    match_fn,
) -> Tuple[float, List[float]]:
    """Shared RCI computation with pluggable per-timestep match function."""
    if not actual_actions or not ideal_actions:
        return 0.0, []

    num_agents = len(actual_actions[0])
    T = len(actual_actions)

    per_agent_scores = np.zeros(num_agents)
    for t in range(min(T, len(ideal_actions))):
        for i in range(num_agents):
            per_agent_scores[i] += match_fn(
                actual_actions[t][i],
                ideal_actions[t][i],
            )

    rci_per_agent = (per_agent_scores / T).tolist()
    rci_overall = float(np.mean(rci_per_agent))

    return rci_overall, rci_per_agent


# ---------------------------------------------------------------------------
# Temporal RCI — streak-aware variant
# ---------------------------------------------------------------------------

def compute_rci_streak(
    actual_actions: List[List[int]],
    ideal_actions: List[List[int]],
    min_streak: int = 10,
) -> Dict[str, any]:
    """Compute temporal RCI based on consecutive matching streaks.

    Addresses thesis concern: per-timestep RCI ignores temporal coherence.
    A defender maintaining correct position for 50 consecutive timesteps
    is more coherent than one who matches randomly with same overall rate.

    Metrics:
        - rci_streak: fraction of timesteps in valid streaks (>= min_streak)
        - mean_streak_length: average length of valid streaks per agent
        - role_switch_rate: number of streak breaks per timestep (lower = more coherent)

    Args:
        actual_actions: list of (num_agents,) action lists per timestep
        ideal_actions: list of (num_agents,) action lists per timestep
        min_streak: minimum consecutive matching timesteps to count as valid streak

    Returns:
        dict with rci_streak, mean_streak_length, role_switch_rate, per_agent details
    """
    if not actual_actions or not ideal_actions:
        return {'rci_streak': 0.0, 'mean_streak_length': 0.0, 'role_switch_rate': 1.0}

    num_agents = len(actual_actions[0])
    T = min(len(actual_actions), len(ideal_actions))

    per_agent_streaks = []  # list of lists of streak lengths
    per_agent_valid_timesteps = []

    for i in range(num_agents):
        streaks = []
        current_streak = 0
        valid_timesteps = 0

        for t in range(T):
            match = f_category(actual_actions[t][i], ideal_actions[t][i])
            if match > 0:
                current_streak += 1
            else:
                if current_streak >= min_streak:
                    streaks.append(current_streak)
                    valid_timesteps += current_streak
                current_streak = 0

        # Handle final streak
        if current_streak >= min_streak:
            streaks.append(current_streak)
            valid_timesteps += current_streak

        per_agent_streaks.append(streaks)
        per_agent_valid_timesteps.append(valid_timesteps)

    # Aggregate
    all_streaks = [s for agent_streaks in per_agent_streaks for s in agent_streaks]
    total_valid = sum(per_agent_valid_timesteps)
    total_possible = num_agents * T

    rci_streak = total_valid / total_possible if total_possible > 0 else 0.0
    mean_streak = float(np.mean(all_streaks)) if all_streaks else 0.0

    # Role switch rate: how often does any agent break a streak?
    total_breaks = sum(max(0, len(s) - 0) for s in per_agent_streaks)  # streak count ≈ break count
    # More precise: breaks = episodes where match=0 after match=1
    total_breaks = 0
    for i in range(num_agents):
        for t in range(1, T):
            prev_match = f_category(actual_actions[t-1][i], ideal_actions[t-1][i])
            curr_match = f_category(actual_actions[t][i], ideal_actions[t][i])
            if prev_match > 0 and curr_match == 0:
                total_breaks += 1

    role_switch_rate = total_breaks / total_possible if total_possible > 0 else 1.0

    return {
        'rci_streak': round(rci_streak, 4),
        'mean_streak_length': round(mean_streak, 2),
        'role_switch_rate': round(role_switch_rate, 4),
        'n_valid_streaks': len(all_streaks),
        'per_agent_streaks': per_agent_streaks,
    }


# ---------------------------------------------------------------------------
# Combined RCI computation
# ---------------------------------------------------------------------------

def compute_rci(
    actual_actions: List[List[int]],
    ideal_actions: List[List[int]],
    include_temporal: bool = True,
    min_streak: int = 10,
) -> Dict[str, any]:
    """Compute all RCI variants.

    Returns dict with:
        - rci_strict: overall strict RCI
        - rci_cat: overall category RCI
        - rci_strict_per_agent: per-agent strict RCI
        - rci_cat_per_agent: per-agent category RCI
        - (if include_temporal) rci_streak: temporal streak-based RCI
    """
    strict_overall, strict_per_agent = compute_rci_strict(actual_actions, ideal_actions)
    cat_overall, cat_per_agent = compute_rci_category(actual_actions, ideal_actions)
    nomove_overall, _ = compute_rci_nomove(actual_actions, ideal_actions)

    result = {
        'rci_strict': strict_overall,
        'rci_cat': cat_overall,
        'rci_nomove': nomove_overall,
        'rci_strict_per_agent': strict_per_agent,
        'rci_cat_per_agent': cat_per_agent,
    }

    if include_temporal:
        streak_result = compute_rci_streak(actual_actions, ideal_actions, min_streak=min_streak)
        result['rci_streak'] = streak_result['rci_streak']
        result['mean_streak_length'] = streak_result['mean_streak_length']
        result['role_switch_rate'] = streak_result['role_switch_rate']
        result['n_valid_streaks'] = streak_result['n_valid_streaks']

    return result
