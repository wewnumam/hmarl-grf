"""Role Coherence Index (RCI) evaluation metric.

RCI = (1/NT) · Σ_i Σ_t f_match(a_{i,t}, a*)

Two variants:
- RCI_strict: f_strict = I(a == a*)  (exact match)
- RCI_cat: f_cat = I(C(a) == C(a*))  (category match)

Action categories:
  passing:   {short_pass, long_pass, high_pass}
  shooting:  {shot}
  movement:  {8 directions, sprint}
  ball_control: {dribble}
  defensive: {idle, release_direction, release_sprint, sliding, release_dribble}
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
    if not actual_actions or not ideal_actions:
        return 0.0, []

    num_agents = len(actual_actions[0])
    T = len(actual_actions)

    per_agent_scores = np.zeros(num_agents)

    for t in range(min(T, len(ideal_actions))):
        for i in range(num_agents):
            per_agent_scores[i] += f_category(
                actual_actions[t][i],
                ideal_actions[t][i],
            )

    rci_per_agent = (per_agent_scores / T).tolist()
    rci_overall = float(np.mean(rci_per_agent))

    return rci_overall, rci_per_agent


def compute_rci(
    actual_actions: List[List[int]],
    ideal_actions: List[List[int]],
) -> Dict[str, any]:
    """Compute both RCI variants.

    Returns dict with:
        - rci_strict: overall strict RCI
        - rci_cat: overall category RCI
        - rci_strict_per_agent: per-agent strict RCI
        - rci_cat_per_agent: per-agent category RCI
    """
    strict_overall, strict_per_agent = compute_rci_strict(actual_actions, ideal_actions)
    cat_overall, cat_per_agent = compute_rci_category(actual_actions, ideal_actions)

    return {
        'rci_strict': strict_overall,
        'rci_cat': cat_overall,
        'rci_strict_per_agent': strict_per_agent,
        'rci_cat_per_agent': cat_per_agent,
    }
