"""Hierarchical policy implementation: High-Level, Mid-Level, Low-Level."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from hmarl.env import (
    get_ball_x, get_ball_position, get_distance, get_player_position,
    get_player_role, ball_owned_by_us,
    ROLE_GK, ROLE_CB, ROLE_LB, ROLE_RB, ROLE_DM, ROLE_CM,
    ROLE_LM, ROLE_RM, ROLE_AM, ROLE_CF,
)

# ---------------------------------------------------------------------------
# Macro strategies (High-Level output)
# ---------------------------------------------------------------------------
STRATEGY_HIGH_PRESSING = 0
STRATEGY_COUNTER_ATTACK = 1
STRATEGY_POSSESSION = 2

MACRO_STRATEGIES = {
    STRATEGY_HIGH_PRESSING: 'High Pressing',
    STRATEGY_COUNTER_ATTACK: 'Counter Attack',
    STRATEGY_POSSESSION: 'Possession Play',
}

# ---------------------------------------------------------------------------
# Sub-goals (Mid-Level output)
# ---------------------------------------------------------------------------
SUBGOAL_ZONAL_MARKING = 0
SUBGOAL_BUILD_UP = 1
SUBGOAL_WING_ATTACK = 2
SUBGOAL_MAN_MARKING = 3
SUBGOAL_CLEARANCE = 4

SUBGOALS = {
    SUBGOAL_ZONAL_MARKING: 'Zonal Marking',
    SUBGOAL_BUILD_UP: 'Build-up',
    SUBGOAL_WING_ATTACK: 'Wing Attack',
    SUBGOAL_MAN_MARKING: 'Man Marking',
    SUBGOAL_CLEARANCE: 'Clearance',
}

NUM_SUBGOALS = 5
SUBGOAL_EMBED_DIM = 16  # Embedding dimension for sub-goal


# ---------------------------------------------------------------------------
# High-Level Policy (Rule-Based)
# ---------------------------------------------------------------------------
class HighLevelPolicy:
    """Rule-based High-Level Policy: global state -> macro strategy.

    Tabel 5 (BAB_4): Decision logic for π^H.
    """

    def decide(self, game_state: Dict) -> int:
        """Select macro strategy based on global state.

        Rules (extended from thesis Table 5):
        - ball_owned_team != our team -> High Pressing
        - ball_owned_team == our team AND ball_x >= 0.3 -> Counter Attack
        - ball_owned_team == our team AND ball_x < 0.3 ->
            check local numerical advantage at ball position
        """
        ball_x = get_ball_x(game_state)
        has_possession = ball_owned_by_us(game_state, team_id=0)

        if not has_possession:
            return STRATEGY_HIGH_PRESSING

        if ball_x >= 0.3:
            return STRATEGY_COUNTER_ATTACK

        # In midfield/back with ball: check numerical advantage
        ball_pos = get_ball_position(game_state)
        our_near = sum(
            1 for i in range(11)
            if get_distance(get_player_position(game_state, 'left', i), ball_pos) < 0.2
        )
        opp_near = sum(
            1 for j in range(11)
            if get_distance(get_player_position(game_state, 'right', j), ball_pos) < 0.2
        )

        return STRATEGY_COUNTER_ATTACK if our_near > opp_near else STRATEGY_POSSESSION


# ---------------------------------------------------------------------------
# Mid-Level Policy (Rule-Based)
# ---------------------------------------------------------------------------
class MidLevelPolicy:
    """Rule-based Mid-Level Policy: (role, macro_strategy, local_state) -> sub-goal.

    Tabel 6 (BAB_4): Decision logic for π^M.
    """

    # Distance thresholds for spatial decisions (GRF coordinates)
    DEFENSIVE_ZONE_X = -0.2
    ATTACKING_ZONE_X = 0.3
    WING_THRESHOLD_Y = 0.15

    def decide(self, game_state: Dict, player_idx: int, macro_strategy: int) -> int:
        """Select sub-goal for a specific player based on role and macro strategy.

        Rules (extended from thesis Table 6):
        - GK: Zonal Marking always
        - CB, FB: Clearance if ball in defensive zone, else Zonal Marking / Build-up
        - CM: Clearance if ball in defensive zone, else Clearance / Build-up
        - WG, CF: Wing Attack if ball on wing, else Man Marking / Build-up
        """
        role = get_player_role(game_state, 'left', player_idx)
        ball_pos = get_ball_position(game_state)
        ball_in_defense = ball_pos[0] < -0.3

        if role == ROLE_GK:
            return SUBGOAL_ZONAL_MARKING

        # Defensive roles: CB, LB, RB
        if role in (ROLE_CB, ROLE_LB, ROLE_RB):
            if ball_in_defense:
                return SUBGOAL_CLEARANCE
            if macro_strategy in (STRATEGY_HIGH_PRESSING, STRATEGY_COUNTER_ATTACK):
                return SUBGOAL_ZONAL_MARKING
            return SUBGOAL_BUILD_UP

        # Midfield roles: DM, CM, AM
        if role in (ROLE_DM, ROLE_CM, ROLE_AM):
            if ball_in_defense:
                return SUBGOAL_CLEARANCE
            if macro_strategy in (STRATEGY_HIGH_PRESSING, STRATEGY_COUNTER_ATTACK):
                return SUBGOAL_CLEARANCE
            return SUBGOAL_BUILD_UP

        # Attack roles: LM (wing), RM (wing), CF (center forward)
        if role in (ROLE_LM, ROLE_RM, ROLE_CF):
            if macro_strategy in (STRATEGY_HIGH_PRESSING, STRATEGY_COUNTER_ATTACK):
                if abs(ball_pos[1]) > 0.15:
                    return SUBGOAL_WING_ATTACK
                return SUBGOAL_MAN_MARKING
            return SUBGOAL_BUILD_UP

        return SUBGOAL_BUILD_UP


# ---------------------------------------------------------------------------
# Low-Level Policy (PPO-Trained Neural Network)
# ---------------------------------------------------------------------------
class SubGoalEmbedding(nn.Module):
    """Learnable embedding for sub-goal discrete values."""

    def __init__(self, num_subgoals: int = NUM_SUBGOALS, embed_dim: int = SUBGOAL_EMBED_DIM):
        super().__init__()
        self.embedding = nn.Embedding(num_subgoals, embed_dim)

    def forward(self, subgoal: torch.Tensor) -> torch.Tensor:
        """Map sub-goal index to embedding vector.

        Args:
            subgoal: (batch_size,) or (batch_size, 1) integer tensor
        Returns:
            (batch_size, embed_dim) embedding
        """
        if subgoal.dim() == 1:
            subgoal = subgoal.unsqueeze(1)
        return self.embedding(subgoal).squeeze(1)


class HierarchicalActorCritic(nn.Module):
    """Actor-Critic for Low-Level Policy with sub-goal conditioning.

    Architecture (from thesis Table 8):
    - Input: concat(observation_local, subgoal_embedding)
    - Shared layers: 2 × 256 ReLU
    - Policy head: 128 ReLU -> 19 linear (Categorical)
    - Value head: 128 ReLU -> 1 linear
    """

    def __init__(
        self,
        obs_dim: int = 115,
        subgoal_embed_dim: int = SUBGOAL_EMBED_DIM,
        hidden_dim: int = 256,
        head_dim: int = 128,
        action_dim: int = 19,
    ):
        super().__init__()

        input_dim = obs_dim + subgoal_embed_dim

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, action_dim),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )

    def forward(self, obs: torch.Tensor, subgoal_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            obs: (batch, obs_dim) - agent observation
            subgoal_embed: (batch, subgoal_embed_dim) - sub-goal embedding
        Returns:
            logits: (batch, action_dim) - action logits
            value: (batch, 1) - state value estimate
        """
        x = torch.cat([obs, subgoal_embed], dim=-1)
        features = self.shared(x)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value

    def get_action_and_value(
        self, obs: torch.Tensor, subgoal_embed: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or evaluate action.

        Returns:
            action, log_prob, entropy, value
        """
        logits, value = self.forward(obs, subgoal_embed)
        dist = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)

    def get_value(self, obs: torch.Tensor, subgoal_embed: torch.Tensor) -> torch.Tensor:
        """Get value estimate only."""
        _, value = self.forward(obs, subgoal_embed)
        return value.squeeze(-1)


# ---------------------------------------------------------------------------
# Hierarchical Controller (ties all levels together)
# ---------------------------------------------------------------------------
class HierarchicalController:
    """Orchestrates the 3-level hierarchy at each timestep.

    - High-Level: global state -> macro_strategy
    - Mid-Level: (role, macro_strategy, local_state) -> sub_goal per agent
    - Low-Level PPO will be called externally with sub_goal embeddings
    """

    def __init__(self):
        self.high_level = HighLevelPolicy()
        self.mid_level = MidLevelPolicy()

    def get_macro_strategy(self, game_state: Dict) -> int:
        """Decide macro strategy from global state."""
        return self.high_level.decide(game_state)

    def get_sub_goals(self, game_state: Dict, macro_strategy: int) -> List[int]:
        """Decide sub-goal for each of the 11 agents."""
        return [
            self.mid_level.decide(game_state, i, macro_strategy)
            for i in range(11)
        ]
