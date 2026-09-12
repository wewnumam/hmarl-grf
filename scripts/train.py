"""HMARL Training Script.

Trains the Hierarchical Multi-Agent Reinforcement Learning architecture
with rule-based High/Mid-Level policies and PPO-trained Low-Level policy.

Algorithm (from thesis BAB_4, Algoritme 1):
1. High-Level (rule-based) -> macro strategy
2. Mid-Level (rule-based) -> sub-goal per agent
3. Low-Level (PPO) -> action per agent
4. Combined reward: r_game + α_H·FAI + α_M·PPR + α_L·RCI
5. PPO update on Low-Level policy parameters
"""

import os
import signal
import sys
import time
from typing import Dict, List, Optional, Tuple

# Ensure project root is on path so `import hmarl` works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from hmarl.utils import set_seed as _set_seed_util, extract_obs_vector as _extract_obs_util, save_checkpoint as _save_ckpt
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except (ImportError, AttributeError):
    HAS_TB = False
    class SummaryWriter:
        def __init__(self, **kw): pass
        def add_scalar(self, *a, **kw): pass
        def close(self): pass

from hmarl.env import (
    create_raw_env, NUM_AGENTS, ACTION_SPACE_SIZE,
    extract_game_state, get_action_category,
)
from hmarl.policy import (
    HierarchicalActorCritic, HierarchicalController,
    SubGoalEmbedding, SUBGOAL_EMBED_DIM,
    STRATEGY_HIGH_PRESSING, STRATEGY_COUNTER_ATTACK,
    STRATEGY_POSSESSION,
)
from hmarl.expert import ExpertPolicyAllAgents
from hmarl.reward import PassTracker, RCITracker, BallProgressionTracker, compute_hierarchical_reward
from hmarl.metrics import (
    compute_win_rate, compute_goal_difference,
    pass_success_ratio, progressive_pass_ratio,
    positional_entropy, team_compactness,
    formation_adherence_index,
)
from hmarl.rci import compute_rci

# ---------------------------------------------------------------------------
# PPO Hyperparameters (from thesis Table 9)
# ---------------------------------------------------------------------------
LEARNING_RATE = 3e-4
GAMMA = 0.99          # Discount factor
GAE_LAMBDA = 0.95     # GAE lambda
CLIP_RANGE = 0.2      # PPO clip range
ENT_COEF = 0.01       # Entropy coefficient (c2)
VF_COEF = 0.5         # Value loss coefficient (c1)
MINIBATCH_SIZE = 64
NUM_EPOCHS = 4        # PPO epochs per update
TOTAL_TIMESTEPS = 3_000_000  # 2-5 juta (thesis)
EPISODE_MAX_STEPS = 3000   # GRF half = 3000 timesteps
LOG_FREQ = 50              # TensorBoard scalars + print every N episodes
SAVE_FREQ = 500            # Checkpoint + eval every N episodes
NUM_EVAL_EPISODES = 10     # Episodes per mid-training eval
DUMP_FREQ = 500          # Enable dump every N episodes
MAX_DUMPS = 10            # Keep at most N dump files

from hmarl.utils import HIDDEN_DIM, HEAD_DIM, OBS_DIM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_DIR = "dumps"
MODEL_DIR = "checkpoints"


def set_seed(seed: int):
    """Set random seeds for reproducibility (delegates to hmarl.utils)."""
    _set_seed_util(seed)


# ---------------------------------------------------------------------------
# Rollout Buffer
# ---------------------------------------------------------------------------
class RolloutBuffer:
    """Stores trajectory data for PPO updates."""

    def __init__(self, capacity: int = EPISODE_MAX_STEPS):
        self.capacity = capacity
        self.reset()

    def reset(self):
        self.observations = []
        self.subgoal_embeds = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.advantages = []
        self.returns = []

        # Buffer tracks per-agent transitions (one per agent per timestep)
        # Rewards are filled retroactively after computing the combined reward
        self.agent_buffer_indices = []  # indices for current timestep's agents

    def add_transition(
        self,
        obs: np.ndarray,
        subgoal_embed: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
    ):
        """Add a single agent's transition (reward filled later)."""
        self.observations.append(obs)
        self.subgoal_embeds.append(subgoal_embed)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(0.0)  # placeholder
        self.values.append(value)
        self.dones.append(False)

    def start_timestep(self):
        """Mark start of a new timestep's agent transitions."""
        self.agent_buffer_indices = []

    def record_agent_index(self):
        """Record current buffer position for the latest agent."""
        self.agent_buffer_indices.append(len(self.rewards) - 1)

    def fill_timestep_reward(self, reward_per_agent: float):
        """Fill reward for all agents in the current timestep."""
        for idx in self.agent_buffer_indices:
            if idx < len(self.rewards):
                self.rewards[idx] = reward_per_agent

    def compute_gae(
        self,
        last_value: float,
        gamma: float = GAMMA,
        gae_lambda: float = GAE_LAMBDA,
    ):
        """Compute Generalized Advantage Estimation."""
        T = len(self.rewards)
        self.advantages = [0.0] * T
        self.returns = [0.0] * T

        gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_value
                next_done = self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t]

            delta = (
                self.rewards[t]
                + gamma * next_value * (1.0 - float(self.dones[t]))
                - self.values[t]
            )
            gae = delta + gamma * gae_lambda * (1.0 - float(self.dones[t])) * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

    def get_batches(self, batch_size: int = MINIBATCH_SIZE):
        """Yield random mini-batches."""
        T = len(self.observations)
        indices = np.random.permutation(T)

        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            batch_indices = indices[start:end]

            yield {
                'obs': torch.FloatTensor(np.array([self.observations[i] for i in batch_indices])).to(DEVICE),
                'subgoal_embed': torch.FloatTensor(np.array([self.subgoal_embeds[i] for i in batch_indices])).to(DEVICE),
                'actions': torch.LongTensor([self.actions[i] for i in batch_indices]).to(DEVICE),
                'log_probs_old': torch.FloatTensor([self.log_probs[i] for i in batch_indices]).to(DEVICE),
                'advantages': torch.FloatTensor([self.advantages[i] for i in batch_indices]).to(DEVICE),
                'returns': torch.FloatTensor([self.returns[i] for i in batch_indices]).to(DEVICE),
            }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class HMARLTrainer:
    """Main training loop for HMARL architecture."""

    def __init__(
        self,
        total_timesteps: int = TOTAL_TIMESTEPS,
        log_dir: str = LOG_DIR,
        model_dir: str = MODEL_DIR,
        render: bool = False,
    ):
        self.total_timesteps = total_timesteps
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.render = render

        os.makedirs(model_dir, exist_ok=True)

        # Create environment (use raw for full game state access)
        self.env = create_raw_env(render=render, write_dumps=False)

        # Hierarchical controller (rule-based high/mid)
        self.controller = HierarchicalController()

        # Expert policy (for RCI)
        self.expert = ExpertPolicyAllAgents()

        # Sub-goal embedding
        self.subgoal_embedding = SubGoalEmbedding().to(DEVICE)

        # Low-Level Policy (PPO-trained)
        self.policy = HierarchicalActorCritic(
            obs_dim=OBS_DIM,
            subgoal_embed_dim=SUBGOAL_EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            head_dim=HEAD_DIM,
            action_dim=ACTION_SPACE_SIZE,
        ).to(DEVICE)

        # Optimizer (shared for policy + subgoal embedding)
        params = list(self.policy.parameters()) + list(self.subgoal_embedding.parameters())
        self.optimizer = optim.Adam(params, lr=LEARNING_RATE, eps=1e-5)

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Trackers
        self.pass_tracker = PassTracker()
        self.rci_tracker = RCITracker(num_agents=NUM_AGENTS)
        self.ball_progression_tracker = BallProgressionTracker()

        # Logging
        self.writer = SummaryWriter(log_dir=f"{log_dir}/hmarl_runs")

        # Log hyperparams as scalars (add_hparams removed in newer TensorBoard)
        hparam_dict = {
            'lr': LEARNING_RATE, 'gamma': GAMMA,
            'gae_lambda': GAE_LAMBDA, 'clip_range': CLIP_RANGE,
            'ent_coef': ENT_COEF, 'vf_coef': VF_COEF,
            'minibatch_size': MINIBATCH_SIZE, 'num_epochs': NUM_EPOCHS,
            'hidden_dim': HIDDEN_DIM, 'head_dim': HEAD_DIM,
        }
        for k, v in hparam_dict.items():
            self.writer.add_scalar(f'hparams/{k}', v, 0)

        # Training state
        self.global_step = 0
        self.episode_count = 0

    def _get_obs_vector(self, game_state: Dict, player_idx: int) -> np.ndarray:
        """Extract 115-dim observation vector (delegates to hmarl.utils)."""
        return _extract_obs_util(game_state, player_idx, OBS_DIM)

    def _get_subgoal_embed(self, sub_goal: int) -> np.ndarray:
        """Get sub-goal embedding vector."""
        with torch.no_grad():
            sg_tensor = torch.LongTensor([sub_goal]).to(DEVICE)
            embed = self.subgoal_embedding(sg_tensor)
            return embed.cpu().numpy().flatten()

    def _run_episode(self, training: bool = True) -> Dict:
        """Run one episode.

        Returns episode statistics.
        """
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs_raw, info = reset_result
        else:
            obs_raw = reset_result; info = {}
        # Raw obs is list[dict]; extract_game_state handles this
        game_state = extract_game_state(obs_raw)

        self.pass_tracker.reset()
        self.rci_tracker.reset()
        self.ball_progression_tracker.reset()

        episode_reward = 0.0
        episode_steps = 0
        all_actual_actions = []
        all_ideal_actions = []
        all_game_states = []

        prev_game_state = None
        done = False

        while not done and episode_steps < EPISODE_MAX_STEPS:
            # ---- Tahap 1: High-Level Policy (rule-based) ----
            macro_strategy = self.controller.get_macro_strategy(game_state)

            # ---- Tahap 2: Mid-Level Policy (rule-based) ----
            sub_goals = self.controller.get_sub_goals(game_state, macro_strategy)

            # ---- Expert ideal actions (for RCI) ----
            ideal_actions = self.expert.get_ideal_actions(
                game_state, sub_goals, macro_strategy,
            )

            # ---- Tahap 3: Low-Level Policy (PPO) per agent ----
            joint_actions = []
            step_log_probs = []
            step_values = []

            if training:
                self.buffer.start_timestep()

            for i in range(NUM_AGENTS):
                obs_vec = self._get_obs_vector(game_state, i)
                sg_embed = self._get_subgoal_embed(sub_goals[i])

                if training:
                    with torch.no_grad():
                        obs_t = torch.FloatTensor(obs_vec).unsqueeze(0).to(DEVICE)
                        sg_t = torch.FloatTensor(sg_embed).unsqueeze(0).to(DEVICE)
                        action, log_prob, _, value = self.policy.get_action_and_value(obs_t, sg_t)
                        action = action.item()
                        log_prob = log_prob.item()
                        value = value.item()
                else:
                    with torch.no_grad():
                        obs_t = torch.FloatTensor(obs_vec).unsqueeze(0).to(DEVICE)
                        sg_t = torch.FloatTensor(sg_embed).unsqueeze(0).to(DEVICE)
                        logits, value = self.policy(obs_t, sg_t)
                        action = logits.argmax(dim=-1).item()
                        log_prob = 0.0
                        value = value.item()

                joint_actions.append(action)
                step_log_probs.append(log_prob)
                step_values.append(value)

                # Store in buffer
                if training:
                    self.buffer.add_transition(
                        obs=obs_vec,
                        subgoal_embed=sg_embed,
                        action=action,
                        log_prob=log_prob,
                        value=value,
                    )
                    self.buffer.record_agent_index()

            # ---- Execute in environment ----
            step_result = self.env.step(joint_actions)
            if len(step_result) == 5:
                obs_raw, game_reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs_raw, game_reward, done, info = step_result
            team_reward = float(np.sum(game_reward))
            new_game_state = extract_game_state(obs_raw)

            # ---- Tahap 4: Compute combined reward ----
            self.pass_tracker.update(new_game_state, prev_game_state)
            total_reward, reward_breakdown = compute_hierarchical_reward(
                game_reward=team_reward,
                game_state=new_game_state,
                actual_actions=joint_actions,
                ideal_actions=ideal_actions,
                pass_tracker=self.pass_tracker,
                rci_tracker=self.rci_tracker,
                ball_progression_tracker=self.ball_progression_tracker,
            )

            # Update buffer rewards (per-agent share of team reward)
            if training:
                self.buffer.fill_timestep_reward(total_reward / NUM_AGENTS)
                # Mark done for all agents in this timestep if episode ended
                if done:
                    for idx in self.buffer.agent_buffer_indices:
                        if idx < len(self.buffer.dones):
                            self.buffer.dones[idx] = True

            episode_reward += total_reward
            episode_steps += 1
            self.global_step += 1

            all_actual_actions.append(joint_actions)
            all_ideal_actions.append(ideal_actions)
            all_game_states.append(new_game_state)

            prev_game_state = game_state
            game_state = new_game_state

        # ---- PPO Update ----
        if training and len(self.buffer.observations) > 0:
            # Compute last value for GAE
            with torch.no_grad():
                last_obs = torch.FloatTensor(self.buffer.observations[-1]).unsqueeze(0).to(DEVICE)
                last_sg = torch.FloatTensor(self.buffer.subgoal_embeds[-1]).unsqueeze(0).to(DEVICE)
                last_value = self.policy.get_value(last_obs, last_sg).item()

            self.buffer.compute_gae(last_value)
            self._ppo_update()

        self.episode_count += 1

        # Determine match result
        score_left = game_state.get('score', [0, 0])[0] if isinstance(game_state.get('score'), (list, tuple)) else 0
        score_right = game_state.get('score', [0, 0])[1] if isinstance(game_state.get('score'), (list, tuple)) else 0

        return {
            'episode_reward': episode_reward,
            'episode_length': episode_steps,
            'score_left': score_left,
            'score_right': score_right,
            'all_actual_actions': all_actual_actions,
            'all_ideal_actions': all_ideal_actions,
            'all_game_states': all_game_states,
            'reward_breakdown': reward_breakdown,
            'ball_progression_total': self.ball_progression_tracker.total_progression,
        }

    def _ppo_update(self):
        """Perform PPO update on collected rollout data."""
        self.policy.train()
        self.subgoal_embedding.train()

        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for epoch in range(NUM_EPOCHS):
            for batch in self.buffer.get_batches(MINIBATCH_SIZE):
                obs = batch['obs']
                sg_embed = batch['subgoal_embed']
                actions = batch['actions']
                old_log_probs = batch['log_probs_old']
                advantages = batch['advantages']
                returns = batch['returns']

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Forward pass
                _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                    obs, sg_embed, actions,
                )

                # Policy loss (clipped)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(new_values, returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Combined loss
                loss = (
                    policy_loss
                    + VF_COEF * value_loss
                    + ENT_COEF * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                total_pg_loss += policy_loss.item()
                total_v_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                num_updates += 1

        self.buffer.reset()

        if num_updates > 0:
            self.writer.add_scalar('loss/policy', total_pg_loss / num_updates, self.global_step)
            self.writer.add_scalar('loss/value', total_v_loss / num_updates, self.global_step)
            self.writer.add_scalar('loss/entropy', total_entropy / num_updates, self.global_step)

    def _init_training_log(self):
        """Initialize per-episode training log for visualizations."""
        self._train_log = {
            'episode_rewards': [],
            'episode_rci_cat': [],
            'episode_rci_strict': [],
            'episode_fai': [],
            'episode_ppr': [],
            'episode_compactness': [],
            'episode_goals_for': [],
            'episode_goals_against': [],
            'episode_ball_progression': [],
            'episode_action_distribution': [],  # [counts per action] per episode
        }

    def _append_training_log(self, stats: Dict):
        """Append per-episode metrics to training log."""
        self._train_log['episode_rewards'].append(float(stats['episode_reward']))
        self._train_log['episode_goals_for'].append(int(stats['score_left']))
        self._train_log['episode_goals_against'].append(int(stats['score_right']))

        # Compute per-episode RCI
        actual = stats.get('all_actual_actions', [])
        ideal = stats.get('all_ideal_actions', [])
        if actual and ideal:
            from hmarl.rci import compute_rci
            rci_res = compute_rci(actual, ideal)
            self._train_log['episode_rci_cat'].append(rci_res['rci_cat'])
            self._train_log['episode_rci_strict'].append(rci_res['rci_strict'])
        else:
            self._train_log['episode_rci_cat'].append(0.0)
            self._train_log['episode_rci_strict'].append(0.0)

        # Compute per-episode FAI, PPR, compactness from game states
        game_states = stats.get('all_game_states', [])
        if game_states:
            fai_mean, _ = formation_adherence_index(game_states)
            ppr_stats = 0.0
            if stats.get('all_actual_actions'):
                _, _, ppr_val = progressive_pass_ratio(
                    stats.get('all_actual_actions', []), game_states,
                )
                ppr_stats = ppr_val
            tc_mean, _, _, _ = team_compactness(game_states)
            self._train_log['episode_fai'].append(fai_mean)
            self._train_log['episode_ppr'].append(ppr_stats if isinstance(ppr_stats, float) else 0.0)
            self._train_log['episode_compactness'].append(tc_mean)
        else:
            self._train_log['episode_fai'].append(0.0)
            self._train_log['episode_ppr'].append(0.0)
            self._train_log['episode_compactness'].append(0.0)

        # Ball progression (sum of ball x-delta over episode)
        ball_prog = stats.get('ball_progression_total', 0.0)
        self._train_log['episode_ball_progression'].append(float(ball_prog))

        # Action distribution: count of each action across all agents in episode
        all_actions = stats.get('all_actual_actions', [])
        action_counts = [0] * 19
        for timestep_actions in all_actions:
            for a in timestep_actions:
                if 0 <= a < 19:
                    action_counts[a] += 1
        self._train_log['episode_action_distribution'].append(action_counts)

    def _save_training_log(self):
        """Save training log to JSON for visualization."""
        import json as _json
        path = os.path.join(self.log_dir, 'training_log.json')
        # Convert numpy types
        serializable = {}
        for k, v in self._train_log.items():
            if isinstance(v, list):
                serializable[k] = [float(x) if hasattr(x, 'item') else x for x in v]
            else:
                serializable[k] = v
        with open(path, 'w') as f:
            _json.dump(serializable, f, indent=2)
        print(f"Training log saved: {path}")

    def train(self):
        """Main training loop."""
        print(f"HMARL Training | Device: {DEVICE}")
        print(f"Total timesteps: {self.total_timesteps:,}")
        print(f"Log dir: {self.log_dir}")
        print(f"Dump every {DUMP_FREQ} episodes (max {MAX_DUMPS} kept)")
        print(f"{'='*60}")

        start_time = time.time()
        episode_rewards = []
        self._init_training_log()

        while self.global_step < self.total_timesteps:
            # --- Selective dump: recreate env with dumps at interval ---
            dump_this_episode = (
                DUMP_FREQ > 0
                and self.episode_count % DUMP_FREQ == 0
                and self.episode_count > 0
            )
            if dump_this_episode:
                self.env.close()
                self.env = create_raw_env(render=self.render, write_dumps=True)
                print(f"  [DUMP] Episode {self.episode_count} — dumps enabled")

            stats = self.run_episode(training=True)

            if dump_this_episode:
                self.env.close()
                self.env = create_raw_env(render=self.render, write_dumps=False)
                self._cleanup_dumps()

            episode_rewards.append(stats['episode_reward'])
            self._append_training_log(stats)

            # Logging
            if self.episode_count % LOG_FREQ == 0 and self.episode_count > 0:
                avg_reward = np.mean(episode_rewards[-100:])
                elapsed = time.time() - start_time
                steps_per_sec = self.global_step / max(elapsed, 1)

                # Rolling window for thesis-critical metrics (last 50 eps)
                window = min(50, len(self._train_log['episode_rci_cat']))
                rci_cat_avg = np.mean(self._train_log['episode_rci_cat'][-window:])
                rci_strict_avg = np.mean(self._train_log['episode_rci_strict'][-window:])
                fai_avg = np.mean(self._train_log['episode_fai'][-window:])
                ppr_avg = np.mean(self._train_log['episode_ppr'][-window:])
                comp_avg = np.mean(self._train_log['episode_compactness'][-window:])

                print(
                    f"Ep {self.episode_count:6d} | "
                    f"Step {self.global_step:8d}/{self.total_timesteps:,} | "
                    f"Avg Reward (100ep): {avg_reward:8.2f} | "
                    f"RCI: {rci_cat_avg:.3f} | FAI: {fai_avg:.3f} | "
                    f"PPR: {ppr_avg:.3f} | Compact: {comp_avg:.3f} | "
                    f"Steps/s: {steps_per_sec:.1f}"
                )

                # Reward curve
                self.writer.add_scalar('reward/episode', stats['episode_reward'], self.episode_count)
                self.writer.add_scalar('reward/avg_100', avg_reward, self.episode_count)
                self.writer.add_scalar('training/episode_length', stats['episode_length'], self.episode_count)

                # Reward breakdown (last episode)
                rb = stats.get('reward_breakdown', {})
                if rb:
                    self.writer.add_scalar('reward/game', rb.get('game_reward', 0), self.episode_count)
                    self.writer.add_scalar('reward/r_high', rb.get('r_high', 0), self.episode_count)
                    self.writer.add_scalar('reward/r_mid', rb.get('r_mid', 0), self.episode_count)
                    self.writer.add_scalar('reward/r_low', rb.get('r_low', 0), self.episode_count)

                # Thesis-critical metrics (rolling avg)
                self.writer.add_scalar('metrics/rci_cat', rci_cat_avg, self.episode_count)
                self.writer.add_scalar('metrics/rci_strict', rci_strict_avg, self.episode_count)
                self.writer.add_scalar('metrics/fai', fai_avg, self.episode_count)
                self.writer.add_scalar('metrics/ppr', ppr_avg, self.episode_count)
                self.writer.add_scalar('metrics/compactness', comp_avg, self.episode_count)

            # Save checkpoint + mid-training eval
            if self.episode_count % SAVE_FREQ == 0 and self.episode_count > 0:
                self._save_checkpoint()
                self._save_training_log()

                # Mid-training evaluation
                eval_stats = self._quick_eval(num_episodes=10)
                self.writer.add_scalar('eval/win_rate', eval_stats['win_rate'], self.episode_count)
                self.writer.add_scalar('eval/avg_reward', eval_stats['avg_reward'], self.episode_count)
                self.writer.add_scalar('eval/goal_diff', eval_stats['goal_diff'], self.episode_count)
                print(
                    f"  [EVAL] WR: {eval_stats['win_rate']:.1f}% | "
                    f"Avg Reward: {eval_stats['avg_reward']:.2f} | "
                    f"GD: {eval_stats['goal_diff']}"
                )

        self._save_checkpoint()
        self._save_training_log()
        self.writer.close()
        print(f"\nTraining complete. Total time: {time.time() - start_time:.1f}s")

    def run_episode(self, training: bool = True) -> Dict:
        """Public method to run an episode."""
        if training:
            self.policy.train()
            self.subgoal_embedding.train()
        else:
            self.policy.eval()
            self.subgoal_embedding.eval()

        return self._run_episode(training=training)

    def _save_checkpoint(self):
        """Save model checkpoint with reproducibility metadata."""
        path = os.path.join(self.model_dir, "hmarl_model.pt")
        _save_ckpt(
            path,
            policy_state=self.policy.state_dict(),
            subgoal_embedding_state=self.subgoal_embedding.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            extra={
                'global_step': self.global_step,
                'episode_count': self.episode_count,
                'hyperparams': {
                    'learning_rate': LEARNING_RATE,
                    'gamma': GAMMA,
                    'gae_lambda': GAE_LAMBDA,
                    'clip_range': CLIP_RANGE,
                    'ent_coef': ENT_COEF,
                    'vf_coef': VF_COEF,
                    'minibatch_size': MINIBATCH_SIZE,
                    'num_epochs': NUM_EPOCHS,
                    'total_timesteps': self.total_timesteps,
                    'hidden_dim': HIDDEN_DIM,
                    'head_dim': HEAD_DIM,
                    'obs_dim': OBS_DIM,
                },
            },
        )
        print(f"Checkpoint saved: {path}")

    def _cleanup_dumps(self):
        """Remove old dump files, keep only the most recent MAX_DUMPS."""
        import glob
        dump_files = sorted(
            glob.glob(os.path.join(self.log_dir, "*.dump")),
            key=os.path.getmtime,
        )
        if len(dump_files) > MAX_DUMPS:
            for f in dump_files[: len(dump_files) - MAX_DUMPS]:
                os.remove(f)
            print(f"  [DUMP] Cleaned up {len(dump_files) - MAX_DUMPS} old dumps ({len(dump_files)} → {MAX_DUMPS})")


    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self.policy.load_state_dict(checkpoint['policy_state'])
        self.subgoal_embedding.load_state_dict(checkpoint['subgoal_embedding_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.global_step = checkpoint['global_step']
        self.episode_count = checkpoint['episode_count']
        print(f"Checkpoint loaded: {path} (step {self.global_step})")

    def _quick_eval(self, num_episodes: int = 10) -> Dict:
        """Run quick evaluation during training. Returns eval metrics."""
        self.policy.eval()
        self.subgoal_embedding.eval()

        eval_env = create_raw_env(render=False)
        controller = HierarchicalController()
        expert = ExpertPolicyAllAgents()

        wins = 0
        total_goals_for = 0
        total_goals_against = 0
        total_reward = 0.0

        for _ in range(num_episodes):
            reset_result = eval_env.reset()
            obs_raw = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            game_state = extract_game_state(obs_raw)
            ep_reward = 0.0
            done = False
            step = 0

            while not done and step < EPISODE_MAX_STEPS:
                macro = controller.get_macro_strategy(game_state)
                sub_goals = controller.get_sub_goals(game_state, macro)

                joint_actions = []
                for i in range(NUM_AGENTS):
                    obs_vec = self._get_obs_vector(game_state, i)
                    with torch.no_grad():
                        sg_embed = self.subgoal_embedding(
                            torch.LongTensor([sub_goals[i]]).to(DEVICE)
                        ).cpu().numpy().flatten()
                        obs_t = torch.FloatTensor(obs_vec).unsqueeze(0).to(DEVICE)
                        sg_t = torch.FloatTensor(sg_embed).unsqueeze(0).to(DEVICE)
                        logits, _ = self.policy(obs_t, sg_t)
                        joint_actions.append(logits.argmax(dim=-1).item())

                step_result = eval_env.step(joint_actions)
                if len(step_result) == 5:
                    obs_raw, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs_raw, reward, done, info = step_result
                ep_reward += float(np.sum(reward))
                game_state = extract_game_state(obs_raw)
                step += 1

            score = game_state.get('score', [0, 0])
            gf, ga = (score[0], score[1]) if isinstance(score, (list, tuple)) and len(score) >= 2 else (0, 0)
            if gf > ga:
                wins += 1
            total_goals_for += gf
            total_goals_against += ga
            total_reward += ep_reward

        eval_env.close()
        self.policy.train()
        self.subgoal_embedding.train()

        return {
            'win_rate': (wins / num_episodes * 100.0),
            'avg_reward': total_reward / num_episodes,
            'goal_diff': total_goals_for - total_goals_against,
            'num_episodes': num_episodes,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train HMARL on GRF 11v11")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS,
                        help="Total training timesteps")
    parser.add_argument("--render", action="store_true",
                        help="Render during training")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR)
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--dump-freq", type=int, default=DUMP_FREQ,
                        help="Enable dump every N episodes (0=never)")
    parser.add_argument("--max-dumps", type=int, default=MAX_DUMPS,
                        help="Max dump files to keep")

    args = parser.parse_args()
    DUMP_FREQ = args.dump_freq
    MAX_DUMPS = args.max_dumps

    set_seed(args.seed)

    trainer = HMARLTrainer(
        total_timesteps=args.timesteps,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        render=args.render,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Graceful shutdown: save checkpoint on SIGINT/SIGTERM
    def _shutdown(signum, frame):
        sig_name = signal.Signals(signum).name
        print(f"\n[SHUTDOWN] {sig_name} received — saving checkpoint...")
        trainer._save_checkpoint()
        trainer._save_training_log()
        trainer.writer.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    trainer.train()
