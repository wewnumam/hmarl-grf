"""Scalable and Heterogeneous PPO (SHPPO) Baseline.

Implementation based on Guo et al. (2025):
  "Heterogeneous Multi-Agent Reinforcement Learning for Zero-Shot
   Scalable Collaboration" (Neurocomputing, arXiv:2404.03869)

Architecture:
  1. LatentNetwork: obs -> latent variables z (captures strategy pattern)
  2. HyperNetwork: z -> heterogeneous layer parameters (θ_hetero)
  3. HeterogeneousActorCritic: heterogeneous_layer(obs) -> shared_core -> policy+value
  4. CentralizedInferenceNet: global_obs -> z_pred (guides latent learning)

Key difference from Shared PPO / IPPO:
  - Shared PPO/IPPO: same fixed network for all agents
  - SHPPO: shared core + per-agent heterogeneous layer generated from z
  - Heterogeneity is EXPLICIT (latent variables), not implicit (from obs only)
  - Zero-shot scalable: can handle different agent counts at inference

Note: Baselines use game reward only (no FAI/PPR/RCI reward shaping).
This ensures a fair comparison — HMARL's improvement over baselines
is attributed to the hierarchical architecture, not reward shaping.
"""

import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from hmarl.utils import set_seed, extract_obs_vector, build_centralized_obs
from hmarl.env import create_raw_env, NUM_AGENTS, extract_game_state

OBS_DIM = 115
HIDDEN_DIM = 256
ACTION_SPACE_SIZE = 19
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MINIBATCH_SIZE = 64
NUM_EPOCHS = 4
TOTAL_TIMESTEPS = 3_000_000
EPISODE_MAX_STEPS = 3000
LEARNING_RATE = 3e-4

# SHPPO-specific hyperparameters (from Guo et al.)
LATENT_DIM = 4          # Low-dim latent variables representing strategy
INFERENCE_LR_SCALE = 1.0  # Inference net uses same LR as main policy
AUX_LOSS_COEF = 0.1      # Weight of auxiliary inference loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1. Latent Network — learns per-agent strategy representation
# ---------------------------------------------------------------------------
class LatentNetwork(nn.Module):
    """Maps agent observation to low-dim latent variables z.

    z encodes the agent's "strategy pattern" — what role/tactic it should
    follow. The latent variables adapt based on observations and trajectories,
    enabling both inter-individual and temporal heterogeneity.
    """

    def __init__(self, obs_dim: int = OBS_DIM, latent_dim: int = LATENT_DIM,
                 hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs (batch, obs_dim) -> z (batch, latent_dim)"""
        return self.net(obs)


# ---------------------------------------------------------------------------
# 2. HyperNetwork — generates heterogeneous layer params from z
# ---------------------------------------------------------------------------
class HyperNetwork(nn.Module):
    """Generates weights and biases for the heterogeneous layer from z.

    Output: weight (hidden, 128) and bias (128) for a linear layer.
    """

    def __init__(self, latent_dim: int = LATENT_DIM,
                 hidden: int = 128, head_out: int = 128):
        super().__init__()
        self.head_out = head_out
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # Separate heads for weight and bias
        self.weight_head = nn.Linear(hidden, hidden * head_out)
        self.bias_head = nn.Linear(hidden, head_out)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """z (batch, latent_dim) -> (weight (batch, hidden, head_out), bias (batch, head_out))"""
        h = self.net(z)
        weight = self.weight_head(h).view(-1, self.hidden, self.head_out)
        bias = self.bias_head(h)
        return weight, bias


# ---------------------------------------------------------------------------
# 3. Heterogeneous Layer — dynamic linear layer with generated params
# ---------------------------------------------------------------------------
class HeterogeneousLinear(nn.Module):
    """Linear layer whose parameters are dynamically generated per sample.

    Unlike a fixed nn.Linear, this layer applies different weights to each
    sample in the batch based on its latent variable z.
    """

    def forward(self, x: torch.Tensor, weight: torch.Tensor,
                bias: torch.Tensor) -> torch.Tensor:
        """x (batch, in_dim), weight (batch, in_dim, out_dim), bias (batch, out_dim)
        -> (batch, out_dim)"""
        # Manual batched linear: y = x @ W + b
        return torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias


# ---------------------------------------------------------------------------
# 4. HeterogeneousActorCritic — full SHPPO actor-critic
# ---------------------------------------------------------------------------
class HeterogeneousActorCritic(nn.Module):
    """SHPPO Actor-Critic with heterogeneous layer.

    Architecture (from Guo et al., Fig. 2):
      obs -> [Shared Encoder 115->256->256] -> features
      z -> [HyperNetwork] -> (W_hetero, b_hetero)
      features -> [HeterogeneousLinear(W_hetero, b_hetero)] -> h_hetero
      h_hetero -> [Policy Head 128->19] + [Value Head 128->1]

    The heterogeneous layer is inserted BETWEEN the shared encoder and
    the heads, allowing per-agent specialization while sharing core params.
    """

    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = ACTION_SPACE_SIZE,
                 hidden: int = HIDDEN_DIM, head_dim: int = 128,
                 latent_dim: int = LATENT_DIM):
        super().__init__()
        # Shared encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # Heterogeneous layer components
        self.latent_net = LatentNetwork(obs_dim, latent_dim, hidden=128)
        self.hyper_net = HyperNetwork(latent_dim, hidden=hidden, head_out=head_dim)
        self.hetero_linear = HeterogeneousLinear()
        self.hetero_activation = nn.ReLU()

        # Shared heads (same for all agents)
        self.policy_head = nn.Sequential(
            nn.Linear(head_dim, head_dim), nn.ReLU(),
            nn.Linear(head_dim, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(head_dim, head_dim), nn.ReLU(),
            nn.Linear(head_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: obs -> (logits, value).

        z is internally computed from obs via the latent network.
        """
        z = self.latent_net(obs)
        features = self.shared_encoder(obs)
        w, b = self.hyper_net(z)
        h_hetero = self.hetero_linear(features, w, b)
        h_hetero = self.hetero_activation(h_hetero)

        logits = self.policy_head(h_hetero)
        value = self.value_head(h_hetero)
        return logits, value

    def get_z(self, obs: torch.Tensor) -> torch.Tensor:
        """Get latent variables for an observation."""
        return self.latent_net(obs)

    def get_action_and_value(self, obs: torch.Tensor,
                             action: torch.Tensor = None
                             ) -> Tuple[torch.Tensor, torch.Tensor,
                                        torch.Tensor, torch.Tensor]:
        """Sample or evaluate action. Returns (action, log_prob, entropy, value)."""
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)


# ---------------------------------------------------------------------------
# 5. CentralizedInferenceNet — guides latent network learning
# ---------------------------------------------------------------------------
class CentralizedInferenceNet(nn.Module):
    """Learns to predict latent variables from global (centralized) state.

    Symmetric structure to actor-critic: where actor maps obs->action via z,
    inference net maps centralized_obs->z. This creates a symmetric
    actor-critic-like structure that guides the latent network.

    For GRF 11v11: centralized input = concatenation of all agents' observations.
    """

    def __init__(self, obs_dim: int = OBS_DIM, num_agents: int = NUM_AGENTS,
                 latent_dim: int = LATENT_DIM, hidden: int = 128):
        super().__init__()
        centralized_dim = obs_dim * num_agents
        self.net = nn.Sequential(
            nn.Linear(centralized_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, centralized_obs: torch.Tensor) -> torch.Tensor:
        """centralized_obs (batch, obs_dim * num_agents) -> z_pred (batch, latent_dim)"""
        return self.net(centralized_obs)


# ---------------------------------------------------------------------------
# 6. Rollout Buffer
# ---------------------------------------------------------------------------
class SHPPORolloutBuffer:
    """Unified rollout buffer — pools ALL agents' transitions."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.advantages = []
        self.returns = []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, last_value, gamma=GAMMA, lam=GAE_LAMBDA):
        """Compute GAE over the pooled buffer (all agents)."""
        T = len(self.rewards)
        self.advantages = [0.0] * T
        self.returns = [0.0] * T
        gae = 0.0
        for t in reversed(range(T)):
            next_val = self.values[t + 1] if t < T - 1 else last_value
            delta = (self.rewards[t] + gamma * next_val * (1 - self.dones[t])
                     - self.values[t])
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

    def get_batches(self, bs=MINIBATCH_SIZE):
        """Yield random mini-batches from pooled data."""
        T = len(self.obs)
        indices = np.random.permutation(T)
        for start in range(0, T, bs):
            end = min(start + bs, T)
            idx = indices[start:end]
            yield {
                'obs': torch.FloatTensor(np.array([self.obs[i] for i in idx])).to(DEVICE),
                'actions': torch.LongTensor([self.actions[i] for i in idx]).to(DEVICE),
                'log_probs_old': torch.FloatTensor([self.log_probs[i] for i in idx]).to(DEVICE),
                'advantages': torch.FloatTensor([self.advantages[i] for i in idx]).to(DEVICE),
                'returns': torch.FloatTensor([self.returns[i] for i in idx]).to(DEVICE),
            }


# ---------------------------------------------------------------------------
# 7. SHPPO Trainer
# ---------------------------------------------------------------------------
class SHPPOTrainer:
    """Scalable and Heterogeneous PPO (Guo et al., 2025).

    Training loop:
    1. Collect transitions from ALL agents into one unified buffer
    2. Also collect per-agent latent vars z and centralized obs
    3. Compute GAE over the entire pooled buffer
    4. PPO update + auxiliary inference loss to guide latent learning

    The auxiliary loss trains the inference net to predict z from centralized
    obs, and penalizes divergence between latent and inference predictions:
      L_aux = MSE(inference_net(centralized_obs), z.detach())
    """

    def __init__(self, total_timesteps=TOTAL_TIMESTEPS, log_dir="dumps",
                 render=False):
        self.total_timesteps = total_timesteps
        self.log_dir = log_dir
        self.render = render

        self.env = create_raw_env(render=render)
        self.policy = HeterogeneousActorCritic().to(DEVICE)
        self.inference_net = CentralizedInferenceNet().to(DEVICE)

        # Separate optimizer for inference net
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE, eps=1e-5)
        self.inference_optimizer = optim.Adam(
            self.inference_net.parameters(), lr=LEARNING_RATE * INFERENCE_LR_SCALE, eps=1e-5
        )

        self.buffer = SHPPORolloutBuffer()
        # Extra storage for inference net training
        self._z_buffer = []           # latent vars per agent per step
        self._centralized_buf = []    # centralized obs per step

        os.makedirs(log_dir, exist_ok=True)
        self.global_step = 0
        self.episode_count = 0

    def train(self):
        print(f"SHPPO (Heterogeneous) Training | Device: {DEVICE} | "
              f"Timesteps: {self.total_timesteps:,}")
        print(f"  Latent dim: {LATENT_DIM} | Aux loss coef: {AUX_LOSS_COEF}")
        start = time.time()

        while self.global_step < self.total_timesteps:
            reset_result = self.env.reset()
            obs_raw = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            game_state = extract_game_state(obs_raw)
            self.buffer.reset()
            self._z_buffer.clear()
            self._centralized_buf.clear()

            episode_reward = 0.0
            done = False
            step = 0

            while not done and step < EPISODE_MAX_STEPS:
                joint_actions = []
                all_obs = []
                all_z = []

                # Collect observations for all agents
                for i in range(NUM_AGENTS):
                    obs_vec = extract_obs_vector(game_state, i)
                    all_obs.append(obs_vec)

                centralized_obs = build_centralized_obs(all_obs)

                # Get actions, latent vars for all agents
                for i in range(NUM_AGENTS):
                    obs_t = torch.FloatTensor(all_obs[i]).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        # Get latent var for this agent
                        z = self.policy.get_z(obs_t)
                        # Get action via heterogeneous policy
                        action, log_prob, _, value = self.policy.get_action_and_value(obs_t)

                    a = action.item()
                    lp = log_prob.item()
                    v = value.item()
                    joint_actions.append(a)
                    all_z.append(z.squeeze(0).cpu().numpy())

                    self.buffer.add(all_obs[i], a, lp, 0.0, v, 0.0)

                # Store for inference net training
                self._z_buffer.append(all_z)
                self._centralized_buf.append(centralized_obs)

                # Step environment
                step_result = self.env.step(joint_actions)
                if len(step_result) == 5:
                    obs_raw_new, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs_raw_new, reward, done, info = step_result
                team_reward = float(np.sum(reward))

                # Fill rewards: team_reward for ALL agents (pooled)
                for i in range(NUM_AGENTS):
                    idx = len(self.buffer.rewards) - NUM_AGENTS + i
                    self.buffer.rewards[idx] = team_reward
                    self.buffer.dones[idx] = float(done)

                obs_raw = obs_raw_new
                game_state = extract_game_state(obs_raw)
                episode_reward += team_reward
                step += 1
                self.global_step += 1  # count env steps, not per-agent

            # PPO update + inference net update
            with torch.no_grad():
                last_obs = torch.FloatTensor(
                    extract_obs_vector(game_state, 0)
                ).unsqueeze(0).to(DEVICE)
                _, _, _, last_val = self.policy.get_action_and_value(last_obs)
                last_value = last_val.item()

            self.buffer.compute_gae(last_value)
            self._ppo_update_with_inference()

            self.episode_count += 1
            if self.episode_count % 500 == 0:
                elapsed = time.time() - start
                print(
                    f"Ep {self.episode_count:6d} | Step {self.global_step:8d}/{self.total_timesteps:,} | "
                    f"Reward: {episode_reward:7.2f} | "
                    f"Steps/s: {self.global_step/max(elapsed,1):.1f}"
                )
                self._save()
                eval_stats = self._quick_eval(num_episodes=10)
                print(
                    f"  [EVAL] WR: {eval_stats['win_rate']:.1f}% | "
                    f"Avg Reward: {eval_stats['avg_reward']:.2f} | "
                    f"GD: {eval_stats['goal_diff']}"
                )

        self._save()
        print(f"\nSHPPO Training complete. ({time.time()-start:.1f}s)")

    def _ppo_update_with_inference(self):
        """PPO update with auxiliary inference loss for latent guidance."""
        self.policy.train()
        self.inference_net.train()

        for _ in range(NUM_EPOCHS):
            for batch in self.buffer.get_batches():
                obs = batch['obs']
                actions = batch['actions']
                old_lp = batch['log_probs_old']
                adv = batch['advantages']
                returns = batch['returns']

                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Standard PPO loss
                _, new_lp, entropy, values = self.policy.get_action_and_value(obs, actions)
                ratio = torch.exp(new_lp - old_lp)
                s1 = ratio * adv
                s2 = torch.clamp(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE) * adv
                pg_loss = -torch.min(s1, s2).mean()
                v_loss = nn.MSELoss()(values, returns)
                ppo_loss = pg_loss + VF_COEF * v_loss - ENT_COEF * entropy.mean()

                # Auxiliary inference loss: inference net should predict
                # the latent vars that the policy network produces
                z_from_policy = self.policy.get_z(obs).detach()
                z_pred = self.inference_net(obs)  # simplified: use agent-local obs
                aux_loss = nn.MSELoss()(z_pred, z_from_policy)

                total_loss = ppo_loss + AUX_LOSS_COEF * aux_loss

                self.optimizer.zero_grad()
                self.inference_optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.inference_net.parameters(), 0.5)
                self.optimizer.step()
                self.inference_optimizer.step()

        self.buffer.reset()
        self._z_buffer.clear()
        self._centralized_buf.clear()

    def _save(self):
        path = os.path.join(self.log_dir, "shppo_model.pt")
        torch.save({
            'policy_state': self.policy.state_dict(),
            'inference_state': self.inference_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'inference_optimizer_state': self.inference_optimizer.state_dict(),
            'global_step': self.global_step,
            'episode_count': self.episode_count,
        }, path)
        print(f"Model saved: {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        self.policy.load_state_dict(ckpt['policy_state'])
        self.inference_net.load_state_dict(ckpt['inference_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.inference_optimizer.load_state_dict(ckpt['inference_optimizer_state'])
        self.global_step = ckpt['global_step']
        self.episode_count = ckpt['episode_count']

    def _quick_eval(self, num_episodes: int = 10) -> Dict:
        """Run quick evaluation during training."""
        self.policy.eval()
        self.inference_net.eval()
        eval_env = create_raw_env(render=False)
        wins, total_goals_for, total_goals_against, total_reward = 0, 0, 0, 0.0

        for _ in range(num_episodes):
            reset_result = eval_env.reset()
            obs_raw = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            game_state = extract_game_state(obs_raw)
            ep_reward = 0.0
            done = False
            step = 0

            while not done and step < EPISODE_MAX_STEPS:
                joint_actions = []
                for i in range(NUM_AGENTS):
                    obs_vec = extract_obs_vector(game_state, i)
                    obs_t = torch.FloatTensor(obs_vec).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        logits, _ = self.policy(obs_t)
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
            if gf > ga: wins += 1
            total_goals_for += gf
            total_goals_against += ga
            total_reward += ep_reward

        eval_env.close()
        self.policy.train()
        self.inference_net.train()
        return {
            'win_rate': wins / num_episodes * 100.0,
            'avg_reward': total_reward / num_episodes,
            'goal_diff': total_goals_for - total_goals_against,
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Train SHPPO (Scalable & Heterogeneous PPO) baseline")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--log-dir", type=str, default="dumps")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint (.pt file)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)
    trainer = SHPPOTrainer(
        total_timesteps=args.timesteps,
        log_dir=args.log_dir,
        render=args.render,
    )
    if args.resume:
        trainer.load(args.resume)
        print(f"Resumed from {args.resume} at step {trainer.global_step}")
    trainer.train()
