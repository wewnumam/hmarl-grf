"""Hyperparameter Sweep for HMARL using Optuna.

Usage:
    python scripts/sweep.py --trials 50 --timesteps 500000
    python scripts/sweep.py --trials 20 --timesteps 300000 --study-name hmarl_sweep

Requires: pip install optuna
"""

import argparse
import json
import os
import sys
import time
import traceback
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Also add scripts dir so we can import train module directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from hmarl.env import create_raw_env, NUM_AGENTS, extract_game_state, ACTION_SPACE_SIZE
from hmarl.policy import HierarchicalActorCritic, HierarchicalController, SubGoalEmbedding, SUBGOAL_EMBED_DIM
from hmarl.expert import ExpertPolicyAllAgents
from hmarl.reward import PassTracker, RCITracker, compute_hierarchical_reward
from hmarl.metrics import compute_win_rate, compute_goal_difference, formation_adherence_index, team_compactness
from hmarl.rci import compute_rci
from hmarl.utils import (
    set_seed, OBS_DIM, HIDDEN_DIM, HEAD_DIM, EPISODE_MAX_STEPS,
    ProgressTracker, cleanup_temp_dirs, extract_obs_vector,
)

try:
    import optuna
    from optuna.trial import Trial
except ImportError:
    print("ERROR: optuna not installed. Run: pip install optuna")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "sweep_results"


def evaluate_policy(
    policy: HierarchicalActorCritic,
    subgoal_embedding: SubGoalEmbedding,
    num_episodes: int = 20,
    seed: int = 42,
) -> Dict:
    """Evaluate a trained policy. Returns metrics dict."""
    set_seed(seed)
    env = create_raw_env(render=False)
    controller = HierarchicalController()
    expert = ExpertPolicyAllAgents()

    match_results = []
    all_rewards = []
    all_actual = []
    all_ideal = []
    all_states = []

    for _ in range(num_episodes):
        reset_result = env.reset()
        obs_raw = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        game_state = extract_game_state(obs_raw)
        pass_tracker = PassTracker()
        rci_tracker = RCITracker(NUM_AGENTS)
        ep_reward = 0.0
        ep_actual = []
        ep_ideal = []
        ep_states = []
        prev_gs = None
        done = False
        step = 0

        while not done and step < EPISODE_MAX_STEPS:
            macro = controller.get_macro_strategy(game_state)
            sub_goals = controller.get_sub_goals(game_state, macro)
            ideal = expert.get_ideal_actions(game_state, sub_goals, macro)

            joint_actions = []
            for i in range(NUM_AGENTS):
                obs_vec = extract_obs_vector(game_state, i)

                with torch.no_grad():
                    sg_embed = subgoal_embedding(
                        torch.LongTensor([sub_goals[i]]).to(DEVICE)
                    ).cpu().numpy().flatten()
                    obs_t = torch.FloatTensor(obs_vec).unsqueeze(0).to(DEVICE)
                    sg_t = torch.FloatTensor(sg_embed).unsqueeze(0).to(DEVICE)
                    logits, _ = policy(obs_t, sg_t)
                    joint_actions.append(logits.argmax(dim=-1).item())

            step_result = env.step(joint_actions)
            if len(step_result) == 5:
                obs_raw, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs_raw, reward, done, info = step_result
            team_reward = float(np.sum(reward))
            new_gs = extract_game_state(obs_raw)
            pass_tracker.update(new_gs, prev_gs)
            total_r, _ = compute_hierarchical_reward(team_reward, new_gs, joint_actions, ideal, pass_tracker, rci_tracker)
            ep_reward += total_r
            ep_actual.append(joint_actions)
            ep_ideal.append(ideal)
            ep_states.append(new_gs)
            prev_gs = game_state
            game_state = new_gs
            step += 1

        score = info.get('score', [0, 0])
        gf, ga = (score[0], score[1]) if isinstance(score, (list, tuple)) and len(score) >= 2 else (0, 0)
        result = 'win' if gf > ga else ('loss' if gf < ga else 'draw')
        match_results.append(result)
        all_rewards.append(ep_reward)
        all_actual.extend(ep_actual)
        all_ideal.extend(ep_ideal)
        all_states.extend(ep_states)

    env.close()

    rci_res = compute_rci(all_actual, all_ideal)
    fai_mean, _ = formation_adherence_index(all_states) if all_states else (0.0, 0.0)
    tc_mean, _, _, _ = team_compactness(all_states) if all_states else (0.0, 0.0, 0.0, 0.0)

    return {
        'win_rate': compute_win_rate(match_results),
        'avg_reward': float(np.mean(all_rewards)),
        'goal_difference': compute_goal_difference(
            [1 if r == 'win' else 0 for r in match_results],
            [1 if r == 'loss' else 0 for r in match_results],
        ),
        'rci_cat': rci_res['rci_cat'],
        'rci_strict': rci_res['rci_strict'],
        'fai': fai_mean,
        'compactness': tc_mean,
    }


def objective(trial: Trial, timesteps: int, eval_episodes: int, base_seed: int) -> float:
    """Optuna objective: train with sampled hyperparams, return win_rate."""
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.1, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.25, 1.0)
    minibatch_size = trial.suggest_categorical("minibatch_size", [32, 64, 128])
    num_epochs = trial.suggest_int("num_epochs", 2, 8)

    # Monkey-patch training constants
    import train as train_mod
    orig_lr = train_mod.LEARNING_RATE
    orig_gamma = train_mod.GAMMA
    orig_gae = train_mod.GAE_LAMBDA
    orig_clip = train_mod.CLIP_RANGE
    orig_ent = train_mod.ENT_COEF
    orig_vf = train_mod.VF_COEF
    orig_mb = train_mod.MINIBATCH_SIZE
    orig_ep = train_mod.NUM_EPOCHS

    train_mod.LEARNING_RATE = lr
    train_mod.GAMMA = gamma
    train_mod.GAE_LAMBDA = gae_lambda
    train_mod.CLIP_RANGE = clip_range
    train_mod.ENT_COEF = ent_coef
    train_mod.VF_COEF = vf_coef
    train_mod.MINIBATCH_SIZE = minibatch_size
    train_mod.NUM_EPOCHS = num_epochs

    try:
        set_seed(base_seed + trial.number)
        trainer = train_mod.HMARLTrainer(
            total_timesteps=timesteps,
            log_dir=f"sweep_temp/{trial.number}",
            model_dir=f"sweep_temp/{trial.number}/checkpoints",
        )
        trainer.train()

        # Evaluate best model
        ckpt_path = os.path.join(trainer.model_dir, "hmarl_model.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            trainer.policy.load_state_dict(ckpt['policy_state'])
            trainer.subgoal_embedding.load_state_dict(ckpt['subgoal_embedding_state'])
        metrics = evaluate_policy(trainer.policy, trainer.subgoal_embedding,
                                  num_episodes=eval_episodes, seed=base_seed + 1000)
        return metrics['win_rate']

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 0.0
    finally:
        # Restore original constants
        train_mod.LEARNING_RATE = orig_lr
        train_mod.GAMMA = orig_gamma
        train_mod.GAE_LAMBDA = orig_gae
        train_mod.CLIP_RANGE = orig_clip
        train_mod.ENT_COEF = orig_ent
        train_mod.VF_COEF = orig_vf
        train_mod.MINIBATCH_SIZE = orig_mb
        train_mod.NUM_EPOCHS = orig_ep


def main():
    parser = argparse.ArgumentParser(description="HMARL Hyperparameter Sweep")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Training timesteps per trial (use less than full for speed)")
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="Evaluation episodes per trial")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--study-name", type=str, default="hmarl_sweep")
    parser.add_argument("--output", type=str, default="sweep_results/best_params.json")
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs("sweep_temp", exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    print(f"Starting sweep: {args.trials} trials, {args.timesteps:,} steps each")
    print(f"Device: {DEVICE}")
    start = time.time()

    study.optimize(
        lambda trial: objective(trial, args.timesteps, args.eval_episodes, args.seed),
        n_trials=args.trials,
        show_progress_bar=True,
    )

    elapsed = time.time() - start
    print(f"\nSweep complete in {elapsed:.0f}s")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best win rate: {study.best_value:.1f}%")
    print(f"Best params: {json.dumps(study.best_params, indent=2)}")

    # Save best params
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result = {
        'best_params': study.best_params,
        'best_win_rate': study.best_value,
        'best_trial': study.best_trial.number,
        'num_trials': args.trials,
        'timesteps_per_trial': args.timesteps,
    }
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {args.output}")

    # Save full study
    study_path = os.path.join(SAVE_DIR, f"{args.study_name}.pkl")
    import pickle
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    print(f"Full study saved to {study_path}")

    cleanup_temp_dirs(['sweep_temp'])


if __name__ == "__main__":
    main()
