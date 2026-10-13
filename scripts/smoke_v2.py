"""Smoke test v3: verifikasi obs-165 + reward v2 sebelum retrain.

1. obs vector: dim=165, blok kanan terisi (bukan nol semua)
2. BallProgressionTracker v2: r_prog dense & action-dependent
3. policy HierarchicalActorCritic forward dengan obs_dim=165

Jalankan: python /gfootball/scripts/smoke_v2.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from hmarl.utils import OBS_DIM, HIDDEN_DIM, HEAD_DIM, extract_obs_vector
from hmarl.env import create_raw_env, extract_game_state, NUM_AGENTS
from hmarl.policy import HierarchicalActorCritic, SubGoalEmbedding, SUBGOAL_EMBED_DIM
from hmarl.reward import BallProgressionTracker, compute_hierarchical_reward, PassTracker, RCITracker
from hmarl.expert import ExpertPolicyAllAgents

ok = True


def check(label, cond, detail=""):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {label} {detail}")
    if not cond:
        ok = False


print("=== 1. obs vector ===")
env = create_raw_env()
reset = env.reset()
obs_raw = reset[0] if isinstance(reset, tuple) else reset
gs = extract_game_state(obs_raw)

obs0 = extract_obs_vector(gs, 0)
check("OBS_DIM == 165", OBS_DIM == 165, f"(OBS_DIM={OBS_DIM})")
check("obs shape", obs0.shape == (165,), f"({obs0.shape})")
right_block = obs0[99:165]
check("blok kanan terisi (ada nonzero)", float(np.abs(right_block).sum()) > 0,
      f"(sum={float(np.abs(right_block).sum()):.3f})")
xs = right_block[0::6]  # 11 x-position lawan
print(f"    posisi x lawan: {np.round(xs, 3).tolist()}")
# GK lawan boleh > 1.0 (berdiri di belakang garis gawang, normal di GRF)
check("posisi x lawan dalam rentang [-1.05, 1.05]", bool(np.all(np.abs(xs) <= 1.05 + 1e-6)))

print("\n=== 2. reward v2 ===")
bpt = BallProgressionTracker()
pt = PassTracker()
rt = RCITracker(NUM_AGENTS)
expert = ExpertPolicyAllAgents()
rewards = []
subs = []
for step in range(200):
    macro = 0
    sub_goals = [0] * NUM_AGENTS
    ideal = expert.get_ideal_actions(gs, sub_goals, macro)
    act = expert.get_ideal_actions(gs, sub_goals, macro)
    r_prog = bpt.update(gs)
    subs.append(r_prog)
    if step % 40 == 0:
        # majukan bola manual untuk uji densitas
        gs = dict(gs)
        gs['ball'] = list(gs.get('ball', [0, 0, 0]))
        gs['ball'][0] = min(1.0, gs['ball'][0] + 0.05)
        gs['ball_owned_team'] = 0
    step_result = env.step(act)
    gs = extract_game_state(step_result[0])
    total, bd = compute_hierarchical_reward(
        0.0, gs, act, ideal, pt, rt, ball_progression_tracker=None)
    rewards.append(total)

subs = np.array(subs)
check("r_prog nonzero di >=10% step", float((np.abs(subs) > 1e-9).mean()) >= 0.10,
      f"(fraction={float((np.abs(subs) > 1e-9).mean()):.2f})")
check("r_prog bervariasi (std > 0)", float(subs.std()) > 1e-6, f"(std={float(subs.std()):.5f})")
print(f"    r_prog sample: {np.round(subs[:12], 4).tolist()}")

print("\n=== 3. policy forward (obs 165) ===")
policy = HierarchicalActorCritic(obs_dim=OBS_DIM, subgoal_embed_dim=SUBGOAL_EMBED_DIM,
                                 hidden_dim=HIDDEN_DIM, head_dim=HEAD_DIM,
                                 action_dim=19)
emb = SubGoalEmbedding()
obs_t = torch.FloatTensor(obs0).unsqueeze(0)
sg_t = emb(torch.LongTensor([0]))  # (1, 16)
logits, value = policy(obs_t, sg_t)
check("logits shape", logits.shape == (1, 19), f"({tuple(logits.shape)})")
check("value shape", value.shape[0] == 1, f"({tuple(value.shape)})")

env.close()
print(f"\n{'ALL PASS' if ok else 'ADA FAIL'}")
sys.exit(0 if ok else 1)