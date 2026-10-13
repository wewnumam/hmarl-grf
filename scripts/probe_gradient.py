"""Probe 2: apakah PPO benar-benar belajar?

(a) Baca scalar dari tfevents (loss policy/value/entropy, reward)
(b) Bandingkan logits policy terhadap obs identik: checkpoint 3M vs init acak
(c) Sebaran logits (entropy, margin argmax-2)

Jalankan di container:
  python /gfootball/scripts/probe_gradient.py
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from hmarl.utils import OBS_DIM, load_hmarl_checkpoint, HIDDEN_DIM, HEAD_DIM, ACTION_SPACE_SIZE
from hmarl.policy import HierarchicalActorCritic, SUBGOAL_EMBED_DIM


def read_tfevents():
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("tensorboard not installed -> skip (a)")
        return
    files = glob.glob('dumps/hmarl_runs/events.out.tfevents.*')
    if not files:
        print("no tfevents found -> skip (a)")
        return
    ea = EventAccumulator(files[0], size_guidance={'scalars': 0})
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    print("=== (a) tfevents scalar tags ===")
    for t in sorted(tags):
        ev = ea.Scalars(t)
        if not ev:
            continue
        vals = [e.value for e in ev]
        n = len(vals)
        head = vals[: max(1, n // 10)]
        tail = vals[-max(1, n // 10):]
        m = lambda a: float(np.mean(a))
        print(f"  {t:36s} n={n:5d} first10%={m(head):+.6g} last10%={m(tail):+.6g} "
              f"min={min(vals):+.6g} max={max(vals):+.6g}")


def compare_logits():
    print("\n=== (b) checkpoint vs fresh-init logits ===")
    torch.manual_seed(0)
    fresh = HierarchicalActorCritic(
        obs_dim=OBS_DIM, subgoal_embed_dim=SUBGOAL_EMBED_DIM,
        hidden_dim=HIDDEN_DIM, head_dim=HEAD_DIM, action_dim=ACTION_SPACE_SIZE)
    fresh.eval()

    policy, sg_emb, meta = load_hmarl_checkpoint('checkpoints/hmarl_model.pt', device='cpu')
    print("checkpoint meta keys:", sorted(meta.keys()))
    for k in ('timesteps', 'total_timesteps', 'episode', 'updates', 'step'):
        if k in meta:
            print(f"  meta[{k}] = {meta[k]}")

    # obs deterministik: bola di tengah, formasi garis
    def make_obs(seed):
        rng = np.random.RandomState(seed)
        ball = np.array([-0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        vec = list(ball)
        for i in range(11):
            vec += [1.0 if i == 0 else 0.0,
                    -0.8 + 0.1 * i, rng.uniform(-0.3, 0.3),
                    0.0, 0.0, 0.0, 0.0, float(i % 10)]
        vec = (vec + [0.0] * OBS_DIM)[:OBS_DIM]
        return np.array(vec, dtype=np.float32)

    sg_ids = torch.zeros(11, dtype=torch.long)
    sg = sg_emb(sg_ids)

    diffs, ent_ck, ent_fresh, agree = [], [], [], 0
    for s in range(11):
        obs = torch.FloatTensor(make_obs(s)).unsqueeze(0)
        sgv = sg[s].unsqueeze(0)
        with torch.no_grad():
            l_ck, _ = policy(obs, sgv)
            l_fr, _ = fresh(obs, sgv)
        p_ck = torch.softmax(l_ck, -1).numpy()[0]
        p_fr = torch.softmax(l_fr, -1).numpy()[0]
        diffs.append(float(np.abs(p_ck - p_fr).sum()))
        ent_ck.append(float(-(p_ck * np.log(p_ck + 1e-12)).sum()))
        ent_fresh.append(float(-(p_fr * np.log(p_fr + 1e-12)).sum()))
        agree += int(p_ck.argmax() == p_fr.argmax())
        if s < 3:
            top = lambda p: (int(p.argmax()), round(float(p.max()), 4))
            print(f"  obs{s}: ckpt argmax={top(p_ck)} fresh argmax={top(p_fr)} "
                  f"L1(p_ck,p_fr)={diffs[-1]:.4f}")
    print(f"  mean L1(ckpt,fresh) = {float(np.mean(diffs)):.4f}  (0 = identik dgn init)")
    print(f"  argmax agreement = {agree}/11")
    print(f"  mean policy entropy: ckpt={float(np.mean(ent_ck)):.4f} "
          f"fresh={float(np.mean(ent_fresh)):.4f}  (uniform 19 aksi = {np.log(19):.4f})")

    # margin argmax vs runner-up pada 100 obs acak
    margins = []
    for s in range(100):
        obs = torch.FloatTensor(make_obs(1000 + s)).unsqueeze(0)
        with torch.no_grad():
            l, _ = policy(obs, sg[0].unsqueeze(0))
        p = torch.softmax(l, -1).numpy()[0]
        srt = np.sort(p)[::-1]
        margins.append(float(srt[0] - srt[1]))
    print(f"  argmax margin (100 obs): mean={np.mean(margins):.4f} p90={np.percentile(margins,90):.4f}")


if __name__ == '__main__':
    read_tfevents()
    compare_logits()