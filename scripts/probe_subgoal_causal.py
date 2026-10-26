"""P0 — does the sub-goal context causally steer the low-level policy?

Answers thesis rumusan-masalah aspect 1 ("does the sub-goal context bridge the
disconnection?") with a counterfactual intervention on a trained checkpoint.
No training, no new environment rollouts beyond the episodes needed to collect
game states.

Interventions, all applied to the SAME observation and the SAME legal-action
mask, changing only the 16-dim sub-goal embedding fed to the shared trunk:

  actual   : the mid-level sub-goals as produced by HierarchicalController
  cf_<g>   : every agent receives sub-goal g (all 5 sub-goals)
  zero     : zero embedding (no sub-goal context at all)
  roleperm : sub-goals assigned as if left-team roles were permuted

Reported per condition:
  jsd      : mean Jensen-Shannon divergence (log2, in [0,1]) of the action
             distribution vs the `actual` condition
  flip     : share of (step, agent) where argmax action changes
  rci_cat / rci_strict / rci_nomove : RCI recomputed from the counterfactual
             action sequence. RCI is a pure function of (actions, expert
             ideals), so the counterfactual sequence can be replayed offline
             against the SAME game states and the SAME ideal actions.

Usage (inside the container):
  docker run --rm -v "<repo>:/gfootball" -w /gfootball -e PYTHONPATH=/gfootball \
      gfootball:hmarl python3 -u scripts/probe_subgoal_causal.py \
      --ckpt checkpoints_3m/hmarl_model.pt --episodes 5
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from hmarl.env import (
    create_raw_env, NUM_AGENTS, extract_game_state, get_action_mask,
    get_player_role,
    ROLE_GK, ROLE_CB, ROLE_LB, ROLE_RB, ROLE_DM, ROLE_CM,
    ROLE_LM, ROLE_RM, ROLE_AM, ROLE_CF,
)
from hmarl.policy import (
    HierarchicalController, SUBGOALS, NUM_SUBGOALS,
)
from hmarl.expert import ExpertPolicyAllAgents, ExpertPolicyEval
from hmarl.rci import (
    compute_rci_category, compute_rci_strict, compute_rci_nomove,
    f_strict, f_category, f_category_nomove,
)
from hmarl.utils import (
    OBS_DIM, EPISODE_MAX_STEPS, extract_obs_vector, load_hmarl_checkpoint,
)

ROLE_NAME = {ROLE_GK: 'GK', ROLE_CB: 'CB', ROLE_LB: 'LB', ROLE_RB: 'RB',
             ROLE_DM: 'DM', ROLE_CM: 'CM', ROLE_LM: 'LM', ROLE_RM: 'RM',
             ROLE_AM: 'AM', ROLE_CF: 'CF'}


def jsd_log2(p, q, eps=1e-12):
    """Jensen-Shannon divergence in bits (log2), bounded to [0, 1]."""
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * (np.log2(p) - np.log2(m))))
    kl_qm = float(np.sum(q * (np.log2(q) - np.log2(m))))
    return max(0.0, 0.5 * kl_pm + 0.5 * kl_qm)


INVOLVE_RADIUS = 3.0


def involvement(game_state, radius=INVOLVE_RADIUS):
    """Which left-team agents actually take part in the ball at this step?

    Motivation: RCI_cat credits any category match, including
    movement-vs-movement, so a passively standing agent still earns credit.
    That is the confound flagged by the rci_nomove variant. Gating by ball
    involvement isolates the steps where role behaviour is observable.

    Returns (own, near, in_radius) — three boolean lists of NUM_AGENTS:
      own       : agent currently owns the ball (team 0)
      near      : agent nearest to the ball
      in_radius : agent within `radius` units of the ball
    """
    ball = game_state.get('ball', [0.0, 0.0, 0.0])
    bx, by = float(ball[0]), float(ball[1])
    pos = game_state.get('left_team', [])[:NUM_AGENTS]
    owned_team = int(round(float(game_state.get('ball_owned_team', -1))))
    owned_player = int(round(float(game_state.get('ball_owned_player', -1))))

    own = [False] * NUM_AGENTS
    if owned_team == 0 and 0 <= owned_player < NUM_AGENTS:
        own[owned_player] = True

    dist = [float(np.hypot(float(p[0]) - bx, float(p[1]) - by)) for p in pos]
    near = [False] * NUM_AGENTS
    if dist:
        near[int(np.argmin(dist))] = True
    in_radius = [(d <= radius) for d in dist] + [False] * (NUM_AGENTS - len(dist))
    return own, near, in_radius[:NUM_AGENTS]


def gated_rci(acts, ideals, gates, f_match):
    """RCI restricted to (agent, step) pairs where the gate is True."""
    num = den = 0.0
    for t, (a, ideal) in enumerate(zip(acts, ideals)):
        g = gates[t] if t < len(gates) else None
        for i in range(NUM_AGENTS):
            if g is not None and g[i]:
                num += f_match(int(a[i]), int(ideal[i]))
                den += 1.0
    return (num / den) if den else 0.0, den


def softmax_masked(logits, mask):
    z = np.array(logits, dtype=np.float64)
    if mask is not None:
        z = np.where(mask, z, -1e9)
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints_3m/hmarl_model.pt')
    ap.add_argument('--episodes', type=int, default=5)
    ap.add_argument('--out', default=None)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    policy, subgoal_embedding, meta = load_hmarl_checkpoint(args.ckpt, device=str(device))
    policy.eval()
    subgoal_embedding.eval()
    print('checkpoint: %s (global_step=%s)' % (args.ckpt, meta.get('global_step', '?')))

    conditions = ['cf_%d' % g for g in range(NUM_SUBGOALS)] + ['zero', 'roleperm']
    perm = np.random.RandomState(0).permutation(NUM_AGENTS)

    env = create_raw_env()
    controller = HierarchicalController()
    expert = ExpertPolicyAllAgents()
    expert.policy = ExpertPolicyEval()

    # per-condition accumulators
    acc = {c: {'jsd': [], 'flip': 0, 'flip_decided': 0, 'n': 0, 'ndec': 0,
               'pmax': [], 'gap': [], 'acts': [], 'ideals': []}
           for c in ['actual'] + conditions}
    acc['actual'].update({'own': [], 'near': [], 'rad': []})
    role_jsd = {c: {} for c in conditions}
    steps_total = 0
    t0 = time.time()

    for ep in range(args.episodes):
        reset_result = env.reset()
        obs_raw = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        game_state = extract_game_state(obs_raw)
        done = False
        step = 0

        while not done and step < EPISODE_MAX_STEPS:
            macro = controller.get_macro_strategy(game_state)
            sub_goals = controller.get_sub_goals(game_state, macro)
            ideal_actions = expert.get_ideal_actions(game_state, sub_goals, macro)
            mask = get_action_mask(game_state)

            # role-permuted game state -> mid-level sub-goals only
            gs_perm = dict(game_state)
            roles = list(game_state.get('left_team_roles', list(range(NUM_AGENTS))))[:NUM_AGENTS]
            if len(roles) == NUM_AGENTS:
                gs_perm['left_team_roles'] = [roles[perm[i]] for i in range(NUM_AGENTS)]
            sg_perm = controller.get_sub_goals(gs_perm, macro)

            # condition -> per-agent sub-goal index
            cond_sg = {'actual': sub_goals}
            for g in range(NUM_SUBGOALS):
                cond_sg['cf_%d' % g] = [g] * NUM_AGENTS
            cond_sg['roleperm'] = sg_perm

            # build one batched forward: (num_agents * num_forward_conds, ...)
            fwd_conds = ['actual'] + ['cf_%d' % g for g in range(NUM_SUBGOALS)] + ['roleperm']
            obs_batch, sg_batch, owner = [], [], []
            for ci, cname in enumerate(fwd_conds):
                for i in range(NUM_AGENTS):
                    obs_batch.append(extract_obs_vector(game_state, i, OBS_DIM))
                    sg_batch.append(cond_sg[cname][i])
                    owner.append((cname, i))
            with torch.no_grad():
                obs_t = torch.FloatTensor(np.array(obs_batch)).to(device)
                sg_t = subgoal_embedding(torch.LongTensor(sg_batch).to(device))
                logits, _ = policy(obs_t, sg_t)
                logits = logits.cpu().numpy()

            # zero-embedding control: same obs, zero sub-goal vector
            with torch.no_grad():
                obs_z = torch.FloatTensor(np.array(
                    [extract_obs_vector(game_state, i, OBS_DIM) for i in range(NUM_AGENTS)]
                )).to(device)
                sg_z = torch.zeros(NUM_AGENTS, sg_t.shape[-1]).to(device)
                logits_z, _ = policy(obs_z, sg_z)
                logits_z = logits_z.cpu().numpy()

            probs = {}
            for k, (cname, i) in enumerate(owner):
                probs.setdefault(cname, {})[i] = softmax_masked(logits[k], mask)
            probs['zero'] = {i: softmax_masked(logits_z[i], mask) for i in range(NUM_AGENTS)}

            for c in ['actual'] + conditions:
                acc[c]['jsd'].append(
                    float(np.mean([jsd_log2(probs['actual'][i], probs[c][i])
                                   for i in range(NUM_AGENTS)]))
                )
                for i in range(NUM_AGENTS):
                    pa = np.sort(probs['actual'][i])[::-1]
                    acc[c]['pmax'].append(float(pa[0]))
                    acc[c]['gap'].append(float(pa[0] - pa[1]))
                if c != 'actual':
                    for i in range(NUM_AGENTS):
                        pa = np.sort(probs['actual'][i])[::-1]
                        decided = (pa[0] - pa[1]) >= 0.05
                        if int(np.argmax(probs[c][i])) != int(np.argmax(probs['actual'][i])):
                            acc[c]['flip'] += 1
                            acc[c]['flip_decided'] += 1 if decided else 0
                        acc[c]['n'] += 1
                        acc[c]['ndec'] += 1 if decided else 0
                        role = ROLE_NAME.get(get_player_role(game_state, 'left', i), '?')
                        role_jsd[c].setdefault(role, []).append(jsd_log2(probs['actual'][i], probs[c][i]))

            for c in ['actual'] + conditions:
                acc[c]['acts'].append([int(np.argmax(probs[c][i])) for i in range(NUM_AGENTS)])
                acc[c]['ideals'].append([int(a) for a in ideal_actions])

            own_g, near_g, rad_g = involvement(game_state)
            acc['actual']['own'].append(own_g)
            acc['actual']['near'].append(near_g)
            acc['actual']['rad'].append(rad_g)

            joint = [int(np.argmax(probs['actual'][i])) for i in range(NUM_AGENTS)]
            step_result = env.step(joint)
            if len(step_result) == 5:
                obs_raw_new, _, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                obs_raw_new, _, done, _ = step_result
            obs_raw = obs_raw_new
            game_state = extract_game_state(obs_raw)
            step += 1
            steps_total += 1

        print('  episode %d: %d steps (%.0fs elapsed)' % (ep + 1, step, time.time() - t0))

    env.close()

    # RCI replay per condition
    summary = {'ckpt': args.ckpt, 'episodes': args.episodes, 'steps': steps_total,
               'num_subgoals': NUM_SUBGOALS, 'conditions': {}}

    gates = {'own': acc['actual']['own'], 'near': acc['actual']['near'],
             'rad': acc['actual']['rad']}
    summary['involvement'] = {
        'radius': INVOLVE_RADIUS,
        'owner_share': float(np.mean([np.mean(g) for g in gates['own']])) if gates['own'] else 0.0,
        'in_radius_share': float(np.mean([np.mean(g) for g in gates['rad']])) if gates['rad'] else 0.0,
        'steps_with_any_owner': float(np.mean([any(g) for g in gates['own']])) if gates['own'] else 0.0,
    }

    for c in ['actual'] + conditions:
        acts, ideals = acc[c]['acts'], acc[c]['ideals']
        rci_cat, _ = compute_rci_category(acts, ideals)
        rci_strict, _ = compute_rci_strict(acts, ideals)
        rci_nomove, _ = compute_rci_nomove(acts, ideals)
        gated = {}
        for gname, gate in gates.items():
            for fname, fn in (('cat', f_category), ('strict', f_strict),
                              ('nomove', f_category_nomove)):
                val, den = gated_rci(acts, ideals, gate, fn)
                gated['%s_%s' % (gname, fname)] = val
                gated['%s_%s_n' % (gname, fname)] = den
        summary['conditions'][c] = {
            'jsd_mean': float(np.mean(acc[c]['jsd'])) if acc[c]['jsd'] else 0.0,
            'jsd_p95': float(np.percentile(acc[c]['jsd'], 95)) if acc[c]['jsd'] else 0.0,
            'flip_rate': (acc[c]['flip'] / float(acc[c]['n'])) if c != 'actual' and acc[c]['n'] else 0.0,
            'flip_rate_decided': (acc[c]['flip_decided'] / float(acc[c]['ndec']))
                                 if c != 'actual' and acc[c]['ndec'] else 0.0,
            'decided_share': (acc[c]['ndec'] / float(acc[c]['n'])) if acc[c]['n'] else 0.0,
            'pmax_mean': float(np.mean(acc[c]['pmax'])) if acc[c]['pmax'] else 0.0,
            'top2_gap_mean': float(np.mean(acc[c]['gap'])) if acc[c]['gap'] else 0.0,
            'rci_cat': rci_cat, 'rci_strict': rci_strict, 'rci_nomove': rci_nomove,
            'gated': gated,
        }
        if c in role_jsd:
            summary['conditions'][c]['jsd_by_role'] = {
                r: float(np.mean(v)) for r, v in sorted(role_jsd[c].items())
            }

    print('\n%-9s %8s %7s %8s %7s %7s %8s %9s %9s %10s' % (
        'condition', 'JSD', 'flip', 'flip_dec', 'p_max', 'gap', 'decided',
        'RCI_cat', 'RCI_str', 'RCI_nomove'))
    for c in ['actual'] + conditions:
        s = summary['conditions'][c]
        print('%-9s %8.5f %6.1f%% %7.1f%% %7.3f %7.3f %7.1f%% %9.4f %9.4f %10.4f' % (
            c, s['jsd_mean'], 100.0 * s['flip_rate'], 100.0 * s['flip_rate_decided'],
            s['pmax_mean'], s['top2_gap_mean'], 100.0 * s['decided_share'],
            s['rci_cat'], s['rci_strict'], s['rci_nomove']))

    print('\nJSD by role (vs actual):')
    for c in conditions:
        br = summary['conditions'][c].get('jsd_by_role', {})
        if br:
            print('  %-9s %s' % (c, '  '.join('%s %.5f' % (r, v) for r, v in br.items())))

    inv = summary['involvement']
    print('\ninvolvement (radius %.1f): owner_share %.3f | in_radius_share %.3f | steps_with_owner %.1f%%'
          % (inv['radius'], inv['owner_share'], inv['in_radius_share'],
             100.0 * inv['steps_with_any_owner']))
    for gname, label in (('near', 'nearest-to-ball'), ('rad', 'in-radius'), ('own', 'ball-owner')):
        print('\nRCI gated by %s (share of ungated -> gated):' % label)
        print('%-9s %10s %10s %10s' % ('condition', 'cat', 'strict', 'nomove'))
        for c in ['actual'] + conditions:
            g = summary['conditions'][c]['gated']
            print('%-9s %10.4f %10.4f %10.4f' % (
                c, g['%s_cat' % gname], g['%s_strict' % gname], g['%s_nomove' % gname]))

    print('\nreference: pre-SUBGOAL_SCALE=3.0 measurement was JSD 4e-5..4e-4, '
          'flip 2.5-4.3%% (obs103 checkpoint)')

    out = args.out or os.path.join('_probe', 'cf_subgoal_%s.json' % os.path.basename(
        os.path.dirname(args.ckpt) or 'ckpt'))
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w') as fh:
        json.dump(summary, fh, indent=1)
    print('\nwrote %s' % out)

    # Gate masks for offline recomputation of any future RCI variant (no re-run needed)
    npz = os.path.join(os.path.dirname(out) or '.', 'involve_%s.npz' % os.path.basename(
        os.path.dirname(args.ckpt) or 'ckpt'))
    np.savez_compressed(
        npz,
        own=np.packbits(np.array(acc['actual']['own'], dtype=np.uint8).ravel()),
        near=np.packbits(np.array(acc['actual']['near'], dtype=np.uint8).ravel()),
        rad=np.packbits(np.array(acc['actual']['rad'], dtype=np.uint8).ravel()),
        shape=np.array(acc['actual']['own']).shape,
    )
    print('wrote %s' % npz)


if __name__ == '__main__':
    main()
