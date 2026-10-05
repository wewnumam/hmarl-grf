"""Diagnostic probe5: full team shape at goal moments.

On each goal conceded, dump: ball pos, all 11 left player positions+roles,
nearest left player to ball, opponent ball carrier pos. Also track
positions 10 steps before the goal to see defensive breakdown.
"""
import sys, time
sys.path.insert(0, '/gfootball')
from hmarl.env import (
    create_raw_env, extract_game_state, NUM_AGENTS,
    get_ball_position, get_player_position, get_distance,
    get_player_role, ACTION_NAMES, ROLE_NAMES,
)
from hmarl.policy import HierarchicalController
from hmarl.expert import ExpertPolicy
from hmarl.utils import set_seed

def fmt_team(gs):
    b = get_ball_position(gs)
    parts = []
    nearest = (None, 1e9)
    for i in range(NUM_AGENTS):
        p = get_player_position(gs, 'left', i)
        r = ROLE_NAMES.get(get_player_role(gs, 'left', i), '?')
        d = get_distance(p, b)
        if d < nearest[1]:
            nearest = (i, d, p, r)
        parts.append(f"P{i}:{r}({p[0]:+.2f},{p[1]:+.2f})d={d:.2f}")
    return b, parts, nearest

set_seed(42)
env = create_raw_env(env_name="11_vs_11_stochastic", num_agents=11, render=False)
controller = HierarchicalController()
expert = ExpertPolicy(d_tackle=0.05, d_safe=0.15, d_shoot=0.30)

r = env.reset()
obs = r[0] if isinstance(r, tuple) else r
gs = extract_game_state(obs)

ring = []  # last 10 states
done, step = False, 0
prev_score = [0, 0]
t0 = time.time()
while not done and step < 3000:
    macro = controller.get_macro_strategy(gs)
    sub_goals = controller.get_sub_goals(gs, macro)
    actions = [expert.get_ideal_action(gs, i, sub_goals[i], macro) for i in range(NUM_AGENTS)]

    score = gs.get('score', [0, 0])
    if score != prev_score:
        side = "US" if score[0] > prev_score[0] else "THEM"
        print(f"\n##### GOAL {side} {prev_score}->{score} step{step} #####", flush=True)
        b, parts, nearest = fmt_team(gs)
        print(f"  ball=({b[0]:+.2f},{b[1]:+.2f}) nearest_left=P{nearest[0]}({nearest[3]}) d={nearest[1]:.2f}", flush=True)
        # opponent carrier
        ot = gs.get('ball_owned_team', -1)
        op = gs.get('ball_owned_player', -1)
        if ot == 1 and op >= 0:
            op_p = get_player_position(gs, 'right', op)
            print(f"  opp_carrier=P{op} pos=({op_p[0]:+.2f},{op_p[1]:+.2f})", flush=True)
        print("  left team: " + " ".join(parts), flush=True)
        # ring: 10 steps before
        print("  --- 10 steps before ---", flush=True)
        for h in ring[-10:]:
            hb, hp, hn = fmt_team(h)
            print(f"    ball=({hb[0]:+.2f},{hb[1]:+.2f}) nearest=P{hn[0]}({hn[3]}) d={hn[1]:.2f} macro={h.get('_macro')}", flush=True)
        prev_score = list(score)

    ring.append({**gs, '_macro': macro})
    if len(ring) > 12:
        ring.pop(0)

    res = env.step(actions)
    if len(res) == 5:
        obs, _, term, trunc, _ = res
        done = term or trunc
    else:
        obs, _, done, _ = res
    gs = extract_game_state(obs)
    step += 1

print(f"\nFINAL {gs.get('score')} steps={step} ({time.time()-t0:.0f}s)", flush=True)
print("PROBE5 DONE", flush=True)
