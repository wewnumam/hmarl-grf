"""Probe4: trace every left-possession window on 11v11.

For each window: carrier role/pos, action sequence, nearest-opp dist,
pass target when passing, how window ended (steps held, ball end pos).
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

def min_opp(gs, idx):
    p = get_player_position(gs, 'left', idx)
    return min(get_distance(p, get_player_position(gs, 'right', j)) for j in range(11))

set_seed(42)
env = create_raw_env(env_name="11_vs_11_stochastic", num_agents=11, render=False)
controller = HierarchicalController()
expert = ExpertPolicy(d_tackle=0.05, d_safe=0.15, d_shoot=0.30)

r = env.reset()
obs = r[0] if isinstance(r, tuple) else r
gs = extract_game_state(obs)

done, step = False, 0
window = None  # dict while left has ball
goal_events = []
t0 = time.time()
prev_score = [0, 0]

while not done and step < 3000:
    macro = controller.get_macro_strategy(gs)
    sub_goals = controller.get_sub_goals(gs, macro)
    actions = [expert.get_ideal_action(gs, i, sub_goals[i], macro) for i in range(NUM_AGENTS)]

    score = gs.get('score', [0, 0])
    if score != prev_score:
        side = "US" if score[0] > prev_score[0] else "THEM"
        ball = get_ball_position(gs)
        print(f"step{step}: GOAL {side} {prev_score}->{score} ball@({ball[0]:+.2f},{ball[1]:+.2f})", flush=True)
        prev_score = list(score)

    team = gs.get('ball_owned_team', -1)
    pl = gs.get('ball_owned_player', -1)

    if team == 0 and pl >= 0:
        if window is None or window['carrier'] != pl:
            if window is not None:
                print(f"  [w{window['id']}] end step{step}: P{window['carrier']}({window['role']}) held={step-window['start']} ball@({get_ball_position(gs)[0]:+.2f},{get_ball_position(gs)[1]:+.2f}) acts={window['acts']}", flush=True)
            role = get_player_role(gs, 'left', pl)
            window = {'id': step, 'carrier': pl, 'role': ROLE_NAMES.get(role, role),
                      'start': step, 'acts': [], 'start_pos': None}
        if window['start_pos'] is None:
            window['start_pos'] = tuple(get_player_position(gs, 'left', pl))
        a = actions[pl]
        window['acts'].append(ACTION_NAMES[a])
        if len(window['acts']) <= 12:
            p = get_player_position(gs, 'left', pl)
            b = get_ball_position(gs)
            print(f"step{step}: P{pl}({window['role']}) act={ACTION_NAMES[a]:14s} pos=({p[0]:+.2f},{p[1]:+.2f}) ball=({b[0]:+.2f},{b[1]:+.2f}) min_opp={min_opp(gs, pl):.3f} subgoal={sub_goals[pl]} macro={macro}", flush=True)
    else:
        if window is not None:
            print(f"  [w{window['id']}] LOST step{step}: P{window['carrier']}({window['role']}) held={step-window['start']} ball@({get_ball_position(gs)[0]:+.2f},{get_ball_position(gs)[1]:+.2f}) acts={window['acts']}", flush=True)
            window = None

    res = env.step(actions)
    if len(res) == 5:
        obs, _, term, trunc, _ = res
        done = term or trunc
    else:
        obs, _, done, _ = res
    gs = extract_game_state(obs)
    step += 1

print(f"FINAL {gs.get('score')} steps={step} ({time.time()-t0:.0f}s)", flush=True)
print("PROBE4 DONE", flush=True)
