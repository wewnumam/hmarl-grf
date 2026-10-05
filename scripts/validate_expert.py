"""Multi-episode validation of current expert (10 eps).

Runs 10 episodes of 11_vs_11_stochastic with current ExpertPolicy,
prints per-episode score + aggregate WR/GF/GA. ~35s/ep on CPU.
"""
import sys, time
sys.path.insert(0, '/gfootball')
from hmarl.env import create_raw_env, extract_game_state
from hmarl.policy import HierarchicalController
from hmarl.expert import ExpertPolicy
from hmarl.utils import set_seed

N_EPS = 10
MAX_STEPS = 3000
set_seed(42)
env = create_raw_env(env_name="11_vs_11_stochastic", num_agents=11, render=False)
controller = HierarchicalController()
expert = ExpertPolicy(d_tackle=0.05, d_safe=0.15, d_shoot=0.30)

wins = draws = losses = 0
gf_total = ga_total = 0
t0 = time.time()
for ep in range(N_EPS):
    r = env.reset()
    obs = r[0] if isinstance(r, tuple) else r
    gs = extract_game_state(obs)
    done, step = False, 0
    ep_gf = ep_ga = 0
    while not done and step < MAX_STEPS:
        macro = controller.get_macro_strategy(gs)
        sub_goals = controller.get_sub_goals(gs, macro)
        actions = [expert.get_ideal_action(gs, i, sub_goals[i], macro) for i in range(11)]
        res = env.step(actions)
        if len(res) == 5:
            obs, _, term, trunc, _ = res
            done = term or trunc
        else:
            obs, _, done, _ = res
        gs = extract_game_state(obs)
        s = gs.get('score', [0, 0])
        ep_gf, ep_ga = max(ep_gf, s[0]), max(ep_ga, s[1])
        step += 1
    gf_total += ep_gf
    ga_total += ep_ga
    if ep_gf > ep_ga:
        wins += 1
        r_ = 'W'
    elif ep_gf == ep_ga:
        draws += 1
        r_ = 'D'
    else:
        losses += 1
        r_ = 'L'
    print(f"ep{ep:2d}: {r_} {ep_gf}-{ep_ga} steps={step} t={time.time()-t0:.0f}s", flush=True)

print(f"\nAGGREGATE: W/D/L={wins}/{draws}/{losses} WR={100*wins/N_EPS:.0f}% "
      f"GF/ep={gf_total/N_EPS:.1f} GA/ep={ga_total/N_EPS:.1f} ({time.time()-t0:.0f}s)", flush=True)
print("VALIDATION DONE", flush=True)
