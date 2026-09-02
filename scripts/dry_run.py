"""Dry Run / Smoke Test for HMARL-GRF pipeline.

Usage (inside Docker):
    python scripts/dry_run.py              # quick: AST + components + mock pipeline
    python scripts/dry_run.py --full       # full: includes actual GRF env (1 episode, 50 steps)
    python scripts/dry_run.py --ast-only   # host-side: syntax check only, no imports

What it tests:
    Layer 1 (ast-only): parse all .py files for syntax errors
    Layer 2 (default):  import all modules, unit-test RL components
    Layer 3 (--full):   actual GRF env, 1 episode, 50 steps, PPO backward pass
"""

import ast
import os
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASSED = 0
FAILED = 0

def check(name, fn):
    global PASSED, FAILED
    try:
        fn()
        PASSED += 1
        print(f"  PASS  {name}")
    except Exception as e:
        FAILED += 1
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Layer 1: AST syntax check (no imports needed, runs on host)
# ---------------------------------------------------------------------------
def layer1_ast_check():
    print("\n=== Layer 1: AST Syntax Check ===")
    root = Path(__file__).resolve().parent.parent
    py_files = sorted(root.rglob("*.py"))
    ast_failed = 0
    for fpath in py_files:
        # skip hidden dirs
        if any(p.startswith('.') for p in fpath.relative_to(root).parts):
            continue
        try:
            ast.parse(fpath.read_text(encoding="utf-8"), filename=str(fpath))
        except SyntaxError as e:
            print(f"  SYNTAX ERROR in {fpath.relative_to(root)}: {e}")
            ast_failed += 1
    if ast_failed == 0:
        print(f"  All {len(py_files)} .py files parse OK")
    else:
        print(f"  {ast_failed}/{len(py_files)} files have syntax errors")
    return ast_failed == 0


# ---------------------------------------------------------------------------
# Layer 2: Component unit tests (needs torch, numpy, gfootball)
# ---------------------------------------------------------------------------
def layer2_components():
    print("\n=== Layer 2: Component Unit Tests ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    # -- Imports --
    def test_imports():
        import hmarl.env
        import hmarl.policy
        import hmarl.expert
        import hmarl.reward
        import hmarl.rci
        import hmarl.metrics
        import hmarl.utils
    check("import all modules", test_imports)

    # -- Seed --
    def test_seed():
        from hmarl.utils import set_seed
        set_seed(42)
    check("set_seed", test_seed)

    # -- Policy forward pass --
    def test_policy_forward():
        import torch
        from hmarl.policy import HierarchicalActorCritic, SubGoalEmbedding, SUBGOAL_EMBED_DIM
        from hmarl.utils import OBS_DIM, ACTION_SPACE_SIZE
        device = torch.device("cpu")
        policy = HierarchicalActorCritic(
            obs_dim=OBS_DIM, subgoal_embed_dim=SUBGOAL_EMBED_DIM,
            hidden_dim=256, head_dim=128, action_dim=ACTION_SPACE_SIZE,
        ).to(device)
        sg_emb = SubGoalEmbedding().to(device)
        obs = torch.randn(4, OBS_DIM)
        sg = torch.randint(0, 5, (4,))
        sg_embed = sg_emb(sg)
        logits, value = policy(obs, sg_embed)
        assert logits.shape == (4, 19), f"bad logits shape: {logits.shape}"
        assert value.shape == (4, 1), f"bad value shape: {value.shape}"
        # get_action_and_value
        action, log_prob, entropy, val = policy.get_action_and_value(obs, sg_embed)
        assert action.shape == (4,)
        assert log_prob.shape == (4,)
    check("policy forward pass", test_policy_forward)

    # -- Policy backward pass --
    def test_policy_backward():
        import torch
        import torch.nn as nn
        from hmarl.policy import HierarchicalActorCritic, SubGoalEmbedding
        from hmarl.utils import OBS_DIM, ACTION_SPACE_SIZE
        policy = HierarchicalActorCritic(obs_dim=OBS_DIM, action_dim=ACTION_SPACE_SIZE).to("cpu")
        sg_emb = SubGoalEmbedding().to("cpu")
        obs = torch.randn(8, OBS_DIM)
        sg = sg_emb(torch.randint(0, 5, (8,)))
        action, log_prob, entropy, value = policy.get_action_and_value(obs, sg, torch.randint(0, 19, (8,)))
        loss = -log_prob.mean() + 0.5 * nn.functional.mse_loss(value, torch.randn(8))
        loss.backward()
        # check gradients exist
        for p in policy.parameters():
            if p.grad is not None:
                return
        raise RuntimeError("no gradients computed")
    check("policy backward pass", test_policy_backward)

    # -- Obs vector extraction --
    def test_obs_vector():
        import numpy as np
        from hmarl.utils import extract_obs_vector, OBS_DIM
        gs = _make_mock_game_state()
        vec = extract_obs_vector(gs, 0, OBS_DIM)
        assert vec.shape == (OBS_DIM,), f"obs dim mismatch: {vec.shape}"
        assert not np.any(np.isnan(vec)), "NaN in obs vector"
    check("extract_obs_vector", test_obs_vector)

    # -- Game state extraction --
    def test_game_state():
        from hmarl.env import extract_game_state
        gs = _make_mock_game_state()
        gs2 = extract_game_state(gs)
        assert 'ball' in gs2
        assert 'left_team' in gs2
        assert len(gs2['left_team']) >= 11
    check("extract_game_state", test_game_state)

    # -- High-Level policy --
    def test_high_level():
        from hmarl.policy import HierarchicalController, STRATEGY_HIGH_PRESSING, STRATEGY_POSSESSION, STRATEGY_COUNTER_ATTACK
        ctrl = HierarchicalController()
        gs = _make_mock_game_state()
        # ball not owned -> high pressing
        gs['ball_owned_team'] = 1
        s = ctrl.get_macro_strategy(gs)
        assert s == STRATEGY_HIGH_PRESSING
        # ball owned, back half -> possession
        gs['ball_owned_team'] = 0
        gs['ball'] = [-0.5, 0.0, 0.0]
        s = ctrl.get_macro_strategy(gs)
        assert s == STRATEGY_POSSESSION
        # ball owned, front half -> counter
        gs['ball'] = [0.5, 0.0, 0.0]
        s = ctrl.get_macro_strategy(gs)
        assert s == STRATEGY_COUNTER_ATTACK
    check("high-level policy", test_high_level)

    # -- Mid-Level policy --
    def test_mid_level():
        from hmarl.policy import HierarchicalController
        ctrl = HierarchicalController()
        gs = _make_mock_game_state()
        gs['ball_owned_team'] = 0
        gs['ball'] = [0.5, 0.0, 0.0]
        macro = ctrl.get_macro_strategy(gs)
        sub_goals = ctrl.get_sub_goals(gs, macro)
        assert len(sub_goals) == 11
        for sg in sub_goals:
            assert 0 <= sg <= 4, f"bad sub_goal: {sg}"
    check("mid-level policy", test_mid_level)

    # -- Expert policy --
    def test_expert():
        from hmarl.expert import ExpertPolicyAllAgents
        from hmarl.policy import HierarchicalController
        ctrl = HierarchicalController()
        expert = ExpertPolicyAllAgents()
        gs = _make_mock_game_state()
        gs['ball_owned_team'] = 0
        gs['ball'] = [0.5, 0.0, 0.0]
        gs['ball_owned_player'] = 5
        macro = ctrl.get_macro_strategy(gs)
        sub_goals = ctrl.get_sub_goals(gs, macro)
        ideals = expert.get_ideal_actions(gs, sub_goals, macro)
        assert len(ideals) == 11
        for a in ideals:
            assert 0 <= a <= 18, f"bad ideal action: {a}"
    check("expert policy", test_expert)

    # -- Reward computation --
    def test_reward():
        from hmarl.reward import PassTracker, RCITracker, compute_hierarchical_reward
        gs = _make_mock_game_state()
        pt = PassTracker()
        rt = RCITracker(11)
        total, breakdown = compute_hierarchical_reward(
            game_reward=1.0, game_state=gs,
            actual_actions=[5]*11, ideal_actions=[5]*11,
            pass_tracker=pt, rci_tracker=rt,
        )
        assert isinstance(total, float)
        assert 'fai' in breakdown
        assert 'ppr' in breakdown
        assert 'rci_avg' in breakdown
    check("reward computation", test_reward)

    # -- FAI --
    def test_fai():
        from hmarl.reward import compute_fai
        gs = _make_mock_game_state()
        gs['ball_owned_team'] = 0
        gs['ball'] = [0.0, 0.0, 0.0]
        fai = compute_fai(gs, 11)
        assert 0.0 <= fai <= 1.0, f"FAI out of range: {fai}"
    check("FAI computation", test_fai)

    # -- RCI --
    def test_rci():
        from hmarl.rci import compute_rci
        actual = [[5, 11, 3, 5, 9, 11, 12, 0, 1, 2, 5] for _ in range(20)]
        ideal = [[5, 11, 3, 5, 9, 11, 12, 0, 1, 2, 5] for _ in range(20)]
        res = compute_rci(actual, ideal)
        assert res['rci_strict'] == 1.0, f"RCI strict should be 1.0, got {res['rci_strict']}"
        assert res['rci_cat'] == 1.0
    check("RCI computation", test_rci)

    # -- Metrics --
    def test_metrics():
        from hmarl.metrics import compute_win_rate, compute_goal_difference
        wr = compute_win_rate(['win', 'win', 'loss', 'draw'])
        assert abs(wr - 50.0) < 0.01, f"WR wrong: {wr}"
        gd = compute_goal_difference([3, 1, 2], [1, 2, 0])
        assert gd == 3, f"GD wrong: {gd}"
    check("metrics", test_metrics)

    # -- Team compactness --
    def test_compactness():
        from hmarl.metrics import team_compactness
        gs_list = [_make_mock_game_state() for _ in range(10)]
        mean, std, mn, mx = team_compactness(gs_list)
        assert mean >= 0
    check("team_compactness", test_compactness)

    # -- FAI metric --
    def test_fai_metric():
        from hmarl.metrics import formation_adherence_index
        gs_list = [_make_mock_game_state() for _ in range(5)]
        mean, std = formation_adherence_index(gs_list)
        assert 0.0 <= mean <= 1.0
    check("formation_adherence_index (metric)", test_fai_metric)

    # -- Rollout buffer --
    def test_buffer():
        import numpy as np
        # inline test since RolloutBuffer is in train.py
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from train import RolloutBuffer
        buf = RolloutBuffer(capacity=100)
        # Add initial transitions
        for _ in range(32):
            buf.add_transition(
                obs=np.random.randn(115).astype(np.float32),
                subgoal_embed=np.random.randn(16).astype(np.float32),
                action=np.random.randint(0, 19),
                log_prob=-1.0,
                value=0.5,
            )
        # Simulate one timestep: start_timestep + add 11 agents + record + fill reward
        buf.start_timestep()
        for _ in range(11):
            buf.add_transition(
                obs=np.random.randn(115).astype(np.float32),
                subgoal_embed=np.random.randn(16).astype(np.float32),
                action=np.random.randint(0, 19),
                log_prob=-1.0,
                value=0.5,
            )
            buf.record_agent_index()
        buf.fill_timestep_reward(0.5)
        # The 11 new transitions are indices 32-42
        assert buf.rewards[32] == 0.5, f"reward[32]={buf.rewards[32]}"
        assert buf.rewards[42] == 0.5, f"reward[42]={buf.rewards[42]}"
        assert buf.rewards[0] == 0.0, "old reward should be untouched"
        # GAE
        buf.dones[-1] = True
        buf.compute_gae(last_value=0.0)
        assert len(buf.advantages) == 43
        assert len(buf.returns) == 43
        # batches
        count = 0
        for batch in buf.get_batches(8):
            assert 'obs' in batch
            count += 1
        assert count > 0
    check("rollout buffer", test_buffer)

    # -- Action category --
    def test_action_category():
        from hmarl.env import get_action_category, ACTION_CATEGORIES
        assert get_action_category(11) == 'passing'   # short_pass
        assert get_action_category(12) == 'shooting'   # shot
        assert get_action_category(3) == 'movement'     # top
        assert get_action_category(17) == 'ball_control'  # dribble
        assert get_action_category(0) == 'defensive'    # idle
    check("action category", test_action_category)


# ---------------------------------------------------------------------------
# Layer 2b: Mock pipeline (full training loop, mock env, no GRF needed)
# ---------------------------------------------------------------------------
def layer2b_mock_pipeline():
    print("\n=== Layer 2b: Mock Pipeline (1 episode, 50 steps) ===")
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

    from hmarl.utils import set_seed, extract_obs_vector, OBS_DIM
    from hmarl.policy import (
        HierarchicalActorCritic, HierarchicalController,
        SubGoalEmbedding, SUBGOAL_EMBED_DIM,
    )
    from hmarl.expert import ExpertPolicyAllAgents
    from hmarl.reward import PassTracker, RCITracker, compute_hierarchical_reward
    from train import RolloutBuffer

    set_seed(42)
    NUM_STEPS = 50
    NUM_AGENTS = 11

    # mock game state generator
    def mock_gs(step):
        gs = _make_mock_game_state()
        # simulate ball moving forward
        gs['ball'] = [-0.9 + step * 0.04, np.sin(step * 0.1) * 0.2, 0.0]
        gs['ball_owned_team'] = 0 if step % 5 != 0 else 1
        gs['ball_owned_player'] = step % 11 if gs['ball_owned_team'] == 0 else -1
        # randomize player positions slightly
        for i in range(11):
            gs['left_team'][i] = [
                gs['left_team'][i][0] + np.random.uniform(-0.02, 0.02),
                gs['left_team'][i][1] + np.random.uniform(-0.01, 0.01),
            ]
        return gs

    # init components
    device = torch.device("cpu")
    controller = HierarchicalController()
    expert = ExpertPolicyAllAgents()
    subgoal_embedding = SubGoalEmbedding().to(device)
    policy = HierarchicalActorCritic(
        obs_dim=OBS_DIM, subgoal_embed_dim=SUBGOAL_EMBED_DIM,
        hidden_dim=256, head_dim=128, action_dim=19,
    ).to(device)
    optimizer = optim.Adam(
        list(policy.parameters()) + list(subgoal_embedding.parameters()),
        lr=3e-4, eps=1e-5,
    )
    buffer = RolloutBuffer(capacity=NUM_STEPS)
    pass_tracker = PassTracker()
    rci_tracker = RCITracker(NUM_AGENTS)

    # run mock episode
    game_state = mock_gs(0)
    episode_reward = 0.0
    done = False
    step = 0

    t0 = time.time()
    while not done and step < NUM_STEPS:
        macro = controller.get_macro_strategy(game_state)
        sub_goals = controller.get_sub_goals(game_state, macro)
        ideal_actions = expert.get_ideal_actions(game_state, sub_goals, macro)

        buffer.start_timestep()
        joint_actions = []
        for i in range(NUM_AGENTS):
            obs_vec = extract_obs_vector(game_state, i, OBS_DIM)
            with torch.no_grad():
                sg_embed = subgoal_embedding(
                    torch.LongTensor([sub_goals[i]]).to(device)
                ).cpu().numpy().flatten()
            obs_t = torch.FloatTensor(obs_vec).unsqueeze(0).to(device)
            sg_t = torch.FloatTensor(sg_embed).unsqueeze(0).to(device)
            action, log_prob, _, value = policy.get_action_and_value(obs_t, sg_t)
            a = action.item()
            joint_actions.append(a)
            buffer.add_transition(obs_vec, sg_embed, a, log_prob.item(), value.item())
            buffer.record_agent_index()

        # mock env step
        reward = float(np.random.uniform(-0.1, 0.3))
        new_game_state = mock_gs(step + 1)
        pass_tracker.update(new_game_state, game_state)

        total_r, _ = compute_hierarchical_reward(
            reward, new_game_state, joint_actions, ideal_actions,
            pass_tracker, rci_tracker,
        )
        buffer.fill_timestep_reward(total_r / NUM_AGENTS)
        episode_reward += total_r

        if step >= NUM_STEPS - 1:
            for idx in buffer.agent_buffer_indices:
                if idx < len(buffer.dones):
                    buffer.dones[idx] = True

        prev_gs = game_state
        game_state = new_game_state
        step += 1
        if step % 10 == 0:
            print(f"    step {step}/{NUM_STEPS} reward={episode_reward:.3f}")

    elapsed = time.time() - t0

    # PPO update
    def test_ppo_update():
        if len(buffer.observations) == 0:
            raise RuntimeError("empty buffer")
        with torch.no_grad():
            last_obs = torch.FloatTensor(buffer.observations[-1]).unsqueeze(0).to(device)
            last_sg = torch.FloatTensor(buffer.subgoal_embeds[-1]).unsqueeze(0).to(device)
            last_value = policy.get_value(last_obs, last_sg).item()
        buffer.compute_gae(last_value)

        policy.train()
        subgoal_embedding.train()
        for batch in buffer.get_batches(8):
            obs = batch['obs'].to(device)
            sg = batch['subgoal_embed'].to(device)
            acts = batch['actions'].to(device)
            old_lp = batch['log_probs_old'].to(device)
            adv = batch['advantages'].to(device)
            ret = batch['returns'].to(device)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            _, new_lp, entropy, new_val = policy.get_action_and_value(obs, sg, acts)
            ratio = torch.exp(new_lp - old_lp)
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 0.8, 1.2) * adv
            pg_loss = -torch.min(s1, s2).mean()
            v_loss = nn.functional.mse_loss(new_val, ret)
            loss = pg_loss + 0.5 * v_loss - 0.01 * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

    check("PPO backward update (mock pipeline)", test_ppo_update)
    print(f"  Mock pipeline: {NUM_STEPS} steps in {elapsed:.2f}s ({NUM_STEPS/elapsed:.0f} steps/s)")


# ---------------------------------------------------------------------------
# Layer 3: Full GRF integration (--full flag, needs Docker env)
# ---------------------------------------------------------------------------
def layer3_full_integration():
    print("\n=== Layer 3: GRF Integration (1 episode, 50 steps) ===")
    import numpy as np
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

    from hmarl.env import create_raw_env, NUM_AGENTS, extract_game_state
    from hmarl.utils import set_seed, extract_obs_vector, OBS_DIM
    from hmarl.policy import (
        HierarchicalActorCritic, HierarchicalController,
        SubGoalEmbedding, SUBGOAL_EMBED_DIM,
    )
    from hmarl.expert import ExpertPolicyAllAgents
    from hmarl.reward import PassTracker, RCITracker, compute_hierarchical_reward
    from hmarl.metrics import compute_win_rate

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        env = create_raw_env(render=False, write_dumps=False)
    except Exception as e:
        print(f"  SKIP: cannot create GRF env: {e}")
        print("  (This is expected if not running inside Docker)")
        return

    controller = HierarchicalController()
    expert = ExpertPolicyAllAgents()
    subgoal_embedding = SubGoalEmbedding().to(device)
    policy = HierarchicalActorCritic(
        obs_dim=OBS_DIM, subgoal_embed_dim=SUBGOAL_EMBED_DIM,
        hidden_dim=256, head_dim=128, action_dim=19,
    ).to(device)

    MAX_STEPS = 50

    def test_grf_reset():
        reset_result = env.reset()
        obs_raw = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        gs = extract_game_state(obs_raw)
        assert 'ball' in gs, "no ball in game state"
        assert 'left_team' in gs, "no left_team in game state"
    check("env.reset + extract_game_state", test_grf_reset)

    def test_grf_step():
        reset_result = env.reset()
        obs_raw = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        gs = extract_game_state(obs_raw)
        pass_tracker = PassTracker()
        rci_tracker = RCITracker(NUM_AGENTS)

        steps_done = 0
        prev_gs = None
        total_reward = 0.0
        t0 = time.time()

        while steps_done < MAX_STEPS:
            macro = controller.get_macro_strategy(gs)
            sub_goals = controller.get_sub_goals(gs, macro)
            ideals = expert.get_ideal_actions(gs, sub_goals, macro)

            joint_actions = []
            for i in range(NUM_AGENTS):
                obs_vec = extract_obs_vector(gs, i, OBS_DIM)
                with torch.no_grad():
                    sg_embed = subgoal_embedding(
                        torch.LongTensor([sub_goals[i]]).to(device)
                    ).cpu().numpy().flatten()
                    obs_t = torch.FloatTensor(obs_vec).unsqueeze(0).to(device)
                    sg_t = torch.FloatTensor(sg_embed).unsqueeze(0).to(device)
                    logits, _ = policy(obs_t, sg_t)
                    joint_actions.append(logits.argmax(dim=-1).item())

            step_result = env.step(joint_actions)
            if len(step_result) == 5:
                obs_raw, game_reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs_raw, game_reward, done, info = step_result

            if isinstance(obs_raw, np.ndarray) and obs_raw.ndim == 2:
                from eval import _extract_dict_from_simple
                new_gs = _extract_dict_from_simple(obs_raw)
            else:
                new_gs = extract_game_state(obs_raw)

            pass_tracker.update(new_gs, prev_gs)
            tr, _ = compute_hierarchical_reward(
                float(np.sum(game_reward)), new_gs, joint_actions, ideals,
                pass_tracker, rci_tracker,
            )
            total_reward += tr
            prev_gs = gs
            gs = new_gs
            steps_done += 1
            if done:
                break

        elapsed = time.time() - t0
        print(f"    {steps_done} GRF steps in {elapsed:.2f}s ({steps_done/elapsed:.0f} steps/s)")
        assert steps_done > 0, "no steps completed"

    check("env.step full loop (50 steps)", test_grf_step)

    def test_grf_backward():
        """One PPO update on real GRF rollout data."""
        import torch.nn as nn
        import torch.optim as optim
        from train import RolloutBuffer

        reset_result = env.reset()
        obs_raw = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        gs = extract_game_state(obs_raw)
        buffer = RolloutBuffer(capacity=50)
        pass_tracker = PassTracker()
        rci_tracker = RCITracker(NUM_AGENTS)

        policy.train()
        subgoal_embedding.train()
        optimizer = optim.Adam(
            list(policy.parameters()) + list(subgoal_embedding.parameters()),
            lr=3e-4, eps=1e-5,
        )

        prev_gs = None
        for step in range(10):
            macro = controller.get_macro_strategy(gs)
            sub_goals = controller.get_sub_goals(gs, macro)
            ideals = expert.get_ideal_actions(gs, sub_goals, macro)
            buffer.start_timestep()
            joint_actions = []
            for i in range(NUM_AGENTS):
                obs_vec = extract_obs_vector(gs, i, OBS_DIM)
                with torch.no_grad():
                    sg_embed = subgoal_embedding(
                        torch.LongTensor([sub_goals[i]]).to(device)
                    ).cpu().numpy().flatten()
                obs_t = torch.FloatTensor(obs_vec).unsqueeze(0).to(device)
                sg_t = torch.FloatTensor(sg_embed).unsqueeze(0).to(device)
                a, lp, _, v = policy.get_action_and_value(obs_t, sg_t)
                joint_actions.append(a.item())
                buffer.add_transition(obs_vec, sg_embed, a.item(), lp.item(), v.item())
                buffer.record_agent_index()

            step_result = env.step(joint_actions)
            if len(step_result) == 5:
                obs_raw, game_reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs_raw, game_reward, done, info = step_result

            new_gs = extract_game_state(obs_raw)
            pass_tracker.update(new_gs, prev_gs)
            tr, _ = compute_hierarchical_reward(
                float(np.sum(game_reward)), new_gs, joint_actions, ideals,
                pass_tracker, rci_tracker,
            )
            buffer.fill_timestep_reward(tr / NUM_AGENTS)
            if done:
                for idx in buffer.agent_buffer_indices:
                    if idx < len(buffer.dones):
                        buffer.dones[idx] = True
            prev_gs = gs
            gs = new_gs

        with torch.no_grad():
            last_obs = torch.FloatTensor(buffer.observations[-1]).unsqueeze(0).to(device)
            last_sg = torch.FloatTensor(buffer.subgoal_embeds[-1]).unsqueeze(0).to(device)
            last_val = policy.get_value(last_obs, last_sg).item()
        buffer.compute_gae(last_val)
        for batch in buffer.get_batches(8):
            _, new_lp, entropy, new_val = policy.get_action_and_value(
                batch['obs'], batch['subgoal_embed'], batch['actions'],
            )
            ratio = torch.exp(new_lp - batch['log_probs_old'])
            adv = batch['advantages']
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            loss = -torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv).mean()
            loss = loss + 0.5 * nn.functional.mse_loss(new_val, batch['returns'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    check("GRF PPO backward (10 steps)", test_grf_backward)

    env.close()


# ---------------------------------------------------------------------------
# Mock game state generator
# ---------------------------------------------------------------------------
def _make_mock_game_state():
    """Create a valid mock GRF game state dict for unit tests."""
    gs = {
        'ball': [0.0, 0.0, 0.0],
        'ball_direction': [0.0, 0.0, 0.0],
        'ball_rotation': [0.0, 0.0, 0.0],
        'ball_owned_team': 0,
        'ball_owned_player': 5,
        'left_team': [
            [-0.95, 0.0],   # GK
            [-0.5, -0.15],  # CB
            [-0.5, 0.15],   # CB
            [-0.3, -0.35],  # LB
            [-0.3, 0.35],   # RB
            [-0.1, 0.0],    # DM
            [0.1, -0.2],    # CM
            [0.1, 0.2],     # CM
            [0.3, -0.35],   # LM
            [0.3, 0.35],    # RM
            [0.5, 0.0],     # CF
        ],
        'left_team_roles': [0, 1, 1, 2, 3, 4, 5, 5, 6, 7, 9],
        'left_team_direction': [[0, 0]] * 11,
        'left_team_tired_factor': [0.0] * 11,
        'left_team_yellow_card': [0] * 11,
        'left_team_active': [True] * 11,
        'right_team': [
            [0.95, 0.0], [0.5, -0.15], [0.5, 0.15],
            [0.3, -0.35], [0.3, 0.35], [0.1, 0.0],
            [-0.1, -0.2], [-0.1, 0.2], [-0.3, -0.35],
            [-0.3, 0.35], [-0.5, 0.0],
        ],
        'right_team_roles': [0, 1, 1, 2, 3, 4, 5, 5, 6, 7, 9],
        'steps_left': 3000,
        'score': [0, 0],
        'game_mode': 0,
    }
    return gs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  HMARL-GRF Dry Run / Smoke Test")
    print("=" * 60)

    ast_only = "--ast-only" in sys.argv
    full = "--full" in sys.argv

    # Layer 1 always runs
    ast_ok = layer1_ast_check()
    if ast_only:
        sys.exit(0 if ast_ok else 1)

    # Layer 2
    layer2_components()
    layer2b_mock_pipeline()

    # Layer 3 (optional)
    if full:
        layer3_full_integration()

    # Summary
    print("\n" + "=" * 60)
    if FAILED == 0:
        print(f"  ALL {PASSED} CHECKS PASSED")
    else:
        print(f"  {PASSED} passed, {FAILED} FAILED")
    print("=" * 60)
    sys.exit(0 if FAILED == 0 else 1)


if __name__ == "__main__":
    main()
