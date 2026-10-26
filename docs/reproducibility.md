# Reproducibility

Everything runs in Docker. Nothing is installed on the host Python.

## 1. Environment

The runtime image is Google Research Football (TF 1.15, Python 3.6) with the RL
stack layered on top:

```bash
# (a) pre-fetch the torch wheel (single-connection download stalls on this
#     network; the script uses 4 parallel range requests + sha256 check)
bash scratch/dl_torch.sh        # -> wheels/torch-1.10.2-cp36-cp36m-manylinux1_x86_64.whl

# (b) GRF base image — from a GRF source clone (see gfootball/doc/docker.md)
docker build --build-arg DOCKER_BASE=tensorflow/tensorflow:1.15.2-gpu-py3 -t gfootball .

# (c) RL stack on top of it — from this repo
docker build -f Dockerfile.hmarl -t gfootball:hmarl .
```

`Dockerfile.hmarl` pins the versions every log in `dumps*/` reports:
`torch==1.10.2+cu102`, `stable_baselines3==1.3.0`, `gym==0.19.0`. Do not
"upgrade" them — the archived training logs were produced with exactly these.

Two deliberate deviations from the original install line:

- **torch comes from a local wheel, not from `download.pytorch.org`.** That host
  is unreachable here (HTTP 206 headers then 0 B/s) and `pip` aborted after
  ~39 min mid-download of the 882 MB wheel. The PyPI manylinux wheel is the same
  cu102 build — `torch.__version__` still reports `1.10.2+cu102`.
- **torchvision is not installed.** Nothing in this repo imports it, and PyPI
  publishes no cp36/x86_64 wheel for 0.11.3 (only aarch64).

`docker-compose.yml` runs `scripts/train_loop.sh` as PID 1 (auto-resume on
crash/reboot, exit 0 when the `.DONE` marker appears).

## 2. Reported runs

| run | dirs | config | device |
|---|---|---|---|
| Main A (reported in BAB V/VI) | `dumps/`, `checkpoints/` | `11_vs_11_stochastic`, no checkpoint reward, GAE 0.95, ent 0.01 | CPU |
| 3M main experiment | `dumps_3m/`, `checkpoints_3m/` | same as Main A, full 3,000,000 steps | CPU |
| obs10x ladder (archived, not reported) | `dumps_obs100…106/`, `checkpoints_obs10x/` | `11_vs_11_easy_stochastic` + checkpoint reward, GAE 0.99, ent 0.03/0.01 | CPU |
| full3m probe (archived) | `dumps_full3m/`, `checkpoints_full3m/` | easy scenario + checkpoint reward, stopped at 750k | CPU |

All runs are CPU: every `training_log.json` records `cuda_available: false`, and
`scripts/train.py` calls `torch.set_num_threads(1)`. Measured throughput on the
main config ≈ **10.6 env steps/s** (Main A: 1,500,000 steps in 141,267 s), so the
full 3M run takes ≈ **78 h** wall clock. The ~31 steps/s quoted in `README.md`
is a short-window measurement that excludes mid-training evaluation and
checkpointing; budget with the end-to-end 10.6 figure.

Hyperparameters live at the top of `scripts/train.py` and must match Table 9 of
the thesis: `LEARNING_RATE=3e-4`, `GAMMA=1.0`, `GAE_LAMBDA=0.95`,
`CLIP_RANGE=0.2`, `ENT_COEF=0.01`, `VF_COEF=0.5`, `MINIBATCH_SIZE=64`,
`NUM_EPOCHS=4`, `EPISODE_MAX_STEPS=3000`. Reward coefficients live at the top of
`hmarl/reward.py` (`ALPHA_HIGH=0.005`, `ALPHA_MID=0.01`, `ALPHA_LOW=0.001`,
`ALPHA_PROG=3.0`, `GAME_WIN=+30`, `GAME_LOSE=-15`, `CONCEDE_PENALTY=-0.5`,
goal bonus `+5.0` raw = `+25` scaled).

## 3. Launching

```bash
docker compose up -d          # start/resume the run configured in compose
docker compose logs -f        # follow
docker stop gfootball-dev     # clean stop (trainer saves ckpt + log)
```

A finished run writes `$LOG_DIR/.DONE` and the loop exits 0.

## 4. Evaluation and batteries

All battery scripts shard by seed and write JSON aggregates; run them one
container at a time (the WSL2 VM has ~3.7 GB RAM — two containers OOM at 137).

```bash
scripts/multiseed_battery.sh   # 4 algos × 100k × seeds 42-45 -> evaluation_results_multiseed/
scripts/baseline_battery.sh    # 4 algos × 100k × seed 42   -> evaluation_results_baseline/
scripts/rci_battery.sh         # RCI validity, n=16         -> evaluation_results_multiseed/rci_validity.json
scripts/ablation_battery.sh    # 8 arms × 50k × seeds 42-45 -> evaluation_results_ablation/
```

Diagnostics: `scripts/probe_gk.py`, `probe_shot_path.py`, `probe_gradient.py`,
`validate_rci.py`, `validate_expert.py`, and
`validate_expert_criteria.py` (expert ground-truth criteria (a)(b)(c) of BAB IV
→ `evaluation_results/expert_ground_truth.json`).

## 5. Caveats on the archived results

- The per-seed checkpoints of the battery runs (`*_temp_seed*` dirs) were
  deleted after the aggregates were merged. The JSON in `evaluation_results_*`
  is aggregate-only; re-running a battery retrains from scratch.
- `evaluation_results/` (12 Oct) is superseded: it evaluated a single 100-episode
  run, and its `stat_test_results.json` is broken (`All numbers are identical in
  mannwhitneyu` — the archived HMARL rewards were `[0, 0, 0]`, and its random
  policy RCI of 0.000 was an artifact of `evaluate_random` not computing RCI).
  Cite `evaluation_results_multiseed/` instead.
- `evaluation_results_obs106/` is empty: the obs10x ladder was evaluated only up
  to obs100. The ladder is archived, not reported.
- SHPPO (`hmarl/shppo.py`) is implemented but not part of the reported
  comparison: measured throughput is ~0.7 CPU steps/s, i.e. ~41 h per 100k-step
  seed, which makes a 4-seed battery infeasible. BAB VI/VII state this
  explicitly.
