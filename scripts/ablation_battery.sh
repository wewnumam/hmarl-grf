#!/bin/sh
# Multi-seed ablation study, sharded across processes INSIDE one container.
#
# 7 arms x 4 seeds x 100k steps would be ~16 h sequentially. Sharding by
# (arm-block, seed) cuts that to ~4 h. Same RAM rule as multiseed_battery.sh:
# N *processes* in one container fit in 3.7 GB, two *containers* do not.
#
# Each shard owns its own --temp-root so the per-shard checkpoint dirs cannot
# collide or be deleted by a sibling's cleanup. --merge-inputs pools the shards.
#
# Answers RQ1: which reward component (FAI / PPR / RCI / progression / GK
# penalty) or structural choice (sub-goal conditioning, expert oracle) actually
# drives the measured behaviour.
set -e

STEPS=${STEPS:-100000}
EVAL_EPISODES=${EVAL_EPISODES:-10}
SEEDS=${SEEDS:-"42 43 44 45"}
CONFIGS=${CONFIGS:-"full_hmarl no_fai no_ppr no_rci no_reward_shaping no_gk_penalty flat_policy random_expert"}
# How many shards to run at once. Each shard = one (--seed, --configs subset).
PARALLEL=${PARALLEL:-4}
OUT=${OUT:-evaluation_results_multiseed_ablation}

mkdir -p "$OUT"

echo "=== ablation battery: configs [$CONFIGS] x seeds [$SEEDS] @ $STEPS steps, $EVAL_EPISODES eval ep ==="

i=0
for s in $SEEDS; do
    i=$((i + 1))
    # Rotate which arms go first in each shard so a wall-clock cutoff still
    # leaves every arm represented in at least one finished shard.
    echo "  shard seed=$s -> $OUT/shard_seed${s}.json"
    python3 -u scripts/ablation.py \
        --configs $CONFIGS \
        --timesteps "$STEPS" \
        --eval-episodes "$EVAL_EPISODES" \
        --seed "$s" \
        --temp-root "ablation_temp_seed${s}" \
        --output "$OUT/shard_seed${s}.json" \
        > "$OUT/shard_seed${s}.log" 2>&1 &
    if [ "$i" -ge "$PARALLEL" ]; then
        wait
        i=0
    fi
done
wait
echo "=== shards finished ==="

python3 -u - "$OUT" <<'PY'
import glob, json, os, sys
out = sys.argv[1]
pooled = {}
for p in sorted(glob.glob(os.path.join(out, "shard_seed*.json"))):
    data = json.load(open(p))
    for name, res in data.get("results", {}).items():
        tgt = pooled.setdefault(name, {"description": res.get("description", "")})
        for k, v in res.items():
            if k == "description":
                continue
            if isinstance(v, list):
                tgt[k] = tgt.get(k, []) + v
for name, res in pooled.items():
    n = len(res.get("goal_differences", []))
    res["n_seeds"] = n
    for lst, mean_k, std_k in (("goal_differences", "goal_difference_mean", "goal_difference_std"),
                               ("rewards", "reward_mean", "reward_std"),
                               ("wins", "win_rate_mean", "win_rate_std")):
        vals = res.get(lst)
        if vals:
            m = sum(vals) / len(vals)
            res[mean_k] = m
            res[std_k] = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
merged = {
    "meta": {"script": "scripts/ablation.py (merged shards)", "shards": len(glob.glob(os.path.join(out, "shard_seed*.json")))},
    "results": pooled,
}
with open(os.path.join(out, "ablation_results.json"), "w") as f:
    json.dump(merged, f, indent=2)

print(f"\n{'='*78}")
print(f"  ABLATION SUMMARY (pooled over seeds)")
print(f"{'='*78}")
print(f"  {'Config':<22} {'n':>3} {'WR%':>8} {'Reward':>20} {'Goal Diff':>20}")
for name, res in sorted(pooled.items(), key=lambda kv: -kv[1].get("reward_mean", -1e9)):
    print(f"  {name:<22} {res['n_seeds']:>3} {res['win_rate_mean']:>7.1f}% "
          f"{res['reward_mean']:>13.2f}±{res['reward_std']:<6.2f} "
          f"{res['goal_difference_mean']:>13.2f}±{res['goal_difference_std']:<6.2f}")
print(f"{'='*78}")
PY

echo
echo "=== DONE: $OUT/ablation_results.json ==="
