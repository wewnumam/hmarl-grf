#!/bin/sh
# Multi-seed controlled battery, sharded across processes INSIDE one container.
#
# Why sharded: one seed of (hmarl + ippo + mappo + random) at 100k steps takes
# ~3.5 h sequentially. WSL2 has 3.7 GB RAM: two *containers* OOM (exit 137),
# but N *processes* in one container fit (measured: 3 procs -> 1454 MB peak,
# MemAvailable 2550 MB). Each shard gets its own --temp-root so the temp
# cleanup cannot delete a sibling's working directory mid-run.
#
# Answers: RQ2 (RCI discrimination + internal consistency need n>=4 seeds) and
# RQ3 (HMARL vs IPPO/MAPPO with a real statistical test instead of n=1).
set -e

STEPS=${STEPS:-100000}
EVAL_EPISODES=${EVAL_EPISODES:-10}
SEEDS=${SEEDS:-"42 43 44 45"}
ALGOS=${ALGOS:-"hmarl ippo mappo random"}
OUT=${OUT:-evaluation_results_multiseed}

mkdir -p "$OUT"

NUM=0
for s in $SEEDS; do NUM=$((NUM + 1)); done
echo "=== multi-seed battery: $NUM shards x [$ALGOS] @ $STEPS steps, $EVAL_EPISODES eval ep ==="

# Launch every shard in the background; each writes its own file.
for s in $SEEDS; do
    echo "  shard seed=$s -> $OUT/shard_seed${s}.json"
    python3 -u scripts/stat_test.py \
        --algorithms $ALGOS \
        --timesteps "$STEPS" \
        --seeds 1 \
        --base-seed "$s" \
        --temp-root "stat_temp_seed${s}" \
        --eval-episodes "$EVAL_EPISODES" \
        --output "$OUT/shard_seed${s}.json" \
        > "$OUT/shard_seed${s}.log" 2>&1 &
done

FAIL=0
for job in $(jobs -p); do
    wait "$job" || FAIL=1
done
echo "=== shards finished (fail=$FAIL) ==="

# Pool the shards and run the statistical tests over all seeds at once.
python3 -u scripts/stat_test.py \
    --merge-inputs "$OUT"/shard_seed*.json \
    --output "$OUT/stat_test_results.json" 2>&1 | tr '\r' '\n'

echo
echo "=== RCI validity on the pooled seeds (construct + discrimination + consistency) ==="
python3 -u scripts/validate_rci.py \
    --load-results "$OUT/stat_test_results.json" \
    --output "$OUT/rci_validity.json" 2>&1 | tr '\r' '\n'

echo
echo "=== DONE: $OUT/stat_test_results.json + $OUT/rci_validity.json ==="
