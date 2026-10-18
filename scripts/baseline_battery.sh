#!/bin/sh
# Aspect 3: train the flat baselines (IPPO, SHPPO, MAPPO) and HMARL-eval them,
# plus statistical tests vs HMARL. This is the comparison promised in
# "Tujuan Penelitian" (IPPO / SHPPO / MAPPO) which has never been run.
#
# IMPORTANT (hardware guard): run SEQUENTIALLY. stat_test.py trains each
# algorithm in a Python loop inside ONE process, so it is already sequential
# (no two containers). RAM on this WSL2 VM is 3.7 GB.
#
# Budget: keep small first. 300k steps mirrors the other legs; raise only if
# the tiny run shows signal. Edit STEPS below.
set -e

STEPS=${STEPS:-100000}
SEEDS=${SEEDS:-1}
EVAL_EPISODES=${EVAL_EPISODES:-10}
OUT=${OUT:-evaluation_results_baseline}
ALGOS=${ALGOS:-"hmarl random ippo shppo mappo"}
CKPT=${CKPT:-checkpoints_obs106/hmarl_model.pt}

mkdir -p "$OUT"

echo "=== 1/2 Baseline comparison: $ALGOS ==="
echo "    steps=$STEPS seeds=$SEEDS eval_episodes=$EVAL_EPISODES -> $OUT"

# --eval-only is INTENTIONALLY absent: flat baselines have no checkpoint yet,
# so stat_test must actually train them (see the eval-only skip branch).
python3 -u scripts/stat_test.py \
    --algorithms $ALGOS \
    --timesteps "$STEPS" \
    --seeds "$SEEDS" \
    --eval-episodes "$EVAL_EPISODES" \
    --output "$OUT/stat_test_results.json" 2>&1 | tr '\r' '\n'

echo
echo "=== 2/2 RCI validity: construct + discrimination + sensitivity ==="
# --sensitivity fills the "skipped" gap in the existing rci_validity.json.
python3 -u scripts/validate_rci.py \
    --load-results "$OUT/stat_test_results.json" \
    --sensitivity \
    --model-path "$CKPT" \
    --eval-episodes "${SENS_EPISODES:-5}" \
    --output "$OUT/rci_validity.json" 2>&1 | tr '\r' '\n'

echo
echo "=== DONE: $OUT/stat_test_results.json + $OUT/rci_validity.json ==="
