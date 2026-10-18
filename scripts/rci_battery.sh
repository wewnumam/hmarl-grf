#!/bin/sh
# Aspek 2 battery: refresh RCI validity evidence on the newest checkpoint
# (obs106, step 1650889) instead of the stale pre-fix runs.
#
#   --eval-only  : seeds vary env rollouts on ONE checkpoint (no retraining)
#   --load-results: reuse that JSON for construct/discrimination validity
#   --sensitivity: fills the "skipped" hole (needs --model-path)
#
# Run inside the container:
#   docker exec -w /gfootball -e PYTHONPATH=/gfootball gfootball-dev sh scripts/rci_battery.sh
set -e

CKPT=${CKPT:-checkpoints_obs106/hmarl_model.pt}
OUT=${OUT:-evaluation_results_obs106}
EPISODES=${EPISODES:-30}
SEEDS=${SEEDS:-3}

mkdir -p "$OUT"

echo "=== 1/2 stat_test --eval-only (HMARL vs random) | ckpt=$CKPT ==="
python3 -u scripts/stat_test.py \
    --eval-only --model-path "$CKPT" \
    --algorithms hmarl random \
    --seeds "$SEEDS" --eval-episodes "$EPISODES" \
    --output "$OUT/stat_test_results.json"

echo
echo "=== 2/2 validate_rci --load-results + --sensitivity ==="
python3 -u scripts/validate_rci.py \
    --load-results "$OUT/stat_test_results.json" \
    --sensitivity --model-path "$CKPT" \
    --eval-episodes 5 \
    --output "$OUT/rci_validity.json"

echo
echo "=== DONE: $OUT/rci_validity.json ==="
