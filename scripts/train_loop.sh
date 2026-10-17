#!/usr/bin/env bash
# Auto-resuming HMARL training loop (container PID1).
#   start/resume : docker compose up -d
#   clean stop   : docker stop gfootball-dev   (SIGTERM -> trainer saves ckpt+log)
#   config       : env TSTEPS / LOG_DIR / MODEL_DIR / INIT_CKPT (compose passthrough)
# Crash, OOM, or VM/container reboot -> train.py relaunched from latest checkpoint.
# Completed runs (.DONE marker) -> exits 0, restart=on-failure keeps it down.
set -u
cd /gfootball

LOG_DIR=${LOG_DIR:-dumps_obs101}
MODEL_DIR=${MODEL_DIR:-checkpoints_obs101}
TSTEPS=${TSTEPS:-300000}
INIT_CKPT=${INIT_CKPT:-}          # optional warm-start checkpoint (used only once)
DONE="/gfootball/$LOG_DIR/.DONE"
CKPT="/gfootball/$MODEL_DIR/hmarl_model.pt"

child=""
trap '[ -n "$child" ] && kill -TERM "$child" 2>/dev/null; wait 2>/dev/null; exit 0' TERM INT

while true; do
  rm -f "$DONE"
  CMD=(python3 -u scripts/train.py --timesteps "$TSTEPS"
       --log-dir "$LOG_DIR" --model-dir "$MODEL_DIR" --dump-freq 0)
  if [ -f "$CKPT" ]; then
    CMD+=(--resume "$CKPT")
  elif [ -n "$INIT_CKPT" ] && [ -f "$INIT_CKPT" ]; then
    CMD+=(--resume "$INIT_CKPT")
  fi
  echo "[LOOP] starting: ${CMD[*]}"
  PYTHONPATH=/gfootball "${CMD[@]}" &
  child=$!
  wait "$child"
  rc=$?
  child=""
  if [ -f "$DONE" ]; then
    echo "[LOOP] timesteps target reached — training complete"
    exit 0
  fi
  echo "[LOOP] train.py exited rc=$rc without completion — resuming in 10s"
  sleep 10
done
