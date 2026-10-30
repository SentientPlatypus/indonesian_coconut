#!/usr/bin/env bash
# Pool evals and keep-best for a training run.
# Polls the checkpoint dir; each time the newest checkpoint passes the next
# multiple of EVERY steps, exports it and runs tools/pool_eval/keep_best.py at
# low priority: GAMES games against every bot in the pool (1v1.pt, the saved
# PPO checkpoints, Nexto, Necto, Element, Immortal), one JSON line per eval in
# $OUT, and promotion to $BEST when it beats the current best (pool score and
# head-to-head). $BEST is outside the checkpoint dir, so rotation never removes it.
#
#   tools/eval_loop.sh <checkpoints> <out.jsonl> <best dir>
# env: TRAIN_BIN, PY (python with rlgym 2 + torch), POOL (pool.toml), EVERY (100000000),
#      GAMES (200), PROCS (8)
set -euo pipefail
CKPTS=$1 OUT=$2 BEST=$3
HERE=$(cd "$(dirname "$0")" && pwd)
EVERY=${EVERY:-100000000} GAMES=${GAMES:-200} PROCS=${PROCS:-8}
POOL=${POOL:-$HERE/pool_eval/pool.toml}
WORK=$(dirname "$OUT")/eval_tmp
mkdir -p "$WORK"
next=$EVERY
if [[ -s $OUT ]]; then
  last=$(tail -1 "$OUT" | "$PY" -c 'import json,sys; print(json.loads(sys.stdin.read())["steps"])')
  next=$(( (last / EVERY + 1) * EVERY ))
fi
while true; do
  newest=$(ls "$CKPTS" | grep -E '^[0-9]+$' | sort -n | tail -1 || true)
  if [[ -n $newest && -f $CKPTS/$newest/stats.toml ]]; then
    steps=$(sed -nE 's/^cumulative_timesteps = ([0-9]+)/\1/p' "$CKPTS/$newest/stats.toml")
    if (( steps >= next )); then
      sleep 30   # let the save finish
      snap=$WORK/ckpt_$steps
      rm -rf "$snap" && cp -r "$CKPTS/$newest" "$snap"   # rotation may delete the original
      st=$WORK/policy_$steps.safetensors
      nice -n 19 "$TRAIN_BIN" export "$snap" "$st" > /dev/null
      log=$WORK/keep_best_$steps.jsonl
      nice -n 19 "$PY" "$HERE/pool_eval/keep_best.py" --checkpoint "$snap" --policy "$st" \
          --best "$BEST" --pool "$POOL" --games "$GAMES" --procs "$PROCS" --log "$log" \
          > "$WORK/keep_best_$steps.log" 2>&1
      "$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); r["steps"]=int(sys.argv[2]); print(json.dumps(r))' \
          "$log" "$steps" >> "$OUT"
      rm -rf "$snap" "$st" "$log"
      next=$(( (steps / EVERY + 1) * EVERY ))
    fi
  fi
  sleep 60
done
