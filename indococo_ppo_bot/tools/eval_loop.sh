#!/usr/bin/env bash
# Head-to-head evals of a training run against a reference policy.
# Polls the checkpoint dir; each time the newest checkpoint passes the next
# multiple of EVERY steps, exports it and runs h2h_1v1.py at low priority.
# Appends one JSON line per eval to $OUT.
#
#   tools/eval_loop.sh <checkpoints> <out.jsonl> <reference.pt>
# env: TRAIN_BIN, H2H (path to h2h_1v1.py), PY, EVERY (100000000), GAMES (400), PROCS (8)
# h2h_1v1.py imports eval_match/v4_env/loop_config from $INDOCOCO_REPO; those
# Python files are on master only (e.g. git worktree add /tmp/ic_master master).
set -euo pipefail
CKPTS=$1 OUT=$2 REF=$3
EVERY=${EVERY:-100000000} GAMES=${GAMES:-400} PROCS=${PROCS:-8}
WORK=$(dirname "$OUT")/eval_tmp
mkdir -p "$WORK"
next=$EVERY
if [[ -s $OUT ]]; then
  last=$(tail -1 "$OUT" | sed -E 's/.*"steps": ([0-9]+).*/\1/')
  next=$(( (last / EVERY + 1) * EVERY ))
fi
while true; do
  newest=$(ls "$CKPTS" | grep -E '^[0-9]+$' | sort -n | tail -1 || true)
  if [[ -n $newest && -f $CKPTS/$newest/stats.toml ]]; then
    steps=$(sed -nE 's/^cumulative_timesteps = ([0-9]+)/\1/p' "$CKPTS/$newest/stats.toml")
    if (( steps >= next )); then
      sleep 30   # let the save finish
      st=$WORK/policy_$steps.safetensors
      nice -n 19 "$TRAIN_BIN" export "$CKPTS/$newest" "$st" > /dev/null
      res=$WORK/h2h_$steps.json
      (cd "$(dirname "$H2H")" && nice -n 19 "$PY" "$(basename "$H2H")" --candidate "$st" --opponent "$REF" \
          --games "$GAMES" --procs "$PROCS" --out "$res" > "$WORK/h2h_$steps.log" 2>&1)
      printf '{"time": "%s", "steps": %s, "checkpoint": "%s", "h2h": %s}\n' \
          "$(TZ=America/New_York date -Iseconds)" "$steps" "$newest" "$("$PY" -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))))' "$res")" >> "$OUT"
      rm -f "$st"
      next=$(( (steps / EVERY + 1) * EVERY ))
    fi
  fi
  sleep 60
done
