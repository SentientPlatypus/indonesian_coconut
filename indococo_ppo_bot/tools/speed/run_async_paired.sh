#!/bin/bash
# Paired interleaved off / async / async_prio speed-bench in a mid window
# (GPU free, CPU contended — recorded). 12 iters/block, drop first 3 later.
# Optional: SPEED_VARIANTS and ITERS. Also used for a short clean/GAP bench.
set -euo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/lib_env.sh"
BIN=${BIN:?set BIN}
ITERS=${ITERS:-12}
ROUNDS=${ROUNDS:-3}
VARIANTS=${SPEED_VARIANTS:-off,async,async_prio}
OUT=${OUTDIR:-$W/bench/async_paired}
CK=${CK:-$W/ckpt/tm10/1795503212}
SRC=${SRC_TOML:-$W/bench/profile/train.toml}
GUARD=${GUARD:-$W/repo/indococo_ppo_bot/tools/speed/gpu_guard.sh}
[[ -x $GUARD ]] || GUARD=$W/gpu_guard.sh
mkdir -p "$OUT"
cd "$W/repo"

st=$(tm11_state)
kind=${st%% *}
echo "$(date -Is) window=$st variants=$VARIANTS iters=$ITERS rounds=$ROUNDS" | tee "$OUT/run.log"
if [[ $kind == TRAIN ]]; then
  echo "$(date -Is) REFUSE: GPU is TM11 TRAIN" | tee -a "$OUT/run.log"
  exit 4
fi

# speed-bench runs variants sequentially in one process. Rotate ABC by
# invoking once per block so a mid-window abort still keeps completed blocks.
# Interleave: for r in 1..ROUNDS: run VARIANTS in order, then we analyze ABA/ABC.
export SPEED_VARIANTS
export SPEED_BENCH_KEEP=1
export TMPDIR=$W/tmp

# Point the train.toml checkpoints at a scratch copy under OUT.
python3 - "$SRC" "$OUT/train.toml" "$OUT/ckpt" "$CK" << 'PY'
import sys, pathlib, re, shutil, time
src, dest, ckpt_root, seed = map(pathlib.Path, sys.argv[1:])
text = src.read_text()
ckpt_root.mkdir(parents=True, exist_ok=True)
ck = ckpt_root / str(int(time.time()))
if ck.exists():
    shutil.rmtree(ck)
shutil.copytree(seed, ck)
text = re.sub(r'^checkpoints\s*=.*$', f'checkpoints = "{ckpt_root}"', text, count=1, flags=re.M)
# mb 50k while TM11 lives — leave whatever the template already has if 50000
if re.search(r'^critic_warmup_steps\s*=', text, re.M):
    text = re.sub(r'^critic_warmup_steps\s*=.*$', 'critic_warmup_steps = 0', text, count=1, flags=re.M)
else:
    text = text.replace('[lr_schedule]', 'critic_warmup_steps = 0\n\n[lr_schedule]', 1)
pathlib.Path(dest).write_text(text)
print("toml", dest, "seed", ck)
PY

block=0
for r in $(seq 1 "$ROUNDS"); do
  st=$(tm11_state); kind=${st%% *}
  if [[ $kind == TRAIN ]]; then
    echo "$(date -Is) STOP: TM11 returned before block $r ($st)" | tee -a "$OUT/run.log"
    exit 0
  fi
  block=$r
  dir=$OUT/block_$r
  mkdir -p "$dir"
  echo "$(date -Is) BLOCK $r START $st" | tee -a "$OUT/run.log"
  SPEED_VARIANTS="$VARIANTS" nice -n 19 "$BIN" speed-bench "$OUT/train.toml" "$ITERS" \
    >"$dir/stdout" 2>"$dir/stderr" &
  pid=$!
  echo $pid > "$dir/pid"
  if [[ -x $GUARD ]]; then
    nice -n 19 "$GUARD" "$pid" >>"$OUT/gpu_guard_b${r}.log" 2>&1 &
  fi

  rc=0
  while kill -0 "$pid" 2>/dev/null; do
    st=$(tm11_state)
    kind=${st%% *}
    kind=${kind,,}
    if [[ $kind == train ]]; then
      echo "$(date -Is) ABORT: TM11 returned ($st); killing our pid $pid" | tee -a "$OUT/run.log"
      kill "$pid" 2>/dev/null || true
      wait "$pid" || true
      echo "$(date -Is) ABORTED exit=99" | tee -a "$OUT/run.log"
      exit 99
    fi
    sleep 5
  done
  wait "$pid"
  rc=$?
  echo "$(date -Is) BLOCK $r DONE exit=$rc $(tm11_state)" | tee -a "$OUT/run.log"
  if [[ $rc -ne 0 ]]; then
    echo "block $r failed" | tee -a "$OUT/run.log"
    # do not start another block on failure
    break
  fi
done
echo "$(date -Is) ASYNC_PAIRED_DONE blocks=$block" | tee -a "$OUT/run.log"
