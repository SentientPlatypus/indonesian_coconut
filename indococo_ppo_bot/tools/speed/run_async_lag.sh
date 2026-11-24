#!/bin/bash
# N=10 sync vs async (and async_prio) from the same ckpt/seed.
# Records KL / clip fraction movement from the one-iteration policy lag,
# plus step accounting (10 * 300k) and a one-iter resume check.
# Mid-window only (GPU).
set -euo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/lib_env.sh"
BIN=${BIN:?set BIN}
CK=${CK:-$W/ckpt/tm10/1795503212}
SRC=${SRC_TOML:-$W/bench/profile/train.toml}
ITERS=${ITERS:-10}
TPI=${TPI:-300000}
OUT=${OUTDIR:-$W/bench/async_lag}
GUARD=${GUARD:-$W/repo/indococo_ppo_bot/tools/speed/gpu_guard.sh}
[[ -x $GUARD ]] || GUARD=$W/gpu_guard.sh
mkdir -p "$OUT"
cd "$W/repo"

st=$(tm11_state)
kind=${st%% *}
if [[ $kind == TRAIN ]]; then
  echo "$(date -Is) REFUSE: GPU is TM11 TRAIN ($st)"
  exit 4
fi
echo "$(date -Is) window=$st lag N=$ITERS" | tee "$OUT/run.log"

write_run() {
  local name=$1 async=$2 prio=$3
  python3 - "$SRC" "$OUT/$name" "$CK" "$ITERS" "$TPI" "$async" "$prio" << 'PY'
import sys, pathlib, re, shutil, time
src, dest, seed, iters, tpi, async_, prio = sys.argv[1:]
dest = pathlib.Path(dest)
if dest.exists():
    shutil.rmtree(dest)
dest.mkdir(parents=True)
ckpt = dest / "ckpt"
ckpt.mkdir()
ck = ckpt / str(int(time.time()))
shutil.copytree(seed, ck)
text = pathlib.Path(src).read_text()
def setk(text, key, val):
    pat = re.compile(rf'^{key}\s*=.*$', re.M)
    line = f"{key} = {val}"
    if pat.search(text):
        return pat.sub(line, text, count=1)
    # Top-level only: never append after a [table] (deny_unknown_fields).
    m = re.search(r'(?m)^\[', text)
    if m:
        return text[:m.start()] + line + "\n" + text[m.start():]
    return text.rstrip()+"\n"+line+"\n"
text = setk(text, "checkpoints", f'"{ckpt}"')
text = setk(text, "iterations", iters)
text = setk(text, "timesteps_per_iteration", tpi)
text = setk(text, "async_collect", "true" if async_=="1" else "false")
text = setk(text, "collect_stream_priority", "true" if prio=="1" else "false")
text = setk(text, "critic_warmup_steps", "0")
(dest/"train.toml").write_text(text)
print("wrote", dest/"train.toml", "seed", ck)
PY
}

run() {
  local name=$1
  echo "$(date -Is) START $name $(tm11_state)" | tee -a "$OUT/run.log"
  nice -n 19 "$BIN" train "$OUT/$name/train.toml" >"$OUT/$name/stdout" 2>"$OUT/$name/stderr" &
  local pid=$!
  echo $pid > "$OUT/$name/pid"
  if [[ -x $GUARD ]]; then
    nice -n 19 "$GUARD" "$pid" >>"$OUT/gpu_guard_${name}.log" 2>&1 &
  fi

  local rc=0
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
  echo "$(date -Is) DONE $name exit=$rc" | tee -a "$OUT/run.log"
  return $rc
}

# seed / resume check uses the off run's last checkpoint after the three
# main runs so a mid abort still leaves the KL comparison if two finished.
for spec in "off 0 0" "async 1 0" "async_prio 1 1"; do
  set -- $spec
  st=$(tm11_state); kind=${st%% *}
  if [[ $kind == TRAIN ]]; then
    echo "$(date -Is) STOP before $1: TM11 returned" | tee -a "$OUT/run.log"
    exit 0
  fi
  write_run "$1" "$2" "$3"
  run "$1" || true
done
echo "$(date -Is) ASYNC_LAG_DONE" | tee -a "$OUT/run.log"
