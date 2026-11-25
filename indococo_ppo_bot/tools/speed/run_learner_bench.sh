#!/bin/bash
# Interleaved learn-replay arms in one team mid window. Never touches TM11/TM12.
# mb 50k, GPU_GUARD_MAX_MIB=9800, 2 s watchdog kills only our pid,
# no new arm with under MIN_LEFT_S (default 150) of a 9-minute window left.
set -euo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/lib_env.sh"
BIN=${BIN:?}
OUT=${OUT:?}
ROUND=${ROUND:?}
CK=${CK:-$W/ckpt/tm10/1795503212}
MEM=${MEM:-$W/bench/profile/memory}
SRC_TOML=${SRC_TOML:-$W/bench/profile/train.toml}
ARMS=${ARMS:-"base l1 base l1"}
REPEAT=${REPEAT:-6}
MB=${MB:-50000}
MIN_LEFT_S=${MIN_LEFT_S:-150}
WINDOW_S=${WINDOW_S:-540}   # 9 minutes
# A live TM12/TMn trainer is TRAIN via tm_state. Do not start in that case.
st=$(tm_state)
kind=${st%% *}
kind=${kind,,}
if [[ $kind == train ]]; then
  echo "$(date -Is) REFUSE trainer is back ($st)"
  exit 4
fi

mid_t0=${MID_START_EPOCH:-$(mid_start_epoch)}
export MID_START_EPOCH=$mid_t0
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/$ROUND"
echo "$(date -Is) learner-bench start ROUND=$ROUND ARMS=$ARMS MB=$MB REPEAT=$REPEAT BENCH_ENV=${BENCH_ENV:-det} mid_t0=$mid_t0 $st" | tee -a "$OUT/$ROUND/run.log"

for arm in $ARMS; do
  st=$(tm_state)
  kind=${st%% *}
  kind=${kind,,}
  if [[ $kind == train ]]; then
    echo "$(date -Is) STOP trainer back ($st)" | tee -a "$OUT/$ROUND/run.log"
    exit 0
  fi
  now=$(date +%s)
  if (( now - mid_t0 > WINDOW_S - MIN_LEFT_S )); then
    echo "$(date -Is) STOP window ending (elapsed=$((now-mid_t0))s left=$((WINDOW_S - (now-mid_t0)))s)" | tee -a "$OUT/$ROUND/run.log"
    exit 0
  fi
  dest=$OUT/$ROUND/$arm
  # Unique dest if the same arm repeats in one round (base appears twice).
  n=1
  while [[ -e $dest ]]; do
    n=$((n + 1))
    dest=$OUT/$ROUND/${arm}_$n
  done
  mkdir -p "$dest"
  toml=$dest/train.toml
  python3 "$HERE/write_replay_toml.py" --src "$SRC_TOML" --arm "$arm" --mb "$MB" --out "$toml"
  echo "$(date -Is) START $arm dest=$dest" | tee -a "$OUT/$ROUND/run.log"
  # learn-replay loads ./collision_meshes from cwd. The scratch repo has a
  # read-only symlink to the official meshes; never write there.
  (
    cd "$W/repo"
    LEARN_REPLAY_REPEAT=$REPEAT INDOCOCO_SPEED_PROFILE=1 \
      nice -n 19 taskset -c 0-7 \
      "$BIN" learn-replay "$toml" "$CK" "$MEM" "$dest" 0
  ) &> "$dest/log" &
  pid=$!
  echo $pid > "$dest/pid"
  GPU_GUARD_MAX_MIB=${GPU_GUARD_MAX_MIB:-9800} GPU_GUARD_LOG=$dest/guard.log \
    "$HERE/gpu_guard.sh" "$pid" &
  guard=$!
  aborted=0
  while kill -0 $pid 2>/dev/null; do
    st=$(tm_state)
    kind=${st%% *}
    kind=${kind,,}
    if [[ $kind == train ]]; then
      echo "$(date -Is) ABORT $arm trainer back ($st); killing our pid $pid" | tee -a "$OUT/$ROUND/run.log"
      kill $pid || true
      aborted=1
      break
    fi
    sleep 2
  done
  set +e
  wait $pid
  rc=$?
  set -e
  kill $guard 2>/dev/null || true
  wait $guard 2>/dev/null || true
  if [[ $aborted -eq 1 ]]; then
    echo "$(date -Is) ABORTED $arm rc=$rc" | tee -a "$OUT/$ROUND/run.log"
    exit 0
  fi
  echo "$(date -Is) DONE $arm rc=$rc dest=$dest" | tee -a "$OUT/$ROUND/run.log"
done
echo "$(date -Is) LEARNER_BENCH_DONE ROUND=$ROUND" | tee -a "$OUT/$ROUND/run.log"
