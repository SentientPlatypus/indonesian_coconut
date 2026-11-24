#!/bin/bash
# Wait for a clean TM11 mid window, then run a 12-iter profile + memory dump.
# Never touches the TM11 trainer, exp_TM11 writes, team_best, or BEST.
# Kill by exact PID only. Nice 19. One GPU test at a time.
set -euo pipefail
W=/opt/dlami/nvme/speedup_work
BIN=/opt/dlami/nvme/ippo_bins/indococo-train-speed-48e4dd3H1
SRC_TOML=/home/ubuntu/ippo_runs/exp_TM11/train_tm.toml
SRC_REW=/home/ubuntu/ippo_runs/exp_TM11/tm_rewards.toml
CK_SRC=$W/ckpt/tm10
OUT=$W/bench/profile
LOG=$W/logs/profile12.log
GUARD=$W/gpu_guard.sh
STOP=$OUT/TF32_STOP
mkdir -p "$OUT" "$W/logs" "$W/tmp" "$W/bench/profile/memory"

# Env (do not source anything that cds into a live run dir after we start)
# fr9_env is needed for LIBTORCH; it may cd — call it in a subshell and export.
# shellcheck disable=SC1091
set +u
# Capture LIBTORCH without keeping fr9's cd.
eval "$(
  # shellcheck disable=SC1090
  source /home/ubuntu/ippo_runs/fr9_env.sh >/dev/null
  printf 'export LIBTORCH=%q\n' "${LIBTORCH:-}"
  printf 'export LD_LIBRARY_PATH=%q\n' "${LD_LIBRARY_PATH:-}"
  printf 'export PATH=%q\n' "${PATH:-}"
)"
set -u
export TMPDIR=$W/tmp TMP=$W/tmp TEMP=$W/tmp
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export INDOCOCO_SPEED_PROFILE=1
export INDOCOCO_DUMP_MEMORY=$OUT/memory
# Match TM11 obs width (270). Missing these made speed-bench panic on TM3_final (92 vs 270).
export INDOCOCO_OBS_EXT=1 INDOCOCO_OBS_TEAM=1 INDOCOCO_OBS_PRED=1
export SPEED_BENCH_KEEP=1
export SPEED_VARIANTS=off
export PYTHONDONTWRITEBYTECODE=1

# Build a scratch toml. Never write under exp_TM11.
cp -a "$SRC_REW" "$OUT/tm_rewards.toml"
python3 - "$SRC_TOML" "$OUT/train.toml" "$OUT/tm_rewards.toml" "$CK_SRC" << 'PY'
import sys, re
src, dst, rew, ck = sys.argv[1:]
t = open(src).read()
t = re.sub(r'(?m)^reward_config = ".*"', f'reward_config = "{rew}"', t, count=1)
t = re.sub(r'(?m)^checkpoints = ".*"', f'checkpoints = "{ck}"', t, count=1)
t = re.sub(r'(?m)^mini_batch_size = .*', 'mini_batch_size = 50_000', t, count=1)
t = re.sub(r'(?m)^iterations = .*', 'iterations = 12', t, count=1)
# Keep epochs = 1 and bf16 false; assert:
assert re.search(r'(?m)^epochs = 1$', t), t
assert 'policy_bf16 = false' in t and 'critic_bf16 = false' in t
open(dst, "w").write(t)
print("wrote", dst)
PY

tm11_train_pid() {
  # Exact argv match, no pgrep -f. Never prints a pid we would kill except for display.
  python3 - << 'PY'
import pathlib
for p in pathlib.Path("/proc").iterdir():
    if not p.name.isdigit():
        continue
    try:
        cmd = open(p/"cmdline", "rb").read().replace(b"\x00", b" ").decode()
    except Exception:
        continue
    if cmd.startswith("/home/ubuntu/ippo_runs/exp_TM11/indococo-train train"):
        print(p.name)
        break
PY
}

gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

echo "$(date -Is) waiter start; will launch $BIN speed-bench when TM11 train pid is gone and GPU used is low"
# Wait up to 45 minutes for a clean window.
deadline=$(( $(date +%s) + 45*60 ))
while :; do
  now=$(date +%s)
  if (( now > deadline )); then
    echo "$(date -Is) TIMEOUT waiting for mid window" | tee -a "$LOG"
    exit 3
  fi
  pid=$(tm11_train_pid || true)
  used=$(gpu_used_mib || echo 99999)
  if [[ -z ${pid:-} ]] && [[ ${used:-99999} -lt 3000 ]]; then
    echo "$(date -Is) CLEAN WINDOW used=${used}MiB — launching profile" | tee -a "$LOG"
    break
  fi
  echo "$(date -Is) waiting tm11=${pid:-none} used=${used}MiB"
  sleep 5
done

# Launch trainer
: > "$LOG"
cd "$W/repo/indococo_ppo_bot"
# MESH_REPO for collision meshes: our scratch symlink
export RUST_BACKTRACE=1
set +e
nice -n 19 "$BIN" speed-bench "$OUT/train.toml" 12 >>"$LOG" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$OUT/train.pid"
echo "$(date -Is) train pid=$TRAIN_PID" | tee -a "$LOG"
# Guard watches only our pid.
nice -n 19 "$GUARD" "$TRAIN_PID" >>"$W/logs/gpu_guard_${TRAIN_PID}.log" 2>&1 &
GUARD_PID=$!
echo "$GUARD_PID" > "$OUT/guard.pid"

# Watch for TF32 line; if cublas=0 with override=1, stop our job only.
for _ in $(seq 1 120); do
  if grep -q 'TF32 cublas=' "$LOG" 2>/dev/null; then
    line=$(grep 'TF32 cublas=' "$LOG" | head -1)
    echo "$(date -Is) $line" | tee -a "$OUT/tf32.txt"
    if echo "$line" | grep -q 'cublas=0' && echo "$line" | grep -q 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=Some("1")'; then
      echo "$(date -Is) TF32_STOP: cublas is OFF despite override=1. Killing our pid $TRAIN_PID only." | tee -a "$LOG" "$STOP"
      kill "$TRAIN_PID" || true
      wait "$TRAIN_PID" 2>/dev/null || true
      exit 4
    fi
    break
  fi
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "$(date -Is) trainer exited before TF32 line" | tee -a "$LOG"
    wait "$TRAIN_PID"
    exit $?
  fi
  sleep 1
done

# Wait for trainer; do not let this overrun a returning TM11 too long — report but don't kill TM11.
wait "$TRAIN_PID"
rc=$?
echo "$(date -Is) trainer exit=$rc" | tee -a "$LOG"
wait "$GUARD_PID" 2>/dev/null || true
# Record peak from guard log
if [[ -f $W/logs/gpu_guard_${TRAIN_PID}.log ]]; then
  echo "----- guard -----" | tee -a "$LOG"
  grep -E 'our=|TRIP|CONTAMINATED|guard ' "$W/logs/gpu_guard_${TRAIN_PID}.log" | tail -20 | tee -a "$LOG"
fi
exit $rc
