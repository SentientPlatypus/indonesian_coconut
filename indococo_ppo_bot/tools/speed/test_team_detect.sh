#!/usr/bin/env bash
# Stub tests for team trainer detection (any exp_TM* / official tm*-* bin).
# CPU-only. Does not touch live trainers, team_best, or EC2.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=/dev/null
source "$HERE/lib_detect.sh"
PYT=${PYT:-python3}
DET=${TEAM_DETECT_PY:-$HERE/team_detect.py}
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1: $2"; fails=$((fails + 1)); }

classify() {
  "$PYT" "$DET" classify "$@"
}

echo "=== classify fixtures ==="

expect_class() {
  local want=$1 name=$2
  shift 2
  local got
  got=$(classify "$@")
  if [[ $got == "$want" ]]; then
    pass "classify-$name"
  else
    fail "classify-$name" "want $want got $got argv=$*"
  fi
}

# Live team trainers
expect_class TRAINER exp-tm11 \
  /home/ubuntu/ippo_runs/exp_TM11/indococo-train train /home/ubuntu/ippo_runs/exp_TM11/train_tm.toml
expect_class TRAINER exp-tm12 \
  /home/ubuntu/ippo_runs/exp_TM12/indococo-train train /home/ubuntu/ippo_runs/exp_TM12/train_tm.toml
expect_class TRAINER exp-tm5r \
  /home/ubuntu/ippo_runs/exp_TM5r/indococo-train train /tmp/train.toml
expect_class TRAINER official-tm12 \
  /opt/dlami/nvme/ippo_bins/indococo-train-tm12-c1b0a99 train /home/ubuntu/ippo_runs/exp_TM12/train_tm.toml
expect_class TRAINER official-tm11 \
  /opt/dlami/nvme/ippo_bins/indococo-train-tm11-adf4e93 train /home/ubuntu/ippo_runs/exp_TM11/train_tm.toml

# Must NOT match: speed benches, waiters, inspect lines, evals without train
expect_class OTHER speed-bench \
  /opt/dlami/nvme/ippo_bins/indococo-train-speed-48e4dd3H1 train /opt/dlami/nvme/speedup_work/bench/train.toml
expect_class OTHER speed-learn-replay \
  /opt/dlami/nvme/ippo_bins/indococo-train-speed-a56ff63n1 learn-replay /tmp/t.toml /tmp/ck /tmp/mem /tmp/out 0
expect_class OTHER wait-tm12 \
  bash /home/ubuntu/ippo_runs/code_tm12/indococo_ppo_bot/experiments/wait_tm11_launch_tm12.sh
expect_class OTHER wait-tm12-bashc \
  bash -c 'B12=/opt/dlami/nvme/ippo_bins/indococo-train-tm12-c1b0a99 bash /home/ubuntu/ippo_runs/code_tm12/indococo_ppo_bot/experiments/wait_tm11_launch_tm12.sh; exec bash'
expect_class OTHER official-launcher \
  bash /home/ubuntu/ippo_runs/code_tm11/indococo_ppo_bot/experiments/team_launch_tm11.sh
expect_class OTHER ssh-inspect \
  ssh -i /home/ubuntu/.ssh/key ubuntu@host 'ps -eo args | grep exp_TM11/indococo-train train'
expect_class OTHER tee-log \
  tee -a /home/ubuntu/ippo_runs/exp_TM11/logs/train.log

# Mid evals for any exp_TM*, including official eval bin with a TM ckpt path
expect_class MID mid-h2h-tm11 \
  /home/ubuntu/ippo_runs/exp_TM11/indococo-train h2h /home/ubuntu/ippo_runs/exp_TM11/checkpoints/1 /tmp/bt 200 8
expect_class MID mid-gal-tm12 \
  /opt/dlami/nvme/ippo_bins/indococo-train-tm12-c1b0a99 galactus-h2h /home/ubuntu/ippo_runs/exp_TM12/checkpoints/9 /tmp/gal 200 8
expect_class MID mid-nexto-tm12 \
  /opt/dlami/nvme/ippo_bins/indococo-train-tm11-adf4e93 nexto-h2h /home/ubuntu/ippo_runs/exp_TM12/checkpoints/9 /tmp/nexto 200 8

# A speed-work h2h (no exp_TM* path) must not look like an official mid
expect_class OTHER speed-h2h \
  /opt/dlami/nvme/ippo_bins/indococo-train-speed-a56ff63n1 nexto-h2h /opt/dlami/nvme/speedup_work/ckpt/a /tmp/nexto 40 4

echo "=== stub /proc state + pids ==="
STUB=$(mktemp -d "${TMPDIR:-/tmp}/team_detect.XXXXXX")
cleanup_ok=0
trap 'if [[ $cleanup_ok == 1 ]]; then rm -rf "$STUB"; fi' EXIT

write_cmd() {
  local dir=$1
  shift
  mkdir -p "$dir"
  # NUL-separated cmdline, as in /proc
  "$PYT" - "$dir/cmdline" "$@" << 'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
parts = sys.argv[2:]
path.write_bytes(b"\0".join(p.encode() for p in parts) + b"\0")
PY
}

# Case A: TM12 trainer present
write_cmd "$STUB/3203324" /home/ubuntu/ippo_runs/exp_TM12/indococo-train train /home/ubuntu/ippo_runs/exp_TM12/train_tm.toml
write_cmd "$STUB/3137658" bash /home/ubuntu/ippo_runs/code_tm12/indococo_ppo_bot/experiments/wait_tm11_launch_tm12.sh
write_cmd "$STUB/7" /opt/dlami/nvme/ippo_bins/indococo-train-speed-48e4dd3H1 train /tmp/scratch.toml
export TEAM_PROC_ROOT=$STUB
export TEAM_GPU_USED_MIB=12492
st=$(tm_state)
if [[ $st == "TRAIN pid=3203324 used=12492MiB" ]]; then
  pass state-tm12-train
else
  fail state-tm12-train "got $st"
fi
pid=$(tm_train_pid)
if [[ $pid == 3203324 ]]; then
  pass pid-tm12
else
  fail pid-tm12 "got $pid"
fi
# waiter and speed bench must not appear
pids=$(tm_train_pids | tr '\n' ',')
if [[ $pids == "3203324," || $pids == "3203324" ]]; then
  pass pids-only-trainer
else
  fail pids-only-trainer "got $pids"
fi

# Case B: official tm12 bin + TM11 gone
rm -rf "$STUB/3203324"
write_cmd "$STUB/4000001" /opt/dlami/nvme/ippo_bins/indococo-train-tm12-c1b0a99 train /home/ubuntu/ippo_runs/exp_TM12/train_tm.toml
export TEAM_GPU_USED_MIB=12500
st=$(tm_state)
if [[ $st == "TRAIN pid=4000001 used=12500MiB" ]]; then
  pass state-official-tm12-bin
else
  fail state-official-tm12-bin "got $st"
fi

# Case C: mid only (trainer gone)
rm -rf "$STUB/4000001"
write_cmd "$STUB/5000001" /home/ubuntu/ippo_runs/exp_TM12/indococo-train h2h /home/ubuntu/ippo_runs/exp_TM12/checkpoints/1 /tmp/bt 200 8
write_cmd "$STUB/5000002" /opt/dlami/nvme/ippo_bins/indococo-train-tm12-c1b0a99 galactus-h2h /home/ubuntu/ippo_runs/exp_TM12/checkpoints/1 /tmp/gal 200 8
# a speed h2h must not count
write_cmd "$STUB/5000003" /opt/dlami/nvme/ippo_bins/indococo-train-speed-x nexto-h2h /opt/dlami/nvme/speedup_work/ckpt/a /tmp/n 40 4
export TEAM_GPU_USED_MIB=0
st=$(tm_state)
if [[ $st == "MID used=0MiB ev=5000001,5000002" ]]; then
  pass state-tm12-mid
else
  fail state-tm12-mid "got $st"
fi
if [[ -z $(tm_train_pid) ]]; then
  pass pid-empty-during-mid
else
  fail pid-empty-during-mid "got $(tm_train_pid)"
fi

# Case D: gap
rm -rf "$STUB/5000001" "$STUB/5000002" "$STUB/5000003"
export TEAM_GPU_USED_MIB=0
st=$(tm_state)
if [[ $st == "GAP used=0MiB" ]]; then
  pass state-gap
else
  fail state-gap "got $st"
fi

# Case E: TM11 still detected (regression)
write_cmd "$STUB/2594588" /home/ubuntu/ippo_runs/exp_TM11/indococo-train train /home/ubuntu/ippo_runs/exp_TM11/train_tm.toml
export TEAM_GPU_USED_MIB=12226
st=$(tm_state)
if [[ $st == "TRAIN pid=2594588 used=12226MiB" ]]; then
  pass state-tm11-still
else
  fail state-tm11-still "got $st"
fi

echo "=== wait_tm_state.sh stubs ==="
# Mid already open (no trainer, mid eval present): READY within timeout.
rm -rf "$STUB"/[0-9]*
write_cmd "$STUB/6000001" /home/ubuntu/ippo_runs/exp_TM12/indococo-train h2h /home/ubuntu/ippo_runs/exp_TM12/checkpoints/1 /tmp/bt 40 4
export TEAM_GPU_USED_MIB=0
set +e
"$HERE/wait_tm_state.sh" mid 15 >"$STUB/wait_mid.log" 2>&1
wrc=$?
set -e
if [[ $wrc -eq 0 ]] && grep -q READY "$STUB/wait_mid.log"; then
  pass wait-mid-ready
else
  fail wait-mid-ready "rc=$wrc log=$(cat $STUB/wait_mid.log)"
fi

# Trainer present, want mid, short timeout: TIMEOUT, exit 3. Also via wrapper name.
rm -rf "$STUB"/[0-9]*
write_cmd "$STUB/7000001" /home/ubuntu/ippo_runs/exp_TM12/indococo-train train /tmp/t.toml
export TEAM_GPU_USED_MIB=12000
set +e
"$HERE/wait_tm11_state.sh" mid 3 >"$STUB/wait_timeout.log" 2>&1
wrc=$?
set -e
if [[ $wrc -eq 3 ]] && grep -q TIMEOUT "$STUB/wait_timeout.log"; then
  pass wait-mid-timeout-on-train
else
  fail wait-mid-timeout-on-train "rc=$wrc log=$(cat $STUB/wait_timeout.log)"
fi

echo "=== gpu_guard.sh stubs (kills only our pid) ==="
# Real sleep processes, with stub /proc cmdlines and a fake nvidia-smi.
sleep 60 &
trainer_sleep=$!
sleep 60 &
our_sleep=$!
# Give them a moment to exist
sleep 0.1
if ! kill -0 "$trainer_sleep" 2>/dev/null || ! kill -0 "$our_sleep" 2>/dev/null; then
  fail guard-spawn "could not start sleep stubs trainer=$trainer_sleep our=$our_sleep"
else
  rm -rf "$STUB"/[0-9]*
  write_cmd "$STUB/$trainer_sleep" /home/ubuntu/ippo_runs/exp_TM12/indococo-train train /tmp/t.toml
  write_cmd "$STUB/$our_sleep" /opt/dlami/nvme/ippo_bins/indococo-train-speed-x train /tmp/scratch.toml
  SMIDIR=$STUB/smi
  mkdir -p "$SMIDIR"
  cat > "$SMIDIR/nvidia-smi" << 'SMI'
#!/bin/bash
if [[ "$*" == *query-compute-apps* ]]; then
  cat "${NVIDIA_SMI_APPS_FILE:?}"
elif [[ "$*" == *query-gpu=memory.total* ]]; then
  echo "${NVIDIA_SMI_TOTAL:-23034}"
elif [[ "$*" == *query-gpu=memory.used* ]]; then
  echo "${NVIDIA_SMI_USED:-12000}"
else
  exit 0
fi
SMI
  chmod +x "$SMIDIR/nvidia-smi"
  APPS=$STUB/apps.csv
  # Both on GPU, our use under the cap — CONTAMINATED, kill nobody
  printf '%s, 8000\n%s, 12492\n' "$our_sleep" "$trainer_sleep" > "$APPS"
  export PATH="$SMIDIR:$PATH"
  export NVIDIA_SMI_APPS_FILE=$APPS
  export NVIDIA_SMI_TOTAL=23034
  export GPU_GUARD_LOG=$STUB/guard.log
  # lib_detect must still read TEAM_PROC_ROOT
  unset GPU_GUARD_MAX_MIB
  "$HERE/gpu_guard.sh" "$our_sleep" >"$STUB/guard_out.log" 2>&1 &
  gpid=$!
  # Wait for at least one log line after start
  for i in $(seq 1 20); do
    if grep -q CONTAMINATED "$GPU_GUARD_LOG" 2>/dev/null; then
      break
    fi
    sleep 0.3
  done
  still_our=0 still_tr=0
  kill -0 "$our_sleep" 2>/dev/null && still_our=1
  kill -0 "$trainer_sleep" 2>/dev/null && still_tr=1
  if [[ $still_our -eq 1 && $still_tr -eq 1 ]] && grep -q CONTAMINATED "$GPU_GUARD_LOG"; then
    pass guard-contaminated-no-kill
  else
    fail guard-contaminated-no-kill "our=$still_our trainer=$still_tr log=$(cat $GPU_GUARD_LOG 2>/dev/null) out=$(cat $STUB/guard_out.log 2>/dev/null)"
  fi

  # Now trip the cap on OUR pid only. Trainer must survive.
  printf '%s, 9900\n%s, 12492\n' "$our_sleep" "$trainer_sleep" > "$APPS"
  for i in $(seq 1 20); do
    if grep -q TRIP "$GPU_GUARD_LOG" 2>/dev/null; then
      break
    fi
    sleep 0.3
  done
  # Bound: if the trip did not fire, do not sit on the 60s sleep.
  for i in $(seq 1 20); do
    if ! kill -0 "$gpid" 2>/dev/null; then
      break
    fi
    sleep 0.3
  done
  if kill -0 "$gpid" 2>/dev/null; then
    kill "$gpid" 2>/dev/null || true
  fi
  wait "$gpid" 2>/dev/null || true
  still_our=0 still_tr=0
  kill -0 "$our_sleep" 2>/dev/null && still_our=1
  kill -0 "$trainer_sleep" 2>/dev/null && still_tr=1
  if [[ $still_our -eq 0 && $still_tr -eq 1 ]] && grep -q "kill $our_sleep" "$GPU_GUARD_LOG"; then
    pass guard-trips-only-our-pid
  else
    fail guard-trips-only-our-pid "our=$still_our trainer=$still_tr log=$(cat $GPU_GUARD_LOG 2>/dev/null)"
  fi
  # Cleanup leftover sleeps by exact PID only
  if kill -0 "$trainer_sleep" 2>/dev/null; then
    kill "$trainer_sleep" 2>/dev/null || true
  fi
  if kill -0 "$our_sleep" 2>/dev/null; then
    kill "$our_sleep" 2>/dev/null || true
  fi
fi

echo "=== BENCH_ENV split ==="
# lib_env.sh is source-safe on the box (fr9_env optional).
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS CUBLAS_WORKSPACE_CONFIG BENCH_ENV
set +u
# shellcheck source=/dev/null
source "$HERE/lib_env.sh"
set -u
if [[ ${BENCH_ENV} == det && ${OMP_NUM_THREADS} == 1 && ${CUBLAS_WORKSPACE_CONFIG} == :4096:8 ]]; then
  pass bench-env-det-default
else
  fail bench-env-det-default "BENCH_ENV=${BENCH_ENV:-} OMP=${OMP_NUM_THREADS:-} CUBLAS=${CUBLAS_WORKSPACE_CONFIG:-}"
fi
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS CUBLAS_WORKSPACE_CONFIG
export BENCH_ENV=prod
set +u
# shellcheck source=/dev/null
source "$HERE/lib_env.sh"
set -u
if [[ ${BENCH_ENV} == prod && -z ${OMP_NUM_THREADS+x} && -z ${CUBLAS_WORKSPACE_CONFIG+x} && ${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE} == 1 ]]; then
  pass bench-env-prod-no-caps
else
  fail bench-env-prod-no-caps "BENCH_ENV=${BENCH_ENV:-} OMP=${OMP_NUM_THREADS-UNSET} CUBLAS=${CUBLAS_WORKSPACE_CONFIG-UNSET} TF32=${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-}"
fi

echo "==== results: $fails failed ===="
if [[ $fails -eq 0 ]]; then
  cleanup_ok=1
  exit 0
fi
echo "scratch left at $STUB"
exit 1
