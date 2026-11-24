#!/bin/bash
# Wait for a team-trainer mid window (trainer gone, GPU used < 3000 MiB), then
# learn-replay the existing dump with the EV-fixed binary. Does not overwrite
# the pre-fix replay_{a,b,prof} dirs. Never touches the live team trainer.
set -euo pipefail
W=/opt/dlami/nvme/speedup_work
BIN=/opt/dlami/nvme/ippo_bins/indococo-train-speed-0072244EV
OUT=$W/bench/profile
CK=$W/ckpt/tm10/1795503212
MEM=$OUT/memory
LOG=$W/logs/learn_replay_ev.log
GUARD=$W/gpu_guard.sh
mkdir -p "$W/logs"

set +u
eval "$(
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
export INDOCOCO_OBS_EXT=1 INDOCOCO_OBS_TEAM=1 INDOCOCO_OBS_PRED=1
export PYTHONDONTWRITEBYTECODE=1

# Detect any team trainer (exp_TM* / official indococo-train-tm*-*).
# shellcheck source=/dev/null
source "$(dirname "$0")/lib_detect.sh"


: > "$LOG"
echo "$(date -Is) ev-replay waiter start; bin=$BIN" | tee -a "$LOG"
deadline=$(( $(date +%s) + 90*60 ))
while :; do
  now=$(date +%s)
  if (( now > deadline )); then
    echo "$(date -Is) TIMEOUT waiting for mid window" | tee -a "$LOG"
    exit 3
  fi
  pid=$(tm_train_pid || true)
  used=$(gpu_used_mib || echo 99999)
  if [[ -z ${pid:-} ]] && [[ ${used:-99999} -lt 3000 ]]; then
    echo "$(date -Is) CLEAN WINDOW used=${used}MiB — launching learn-replay" | tee -a "$LOG"
    break
  fi
  echo "$(date -Is) waiting trainer=${pid:-none} used=${used}MiB"
  sleep 5
done

# learn-replay resolves ./collision_meshes from cwd (do not stay in $HOME).
cd "$W/repo"

run_one() {
  local name=$1 dest=$2
  rm -rf "$dest"
  mkdir -p "$dest"
  echo "$(date -Is) START $name" | tee -a "$LOG"
  nice -n 19 "$BIN" learn-replay "$OUT/train.toml" "$CK" "$MEM" "$dest" 0 >>"$LOG" 2>&1 &
  local pid=$!
  echo $pid > "$OUT/${name}.pid"
  nice -n 19 "$GUARD" "$pid" >>"$W/logs/gpu_guard_${pid}.log" 2>&1 &
  wait "$pid"
  local rc=$?
  echo "$(date -Is) END $name rc=$rc" | tee -a "$LOG"
  return $rc
}

run_one ev_a "$OUT/replay_ev_a"
run_one ev_b "$OUT/replay_ev_b"

python3 - << 'PY' | tee -a "$LOG"
import json, hashlib, pathlib
W = pathlib.Path("/opt/dlami/nvme/speedup_work/bench/profile")

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def load(d):
    return json.loads((W/d/"metrics.json").read_text())

a, b = load("replay_a"), load("replay_ev_a")
bb = load("replay_ev_b")
print("======= EV before/after learn-replay (seed 0, same dump) =======")
print("before PPO/explained variance:", a.get("PPO/explained variance"))
print("after  PPO/explained variance:", b.get("PPO/explained variance"))
print("after_b PPO/explained variance:", bb.get("PPO/explained variance"))
timing_pfx = ("PPO/optimizer", "PPO/pipeline", "PPO/training", "PPO/batch", "PPO/trunc bootstrap time", "PPO/upload", "PPO/value inference", "GAE/time", "Prof/")
diffs = []
for k in sorted(set(list(a) + list(b))):
    if k.startswith(timing_pfx) or k == "PPO/explained variance":
        continue
    if a.get(k) != b.get(k):
        diffs.append((k, a.get(k), b.get(k)))
print("non-timing metric diffs besides EV:", diffs or "NONE")
print("ev_a vs ev_b EV equal:", b.get("PPO/explained variance") == bb.get("PPO/explained variance"))
print("actor sha before:", sha(W/"replay_a/actor.bpk"))
print("actor sha after :", sha(W/"replay_ev_a/actor.bpk"))
print("actor sha ev_b  :", sha(W/"replay_ev_b/actor.bpk"))
print("critic sha before:", sha(W/"replay_a/critic.bpk"))
print("critic sha after :", sha(W/"replay_ev_a/critic.bpk"))
print("critic sha ev_b  :", sha(W/"replay_ev_b/critic.bpk"))
print("old_values sha before:", sha(W/"replay_a/old_values.f32"))
print("old_values sha after :", sha(W/"replay_ev_a/old_values.f32"))
print("WEIGHTS_IDENTICAL", sha(W/"replay_a/actor.bpk")==sha(W/"replay_ev_a/actor.bpk") and sha(W/"replay_a/critic.bpk")==sha(W/"replay_ev_a/critic.bpk"))
print("EV_CHANGED", a.get("PPO/explained variance") != b.get("PPO/explained variance"))
PY
echo "$(date -Is) EV_REPLAYS_DONE" | tee -a "$LOG"
