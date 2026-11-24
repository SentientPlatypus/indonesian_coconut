#!/usr/bin/env bash
# Short GPU critic-extras smoke for TM12 lever 2. Small minibatch, memory
# watchdog. Writes $W12/SMOKE.json with pass=true only if the run exits 0,
# value loss is finite and in (0, 10], and explained variance is finite and
# >= 0.30. Never touches the TM11 trainer. Kill the watchdog by exact PID.
set -u
R=${R:-/home/ubuntu/ippo_runs}
C=${C:-$R/code_tm12/indococo_ppo_bot}
W12=$R/tm12_work
mkdir -p "$W12"
log() { echo "$(date '+%F %T %Z') $*" | tee -a "$W12/smoke.log"; }
die() { log "FATAL: $*"; echo '{"pass":false,"reason":"'"$*"'"}' > "$W12/SMOKE.json"; exit 1; }
source "$C/experiments/lib/paths.env"
B12=${B12:?set B12}
START_CK=${START_CK:-$("$PYT" -c 'import json; print(json.load(open("'"$W12"'/START_DECISION.json"))["start"])' 2>/dev/null)}
[[ -x $B12 ]] || die "B12 missing"
[[ -s $START_CK/actor.bpk && -s $START_CK/critic.bpk ]] || die "start ckpt incomplete"

# Refuse if a trainer is currently using the GPU (TM11 or anything else).
if [[ ${HEAVY_OK:-0} != 1 ]]; then
  for d in /proc/[0-9]*; do
    cmd=$(tr '\0' ' ' <"$d/cmdline" 2>/dev/null) || continue
    case $cmd in *indococo-train\ train*) die "a trainer is already running; not taking the GPU" ;; esac
  done
fi

SMOKE=$W12/critic_smoke
rm -rf "$SMOKE"
mkdir -p "$SMOKE/logs" "$SMOKE/seed"
export INDOCOCO_OBS_EXT=1 INDOCOCO_OBS_TEAM=1 INDOCOCO_OBS_PRED=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export INDOCOCO_METRICS=$SMOKE/logs/metrics.jsonl
cd "$MESH_REPO"
( nice -n 19 "$B12" widen-critic-extras "$START_CK" "$SMOKE/seed" ) > "$SMOKE/widen.log" 2>&1 || die "widen-critic-extras failed"

# Tiny train toml: 3 iterations, small TPI/MB, 1 epoch, no bf16, extras on.
# games/threads kept small so this is a smoke, not a 100M chunk.
TPI=8192; MB=2048
cat > "$SMOKE/train.toml" <<TOML
reward_config = "$R/exp_TM11/tm_rewards.toml"
checkpoints = "$SMOKE/checkpoints"
backend = "torch"
threads = 4
pools = 1
games = 16
timesteps_per_iteration = $TPI
mini_batch_size = $MB
epochs = 1
policy_lr = 5e-5
critic_lr = 5e-5
entropy = 0.016
gamma = 0.995
lambda = 0.95
policy_bf16 = false
critic_bf16 = false
save_every = 1_000_000_000
keep_checkpoints = 2
iterations = 3
critic_warmup_steps = 0
self_play_old_versions = false
skill_tracker = false
[critic]
extra_inputs = ["role_onehot", "eta_rank", "mate_roles"]
TOML
# Prefer TM11 reward file if present; else skip smoke rather than invent rewards.
[[ -s $R/exp_TM11/tm_rewards.toml ]] || die "no TM11 reward config to borrow for the smoke"

# Seed the smoke checkpoints from the widened start.
mkdir -p "$SMOKE/checkpoints/1"
cp "$SMOKE/seed/actor.bpk" "$SMOKE/seed/critic.bpk" "$SMOKE/checkpoints/1/"
if [[ -s $SMOKE/seed/stats.toml ]]; then cp "$SMOKE/seed/stats.toml" "$SMOKE/checkpoints/1/"; fi
if [[ -s $START_CK/stats.toml && ! -s $SMOKE/checkpoints/1/stats.toml ]]; then
  cp "$START_CK/stats.toml" "$SMOKE/checkpoints/1/stats.toml"
fi

# Memory watchdog: abort if used GPU memory rises more than 4 GiB above the
# baseline (small-minibatch smoke should stay well under that).
MEM_LOG=$SMOKE/gpu_mem.log
WATCHDOG_PID=""
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 > "$SMOKE/gpu_base.txt"
  base=$(cat "$SMOKE/gpu_base.txt")
  (
    while :; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      echo "$(date '+%T') $used" >> "$MEM_LOG"
      if [[ -n $used && -n $base ]]; then
        if (( used > base + 4096 )); then
          echo "WATCHDOG oom-risk used=${used}MiB base=${base}MiB" >> "$MEM_LOG"
          exit 42
        fi
      fi
      sleep 2
    done
  ) &
  WATCHDOG_PID=$!
fi
cleanup() {
  if [[ -n ${WATCHDOG_PID:-} ]] && kill -0 "$WATCHDOG_PID" 2>/dev/null; then
    kill "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

set +e
timeout --kill-after=60 600 nice -n 19 "$B12" train "$SMOKE/train.toml" > "$SMOKE/train.log" 2>&1
status=$?
set -e
if [[ -n ${WATCHDOG_PID:-} ]] && ! kill -0 "$WATCHDOG_PID" 2>/dev/null; then
  wait "$WATCHDOG_PID"; w=$?
  [[ $w -eq 42 ]] && die "GPU memory watchdog tripped (see $MEM_LOG)"
fi
[[ $status -eq 0 ]] || die "smoke train exit $status (see $SMOKE/train.log)"

"$PYT" - "$SMOKE/logs/metrics.jsonl" "$SMOKE/train.log" "$W12/SMOKE.json" <<'PY'
import json, math, os, re, sys
metrics_path, log_path, out_path = sys.argv[1:4]

def num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None

rows = []
try:
    for line in open(metrics_path):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
except OSError:
    pass

# Also scrape the train log for the same keys if jsonl is empty.
text = open(log_path).read() if os.path.isfile(log_path) else ""

def last_metric(*names):
    for row in reversed(rows):
        for n in names:
            if n in row and num(row[n]) is not None:
                return num(row[n])
    for n in names:
        m = re.findall(rf"{re.escape(n)}[^0-9\-]*(-?[0-9]+(?:\.[0-9]+)?)", text)
        if m:
            try:
                return float(m[-1])
            except ValueError:
                pass
    return None

vloss = last_metric("PPO/value loss", "PPO/critic loss", "Update/value loss", "value_loss")
ev = last_metric("PPO/explained variance", "explained variance")
reason = []
ok = True
if vloss is None:
    ok = False; reason.append("value loss missing")
elif not (0 < vloss <= 10):
    ok = False; reason.append(f"value loss {vloss} outside (0, 10]")
if ev is None:
    ok = False; reason.append("explained variance missing")
elif ev < 0.30:
    ok = False; reason.append(f"explained variance {ev} < 0.30")
payload = {"pass": bool(ok), "value_loss": vloss, "explained_variance": ev, "reason": "; ".join(reason) or "ok"}
open(out_path, "w").write(json.dumps(payload) + "\n")
print(json.dumps(payload))
sys.exit(0 if ok else 1)
PY
