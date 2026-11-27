#!/usr/bin/env bash
# Short GPU smoke for TM13 (P1+L1+L2 speedups, no L3, no critic extras).
# Passes only if the run exits 0, the log has no panic, and measured
# Throughput/overall (or SPEED sps=) is >= 55k. Optionally exercises lever 4
# (LEVER4=1) with empty pre-goal banks (Replay load is a no-op, must not panic).
# Never touches the TM12 trainer, exp_TM12, code_tm12, or those tmux sessions.
# Refuse if a trainer is already on the GPU unless HEAVY_OK=1. Kill only the
# watchdog PID this script started. Do not launch the full TM13 run.
set -u
R=${R:-/home/ubuntu/ippo_runs}
C=${C:-$R/code_tm13/indococo_ppo_bot}
W13=$R/tm13_work
mkdir -p "$W13"
log() { echo "$(date '+%F %T %Z') $*" | tee -a "$W13/smoke.log"; }
die() { log "FATAL: $*"; echo '{"pass":false,"reason":"'"$*"'"}' > "$W13/SMOKE.json"; exit 1; }
# shellcheck disable=SC1091
source "$C/experiments/lib/paths.env"
B13=${B13:?set B13 to the TM13 binary}
START_CK=${START_CK:-/home/ubuntu/ippo_runs/exp_TM10/checkpoints/1795503212}
[[ -x $B13 ]] || die "B13 missing"
[[ -s $START_CK/actor.bpk && -s $START_CK/critic.bpk ]] || die "start ckpt incomplete"

if [[ ${HEAVY_OK:-0} != 1 ]]; then
  for d in /proc/[0-9]*; do
    cmd=$(tr '\0' ' ' <"$d/cmdline" 2>/dev/null) || continue
    case $cmd in *indococo-train\ train*) die "a trainer is already running; not taking the GPU" ;; esac
  done
fi

SMOKE=$W13/tm13_smoke
rm -rf "$SMOKE"
mkdir -p "$SMOKE/logs" "$SMOKE/checkpoints/1" "$SMOKE/pregoal"
export INDOCOCO_OBS_EXT=1 INDOCOCO_OBS_TEAM=1 INDOCOCO_OBS_PRED=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export INDOCOCO_METRICS=$SMOKE/logs/metrics.jsonl
export INDOCOCO_SPEED=1
cd "$MESH_REPO" || die "MESH_REPO missing"

cp "$START_CK/actor.bpk" "$START_CK/critic.bpk" "$SMOKE/checkpoints/1/"
if [[ -s $START_CK/stats.toml ]]; then cp "$START_CK/stats.toml" "$SMOKE/checkpoints/1/"; fi

# Render a tiny train toml from the TM13 template (placeholders + speed keys).
# Defaults match a short production-like step (24 threads / 2 pools / 384
# games, 300k TPI). The 16-game / 4-thread toy batch only collects ~16k sps
# and cannot clear the 55k bar even when the full run is ~80k.
TPI=${SMOKE_TPI:-300000}
MB=${SMOKE_MB:-100000}
ITERS=${SMOKE_ITERS:-2}
THREADS=${SMOKE_THREADS:-24}
POOLS=${SMOKE_POOLS:-2}
GAMES=${SMOKE_GAMES:-384}
REWARD=${REWARD:-$C/experiments/tm13.toml}
[[ -s $REWARD ]] || die "tm13.toml missing at $REWARD"
PREGOAL=$SMOKE/pregoal
L4_WEIGHT=0.0
[[ ${LEVER4:-0} == 1 ]] && L4_WEIGHT=0.15
cat > "$SMOKE/train.toml" <<TOML
reward_config = "$REWARD"
checkpoints = "$SMOKE/checkpoints"
backend = "torch"
threads = $THREADS
pools = $POOLS
games = $GAMES
timesteps_per_iteration = $TPI
mini_batch_size = $MB
epochs = 1
policy_lr = 5e-5
critic_lr = 5e-5
entropy = 0.008
gamma = 0.995
lambda = 0.95
policy_bf16 = false
critic_bf16 = false
save_every = 1_000_000_000
keep_checkpoints = 2
iterations = $ITERS
critic_warmup_steps = 0
self_play_old_versions = false
skill_tracker = false
nf_intra_threads = ${NF_INTRA:-2}
mask_upload = "auto"
pinned_upload = true
[critic]
extra_inputs = []
[[state_setters]]
kind = "kickoff"
weight = 1.0
[[state_setters]]
kind = "replay"
weight = $L4_WEIGHT
paths = { "2" = "$PREGOAL/pregoal_2v2.f32", "3" = "$PREGOAL/pregoal_3v3.f32" }
TOML

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
        # Production-like 384-game smoke sits around 10-16 GiB on the L4.
        # 20 GiB leaves headroom under the 23 GiB card; 4 GiB was for the toy batch.
        if (( used > 20480 )); then
          echo "WATCHDOG oom-risk used=${used}MiB cap=20480MiB" >> "$MEM_LOG"
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
timeout --kill-after=60 600 nice -n 19 "$B13" train "$SMOKE/train.toml" > "$SMOKE/train.log" 2>&1
status=$?
set -e
if [[ -n ${WATCHDOG_PID:-} ]] && ! kill -0 "$WATCHDOG_PID" 2>/dev/null; then
  wait "$WATCHDOG_PID"; w=$?
  [[ $w -eq 42 ]] && die "GPU memory watchdog tripped (see $MEM_LOG)"
fi
if grep -Eiq 'panic|SIGABRT|fatal runtime' "$SMOKE/train.log"; then
  die "panic or fatal in smoke train log (see $SMOKE/train.log)"
fi
[[ $status -eq 0 ]] || die "smoke train exit $status (see $SMOKE/train.log)"

"$PYT" - "$SMOKE/logs/metrics.jsonl" "$SMOKE/train.log" "$W13/SMOKE.json" <<'PY'
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

text = open(log_path).read() if os.path.isfile(log_path) else ""

def last_metric(*names):
    for row in reversed(rows):
        for n in names:
            if n in row and num(row[n]) is not None:
                return num(row[n])
    for n in names:
        m = re.findall(rf"{re.escape(n)}[^0-9\-]*(-?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)", text)
        if m:
            try:
                return float(m[-1])
            except ValueError:
                pass
    return None

sps = last_metric("Throughput/overall", "throughput/overall")
if sps is None:
    m = re.findall(r"sps=([0-9]+(?:\.[0-9]+)?)", text)
    if m:
        try:
            sps = float(m[-1])
        except ValueError:
            sps = None

reason = []
ok = True
if sps is None:
    ok = False
    reason.append("throughput missing")
elif sps < 55_000:
    ok = False
    reason.append(f"throughput {sps:.1f} < 55000")
payload = {"pass": bool(ok), "sps": sps, "lever4": os.environ.get("LEVER4", "0"), "reason": "; ".join(reason) or "ok"}
open(out_path, "w").write(json.dumps(payload) + "\n")
print(json.dumps(payload))
sys.exit(0 if ok else 1)
PY
