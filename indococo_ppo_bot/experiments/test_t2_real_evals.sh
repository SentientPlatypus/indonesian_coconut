#!/usr/bin/env bash
# T2: 40-game real-eval identity for team_launch_next.sh.
# One heavy CPU job. Must run while TM11 *trains*, not during TM11 mid-evals.
# Scratch only: never writes exp_TM11, team_best, BEST, or indonesian_coconut.
# Kill by exact PID only. Evals already nice 19 via run_eval.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
W=${SPEEDUP_W:-/opt/dlami/nvme/speedup_work}
C=${C:-$W/repo/indococo_ppo_bot}
EVAL_BIN=${EVAL_BIN:-/opt/dlami/nvme/ippo_bins/indococo-train-speed-0072244EV}
CK=${T2_CK:-$W/ckpt/tm10/1795503212}
BT=${BT_PIN:-/home/ubuntu/ippo_runs/exp_TM9/checkpoints/1795388102}
PYT=${PYT:-/home/ubuntu/coco-venv/bin/python}
MESH_REPO=${MESH_REPO:-$W/repo}
PHASE_FILE=$W/logs/t2_phase.txt
LOG=$W/logs/t2_real_evals.log
mkdir -p "$W/logs" "$W/mids_test/t2"
: > "$LOG"
log() { echo "$(date -Is) $*" | tee -a "$LOG"; }
fails=0
pass() { log "PASS $1"; }
fail() { log "FAIL $1: $2"; fails=$((fails + 1)); }

set_phase() { echo "$1" > "$PHASE_FILE"; log "PHASE $1"; }

tm11_train_pid() {
  "$PYT" - << 'PY'
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

tm11_steps() {
  "$PYT" - << 'PY'
from pathlib import Path
data = Path("/home/ubuntu/ippo_runs/exp_TM11/logs/train.log").read_bytes()[-120000:].decode("utf-8","replace")
steps = None
for line in data.splitlines():
    s = line.strip()
    if s.startswith("Cumulative/steps:"):
        try:
            steps = int(s.split(":")[1].strip())
        except ValueError:
            pass
if steps is not None:
    print(steps)
PY
}

# Next mid is the last completed mid + 100M. Do not start a new phase if
# fewer than $1 minutes of TM11 train remain (estimated at 38.7 min/100M).
minutes_left() {
  local now mid left
  now=$(tm11_steps || true)
  mid=$(ls /home/ubuntu/ippo_runs/exp_TM11/mid 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
  if [[ -z ${now:-} || -z ${mid:-} ]]; then echo 0; return; fi
  left=$(( mid + 100000000 - now ))
  "$PYT" -c "print(max(0, $left / 100000000 * 38.7))"
}

guard_ok() {
  local need=${1:-8}
  local pid left
  pid=$(tm11_train_pid || true)
  if [[ -z ${pid:-} ]]; then
    log "GUARD: TM11 trainer is gone (mid window). Stopping T2."
    return 1
  fi
  left=$(minutes_left)
  log "GUARD: tm11_pid=$pid minutes_left=$left need=$need"
  "$PYT" -c "import sys; sys.exit(0 if float('$left') >= $need else 1)"
}

# Libtorch from fr9 without keeping its cd.
set +u
eval "$(
  source /home/ubuntu/ippo_runs/fr9_env.sh >/dev/null
  printf 'export LIBTORCH=%q\n' "${LIBTORCH:-}"
  printf 'export LD_LIBRARY_PATH=%q\n' "${LD_LIBRARY_PATH:-}"
  printf 'export PATH=%q\n' "${PATH:-}"
)"
set -u
export TMPDIR=$W/tmp TMP=$W/tmp TEMP=$W/tmp
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export INDOCOCO_OBS_EXT=1 INDOCOCO_OBS_TEAM=1 INDOCOCO_OBS_PRED=1
export PYTHONDONTWRITEBYTECODE=1
export PYT MESH_REPO

if [[ ! -x $EVAL_BIN ]]; then log "FATAL: EVAL_BIN $EVAL_BIN"; exit 1; fi
if [[ ! -s $CK/actor.bpk ]]; then log "FATAL: ckpt $CK"; exit 1; fi
if [[ ! -s $BT/actor.bpk ]]; then log "FATAL: BT $BT"; exit 1; fi
if [[ ! -x $C/experiments/team_launch_next.sh ]]; then log "FATAL: launcher missing at $C"; exit 1; fi

START_STEPS=$(sed -nE 's/^cumulative_timesteps = ([0-9]+)/\1/p' "$CK/stats.toml")
[[ -n $START_STEPS ]] || { log "FATAL: no steps in $CK/stats.toml"; exit 1; }

R_ROOT=$W/mids_test/t2/runs
mkdir -p "$R_ROOT" "$W/mids_test/t2"
STUB_WRAP=$W/mids_test/t2/train_wrapper.sh
cat > "$STUB_WRAP" << 'WRAP'
#!/usr/bin/env bash
set -u
# If this is a train launch, bump the latest checkpoint by CHUNK_T and exit.
# Any other subcommand is a real eval: exec the speed binary.
if [[ "${1:-}" == "train" ]]; then
  sleep "${STUB_SLEEP:-1}"
  latest=$(ls "$X/checkpoints" | grep -E '^[0-9]+$' | sort -n | tail -1)
  prev=$X/checkpoints/$latest
  now=$(sed -nE 's/^cumulative_timesteps = ([0-9]+)/\1/p' "$prev/stats.toml")
  next=$((now + CHUNK_T))
  base=$(date +%s)
  nd=$X/checkpoints/$base
  while [[ -d $nd ]]; do base=$((base + 1)); nd=$X/checkpoints/$base; done
  mkdir -p "$nd"
  cp "$prev/actor.bpk" "$prev/critic.bpk" "$nd/"
  sed -E "s/^cumulative_timesteps = .*/cumulative_timesteps = $next/" "$prev/stats.toml" > "$nd/stats.toml"
  exit 0
fi
exec "$T2_EVAL_BIN" "$@"
WRAP
chmod +x "$STUB_WRAP"

seed_exp() {
  local x=$1 steps=$2
  mkdir -p "$x/logs" "$x/checkpoints" "$x/opponents" "$x/cfg" "$x/seed" "$x/mid"
  rm -rf "$x/cfg/experiments" "$x/cfg/config"
  cp -a "$C/experiments" "$x/cfg/experiments"
  mkdir -p "$x/cfg/config"
  if [[ -d $C/config ]]; then cp -a "$C/config/." "$x/cfg/config/"; fi
  local init=1000
  mkdir -p "$x/checkpoints/$init" "$x/seed"
  cp -a "$CK/actor.bpk" "$CK/critic.bpk" "$x/checkpoints/$init/"
  cp -a "$CK/actor.bpk" "$CK/critic.bpk" "$x/seed/"
  sed -E "s/^cumulative_timesteps = .*/cumulative_timesteps = $steps/" "$CK/stats.toml" > "$x/checkpoints/$init/stats.toml"
  cp -a "$x/checkpoints/$init/stats.toml" "$x/seed/stats.toml"
  echo "$x/seed $steps TM11" > "$x/START"
  echo LOW=0 > "$x/guard.env"
  cp "$STUB_WRAP" "$x/indococo-train"
  chmod +x "$x/indococo-train"
  # Reward file the h2h path expects (launcher also rewrites this).
  mkdir -p "$x/cfg/experiments"
  printf 'extends = "%s"\n[team]\nspirit_origin = %s.0\n' "$x/cfg/experiments/tm11.toml" "$steps" > "$x/tm_rewards.toml"
}

compare_json() {
  # compare_json label a.json b.json
  local label=$1 a=$2 b=$3
  "$PYT" - "$label" "$a" "$b" << 'PY'
import json, math, sys
label, pa, pb = sys.argv[1], sys.argv[2], sys.argv[3]
DROP = {"seconds"}

def load(p):
    t = open(p).read().strip() or "{}"
    return json.loads(t)

def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in DROP}
    if isinstance(o, list):
        return [strip(x) for x in o]
    return o

a, b = load(pa), load(pb)
sa, sb = strip(a), strip(b)
if sa == sb:
    print("IDENTICAL", label)
    sys.exit(0)

# Binomial fallback on win_rate_decisive / wins-losses
def wr(o):
    if not isinstance(o, dict):
        return None
    if "win_rate_decisive" in o:
        return float(o["win_rate_decisive"])
    g = o.get("games")
    w = o.get("wins")
    l = o.get("losses")
    if g and w is not None and l is not None and (w + l) > 0:
        return w / (w + l)
    return None

def games(o):
    if isinstance(o, dict) and "games" in o:
        return int(o["games"])
    return None

ra, rb = wr(a), wr(b)
ga, gb = games(a), games(b)
g = ga or gb or 0
if ra is not None and rb is not None and g > 0:
    # 2-sigma of a binomial at the pooled rate, on the decisive-game count.
    p = (ra + rb) / 2.0
    sigma = math.sqrt(max(p * (1 - p) / g, 0.0))
    delta = abs(ra - rb)
    ok = delta <= (2 * sigma if sigma > 0 else 0)
    print(("BINOM_OK" if ok else "BINOM_FAIL"), label, "wa", ra, "wb", rb, "delta", delta, "2sigma", 2 * sigma, "games", g)
    sys.exit(0 if ok else 1)
print("DIFF", label)
print("A", json.dumps(sa)[:500])
print("B", json.dumps(sb)[:500])
sys.exit(1)
PY
}

run_h2h_once() {
  local dest=$1
  mkdir -p "$(dirname "$dest")"
  local ev_threads=${EV_THREADS:-8}
  set +e
  ( cd "$MESH_REPO" && \
    env INDOCOCO_TEAM_SIZE=2 H2H_REWARD_CONFIG="$H2H_REWARD" \
      nice -n 19 "$EVAL_BIN" h2h "$CK" "$BT" 40 "$ev_threads" ) > "$dest.raw" 2> "${dest}.err"
  local rc=$?
  set -e
  # last line of stdout is the JSON, same as run_eval
  tail -1 "$dest.raw" > "$dest"
  return $rc
}

# ---------- setup a rewards file and confirm TM11 is training ----------
if ! guard_ok 5; then
  log "T2 not started: TM11 is not in a training window with enough time."
  exit 4
fi

X0=$R_ROOT/exp_T2H2H
seed_exp "$X0" "$START_STEPS"
H2H_REWARD=$X0/tm_rewards.toml

set_phase h2h_a
log "h2h_2v2 run A (40 games)"
if ! run_h2h_once "$W/mids_test/t2/h2h_2v2_a.json"; then
  fail t2-h2h-a "eval exited non-zero; err=$(tail -20 $W/mids_test/t2/h2h_2v2_a.json.err 2>/dev/null)"
fi

if ! guard_ok 5; then
  log "Stopping after h2h A; mid window too close."
  echo "$fails" > "$W/mids_test/t2/fails"
  exit 4
fi

set_phase h2h_b
log "h2h_2v2 run B (40 games)"
if ! run_h2h_once "$W/mids_test/t2/h2h_2v2_b.json"; then
  fail t2-h2h-b "eval exited non-zero; err=$(tail -20 $W/mids_test/t2/h2h_2v2_b.json.err 2>/dev/null)"
fi

if compare_json t2-h2h-2v2-repeat "$W/mids_test/t2/h2h_2v2_a.json" "$W/mids_test/t2/h2h_2v2_b.json"; then
  pass t2-h2h-2v2-repeat
else
  fail t2-h2h-2v2-repeat "see $LOG"
fi

run_launcher_mid() {
  local run=$1 mode=$2
  export R="$R_ROOT"
  export RUN="$run"
  export VARIANT=TM11
  export C
  export X="$R_ROOT/exp_$run"
  export B0="$STUB_WRAP"
  export B="$X/indococo-train"
  export T2_EVAL_BIN="$EVAL_BIN"
  export EVAL_BIN
  export PYT
  export MESH_REPO
  export BT_PIN="$BT"
  export BT="$BT"
  export SKIP_PREFLIGHT=1
  export SKIP_FINALS=1
  export CHUNK_T=100000000
  export CHUNK=100000000
  export STEPS=100000000
  export TPI=300000
  export STUB_SLEEP=2
  export MID_GAMES=40
  export GAL_GAMES=40
  export GAL1_GAMES=40
  export NEXTO_MID=40
  export NECTO_MID=24
  export MEGA_GAMES=24
  export EV_THREADS=8
  export THREADS=2
  export GAMES=2
  export CONCURRENT_MIDS="$mode"
  export INDOCOCO_OBS_PRED=1
  unset EVAL_STUB EVAL_STUB_DIR EVAL_STUB_SLEEP
  seed_exp "$X" "$START_STEPS"
  log "launcher $run CONCURRENT_MIDS=$mode"
  set +e
  bash "$C/experiments/team_launch_next.sh" >"$W/mids_test/t2/${run}.log" 2>&1
  local rc=$?
  set -e
  return $rc
}

if ! guard_ok 10; then
  log "Skipping full mid sets this window (not enough train time left)."
  echo "$fails" > "$W/mids_test/t2/fails"
  exit 4
fi

set_phase mid_mode0
if run_launcher_mid T2M0 0; then
  pass t2-mode0-launcher
else
  fail t2-mode0-launcher "rc nonzero log=$(tail -30 $W/mids_test/t2/T2M0.log)"
fi

if ! guard_ok 10; then
  log "Skipping mode-1 mid set this window (not enough train time left)."
  echo "$fails" > "$W/mids_test/t2/fails"
  exit 4
fi

set_phase mid_mode1
if run_launcher_mid T2M1 1; then
  pass t2-mode1-launcher
else
  fail t2-mode1-launcher "rc nonzero log=$(tail -30 $W/mids_test/t2/T2M1.log)"
fi

# Compare mid.jsonl (drop seconds if any sneak in) and each eval JSON.
M0=$R_ROOT/exp_T2M0
M1=$R_ROOT/exp_T2M1
if [[ -f $M0/logs/mid.jsonl && -f $M1/logs/mid.jsonl ]]; then
  if compare_json t2-mid-jsonl "$M0/logs/mid.jsonl" "$M1/logs/mid.jsonl"; then
    pass t2-mid-jsonl
  else
    fail t2-mid-jsonl "mode0 vs mode1 mid.jsonl differ"
  fi
else
  fail t2-mid-jsonl "missing mid.jsonl m0=$(ls -l $M0/logs 2>/dev/null) m1=$(ls -l $M1/logs 2>/dev/null)"
fi

# Per-eval JSON in the mid dir (use the only numeric mid folder).
mid0=$(ls "$M0/mid" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
mid1=$(ls "$M1/mid" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
if [[ -n ${mid0:-} && -n ${mid1:-} ]]; then
  for f in h2h_2v2 h2h_3v3 gal_2v2 gal_3v3 gal_1v1 mega_2v2 nexto_2v2 nexto_3v3 necto_2v2 necto_3v3 h2h_1v1_st25; do
    a="$M0/mid/$mid0/$f.json"
    b="$M1/mid/$mid1/$f.json"
    if [[ -f $a && -f $b ]]; then
      if compare_json "t2-$f" "$a" "$b"; then
        pass "t2-$f"
      else
        fail "t2-$f" "mode0 vs mode1"
      fi
    else
      fail "t2-$f" "missing a=$a b=$b"
    fi
  done
else
  fail t2-mid-dirs "mid0=$mid0 mid1=$mid1"
fi

set_phase t2_done
log "==== T2 results: $fails failed ===="
echo "$fails" > "$W/mids_test/t2/fails"
exit $fails
