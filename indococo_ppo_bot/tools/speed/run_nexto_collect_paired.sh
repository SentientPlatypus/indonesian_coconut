#!/bin/bash
# Nexto-only scratch-trainer collect: B_base vs B_new (lever on).
# Mid-window only (GPU free). 6 iterations, mb 50k, TM10 scratch ckpt.
# Notes CPU contention when mid evals are also running.
set -euo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/lib_env.sh"
BIN_NEW=${BIN_NEW:?set BIN_NEW}
BIN_BASE=${BIN_BASE:-/opt/dlami/nvme/ippo_bins/indococo-train-speed-48e4dd3H1}
CK=${CK:-$W/ckpt/tm10/1795503212}
ITERS=${ITERS:-4}
MB=${MB:-50000}
TPI=${TPI:-300000}
NEXTO=${NEXTO_DIR:-/home/ubuntu/eval_bots/nexto}
REPLICAS=${NEXTO_REPLICAS:-8}
CHUNK=${NEXTO_CHUNK:-64}
INTRA=${NF_INTRA:-2}
OUT=${OUTDIR:-$W/bench/nexto_collect}
ONLY=${ONLY:-both}  # both|base|new
GUARD=${GUARD:-$W/repo/indococo_ppo_bot/tools/speed/gpu_guard.sh}
# gpu_guard also lives at $W/gpu_guard.sh in older layouts
[[ -x $GUARD ]] || GUARD=$W/gpu_guard.sh
mkdir -p "$OUT"
cd "$W/repo"

st=$(tm11_state)
kind=${st%% *}
if [[ $kind == TRAIN ]]; then
  echo "$(date -Is) REFUSE: GPU is TM11 TRAIN ($st)"
  exit 4
fi
echo "$(date -Is) window=$st replicas=$REPLICAS chunk=$CHUNK intra=$INTRA iters=$ITERS mb=$MB"

# Base template: prefer an existing speed-bench train.toml.
SRC=${SRC_TOML:-$W/bench/profile/train.toml}
if [[ ! -f $SRC ]]; then
  SRC=$W/bench/train.toml
fi
cp -a "$SRC" "$OUT/src.toml"

write_toml() {
  local dest=$1 lever=$2
  python3 "$(dirname "$0")/write_nexto_only_toml.py" \
    "$SRC" "$dest" "$lever" "$OUT" "$CK" "$NEXTO" "$ITERS" "$MB" "$TPI" "$REPLICAS" "$CHUNK" "$INTRA"
}

run_one() {
  local name=$1 bin=$2
  local dir=$OUT/$name
  rm -rf "$dir"
  mkdir -p "$dir" "$OUT/${name}_ckpt"
  # seed from TM10 copy, leave source intact
  python3 - "$CK" "$OUT/${name}_ckpt" << 'PY'
import sys, pathlib, shutil, time
src, dest = map(pathlib.Path, sys.argv[1:])
dest.mkdir(parents=True, exist_ok=True)
# new numeric ckpt id so the trainer sees a resume point
ck = dest / str(int(time.time()))
if ck.exists():
    shutil.rmtree(ck)
shutil.copytree(src, ck)
print("seeded", ck)
PY
  write_toml "$dir/train.toml" "$name"
  echo "$(date -Is) START $name bin=$bin" | tee -a "$OUT/run.log"
  # SPEED_BENCH_KEEP not needed; unique dirs
  nice -n 19 "$bin" train "$dir/train.toml" >"$dir/train.stdout" 2>"$dir/train.stderr" &
  local pid=$!
  echo $pid > "$dir/pid"
  if [[ -x $GUARD ]]; then
    nice -n 19 "$GUARD" "$pid" >>"$OUT/gpu_guard_${name}.log" 2>&1 &
  fi
  # Abort if TM11's trainer returns: never share the GPU. Kill only our pid.
  local rc=0
  while kill -0 "$pid" 2>/dev/null; do
    st=$(tm11_state)
    kind=${st%% *}
    kind=${kind,,}
    if [[ $kind == train ]]; then
      echo "$(date -Is) ABORT $name: TM11 returned ($st); killing our pid $pid" | tee -a "$OUT/run.log"
      kill "$pid" 2>/dev/null || true
      wait "$pid" || true
      echo "$(date -Is) DONE $name exit=99 (aborted for TM11)" | tee -a "$OUT/run.log"
      return 99
    fi
    sleep 5
  done
  wait "$pid"
  rc=$?
  echo "$(date -Is) DONE $name exit=$rc" | tee -a "$OUT/run.log"
  return $rc
}

echo "$(date -Is) NEXTO_COLLECT start ONLY=$ONLY" | tee -a "$OUT/run.log"
if [[ $ONLY == both || $ONLY == base ]]; then
  run_one base "$BIN_BASE" || {
    rc=$?
    echo "$(date -Is) base failed/aborted rc=$rc" | tee -a "$OUT/run.log"
    if [[ $rc -eq 99 ]]; then exit 0; fi
    exit $rc
  }
  st=$(tm11_state); kind=${st%% *}; kind=${kind,,}
  if [[ $kind == train ]]; then
    echo "$(date -Is) STOP after base: TM11 returned ($st)" | tee -a "$OUT/run.log"
    exit 0
  fi
fi
if [[ $ONLY == both || $ONLY == new ]]; then
  run_one new "$BIN_NEW" || {
    rc=$?
    echo "$(date -Is) new failed/aborted rc=$rc" | tee -a "$OUT/run.log"
    if [[ $rc -eq 99 ]]; then exit 0; fi
    exit $rc
  }
fi
echo "$(date -Is) NEXTO_COLLECT_DONE" | tee -a "$OUT/run.log"
# extract Timing/collection (and Collect/inference time if present)
python3 - "$OUT" << 'PY'
import pathlib,re,sys,statistics
out=pathlib.Path(sys.argv[1])
pat=re.compile(r'(Timing/collection|Collect/inference time|Timing/consumption|Timing/total|wall)\s*[:=]\s*([0-9.]+)')
# also rlgymppo typical lines
for name in ("base","new"):
    text=(out/name/"train.stdout").read_text() if (out/name/"train.stdout").exists() else ""
    err=(out/name/"train.stderr").read_text() if (out/name/"train.stderr").exists() else ""
    blob=text+"\n"+err
    print(f"==== {name} bytes={len(blob)} ====")
    keys={}
    for k,v in pat.findall(blob):
        keys.setdefault(k,[]).append(float(v))
    for k,vs in keys.items():
        print(f"  {k}: n={len(vs)} last5={vs[-5:]} mean={sum(vs)/len(vs):.3f}")
    # print iteration summary lines
    for line in blob.splitlines():
        if "Timing/" in line or "Collect/" in line or "ITERATION" in line or "iteration" in line.lower() and "time" in line.lower():
            if any(s in line for s in ("Timing/","Collect/","s/iter","sec/iter","iteration ")):
                print("  |", line[:240])
PY
