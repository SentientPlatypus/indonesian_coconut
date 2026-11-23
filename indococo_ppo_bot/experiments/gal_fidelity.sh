#!/usr/bin/env bash
# Report-only headline Galactus metric for a finished run's final checkpoint
# (EXPERIMENTS.md "Eval fidelity"). Call from a run's finals, e.g.
#   bash $C/experiments/gal_fidelity.sh $bestck $F team 1,2,3 $B      # TM runs (270/235 team obs)
#   bash $C/experiments/gal_fidelity.sh $bestck $F 1v1 1 $B           # ST runs (106-wide)
# Writes $F/gal_fidelity.jsonl (one line per mode: py_exact = headline,
# py_jitter, rust, gap = rust - py_jitter, alarm if gap > 0.05) and
# $F/gal_fidelity.log. Never gates promotion. Env: GAL_FID_GAMES (400),
# GAL_FID_PROCS (8), GAL_FID_RUST_BIN (binary for the Rust leg; default BIN.
# It must honour H2H_JITTER=1, e.g. exp_TM10/indococo-train; older ST binaries
# ignore it and the Rust leg then has one game per spawn). Runs at nice 19.
set -u
CK=$1; F=$2; OBS=${3:-team}; MODES=${4:-1,2,3}; BIN=${5:-}
C=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYT=${PYT:-/home/ubuntu/coco-venv/bin/python}
export LD_LIBRARY_PATH=${TORCH_LIB:-/home/ubuntu/ippo_runs/e8b2_1v1/bin/sp/torch/lib}:${LD_LIBRARY_PATH:-}
mkdir -p "$F"
ST=$F/gal_fidelity_policy.safetensors
case $OBS in
  team)    ENVS="INDOCOCO_OBS_EXT=1 INDOCOCO_OBS_TEAM=1 INDOCOCO_OBS_PRED=1"; KIND=indococo_team ;;
  team235) ENVS="INDOCOCO_OBS_EXT=1 INDOCOCO_OBS_TEAM=1"; KIND=indococo_team ;;
  1v1)     ENVS="INDOCOCO_OBS_EXT=1"; KIND=indococo ;;
  *) echo "obs must be team|team235|1v1" >&2; exit 2 ;;
esac
if [[ -n "$BIN" ]]; then env $ENVS nice -n 19 "$BIN" export "$CK" "$ST" > /dev/null 2>&1 || { echo "export failed" >&2; exit 1; }
else ST=$CK; fi
RUST=(); [[ -n "$BIN" ]] && RUST=(--rust-bin "${GAL_FID_RUST_BIN:-$BIN}" --rust-ckpt "$CK" --rust-env "$OBS")
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 $PYT "$C/tools/pool_eval/gal_fidelity.py" --candidate "$ST" --kind $KIND \
  --modes "$MODES" --games "${GAL_FID_GAMES:-400}" --procs "${GAL_FID_PROCS:-8}" "${RUST[@]}" \
  --label "$(basename "$F")" --out "$F/gal_fidelity.jsonl" > "$F/gal_fidelity.log" 2>&1
tail -5 "$F/gal_fidelity.log"
