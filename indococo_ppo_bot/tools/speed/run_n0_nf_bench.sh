#!/bin/bash
# N0: CPU microbench of the Nexto infer path. nice 19. Aborts between
# configs if TM11 leaves the training window (mid-evals need the CPU).
# Usage: BIN=... bash run_n0_nf_bench.sh
set -euo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/lib_env.sh"
# nf-bench measures intra-op threads; do not inherit the trainer-job OMP=1 caps.
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS TORCH_NUM_THREADS
BIN=${BIN:?set BIN}
NEXTO=${NEXTO_DIR:-/home/ubuntu/eval_bots/nexto}
OUT=${OUT:-$W/logs/n0_nf_bench.jsonl}
ITERS=${ITERS:-8}
CALLERS=${CALLERS:-2}
mkdir -p "$(dirname "$OUT")"
cd "$W/repo"

stop_if_not_train() {
  local st kind
  st=$(tm11_state)
  kind=${st%% *}
  if [[ $kind != TRAIN ]]; then
    echo "$(date -Is) STOP: TM11 left TRAIN ($st)" | tee -a "$W/logs/n0_nf_bench.log"
    return 1
  fi
  return 0
}

run_one() {
  local ts rows iters replicas intra chunk
  ts=$1; rows=$2; iters=$3; replicas=$4; intra=$5; chunk=$6
  echo "$(date -Is) RUN team=$ts rows=$rows iters=$iters replicas=$replicas intra=$intra chunk=$chunk callers=$CALLERS"
  # shellcheck disable=SC2086
  nice -n 19 "$BIN" nf-bench "$NEXTO" "$ts" "$rows" "$iters" "$replicas" "$intra" "$chunk" "$CALLERS" \
    | tee -a "$W/logs/n0_nf_bench.log"
}

echo "$(date -Is) N0 start BIN=$BIN ITERS=$ITERS" | tee "$W/logs/n0_nf_bench.log"
: > "$OUT"
stop_if_not_train

# Baseline first, more iters, both team sizes and row counts.
for spec in "2 384 16" "2 192 16" "3 576 16" "3 192 16"; do
  set -- $spec
  stop_if_not_train || exit 0
  run_one "$1" "$2" "$3" 1 0 0
done

# Full grid, callers=2. Baseline already recorded; still include 1/0/0 at ITERS.
for ts_rows in "2 384" "2 192" "3 576" "3 192"; do
  set -- $ts_rows
  ts=$1; rows=$2
  for replicas in 1 2 4 8; do
    for intra in 0 1 2 4 8; do
      for chunk in 0 64 128; do
        stop_if_not_train || exit 0
        # skip the exact baseline already run with 16 iters, but do a short confirm once
        if [[ $replicas == 1 && $intra == 0 && $chunk == 0 ]]; then
          continue
        fi
        run_one "$ts" "$rows" "$ITERS" "$replicas" "$intra" "$chunk"
      done
    done
  done
done
echo "$(date -Is) N0_ALL_DONE" | tee -a "$W/logs/n0_nf_bench.log"
