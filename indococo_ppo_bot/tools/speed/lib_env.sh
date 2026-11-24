# shellcheck shell=bash
# Common trainer env for speed jobs on the trainer host. Source after set -u is off.
# BENCH_ENV=det (default): OMP=1 and CUBLAS_WORKSPACE_CONFIG=:4096:8, for SHA/determinism.
# BENCH_ENV=prod: no OMP or cuBLAS caps; matches the live TM trainer environment.
W=/opt/dlami/nvme/speedup_work
if [[ -f /home/ubuntu/ippo_runs/fr9_env.sh ]]; then
  set +u
  eval "$(
    source /home/ubuntu/ippo_runs/fr9_env.sh >/dev/null
    printf 'export LIBTORCH=%q\n' "${LIBTORCH:-}"
    printf 'export LD_LIBRARY_PATH=%q\n' "${LD_LIBRARY_PATH:-}"
    printf 'export PATH=%q\n' "${PATH:-}"
  )"
  set -u
fi
export TMPDIR=$W/tmp TMP=$W/tmp TEMP=$W/tmp
BENCH_ENV=${BENCH_ENV:-det}
export BENCH_ENV
if [[ $BENCH_ENV == det ]]; then
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
elif [[ $BENCH_ENV == prod ]]; then
  unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
  unset CUBLAS_WORKSPACE_CONFIG
else
  echo "BENCH_ENV must be prod or det (got ${BENCH_ENV})" >&2
  return 2 2>/dev/null || exit 2
fi
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export INDOCOCO_OBS_EXT=1 INDOCOCO_OBS_TEAM=1 INDOCOCO_OBS_PRED=1
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$W/logs" "$W/tmp" 2>/dev/null || true

# shellcheck source=/dev/null
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_detect.sh"
