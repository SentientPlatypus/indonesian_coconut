# shellcheck shell=bash
# Common trainer env for speed jobs on the EC2 host. Source after set -u is off.
W=/opt/dlami/nvme/speedup_work
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
mkdir -p "$W/logs" "$W/tmp"

tm11_train_pid() {
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

mid_eval_pids() {
  python3 - << 'PY'
import pathlib
need = ("galactus-h2h", "megabyte-h2h", "nexto-h2h", "necto-h2h", " h2h ")
for p in pathlib.Path("/proc").iterdir():
    if not p.name.isdigit():
        continue
    try:
        cmd = open(p/"cmdline", "rb").read().replace(b"\x00", b" ").decode()
    except Exception:
        continue
    if any(s in cmd for s in need) and "exp_TM11" in cmd or (
        any(s in cmd for s in ("galactus-h2h", "megabyte-h2h", "nexto-h2h", "necto-h2h"))
    ):
        # mid evals use EVAL_BIN from the TM11 run or official eval bin
        print(p.name)
PY
}

# TRAIN = TM11 trainer alive (GPU busy, collectors on CPU).
# MID   = trainer gone, typically mid-evals on CPU, GPU free.
# GAP   = trainer gone and no mid-eval processes (rare, short).
tm11_state() {
  local pid used ev
  pid=$(tm11_train_pid || true)
  used=$(gpu_used_mib || echo 99999)
  ev=$(mid_eval_pids || true)
  if [[ -n ${pid:-} ]]; then
    echo "TRAIN pid=$pid used=${used}MiB"
    return 0
  fi
  if [[ -n ${ev:-} ]]; then
    echo "MID used=${used}MiB ev=$(echo "$ev" | tr '\n' ',')"
    return 0
  fi
  echo "GAP used=${used}MiB"
}
