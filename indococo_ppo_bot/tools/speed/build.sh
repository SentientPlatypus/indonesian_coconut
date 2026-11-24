#!/bin/bash
# shellcheck disable=SC1091  # fr9_env.sh exists on the trainer host only
# Build indococo-train into a uniquely named speed binary. Never overwrites.
# Modeled on /opt/dlami/nvme/speed_757cbdf_build.sh.
set -euo pipefail
W=/opt/dlami/nvme/speedup_work
source /home/ubuntu/ippo_runs/fr9_env.sh >/dev/null
# fr9_env.sh cds and sets CARGO_TARGET_DIR — override both.
export RIPPLE_CUDA_INCLUDE=/opt/dlami/nvme/torch29/lib/python3.12/site-packages/triton/backends/nvidia/include
export CARGO_TARGET_DIR=$W/target
export TMPDIR=$W/tmp TMP=$W/tmp TEMP=$W/tmp
export XDG_CACHE_HOME=$W/cache
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 RAYON_NUM_THREADS=2
unset DEP_TORCH_SYS_LIBTORCH_LIB CPLUS_INCLUDE_PATH CUDA_HOME CUDA_PATH
mkdir -p "$CARGO_TARGET_DIR" "$TMPDIR" "$XDG_CACHE_HOME" "$W/logs"

SRC=${SPEEDUP_SRC:-$W/repo}
cd "$SRC/indococo_ppo_bot"
if git rev-parse HEAD >/dev/null 2>&1; then
  HEAD=$(git rev-parse HEAD)
  SHA=$(git rev-parse --short=9 HEAD)
  dirty=$(git status --porcelain | tr "\n" ";")
elif [ -f "$W/SHA" ]; then
  HEAD=$(tr -d '[:space:]' < "$W/SHA")
  SHA=${HEAD:0:9}
  dirty="no-git-checkout"
else
  echo "NO_SHA: provide a git checkout or $W/SHA"
  exit 9
fi
echo "==== BEGIN $(date -Is) ===="
echo "HEAD=$HEAD"
echo "short=$SHA"
echo "dirty=$dirty"
echo "rustc=$(rustc --version) cargo=$(cargo --version)"
echo "CARGO_TARGET_DIR=$CARGO_TARGET_DIR"
echo "pwd=$(pwd)"
echo "LIBTORCH=$LIBTORCH"

echo "==== CARGO TEST LIB (GPU hidden) $(date -Is) ===="
export CUDA_VISIBLE_DEVICES=""
nice -n 19 cargo test --locked --release --lib --features torch,ripple -j4 --no-fail-fast
echo "==== CARGO TEST LIB EXIT=$? $(date -Is) ===="

echo "==== CARGO TEST BIN (GPU hidden) $(date -Is) ===="
nice -n 19 cargo test --locked --release --bin indococo-train --features torch,ripple -j4 --no-fail-fast
echo "==== CARGO TEST BIN EXIT=$? $(date -Is) ===="

echo "==== CARGO BUILD BIN (GPU hidden) $(date -Is) ===="
nice -n 19 cargo build --locked --release --features torch,ripple -j4 --bin indococo-train
echo "==== CARGO BUILD EXIT=$? $(date -Is) ===="

BIN=$CARGO_TARGET_DIR/release/indococo-train
ls -la "$BIN"
DEST=/opt/dlami/nvme/ippo_bins/indococo-train-speed-$SHA
if [ -e "$DEST" ]; then
  DEST=${DEST}-$(date +%Y%m%d%H%M%S)
  echo "name already existed, using $DEST"
fi
cp -a "$BIN" "$DEST.tmp"
chmod 755 "$DEST.tmp"
mv -f "$DEST.tmp" "$DEST"
md5sum "$BIN" "$DEST"
ls -la "$DEST"
echo "BIN=$DEST"
echo "==== ALL_DONE $(date -Is) ===="
