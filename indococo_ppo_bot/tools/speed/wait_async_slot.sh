#!/bin/bash
# Wait for a mid/gap window with the 2nd heavy slot free (no TM12 job).
# Usage: wait_async_slot.sh [timeout_sec]
# Does not kill anything. TM12 is detected via /proc cwd and cmdline files,
# never by pgrep -f.
set -euo pipefail
timeout_sec=${1:-7200}
# shellcheck source=/dev/null
source "$(dirname "$0")/lib_env.sh"

# Returns 0 if a TM12 heavy job is visible. Own PID is ignored.
tm12_heavy() {
  python3 - <<'PY'
import os, pathlib, sys
me = os.getpid()
ppid = os.getppid()
needles_cwd = ("code_tm12", "tm12_target")
needles_cmd = ("tm12_build.sh", "indococo-train-tm12")
for p in pathlib.Path("/proc").iterdir():
    if not p.name.isdigit():
        continue
    pid = int(p.name)
    if pid in (me, ppid):
        continue
    try:
        cwd = os.readlink(p / "cwd")
    except Exception:
        cwd = ""
    try:
        cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except Exception:
        cmd = ""
    if any(n in cwd for n in needles_cwd):
        print(f"TM12_HEAVY pid={pid} cwd={cwd} cmd={cmd[:160]}")
        sys.exit(0)
    # Require the trainer path to be the command start, or a build script path
    # as a real argv token, so an ssh/inspect line cannot false-positive.
    parts = cmd.split()
    if not parts:
        continue
    if parts[0].endswith("indococo-train-tm12") or "/indococo-train-tm12-" in parts[0]:
        print(f"TM12_HEAVY pid={pid} bin={parts[0]}")
        sys.exit(0)
    if any(x.endswith("tm12_build.sh") or x.endswith("/tm12_build.sh") for x in parts):
        print(f"TM12_HEAVY pid={pid} build={cmd[:160]}")
        sys.exit(0)
sys.exit(1)
PY
}

deadline=$(( $(date +%s) + timeout_sec ))
stable=0
need=8
echo "$(date -Is) slot-waiter start timeout=${timeout_sec}s"
while :; do
  now=$(date +%s)
  if (( now > deadline )); then
    echo "$(date -Is) TIMEOUT last=$(tm11_state)"
    exit 3
  fi
  st=$(tm11_state)
  kind=${st%% *}
  kind=${kind,,}
  heavy_msg=""
  if heavy_msg=$(tm12_heavy); then
    heavy=1
  else
    heavy=0
    heavy_msg="none"
  fi
  echo "$(date -Is) STATE=${kind^^} $st tm12=$heavy_msg"
  if [[ ( $kind == mid || $kind == gap ) && $heavy -eq 0 ]]; then
    stable=$((stable + 5))
    if (( stable >= need )); then
      echo "$(date -Is) READY $st tm12=none"
      exit 0
    fi
  else
    stable=0
  fi
  sleep 5
done
