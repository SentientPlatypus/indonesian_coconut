# Shared eval / chain helpers for future launchers (TM11 onward).
#
# New files only: nothing here is sourced by team_launch_tm10.sh, tm_TM10.env,
# tm10_window.sh, the ic_ppo audit script, or the live code_tm copy.
#
#   source experiments/lib/chain.sh
#   W=$X/mid/$now run_eval h2h_2v2 env INDOCOCO_TEAM_SIZE=2 "$B" h2h "$ck" "$BT" 200 8
#
# run_eval wraps the command in `timeout --kill-after=60 ${EVAL_TIMEOUT:-5400}`
# at nice 19, in its own session so a parent exit/kill can reap the process
# group by PID (never pkill -f). Keeps stderr in $W/<name>.err (not /dev/null),
# prints the last stdout line on success, and on failure prints `{}` and logs
# EVAL_FAIL.

# shellcheck shell=bash

run_eval() {
  local name=$1
  shift
  local tout=${EVAL_TIMEOUT:-5400}
  local dir=${W:-.}
  mkdir -p "$dir"
  local err=$dir/${name}.err
  local out tmp status e_was=0 pid="" saved_traps
  tmp=$(mktemp "${TMPDIR:-/tmp}/run_eval.XXXXXX") || { echo "EVAL_FAIL $name (mktemp)" >&2; echo '{}'; return 0; }

  saved_traps=$(trap -p EXIT INT TERM)
  [[ $- == *e* ]] && e_was=1
  set +e
  # New session: the timeout process is the session/process-group leader, so
  # kill -- -$pid cleans up its children if this shell is interrupted.
  # EVAL_TASKSET, when set (e.g. 24-31), pins the eval process group with
  # taskset. Default unset matches TM11 (no pin).
  if [[ -n ${EVAL_TASKSET:-} ]]; then
    setsid timeout --kill-after=60 "$tout" taskset -c "$EVAL_TASKSET" nice -n 19 "$@" >"$tmp" 2>"$err" &
  else
    setsid timeout --kill-after=60 "$tout" nice -n 19 "$@" >"$tmp" 2>"$err" &
  fi
  pid=$!
  # shellcheck disable=SC2064 # expand pid and tmp now, not at trap time
  trap 'if [[ -n "${pid:-}" ]]; then kill -- -"$pid" 2>/dev/null || true; fi; rm -f "'"$tmp"'"' EXIT INT TERM
  wait "$pid"
  status=$?
  pid=""
  trap - EXIT INT TERM
  if [[ -n $saved_traps ]]; then
    eval "$saved_traps"
  fi
  (( e_was )) && set -e

  if [[ $status -ne 0 ]]; then
    echo "EVAL_FAIL $name (exit $status, stderr $err)" >&2
    echo '{}'
    rm -f "$tmp"
    return 0
  fi
  out=$(tail -1 "$tmp")
  rm -f "$tmp"
  if [[ -z "$out" ]]; then
    echo "EVAL_FAIL $name (empty stdout, stderr $err)" >&2
    echo '{}'
    return 0
  fi
  printf '%s\n' "$out"
}
