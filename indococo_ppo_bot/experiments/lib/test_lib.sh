#!/usr/bin/env bash
# Unit tests for experiments/lib (run_eval timeout) and lint_launch.sh.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
EXP=$(cd "$HERE/.." && pwd)
fails=0
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# --- run_eval: a command that sleeps longer than the timeout returns {} quickly
# and writes $W/<name>.err plus EVAL_FAIL on stderr.
start=$(date +%s)
set +e
out=$(EVAL_TIMEOUT=1 W=$tmp bash -c 'source "'"$HERE"'/chain.sh"; run_eval sleep_test sleep 30' 2>"$tmp/log")
status=$?
set -e
elapsed=$(( $(date +%s) - start ))
if [[ $out == '{}' && $status -eq 0 && $elapsed -lt 10 && -f $tmp/sleep_test.err ]] && grep -q 'EVAL_FAIL sleep_test' "$tmp/log"; then
  echo "PASS run_eval-timeout (${elapsed}s)"
else
  echo "FAIL run_eval-timeout out=$out status=$status elapsed=$elapsed log=$(cat "$tmp/log") err=$(ls -l "$tmp/sleep_test.err" 2>&1)"
  fails=$((fails + 1))
fi

# --- run_eval success: last line of stdout, no EVAL_FAIL
printf '%s\n' 'first' '{"ok":1}' >"$tmp/echo_payload.txt"
set +e
out=$(W=$tmp EVAL_TIMEOUT=5 bash -c 'source "'"$HERE"'/chain.sh"; run_eval echo_test cat "'"$tmp"'/echo_payload.txt"' 2>"$tmp/log2")
status=$?
set -e
if [[ $out == '{"ok":1}' && $status -eq 0 ]] && ! grep -q EVAL_FAIL "$tmp/log2"; then
  echo "PASS run_eval-ok"
else
  echo "FAIL run_eval-ok out=$out status=$status log=$(cat "$tmp/log2")"
  fails=$((fails + 1))
fi

# --- lint: train_tm10.toml + tm_TM10.env should pass (epochs=1, no bf16, CHUNK 100M).
# Warnings (live TM10 binary, local numpy) are allowed.
set +e
"$EXP/lint_launch.sh" "$EXP/train_tm10.toml" "$EXP/tm_TM10.env" >"$tmp/lint_tm10.out"
status=$?
set -e
if [[ $status -eq 0 ]]; then
  echo "PASS lint-tm10"
else
  echo "FAIL lint-tm10 (expected pass):"
  cat "$tmp/lint_tm10.out"
  fails=$((fails + 1))
fi

# --- lint: train_st36p.toml has critic_bf16 = true -> must fail
set +e
"$EXP/lint_launch.sh" "$EXP/train_st36p.toml" "$EXP/st_ST36.env" >"$tmp/lint_st36.out"
status=$?
set -e
if [[ $status -ne 0 ]] && grep -q 'bf16' "$tmp/lint_st36.out"; then
  echo "PASS lint-st36p-bf16"
else
  echo "FAIL lint-st36p (expected non-zero exit mentioning bf16, got $status):"
  cat "$tmp/lint_st36.out"
  fails=$((fails + 1))
fi

# --- paths.env prepends $LIBTORCH_LIB and keeps an existing LD_LIBRARY_PATH
got=$(
  export LIBTORCH_LIB=/tmp/fake_torch_lib
  export LD_LIBRARY_PATH=/already/lib:/other
  export R=/tmp/fake_r
  # shellcheck source=paths.env
  source "$HERE/paths.env"
  printf '%s' "$LD_LIBRARY_PATH"
)
if [[ $got == /tmp/fake_torch_lib:/already/lib:/other ]]; then
  echo "PASS paths-prepend-existing"
else
  echo "FAIL paths-prepend-existing got=$got"
  fails=$((fails + 1))
fi
got=$(
  unset LD_LIBRARY_PATH
  export LIBTORCH_LIB=/tmp/fake_torch_lib
  export R=/tmp/fake_r
  # shellcheck source=paths.env
  source "$HERE/paths.env"
  printf '%s' "$LD_LIBRARY_PATH"
)
if [[ $got == /tmp/fake_torch_lib ]]; then
  echo "PASS paths-prepend-empty"
else
  echo "FAIL paths-prepend-empty got=$got"
  fails=$((fails + 1))
fi

# --- bash syntax
for s in "$HERE/chain.sh" "$HERE/paths.env" "$EXP/lint_launch.sh" "$EXP/tm10_vs_tm9_report.sh"; do
  if bash -n "$s"; then
    echo "PASS bash -n $(basename "$s")"
  else
    echo "FAIL bash -n $s"
    fails=$((fails + 1))
  fi
done

if [[ $fails -eq 0 ]]; then
  echo "RESULT: lib tests passed"
  exit 0
fi
echo "RESULT: $fails lib test(s) failed"
exit 1
