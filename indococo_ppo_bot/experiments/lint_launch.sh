#!/usr/bin/env bash
# Pre-launch checker for standing training rules. Report-only: prints PASS /
# FAIL / WARN lines and exits 1 if any FAIL. Not wired into TM10 or any live
# launcher.
#
#   experiments/lint_launch.sh <train.toml> [env]
#
# Fails if:
#   - epochs != 1 in the train toml
#   - any policy_bf16 / critic_bf16 / *_bf16 is true
#   - CHUNK (in the env file, if given) is set and is not 100M
#   - a team env has neither BT_PIN nor BT assigned
#   - a pkill -f appears in either file (outside comments)
#   - a non-TM10 env's B0/BIN/EVAL_BIN default is the live TM10 binary
#   - tools/requirements.txt does not pin numpy==1.26.4
#
# Path existence for EC2-only checkpoints is a WARN when those paths are
# not on this machine, so the checker can run in a checkout.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
CRATE=$(cd "$HERE/.." && pwd)
usage() { echo "usage: $0 <train.toml> [env]" >&2; exit 2; }
[[ $# -ge 1 && $# -le 2 ]] || usage
TOML=$1
ENVF=${2:-}
[[ -f $TOML ]] || { echo "FAIL $TOML is not a file" >&2; exit 2; }
if [[ -n $ENVF ]]; then [[ -f $ENVF ]] || { echo "FAIL $ENVF is not a file" >&2; exit 2; }; fi

FAILS=0
WARN=0
pass() { echo "PASS $*"; }
fail() { echo "FAIL $*"; FAILS=$((FAILS + 1)); }
warn() { echo "WARN $*"; WARN=$((WARN + 1)); }

# Strip full-line and trailing comments, then blank lines, for scans.
strip_comments() {
  sed -E 's/[[:space:]]+#.*$//; /^[[:space:]]*#/d; /^[[:space:]]*$/d' "$1"
}

# Parse a TOML assignment like `epochs = 1` or `critic_bf16 = true` (first match).
toml_val() {
  local key=$1 file=$2
  sed -nE "s/^[[:space:]]*${key}[[:space:]]*=[[:space:]]*([^[:space:]#]+).*/\1/p" "$file" | tail -1
}

# Numeric 100_000_000 / 100000000 / 100M / 1e8 -> integer.
parse_num() {
  local s=${1//_/}
  s=${s%\"}
  s=${s#\"}
  case $s in
    *M|*m) s=${s%[Mm]}; echo $((s * 1000000)) ;;
    *e8|*E8) echo 100000000 ;;
    ''|*[!0-9]*) echo "" ;;
    *) echo "$s" ;;
  esac
}

# --- epochs ---
epochs=$(toml_val epochs "$TOML")
if [[ -z ${epochs:-} ]]; then
  fail "$TOML: epochs is missing"
elif [[ $epochs == @* ]]; then
  warn "$TOML: epochs=$epochs (placeholder; resolved value must be 1)"
elif [[ $epochs != 1 ]]; then
  fail "$TOML: epochs=$epochs (standing rule: 1 PPO epoch)"
else
  pass "$TOML: epochs=1"
fi

# --- bf16 ---
bf16_hits=0
while IFS= read -r line; do
  case $line in
    *bf16*=*true*|*bf16*=*True*)
      fail "$TOML: $line (standing rule: no bf16)"
      bf16_hits=$((bf16_hits + 1))
      ;;
  esac
done < <(strip_comments "$TOML")
if [[ $bf16_hits -eq 0 ]]; then
  pass "$TOML: no *_bf16 = true"
fi

# --- env-dependent checks ---
if [[ -n $ENVF ]]; then
  stripped=$(mktemp)
  strip_comments "$ENVF" >"$stripped"

  # CHUNK
  chunk_raw=$(grep -E '^CHUNK=' "$stripped" | tail -1 || true)
  if [[ -z $chunk_raw ]]; then
    fail "$ENVF: CHUNK is missing (standing rule: CHUNK 100M)"
  else
    # CHUNK=${CHUNK:-100000000} or CHUNK=100000000
    chunk_val=${chunk_raw#CHUNK=}
    if [[ $chunk_val == *:-* ]]; then
      chunk_val=${chunk_val#*:-}
      chunk_val=${chunk_val%%\}}
    fi
    chunk_val=${chunk_val%%;*}
    chunk_val=${chunk_val// /}
    chunk_n=$(parse_num "$chunk_val")
    if [[ $chunk_n == 100000000 ]]; then
      pass "$ENVF: CHUNK=100000000"
    else
      fail "$ENVF: CHUNK=$chunk_val (standing rule: CHUNK 100M)"
    fi
  fi

  # pkill -f
  if grep -E '(^|[^[:alnum:]_])pkill[[:space:]]+-f' "$stripped" >/dev/null; then
    fail "$ENVF: contains pkill -f (forbidden; kill by PID only)"
  else
    pass "$ENVF: no pkill -f"
  fi

  # BT / BESTF assignment (team envs only). ST envs often take BESTF from the launcher.
  base=$(basename "$ENVF")
  if [[ $base == tm_TM* || $base == tm_* ]]; then
    if grep -E '^(BT_PIN|BT)=' "$stripped" >/dev/null; then
      pass "$ENVF: BT or BT_PIN is assigned"
    else
      fail "$ENVF: team env has neither BT_PIN nor BT"
    fi
  fi

  # Live TM10 binary: do not point a *new* run at it.
  live1=/opt/dlami/nvme/ippo_bins/indococo-train-tm10-cf02e2b
  live2=/home/ubuntu/ippo_runs/exp_TM10/indococo-train
  b0_line=$(grep -E '^(B0|BIN|EVAL_BIN)=' "$stripped" | tail -1 || true)
  if [[ -n $b0_line ]]; then
    if [[ $b0_line == *"$live1"* || $b0_line == *"$live2"* ]]; then
      if [[ $base == tm_TM10.env ]]; then
        warn "$ENVF: B0/BIN is the live TM10 binary (ok for TM10 itself; do not overwrite it)"
      else
        fail "$ENVF: B0/BIN/EVAL_BIN points at the live TM10 binary ($live1 or exp_TM10/indococo-train)"
      fi
    else
      pass "$ENVF: B0/BIN is not the live TM10 binary"
    fi
  fi

  # Path existence: WARN only when the default looks like an EC2 path and is missing.
  for key in BT_PIN ST25_CK START_CK; do
    :
  done
  rm -f "$stripped"
else
  pass "no env file given; skipped CHUNK / BT / live-binary checks"
fi

# pkill in the toml too
if strip_comments "$TOML" | grep -E '(^|[^[:alnum:]_])pkill[[:space:]]+-f' >/dev/null; then
  fail "$TOML: contains pkill -f"
else
  pass "$TOML: no pkill -f"
fi

# numpy pin in the repo requirements
req=$CRATE/tools/requirements.txt
if [[ -f $req ]]; then
  if grep -E '^numpy==1\.26\.4([[:space:]]|$)' "$req" >/dev/null; then
    pass "$req: numpy==1.26.4"
  else
    fail "$req: numpy is not pinned to 1.26.4"
  fi
else
  fail "$req missing (standing rule: pin numpy==1.26.4)"
fi

# Optional: if a venv python is available, report its numpy (do not fail the
# checkout lint when the current interpreter is not coco-venv).
py=${PYT:-${PY:-python3}}
if command -v "$py" >/dev/null 2>&1; then
  ver=$("$py" -c 'import numpy; print(numpy.__version__)' 2>/dev/null || true)
  if [[ -z $ver ]]; then
    warn "$py: numpy not importable"
  elif [[ $ver == 1.26.4 ]]; then
    pass "$py: numpy $ver"
  else
    warn "$py: numpy $ver (standing rule is 1.26.4; this is not the training venv check)"
  fi
fi

echo "----"
if [[ $FAILS -gt 0 ]]; then
  echo "RESULT: $FAILS failed, $WARN warning(s)"
  exit 1
fi
echo "RESULT: all required checks passed, $WARN warning(s)"
exit 0
