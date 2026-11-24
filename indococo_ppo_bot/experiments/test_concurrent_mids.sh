#!/usr/bin/env bash
# T0 render identity and T1 stub-logic tests for team_launch_next.sh.
# CPU-light, no GPU. Does not touch TM11, team_best, or live launchers.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
C_REPO=$(cd "$HERE/.." && pwd)
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1: $2"; fails=$((fails + 1)); }

PYT=${PYT:-$(command -v python3)}
TMP=$(mktemp -d "${TMPDIR:-/tmp}/mids_test.XXXXXX")
# Leave TMP in place on failure for inspection; remove only on full pass at the end.
cleanup_ok=0
trap 'if [[ $cleanup_ok == 1 ]]; then rm -rf "$TMP"; fi' EXIT

# ---------- helpers ----------
make_ck() {
  local dir=$1 steps=$2
  mkdir -p "$dir"
  printf 'dummy-actor\n' > "$dir/actor.bpk"
  printf 'dummy-critic\n' > "$dir/critic.bpk"
  printf 'cumulative_timesteps = %s\ncumulative_model_updates = 0\ncumulative_epochs = 0\ntransfer_rounds = 0\n\n[return_stat]\nmean = 0.0\nmean_sqr = 0.0\ncount = 0\n' "$steps" > "$dir/stats.toml"
}

# Canned h2h JSON. $1=games $2=win_rate_decisive $3=passes_min etc via env or defaults.
canned_h2h() {
  local games=$1 wr=$2
  local wins losses
  wins=$(awk -v g="$games" -v r="$wr" 'BEGIN{printf "%d", int(g*r+0.5)}')
  losses=$((games - wins))
  local passes=${3:-2.0} deep=${4:-0.3} near=${5:-0.2} touch=${6:-2.0} depth=${7:-2000}
  cat <<JSON
{"games": $games, "wins": $wins, "losses": $losses, "draws": 0, "win_rate_decisive": $wr, "seconds": 1.0, "team": {"a": {"passes_per_min": $passes, "double_commit_ratio": 0.1, "pair_lt800_ratio": 0.1, "two_near_ball1500_ratio": $near, "double_commit_wide_ratio": 0.1, "deep_last_man_ratio": $deep, "no_challenger_ratio": 0.01, "role_swaps_per_min": 0.1, "touch_share_first": 0.5, "first_mean_stint_s": 4.0, "touches_per_car_min": $touch, "deepest_car_depth_uu": $depth, "demos_per_min": 0.1, "bumps_per_min": 0.1}}}
JSON
}

canned_wr() {
  local games=$1 wr=$2
  local wins losses
  wins=$(awk -v g="$games" -v r="$wr" 'BEGIN{printf "%d", int(g*r+0.5)}')
  losses=$((games - wins))
  printf '{"games": %s, "wins": %s, "losses": %s, "draws": 0, "win_rate_decisive": %s, "seconds": 1.0}\n' "$games" "$wins" "$losses" "$wr"
}

# Write a full canned mid table under $1 for steps $2, with h2h rate $3 (and optional low flag).
write_canned_mid() {
  local dest=$1 steps=$2 h2h=$3
  mkdir -p "$dest/$steps"
  canned_h2h 40 "$h2h" > "$dest/$steps/h2h_2v2.json"
  canned_h2h 40 "$h2h" > "$dest/$steps/h2h_3v3.json"
  canned_wr 40 0.6 > "$dest/$steps/gal_2v2.json"
  canned_wr 40 0.55 > "$dest/$steps/gal_3v3.json"
  canned_wr 40 0.5 > "$dest/$steps/gal_1v1.json"
  canned_wr 24 0.7 > "$dest/$steps/mega_2v2.json"
  canned_wr 40 0.6 > "$dest/$steps/nexto_2v2.json"
  canned_wr 40 0.5 > "$dest/$steps/nexto_3v3.json"
  canned_wr 24 0.95 > "$dest/$steps/necto_2v2.json"
  canned_wr 24 0.95 > "$dest/$steps/necto_3v3.json"
  canned_wr 40 0.5 > "$dest/$steps/h2h_1v1_st25.json"
}

# Train stub: sleep S seconds, then add CHUNK_T to the latest checkpoint's steps
# and write a new numbered checkpoint dir (epoch-like, unique).
write_train_stub() {
  local path=$1
  cat > "$path" << 'STUB'
#!/usr/bin/env bash
set -u
# X and CHUNK_T and STUB_SLEEP are exported by the test.
sleep "${STUB_SLEEP:-0}"
latest=$(ls "$X/checkpoints" | grep -E '^[0-9]+$' | sort -n | tail -1)
prev=$X/checkpoints/$latest
now=$(sed -nE 's/^cumulative_timesteps = ([0-9]+)/\1/p' "$prev/stats.toml")
next=$((now + CHUNK_T))
# Same naming as the real trainer (unix epoch). Must sort numerically after
# a guard rollback dir, which is also date +%s. A 1XXXXXXXXX name would
# sort *before* a rollback timestamp and look like steps did not advance.
base=$(date +%s)
nd=$X/checkpoints/$base
while [[ -d $nd ]]; do base=$((base + 1)); nd=$X/checkpoints/$base; done
mkdir -p "$nd"
cp "$prev/actor.bpk" "$prev/critic.bpk" "$nd/"
sed -E "s/^cumulative_timesteps = .*/cumulative_timesteps = $next/" "$prev/stats.toml" > "$nd/stats.toml"
exit 0
STUB
  chmod +x "$path"
}

# Seed an experiment dir so the launcher skips widen and starts at $steps.
seed_exp() {
  local x=$1 c=$2 steps=$3 stub=$4
  mkdir -p "$x/logs" "$x/checkpoints" "$x/opponents" "$x/cfg" "$x/seed" "$x/mid"
  rm -rf "$x/cfg/experiments" "$x/cfg/config"
  cp -a "$c/experiments" "$x/cfg/experiments"
  mkdir -p "$x/cfg/config"
  if [[ -d $c/config ]]; then cp -a "$c/config/." "$x/cfg/config/"; fi
  make_ck "$x/checkpoints/1000" "$steps"
  make_ck "$x/seed" "$steps"
  echo "$x/seed $steps ${VARIANT:-TM11}" > "$x/START"
  echo LOW=0 > "$x/guard.env"
  cp "$stub" "$x/indococo-train"
  chmod +x "$x/indococo-train"
}

common_env() {
  # Usage: common_env writes exports into the current shell given X, C, stub, BT.
  export R="$R_ROOT"
  export RUN
  export VARIANT=TM11
  export C
  export X
  export B0="$STUB"
  export B="$X/indococo-train"
  export PYT
  export MESH_REPO="$TMP/mesh"
  export BT_PIN="$BT"
  export BT="$BT"
  export ST25_CK="$TMP/no_st25"
  export SKIP_PREFLIGHT=1
  export CHUNK="$CHUNK_T"
  export STEPS
  export TPI=1000
  export MB=1000
  export LR=5e-5
  export ENT=0.016
  export GAMMA=0.995
  export OPPCH=0.50
  export PASTW=1.0
  export HOLDREL=4000000000
  export DECAY=1000000000
  export MINLR=0.4
  export WARM_REL=0
  export THREADS=2
  export GAMES=2
  export MID_GAMES=40
  export GAL_GAMES=40
  export GAL1_GAMES=40
  export MEGA_GAMES=24
  export NEXTO_MID=40
  export NECTO_MID=24
  export EVAL_STUB=1
  export EVAL_STUB_DIR
  export EVAL_STUB_SLEEP
  export STUB_SLEEP
  export CHUNK_T
  export GAL_FIDELITY=0
  export INDOCOCO_OBS_PRED=0
  unset GAL_WEIGHT MEGA_WEIGHT NEXTO_WEIGHT NECTO_WEIGHT EXTRA_OPPS RIPPLE_WEIGHT
}

mkdir -p "$TMP/mesh"
R_ROOT=$TMP/runs
mkdir -p "$R_ROOT"
BT=$TMP/bt
make_ck "$BT" 0
STUB=$TMP/train_stub.sh
write_train_stub "$STUB"

# =====================================================================
# T0: render identity, mode 0, TM11 env values, vs a DRY_RENDER-injected
# copy of team_launch_tm11.sh (the repo file is not edited).
# =====================================================================
t0() {
  local steps=57957300000
  CHUNK_T=100000000
  STEPS=100000000
  RUN=T0NEXT
  C=$C_REPO
  X=$R_ROOT/exp_$RUN
  seed_exp "$X" "$C" "$steps" "$STUB"
  EVAL_STUB_DIR=$TMP/canned_t0
  EVAL_STUB_SLEEP=0
  STUB_SLEEP=0
  common_env
  # Use the real TM11 reward/train templates and weights for a meaningful diff.
  # Re-source TM11 env into a subshell-like export of the render-relevant vars,
  # but keep our paths. We'll manually set the TM11 template names and weights.
  export REWARD_TMPL=tm11.toml
  export TRAIN_TMPL=train_tm11.toml
  export START_IS_TEAM=1
  export OBS_PRED=1
  export GAL_WEIGHT=0.21
  export NEXTO_WEIGHT=0.30
  export NEXTO_STOCH=0.2
  export GAL_DIR=/home/ubuntu/eval_bots/galactus/rust
  export NEXTO_DIR=/home/ubuntu/eval_bots/nexto
  # Need cfg to include train_tm11.toml (copied from C).
  set +e
  DRY_RENDER=1 CONCURRENT_MIDS=0 bash "$C/experiments/team_launch_next.sh" >"$TMP/t0_next.log" 2>&1
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    fail t0-next-render "exit $rc log=$(tail -20 $TMP/t0_next.log)"
    return
  fi

  # Temp tree used only to hold a regular-file *copy* of the TM11 launcher
  # with a DRY_RENDER hook. Never symlink team_launch_tm11.sh — writing
  # through that symlink would overwrite the official launcher.
  local C_TM11=$TMP/c_tm11
  mkdir -p "$C_TM11/experiments"
  for f in "$C_REPO"/experiments/*; do
    local base
    base=$(basename "$f")
    [[ $base == team_launch_tm11.sh ]] && continue
    ln -sfn "$f" "$C_TM11/experiments/$base"
  done
  cp -a "$C_REPO/experiments/team_launch_tm11.sh" "$C_TM11/experiments/team_launch_tm11.sh"
  # Build the patched tm11 launcher in Python more carefully:
  python3 - "$C_REPO/experiments/team_launch_tm11.sh" "$C_TM11/experiments/team_launch_tm11.sh" << 'PY'
from pathlib import Path
import sys
src = Path(sys.argv[1]).read_text()
# Take the DRY_RENDER block from team_launch_next.sh between the two markers
# by reconstructing the same logic as the while-loop render + exit.
hook = r'''source $X/guard.env
if [[ "${DRY_RENDER:-0}" == 1 ]]; then
  now=$(steps_in $X/checkpoints/$(latest))
  stop=$(( now + CHUNK )); (( stop > END )) && stop=$END
  left=$(( (stop - now + ${TPI//_/} - 1) / ${TPI//_/} ))
  sed -e "s|@RUN@|$X|g" -e "s|@ITERATIONS@|$left|" -e "s|@WARM@|$((START0 + ${WARM_REL:-20000000}))|" -e "s|@HOLD@|$((START0 + HOLDREL))|" \
      -e "s|@SNAP@|$SNAP/train.f32|" -e "s|@TPI@|$TPI|" -e "s|@MB@|$MB|" -e "s|@LR@|$LR|g" -e "s|@ENT@|$ENT|" -e "s|@GAMMA@|$GAMMA|" \
      -e "s|@OPPCH@|$OPPCH|" -e "s|@PASTW@|${PASTW:-1.0}|" -e "s|@DECAY@|$DECAY|" -e "s|@MINLR@|$MINLR|" -e "s|^threads = .*|threads = $THREADS|" -e "s|^games = .*|games = ${GAMES:-384}|" \
      $X/cfg/experiments/${TRAIN_TMPL:-train_tm1.toml} > $X/train_tm.toml
  [[ -n "${GAL_WEIGHT:-}" ]] && printf '\n[[opponents.external]]\nname = "galactus"\nkind = "galactus"\npath = "%s"\nweight = %s\n' "$GAL_DIR" "$GAL_WEIGHT" >> $X/train_tm.toml
  [[ -n "${MEGA_WEIGHT:-}" ]] && printf '\n[[opponents.external]]\nname = "megabyte"\nkind = "megabyte"\npath = "%s"\nweight = %s\n' "$MEGA_DIR" "$MEGA_WEIGHT" >> $X/train_tm.toml
  [[ -n "${NEXTO_WEIGHT:-}" ]] && printf '\n[[opponents.external]]\nname = "nexto"\nkind = "nexto"\npath = "%s"\nweight = %s\nstochastic = %s\n' "$NEXTO_DIR" "$NEXTO_WEIGHT" "${NEXTO_STOCH:-0.0}" >> $X/train_tm.toml
  [[ -n "${NECTO_WEIGHT:-}" ]] && printf '\n[[opponents.external]]\nname = "necto"\nkind = "necto"\npath = "%s"\nweight = %s\nstochastic = %s\n' "$NECTO_DIR" "$NECTO_WEIGHT" "${NECTO_STOCH:-0.0}" >> $X/train_tm.toml
  for e in ${EXTRA_OPPS:-}; do n=${e%%:*}
    printf '\n[[opponents.fixed]]\nname = "%s"\ncheckpoint = "%s"\nweight = %s\n' "$n" "$X/opponents/$n" "${e##*:}" >> $X/train_tm.toml; done
  log "$RUN DRY_RENDER wrote $X/train_tm.toml and $X/tm_rewards.toml"
  exit 0
fi
while :; do
'''
old = "source $X/guard.env\nwhile :; do\n"
if old not in src:
    raise SystemExit("tm11 hook point missing")
src = src.replace(old, hook, 1)
selfchk = '''[[ $SELF == "$(readlink -f "$C")/experiments/team_launch_tm11.sh" ]] || { echo "FATAL: must run $C/experiments/team_launch_tm11.sh (got $SELF)" >&2; exit 1; }'''
loose = '''[[ $SELF == *"/experiments/team_launch_tm11.sh" ]] || { echo "FATAL: must run a team_launch_tm11.sh (got $SELF)" >&2; exit 1; }'''
if selfchk not in src:
    raise SystemExit("tm11 self-check missing")
src = src.replace(selfchk, loose, 1)
# Official tm11 always calls preflight (numpy 1.26.4, lint, ...). The temp
# copy skips it under DRY_RENDER or SKIP_PREFLIGHT so T0 can run on the box.
needle = "\npreflight\n"
if needle not in src:
    raise SystemExit("tm11 preflight call missing")
src = src.replace(needle, "\nif [[ \"${SKIP_PREFLIGHT:-0}\" != 1 && \"${DRY_RENDER:-0}\" != 1 ]]; then\n  preflight\nfi\n", 1)
Path(sys.argv[2]).write_text(src)
PY
  chmod +x "$C_TM11/experiments/team_launch_tm11.sh"
  local X2=$R_ROOT/exp_T0TM11
  RUN=T0TM11
  X=$X2
  seed_exp "$X" "$C_REPO" "$steps" "$STUB"
  # cfg must still have the templates; seed_exp copied from C_REPO, good.
  export X RUN
  set +e
  DRY_RENDER=1 SKIP_PREFLIGHT=1 C=$C_REPO RUN=T0TM11 VARIANT=TM11 \
    bash "$C_TM11/experiments/team_launch_tm11.sh" >"$TMP/t0_tm11.log" 2>&1
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    fail t0-tm11-render "exit $rc log=$(tail -20 $TMP/t0_tm11.log)"
    return
  fi
  # Path-substitute the two X roots and compare.
  local a=$TMP/t0_next_train.toml b=$TMP/t0_tm11_train.toml
  sed "s|$R_ROOT/exp_T0NEXT|__X__|g; s|$C_REPO|__C__|g" "$R_ROOT/exp_T0NEXT/train_tm.toml" > "$a"
  sed "s|$R_ROOT/exp_T0TM11|__X__|g; s|$C_REPO|__C__|g" "$X2/train_tm.toml" > "$b"
  local ra=$TMP/t0_next_rew.toml rb=$TMP/t0_tm11_rew.toml
  sed "s|$R_ROOT/exp_T0NEXT|__X__|g; s|$C_REPO|__C__|g" "$R_ROOT/exp_T0NEXT/tm_rewards.toml" > "$ra"
  sed "s|$R_ROOT/exp_T0TM11|__X__|g; s|$C_REPO|__C__|g" "$X2/tm_rewards.toml" > "$rb"
  if diff -u "$b" "$a" >/tmp/t0_train.diff && diff -u "$rb" "$ra" >/tmp/t0_rew.diff; then
    pass t0-render-identity
  else
    fail t0-render-identity "train or rewards differ (see /tmp/t0_*.diff)"
    tail -40 /tmp/t0_train.diff /tmp/t0_rew.diff
  fi
}

# T0b: NF_INTRA=2 ASYNC_PRIO=1 keys land at top-level (before the first table),
# not inside the last [[opponents.*]] table. Nexto gets replicas/chunk_rows;
# Necto does not. Defaults (no env) stay inert — covered by t0 identity.
t0_speed_keys() {
  local steps=57957300000
  CHUNK_T=100000000
  STEPS=100000000
  RUN=T0SPEED
  C=$C_REPO
  X=$R_ROOT/exp_$RUN
  seed_exp "$X" "$C" "$steps" "$STUB"
  EVAL_STUB_DIR=$TMP/canned_t0speed
  EVAL_STUB_SLEEP=0
  STUB_SLEEP=0
  common_env
  export REWARD_TMPL=tm11.toml
  export TRAIN_TMPL=train_tm11.toml
  export START_IS_TEAM=1
  export OBS_PRED=1
  export GAL_WEIGHT=0.21
  export NEXTO_WEIGHT=0.30
  export NEXTO_STOCH=0.2
  export GAL_DIR=/home/ubuntu/eval_bots/galactus/rust
  export NEXTO_DIR=/home/ubuntu/eval_bots/nexto
  export NF_INTRA=2
  export ASYNC_PRIO=1
  export NEXTO_REPLICAS=8
  export NEXTO_CHUNK=64
  set +e
  DRY_RENDER=1 CONCURRENT_MIDS=0 bash "$C/experiments/team_launch_next.sh" >"$TMP/t0_speed.log" 2>&1
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    unset NF_INTRA ASYNC_PRIO ASYNC NEXTO_REPLICAS NEXTO_CHUNK
    fail t0-speed-render "exit $rc log=$(tail -20 $TMP/t0_speed.log)"
    return
  fi
  local toml=$X/train_tm.toml
  set +e
  "$PYT" - "$toml" << 'PY'
import sys, pathlib
try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

path = pathlib.Path(sys.argv[1])
text = path.read_text()
data = tomllib.loads(text)

def fail(msg):
    print("TOML_FAIL", msg)
    sys.exit(1)

for key, want in (("nf_intra_threads", 2), ("async_collect", True), ("collect_stream_priority", True)):
    if key not in data:
        fail(f"missing top-level {key}")
    if data[key] != want:
        fail(f"{key}={data[key]!r} want {want!r}")

first = None
for line in text.splitlines():
    if line.startswith("["):
        first = line
        break
if first is None:
    fail("no table found")
table_pos = text.find(first)
for key in ("nf_intra_threads", "async_collect", "collect_stream_priority"):
    pos = text.find(f"{key} =")
    if pos < 0 or pos > table_pos:
        fail(f"{key} is not before first table {first}")

externals = data.get("opponents", {}).get("external", [])
if not externals:
    fail("no opponents.external")
for ev in externals:
    for key in ("nf_intra_threads", "async_collect", "collect_stream_priority"):
        if key in ev:
            fail(f"{key} leaked into opponents.external name={ev.get('name')}")

by_name = {ev["name"]: ev for ev in externals}
if "nexto" not in by_name:
    fail("nexto missing")
if by_name["nexto"].get("replicas") != 8 or by_name["nexto"].get("chunk_rows") != 64:
    fail(f"nexto replicas/chunk_rows = {by_name['nexto']}")
print("TOML_OK first_table", first, "externals", list(by_name))
PY
  local prc=$?
  set -e
  if [[ $prc -eq 0 ]]; then
    pass t0-speed-keys-top-level
  else
    fail t0-speed-keys-top-level "see $toml log=$(tail -5 $TMP/t0_speed.log)"
  fi

  # Isolated fixture: necto is absent from tm_TM11.env, so exercise both tables
  # against the same functions the launcher uses.
  local fns=$TMP/append_fns.sh
  "$PYT" - "$C/experiments/team_launch_next.sh" "$fns" << 'PY'
from pathlib import Path
import sys
src = Path(sys.argv[1]).read_text().splitlines()
out = []
grab = False
for line in src:
    if line.startswith("append_speed_toml()"):
        grab = True
    if grab:
        out.append(line)
    if grab and line.startswith("append_nf_block()"):
        # include the body until the closing brace at column 0
        continue
    if grab and line == "}" and "append_nf_block" in "\n".join(out):
        # last closing of append_nf_block
        if out[-1] == "}" and any(l.startswith("append_nf_block()") for l in out):
            # stop after this closing if we have both functions
            nclose = sum(1 for l in out if l == "}")
            nfn = sum(1 for l in out if l.startswith("append_"))
            if nclose >= 2 and nfn >= 2:
                break
Path(sys.argv[2]).write_text("\n".join(out) + "\n")
PY
  # shellcheck source=/dev/null
  source "$fns"
  local fix=$TMP/nf_fixture.toml
  cat > "$fix" << 'EOF'
epochs = 1
policy_bf16 = false
critic_bf16 = false

[[opponents.external]]
name = "nexto"
kind = "nexto"
path = "/tmp/nexto"
weight = 0.3
stochastic = 0.2
EOF
  NEXTO_REPLICAS=8 NEXTO_CHUNK=64 append_nf_block "$fix"
  cat >> "$fix" << 'EOF'

[[opponents.external]]
name = "necto"
kind = "necto"
path = "/tmp/necto"
weight = 0.1
stochastic = 0.0
EOF
  NF_INTRA=2 ASYNC_PRIO=1 append_speed_toml "$fix"
  set +e
  "$PYT" - "$fix" << 'PY'
import sys, pathlib
try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore
text = pathlib.Path(sys.argv[1]).read_text()
data = tomllib.loads(text)
assert data["nf_intra_threads"] == 2 and data["async_collect"] is True
first = next(l for l in text.splitlines() if l.startswith("["))
assert text.find("nf_intra_threads =") < text.find(first)
by = {e["name"]: e for e in data["opponents"]["external"]}
assert by["nexto"]["replicas"] == 8 and by["nexto"]["chunk_rows"] == 64
assert "replicas" not in by["necto"] and "chunk_rows" not in by["necto"]
for e in by.values():
    assert "nf_intra_threads" not in e
print("FIXTURE_OK")
PY
  local frc=$?
  set -e
  unset NF_INTRA ASYNC_PRIO ASYNC NEXTO_REPLICAS NEXTO_CHUNK
  if [[ $frc -eq 0 ]]; then
    pass t0-nf-block-nexto-only
  else
    fail t0-nf-block-nexto-only "fixture=$(cat $fix)"
  fi
}


# Run one stubbed launcher. Args: name mode S E h2h_rate chunks
run_stub_case() {
  local name=$1 mode=$2 s=$3 e=$4 h2h=$5 chunks=$6
  RUN=$name
  C=$C_REPO
  X=$R_ROOT/exp_$RUN
  local start=1000000
  CHUNK_T=100000
  STEPS=$((chunks * CHUNK_T))
  seed_exp "$X" "$C" "$start" "$STUB"
  EVAL_STUB_DIR=$TMP/canned_$name
  mkdir -p "$EVAL_STUB_DIR"
  local i steps
  for i in $(seq 1 "$chunks"); do
    steps=$((start + i * CHUNK_T))
    if [[ $h2h == LOW && $i -le 2 ]]; then
      write_canned_mid "$EVAL_STUB_DIR" "$steps" 0.30
    else
      write_canned_mid "$EVAL_STUB_DIR" "$steps" 0.60
    fi
  done
  EVAL_STUB_SLEEP=$e
  STUB_SLEEP=$s
  common_env
  export REWARD_TMPL=tm11.toml TRAIN_TMPL=train_tm11.toml START_IS_TEAM=1 OBS_PRED=0
  unset GAL_WEIGHT NEXTO_WEIGHT MEGA_WEIGHT NECTO_WEIGHT EXTRA_OPPS
  set +e
  CONCURRENT_MIDS=$mode bash "$C/experiments/team_launch_next.sh" >"$X/logs/case.log" 2>&1
  local rc=$?
  set -e
  echo $rc
}

t0
t0_speed_keys

# T1(a): E < S, mode 1: every MID_WAIT is 0; mid.jsonl identical to mode 0.
# 2 chunks, S=2, E=0.
echo "=== T1a E<S ==="
rc0=$(run_stub_case T1A0 0 2 0 0.60 2)
rc1=$(run_stub_case T1A1 1 2 0 0.60 2)
if [[ $rc0 -eq 0 && $rc1 -eq 0 ]]; then
  # mid.jsonl: compare after substituting ckpt paths and X
  sed -E "s|$R_ROOT/exp_T1A0|__X__|g; s|/checkpoints/[0-9]+|/checkpoints/__CK__|g" "$R_ROOT/exp_T1A0/logs/mid.jsonl" > "$TMP/t1a0.jsonl"
  sed -E "s|$R_ROOT/exp_T1A1|__X__|g; s|/checkpoints/[0-9]+|/checkpoints/__CK__|g" "$R_ROOT/exp_T1A1/logs/mid.jsonl" > "$TMP/t1a1.jsonl"
  if diff -u "$TMP/t1a0.jsonl" "$TMP/t1a1.jsonl"; then
    pass t1a-mid-jsonl-identical
  else
    fail t1a-mid-jsonl-identical "files differ"
  fi
  if grep -q 'MID_WAIT 0s' "$R_ROOT/exp_T1A1/logs/train.log" && ! grep -qE 'MID_WAIT [1-9]' "$R_ROOT/exp_T1A1/logs/train.log"; then
    pass t1a-mid-wait-zero
  else
    fail t1a-mid-wait-zero "log=$(grep MID_WAIT $R_ROOT/exp_T1A1/logs/train.log)"
  fi
else
  fail t1a-run "rc0=$rc0 rc1=$rc1 logs: $(tail -15 $R_ROOT/exp_T1A0/logs/case.log) :: $(tail -15 $R_ROOT/exp_T1A1/logs/case.log)"
fi

# T1(b): E > S, positive MID_WAIT, still correct mid.jsonl (same as mode 0 with same stubs).
echo "=== T1b E>S ==="
rc0=$(run_stub_case T1B0 0 1 3 0.60 2)
rc1=$(run_stub_case T1B1 1 1 3 0.60 2)
if [[ $rc0 -eq 0 && $rc1 -eq 0 ]]; then
  sed -E "s|$R_ROOT/exp_T1B0|__X__|g; s|/checkpoints/[0-9]+|/checkpoints/__CK__|g" "$R_ROOT/exp_T1B0/logs/mid.jsonl" > "$TMP/t1b0.jsonl"
  sed -E "s|$R_ROOT/exp_T1B1|__X__|g; s|/checkpoints/[0-9]+|/checkpoints/__CK__|g" "$R_ROOT/exp_T1B1/logs/mid.jsonl" > "$TMP/t1b1.jsonl"
  if diff -u "$TMP/t1b0.jsonl" "$TMP/t1b1.jsonl"; then
    pass t1b-mid-jsonl-identical
  else
    fail t1b-mid-jsonl-identical "files differ"
  fi
  if grep -qE 'MID_WAIT [1-9][0-9]*s' "$R_ROOT/exp_T1B1/logs/train.log"; then
    pass t1b-mid-wait-positive
  else
    fail t1b-mid-wait-positive "log=$(grep MID_WAIT $R_ROOT/exp_T1B1/logs/train.log)"
  fi
else
  fail t1b-run "rc0=$rc0 rc1=$rc1 $(tail -20 $R_ROOT/exp_T1B1/logs/case.log)"
fi

# T1(c): canned h2h < 0.40 for two chunks; rollback one boundary later than mode 0.
echo "=== T1c rollback lag ==="
rc0=$(run_stub_case T1C0 0 1 0 LOW 3)
rc1=$(run_stub_case T1C1 1 1 0 LOW 3)
if [[ $rc0 -eq 0 && $rc1 -eq 0 ]]; then
  # Mode 0: after chunk 2 book, rollback fires; a new ckpt is written with steps=chunk2.
  # Mode 1: after chunk 2 train, we wait and book chunk 1 only (one LOW); after chunk 3
  # train we book chunk 2 (second LOW) and rollback — one boundary later.
  g0=$(grep -c 'GUARD: 2 chunks' "$R_ROOT/exp_T1C0/logs/train.log" || true)
  g1=$(grep -c 'GUARD: 2 chunks' "$R_ROOT/exp_T1C1/logs/train.log" || true)
  # Compare when the rollback log line appears relative to chunk-done / MID_LAUNCHED lines.
  # Mode 0 rollback after the second "chunk done". Mode 1 rollback after the second wait
  # (which is after the third train / second booked mid... actually after booking mid 2,
  # which happens after chunk 3 train).
  if [[ $g0 -ge 1 && $g1 -ge 1 ]]; then
    pass t1c-both-rolled-back
  else
    fail t1c-both-rolled-back "g0=$g0 g1=$g1"
  fi
  # Mode 0 guard.env after full run should be LOW=0 (reset on rollback). Same in mode 1.
  if diff -u "$R_ROOT/exp_T1C0/guard.env" "$R_ROOT/exp_T1C1/guard.env"; then
    pass t1c-guard-env-matches
  else
    fail t1c-guard-env-matches "0=$(cat $R_ROOT/exp_T1C0/guard.env) 1=$(cat $R_ROOT/exp_T1C1/guard.env)"
  fi
  # Lag: mode 0 rollback logged before the third train; mode 1 after.
  python3 - << PY
import re, pathlib
def events(p):
    out=[]
    for line in pathlib.Path(p).read_text().splitlines():
        if "GUARD: 2 chunks" in line: out.append("rollback")
        if " train: " in line: out.append("train")
        if "chunk done" in line: out.append("book")
        if "MID_LAUNCHED" in line: out.append("launch")
        if "MID_WAIT" in line: out.append("wait")
    return out
e0=events("$R_ROOT/exp_T1C0/logs/train.log")
e1=events("$R_ROOT/exp_T1C1/logs/train.log")
# Mode 0: train, book, train, book, rollback, train, book
# Mode 1: train, launch, train, wait, book, launch, train, wait, book, rollback
open("$TMP/t1c_events.txt","w").write("mode0 "+str(e0)+"\\nmode1 "+str(e1)+"\\n")
ok0 = e0.count("rollback")==1 and e0.index("rollback") < len([x for x in e0]) and e0[:e0.index("rollback")].count("train")==2
ok1 = e1.count("rollback")==1 and e1[:e1.index("rollback")].count("train")==3
raise SystemExit(0 if ok0 and ok1 else 1)
PY
  if [[ $? -eq 0 ]]; then
    pass t1c-rollback-one-boundary-later
  else
    fail t1c-rollback-one-boundary-later "events=$(cat $TMP/t1c_events.txt)"
  fi
else
  fail t1c-run "rc0=$rc0 rc1=$rc1 $(tail -15 $R_ROOT/exp_T1C1/logs/case.log)"
fi

# T1(e): final pending mid is booked before pick (covered if mid.jsonl has all chunks
# and TRAIN_DONE exists). 2 chunks mode 1.
echo "=== T1e final book before pick ==="
if [[ -f $R_ROOT/exp_T1A1/TRAIN_DONE ]]; then
  n=$(wc -l < "$R_ROOT/exp_T1A1/logs/mid.jsonl")
  if [[ $n -eq 2 ]]; then
    pass t1e-final-mid-booked-before-pick
  else
    fail t1e-final-mid-booked-before-pick "mid lines=$n"
  fi
else
  fail t1e-final-mid-booked-before-pick "no TRAIN_DONE"
fi

# T1(f): PASSIVE_ALARM lines same as mode 0, one chunk later.
# Trigger nexto alarm? Our canned nexto is 0.6/0.5 which is above thresholds.
# Use two mids with nexto_2v2 < 0.40. Rebuild canned for a dedicated case.
echo "=== T1f passive alarm lag ==="
write_alarm_canned() {
  local dest=$1 steps=$2
  write_canned_mid "$dest" "$steps" 0.60
  canned_wr 40 0.20 > "$dest/$steps/nexto_2v2.json"  # < 0.40 triggers nexto2
  canned_wr 40 0.10 > "$dest/$steps/nexto_3v3.json"  # < 0.20 triggers nexto3
}
RUN=T1F0; C=$C_REPO; X=$R_ROOT/exp_$RUN
CHUNK_T=100000; STEPS=$((2*CHUNK_T)); start=1000000
seed_exp "$X" "$C" "$start" "$STUB"
EVAL_STUB_DIR=$TMP/canned_T1F
write_alarm_canned "$EVAL_STUB_DIR" $((start+CHUNK_T))
write_alarm_canned "$EVAL_STUB_DIR" $((start+2*CHUNK_T))
EVAL_STUB_SLEEP=0; STUB_SLEEP=1
common_env
export REWARD_TMPL=tm11.toml TRAIN_TMPL=train_tm11.toml
unset GAL_WEIGHT NEXTO_WEIGHT
set +e
CONCURRENT_MIDS=0 bash "$C/experiments/team_launch_next.sh" >"$X/logs/case.log" 2>&1
rc0=$?
set -e
RUN=T1F1; X=$R_ROOT/exp_$RUN
seed_exp "$X" "$C" "$start" "$STUB"
common_env
set +e
CONCURRENT_MIDS=1 bash "$C/experiments/team_launch_next.sh" >"$X/logs/case.log" 2>&1
rc1=$?
set -e
if [[ $rc0 -eq 0 && $rc1 -eq 0 && -f $R_ROOT/exp_T1F0/PASSIVE_ALARM && -f $R_ROOT/exp_T1F1/PASSIVE_ALARM ]]; then
  # Same alarm text; steps may differ by one chunk (mode 1 books mid2 later,
  # but both book mid2 at the same step number). Compare the alarm payload.
  cut -d' ' -f2- "$R_ROOT/exp_T1F0/PASSIVE_ALARM" > "$TMP/pa0"
  cut -d' ' -f2- "$R_ROOT/exp_T1F1/PASSIVE_ALARM" > "$TMP/pa1"
  if diff -u "$TMP/pa0" "$TMP/pa1"; then
    pass t1f-passive-alarm-payload
  else
    fail t1f-passive-alarm-payload "payloads differ"
  fi
else
  fail t1f-run "rc0=$rc0 rc1=$rc1 pa0=$(cat $R_ROOT/exp_T1F0/PASSIVE_ALARM 2>/dev/null) pa1=$(cat $R_ROOT/exp_T1F1/PASSIVE_ALARM 2>/dev/null) $(tail -10 $R_ROOT/exp_T1F1/logs/case.log)"
fi

# T1(d): kill the launcher by exact PID during an eval, then restart; recovery
# finishes the missing mid with no duplicate lines.
echo "=== T1d restart recovery ==="
RUN=T1D; C=$C_REPO; X=$R_ROOT/exp_$RUN
CHUNK_T=100000; STEPS=$((2*CHUNK_T)); start=1000000
seed_exp "$X" "$C" "$start" "$STUB"
EVAL_STUB_DIR=$TMP/canned_T1D
write_canned_mid "$EVAL_STUB_DIR" $((start+CHUNK_T)) 0.60
write_canned_mid "$EVAL_STUB_DIR" $((start+2*CHUNK_T)) 0.60
EVAL_STUB_SLEEP=4
STUB_SLEEP=0
common_env
export REWARD_TMPL=tm11.toml TRAIN_TMPL=train_tm11.toml
unset GAL_WEIGHT NEXTO_WEIGHT
set +e
CONCURRENT_MIDS=1 bash "$C/experiments/team_launch_next.sh" >"$X/logs/case.log" 2>&1 &
LPID=$!
# Wait until a mid is in flight (EVAL_PID exists) then kill the launcher only.
for i in $(seq 1 30); do
  if ls "$X"/mid/*/EVAL_PID >/dev/null 2>&1; then
    break
  fi
  sleep 0.2
done
if ! kill -0 $LPID 2>/dev/null; then
  fail t1d-launcher-died-early "log=$(cat $X/logs/case.log)"
else
  kill $LPID
  wait $LPID 2>/dev/null
  # Restart
  CONCURRENT_MIDS=1 bash "$C/experiments/team_launch_next.sh" >>"$X/logs/case.log" 2>&1
  rc=$?
  set -e
  # mid.jsonl line count is checked via the python unique/dup pass below
  dups=$("$PYT" -c 'import json,sys,collections
p=sys.argv[1]
c=collections.Counter()
for line in open(p):
    if line.strip():
        c[json.loads(line)["steps"]]+=1
d=[(s,n) for s,n in c.items() if n>1]
print(len(c), len(d))
' "$X/logs/mid.jsonl")
  nunique=${dups% *}; ndup=${dups#* }
  if [[ $rc -eq 0 && $ndup -eq 0 && $nunique -eq 2 && -f $X/TRAIN_DONE ]]; then
    pass t1d-recovery-no-duplicates
  else
    fail t1d-recovery-no-duplicates "rc=$rc unique=$nunique dups=$ndup mid=$(cat $X/logs/mid.jsonl 2>/dev/null) log=$(tail -20 $X/logs/case.log)"
  fi
fi
set -e

echo "==== results: $fails failed ===="
if [[ $fails -eq 0 ]]; then
  cleanup_ok=1
  exit 0
fi
echo "scratch left at $TMP"
exit 1
