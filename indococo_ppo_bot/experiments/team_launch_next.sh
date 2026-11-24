#!/usr/bin/env bash
# team_launch_next.sh: generalized copy of team_launch_tm11.sh for TM12 onward.
# The TM11 launcher is not edited. Mid eval/bookkeeping live in lib/mids.sh.
#
# CONCURRENT_MIDS (default 0): 0 is byte-equivalent to TM11 (train, mid_evals,
# mid_book). 1 runs mid_evals of chunk k in the background while chunk k+1
# trains, then mid_book at the k+1 boundary. A guard rollback therefore
# applies to chunk k+2 instead of k+1 (one chunk of extra latency; same for
# PASSIVE_ALARM). Only one eval set is in flight; mid.jsonl stays in steps
# order. Monitoring parsers will see MID_LAUNCHED first and the mid line one
# chunk later.
#
# DRY_RENDER=1: write tm_rewards.toml and train_tm.toml into $X and exit
# before training. SKIP_PREFLIGHT=1 is for stub tests only (default 0).
# EVAL_OMP, when set, is OMP_NUM_THREADS for eval commands only.
# EVAL_TORCH_THREADS, when set, is also OMP/MKL/OPENBLAS/TORCH_NUM_THREADS for
# eval commands (defaults to EVAL_OMP if that is set). EVAL_TASKSET pins evals
# (e.g. 24-31). TRAIN_TASKSET pins the trainer. All default unset (TM11).
# SKIP_FINALS=1 exits after TRAIN_DONE / mid bookkeeping (tests only; default 0).
# NEXTO_REPLICAS, NEXTO_CHUNK, NF_INTRA, ASYNC, ASYNC_PRIO: optional toml
# keys, written only when set. Defaults keep TM11 rendering.
#
# BT pin, fail-closed mid/guard/alarm/finals, numpy 1.26.4, CHUNK 100M, no
# automatic follow-up: same as TM11.
#
# shellcheck disable=SC1090,SC1091,SC2010,SC2046,SC2086,SC2034
# Quoting and latest() listing match team_launch_tm11.sh; mid_evals/mid_book
# stay aligned with that launcher. W is the TM11 eval-output directory.

set -u
R=${R:-/home/ubuntu/ippo_runs}
RUN=${RUN:-TM11}
VARIANT=${VARIANT:-$RUN}
X=$R/exp_$RUN
C=${C:-$R/code_${RUN,,}/indococo_ppo_bot}
SELF=$(readlink -f "${BASH_SOURCE[0]}")
[[ $SELF == "$(readlink -f "$C")/experiments/team_launch_next.sh" ]] || { echo "FATAL: must run $C/experiments/team_launch_next.sh (got $SELF)" >&2; exit 1; }
case "$(readlink -f "$C")" in "$R/code_tm"|"$R/code_tm/"*) echo "FATAL: C points at the live TM10 copy" >&2; exit 1 ;; esac
source "$C/experiments/lib/paths.env"
source "$C/experiments/lib/chain.sh"
THREADS=${THREADS:-24}; EV_THREADS=${EV_THREADS:-8}
MID_GAMES=${MID_GAMES:-200}; GAL_GAMES=${GAL_GAMES:-200}; GAL1_GAMES=${GAL1_GAMES:-200}; MEGA_GAMES=${MEGA_GAMES:-104}
NEXTO_MID=${NEXTO_MID:-200}; NECTO_MID=${NECTO_MID:-104}
FIN_GAMES=${FIN_GAMES:-504}; FIN_GAL=${FIN_GAL:-256}; NF_FIN_GAMES=${NF_FIN_GAMES:-504}
GAL_JITTER=${GAL_JITTER:-1}
export INDOCOCO_OBS_EXT=1 INDOCOCO_OBS_TEAM=1 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 INDOCOCO_METRICS=$X/logs/metrics.jsonl
unset RIPPLE_WEIGHT
EVP="$PYT $C/experiments/tm11_eval.py"
mkdir -p $X/logs $X/checkpoints $X/opponents
log() { echo "$(date '+%F %T %Z') $*" | tee -a $X/logs/train.log; }
die() { log "FATAL: $*"; exit 1; }
source $C/experiments/tm_$VARIANT.env || die "no tm_$VARIANT.env"
preflight() {
  "$PYT" -c 'import numpy,sys; sys.exit(0 if numpy.__version__=="1.26.4" else 1)' || die "numpy in $PYT is not 1.26.4"
  [[ ${CHUNK//_/} == 100000000 ]] || die "CHUNK=$CHUNK (rule: 100M)"
  bash "$C/experiments/lint_launch.sh" "$C/experiments/$TRAIN_TMPL" "$C/experiments/tm_$VARIANT.env" > "$X/logs/lint.log" 2>&1 \
    || die "lint_launch.sh failed: $(grep '^FAIL' "$X/logs/lint.log" | head -3 | tr '\n' ';')"
  [[ -x $B0 ]] || die "B0 $B0 missing"
  local live
  for live in "$R/exp_TM10/indococo-train" /opt/dlami/nvme/ippo_bins/indococo-train-tm10-*; do
    [[ -e $live ]] && cmp -s "$B0" "$live" && die "B0 is byte-identical to the TM10 binary $live"
  done
  [[ -s $C/experiments/tm11_eval.py ]] || die "tm11_eval.py missing"
}
if [[ "${SKIP_PREFLIGHT:-0}" != 1 ]]; then
  preflight
fi
# TM3: the prediction obs is per variant (explicit, so a re-exec never inherits it).
export INDOCOCO_OBS_PRED=${OBS_PRED:-0}
steps_in() { sed -nE 's/^cumulative_timesteps = ([0-9]+)/\1/p' $1/stats.toml; }
latest() { ls $X/checkpoints | grep -E '^[0-9]+$' | sort -n | tail -1; }
if ! ls $X/checkpoints 2>/dev/null | grep -qE '^[0-9]+$'; then
  [[ -s $START_CK/critic.bpk ]] || die "START_CK $START_CK has no critic"
  START=$(steps_in $START_CK)
  INIT=$(basename "$START_CK"); [[ "$INIT" =~ ^[0-9]+$ ]] && (( INIT < $(date +%s) )) || INIT=$(( $(date +%s) - 60 ))
  mkdir -p $X/cfg; cp -r $C/experiments $C/config $X/cfg/; cp $B0 $X/indococo-train
  log "$RUN ($VARIANT) start from $START_CK ($START), team seed (widen-team unless START_IS_TEAM=1)"
  if [[ "${START_IS_TEAM:-0}" == 1 && "$INDOCOCO_OBS_PRED" == 1 ]]; then   # TM3: 235 -> 270 (or copy if 270)
    ( cd "$MESH_REPO" && nice -n 19 $X/indococo-train widen-pred $START_CK $X/seed ) > $X/logs/widen.log 2>&1 || die "widen-pred failed"
    grep WIDEN_PRED $X/logs/widen.log | tee -a $X/logs/train.log
  elif [[ "${START_IS_TEAM:-0}" == 1 ]]; then   # follow-up: already team-ext
    mkdir -p $X/seed; cp $START_CK/actor.bpk $START_CK/critic.bpk $START_CK/stats.toml $X/seed/
  else
    ( cd "$MESH_REPO" && nice -n 19 $X/indococo-train widen-team $START_CK $X/seed ) > $X/logs/widen.log 2>&1 || die "widen-team failed"
    grep WIDEN_TEAM $X/logs/widen.log | tee -a $X/logs/train.log
  fi
  mkdir -p $X/checkpoints/$INIT; cp $X/seed/actor.bpk $X/seed/critic.bpk $X/checkpoints/$INIT/
  sed -E "s/^cumulative_timesteps = .*/cumulative_timesteps = $START/" $X/seed/stats.toml > $X/checkpoints/$INIT/stats.toml
  for e in ${EXTRA_OPPS:-}; do n=${e%%:*}; r=${e#*:}; d=${r%:*}
    mkdir -p $X/opponents/$n; cp $d/actor.bpk $X/opponents/$n/ || die "extra opp $e"; done
  echo "$START_CK $START $VARIANT" > $X/START; echo LOW=0 > $X/guard.env
  # BEST-team: the shared team reference. First run: the widened seed (= the 1v1 policy).
  tb=$(cat "$R/team_best/FINAL_CKPT" 2>/dev/null || true)
  [[ "$tb" == "$BT_PIN" ]] || log "WARNING: team_best ($tb) != BT_PIN ($BT_PIN); TM11 gates vs BT_PIN. team_best is not changed."
fi
B=$X/indococo-train
# Eval binary for Galactus / Megabyte / Nexto / Necto: the run's own copy by
# default (B0 must then be tm4pre or later); EVAL_BIN=<binary> to override.
EVAL_BIN=${EVAL_BIN:-$B}
read _ START0 _ < $X/START; END=$((START0 + STEPS))
# BT is pinned to the TM9 final (BT_PIN). A team_best update cannot retarget it.
BT=${BT_PIN:?BT_PIN unset}
[[ -s "$BT/actor.bpk" ]] || die "BT pin missing or empty: $BT"
# Reward config: the team template with the team-spirit origin = run start.
printf 'extends = "%s"\n[team]\nspirit_origin = %s.0\n' "$X/cfg/experiments/${REWARD_TMPL:-tm1.toml}" "$START0" > $X/tm_rewards.toml
ev() {
  local name=$1; shift
  if [[ "${EVAL_STUB:-0}" == 1 ]]; then
    # Only mid_evals sets EVAL_STUB_STEPS. Finals ev() must not mkdir mid/0.
    if [[ -n "${EVAL_STUB_STEPS:-}" ]]; then
      local stubdir="${X}/mid/${EVAL_STUB_STEPS}"
      mkdir -p "$stubdir"
      if [[ "${EVAL_STUB_SLEEP:-0}" != 0 && ! -f "$stubdir/.stub_slept" ]]; then
        sleep "${EVAL_STUB_SLEEP}"
        : > "$stubdir/.stub_slept"
      fi
      local src="${EVAL_STUB_DIR:-}/${EVAL_STUB_STEPS}/${name}.json"
    else
      local src=""
    fi
    if [[ -n $src && -f $src ]]; then cat "$src"; else echo '{}'; fi
    return 0
  fi
  if [[ -n ${EVAL_OMP:-} || -n ${EVAL_TORCH_THREADS:-} ]]; then
    local ot=${EVAL_TORCH_THREADS:-$EVAL_OMP}
    ( cd "$MESH_REPO" &&       OMP_NUM_THREADS="$ot" MKL_NUM_THREADS="$ot" OPENBLAS_NUM_THREADS="$ot" TORCH_NUM_THREADS="$ot"       run_eval "$name" "$@" ) 2>>"$X/logs/eval_fail.log"
  else
    ( cd "$MESH_REPO" && run_eval "$name" "$@" ) 2>>"$X/logs/eval_fail.log"
  fi
}
h2h_mode() { # n a b games name
  ev "$5" env INDOCOCO_TEAM_SIZE="$1" H2H_REWARD_CONFIG="$X/tm_rewards.toml" "$B" h2h "$2" "$3" "$4" "$EV_THREADS"; }
gal_mode() { ev "$4" env H2H_JITTER="$GAL_JITTER" INDOCOCO_TEAM_SIZE="$1" "$EVAL_BIN" galactus-h2h "$2" "$GAL_DIR" "$3" "$EV_THREADS"; }
mega_mode() { ev "$4" env H2H_JITTER="$GAL_JITTER" INDOCOCO_TEAM_SIZE="$1" "$EVAL_BIN" megabyte-h2h "$2" "$MEGA_DIR" "$3" "$EV_THREADS"; }
nf_mode() { # n kind ckpt games name [traj_prefix]
  local d=$NEXTO_DIR; [[ $2 == necto ]] && d=$NECTO_DIR
  ev "$5" env INDOCOCO_TEAM_SIZE="$1" ${6:+TRAJ_DUMP="$6"} "$EVAL_BIN" "$2-h2h" "$3" "$d" "$4" "$EV_THREADS"; }

# Optional speed keys, written only when the matching env is set (default off).
# Insert before the first [table]. Appending after [[opponents.*]] would land
# the keys inside that table, and deny_unknown_fields would refuse to start.
append_speed_toml() {
  local f=$1
  local lines=()
  if [[ -n ${NF_INTRA:-} ]]; then
    lines+=("nf_intra_threads = ${NF_INTRA}")
  fi
  if [[ "${ASYNC_PRIO:-0}" == 1 ]]; then
    lines+=("async_collect = true" "collect_stream_priority = true")
  elif [[ "${ASYNC:-0}" == 1 ]]; then
    lines+=("async_collect = true")
  fi
  if (( ${#lines[@]} == 0 )); then
    return 0
  fi
  python3 - "$f" "${lines[@]}" << 'PY'
import pathlib, re, sys
path = pathlib.Path(sys.argv[1])
text = path.read_text()
m = re.search(r"(?m)^\[", text)
head, tail = (text[:m.start()], text[m.start():]) if m else (text, "")
for line in sys.argv[2:]:
    key = line.split("=", 1)[0].strip()
    head, n = re.subn(rf"(?m)^{re.escape(key)}\s*=.*\n", line + "\n", head, count=1)
    if n == 0:
        if head and not head.endswith("\n"):
            head += "\n"
        head += line + "\n"
path.write_text(head + tail)
PY
}

append_nf_block() {
  local f=$1
  [[ -n ${NEXTO_REPLICAS:-} ]] && printf 'replicas = %s\n' "$NEXTO_REPLICAS" >> "$f"
  [[ -n ${NEXTO_CHUNK:-} ]] && printf 'chunk_rows = %s\n' "$NEXTO_CHUNK" >> "$f"
}
source "$C/experiments/lib/mids.sh"
source $X/guard.env

# DRY_RENDER: write the same train_tm.toml / tm_rewards.toml the loop would
# use at the current checkpoint and exit before training.
if [[ "${DRY_RENDER:-0}" == 1 ]]; then
  now=$(steps_in $X/checkpoints/$(latest))
  stop=$(( now + CHUNK )); (( stop > END )) && stop=$END
  left=$(( (stop - now + ${TPI//_/} - 1) / ${TPI//_/} ))
  sed -e "s|@RUN@|$X|g" -e "s|@ITERATIONS@|$left|" -e "s|@WARM@|$((START0 + ${WARM_REL:-20000000}))|" -e "s|@HOLD@|$((START0 + HOLDREL))|"       -e "s|@SNAP@|$SNAP/train.f32|" -e "s|@TPI@|$TPI|" -e "s|@MB@|$MB|" -e "s|@LR@|$LR|g" -e "s|@ENT@|$ENT|" -e "s|@GAMMA@|$GAMMA|"       -e "s|@OPPCH@|$OPPCH|" -e "s|@PASTW@|${PASTW:-1.0}|" -e "s|@DECAY@|$DECAY|" -e "s|@MINLR@|$MINLR|" -e "s|^threads = .*|threads = $THREADS|" -e "s|^games = .*|games = ${GAMES:-384}|"       $X/cfg/experiments/${TRAIN_TMPL:-train_tm1.toml} > $X/train_tm.toml
  [[ -n "${GAL_WEIGHT:-}" ]] && printf '\n[[opponents.external]]\nname = "galactus"\nkind = "galactus"\npath = "%s"\nweight = %s\n' "$GAL_DIR" "$GAL_WEIGHT" >> $X/train_tm.toml
  [[ -n "${MEGA_WEIGHT:-}" ]] && printf '\n[[opponents.external]]\nname = "megabyte"\nkind = "megabyte"\npath = "%s"\nweight = %s\n' "$MEGA_DIR" "$MEGA_WEIGHT" >> $X/train_tm.toml
  [[ -n "${NEXTO_WEIGHT:-}" ]] && printf '\n[[opponents.external]]\nname = "nexto"\nkind = "nexto"\npath = "%s"\nweight = %s\nstochastic = %s\n' "$NEXTO_DIR" "$NEXTO_WEIGHT" "${NEXTO_STOCH:-0.0}" >> $X/train_tm.toml
  [[ -n "${NEXTO_WEIGHT:-}" ]] && append_nf_block $X/train_tm.toml
  [[ -n "${NECTO_WEIGHT:-}" ]] && printf '\n[[opponents.external]]\nname = "necto"\nkind = "necto"\npath = "%s"\nweight = %s\nstochastic = %s\n' "$NECTO_DIR" "$NECTO_WEIGHT" "${NECTO_STOCH:-0.0}" >> $X/train_tm.toml
  for e in ${EXTRA_OPPS:-}; do n=${e%%:*}
    printf '\n[[opponents.fixed]]\nname = "%s"\ncheckpoint = "%s"\nweight = %s\n' "$n" "$X/opponents/$n" "${e##*:}" >> $X/train_tm.toml; done
  append_speed_toml $X/train_tm.toml
  log "$RUN DRY_RENDER wrote $X/train_tm.toml and $X/tm_rewards.toml"
  exit 0
fi

CONCURRENT_MIDS=${CONCURRENT_MIDS:-0}
PEND_CK=; PEND_NOW=; MID_PID=
if [[ $CONCURRENT_MIDS == 1 ]]; then
  mid_recover
fi

while :; do
  now=$(steps_in $X/checkpoints/$(latest)); (( now >= END )) && break
  stop=$(( now + CHUNK )); (( stop > END )) && stop=$END
  left=$(( (stop - now + ${TPI//_/} - 1) / ${TPI//_/} ))
  sed -e "s|@RUN@|$X|g" -e "s|@ITERATIONS@|$left|" -e "s|@WARM@|$((START0 + ${WARM_REL:-20000000}))|" -e "s|@HOLD@|$((START0 + HOLDREL))|" \
      -e "s|@SNAP@|$SNAP/train.f32|" -e "s|@TPI@|$TPI|" -e "s|@MB@|$MB|" -e "s|@LR@|$LR|g" -e "s|@ENT@|$ENT|" -e "s|@GAMMA@|$GAMMA|" \
      -e "s|@OPPCH@|$OPPCH|" -e "s|@PASTW@|${PASTW:-1.0}|" -e "s|@DECAY@|$DECAY|" -e "s|@MINLR@|$MINLR|" -e "s|^threads = .*|threads = $THREADS|" -e "s|^games = .*|games = ${GAMES:-384}|" \
      $X/cfg/experiments/${TRAIN_TMPL:-train_tm1.toml} > $X/train_tm.toml
  [[ -n "${GAL_WEIGHT:-}" ]] && printf '\n[[opponents.external]]\nname = "galactus"\nkind = "galactus"\npath = "%s"\nweight = %s\n' "$GAL_DIR" "$GAL_WEIGHT" >> $X/train_tm.toml
  [[ -n "${MEGA_WEIGHT:-}" ]] && printf '\n[[opponents.external]]\nname = "megabyte"\nkind = "megabyte"\npath = "%s"\nweight = %s\n' "$MEGA_DIR" "$MEGA_WEIGHT" >> $X/train_tm.toml
  [[ -n "${NEXTO_WEIGHT:-}" ]] && printf '\n[[opponents.external]]\nname = "nexto"\nkind = "nexto"\npath = "%s"\nweight = %s\nstochastic = %s\n' "$NEXTO_DIR" "$NEXTO_WEIGHT" "${NEXTO_STOCH:-0.0}" >> $X/train_tm.toml
  [[ -n "${NEXTO_WEIGHT:-}" ]] && append_nf_block $X/train_tm.toml
  [[ -n "${NECTO_WEIGHT:-}" ]] && printf '\n[[opponents.external]]\nname = "necto"\nkind = "necto"\npath = "%s"\nweight = %s\nstochastic = %s\n' "$NECTO_DIR" "$NECTO_WEIGHT" "${NECTO_STOCH:-0.0}" >> $X/train_tm.toml
  for e in ${EXTRA_OPPS:-}; do n=${e%%:*}
    printf '\n[[opponents.fixed]]\nname = "%s"\ncheckpoint = "%s"\nweight = %s\n' "$n" "$X/opponents/$n" "${e##*:}" >> $X/train_tm.toml; done
  append_speed_toml $X/train_tm.toml
  grep -qE '^epochs = 1$' $X/train_tm.toml || die "rendered train_tm.toml: epochs != 1"
  ! grep -qE '^[a-z_]*bf16 = true' $X/train_tm.toml || die "rendered train_tm.toml: bf16 on"
  log "$RUN train: $left iterations from $now"
  if [[ -n ${TRAIN_TASKSET:-} ]]; then
    ( cd "$MESH_REPO" && nice -n 19 taskset -c "$TRAIN_TASKSET" "$B" train $X/train_tm.toml ) 2>&1 | tee -a $X/logs/train.log
  else
    ( cd "$MESH_REPO" && nice -n 19 "$B" train $X/train_tm.toml ) 2>&1 | tee -a $X/logs/train.log
  fi
  [[ ${PIPESTATUS[0]} -eq 0 ]] || die "train failed"
  prev=$now; ck=$X/checkpoints/$(latest); now=$(steps_in $ck)
  (( now > prev )) || die "steps did not advance ($prev -> $now)"
  if [[ $CONCURRENT_MIDS == 1 ]]; then
    if [[ -n $MID_PID ]]; then
      t0=$(date +%s)
      wait $MID_PID
      log "MID_WAIT $(( $(date +%s)-t0 ))s for $PEND_NOW"
      mid_book "$PEND_CK" "$PEND_NOW"
      MID_PID=; PEND_CK=; PEND_NOW=
      # rollback (if any) created a newer ckpt dir; next iteration of this
      # loop already finished training the just-completed chunk, so the
      # rolled-back state is used for chunk k+2. Re-read latest after book.
    fi
    ( mid_evals "$ck" "$now" ) &
    MID_PID=$!
    mkdir -p "$X/mid/$now"
    echo $MID_PID > "$X/mid/$now/EVAL_PID"
    PEND_CK=$ck
    PEND_NOW=$now
    log "MID_LAUNCHED $now pid $MID_PID"
  else
    mid_evals "$ck" "$now"
    mid_book "$ck" "$now"
  fi
done
if [[ -n ${MID_PID:-} ]]; then
  wait $MID_PID
  mid_book "$PEND_CK" "$PEND_NOW"
  MID_PID=; PEND_CK=; PEND_NOW=
fi
# Mode-1 restart: the last train may already have finished (now >= END) before
# mid_evals for that ckpt was launched. Evaluate it before pick/finals.
if [[ ${CONCURRENT_MIDS:-0} == 1 ]]; then
  lastck=$X/checkpoints/$(latest)
  last=$(steps_in "$lastck")
  if [[ -n $last && $last =~ ^[1-9][0-9]+$ ]] && ! mid_jsonl_has_steps "$last"; then
    if [[ -d $lastck && $lastck != "$X/checkpoints" && $lastck != "$X/checkpoints/" && -n ${START0:-} ]] && (( last > START0 )); then
      log "MID_CATCHUP evals for $last (latest ckpt had no mid line)"
      mid_evals "$lastck" "$last"
      mid_book "$lastck" "$last"
    fi
  fi
fi
touch $X/TRAIN_DONE
if [[ "${SKIP_FINALS:-0}" == 1 ]]; then
  log "$RUN SKIP_FINALS=1; exiting after mid bookkeeping"
  exit 0
fi
bestck=$($EVP pick "$X/logs/mid.jsonl" 2>/dev/null)
[[ -n "$bestck" && -d "$bestck" ]] || { log "no complete mid; final = latest checkpoint"; bestck=$X/checkpoints/$(latest); }
F=$R/${RUN,,}_final; mkdir -p "$F"; W=$F
echo "$bestck" > "$F/FINAL_CKPT"; steps_in "$bestck" > "$F/STEPS"; echo "$BT" > "$F/BEST_TEAM"
ev export "$B" export "$bestck" "$F/final_policy.safetensors" > /dev/null
sha256sum "$F/final_policy.safetensors" > "$F/sha.txt" 2>/dev/null
for n in 2 3; do
  h2h_mode $n "$bestck" "$BT" "$FIN_GAMES" h2h_${n}v${n} > "$F/h2h_${n}v${n}.json"
  gal_mode $n "$bestck" "$FIN_GAL" gal_${n}v${n} > "$F/gal_${n}v${n}.json"
  gal_mode $n "$BT" "$FIN_GAL" gal_${n}v${n}_best > "$F/gal_${n}v${n}_best.json"
  for k in nexto necto; do
    nf_mode $n $k "$bestck" "$NF_FIN_GAMES" ${k}_${n}v${n} $([[ $k == nexto ]] && echo "$F/nexto_${n}v${n}") > "$F/${k}_${n}v${n}.json"
    nf_mode $n $k "$BT" "$NF_FIN_GAMES" ${k}_${n}v${n}_best > "$F/${k}_${n}v${n}_best.json"
  done
done
gal_mode 1 "$bestck" "$FIN_GAMES" gal_1v1 > "$F/gal_1v1.json"; mega_mode 2 "$bestck" "$FIN_GAL" mega_2v2 > "$F/mega_2v2.json"
[[ -s $ST25_CK/actor.bpk ]] && ev h2h_1v1_st25 env INDOCOCO_TEAM_SIZE=1 "$B" h2h "$bestck" "$ST25_CK" "$FIN_GAMES" "$EV_THREADS" > "$F/h2h_1v1_st25.json"
timeout --kill-after=60 1800 nice -n 19 "$PYT" "$C/tools/team/tm8_audit/tm8_audit_traces.py" "nexto_2v2:a:$F/nexto_2v2.nexto.*.jsonl" "nexto_3v3:a:$F/nexto_3v3.nexto.*.jsonl" > "$F/kickoff_audit.jsonl" 2> "$F/kickoff_audit.err" || true
if $EVP gate "$F" > "$F/TEAM_PROMOTE.json.tmp" 2>>"$X/logs/eval_fail.log"; then mv "$F/TEAM_PROMOTE.json.tmp" "$F/TEAM_PROMOTE.json"
else echo '{"promote_team":false,"error":"gate script failed"}' > "$F/TEAM_PROMOTE.json"; fi
touch "$F/DONE"; log "$RUN finals: $(cat "$F/TEAM_PROMOTE.json")"
# Opt-in (GAL_FIDELITY=1), report-only: headline Galactus metric on the C++ engine
# (Python harness, exact spawns) for 1v1/2v2/3v3 + Rust-vs-Python gap alarm ->
# $F/gal_fidelity.jsonl (EXPERIMENTS.md "Eval fidelity"). Detached, nice 19; never gates.
if [[ "${GAL_FIDELITY:-0}" == 1 ]]; then
  [[ "${OBS_PRED:-0}" == 1 ]] && GF_OBS=team || GF_OBS=team235
  GF="$C/experiments/gal_fidelity.sh"
  if [[ -f "$GF" && -f "$C/tools/pool_eval/gal_fidelity.py" ]]; then
    # flock -n: at most one fidelity run at a time (<= 2 heavy jobs on the box incl.
    # the next training run); a second finals while one runs is skipped, not stacked.
    setsid nohup flock -n "$R/.gal_fidelity.lock" bash "$GF" $bestck $F $GF_OBS 1,2,3 $B > $F/gal_fidelity.out 2>&1 < /dev/null &
  else
    log "GAL_FIDELITY: $GF missing, skipped"
  fi
fi
log "$RUN finished. No automatic follow-up: the next run needs a cross-exam and the user's OK."
