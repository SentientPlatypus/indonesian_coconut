#!/usr/bin/env bash
# TM12: team_launch_tm11.sh plus Nexto kickoff_script, critic extras (if smoke
# passed), Nexto-game mid keys, and a report-only TM9 finals panel (400 g/mode).
# BT is pinned to the TM10 final (BT_PIN / team_best at staging, user-approved
# 2026-11-24). team_best is read-only: never written. Every eval goes through
# run_eval. Mid lines, the candidate score, the 2-chunk guard, alarms and the
# finals gate are fail-closed in tm12_eval.py. Necto is reported but not scored
# (held-out kickoff audit only). No KWI-19 speedups. No automatic follow-up.
#
set -u
R=${R:-/home/ubuntu/ippo_runs}; RUN=${RUN:-TM12}; VARIANT=${VARIANT:-TM12}; X=$R/exp_$RUN
C=${C:-$R/code_tm12/indococo_ppo_bot}
SELF=$(readlink -f "${BASH_SOURCE[0]}")
[[ $SELF == "$(readlink -f "$C")/experiments/team_launch_tm12.sh" ]] || { echo "FATAL: must run $C/experiments/team_launch_tm12.sh (got $SELF)" >&2; exit 1; }
case "$(readlink -f "$C")" in "$R/code_tm"|"$R/code_tm/"*|"$R/code_tm11"|"$R/code_tm11/"*) echo "FATAL: C points at a live earlier-run copy" >&2; exit 1 ;; esac
source "$C/experiments/lib/paths.env"
source "$C/experiments/lib/chain.sh"
THREADS=${THREADS:-24}; EV_THREADS=${EV_THREADS:-8}
MID_GAMES=${MID_GAMES:-200}; GAL_GAMES=${GAL_GAMES:-200}; GAL1_GAMES=${GAL1_GAMES:-200}; MEGA_GAMES=${MEGA_GAMES:-104}
NEXTO_MID=${NEXTO_MID:-200}; NECTO_MID=${NECTO_MID:-104}
FIN_GAMES=${FIN_GAMES:-504}; FIN_GAL=${FIN_GAL:-256}; NF_FIN_GAMES=${NF_FIN_GAMES:-504}
GAL_JITTER=${GAL_JITTER:-1}
export INDOCOCO_OBS_EXT=1 INDOCOCO_OBS_TEAM=1 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 INDOCOCO_METRICS=$X/logs/metrics.jsonl
unset RIPPLE_WEIGHT
EVP="$PYT $C/experiments/tm12_eval.py"
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
  for live in "$R/exp_TM10/indococo-train" "$R/exp_TM11/indococo-train" /opt/dlami/nvme/ippo_bins/indococo-train-tm10-* /opt/dlami/nvme/ippo_bins/indococo-train-tm11-*; do
    [[ -e $live ]] && cmp -s "$B0" "$live" && die "B0 is byte-identical to a live earlier-run binary $live"
  done
  [[ -s $C/experiments/tm12_eval.py ]] || die "tm12_eval.py missing"
}
preflight
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
  if [[ "${CRITIC_EXTRAS:-1}" == 1 ]]; then
    ( cd "$MESH_REPO" && nice -n 19 $X/indococo-train widen-critic-extras $X/seed $X/seed ) > $X/logs/widen_critic.log 2>&1 || die "widen-critic-extras failed"
    grep WIDEN_CRITIC $X/logs/widen_critic.log | tee -a $X/logs/train.log
  fi
  mkdir -p $X/checkpoints/$INIT; cp $X/seed/actor.bpk $X/seed/critic.bpk $X/checkpoints/$INIT/
  sed -E "s/^cumulative_timesteps = .*/cumulative_timesteps = $START/" $X/seed/stats.toml > $X/checkpoints/$INIT/stats.toml
  for e in ${EXTRA_OPPS:-}; do n=${e%%:*}; r=${e#*:}; d=${r%:*}
    mkdir -p $X/opponents/$n; cp $d/actor.bpk $X/opponents/$n/ || die "extra opp $e"; done
  echo "$START_CK $START $VARIANT" > $X/START; echo LOW=0 > $X/guard.env
  # BEST-team: the shared team reference. First run: the widened seed (= the 1v1 policy).
  tb=$(cat "$R/team_best/FINAL_CKPT" 2>/dev/null || true)
  [[ "$tb" == "$BT_PIN" ]] || log "WARNING: team_best ($tb) != BT_PIN ($BT_PIN); TM12 gates vs BT_PIN. team_best is not changed."
fi
B=$X/indococo-train
# Eval binary for Galactus / Megabyte / Nexto / Necto: the run's own copy by
# default (B0 must then be tm4pre or later); EVAL_BIN=<binary> to override.
EVAL_BIN=${EVAL_BIN:-$B}
read _ START0 _ < $X/START; END=$((START0 + STEPS))
# BT is pinned to the TM10 final (BT_PIN). A team_best update cannot retarget it.
BT=${BT_PIN:?BT_PIN unset}
[[ -s "$BT/actor.bpk" ]] || die "BT pin missing or empty: $BT"
# Reward config: the team template with the team-spirit origin = run start.
printf 'extends = "%s"\n[team]\nspirit_origin = %s.0\n' "$X/cfg/experiments/${REWARD_TMPL:-tm1.toml}" "$START0" > $X/tm_rewards.toml
ev() { local name=$1; shift; ( cd "$MESH_REPO" && run_eval "$name" "$@" ) 2>>"$X/logs/eval_fail.log"; }
h2h_mode() { # n a b games name
  ev "$5" env INDOCOCO_TEAM_SIZE="$1" H2H_REWARD_CONFIG="$X/tm_rewards.toml" "$B" h2h "$2" "$3" "$4" "$EV_THREADS"; }
gal_mode() { ev "$4" env H2H_JITTER="$GAL_JITTER" INDOCOCO_TEAM_SIZE="$1" "$EVAL_BIN" galactus-h2h "$2" "$GAL_DIR" "$3" "$EV_THREADS"; }
mega_mode() { ev "$4" env H2H_JITTER="$GAL_JITTER" INDOCOCO_TEAM_SIZE="$1" "$EVAL_BIN" megabyte-h2h "$2" "$MEGA_DIR" "$3" "$EV_THREADS"; }
nf_mode() { # n kind ckpt games name [traj_prefix]
  local d=$NEXTO_DIR; [[ $2 == necto ]] && d=$NECTO_DIR
  ev "$5" env INDOCOCO_TEAM_SIZE="$1" ${6:+TRAJ_DUMP="$6"} "$EVAL_BIN" "$2-h2h" "$3" "$d" "$4" "$EV_THREADS"; }
source $X/guard.env
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
  if [[ -n "${NEXTO_WEIGHT:-}" ]]; then
    ks=${NEXTO_KICKOFF_SCRIPT:-true}
    printf '\n[[opponents.external]]\nname = "nexto"\nkind = "nexto"\npath = "%s"\nweight = %s\nstochastic = %s\nkickoff_script = %s\n' "$NEXTO_DIR" "$NEXTO_WEIGHT" "${NEXTO_STOCH:-0.0}" "$ks" >> $X/train_tm.toml
  fi
  [[ -n "${NECTO_WEIGHT:-}" ]] && printf '\n[[opponents.external]]\nname = "necto"\nkind = "necto"\npath = "%s"\nweight = %s\nstochastic = %s\n' "$NECTO_DIR" "$NECTO_WEIGHT" "${NECTO_STOCH:-0.0}" >> $X/train_tm.toml
  for e in ${EXTRA_OPPS:-}; do n=${e%%:*}
    printf '\n[[opponents.fixed]]\nname = "%s"\ncheckpoint = "%s"\nweight = %s\n' "$n" "$X/opponents/$n" "${e##*:}" >> $X/train_tm.toml; done
  if [[ "${CRITIC_EXTRAS:-1}" != 1 ]]; then
    "$PYT" -c 'from pathlib import Path
p=Path("'"$X"'/train_tm.toml"); t=p.read_text()
t=t.replace("extra_inputs = [\"role_onehot\", \"eta_rank\", \"mate_roles\"]","extra_inputs = []")
p.write_text(t)'
  fi
  grep -qE '^epochs = 1$' $X/train_tm.toml || die "rendered train_tm.toml: epochs != 1"
  ! grep -qE '^[a-z_]*bf16 = true' $X/train_tm.toml || die "rendered train_tm.toml: bf16 on"
  log "$RUN train: $left iterations from $now"
  ( cd "$MESH_REPO" && nice -n 19 "$B" train $X/train_tm.toml ) 2>&1 | tee -a $X/logs/train.log
  [[ ${PIPESTATUS[0]} -eq 0 ]] || die "train failed"
  prev=$now; ck=$X/checkpoints/$(latest); now=$(steps_in $ck)
  (( now > prev )) || die "steps did not advance ($prev -> $now)"
  W=$X/mid/$now; mkdir -p "$W"
  h2h_mode 2 "$ck" "$BT" "$MID_GAMES" h2h_2v2 > "$W/h2h_2v2.json"
  h2h_mode 3 "$ck" "$BT" "$MID_GAMES" h2h_3v3 > "$W/h2h_3v3.json"
  gal_mode 2 "$ck" "$GAL_GAMES" gal_2v2 > "$W/gal_2v2.json"; gal_mode 3 "$ck" "$GAL_GAMES" gal_3v3 > "$W/gal_3v3.json"
  gal_mode 1 "$ck" "$GAL1_GAMES" gal_1v1 > "$W/gal_1v1.json"
  mega_mode 2 "$ck" "$MEGA_GAMES" mega_2v2 > "$W/mega_2v2.json"
  for n in 2 3; do
    nf_mode $n nexto "$ck" "$NEXTO_MID" nexto_${n}v${n} "$W/nexto_${n}v${n}" > "$W/nexto_${n}v${n}.json"
    nf_mode $n necto "$ck" "$NECTO_MID" necto_${n}v${n} "$W/necto_${n}v${n}" > "$W/necto_${n}v${n}.json"
  done
  AUDIT_PY=$C/tools/team/tm8_audit/tm8_audit_traces.py
  timeout --kill-after=60 1800 nice -n 19 "$PYT" "$AUDIT_PY" "nexto_2v2:a:$W/nexto_2v2.nexto.*.jsonl" "nexto_3v3:a:$W/nexto_3v3.nexto.*.jsonl" \
    > "$W/kickoff_audit.jsonl" 2> "$W/kickoff_audit.err" || log "kickoff audit failed at $now (see $W/kickoff_audit.err)"
  timeout --kill-after=60 1800 nice -n 19 "$PYT" "$AUDIT_PY" "necto_2v2:a:$W/necto_2v2.necto.*.jsonl" "necto_3v3:a:$W/necto_3v3.necto.*.jsonl" \
    > "$W/necto_kickoff_audit.jsonl" 2> "$W/necto_kickoff_audit.err" || log "necto kickoff audit failed at $now (see $W/necto_kickoff_audit.err)"
  if [[ -s $ST25_CK/actor.bpk ]]; then ev h2h_1v1_st25 env INDOCOCO_TEAM_SIZE=1 "$B" h2h "$ck" "$ST25_CK" "$MID_GAMES" "$EV_THREADS" > "$W/h2h_1v1_st25.json"; else echo '{}' > "$W/h2h_1v1_st25.json"; fi
  line=$($EVP mid "$W" "$now" "$ck" --expect h2h_2v2=$MID_GAMES h2h_3v3=$MID_GAMES gal_2v2=$GAL_GAMES gal_3v3=$GAL_GAMES \
        gal_1v1=$GAL1_GAMES mega_2v2=$MEGA_GAMES nexto_2v2=$NEXTO_MID nexto_3v3=$NEXTO_MID necto_2v2=$NECTO_MID necto_3v3=$NECTO_MID \
        h2h_1v1_st25=$MID_GAMES 2>>"$X/logs/eval_fail.log") \
    || line="{\"steps\":$now,\"ckpt\":\"$ck\",\"complete\":false,\"missing\":[\"mid_writer\"],\"score\":null}"
  echo "$line" >> "$X/logs/mid.jsonl"
  log "$RUN chunk done at $now: $line"
  g=$($EVP guard "$X/logs/mid.jsonl" 2>>"$X/logs/eval_fail.log" || echo MISSING)
  case $g in
    LOW) LOW=$((LOW + 1)) ;;
    OK) LOW=0 ;;
    *) log "GUARD: last mid has a missing h2h; LOW stays $LOW" ;;
  esac
  if (( LOW >= 2 )); then
    src=$($EVP best_h2h "$X/logs/mid.jsonl" 2>/dev/null)
    [[ -n "$src" && -d "$src" ]] || src=$X/seed
    nd=$X/checkpoints/$(date +%s); mkdir -p $nd; cp $src/actor.bpk $src/critic.bpk $nd/
    sed -E "s/^cumulative_timesteps = .*/cumulative_timesteps = $now/" $src/stats.toml > $nd/stats.toml
    LOW=0; log "GUARD: 2 chunks with min(h2h 2v2, 3v3) < 0.40 vs BT; rolled back to $src as $nd"
  fi
  pa=$(NCR_BASE_2=${NCR_BASE_2:-} NCR_BASE_3=${NCR_BASE_3:-} H2H_ALARM=${H2H_ALARM:-} PASSIVE_BASE=${PASSIVE_BASE:-} \
       TOUCH_DROP=${TOUCH_DROP:-0.15} DEPTH_DROP=${DEPTH_DROP:-0.20} $EVP alarm "$X/logs/mid.jsonl" 2>>"$X/logs/eval_fail.log") || pa="alarm_script_failed"
  if [[ -n "$pa" ]]; then log "PASSIVE ALARM at $now (2 mids in a row): $pa"; echo "$now $pa" >> $X/PASSIVE_ALARM; fi
  echo LOW=$LOW > $X/guard.env
done
touch $X/TRAIN_DONE
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
    nf_mode $n $k "$bestck" "$NF_FIN_GAMES" ${k}_${n}v${n} "$F/${k}_${n}v${n}" > "$F/${k}_${n}v${n}.json"
    nf_mode $n $k "$BT" "$NF_FIN_GAMES" ${k}_${n}v${n}_best > "$F/${k}_${n}v${n}_best.json"
  done
done
gal_mode 1 "$bestck" "$FIN_GAMES" gal_1v1 > "$F/gal_1v1.json"; mega_mode 2 "$bestck" "$FIN_GAL" mega_2v2 > "$F/mega_2v2.json"
# Report-only continuity panel vs the TM9 final (400 g/mode). Never a gate.
if [[ -s ${TM9_PIN:-}/actor.bpk ]]; then
  for n in 2 3; do
    h2h_mode $n "$bestck" "$TM9_PIN" "${TM9_GAMES:-400}" h2h_vs_tm9_${n}v${n} > "$F/h2h_vs_tm9_${n}v${n}.json"
  done
else
  log "TM9 pin missing; skipped report-only TM9 panel"
fi
[[ -s $ST25_CK/actor.bpk ]] && ev h2h_1v1_st25 env INDOCOCO_TEAM_SIZE=1 "$B" h2h "$bestck" "$ST25_CK" "$FIN_GAMES" "$EV_THREADS" > "$F/h2h_1v1_st25.json"
timeout --kill-after=60 1800 nice -n 19 "$PYT" "$C/tools/team/tm8_audit/tm8_audit_traces.py" "nexto_2v2:a:$F/nexto_2v2.nexto.*.jsonl" "nexto_3v3:a:$F/nexto_3v3.nexto.*.jsonl" > "$F/kickoff_audit.jsonl" 2> "$F/kickoff_audit.err" || true
timeout --kill-after=60 1800 nice -n 19 "$PYT" "$C/tools/team/tm8_audit/tm8_audit_traces.py" "necto_2v2:a:$F/necto_2v2.necto.*.jsonl" "necto_3v3:a:$F/necto_3v3.necto.*.jsonl" > "$F/necto_kickoff_audit.jsonl" 2> "$F/necto_kickoff_audit.err" || true
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
