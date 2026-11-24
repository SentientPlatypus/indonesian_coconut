# shellcheck shell=bash
# Mid-eval and mid-bookkeeping helpers factored out of team_launch_tm11.sh.
# Bodies inside mid_evals / mid_book are byte-for-byte the TM11 loop
# (W=$X/mid/$now through the ST25 eval; then $EVP mid through guard.env).
# Sourced by team_launch_next.sh only. The TM11 launcher is not edited.
#
# CONCURRENT_MIDS=1: mid_evals of chunk k run while chunk k+1 trains.
# Bookkeeping (guard, rollback, PASSIVE_ALARM) then happens at the k+1
# boundary, so a rollback applies to chunk k+2 instead of k+1 — one chunk
# of extra latency, documented and accepted.

# shellcheck disable=SC2154,SC2034,SC2010,SC2086
# X, BT, ev, h2h_mode come from the launcher. EVAL_STUB_STEPS and PEND_* are
# written here for ev()/the launcher. latest()-style listing matches TM11.

mid_evals() {
  local ck=$1 now=$2
  if [[ "${EVAL_STUB:-0}" == 1 ]]; then
    EVAL_STUB_STEPS=$now
  fi
  W=$X/mid/$now; mkdir -p "$W"
  h2h_mode 2 "$ck" "$BT" "$MID_GAMES" h2h_2v2 > "$W/h2h_2v2.json"
  h2h_mode 3 "$ck" "$BT" "$MID_GAMES" h2h_3v3 > "$W/h2h_3v3.json"
  gal_mode 2 "$ck" "$GAL_GAMES" gal_2v2 > "$W/gal_2v2.json"; gal_mode 3 "$ck" "$GAL_GAMES" gal_3v3 > "$W/gal_3v3.json"
  gal_mode 1 "$ck" "$GAL1_GAMES" gal_1v1 > "$W/gal_1v1.json"
  mega_mode 2 "$ck" "$MEGA_GAMES" mega_2v2 > "$W/mega_2v2.json"
  for n in 2 3; do
    nf_mode $n nexto "$ck" "$NEXTO_MID" nexto_${n}v${n} "$W/nexto_${n}v${n}" > "$W/nexto_${n}v${n}.json"
    nf_mode $n necto "$ck" "$NECTO_MID" necto_${n}v${n} > "$W/necto_${n}v${n}.json"
  done
  if [[ "${EVAL_STUB:-0}" != 1 ]]; then
    AUDIT_PY=$C/tools/team/tm8_audit/tm8_audit_traces.py
    timeout --kill-after=60 1800 nice -n 19 "$PYT" "$AUDIT_PY" "nexto_2v2:a:$W/nexto_2v2.nexto.*.jsonl" "nexto_3v3:a:$W/nexto_3v3.nexto.*.jsonl" \
      > "$W/kickoff_audit.jsonl" 2> "$W/kickoff_audit.err" || log "kickoff audit failed at $now (see $W/kickoff_audit.err)"
  fi
  if [[ -s $ST25_CK/actor.bpk ]]; then ev h2h_1v1_st25 env INDOCOCO_TEAM_SIZE=1 "$B" h2h "$ck" "$ST25_CK" "$MID_GAMES" "$EV_THREADS" > "$W/h2h_1v1_st25.json"; else echo '{}' > "$W/h2h_1v1_st25.json"; fi
  if [[ "${EVAL_STUB:-0}" == 1 && ! -f "$W/kickoff_audit.jsonl" ]]; then
    : > "$W/kickoff_audit.jsonl"
  fi
  touch "$W/EVALS_DONE"
}

mid_book() {
  local ck=$1 now=$2
  W=$X/mid/$now
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
    nd=$X/checkpoints/$(date +%s)
    # Unique dir: tests (and a same-second collision) must not copy a ckpt onto itself
    # (that truncates stats.toml via sed redirect). One-line robustness over TM11.
    while [[ -d $nd ]]; do nd=$X/checkpoints/$((${nd##*/} + 1)); done
    mkdir -p $nd; cp $src/actor.bpk $src/critic.bpk $nd/
    sed -E "s/^cumulative_timesteps = .*/cumulative_timesteps = $now/" $src/stats.toml > $nd/stats.toml
    LOW=0; log "GUARD: 2 chunks with min(h2h 2v2, 3v3) < 0.40 vs BT; rolled back to $src as $nd"
  fi
  pa=$(NCR_BASE_2=${NCR_BASE_2:-} NCR_BASE_3=${NCR_BASE_3:-} H2H_ALARM=${H2H_ALARM:-} PASSIVE_BASE=${PASSIVE_BASE:-} \
       TOUCH_DROP=${TOUCH_DROP:-0.15} DEPTH_DROP=${DEPTH_DROP:-0.20} $EVP alarm "$X/logs/mid.jsonl" 2>>"$X/logs/eval_fail.log") || pa="alarm_script_failed"
  if [[ -n "$pa" ]]; then log "PASSIVE ALARM at $now (2 mids in a row): $pa"; echo "$now $pa" >> $X/PASSIVE_ALARM; fi
  echo LOW=$LOW > $X/guard.env
}

# True if mid.jsonl already has a line whose "steps" equals $1.
mid_jsonl_has_steps() {
  local steps=$1
  local f=$X/logs/mid.jsonl
  [[ -f $f ]] || return 1
  "$PYT" -c 'import json,sys
p,s=sys.argv[1],int(sys.argv[2])
try:
    lines=open(p)
except OSError:
    sys.exit(1)
for line in lines:
    line=line.strip()
    if not line: continue
    try:
        d=json.loads(line)
    except ValueError:
        continue
    if d.get("steps")==s:
        sys.exit(0)
sys.exit(1)
' "$f" "$steps"
}

# True if pid is still alive and its cmdline contains $X. Never kills it.
eval_pid_is_ours() {
  local pid=${1:-}
  [[ -n $pid && -d /proc/$pid ]] || return 1
  local cmd
  cmd=$("$PYT" -c 'import pathlib,sys
p=pathlib.Path("/proc")/sys.argv[1]/"cmdline"
sys.stdout.write(p.read_bytes().replace(b"\x00", b" ").decode("utf-8","replace"))
' "$pid" 2>/dev/null || true)
  [[ $cmd == *"$X"* ]]
}

# Mode-1 restart: finish any mid dir that has no mid.jsonl line yet.
# If EVAL_PID is still ours, wait on it (never kill). If EVALS_DONE exists,
# book it; otherwise re-run mid_evals in the background and leave PEND_* set.
mid_recover() {
  PEND_CK=; PEND_NOW=; MID_PID=
  local dir now ck
  [[ -d $X/mid ]] || return 0
  local nows
  nows=$(ls "$X"/mid 2>/dev/null | grep -E '^[0-9]+$' | sort -n || true)
  for now in $nows; do
    dir=$X/mid/$now
    # Skip garbage dirs such as mid/0 (created if a stub ev() defaulted STEPS).
    [[ "$now" =~ ^[1-9][0-9]+$ ]] || continue
    mid_jsonl_has_steps "$now" && continue
    ck=$X/checkpoints/$(ls "$X"/checkpoints | grep -E '^[0-9]+$' | while read -r d; do
      if [[ -f $X/checkpoints/$d/stats.toml ]] && grep -qE "^cumulative_timesteps = ${now}$" "$X/checkpoints/$d/stats.toml"; then
        echo "$d"
      fi
    done | tail -1)
    if [[ -z $ck || ! -d $ck || $ck == "$X/checkpoints/" || $ck == "$X/checkpoints" ]]; then
      ck=$X/checkpoints/$(latest)
    fi
    if [[ -z $ck || ! -d $ck || $ck == "$X/checkpoints/" || $ck == "$X/checkpoints" ]]; then
      log "MID_RECOVER skip $now (no checkpoint)"
      continue
    fi
    if [[ -f $dir/EVAL_PID ]]; then
      local epid
      epid=$(tr -d '[:space:]' < "$dir/EVAL_PID" || true)
      if eval_pid_is_ours "$epid"; then
        log "MID_RECOVER wait existing eval pid $epid for $now"
        wait "$epid" || true
      fi
    fi
    if [[ -f $dir/EVALS_DONE ]]; then
      log "MID_RECOVER book $now"
      mid_book "$ck" "$now"
    else
      log "MID_RECOVER relaunch evals for $now"
      ( mid_evals "$ck" "$now" ) &
      MID_PID=$!
      echo $MID_PID > "$X/mid/$now/EVAL_PID"
      PEND_CK=$ck
      PEND_NOW=$now
    fi
  done
}
