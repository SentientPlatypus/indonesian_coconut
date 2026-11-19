#!/usr/bin/env bash
# TM8: team_launch_tm7.sh with BEST-team pinned to the TM6 final (the current
# team_best, exp_TM6/checkpoints/1795044069) and two more passive-alarm rules
# (EXPERIMENTS.md "TM8 — Cross-exam note"): touches per car-minute (h2h vs
# BEST-team, live play) more than TOUCH_DROP (0.15) below TM7's final, or the
# deepest car sitting deeper: its mean distance to the own back wall more than
# DEPTH_DROP (0.20) below TM7's final, in 2v2 or 3v3, 2 mids in a row. The
# TM7 baseline is PASSIVE_BASE (written by tm8_after_tm7.sh from TM7's final
# checkpoint vs the same BEST-team with the TM8 binary); the rules are skipped
# until it exists. TM7's rules and thresholds are unchanged (h2h vs BEST-team
# < 0.50, no-challenger > 1.5x NCR_BASE, passes, deep, Nexto, near2). Mid
# lines add touch_<n>v<n> / depth_<n>v<n>. Started by tm8_window.sh from its
# own path (team_launch*.sh of earlier runs are not replaced).
# TM7: team_launch_tm6.sh plus one more passive-alarm rule (EXPERIMENTS.md
# "TM7 — Cross-exam note"): h2h vs BEST-team (TM4 cut) below H2H_ALARM
# (tm_TM7.env: 0.50) in 2v2 or 3v3 for 2 mids in a row (h2h2 / h2h3), next to
# TM6's rules (no-challenger above 1.5x NCR_BASE_2 / _3 = 0.061 / 0.049,
# passes, deep, Nexto, near2). Staged next to team_launch_tm6.sh; started by
# tm7_window.sh from its own path (team_launch.sh / team_launch_tm6.sh are not
# replaced).
# TM6: team_launch_tm5r.sh plus the TM6 role metrics in the mids (h2h vs
# BEST-team from a TM6+ binary: role swaps / min, first-man touch share,
# no-challenger rate = share of steps the opponents have the ball in our half
# with no teammate within 1500 uu) and one more passive-alarm rule: the
# no-challenger rate above 1.5x TM5r's baseline (NCR_BASE_2 / NCR_BASE_3 from
# tm_TM6.env) for 2 mids in a row. Staged next to team_launch.sh; the TM6
# cut-over runs it from its own path (team_launch.sh is not replaced).
# TM5r: team_launch_tm5.sh plus a passive-play alarm after each mid
# (EXPERIMENTS.md "TM5r"; log + $X/PASSIVE_ALARM only); otherwise identical.
# TM5 copy of team_launch_tm4pre.sh (docs/TM3_SPACING.md, EXPERIMENTS.md TM5).
# Only change: mid evals also log the live-play spacing metrics of the h2h
# vs BEST-team runs (team.a from a TM5+ binary; null on older binaries):
# pair800 (a teammate pair < 800 uu), near2 (2+ cars within 1500 uu of the
# ball), dcw (2+ within 1000 uu closing >= 200 uu/s), deep (a car goalside
# by >= 1500 uu), per 2v2 / 3v3 -> logs/mid.jsonl and the chunk log line.
# TM4-pre copy of team_launch.sh (EXPERIMENTS.md "TM4-pre"). Same chain,
# state and queue; differences:
# - Nexto / Necto in the league: NEXTO_WEIGHT / NECTO_WEIGHT (variant env) add
#   [[opponents.external]] kind = "nexto" / "necto" (optional NEXTO_STOCH /
#   NECTO_STOCH = share of episodes the bot samples). Needs a run binary (B0)
#   with src/nectofam.rs (tm4pre or later). Weights are for all team sizes:
#   the collector picks one opponent per rollout across all arenas.
# - Mid evals add Nexto and Necto in 2v2 / 3v3 (NF_MID_GAMES each, default
#   150) and finals at NF_FIN_GAMES (500) for the candidate and BEST-team, run
#   with EVAL_BIN (default: the run binary; `nexto-h2h` / `necto-h2h`, CPU
#   bots, kickoff spawn jitter).
# - Galactus / Megabyte evals also run on EVAL_BIN with GAL_JITTER=1 (spawn
#   jitter as in tools/pool_eval; without it a greedy policy vs deterministic
#   Galactus plays one game per kickoff spawn). GAL_JITTER=0: old protocol.
# - Candidate score (final pick): 0.5 * mean(h2h 2v2, h2h 3v3 vs BEST-team)
#   + 0.5 * mean of the external team win rates present (Nexto, Necto,
#   Galactus in 2v2 and 3v3). The 2v2 rollback guard is unchanged.
# - TEAM_PROMOTE.json also reports ext (the same mean) for the candidate and
#   BEST-team; the gate adds "ext not below BEST-team's by more than 0.04".
# Switch (between runs only, never under a running trainer): see EXPERIMENTS.md.
# TM team chain (2v2 / 3v3, EXPERIMENTS.md TM1). Separate from the 1v1 chain:
# its own queue (~/ippo_runs/TEAM_QUEUE), state (exp_<RUN>, <run>_final,
# ~/ippo_runs/team_best) and code copy (C, default ~/ippo_runs/code_tm). It
# never reads or writes ST_QUEUE, BEST/BESTF or any PROMOTE.json.
#   RUN=TM1 VARIANT=TM1 bash team_launch.sh      (resumes; run in its own tmux)
# Fresh start: `widen-team` turns START_CK (a 106-wide 1v1 checkpoint) into a
# 235-wide team-ext seed (actor + critic, new columns zero); follow-ups start
# from the previous final as is (START_IS_TEAM=1). TM3+ (OBS_PRED=1 in the
# variant env): INDOCOCO_OBS_PRED=1 for the run and every eval, and the
# 235-wide start is widened to 270 with `widen-pred`. Chunks of CHUNK
# steps; after each: Rust h2h vs BEST-team in 2v2 and 3v3 (MID_GAMES each,
# kickoffs, both sides) with team metrics, Galactus in 2v2 / 3v3 / 1v1 and
# Megabyte 2v2 -> logs/mid.jsonl. Guard: 2 chunks in a row < 40% in 2v2 vs
# BEST-team -> roll back to the best chunk. Finals (<run>_final): FIN_GAMES
# per mode vs BEST-team and Galactus, 1v1 Galactus, team gate ->
# TEAM_PROMOTE.json (a recommendation only: team_best is never changed by
# this script). Then pops the next "RUN VARIANT" line of TEAM_QUEUE; empty
# queue -> repeats as <RUN>r. Stop: touch ~/ippo_runs/TEAM_NO_FOLLOWUP.
set -u
R=${R:-/home/ubuntu/ippo_runs}; RUN=${RUN:-TM1}; VARIANT=${VARIANT:-TM1}; X=$R/exp_$RUN
C=${C:-$R/code_tm/indococo_ppo_bot}
THREADS=${THREADS:-24}; MID_GAMES=${MID_GAMES:-200}; FIN_GAMES=${FIN_GAMES:-500}; GAL_GAMES=${GAL_GAMES:-100}; EV_THREADS=${EV_THREADS:-8}
export INDOCOCO_OBS_EXT=1 INDOCOCO_OBS_TEAM=1
export LD_LIBRARY_PATH=$R/e8b2_1v1/bin/sp/torch/lib TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 INDOCOCO_METRICS=$X/logs/metrics.jsonl
unset RIPPLE_WEIGHT   # Ripple is a 1v1 bot: never in the team league
PYT=/home/ubuntu/coco-venv/bin/python; SNAP=$R/fr7_snap
GAL_DIR=${GAL_DIR:-/home/ubuntu/eval_bots/galactus/rust}; MEGA_DIR=${MEGA_DIR:-/home/ubuntu/eval_bots/megabyte}
# TM2: 1v1 h2h vs the 1v1 BEST (ST25, 106-wide, remapped on load) at every mid eval.
ST25_CK=${ST25_CK:-$R/exp_ST25/checkpoints/1794413355}
# TM4-pre: Nexto / Necto evals (and jittered Galactus / Megabyte) on EVAL_BIN.
NEXTO_DIR=${NEXTO_DIR:-/home/ubuntu/eval_bots/nexto}; NECTO_DIR=${NECTO_DIR:-/home/ubuntu/eval_bots/necto}
NF_MID_GAMES=${NF_MID_GAMES:-150}; NF_FIN_GAMES=${NF_FIN_GAMES:-500}; GAL_JITTER=${GAL_JITTER:-1}
mkdir -p $X/logs $X/checkpoints $X/opponents
log() { echo "$(date '+%F %T %Z') $*" | tee -a $X/logs/train.log; }
die() { log "FATAL: $*"; exit 1; }
source $C/experiments/tm_$VARIANT.env || die "no tm_$VARIANT.env"
# TM3: the prediction obs is per variant (explicit, so a re-exec never inherits it).
export INDOCOCO_OBS_PRED=${OBS_PRED:-0}
steps_in() { sed -nE 's/^cumulative_timesteps = ([0-9]+)/\1/p' $1/stats.toml; }
latest() { ls $X/checkpoints | grep -E '^[0-9]+$' | sort -n | tail -1; }
jget() { $PYT -c "import json,sys
try:
  d=json.loads(sys.stdin.read().strip().splitlines()[-1]); k='$1'.split('.')
  for p in k: d=d[p]
  print(d)
except Exception: print('null')"; }
if ! ls $X/checkpoints 2>/dev/null | grep -qE '^[0-9]+$'; then
  [[ -s $START_CK/critic.bpk ]] || die "START_CK $START_CK has no critic"
  START=$(steps_in $START_CK)
  INIT=$(basename "$START_CK"); [[ "$INIT" =~ ^[0-9]+$ ]] && (( INIT < $(date +%s) )) || INIT=$(( $(date +%s) - 60 ))
  mkdir -p $X/cfg; cp -r $C/experiments $C/config $X/cfg/; cp $B0 $X/indococo-train
  log "$RUN ($VARIANT) start from $START_CK ($START), team seed (widen-team unless START_IS_TEAM=1)"
  if [[ "${START_IS_TEAM:-0}" == 1 && "$INDOCOCO_OBS_PRED" == 1 ]]; then   # TM3: 235 -> 270 (or copy if 270)
    ( cd /home/ubuntu/indonesian_coconut && nice -n 19 $X/indococo-train widen-pred $START_CK $X/seed ) > $X/logs/widen.log 2>&1 || die "widen-pred failed"
    grep WIDEN_PRED $X/logs/widen.log | tee -a $X/logs/train.log
  elif [[ "${START_IS_TEAM:-0}" == 1 ]]; then   # follow-up: already team-ext
    mkdir -p $X/seed; cp $START_CK/actor.bpk $START_CK/critic.bpk $START_CK/stats.toml $X/seed/
  else
    ( cd /home/ubuntu/indonesian_coconut && nice -n 19 $X/indococo-train widen-team $START_CK $X/seed ) > $X/logs/widen.log 2>&1 || die "widen-team failed"
    grep WIDEN_TEAM $X/logs/widen.log | tee -a $X/logs/train.log
  fi
  mkdir -p $X/checkpoints/$INIT; cp $X/seed/actor.bpk $X/seed/critic.bpk $X/checkpoints/$INIT/
  sed -E "s/^cumulative_timesteps = .*/cumulative_timesteps = $START/" $X/seed/stats.toml > $X/checkpoints/$INIT/stats.toml
  for e in ${EXTRA_OPPS:-}; do n=${e%%:*}; r=${e#*:}; d=${r%:*}
    mkdir -p $X/opponents/$n; cp $d/actor.bpk $X/opponents/$n/ || die "extra opp $e"; done
  echo "$START_CK $START $VARIANT" > $X/START; echo LOW=0 > $X/guard.env
  # BEST-team: the shared team reference. First run: the widened seed (= the 1v1 policy).
  [[ -s $R/team_best/FINAL_CKPT ]] || die "team_best missing: never initialize or change it without approval"
fi
B=$X/indococo-train
# Eval binary for Galactus / Megabyte / Nexto / Necto: the run's own copy by
# default (B0 must then be tm4pre or later); EVAL_BIN=<binary> to override.
EVAL_BIN=${EVAL_BIN:-$B}
read _ START0 _ < $X/START; END=$((START0 + STEPS))
BT=$(cat $R/team_best/FINAL_CKPT)
[[ "$BT" == "${BT_PIN:-$R/exp_TM6/checkpoints/1795044069}" ]] || die "BT must stay the TM6 final (team_best), got $BT"
# Reward config: the team template with the team-spirit origin = run start.
printf 'extends = "%s"\n[team]\nspirit_origin = %s.0\n' "$X/cfg/experiments/${REWARD_TMPL:-tm1.toml}" "$START0" > $X/tm_rewards.toml
h2h_mode() { # n a b games -> json line
  ( cd /home/ubuntu/indonesian_coconut && INDOCOCO_TEAM_SIZE=$1 nice -n 19 $B h2h $2 $3 $4 $EV_THREADS 2>/dev/null | tail -1 ); }
[[ -x $EVAL_BIN ]] || die "EVAL_BIN $EVAL_BIN missing"
gal_mode() { ( cd /home/ubuntu/indonesian_coconut && H2H_JITTER=$GAL_JITTER INDOCOCO_TEAM_SIZE=$1 nice -n 19 $EVAL_BIN galactus-h2h $2 $GAL_DIR $3 $EV_THREADS 2>/dev/null | tail -1 ); }
mega_mode() { ( cd /home/ubuntu/indonesian_coconut && H2H_JITTER=$GAL_JITTER INDOCOCO_TEAM_SIZE=$1 nice -n 19 $EVAL_BIN megabyte-h2h $2 $MEGA_DIR $3 $EV_THREADS 2>/dev/null | tail -1 ); }
nf_mode() { # n kind ckpt games -> json line (CPU bots, greedy, jittered spawns)
  local d=$NEXTO_DIR; [[ $2 == necto ]] && d=$NECTO_DIR
  ( cd /home/ubuntu/indonesian_coconut && INDOCOCO_TEAM_SIZE=$1 nice -n 19 $EVAL_BIN $2-h2h $3 $d $4 $EV_THREADS 2>/dev/null | tail -1 ); }
# Candidate score (TM4-pre): 0.5 * h2h mean + 0.5 * mean of the external team rates present.
SCORE_PY='
def score(d):
    h=((d.get("h2h_2v2") or 0)+(d.get("h2h_3v3") or 0))/2
    ext=[d[k] for k in ("nexto_2v2","nexto_3v3","necto_2v2","necto_3v3","gal_2v2","gal_3v3") if d.get(k) is not None]
    return 0.5*h+0.5*(sum(ext)/len(ext) if ext else 0)
'

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
  [[ -n "${NEXTO_WEIGHT:-}" ]] && printf '\n[[opponents.external]]\nname = "nexto"\nkind = "nexto"\npath = "%s"\nweight = %s\nstochastic = %s\n' "$NEXTO_DIR" "$NEXTO_WEIGHT" "${NEXTO_STOCH:-0.0}" >> $X/train_tm.toml
  [[ -n "${NECTO_WEIGHT:-}" ]] && printf '\n[[opponents.external]]\nname = "necto"\nkind = "necto"\npath = "%s"\nweight = %s\nstochastic = %s\n' "$NECTO_DIR" "$NECTO_WEIGHT" "${NECTO_STOCH:-0.0}" >> $X/train_tm.toml
  for e in ${EXTRA_OPPS:-}; do n=${e%%:*}
    printf '\n[[opponents.fixed]]\nname = "%s"\ncheckpoint = "%s"\nweight = %s\n' "$n" "$X/opponents/$n" "${e##*:}" >> $X/train_tm.toml; done
  log "$RUN train: $left iterations from $now"
  ( cd /home/ubuntu/indonesian_coconut && nice -n 19 $B train $X/train_tm.toml ) 2>&1 | tee -a $X/logs/train.log
  [[ ${PIPESTATUS[0]} -eq 0 ]] || die "train failed"
  prev=$now; ck=$X/checkpoints/$(latest); now=$(steps_in $ck)
  (( now > prev )) || die "steps did not advance ($prev -> $now)"
  W=$X/mid/$now; mkdir -p $W
  h2h_mode 2 $ck $BT $MID_GAMES > $W/h2h_2v2.json; h2h_mode 3 $ck $BT $MID_GAMES > $W/h2h_3v3.json
  gal_mode 2 $ck $GAL_GAMES > $W/gal_2v2.json; gal_mode 3 $ck $GAL_GAMES > $W/gal_3v3.json; gal_mode 1 $ck $((2 * GAL_GAMES)) > $W/gal_1v1.json
  mega_mode 2 $ck $GAL_GAMES > $W/mega_2v2.json
  for n in 2 3; do for k in nexto necto; do nf_mode $n $k $ck $NF_MID_GAMES > $W/${k}_${n}v${n}.json; done; done
  if [[ -s $ST25_CK/actor.bpk ]]; then h2h_mode 1 $ck $ST25_CK $MID_GAMES > $W/h2h_1v1_st25.json; else echo "{}" > $W/h2h_1v1_st25.json; fi
  w2=$(jget win_rate_decisive < $W/h2h_2v2.json); w3=$(jget win_rate_decisive < $W/h2h_3v3.json)
  g2=$(jget win_rate_decisive < $W/gal_2v2.json); g3=$(jget win_rate_decisive < $W/gal_3v3.json); g1=$(jget win_rate_decisive < $W/gal_1v1.json)
  m2=$(jget win_rate_decisive < $W/mega_2v2.json); s1=$(jget win_rate_decisive < $W/h2h_1v1_st25.json)
  x2=$(jget win_rate_decisive < $W/nexto_2v2.json); x3=$(jget win_rate_decisive < $W/nexto_3v3.json)
  c2=$(jget win_rate_decisive < $W/necto_2v2.json); c3=$(jget win_rate_decisive < $W/necto_3v3.json)
  p2=$(jget team.a.passes_per_min < $W/h2h_2v2.json); p3=$(jget team.a.passes_per_min < $W/h2h_3v3.json)
  d2=$(jget team.a.double_commit_ratio < $W/h2h_2v2.json); d3=$(jget team.a.double_commit_ratio < $W/h2h_3v3.json)
  q2=$(jget team.a.pair_lt800_ratio < $W/h2h_2v2.json); q3=$(jget team.a.pair_lt800_ratio < $W/h2h_3v3.json)
  n2=$(jget team.a.two_near_ball1500_ratio < $W/h2h_2v2.json); n3=$(jget team.a.two_near_ball1500_ratio < $W/h2h_3v3.json)
  e2=$(jget team.a.double_commit_wide_ratio < $W/h2h_2v2.json); e3=$(jget team.a.double_commit_wide_ratio < $W/h2h_3v3.json)
  l2=$(jget team.a.deep_last_man_ratio < $W/h2h_2v2.json); l3=$(jget team.a.deep_last_man_ratio < $W/h2h_3v3.json)
  r2=$(jget team.a.no_challenger_ratio < $W/h2h_2v2.json); r3=$(jget team.a.no_challenger_ratio < $W/h2h_3v3.json)
  o2=$(jget team.a.role_swaps_per_min < $W/h2h_2v2.json); o3=$(jget team.a.role_swaps_per_min < $W/h2h_3v3.json)
  f2=$(jget team.a.touch_share_first < $W/h2h_2v2.json); f3=$(jget team.a.touch_share_first < $W/h2h_3v3.json)
  t2=$(jget team.a.first_mean_stint_s < $W/h2h_2v2.json); t3=$(jget team.a.first_mean_stint_s < $W/h2h_3v3.json)
  u2=$(jget team.a.touches_per_car_min < $W/h2h_2v2.json); u3=$(jget team.a.touches_per_car_min < $W/h2h_3v3.json)
  y2=$(jget team.a.deepest_car_depth_uu < $W/h2h_2v2.json); y3=$(jget team.a.deepest_car_depth_uu < $W/h2h_3v3.json)
  echo "{\"steps\":$now,\"ckpt\":\"$ck\",\"h2h_2v2\":$w2,\"h2h_3v3\":$w3,\"gal_2v2\":$g2,\"gal_3v3\":$g3,\"gal_1v1\":$g1,\"mega_2v2\":$m2,\"h2h_1v1_st25\":$s1,\"passes_min_2v2\":$p2,\"passes_min_3v3\":$p3,\"double_commit_2v2\":$d2,\"double_commit_3v3\":$d3,\"nexto_2v2\":$x2,\"nexto_3v3\":$x3,\"necto_2v2\":$c2,\"necto_3v3\":$c3,\"pair800_2v2\":$q2,\"pair800_3v3\":$q3,\"near2_2v2\":$n2,\"near2_3v3\":$n3,\"dcw_2v2\":$e2,\"dcw_3v3\":$e3,\"deep_2v2\":$l2,\"deep_3v3\":$l3,\"ncr_2v2\":$r2,\"ncr_3v3\":$r3,\"swaps_min_2v2\":$o2,\"swaps_min_3v3\":$o3,\"first_touch_2v2\":$f2,\"first_touch_3v3\":$f3,\"first_stint_2v2\":$t2,\"first_stint_3v3\":$t3,\"touch_2v2\":$u2,\"touch_3v3\":$u3,\"depth_2v2\":$y2,\"depth_3v3\":$y3,\"gal_jitter\":$GAL_JITTER}" >> $X/logs/mid.jsonl
  sc=$(tail -1 $X/logs/mid.jsonl | $PYT -c "import json,sys
$SCORE_PY
print(round(score(json.loads(sys.stdin.read())),4))")
  log "$RUN chunk done at $now: vs BEST-team 2v2 $w2 3v3 $w3 | Galactus 2v2 $g2 3v3 $g3 1v1 $g1 | Megabyte 2v2 $m2 | Nexto 2v2 $x2 3v3 $x3 | Necto 2v2 $c2 3v3 $c3 | 1v1 vs ST25 $s1 | score $sc | passes/min $p2 / $p3, double commit $d2 / $d3 | spacing pair800 $q2 / $q3, near2 $n2 / $n3, dcw $e2 / $e3, deep $l2 / $l3 | roles no-challenger $r2 / $r3, swaps/min $o2 / $o3, first-man touch share $f2 / $f3, first stint s $t2 / $t3 | touches/car-min $u2 / $u3, deepest-car depth $y2 / $y3"
  if [[ "$w2" != null ]] && $PYT -c "import sys;sys.exit(0 if $w2 < 0.40 else 1)"; then LOW=$((LOW + 1)); else LOW=0; fi
  if (( LOW >= 2 )); then
    src=$($PYT - $X/logs/mid.jsonl <<'PY'
import json,sys
b=None
for l in open(sys.argv[1]):
    d=json.loads(l); s=((d.get("h2h_2v2") or 0)+(d.get("h2h_3v3") or 0))/2
    if b is None or s>b[0]: b=(s,d["ckpt"])
print(b[1] if b else "")
PY
)
    [[ -n "$src" && -d "$src" ]] || src=$X/seed
    nd=$X/checkpoints/$(date +%s); mkdir -p $nd; cp $src/actor.bpk $src/critic.bpk $nd/
    sed -E "s/^cumulative_timesteps = .*/cumulative_timesteps = $now/" $src/stats.toml > $nd/stats.toml
    LOW=0; log "GUARD: 2 chunks < 40% vs BEST-team (2v2); rolled back to $src as $nd"
  fi
  # TM5r passive-play alarm (log only; never changes the run): over the last 2
  # mids in a row, any of: passes/min below 70% of TM4's (2v2 < 0.85, 3v3 <
  # 1.30); deep last man above Nexto's (2v2 > 0.56, 3v3 > 0.73) while h2h vs
  # BEST-team < 0.48 in that mode; Nexto win rate clearly below TM5's
  # (TM5 mids 2v2 0.43-0.52, 3v3 0.24-0.32: 2v2 < 0.40, 3v3 < 0.20); near2 (2+ cars within 1500 of the ball) below half of
  # Nexto's (2v2 < 0.06, 3v3 < 0.10). Writes $X/PASSIVE_ALARM for the routine.
  # TM6: also the no-challenger rate above 1.5x TM5r's (NCR_BASE_2 / _3).
  pa=$(tail -2 $X/logs/mid.jsonl | NCR_BASE_2=${NCR_BASE_2:-} NCR_BASE_3=${NCR_BASE_3:-} H2H_ALARM=${H2H_ALARM:-} PASSIVE_BASE=${PASSIVE_BASE:-} TOUCH_DROP=${TOUCH_DROP:-0.15} DEPTH_DROP=${DEPTH_DROP:-0.20} $PYT -c "
import json,sys
L=[json.loads(l) for l in sys.stdin if l.strip()]
def f(d):
    g=lambda k: d.get(k) if d.get(k) is not None else float('nan')
    r=[]
    if g('passes_min_2v2')<0.85: r.append('passes2')
    if g('passes_min_3v3')<1.30: r.append('passes3')
    if g('deep_2v2')>0.56 and g('h2h_2v2')<0.48: r.append('deep2')
    if g('deep_3v3')>0.73 and g('h2h_3v3')<0.48: r.append('deep3')
    if g('nexto_2v2')<0.40: r.append('nexto2')
    if g('nexto_3v3')<0.20: r.append('nexto3')
    if g('near2_2v2')<0.06: r.append('near2_2')
    if g('near2_3v3')<0.10: r.append('near2_3')
    import os
    for n in (2,3):
        b=os.environ.get(f'NCR_BASE_{n}') or ''
        if b and g(f'ncr_{n}v{n}')>1.5*float(b): r.append(f'ncr{n}')
    h=os.environ.get('H2H_ALARM') or ''
    for n in (2,3):
        if h and g(f'h2h_{n}v{n}')<float(h): r.append(f'h2h{n}')
    # TM8 passivity vs TM7's final (PASSIVE_BASE json: touch_<n>v<n>, depth_<n>v<n>).
    pb=os.environ.get('PASSIVE_BASE') or ''
    try: base=json.load(open(pb)) if pb else {}
    except Exception: base={}
    td=float(os.environ.get('TOUCH_DROP') or 0.15); dd=float(os.environ.get('DEPTH_DROP') or 0.20)
    for n in (2,3):
        bt=base.get(f'touch_{n}v{n}'); bd=base.get(f'depth_{n}v{n}')
        if bt and g(f'touch_{n}v{n}')<(1-td)*bt: r.append(f'touch{n}')
        if bd and g(f'depth_{n}v{n}')<(1-dd)*bd: r.append(f'depth{n}')
    return set(r)
both=f(L[0])&f(L[1]) if len(L)==2 else set()
print(','.join(sorted(both)))" 2>/dev/null)
  if [[ -n "$pa" ]]; then log "PASSIVE ALARM at $now (2 mids in a row): $pa"; echo "$now $pa" >> $X/PASSIVE_ALARM; fi
  echo LOW=$LOW > $X/guard.env
done
touch $X/TRAIN_DONE
# Final candidate (TM4-pre): best score() chunk (ties -> later), else latest.
bestck=$($PYT -c "import json,sys
$SCORE_PY
b=None
for l in open(sys.argv[1]):
    d=json.loads(l); s=score(d)
    if b is None or s>=b[0]: b=(s,d['ckpt'])
print(b[1] if b else '')" $X/logs/mid.jsonl 2>/dev/null)
[[ -n "$bestck" && -d "$bestck" ]] || bestck=$X/checkpoints/$(latest)
F=$R/${RUN,,}_final; mkdir -p $F; echo $bestck > $F/FINAL_CKPT; echo "$(steps_in $bestck)" > $F/STEPS; echo $BT > $F/BEST_TEAM
$B export $bestck $F/final_policy.safetensors > /dev/null 2>&1; sha256sum $F/final_policy.safetensors > $F/sha.txt 2>/dev/null
for n in 2 3; do
  h2h_mode $n $bestck $BT $FIN_GAMES > $F/h2h_${n}v${n}.json
  gal_mode $n $bestck $((FIN_GAMES / 2)) > $F/gal_${n}v${n}.json
  gal_mode $n $BT $((FIN_GAMES / 2)) > $F/gal_${n}v${n}_best.json
  for k in nexto necto; do
    nf_mode $n $k $bestck $NF_FIN_GAMES > $F/${k}_${n}v${n}.json
    nf_mode $n $k $BT $NF_FIN_GAMES > $F/${k}_${n}v${n}_best.json
  done
done
gal_mode 1 $bestck $FIN_GAMES > $F/gal_1v1.json; mega_mode 2 $bestck $((FIN_GAMES / 2)) > $F/mega_2v2.json
[[ -s $ST25_CK/actor.bpk ]] && h2h_mode 1 $bestck $ST25_CK $FIN_GAMES > $F/h2h_1v1_st25.json
# Team gate (recommendation only): mean h2h >= 0.52 with each mode >= 0.48,
# Galactus per mode not below BEST-team's by more than 0.04, and (TM4-pre) the
# external mean (Nexto, Necto, Galactus x 2v2 / 3v3) not below BEST-team's by
# more than 0.04.
$PYT - $F > $F/TEAM_PROMOTE.json <<'PY'
import json,sys,os
F=sys.argv[1]
def wr(n):
    try: return json.loads(open(os.path.join(F,n)).read().strip().splitlines()[-1])["win_rate_decisive"]
    except Exception: return None
h2,h3=wr("h2h_2v2.json"),wr("h2h_3v3.json"); g2,g3=wr("gal_2v2.json"),wr("gal_3v3.json"); b2,b3=wr("gal_2v2_best.json"),wr("gal_3v3_best.json")
ok=None not in (h2,h3) and (h2+h3)/2>=0.52 and min(h2,h3)>=0.48
for g,b in ((g2,b2),(g3,b3)):
    if g is not None and b is not None and g < b-0.04: ok=False
def ext(sfx):
    v=[wr(f"{k}_{n}v{n}{sfx}.json") for k in ("nexto","necto","gal") for n in (2,3)]
    v=[x for x in v if x is not None]
    return sum(v)/len(v) if v else None
e,eb=ext(""),ext("_best")
if e is not None and eb is not None and e < eb-0.04: ok=False
nf={f"{k}_{n}v{n}{s}":wr(f"{k}_{n}v{n}{s}.json") for k in ("nexto","necto") for n in (2,3) for s in ("","_best")}
score=None if None in (h2,h3) or e is None else round(0.5*(h2+h3)/2+0.5*e,4)
print(json.dumps({"promote_team":bool(ok),"score":score,"ext":e,"best_ext":eb,**nf,"h2h_2v2":h2,"h2h_3v3":h3,"gal_2v2":g2,"gal_3v3":g3,"best_gal_2v2":b2,"best_gal_3v3":b3,"gal_1v1":wr("gal_1v1.json"),"mega_2v2":wr("mega_2v2.json"),"h2h_1v1_st25":wr("h2h_1v1_st25.json"),"note":"recommendation only; team_best/FINAL_CKPT is changed by hand"}))
PY
touch $F/DONE; log "$RUN finals: $(cat $F/TEAM_PROMOTE.json)"
if [[ ! -f $R/TEAM_NO_FOLLOWUP ]]; then
  Q=$R/TEAM_QUEUE; next=$(head -1 $Q 2>/dev/null)
  if [[ -n "$next" ]]; then sed -i 1d $Q; else next="${RUN}r $VARIANT"; fi
  log "follow-up ${next} from $bestck (BEST-team $BT)"
  RUN=${next%% *} VARIANT=${next#* } START_CK=$bestck START_IS_TEAM=1 exec bash "${BASH_SOURCE[0]}"
fi
