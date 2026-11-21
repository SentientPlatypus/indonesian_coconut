#!/usr/bin/env bash
# TM10 after TM9 (EXPERIMENTS.md "TM10 — Cross-exam note"; user-approved Nov
# 21 2026: "Put that in tm10 and run tm10"). Run detached, independent of ssh:
#   mkdir -p /opt/dlami/nvme/tm10_work/cut && setsid nohup bash \
#     /home/ubuntu/ippo_runs/code_tm/indococo_ppo_bot/experiments/tm10_after_tm9.sh \
#     >> /opt/dlami/nvme/tm10_work/cut/cutover.log 2>&1 < /dev/null &
# 1. TEAM_NO_FOLLOWUP stays set so TM9's launcher does not chain a follow-up
#    after its own finals (it still runs them, recommendation only), and
#    TEAM_QUEUE must be empty (BLOCKED otherwise).
# 2. Waits for TM9's trainer to exit normally: the latest checkpoint at or past
#    END (= TM9 START + 5B), no `indococo-train train` process of exp_TM9, the
#    mid directory proving the launcher saw trainer exit 0, and no FATAL in
#    TM9's train.log. If TM9's launcher dies before END, logs BLOCKED and exits
#    without starting TM10.
# 3. Starts TM10 at once in tmux ippo_team window tm10 (tm10_window.sh, from
#    TM9's last checkpoint at END, not a mid-best), then, one job at a time at
#    nice 19 on CORES:
#    a. exports TM9's last checkpoint actor as checkpoints_to_test/tm9_<N>M.bpk
#       and pushes it to indococo-ppo (not promoted);
#    b. passivity baseline for TM10's alarm: h2h TM9 final vs BEST-team (TM7
#       final) 200 games per mode with the TM10 binary -> $D/passive_base.json;
#    c. after TM9's launcher has exited (its own finals), TM9 finals: h2h vs
#       TM7 final (team_best) and vs TM8 final (previous run), 400 games per
#       mode, Nexto 300 and Necto 300 per mode, spacing / role / passivity
#       metrics, plus a Nexto kickoff audit (504 games, TRAJ_DUMP) -> $D/finals/;
#    d. after TM10's first mid line, logs the playmaking / demo / bump metrics
#       of TM10's training log (metrics.jsonl).
# Process checks use patterns anchored at the start of the command line
# (pgrep -f '^...'), so this script never matches itself. Never kills
# anything. Never touches tmux indococo / ippo_1v1, team_best, the RLBot pack,
# rlbot_submission/, 1v1.pt / 2v2.pt / 3v3.pt, or the running TM9.
set -u
R=/home/ubuntu/ippo_runs; X9=$R/exp_TM9; X10=$R/exp_TM10; C=$R/code_tm/indococo_ppo_bot/experiments
REPO=/opt/dlami/nvme/ic_ppo; D=/opt/dlami/nvme/tm10_work/cut; F=$D/finals
B10=${B10:-/opt/dlami/nvme/ippo_bins/indococo-train-tm10-cf02e2b}
B10SHA=${B10SHA:-b29f90fe9786842a83bdd911453142d0af75084043e69bea03582ebad3812ba8}
CORES=${CORES:-28-31}
TM7=$R/exp_TM7/checkpoints/1795170349
TM8=$R/exp_TM8/checkpoints/1795278408
AUDIT_PY=${AUDIT_PY:-/opt/dlami/nvme/ic_ppo/indococo_ppo_bot/tools/team/tm8_audit/tm8_audit_traces.py}
[[ -s $AUDIT_PY ]] || AUDIT_PY=/opt/dlami/nvme/tm8_audit/tm8_audit_traces.py
PY=/home/ubuntu/coco-venv/bin/python
mkdir -p $D $F
log() { echo "$(date '+%F %T %Z') $*"; }
steps_in() { sed -nE 's/^cumulative_timesteps = ([0-9]+)/\1/p' $1/stats.toml; }
latest() { ls $X9/checkpoints | grep -E '^[0-9]+$' | sort -n | tail -1; }
trainer_up() { pgrep -f "^$X9/indococo-train train " > /dev/null; }
launcher_pid() { pgrep -f "^bash $C/team_launch_tm9.sh" | head -1; }
for f in team_launch_tm10.sh tm10_window.sh tm_TM10.env tm10.toml train_tm10.toml tm9.toml train_tm9.toml tm8.toml; do [[ -s $C/$f ]] || { log "BLOCKED: $C/$f missing"; exit 1; }; done
[[ -x $B10 ]] || { log "BLOCKED: TM10 binary $B10 missing"; exit 1; }
[[ "$(sha256sum $B10 | cut -d' ' -f1)" == "$B10SHA" ]] || { log "BLOCKED: TM10 binary sha256 mismatch"; exit 1; }
[[ "$(cat $R/team_best/FINAL_CKPT)" == "$TM7" ]] || { log "BLOCKED: team_best is not the TM7 final ($(cat $R/team_best/FINAL_CKPT))"; exit 1; }
touch $R/TEAM_NO_FOLLOWUP
[[ -s $R/TEAM_QUEUE ]] && { log "BLOCKED: TEAM_QUEUE not empty ($(head -1 $R/TEAM_QUEUE)); TM10 not armed"; exit 1; }
read _ START0 _ < $X9/START; END=$((START0 + 5000000000))
LP=$(launcher_pid); TP=$(pgrep -f "^$X9/indococo-train train " | head -1)
log "armed (pid $$): waiting for TM9 (launcher pid ${LP:-none}, current trainer pid ${TP:-none}) to reach END $END; TM10 binary $B10 ($B10SHA)"
[[ -n "$LP" ]] || { log "BLOCKED: TM9 launcher not found at arm time"; exit 1; }
while :; do
  now=$(steps_in $X9/checkpoints/$(latest))
  # The mid directory is created ONLY after the launcher checked the
  # trainer pipeline exit status == 0. This is the normal-exit receipt.
  if (( now >= END )) && ! trainer_up && [[ -d $X9/mid/$now ]]; then break; fi
  if [[ -z "$(launcher_pid)" ]] && ! trainer_up; then
    log "BLOCKED: TM9 launcher gone at $now < END $END (train.log: $(grep FATAL $X9/logs/train.log | tail -1)); TM10 not started"; exit 1
  fi
  if grep -q "FATAL" $X9/logs/train.log; then log "BLOCKED: FATAL in TM9 train.log: $(grep FATAL $X9/logs/train.log | tail -1); TM10 not started"; exit 1; fi
  sleep 30
done
ck=$X9/checkpoints/$(latest); now=$(steps_in $ck)
if grep -q "FATAL" $X9/logs/train.log; then log "BLOCKED: FATAL in TM9 train.log: $(grep FATAL $X9/logs/train.log | tail -1); TM10 not started"; exit 1; fi
log "TM9 trainer exited normally at $now ($ck)"
# 3. TM10 (the box never idles). Warm start = this END ckpt, not a mid-best.
if [[ -d $X10 ]]; then log "BLOCKED: $X10 already exists; TM10 not started"; exit 1; fi
tmux has-session -t ippo_team 2>/dev/null || tmux new-session -d -s ippo_team -n tm10_hold "sleep 60"
tmux new-window -d -t ippo_team -n tm10 "bash $C/tm10_window.sh $ck"
sleep 120; log "TM10 started: $(pgrep -af "^$X10/indococo-train train" | head -1 || echo 'no trainer yet'); $(tail -2 $X10/logs/train.log 2>/dev/null | tr '\n' ' ')"
pgrep -f "^$X10/indococo-train train" > /dev/null || log "WARNING: no TM10 trainer 120 s after start (see $X10/logs/train.log; tm10_window.sh falls back to TM9r)"
export LD_LIBRARY_PATH=$R/e8b2_1v1/bin/sp/torch/lib INDOCOCO_OBS_EXT=1 INDOCOCO_OBS_TEAM=1 INDOCOCO_OBS_PRED=1
cd /home/ubuntu/indonesian_coconut
# 3a. bpk push of the TM9 final (not promoted).
N=$(( (now + 500000) / 1000000 )); bpk=checkpoints_to_test/tm9_${N}M.bpk
(
  cd $REPO && git pull -q --rebase origin indococo-ppo && cp $ck/actor.bpk $bpk.tmp && mv $bpk.tmp $bpk && git add $bpk \
  && git -c user.name=SentientPlatypus -c user.email=83039642+SentientPlatypus@users.noreply.github.com commit -q \
       -m "checkpoints_to_test: add tm9_${N}M (TM9 final checkpoint at 5B, not promoted; 270-wide pred obs)" \
  && git push -q origin HEAD:indococo-ppo && log "pushed $bpk @ $(git rev-parse --short HEAD) (sha256 $(sha256sum $bpk | cut -c1-16))"
) || log "bpk push FAILED for $bpk (left for a manual push)"
# 3b. Passivity baseline (TM9 final vs TM7 final, TM10 binary).
for n in 2 3; do
  INDOCOCO_TEAM_SIZE=$n nice -n 19 taskset -c $CORES $B10 h2h $ck $TM7 200 4 2>/dev/null | tail -1 > $D/base_h2h_${n}v${n}.json
done
$PY - $D $ck <<'J1' > $D/passive_base.json.tmp && mv $D/passive_base.json.tmp $D/passive_base.json
import json,os,sys
D,ck=sys.argv[1],sys.argv[2]; out={"ckpt":ck}
for n in (2,3):
    a=json.loads(open(os.path.join(D,f"base_h2h_{n}v{n}.json")).read().strip().splitlines()[-1])["team"]["a"]
    out[f"touch_{n}v{n}"]=a["touches_per_car_min"]; out[f"depth_{n}v{n}"]=a["deepest_car_depth_uu"]
print(json.dumps(out))
J1
log "passivity baseline: $(cat $D/passive_base.json 2>/dev/null || echo MISSING)"
# 3c. TM9 finals after TM9's launcher finished its own evals.
while [[ -n "$(launcher_pid)" ]]; do sleep 60; done
log "TM9 launcher exited ($(cat $R/tm9_final/TEAM_PROMOTE.json 2>/dev/null | head -c 300)); TM9 finals on $ck"
echo "$ck vs TM7=$TM7 vs TM8=$TM8" > $F/PAIRS
for n in 2 3; do
  INDOCOCO_TEAM_SIZE=$n nice -n 19 taskset -c $CORES $B10 h2h $ck $TM7 400 4 2>/dev/null | tail -1 > $F/h2h_tm7_${n}v${n}.json
  INDOCOCO_TEAM_SIZE=$n nice -n 19 taskset -c $CORES $B10 h2h $ck $TM8 400 4 2>/dev/null | tail -1 > $F/h2h_tm8_${n}v${n}.json
  for k in nexto necto; do
    INDOCOCO_TEAM_SIZE=$n nice -n 19 taskset -c $CORES $B10 $k-h2h $ck /home/ubuntu/eval_bots/$k 300 4 2>/dev/null | tail -1 > $F/${k}_${n}v${n}.json
  done
  INDOCOCO_TEAM_SIZE=$n TRAJ_DUMP=$F/tm9_nexto_${n}v${n} OMP_NUM_THREADS=1 nice -n 19 taskset -c $CORES $B10 nexto-h2h $ck /home/ubuntu/eval_bots/nexto 504 4 > $F/tm9_nexto_${n}v${n}.result.json 2>$F/tm9_nexto_${n}v${n}.stderr
done
$PY $AUDIT_PY \
  "tm9_nexto_2v2:a:$F/tm9_nexto_2v2.nexto.*.jsonl" "tm9_nexto_3v3:a:$F/tm9_nexto_3v3.nexto.*.jsonl" > $F/kickoff_audit.jsonl 2>$F/kickoff_audit.err || true
$PY - $F <<'J2' > $F/finals.json
import json,os,sys
F=sys.argv[1]; out={}
def ld(n):
    try: return json.loads(open(os.path.join(F,n)).read().strip().splitlines()[-1])
    except Exception: return {}
for n in (2,3):
    for tag in ("tm7","tm8"):
        h=ld(f"h2h_{tag}_{n}v{n}.json"); a=h.get("team",{}).get("a",{})
        out[f"h2h_{tag}_{n}v{n}"]=h.get("win_rate_decisive")
        if tag=="tm7":
            for k,m in (("pair800","pair_lt800_ratio"),("near2","two_near_ball1500_ratio"),("ncr","no_challenger_ratio"),("dcw","double_commit_wide_ratio"),("deep","deep_last_man_ratio"),("passes_min","passes_per_min"),("touch","touches_per_car_min"),("depth","deepest_car_depth_uu"),("ko_dup","kickoff_duplicate_share"),("demos_min","demos_per_min"),("bumps_min","bumps_per_min")):
                out[f"{k}_{n}v{n}"]=a.get(m)
    for k in ("nexto","necto"): out[f"{k}_{n}v{n}"]=ld(f"{k}_{n}v{n}.json").get("win_rate_decisive")
try:
    for l in open(os.path.join(F,"kickoff_audit.jsonl")):
        d=json.loads(l); k=d.get("kickoff") or {}; n=d["label"].split("_")[-1]
        out[f"same_pad_{n}"]=k.get("same_pad_two_nontakers")
        out[f"same_big_{n}"]=k.get("same_big_pad_two_cars")
        out[f"ko_15s_{n}"]=k.get("outcome_1.5s(ball y +-800)")
        out[f"ko_3s_{n}"]=k.get("outcome_3s(ball y +-1500)")
        out[f"ko_ft_{n}"]=k.get("first_touch_ours_only")
        out[f"nontaker_beh_{n}"]=k.get("nontaker_behaviour")
except Exception:
    pass
print(json.dumps(out))
J2
touch $F/DONE; log "TM9 finals: $(cat $F/finals.json)"
# 3d. TM10 playmaking metrics at its first mid.
while [[ ! -s $X10/logs/mid.jsonl ]]; do
  if [[ ! -d $X10 ]] || { ! pgrep -f "^bash $C/team_launch_tm10.sh" > /dev/null && [[ ! -s $X10/logs/mid.jsonl ]]; }; then
    log "playmaking check SKIPPED: TM10 launcher not running and no mid yet"; exit 0
  fi
  sleep 120
done
$PY - $X10/logs/metrics.jsonl <<'J3' > $D/playmaking_first_mid.json
import json,sys
keys=("laymaking","Events/Demos per min","Events/Bumps per min","Reward/demo")
last={}
try:
    for l in open(sys.argv[1]):
        try: d=json.loads(l)
        except Exception: continue
        for k,v in d.items():
            if any(s in k for s in keys): last[k]=v
except Exception as e:
    last={"error":str(e)}
print(json.dumps(last))
J3
log "TM10 first mid $(head -1 $X10/logs/mid.jsonl | head -c 400); playmaking metrics: $(cat $D/playmaking_first_mid.json)"
touch $D/DONE
