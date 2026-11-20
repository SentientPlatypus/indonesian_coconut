#!/usr/bin/env bash
# TM9 after TM8 (EXPERIMENTS.md "TM9 — Cross-exam note"; user-approved Nov 20
# 2026, 3:58 PM ET: "Yes, build TM9 as proposed and arm it to start when TM8
# finishes"). Run detached, independent of ssh:
#   mkdir -p /opt/dlami/nvme/tm9_work/cut && setsid nohup bash \
#     /home/ubuntu/ippo_runs/code_tm/indococo_ppo_bot/experiments/tm9_after_tm8.sh \
#     >> /opt/dlami/nvme/tm9_work/cut/cutover.log 2>&1 < /dev/null &
# 1. touch ~/ippo_runs/TEAM_NO_FOLLOWUP so TM8's launcher does not chain a
#    follow-up after its own finals (it still runs them: nice 19, recommendation
#    only), and TEAM_QUEUE must be empty (BLOCKED otherwise).
# 2. Waits for TM8's trainer to exit normally: the latest checkpoint at or past
#    END (= TM8 START + 5B), no `indococo-train train` process of exp_TM8, the
#    mid directory proving the launcher saw trainer exit 0, and no FATAL in
#    TM8's train.log. Follows the launcher (team_launch_tm8.sh); if it dies
#    before END, logs BLOCKED and exits without starting TM9.
# 3. Starts TM9 at once in tmux ippo_team window tm9 (tm9_window.sh, from TM8's
#    last checkpoint at END, not a mid-best), then, one job at a time at nice 19
#    on CORES:
#    a. exports TM8's last checkpoint actor as checkpoints_to_test/tm8_<N>M.bpk
#       and pushes it to indococo-ppo (not promoted);
#    b. passivity baseline for TM9's alarm: h2h TM8 final vs BEST-team (TM7
#       final) 200 games per mode with the TM9 binary -> $D/passive_base.json
#       (touch_<n>v<n> / depth_<n>v<n>; team_launch_tm9.sh reads it);
#    c. after TM8's launcher has exited (its own finals), TM8 finals: h2h vs
#       TM7 final and vs TM6 final, 400 games per mode, Nexto 300 and Necto 300
#       per mode, spacing / role / passivity metrics, plus a Nexto kickoff
#       same-pad audit (504 games, TRAJ_DUMP) -> $D/finals/;
#    d. after TM9's first mid line, the kickoff eval of that checkpoint vs
#       Nexto in 2v2 / 3v3 (504 games each, tm8_audit_traces.py) ->
#       $D/kickoff/kickoff_eval.json.
# If TM9 dies before its first chunk, tm9_window.sh continues as TM8r (TM8
# binary + tm8.toml / train_tm8.toml, BT=TM7) so the box never idles.
# Never touches tmux indococo / ippo_1v1, team_best, the RLBot pack,
# rlbot_submission/, 1v1.pt / 2v2.pt / 3v3.pt, or the running TM8.
set -u
R=/home/ubuntu/ippo_runs; X8=$R/exp_TM8; X9=$R/exp_TM9; C=$R/code_tm/indococo_ppo_bot/experiments
REPO=/opt/dlami/nvme/ic_ppo; D=/opt/dlami/nvme/tm9_work/cut; F=$D/finals; K=$D/kickoff
B9=${B9:-/opt/dlami/nvme/ippo_bins/indococo-train-tm9-fb49992}
B9SHA=${B9SHA:-125aca14679d32ea470506fe97535223e54ffa8acbad623d53874e4585af4c57}
CORES=${CORES:-28-31}
TM7=$R/exp_TM7/checkpoints/1795170349
TM6=$R/exp_TM6/checkpoints/1795044069
AUDIT_PY=${AUDIT_PY:-/opt/dlami/nvme/ic_ppo/indococo_ppo_bot/tools/team/tm8_audit/tm8_audit_traces.py}
[[ -s $AUDIT_PY ]] || AUDIT_PY=/opt/dlami/nvme/tm8_audit/tm8_audit_traces.py
mkdir -p $D $F $K
log() { echo "$(date '+%F %T %Z') $*"; }
steps_in() { sed -nE 's/^cumulative_timesteps = ([0-9]+)/\1/p' $1/stats.toml; }
latest() { ls $X8/checkpoints | grep -E '^[0-9]+$' | sort -n | tail -1; }
trainer_up() { pgrep -f "^$X8/indococo-train train " > /dev/null; }
launcher_pid() { pgrep -f "bash $C/team_launch_tm8.sh" | head -1; }
for f in team_launch_tm9.sh tm9_window.sh tm_TM9.env tm9.toml train_tm9.toml tm8.toml train_tm8.toml; do [[ -s $C/$f ]] || { log "BLOCKED: $C/$f missing"; exit 1; }; done
[[ -x $B9 ]] || { log "BLOCKED: TM9 binary $B9 missing"; exit 1; }
[[ "$(sha256sum $B9 | cut -d' ' -f1)" == "$B9SHA" ]] || { log "BLOCKED: TM9 binary sha256 mismatch"; exit 1; }
# Do not start if someone moved team_best off TM7; never change it here.
[[ "$(cat $R/team_best/FINAL_CKPT)" == "$TM7" ]] || { log "BLOCKED: team_best is not the TM7 final ($(cat $R/team_best/FINAL_CKPT))"; exit 1; }
touch $R/TEAM_NO_FOLLOWUP
[[ -s $R/TEAM_QUEUE ]] && { log "BLOCKED: TEAM_QUEUE not empty ($(head -1 $R/TEAM_QUEUE)); TM9 not armed"; exit 1; }
read _ START0 _ < $X8/START; END=$((START0 + 5000000000))
LP=$(launcher_pid); TP=$(pgrep -f "^$X8/indococo-train train " | head -1)
log "armed (pid $$): waiting for TM8 (launcher pid ${LP:-none}, current trainer pid ${TP:-none}) to reach END $END; TM9 binary $B9 ($B9SHA)"
while :; do
  now=$(steps_in $X8/checkpoints/$(latest))
  # The mid directory is created ONLY after the launcher checked the
  # trainer pipeline exit status == 0. This is the normal-exit receipt.
  if (( now >= END )) && ! trainer_up && [[ -d $X8/mid/$now ]]; then break; fi
  if [[ -z "$(launcher_pid)" ]] && ! trainer_up; then
    log "BLOCKED: TM8 launcher gone at $now < END $END (train.log: $(grep FATAL $X8/logs/train.log | tail -1)); TM9 not started"; exit 1
  fi
  if grep -q "FATAL" $X8/logs/train.log; then log "BLOCKED: FATAL in TM8 train.log: $(grep FATAL $X8/logs/train.log | tail -1); TM9 not started"; exit 1; fi
  sleep 30
done
ck=$X8/checkpoints/$(latest); now=$(steps_in $ck)
if grep -q "FATAL" $X8/logs/train.log; then log "BLOCKED: FATAL in TM8 train.log: $(grep FATAL $X8/logs/train.log | tail -1); TM9 not started"; exit 1; fi
log "TM8 trainer exited normally at $now ($ck)"
# 3. TM9 (the box never idles). Warm start = this END ckpt, not a mid-best.
if [[ -d $X9 ]]; then log "BLOCKED: $X9 already exists; TM9 not started"; exit 1; fi
tmux has-session -t ippo_team 2>/dev/null || tmux new-session -d -s ippo_team -n tm9_hold "sleep 60"
tmux new-window -d -t ippo_team -n tm9 "bash $C/tm9_window.sh $ck"
sleep 120; log "TM9 started: $(pgrep -af "^$X9/indococo-train train" | head -1 || echo 'no trainer yet'); $(tail -2 $X9/logs/train.log 2>/dev/null | tr '\n' ' ')"
export LD_LIBRARY_PATH=$R/e8b2_1v1/bin/sp/torch/lib INDOCOCO_OBS_EXT=1 INDOCOCO_OBS_TEAM=1 INDOCOCO_OBS_PRED=1
cd /home/ubuntu/indonesian_coconut
# 3a. bpk push of the TM8 final (not promoted).
N=$(( (now + 500000) / 1000000 )); bpk=checkpoints_to_test/tm8_${N}M.bpk
(
  cd $REPO && git pull -q --rebase origin indococo-ppo && cp $ck/actor.bpk $bpk.tmp && mv $bpk.tmp $bpk && git add $bpk \
  && git -c user.name=SentientPlatypus -c user.email=83039642+SentientPlatypus@users.noreply.github.com commit -q \
       -m "checkpoints_to_test: add tm8_${N}M (TM8 final checkpoint at 5B, not promoted; 270-wide pred obs)" \
  && git push -q origin HEAD:indococo-ppo && log "pushed $bpk @ $(git rev-parse --short HEAD) (sha256 $(sha256sum $bpk | cut -c1-16))"
) || log "bpk push FAILED for $bpk (left for a manual push)"
# 3b. Passivity baseline (TM8 final vs TM7 final, TM9 binary).
for n in 2 3; do
  INDOCOCO_TEAM_SIZE=$n nice -n 19 taskset -c $CORES $B9 h2h $ck $TM7 200 4 2>/dev/null | tail -1 > $D/base_h2h_${n}v${n}.json
done
/home/ubuntu/coco-venv/bin/python - $D $ck <<'J1' > $D/passive_base.json.tmp && mv $D/passive_base.json.tmp $D/passive_base.json
import json,os,sys
D,ck=sys.argv[1],sys.argv[2]; out={"ckpt":ck}
for n in (2,3):
    a=json.loads(open(os.path.join(D,f"base_h2h_{n}v{n}.json")).read().strip().splitlines()[-1])["team"]["a"]
    out[f"touch_{n}v{n}"]=a["touches_per_car_min"]; out[f"depth_{n}v{n}"]=a["deepest_car_depth_uu"]
print(json.dumps(out))
J1
log "passivity baseline: $(cat $D/passive_base.json 2>/dev/null || echo MISSING)"
# 3c. TM8 finals after TM8's launcher finished its own evals.
while [[ -n "$(launcher_pid)" ]]; do sleep 60; done
log "TM8 launcher exited ($(cat $R/tm8_final/TEAM_PROMOTE.json 2>/dev/null | head -c 300)); TM8 finals on $ck"
echo "$ck vs TM7=$TM7 vs TM6=$TM6" > $F/PAIRS
for n in 2 3; do
  INDOCOCO_TEAM_SIZE=$n nice -n 19 taskset -c $CORES $B9 h2h $ck $TM7 400 4 2>/dev/null | tail -1 > $F/h2h_tm7_${n}v${n}.json
  INDOCOCO_TEAM_SIZE=$n nice -n 19 taskset -c $CORES $B9 h2h $ck $TM6 400 4 2>/dev/null | tail -1 > $F/h2h_tm6_${n}v${n}.json
  for k in nexto necto; do
    INDOCOCO_TEAM_SIZE=$n nice -n 19 taskset -c $CORES $B9 $k-h2h $ck /home/ubuntu/eval_bots/$k 300 4 2>/dev/null | tail -1 > $F/${k}_${n}v${n}.json
  done
  INDOCOCO_TEAM_SIZE=$n TRAJ_DUMP=$F/tm8_nexto_${n}v${n} OMP_NUM_THREADS=1 nice -n 19 taskset -c $CORES $B9 nexto-h2h $ck /home/ubuntu/eval_bots/nexto 504 4 > $F/tm8_nexto_${n}v${n}.result.json 2>$F/tm8_nexto_${n}v${n}.stderr
done
/home/ubuntu/coco-venv/bin/python $AUDIT_PY \
  "tm8_nexto_2v2:a:$F/tm8_nexto_2v2.nexto.*.jsonl" "tm8_nexto_3v3:a:$F/tm8_nexto_3v3.nexto.*.jsonl" > $F/kickoff_audit.jsonl 2>$F/kickoff_audit.err || true
/home/ubuntu/coco-venv/bin/python - $F <<'J2' > $F/finals.json
import json,os,sys
F=sys.argv[1]; out={}
def ld(n):
    try: return json.loads(open(os.path.join(F,n)).read().strip().splitlines()[-1])
    except Exception: return {}
for n in (2,3):
    for tag in ("tm7","tm6"):
        h=ld(f"h2h_{tag}_{n}v{n}.json"); a=h.get("team",{}).get("a",{})
        out[f"h2h_{tag}_{n}v{n}"]=h.get("win_rate_decisive")
        if tag=="tm7":
            for k,m in (("pair800","pair_lt800_ratio"),("near2","two_near_ball1500_ratio"),("ncr","no_challenger_ratio"),("dcw","double_commit_wide_ratio"),("deep","deep_last_man_ratio"),("passes_min","passes_per_min"),("touch","touches_per_car_min"),("depth","deepest_car_depth_uu"),("ko_dup","kickoff_duplicate_share")):
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
touch $F/DONE; log "TM8 finals: $(cat $F/finals.json)"
# 3d. Kickoff eval of TM9's first mid checkpoint vs Nexto.
while [[ ! -s $X9/logs/mid.jsonl ]]; do
  if [[ ! -d $X9 ]] || { ! pgrep -f "bash $C/team_launch_tm9.sh" > /dev/null && [[ ! -s $X9/logs/mid.jsonl ]]; }; then
    # TM8r fallback may be running instead; still stop this wait.
    log "kickoff eval SKIPPED: TM9 launcher not running and no mid yet"; exit 0
  fi
  sleep 120
done
m1=$(head -1 $X9/logs/mid.jsonl | /home/ubuntu/coco-venv/bin/python -c "import json,sys;print(json.loads(sys.stdin.read())['ckpt'])")
log "kickoff eval on TM9 first mid $m1"
for n in 2 3; do
  INDOCOCO_TEAM_SIZE=$n TRAJ_DUMP=$K/tm9_nexto_${n}v${n} OMP_NUM_THREADS=1 nice -n 19 taskset -c $CORES $B9 nexto-h2h $m1 /home/ubuntu/eval_bots/nexto 504 4 > $K/tm9_nexto_${n}v${n}.result.json 2>$K/tm9_nexto_${n}v${n}.stderr
done
/home/ubuntu/coco-venv/bin/python $AUDIT_PY \
  "tm9_nexto_2v2:a:$K/tm9_nexto_2v2.nexto.*.jsonl" "tm9_nexto_3v3:a:$K/tm9_nexto_3v3.nexto.*.jsonl" > $K/kickoff_eval.jsonl 2>$K/kickoff_eval.err
/home/ubuntu/coco-venv/bin/python - $K <<'J3' > $K/kickoff_eval.json
import json,sys,os
K=sys.argv[1]; out={}
for l in open(os.path.join(K,"kickoff_eval.jsonl")):
    d=json.loads(l); k=d["kickoff"]; n=d["label"].split("_")[-1]
    out[n]={"games":d["games"],"win_share":d["win_share"],"same_pad_two_nontakers":k.get("same_pad_two_nontakers"),
            "same_big_pad_two_cars":k.get("same_big_pad_two_cars"),"first_touch_ours_only":k.get("first_touch_ours_only"),
            "outcome_1.5s":k.get("outcome_1.5s(ball y +-800)"),"outcome_3s":k.get("outcome_3s(ball y +-1500)"),
            "goal_8s_for":k.get("goal_8s_for"),"goal_8s_against":k.get("goal_8s_against"),
            "nontaker_behaviour":k.get("nontaker_behaviour"),
            "nontaker_dball_at_touch":k.get("nontaker_dball_at_touch")}
print(json.dumps(out))
J3
touch $K/DONE; log "TM9 kickoff eval vs Nexto: $(cat $K/kickoff_eval.json)"
