#!/usr/bin/env bash
# TM8 after TM7 (EXPERIMENTS.md "TM8 — Cross-exam note"; user-approved Nov 19
# 2026, 5:05 PM ET: "Auto-start TM8 (tm8v2 + your three tweaks) when TM7
# finishes"). Run detached, independent of ssh:
#   mkdir -p /opt/dlami/nvme/tm8_work/cut && setsid nohup bash \
#     /home/ubuntu/ippo_runs/code_tm/indococo_ppo_bot/experiments/tm8_after_tm7.sh \
#     >> /opt/dlami/nvme/tm8_work/cut/cutover.log 2>&1 < /dev/null &
# 1. touch ~/ippo_runs/TEAM_NO_FOLLOWUP so TM7's launcher does not chain a
#    TM7r after its own finals (it still runs them: nice 19, recommendation
#    only), and TEAM_QUEUE must be empty (BLOCKED otherwise).
# 2. Waits for TM7's trainer to exit normally: the latest checkpoint at or past
#    END (= TM7 START + 5B), no `indococo-train train` process of exp_TM7, the
#    mid directory proving the launcher saw trainer exit 0, and no FATAL in
#    TM7's train.log. Follows the launcher (team_launch_tm7.sh); if it dies
#    before END, logs BLOCKED and exits without starting TM8.
# 3. Starts TM8 at once in tmux ippo_team window tm8 (tm8_window.sh, from TM7's
#    last checkpoint), then, one job at a time at nice 19 on CORES:
#    a. passivity baseline for TM8's alarm: h2h TM7 final vs BEST-team (TM6
#       final) 200 games per mode with the TM8 binary -> $D/passive_base.json
#       (touch_<n>v<n> = touches per car-minute, depth_<n>v<n> = deepest-car
#       distance to the own back wall; team_launch_tm8.sh reads it);
#    b. exports TM7's last checkpoint actor as checkpoints_to_test/tm7_<N>M.bpk
#       and pushes it to indococo-ppo (not promoted);
#    c. after TM7's launcher has exited (its own finals), TM7 finals: h2h vs
#       TM6 final 400 games per mode, Nexto 300 and Necto 300 per mode, spacing
#       / role / passivity metrics -> $D/finals/finals.json;
#    d. after TM8's first mid line, the kickoff eval of that checkpoint vs
#       Nexto in 2v2 / 3v3 (504 games each, TRAJ_DUMP traces analysed by
#       /opt/dlami/nvme/tm8_audit/tm8_audit_traces.py: same-pad rates, kickoff
#       won / lost at 1.5 s and 3 s, first-touch, goals within 8 s) ->
#       $D/kickoff/kickoff_eval.json (compare with the TM6 / TM7 rows in
#       EXPERIMENTS.md "TM6/TM7 team audit").
# Never touches tmux indococo / ippo_1v1, team_best, the RLBot pack,
# rlbot_submission/, 1v1.pt / 2v2.pt / 3v3.pt, or the running TM7.
set -u
R=/home/ubuntu/ippo_runs; X7=$R/exp_TM7; X8=$R/exp_TM8; C=$R/code_tm/indococo_ppo_bot/experiments
REPO=/opt/dlami/nvme/ic_ppo; D=/opt/dlami/nvme/tm8_work/cut; F=$D/finals; K=$D/kickoff
B8=/opt/dlami/nvme/ippo_bins/indococo-train-tm8-5005500; B8SHA=544de3023bb184917ffc1d2edccbdb29f62369a3cbcdad73781d5c6f25a43ff3; CORES=${CORES:-28-31}
TM6=$R/exp_TM6/checkpoints/1795044069
mkdir -p $D $F $K
log() { echo "$(date '+%F %T %Z') $*"; }
steps_in() { sed -nE 's/^cumulative_timesteps = ([0-9]+)/\1/p' $1/stats.toml; }
latest() { ls $X7/checkpoints | grep -E '^[0-9]+$' | sort -n | tail -1; }
trainer_up() { pgrep -f "^$X7/indococo-train train " > /dev/null; }
launcher_pid() { pgrep -f "bash $C/team_launch_tm7.sh" | head -1; }
for f in team_launch_tm8.sh tm8_window.sh tm_TM8.env tm8.toml train_tm8.toml tm7.toml train_tm7.toml; do [[ -s $C/$f ]] || { log "BLOCKED: $C/$f missing"; exit 1; }; done
[[ -x $B8 ]] || { log "BLOCKED: TM8 binary $B8 missing"; exit 1; }
[[ "$(sha256sum $B8 | cut -d' ' -f1)" == "$B8SHA" ]] || { log "BLOCKED: TM8 binary sha256 mismatch"; exit 1; }
[[ "$(cat $R/team_best/FINAL_CKPT)" == "$TM6" ]] || { log "BLOCKED: team_best is not the TM6 final ($(cat $R/team_best/FINAL_CKPT))"; exit 1; }
touch $R/TEAM_NO_FOLLOWUP
[[ -s $R/TEAM_QUEUE ]] && { log "BLOCKED: TEAM_QUEUE not empty ($(head -1 $R/TEAM_QUEUE)); TM8 not armed"; exit 1; }
read _ START0 _ < $X7/START; END=$((START0 + 5000000000))
LP=$(launcher_pid); TP=$(pgrep -f "^$X7/indococo-train train " | head -1)
log "armed (pid $$): waiting for TM7 (launcher pid ${LP:-none}, current trainer pid ${TP:-none}) to reach END $END; TM8 binary $B8 ($B8SHA)"
while :; do
  now=$(steps_in $X7/checkpoints/$(latest))
  # The mid directory is created ONLY after the launcher checked the
  # trainer pipeline exit status == 0. This is the normal-exit receipt.
  if (( now >= END )) && ! trainer_up && [[ -d $X7/mid/$now ]]; then break; fi
  if [[ -z "$(launcher_pid)" ]] && ! trainer_up; then
    log "BLOCKED: TM7 launcher gone at $now < END $END (train.log: $(grep FATAL $X7/logs/train.log | tail -1)); TM8 not started"; exit 1
  fi
  if grep -q "FATAL" $X7/logs/train.log; then log "BLOCKED: FATAL in TM7 train.log: $(grep FATAL $X7/logs/train.log | tail -1); TM8 not started"; exit 1; fi
  sleep 30
done
ck=$X7/checkpoints/$(latest); now=$(steps_in $ck)
if grep -q "FATAL" $X7/logs/train.log; then log "BLOCKED: FATAL in TM7 train.log: $(grep FATAL $X7/logs/train.log | tail -1); TM8 not started"; exit 1; fi
log "TM7 trainer exited normally at $now ($ck)"
# 3. TM8 (the box never idles).
if [[ -d $X8 ]]; then log "BLOCKED: $X8 already exists; TM8 not started"; exit 1; fi
tmux has-session -t ippo_team 2>/dev/null || tmux new-session -d -s ippo_team -n tm8_hold "sleep 60"
tmux new-window -d -t ippo_team -n tm8 "bash $C/tm8_window.sh $ck"
sleep 120; log "TM8 started: $(pgrep -af "^$X8/indococo-train train" | head -1 || echo 'no trainer yet'); $(tail -2 $X8/logs/train.log 2>/dev/null | tr '\n' ' ')"
export LD_LIBRARY_PATH=$R/e8b2_1v1/bin/sp/torch/lib INDOCOCO_OBS_EXT=1 INDOCOCO_OBS_TEAM=1 INDOCOCO_OBS_PRED=1
cd /home/ubuntu/indonesian_coconut
# 3a. Passivity baseline (TM7 final vs TM6 final, TM8 binary, same protocol as TM8's mids).
for n in 2 3; do
  INDOCOCO_TEAM_SIZE=$n nice -n 19 taskset -c $CORES $B8 h2h $ck $TM6 200 4 2>/dev/null | tail -1 > $D/base_h2h_${n}v${n}.json
done
/home/ubuntu/coco-venv/bin/python - $D $ck <<'PY' > $D/passive_base.json.tmp && mv $D/passive_base.json.tmp $D/passive_base.json
import json,os,sys
D,ck=sys.argv[1],sys.argv[2]; out={"ckpt":ck}
for n in (2,3):
    a=json.loads(open(os.path.join(D,f"base_h2h_{n}v{n}.json")).read().strip().splitlines()[-1])["team"]["a"]
    out[f"touch_{n}v{n}"]=a["touches_per_car_min"]; out[f"depth_{n}v{n}"]=a["deepest_car_depth_uu"]
print(json.dumps(out))
PY
log "passivity baseline: $(cat $D/passive_base.json 2>/dev/null || echo MISSING)"
# 3b. bpk push (not promoted).
N=$(( (now + 500000) / 1000000 )); bpk=checkpoints_to_test/tm7_${N}M.bpk
(
  cd $REPO && git pull -q --rebase origin indococo-ppo && cp $ck/actor.bpk $bpk.tmp && mv $bpk.tmp $bpk && git add $bpk \
  && git -c user.name=SentientPlatypus -c user.email=83039642+SentientPlatypus@users.noreply.github.com commit -q \
       -m "checkpoints_to_test: add tm7_${N}M (TM7 final checkpoint at 5B, not promoted; 270-wide pred obs)" \
  && git push -q origin HEAD:indococo-ppo && log "pushed $bpk @ $(git rev-parse --short HEAD) (sha256 $(sha256sum $bpk | cut -c1-16))"
) || log "bpk push FAILED for $bpk (left for a manual push)"
# 3c. TM7 finals after TM7's launcher finished its own evals.
while [[ -n "$(launcher_pid)" ]]; do sleep 60; done
log "TM7 launcher exited ($(cat $R/tm7_final/TEAM_PROMOTE.json 2>/dev/null | head -c 300)); TM7 finals on $ck"
echo "$ck vs $TM6" > $F/PAIRS
for n in 2 3; do
  INDOCOCO_TEAM_SIZE=$n nice -n 19 taskset -c $CORES $B8 h2h $ck $TM6 400 4 2>/dev/null | tail -1 > $F/h2h_${n}v${n}.json
  for k in nexto necto; do
    INDOCOCO_TEAM_SIZE=$n nice -n 19 taskset -c $CORES $B8 $k-h2h $ck /home/ubuntu/eval_bots/$k 300 4 2>/dev/null | tail -1 > $F/${k}_${n}v${n}.json
  done
done
/home/ubuntu/coco-venv/bin/python - $F <<'PY' > $F/finals.json
import json,os,sys
F=sys.argv[1]; out={}
def ld(n):
    try: return json.loads(open(os.path.join(F,n)).read().strip().splitlines()[-1])
    except Exception: return {}
for n in (2,3):
    h=ld(f"h2h_{n}v{n}.json"); a=h.get("team",{}).get("a",{})
    out[f"h2h_{n}v{n}"]=h.get("win_rate_decisive")
    for k in ("nexto","necto"): out[f"{k}_{n}v{n}"]=ld(f"{k}_{n}v{n}.json").get("win_rate_decisive")
    for k,m in (("pair800","pair_lt800_ratio"),("near2","two_near_ball1500_ratio"),("ncr","no_challenger_ratio"),("dcw","double_commit_wide_ratio"),("deep","deep_last_man_ratio"),("passes_min","passes_per_min"),("swaps_min","role_swaps_per_min"),("first_touch","touch_share_first"),("touch","touches_per_car_min"),("depth","deepest_car_depth_uu"),("ko_dup","kickoff_duplicate_share")):
        out[f"{k}_{n}v{n}"]=a.get(m)
print(json.dumps(out))
PY
touch $F/DONE; log "TM7 finals: $(cat $F/finals.json)"
# 3d. Kickoff eval of TM8's first mid checkpoint vs Nexto (traces + tm8_audit_traces.py).
while [[ ! -s $X8/logs/mid.jsonl ]]; do
  if [[ ! -d $X8 ]] || { ! pgrep -f "bash $C/team_launch_tm8.sh" > /dev/null && [[ ! -s $X8/logs/mid.jsonl ]]; }; then
    log "kickoff eval SKIPPED: TM8 launcher not running and no mid yet"; exit 0
  fi
  sleep 120
done
m1=$(head -1 $X8/logs/mid.jsonl | /home/ubuntu/coco-venv/bin/python -c "import json,sys;print(json.loads(sys.stdin.read())['ckpt'])")
log "kickoff eval on TM8 first mid $m1"
for n in 2 3; do
  INDOCOCO_TEAM_SIZE=$n TRAJ_DUMP=$K/tm8_nexto_${n}v${n} OMP_NUM_THREADS=1 nice -n 19 taskset -c $CORES $B8 nexto-h2h $m1 /home/ubuntu/eval_bots/nexto 504 4 > $K/tm8_nexto_${n}v${n}.result.json 2>$K/tm8_nexto_${n}v${n}.stderr
done
/home/ubuntu/coco-venv/bin/python /opt/dlami/nvme/tm8_audit/tm8_audit_traces.py \
  "tm8_nexto_2v2:a:$K/tm8_nexto_2v2.nexto.*.jsonl" "tm8_nexto_3v3:a:$K/tm8_nexto_3v3.nexto.*.jsonl" > $K/kickoff_eval.jsonl 2>$K/kickoff_eval.err
/home/ubuntu/coco-venv/bin/python - $K <<'PY' > $K/kickoff_eval.json
import json,sys,os
K=sys.argv[1]; out={}
for l in open(os.path.join(K,"kickoff_eval.jsonl")):
    d=json.loads(l); k=d["kickoff"]; n=d["label"].split("_")[-1]
    out[n]={"games":d["games"],"win_share":d["win_share"],"same_pad_two_nontakers":k.get("same_pad_two_nontakers"),
            "same_big_pad_two_cars":k.get("same_big_pad_two_cars"),"first_touch_ours_only":k.get("first_touch_ours_only"),
            "outcome_1.5s":k.get("outcome_1.5s(ball y +-800)"),"outcome_3s":k.get("outcome_3s(ball y +-1500)"),
            "goal_8s_for":k.get("goal_8s_for"),"goal_8s_against":k.get("goal_8s_against"),
            "nontaker_dball_at_touch":k.get("nontaker_dball_at_touch")}
print(json.dumps(out))
PY
touch $K/DONE; log "TM8 kickoff eval vs Nexto: $(cat $K/kickoff_eval.json)"
