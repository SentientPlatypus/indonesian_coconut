#!/usr/bin/env bash
# TM7 after TM6 (EXPERIMENTS.md "TM7 — Cross-exam note"). Run detached:
#   mkdir -p /opt/dlami/nvme/tm7_work/cut && nohup bash \
#     /home/ubuntu/ippo_runs/code_tm/indococo_ppo_bot/experiments/tm7_after_tm6.sh \
#     >> /opt/dlami/nvme/tm7_work/cut/cutover.log 2>&1 &
# 1. touch ~/ippo_runs/TEAM_NO_FOLLOWUP so TM6's launcher does not chain a
#    TM6r after its own finals (it still runs them: nice 19, recommendation only).
# 2. Waits for TM6's trainer to exit normally: the latest checkpoint at or past
#    END (= TM6 START + 5B = 37956.7M), no `indococo-train train` process of
#    exp_TM6, mid directory proving the launcher saw trainer exit 0, and no FATAL in TM6's train.log. The trainer PID changes every
#    100M chunk, so the wait follows the launcher (team_launch_tm6.sh) and the
#    checkpoint steps; if the launcher dies before END, logs BLOCKED and exits
#    without starting TM7.
# 3. Starts TM7 at once in tmux ippo_team window tm7 (tm7_window.sh, from TM6's
#    last checkpoint), then in parallel (nice 19, one job at a time):
#    - exports that checkpoint's actor as checkpoints_to_test/tm6_<N>M.bpk and
#      pushes it to indococo-ppo (not promoted);
#    - after TM6's launcher has exited (its last mid + finals, so at most two
#      eval jobs run next to TM7's trainer), TM6 finals as the TM5r cut evals:
#      h2h vs the team BEST (TM4 cut) 400 games per mode, Nexto 300 and Necto
#      300 per mode, spacing / role metrics from the h2h JSON -> $D/finals.
# Never touches tmux indococo / ippo_1v1, team_best, 1v1.pt / 2v2.pt / 3v3.pt.
set -u
R=/home/ubuntu/ippo_runs; X6=$R/exp_TM6; C=$R/code_tm/indococo_ppo_bot/experiments
REPO=/opt/dlami/nvme/ic_ppo; D=/opt/dlami/nvme/tm7_work/cut; F=$D/finals
B6=/opt/dlami/nvme/ippo_bins/indococo-train-tm6-a922fb6; CORES=${CORES:-28-31}
mkdir -p $D $F
# stdout is cutover.log (see the nohup line above).
log() { echo "$(date '+%F %T %Z') $*"; }
steps_in() { sed -nE 's/^cumulative_timesteps = ([0-9]+)/\1/p' $1/stats.toml; }
latest() { ls $X6/checkpoints | grep -E '^[0-9]+$' | sort -n | tail -1; }
trainer_up() { pgrep -f "^$X6/indococo-train train " > /dev/null; }
launcher_pid() { pgrep -f "bash $C/team_launch_tm6.sh" | head -1; }
for f in team_launch_tm7.sh tm7_window.sh tm_TM7.env tm7.toml train_tm7.toml; do [[ -s $C/$f ]] || { log "BLOCKED: $C/$f missing"; exit 1; }; done
[[ -x /opt/dlami/nvme/ippo_bins/indococo-train-tm7-3c49c88 ]] || { log "BLOCKED: TM7 binary missing"; exit 1; }
touch $R/TEAM_NO_FOLLOWUP
read _ START0 _ < $X6/START; END=$((START0 + 5000000000))
LP=$(launcher_pid); TP=$(pgrep -f "^$X6/indococo-train train " | head -1)
log "waiting for TM6 (launcher pid ${LP:-none}, current trainer pid ${TP:-none}) to reach END $END"
while :; do
  now=$(steps_in $X6/checkpoints/$(latest))
  # The mid directory is created ONLY after the launcher checked the
  # trainer pipeline exit status == 0. This is the normal-exit receipt.
  if (( now >= END )) && ! trainer_up && [[ -d $X6/mid/$now ]]; then break; fi
  if [[ -z "$(launcher_pid)" ]] && ! trainer_up; then
    log "BLOCKED: TM6 launcher gone at $now < END $END (train.log: $(grep FATAL $X6/logs/train.log | tail -1)); TM7 not started"; exit 1
  fi
  sleep 30
done
ck=$X6/checkpoints/$(latest); now=$(steps_in $ck)
if grep -q "FATAL" $X6/logs/train.log; then log "BLOCKED: FATAL in TM6 train.log: $(grep FATAL $X6/logs/train.log | tail -1); TM7 not started"; exit 1; fi
log "TM6 trainer exited normally at $now ($ck)"
# 3a. TM7 (the box never idles).
if [[ -d $R/exp_TM7 ]]; then log "BLOCKED: $R/exp_TM7 already exists; TM7 not started"; exit 1; fi
tmux has-session -t ippo_team 2>/dev/null || tmux new-session -d -s ippo_team -n tm7_hold "sleep 60"
tmux new-window -d -t ippo_team -n tm7 "bash $C/tm7_window.sh $ck"
sleep 120; log "TM7 started: $(pgrep -af "^$R/exp_TM7/indococo-train train" | head -1 || echo 'no trainer yet'); $(tail -2 $R/exp_TM7/logs/train.log 2>/dev/null | tr '\n' ' ')"
# 3b. bpk push (not promoted).
N=$(( (now + 500000) / 1000000 )); bpk=checkpoints_to_test/tm6_${N}M.bpk
(
  cd $REPO && git pull -q --rebase origin indococo-ppo && cp $ck/actor.bpk $bpk.tmp && mv $bpk.tmp $bpk && git add $bpk \
  && git -c user.name=SentientPlatypus -c user.email=83039642+SentientPlatypus@users.noreply.github.com commit -q \
       -m "checkpoints_to_test: add tm6_${N}M (TM6 final checkpoint at 5B, not promoted; 270-wide pred obs)" \
  && git push -q origin HEAD:indococo-ppo && log "pushed $bpk @ $(git rev-parse --short HEAD) (sha256 $(sha256sum $bpk | cut -c1-16))"
) || log "bpk push FAILED for $bpk (left for a manual push)"
# 3c. TM6 finals after TM6's launcher finished its own evals.
while [[ -n "$(launcher_pid)" ]]; do sleep 60; done
log "TM6 launcher exited ($(cat $R/tm6_final/TEAM_PROMOTE.json 2>/dev/null | head -c 300)); TM6 finals on $ck"
BT=$(cat $R/team_best/FINAL_CKPT); echo "$ck vs $BT" > $F/PAIRS
export LD_LIBRARY_PATH=$R/e8b2_1v1/bin/sp/torch/lib INDOCOCO_OBS_EXT=1 INDOCOCO_OBS_TEAM=1 INDOCOCO_OBS_PRED=1
cd /home/ubuntu/indonesian_coconut
for n in 2 3; do
  INDOCOCO_TEAM_SIZE=$n nice -n 19 taskset -c $CORES $B6 h2h $ck $BT 400 4 2>/dev/null | tail -1 > $F/h2h_${n}v${n}.json
  for k in nexto necto; do
    INDOCOCO_TEAM_SIZE=$n nice -n 19 taskset -c $CORES $B6 $k-h2h $ck /home/ubuntu/eval_bots/$k 300 4 2>/dev/null | tail -1 > $F/${k}_${n}v${n}.json
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
    for k,m in (("pair800","pair_lt800_ratio"),("near2","two_near_ball1500_ratio"),("ncr","no_challenger_ratio"),("dcw","double_commit_wide_ratio"),("deep","deep_last_man_ratio"),("passes_min","passes_per_min"),("swaps_min","role_swaps_per_min"),("first_touch","touch_share_first")):
        out[f"{k}_{n}v{n}"]=a.get(m)
print(json.dumps(out))
PY
touch $F/DONE; log "TM6 finals: $(cat $F/finals.json)"
