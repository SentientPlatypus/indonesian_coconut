#!/usr/bin/env bash
# TM5r -> TM6 cut-over (EXPERIMENTS.md "TM6 — Cross-exam note"). Run ONLY
# after the user's yes:
#   bash /opt/dlami/nvme/ic_ppo/indococo_ppo_bot/experiments/tm6_cutover.sh 2>&1 | tee -a /opt/dlami/nvme/tm6_work/cut/cutover.log
# 1. Stages the TM6 configs and team_launch_tm6.sh in the live code copy
#    (new files next to team_launch.sh; team_launch.sh is NOT replaced).
# 2. Opens tmux ippo_team:tm6 (tm6_window.sh, waits for GO) so the session
#    survives, then waits until TM5r's running trainer chunk exits (its
#    checkpoint is then on disk; the chain is in mid evals).
# 3. Stops the TM5r chain by PID only (bash team_launch.sh and its
#    descendants; never pkill -f), takes TM5r's latest checkpoint, writes
#    ~/ippo_runs/TM6_START_CK and GO (TM6 starts from it), and the CUT marker.
# 4. Exports TM5r's latest actor as checkpoints_to_test/tm5r_<N>M.bpk,
#    commits and pushes it (not promoted), and starts abbreviated TM5r evals
#    at nice 19 on 4 pinned cores (~/ippo_runs/tm5r_cut).
# team_best is NOT touched. TM5r full finals are skipped.
set -u
R=/home/ubuntu/ippo_runs; C=$R/code_tm/indococo_ppo_bot/experiments
REPO=/opt/dlami/nvme/ic_ppo/indococo_ppo_bot; E=$REPO/experiments; D=/opt/dlami/nvme/tm6_work/cut
B6=/opt/dlami/nvme/ippo_bins/indococo-train-tm6-a922fb6
X5=$R/exp_TM5r
mkdir -p $D
[[ -x $B6 ]] || { echo "missing $B6"; exit 1; }
[[ -e $R/exp_TM6 ]] && { echo "$R/exp_TM6 already exists: not cutting over"; exit 1; }
[[ -s $D/GO ]] && { echo "$D/GO already exists: not cutting over"; exit 1; }
CHAIN=${CHAIN:-$(pgrep -f "^bash $C/team_launch.sh" | tr '\n' ' ')}
set -- $CHAIN; (( $# == 1 )) || { echo "expected one TM5r chain, found: '$CHAIN' (set CHAIN=<pid>)"; exit 1; }
CHAIN=$1
ps -o args= -p $CHAIN | grep -q team_launch.sh || { echo "chain $CHAIN is not team_launch.sh"; exit 1; }
tr '\0' '\n' < /proc/$CHAIN/environ 2>/dev/null | grep -q '^RUN=TM5r$' || { echo "chain $CHAIN is not RUN=TM5r"; exit 1; }
# 1. stage configs + launcher (new files only)
cp $E/tm6.toml $E/train_tm6.toml $E/tm_TM6.env $E/team_launch_tm6.sh $E/tm6_window.sh $E/tm5r_cut_evals.sh $C/
chmod +x $C/team_launch_tm6.sh; bash -n $C/team_launch_tm6.sh || { echo "launcher syntax"; exit 1; }
# 2. TM6 window (keeps ippo_team alive), then wait for the TM5r chunk
tmux new-window -d -t ippo_team -n tm6 "bash $C/tm6_window.sh"
TP=$(pgrep -f "^$X5/indococo-train train" || true)
if [[ -n "$TP" ]]; then echo "$(date) waiting for TM5r trainer $TP to finish its chunk"; while kill -0 $TP 2>/dev/null; do sleep 10; done; fi
echo "$(date) TM5r trainer chunk done"
# 3. stop the TM5r chain by PID only
desc() { local p; for p in $(ps -o pid= --ppid $1); do echo $p; desc $p; done; }
OLD="$CHAIN $(desc $CHAIN | tr '\n' ' ')"
echo "$(date) stopping TM5r chain PIDs: $OLD"; kill $OLD 2>/dev/null || true
sleep 3
TP2=$(pgrep -f "^$X5/indococo-train train" || true); [[ -n "$TP2" ]] && { echo "stopping just-started TM5r trainer $TP2"; kill $TP2; sleep 5; }
for p in $OLD; do kill -0 $p 2>/dev/null && echo "still alive: $p"; done
CK=$X5/checkpoints/$(ls $X5/checkpoints | grep -E '^[0-9]+$' | sort -n | tail -1)
N=$(sed -nE 's/^cumulative_timesteps = ([0-9]+)/\1/p' $CK/stats.toml)
echo "$(date) TM5r latest $CK ($N)"
[[ -s $CK/actor.bpk && -s $CK/critic.bpk ]] || { echo "bad checkpoint $CK"; exit 1; }
echo "$CK" > $R/TM6_START_CK
echo "$CK" > $D/GO; echo "$(date) TM6 starting in ippo_team:tm6 from $CK"
echo "TM5r cut at $N ($CK) for TM6 on $(date -u +%FT%TZ); full finals skipped" > $X5/CUT_FOR_TM6
# 4. bpk export + push, abbreviated TM5r evals
M=$(( (N + 500000) / 1000000 ))
cd /opt/dlami/nvme/ic_ppo
cp $CK/actor.bpk checkpoints_to_test/tm5r_${M}M.bpk
git pull --rebase -q
git add checkpoints_to_test/tm5r_${M}M.bpk
git -c user.name=SentientPlatypus -c user.email=83039642+SentientPlatypus@users.noreply.github.com \
  commit -q -m "checkpoints_to_test: add tm5r_${M}M (TM5r latest at the TM6 cut-over, not promoted; 270-wide pred obs)"
git push -q origin indococo-ppo; git log --oneline -1
mkdir -p $R/tm5r_cut; echo $CK > $R/tm5r_cut/CKPT
nohup bash $E/tm5r_cut_evals.sh $CK $R/tm5r_cut > $R/tm5r_cut/evals.log 2>&1 < /dev/null &
echo "abbreviated TM5r evals pid $!"
