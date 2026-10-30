#!/usr/bin/env bash
# Runs the experiment arms one after another on one GPU (see EXPERIMENTS.md).
#
#   experiments/run_experiments.sh
#
# Each arm gets a run folder $RUNS/exp_<arm> and a tmux session ippo_<arm>
# with a "train" window (its reward stages in order) and an "eval" window
# (tools/eval_loop.sh: pool eval + keep-best every EVERY steps). After
# training the final checkpoint is evaluated too and the eval window closes;
# the session stays for inspection. Resumable: rerun to continue.
# $RUNS/EXPERIMENTS_STATUS.md is refreshed every few minutes.
#
# env: RUNS, BIN (indococo-train), WARM (warm-start checkpoint dir),
#      OPP_1V1 / OPP_430M (opponent checkpoint dirs), MESHES (dir holding
#      collision_meshes/), LIBTORCH_LIB, PY, EVERY, GAMES, PROCS
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CRATE=$(dirname "$HERE")
: "${RUNS:?}" "${BIN:?}" "${WARM:?}" "${OPP_1V1:?}" "${OPP_430M:?}" "${MESHES:?}" "${LIBTORCH_LIB:?}" "${PY:?}"
EVERY=${EVERY:-100000000} GAMES=${GAMES:-100} PROCS=${PROCS:-8}
TS=100000   # timesteps per iteration (train_template.toml)
export LD_LIBRARY_PATH=$LIBTORCH_LIB   # indococo-train export needs LibTorch too

# arm -> "rewards:iterations ..." stages
declare -A STAGES=(
  [X0]="x0.toml:4500"
  [FR]="fr_ramp.toml:1500 fr.toml:3000"
  [ADB]="adb.toml:4500"
)
ARMS=${ARMS:-"X0 FR ADB"}

steps_of() {  # cumulative_timesteps of the newest checkpoint in $1
  local d; d=$(ls "$1" | grep -E '^[0-9]+$' | sort -n | tail -1)
  sed -nE 's/^cumulative_timesteps = ([0-9]+)/\1/p' "$1/$d/stats.toml"
}

status() { "$PY" "$HERE/status.py" "$RUNS" $ARMS > /dev/null || true; }

setup() {  # run folder for arm $1
  local run=$RUNS/exp_$1
  [[ -d $run/checkpoints ]] && return
  mkdir -p "$run"/{cfg,checkpoints,logs,opponents,bots}
  cp -r "$CRATE/config" "$CRATE/experiments" "$run/cfg/"
  cp -r "$WARM" "$run/checkpoints/"
  ln -sfn "$OPP_1V1" "$run/opponents/1v1"
  ln -sfn "$OPP_430M" "$run/opponents/ppo_430M"
  cp "$BIN" "$run/indococo-train"
  steps_of "$run/checkpoints" > "$run/warm_steps"
}

write_arm_script() {  # train script for arm $1: runs each unfinished stage
  local arm=$1 run=$RUNS/exp_$1 end
  end=$(cat "$run/warm_steps")
  {
    echo '#!/usr/bin/env bash'
    echo "set -euo pipefail; cd '$MESHES'"
    echo "export LD_LIBRARY_PATH='$LIBTORCH_LIB' TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 INDOCOCO_METRICS='$run/logs/metrics.jsonl'"
    for stage in ${STAGES[$arm]}; do
      local rewards=${stage%%:*} iters=${stage##*:}
      end=$((end + iters * TS))
      echo "now=\$(ls '$run/checkpoints' | grep -E '^[0-9]+\$' | sort -n | tail -1 | xargs -I{} sed -nE 's/^cumulative_timesteps = ([0-9]+)/\\1/p' '$run/checkpoints/{}/stats.toml')"
      echo "left=\$(( ($end - now + $TS - 1) / $TS ))"
      echo "if (( left > 0 )); then"
      echo "  sed -e 's|@RUN@|$run|g' -e 's|@REWARDS@|$rewards|' -e \"s|@ITERATIONS@|\$left|\" '$run/cfg/experiments/train_template.toml' > '$run/train_$rewards'"
      echo "  echo \"STAGE $rewards: \$left iterations\" | tee -a '$run/logs/train.log'"
      echo "  '$run/indococo-train' train '$run/train_$rewards' 2>&1 | tee -a '$run/logs/train.log'"
      echo "fi"
    done
    echo "touch '$run/TRAIN_DONE'"
  } > "$run/train.sh.new"
  chmod +x "$run/train.sh.new"
  mv "$run/train.sh.new" "$run/train.sh"   # atomic: a running train.sh keeps its copy
}

final_eval() {  # pool eval + keep-best of the newest checkpoint, if not yet logged
  local run=$RUNS/exp_$1 newest steps
  newest=$(ls "$run/checkpoints" | grep -E '^[0-9]+$' | sort -n | tail -1)
  steps=$(steps_of "$run/checkpoints")
  if grep -q "\"steps\": $steps[,}]" "$run/logs/evals.jsonl" 2>/dev/null; then return; fi
  local st=$run/logs/final_policy.safetensors log=$run/logs/final_keep_best.jsonl
  "$run/indococo-train" export "$run/checkpoints/$newest" "$st" > /dev/null
  rm -f "$log"
  nice -n 19 "$PY" "$CRATE/tools/pool_eval/keep_best.py" --checkpoint "$run/checkpoints/$newest" \
      --policy "$st" --best "$run/bots/best" --games "$GAMES" --procs 32 --log "$log" \
      > "$run/logs/final_keep_best.log" 2>&1
  "$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); r["steps"]=int(sys.argv[2]); print(json.dumps(r))' \
      "$log" "$steps" >> "$run/logs/evals.jsonl"
}

for arm in $ARMS; do
  run=$RUNS/exp_$arm
  [[ -f $run/FINISHED ]] && continue
  setup "$arm"
  write_arm_script "$arm"
  session=ippo_$arm
  if ! tmux has-session -t "$session" 2>/dev/null; then
    tmux new-session -d -s "$session" -n train "$run/train.sh"
  fi
  if ! tmux list-windows -t "$session" -F '#W' | grep -qx eval; then
    tmux new-window -d -t "$session" -n eval "env LD_LIBRARY_PATH='$LIBTORCH_LIB' TRAIN_BIN='$run/indococo-train' PY='$PY' EVERY=$EVERY GAMES=$GAMES PROCS=$PROCS '$CRATE/tools/eval_loop.sh' '$run/checkpoints' '$run/logs/evals.jsonl' '$run/bots/best' 2>&1 | tee -a '$run/logs/eval_loop.log'; sleep infinity"
  fi
  until [[ -f $run/TRAIN_DONE ]]; do
    status
    if ! tmux list-windows -t "$session" -F '#W' 2>/dev/null | grep -qx train; then
      echo "$arm: train window exited without TRAIN_DONE; see $run/logs/train.log" >&2
      status; exit 1
    fi
    sleep 300
  done
  tmux kill-window -t "$session:eval" 2>/dev/null || true
  final_eval "$arm"
  touch "$run/FINISHED"
  status
done
status
