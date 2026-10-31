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
#      collision_meshes/), LIBTORCH_LIB, PY, EVERY, GAMES, PROCS,
#      SEED_BEST (optional bots/best dir the arm has to beat),
#      RATES_STEPS (true flip reset rates per eval, `indococo-train rates`
#      over experiments/rates_mix.toml vs OPP_430M; 0 = off)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CRATE=$(dirname "$HERE")
: "${RUNS:?}" "${BIN:?}" "${WARM:?}" "${OPP_1V1:?}" "${OPP_430M:?}" "${MESHES:?}" "${LIBTORCH_LIB:?}" "${PY:?}"
EVERY=${EVERY:-100000000} GAMES=${GAMES:-100} PROCS=${PROCS:-8} RATES_STEPS=${RATES_STEPS:-0}
TS=100000   # timesteps per iteration (train_template.toml)
export LD_LIBRARY_PATH=$LIBTORCH_LIB   # indococo-train export needs LibTorch too

# arm -> stages "rewards:iterations[:template:kickoff_w:air_reset_w] ..."
# (template defaults to train_template.toml)
declare -A STAGES=(
  [X0]="x0.toml:4500"
  [FR]="fr_ramp.toml:1500 fr.toml:3000"
  [ADB]="adb.toml:4500"
  [FR2]="fr2.toml:1000:train_fr2_template.toml:0.35:0.15 fr2.toml:1000:train_fr2_template.toml:0.27:0.23 fr2.toml:1000:train_fr2_template.toml:0.20:0.30"
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
  if [[ -n ${SEED_BEST:-} ]]; then cp -r "$SEED_BEST" "$run/bots/best"; fi
}

write_arm_script() {  # train script for arm $1: runs each unfinished stage
  local arm=$1 run=$RUNS/exp_$1 end
  end=$(cat "$run/warm_steps")
  {
    echo '#!/usr/bin/env bash'
    echo "set -euo pipefail; cd '$MESHES'"
    echo "export LD_LIBRARY_PATH='$LIBTORCH_LIB' TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 INDOCOCO_METRICS='$run/logs/metrics.jsonl'"
    local n=0
    for stage in ${STAGES[$arm]}; do
      local rewards iters template kick air
      IFS=: read -r rewards iters template kick air <<< "$stage"
      template=${template:-train_template.toml}
      n=$((n + 1))
      end=$((end + iters * TS))
      echo "now=\$(ls '$run/checkpoints' | grep -E '^[0-9]+\$' | sort -n | tail -1 | xargs -I{} sed -nE 's/^cumulative_timesteps = ([0-9]+)/\\1/p' '$run/checkpoints/{}/stats.toml')"
      echo "left=\$(( ($end - now + $TS - 1) / $TS ))"
      echo "if (( left > 0 )); then"
      echo "  sed -e 's|@RUN@|$run|g' -e 's|@REWARDS@|$rewards|' -e \"s|@ITERATIONS@|\$left|\" -e 's|@KICKOFF_W@|$kick|' -e 's|@AIR_RESET_W@|$air|' '$run/cfg/experiments/$template' > '$run/train_${n}_$rewards'"
      echo "  echo \"STAGE $n $rewards: \$left iterations\" | tee -a '$run/logs/train.log'"
      echo "  '$run/indococo-train' train '$run/train_${n}_$rewards' 2>&1 | tee -a '$run/logs/train.log'"
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
  local rates=$run/logs/final_rates.json
  echo '{}' > "$rates"
  if (( RATES_STEPS > 0 )); then
    (cd "$MESHES" && nice -n 19 "$run/indococo-train" rates "$run/cfg/experiments/rates_mix.toml" \
        "$run/checkpoints/$newest" "$OPP_430M" "$RATES_STEPS" 16) 2>/dev/null | tail -1 > "$rates.new" \
        && [[ -s $rates.new ]] && mv "$rates.new" "$rates"
  fi
  "$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); r["steps"]=int(sys.argv[2]); r["flip_reset_rates"]=json.load(open(sys.argv[3])); print(json.dumps(r))' \
      "$log" "$steps" "$rates" >> "$run/logs/evals.jsonl"
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
    tmux new-window -d -t "$session" -n eval "env LD_LIBRARY_PATH='$LIBTORCH_LIB' TRAIN_BIN='$run/indococo-train' PY='$PY' EVERY=$EVERY GAMES=$GAMES PROCS=$PROCS RATES_STEPS=$RATES_STEPS RATES_MIX='$run/cfg/experiments/rates_mix.toml' RATES_OPP='$OPP_430M' MESHES='$MESHES' '$CRATE/tools/eval_loop.sh' '$run/checkpoints' '$run/logs/evals.jsonl' '$run/bots/best' 2>&1 | tee -a '$run/logs/eval_loop.log'; sleep infinity"
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
