#!/usr/bin/env bash
# Render ST31 the way st_launch.sh does, for both templates, into st31_validate/{np,p}.
# np = ST30 not promoted (BESTF st25_final -> train_st31.toml), p = ST30 promoted (BESTF st30_final -> train_st31p.toml).
set -eu
R=/home/ubuntu/ippo_runs; S=/opt/dlami/nvme/st31_stage; V=/opt/dlami/nvme/st31_validate
C=/home/ubuntu/ippo_runs/code_fr9/indococo_ppo_bot
rm -rf $V; mkdir -p $V
for case in np:st25_final:20727500000 p:st30_final:21227500000; do
  IFS=: read k bf START0 <<< "$case"; X=$V/$k; FR=$V/fakeR_$k
  mkdir -p $X/cfg $X/checkpoints $X/opponents $FR
  cp -r $C/experiments $C/config $X/cfg/
  cp $S/train_st31.toml $S/train_st31p.toml $S/st_ST31.env $X/cfg/experiments/
  RUN=ST31 BESTF=/nonexistent/$bf ST31_R=$FR START_CK=x bash -uc "source $X/cfg/experiments/st_ST31.env >/dev/null; declare -p TRAIN_TMPL EXTRA_OPPS TPI MB LR ENT GAMMA OPPCH DECAY MINLR HOLDREL GAL_WEIGHT MEGA_WEIGHT CHUNK B0" > $X/env.out
  source $X/env.out; EXTRA_OPPS=${EXTRA_OPPS//$FR/$R}
  mkdir -p $X/opponents/FR4_final; cp $R/exp_FR4/checkpoints/1793481832/actor.bpk $X/opponents/FR4_final/
  for e in $EXTRA_OPPS; do n=${e%%:*}; r=${e#*:}; d=${r%:*}; mkdir -p $X/opponents/$n; cp $d/actor.bpk $X/opponents/$n/; done
  left=334; SNAP=$R/fr7_snap; THREADS=24; GAMES=512
  sed -e "s|@RUN@|$X|g" -e "s|@ITERATIONS@|$left|" -e "s|@WARM@|$((START0 + ${WARM:-10000000}))|" -e "s|@HOLD@|$((START0 + HOLDREL))|" \
      -e "s|@SNAP@|$SNAP/train.f32|" -e "s|@TPI@|$TPI|" -e "s|@MB@|$MB|" -e "s|@LR@|$LR|g" -e "s|@ENT@|$ENT|" -e "s|@GAMMA@|$GAMMA|" \
      -e "s|@OPPCH@|$OPPCH|" -e "s|@DECAY@|$DECAY|" -e "s|@MINLR@|$MINLR|" -e "s|^threads = .*|threads = $THREADS|" -e "s|^games = .*|games = $GAMES|" \
      $X/cfg/experiments/${TRAIN_TMPL} > $X/train_st.toml
  printf "\n[[opponents.external]]\nname = \"galactus\"\nkind = \"galactus\"\npath = \"%s\"\nweight = %s\n" "/home/ubuntu/eval_bots/galactus/rust" "$GAL_WEIGHT" >> $X/train_st.toml
  printf "\n[[opponents.external]]\nname = \"megabyte\"\nkind = \"megabyte\"\npath = \"%s\"\nweight = %s\n" "/home/ubuntu/eval_bots/megabyte" "$MEGA_WEIGHT" >> $X/train_st.toml
  for e in $EXTRA_OPPS; do n=${e%%:*}
    printf '\n[[opponents.fixed]]\nname = "%s"\ncheckpoint = "%s"\nweight = %s\n' "$n" "$X/opponents/$n" "${e##*:}" >> $X/train_st.toml; done
  echo "$k: TRAIN_TMPL=$TRAIN_TMPL B0=$B0 CHUNK=$CHUNK MEGA=$MEGA_WEIGHT GAL=$GAL_WEIGHT tmplfile=$(cat $FR/exp_ST31/TRAIN_TMPL)"
done
