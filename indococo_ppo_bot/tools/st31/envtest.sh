#!/usr/bin/env bash
# st_ST31.env logic under bash -u: fresh/not promoted, resume, resume with changed BESTF, fresh/promoted.
set -u; T=$(mktemp -d); E=/opt/dlami/nvme/st31_stage/st_ST31.env
mkdir -p $T/st25_final $T/st30_final $T/ck25/ $T/ck30; echo "cumulative_timesteps = 1" > $T/ck25/stats.toml; cp $T/ck25/stats.toml $T/ck30/
echo $T/ck25 > $T/st25_final/FINAL_CKPT; echo $T/ck30 > $T/st30_final/FINAL_CKPT
run() { RUN=ST31 ST31_R=$T BESTF=$1 START_CK=$2 START=$3 bash -uc "source $E >/dev/null; echo START_CK=\${START_CK#$T/} START=\${START:-} TRAIN_TMPL=\$TRAIN_TMPL file=\$(cat $T/exp_ST31/TRAIN_TMPL)"; }
echo "1 fresh, ST30 not promoted (launcher passes the ST30 ck, BESTF st25):"; run $T/st25_final $T/ck30 999
echo "2 resume of case 1 (checkpoints exist, BESTF unchanged):"; mkdir -p $T/exp_ST31/checkpoints/1794600000; run $T/st25_final $T/ck30 999
echo "3 resume with a different BESTF (template must stay):"; run $T/st30_final $T/ck30 999
rm -rf $T/exp_ST31
echo "4 fresh, ST30 promoted (BESTF st30, launcher passes the same ck):"; run $T/st30_final $T/ck30 999
rm -rf $T
