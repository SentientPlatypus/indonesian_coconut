#!/bin/bash
cd /opt/dlami/nvme/st31_validate
python3 /opt/dlami/nvme/st31_stage/measure.py mk || exit 1
export LD_LIBRARY_PATH=/home/ubuntu/ippo_runs/e8b2_1v1/bin/sp/torch/lib INDOCOCO_OBS_EXT=1 RIPPLE_DEVICE=cuda
B=/opt/dlami/nvme/ippo_bins/indococo-train-rip-9d4b2a6; CK=/home/ubuntu/ippo_runs/exp_ST25/checkpoints/1794413355
for t in measure_np.toml measure_p.toml; do nice -n19 $B rates $t $CK self 240000 8 > rates_v2_$t.json 2> rates_v2_$t.err; echo "$t exit=$?"; done
grep -il panic rates_v2_*.err || echo NO_PANICS
python3 /opt/dlami/nvme/st31_stage/measure.py report
