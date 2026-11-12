#!/bin/bash
# ST31 bank v2 capture: ST25 (BEST) vs six opponents, windows 1.5/2.25/3.0/3.75/4.5 s, two batches,
# seeds 10000-11549 (cap3_) and 12000-13549 (cap4_): disjoint from diag 0-99, ST30 1000-3399, v1 5000-8549, finals 424242+. CPU, nice 19, sequential.
cd /opt/dlami/nvme/st31_stage
export CUDA_VISIBLE_DEVICES=
P=/home/ubuntu/ippo_runs/st25_final/final_policy.safetensors
for b in 3:10000 4:12000; do
  pre=cap${b%%:*}; o=${b#*:}
  for spec in "fr4 0 400" "galactus 400 300" "dice 800 200" "kira 1000 200" "ripple 1200 200" "chronos 1400 150"; do
    set -- $spec
    nice -n19 ~/coco-venv/bin/python capture_preconcede_v2.py $P $1 $(($o + $2)) $3 ${pre}_$1.f32 > ${pre}_$1.log 2>&1
  done
done
echo done > CAPTURE3_DONE
