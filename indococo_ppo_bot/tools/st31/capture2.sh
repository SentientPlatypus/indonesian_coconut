#!/bin/bash
# ST31 capture: ST25 (BEST) vs six opponents, seeds 7000+ (second batch, more distinct events) (disjoint from diag 0-99 and ST30's 1000-1399/2000-2199/3000-3399), CPU, nice 19, sequential.
cd /opt/dlami/nvme/st31_stage
export CUDA_VISIBLE_DEVICES=
P=/home/ubuntu/ippo_runs/st25_final/final_policy.safetensors
for spec in "fr4 7000 400" "galactus 7400 300" "dice 7800 200" "kira 8000 200" "ripple 8200 200" "chronos 8400 150"; do
  set -- $spec
  nice -n19 ~/coco-venv/bin/python capture_preconcede.py $P $1 $2 $3 cap2_$1.f32 > cap2_$1.log 2>&1
done
echo done > CAPTURE2_DONE
