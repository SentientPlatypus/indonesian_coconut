#!/bin/bash
# TM8 audit scratch build (reward-audit mode). nice 19, cores 28-31, -j 4, own target dir.
source /opt/dlami/nvme/nf/env.sh
TP=$(ls -d /opt/dlami/nvme/torch29/lib/python3*/site-packages/torch)
export LIBTORCH=$TP LD_LIBRARY_PATH=$TP/lib:$LD_LIBRARY_PATH LIBTORCH_BYPASS_VERSION_CHECK=1
export CARGO_TARGET_DIR=/opt/dlami/nvme/tm8_audit/target
cd /opt/dlami/nvme/tm8_audit/code
echo "BUILD START $(date)"
nice -n 19 taskset -c 28-31 cargo build --release -j 4 --features torch,ripple --bin indococo-train 2>&1 | grep -E "^(error|warning: unused)|-->|Finished|^\s+\|" | tail -60
echo "BUILD EXIT ${PIPESTATUS[0]} $(date)"
