#!/bin/bash
# Step 3: traced 1-GPU run at the reference size, then the scaling prediction.
set -euo pipefail
N=/home/ramon/src-method/benches/large/nebius
$N/job.sh step3-trace-g1-m4000 uv run python $N/trace.py /dev/shm/nebius/m4000-c64 \
  --chi-out 2000 --devices 1 --scratch-dir /home/ramon/nebius-data/scratch \
  --out $N/raw/trace_g1_m4000.json
uv run python $N/predict.py $N/raw/probe.json $N/raw/trace_g1_m4000.json > $N/raw/prediction.json
