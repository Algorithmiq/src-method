#!/bin/bash
# Step 4: strong scaling, twice per size, then traced runs for the attribution.
set -euo pipefail
N=/home/ramon/src-method/benches/large/nebius
B="uv run python /home/ramon/src-method/benches/large/bench_large.py --debug scaling"
SCRATCH=/home/ramon/nebius-data/scratch
for rep in 1 2; do
  $N/job.sh step4-m4000-r$rep $B /dev/shm/nebius/m4000-c64 --chi-out 2000 \
    --devices 1 2 4 8 --scratch-dir $SCRATCH --results $N/raw/step4_m4000.jsonl
  $N/job.sh step4-m1000-r$rep $B /dev/shm/nebius/m1000-c64 --chi-out 500 \
    --devices 1 2 4 8 --scratch-dir $SCRATCH --results $N/raw/step4_m1000.jsonl
done
for g in 2 4 8; do
  $N/job.sh step4-trace-g$g-m4000 uv run python $N/trace.py /dev/shm/nebius/m4000-c64 \
    --chi-out 2000 --devices $g --scratch-dir $SCRATCH --out $N/raw/trace_g${g}_m4000.json
done
