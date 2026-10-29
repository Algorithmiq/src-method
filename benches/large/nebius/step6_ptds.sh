#!/bin/bash
# Supplementary, after the step 4 analysis: the collectives run on CuPy's legacy
# null stream, where a peer copy serializes with the copy in the other direction
# (allgather_check.py). Rerun G = 4 and 8 with per-thread default streams, which
# needs no code change, and compare the output with the 1-GPU output of step 5 (a).
set -euo pipefail
N=/home/ramon/src-method/benches/large/nebius
B="env CUPY_CUDA_PER_THREAD_DEFAULT_STREAM=1 uv run python /home/ramon/src-method/benches/large/bench_large.py --debug run"
S=/dev/shm/nebius OUT=$S/out
for g in 8 4; do
  $N/job.sh step6-ptds-g$g-m4000 $B $S/m4000-c64 --chi-out 2000 --devices $g \
    --results $N/raw/step6_ptds_m4000.jsonl --save $OUT/step6-ptds-g$g.npz
  $N/job.sh step6-ptds-g$g-m1000 $B $S/m1000-c64 --chi-out 500 --devices $g \
    --results $N/raw/step6_ptds_m1000.jsonl
done
uv run python $N/distances.py $OUT/step5-a.npz $OUT/step6-ptds-g8.npz \
  $OUT/step6-ptds-g4.npz > $N/raw/step6_distances.jsonl
