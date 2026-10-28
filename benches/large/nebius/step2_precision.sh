#!/bin/bash
# Step 2: complex64 against complex128 at D_M = 1000, chi_out = 500, one GPU.
set -euo pipefail
N=/home/ramon/src-method/benches/large/nebius
B="uv run python /home/ramon/src-method/benches/large/bench_large.py --debug run"
S=/dev/shm/nebius OUT=$S/out; mkdir -p $OUT
common="--chi-out 500 --scratch-dir /home/ramon/nebius-data/scratch --results $N/raw/step2_precision.jsonl"
# Each run is its own process; the repeats show the first-run effect.
for rep in 1 2; do
  $N/job.sh step2-c128-s0-r$rep $B $S/m1000-c128 $common --seed 0 --save $OUT/c128-s0-r$rep.npz
  $N/job.sh step2-c64-s0-r$rep $B $S/m1000-c64 $common --seed 0 --save $OUT/c64-s0-r$rep.npz
done
$N/job.sh step2-c128-s1 $B $S/m1000-c128 $common --seed 1 --save $OUT/c128-s1.npz
uv run python $N/distances.py $OUT/c128-s0-r1.npz $OUT/c64-s0-r1.npz $OUT/c128-s1.npz \
  $OUT/c128-s0-r2.npz > $N/raw/step2_distances.jsonl
uv run python $N/distances.py $OUT/c64-s0-r1.npz $OUT/c64-s0-r2.npz >> $N/raw/step2_distances.jsonl
