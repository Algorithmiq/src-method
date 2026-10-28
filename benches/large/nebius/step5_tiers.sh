#!/bin/bash
# Step 5: one GPU, reference size, complex64, three tier plans; scratch on the root
# disk (no NVMe on this VM). (c) runs twice: as is, where the spill can stay in the
# page cache, and in a scope whose memory cap leaves the page cache no room for it.
set -euo pipefail
N=/home/ramon/src-method/benches/large/nebius
B="uv run python /home/ramon/src-method/benches/large/bench_large.py --debug run"
S=/dev/shm/nebius OUT=$S/out; mkdir -p $OUT
common="$S/m4000-c64 --chi-out 2000 --scratch-dir /home/ramon/nebius-data/scratch --results $N/raw/step5_tiers.jsonl"
$N/job.sh step5-a $B $common --save $OUT/step5-a.npz
$N/job.sh step5-b $B $common --gpu-memory 16GB --save $OUT/step5-b.npz
$N/job.sh step5-c1 $B $common --host-memory 50GB --save $OUT/step5-c1.npz
# Cap: the measured peak RSS of (c1) plus 8 GB.
peak=$(grep -h "Maximum resident set size" $N/logs/step5-c1.log | awk '{print $NF}')
cap=$(( peak / 1024 / 1024 + 8 ))G
echo "c2 MemoryMax=$cap" > $N/logs/step5-c2-cap.log
$N/job.sh step5-c2 systemd-run --user --scope -p MemoryMax=$cap -p MemorySwapMax=0 --quiet \
  $B $common --host-memory 50GB --save $OUT/step5-c2.npz
uv run python $N/distances.py $OUT/step5-a.npz $OUT/step5-b.npz $OUT/step5-c1.npz \
  $OUT/step5-c2.npz > $N/raw/step5_distances.jsonl
