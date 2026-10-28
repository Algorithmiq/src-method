#!/bin/bash
# Run one timed job: strict FP32 math, one job at a time on the node.
# Usage: job.sh <log-name> <command...>
set -euo pipefail
export CUPY_TF32=0 NVIDIA_TF32_OVERRIDE=0
LOG=/home/ramon/src-method/benches/large/nebius/logs/$1.log; shift
cd /home/ramon/src-method
exec 9>/home/ramon/nebius-data/.timed.lock
flock 9
{ echo "# $(date -Is) $*"; echo "# git $(git -C /home/ramon/src-method rev-parse HEAD)"; } > "$LOG"
/usr/bin/time -v "$@" >> "$LOG" 2>&1
