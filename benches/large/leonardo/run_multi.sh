#!/usr/bin/env -S bash -l
# Usage, from this folder: sbatch run_multi.sh <command...>; it runs in the repository root.
# e.g. sbatch run_multi.sh python benches/large/bench_large.py --debug scaling ...
# Add --qos=boost_qos_dbg --time=00:30:00 to sbatch for short runs.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=large_multi
#SBATCH --account=EUHPC_D30_139
#SBATCH --partition=boost_usr_prod
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
# Booster nodes have no local disk and /tmp is a small tmpfs: spill to Lustre scratch.
export TMPDIR=$CINECA_SCRATCH/src-large/tmp
mkdir -p "$TMPDIR"

REPO=$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)
source "$REPO/.venv/bin/activate"
export CUPY_CACHE_DIR=$REPO/.cupy-cache
# The Booster driver (535) predates CUDA 13: load NVIDIA's forward-compat libcuda.
CUDA_COMPAT=${CUDA_COMPAT:-$REPO/.cuda-compat/usr/local/cuda-13.4/compat}
if [[ -d $CUDA_COMPAT ]]; then
  export LD_LIBRARY_PATH=$CUDA_COMPAT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi
cd "$REPO"

echo "NODE: $SLURMD_NODENAME, CPUS: $SLURM_CPUS_PER_TASK, COMMIT: $(git rev-parse --short HEAD)"
echo "COMMAND: $*"
nvidia-smi -L
nvidia-smi topo -m
python -c "import cupy as c; n=c.cuda.runtime.getDeviceCount(); print(n, [[c.cuda.runtime.deviceCanAccessPeer(i,j) for j in range(n)] for i in range(n)])"

# WARM_DIR=<stack>: read it once, so that the first run of the job is not the only cold one.
if [[ -n ${WARM_DIR:-} ]]; then
  start=$SECONDS
  cat "$WARM_DIR"/*/*.npy > /dev/null
  echo "WARMED: $WARM_DIR in $((SECONDS - start)) s"
fi

srun "$@"

sacct --format=JobID,JobName,Elapsed,MaxRSS,MaxDiskRead,MaxDiskWrite,ExitCode \
  --units=G --jobs="$SLURM_JOB_ID"
