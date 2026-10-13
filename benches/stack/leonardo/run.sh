#!/usr/bin/env -S bash -l
# Usage, from this folder: sbatch run.sh evolve [bench_depth.py evolve options]
# accuracy and timing are laptop-sized; run them locally.
# Add --qos=boost_qos_dbg (30 min, 2 nodes) to sbatch for short test runs.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=stack
#SBATCH --account=EUHPC_D30_139
#SBATCH --partition=boost_usr_prod
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

REPO=$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)
source "$REPO/.venv/bin/activate"
export CUPY_CACHE_DIR=$REPO/.cupy-cache
# The Booster driver (535) predates CUDA 13: load NVIDIA's forward-compat libcuda.
CUDA_COMPAT=${CUDA_COMPAT:-$REPO/.cuda-compat/usr/local/cuda-13.4/compat}
if [[ -d $CUDA_COMPAT ]]; then
  export LD_LIBRARY_PATH=$CUDA_COMPAT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi

echo "NODE: $SLURMD_NODENAME, CPUS: $SLURM_CPUS_PER_TASK, ARGS: $*"

srun python "$REPO/benches/stack/bench_depth.py" "$@" --output "logs/$1_$SLURM_JOB_ID.md"
