#!/usr/bin/env bash
# One A100 on the debug QoS; submit from the repository root, e.g.
#   sbatch benches/chebyshev/leonardo/job.sh check
# Several commands: sbatch ... job.sh "prep --n 20" "exact --n 20"
#SBATCH --job-name=cheb
#SBATCH --account=EUHPC_D30_139
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --output=benches/chebyshev/leonardo/logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_PLACES=cores OMP_PROC_BIND=spread
export TMPDIR=$CINECA_SCRATCH/tmp
mkdir -p "$TMPDIR"
export CUPY_CACHE_DIR=$PWD/.cupy-cache
export LD_LIBRARY_PATH=$PWD/.cuda-compat/usr/local/cuda-13.4/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
unset SRC_EINSUM_OPTIMIZE  # gone since the dp default; kept out of the environment
source .venv/bin/activate

for cmd in "$@"; do
  echo "=== $cmd (job $SLURM_JOB_ID, $(hostname))"
  # shellcheck disable=SC2086
  srun python benches/chebyshev/bench_chebyshev.py $cmd
done
