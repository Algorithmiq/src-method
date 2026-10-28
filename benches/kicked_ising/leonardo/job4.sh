#!/usr/bin/env bash
# Four independent single-GPU runs on one node of the debug QoS: each argument is
# the command line of one process, pinned to its own A100, 8 cores and 120 GB.
#   sbatch benches/kicked_ising/leonardo/job4.sh "run cfg ..." "run cfg ..." ...
#SBATCH --job-name=ki4
#SBATCH --account=EUHPC_D30_139
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --output=benches/kicked_ising/leonardo/logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_PLACES=cores OMP_PROC_BIND=spread
export TMPDIR=$CINECA_SCRATCH/tmp
mkdir -p "$TMPDIR"
export CUPY_CACHE_DIR=$PWD/.cupy-cache
export LD_LIBRARY_PATH=$PWD/.cuda-compat/usr/local/cuda-13.4/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
source .venv/bin/activate

for cmd in "$@"; do
  echo "=== $cmd (job $SLURM_JOB_ID, $(hostname))"
  # shellcheck disable=SC2086
  srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --mem=118G --exact \
    python -W ignore::UserWarning benches/kicked_ising/bench_kicked_ising.py $cmd &
done
wait
