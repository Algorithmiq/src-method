#!/usr/bin/env bash
# One A100 on the debug QoS; submit from the repository root, e.g.
#   sbatch benches/kicked_ising/leonardo/job.sh check
# Several commands: sbatch ... job.sh "run w17:5:8:src-k1:256:0 ..." "run ..."
# More GPUs: sbatch --gres=gpu:4 ... job.sh "run ...:src-k1-g4:..."
#SBATCH --job-name=ki
#SBATCH --account=EUHPC_D30_139
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --output=benches/kicked_ising/leonardo/logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_PLACES=cores OMP_PROC_BIND=spread
export TMPDIR=$CINECA_SCRATCH/tmp
mkdir -p "$TMPDIR"
export CUPY_CACHE_DIR=$PWD/.cupy-cache
export LD_LIBRARY_PATH=$PWD/.cuda-compat/usr/local/cuda-13.4/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# The venv carries qiskit, stim and autoray 0.11 beyond the lock; `uv run` would
# sync them away, so the job activates it directly.
source .venv/bin/activate
# quimb 1.15 on CuPy 14 needs autoray >= 0.11; `uv sync` (and prek's uv hooks)
# put the locked 0.8.10 back.
python -c "import autoray, sys; sys.exit(tuple(map(int, autoray.__version__.split('.')[:2])) < (0, 11))" \
  || { echo "autoray < 0.11 in .venv: uv pip install autoray==0.11.0" >&2; exit 1; }

for cmd in "$@"; do
  echo "=== $cmd (job $SLURM_JOB_ID, $(hostname))"
  # shellcheck disable=SC2086
  srun python -W ignore::UserWarning benches/kicked_ising/bench_kicked_ising.py $cmd
done
