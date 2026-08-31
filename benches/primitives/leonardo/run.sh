#!/usr/bin/env -S bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=src_bench
#SBATCH --account=ALQ_prod2526_0
#SBATCH --partition=dcgp_usr_prod
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=28
#SBATCH --mem=440G
#SBATCH --exclusive
#SBATCH --output=logs/1000_%j.out

# Python application's loggin level
export LOG_LEVEL_SRC=DEBUG

# Automatic OpenMP binding
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# Python uv environment
source ../../.venv/bin/activate

# Problem size
NSITES=25
CHI=1000
COMPARE="no"    # "yes", "no"
RUN_TYPE="src"  # "src", "quimb", "both"

# Report
echo "OPENMP THREADS: $OMP_NUM_THREADS"
echo "NODE: $SLURMD_NODENAME"
echo "CPUS: $SLURM_CPUS_PER_TASK"

# Run types
if [ "$RUN_TYPE" = "both" ]; then
  srun --verbose --label python bench_mpo_mpo.py --run quimb --compare $COMPARE --n_sites $NSITES --chi_out $CHI
  srun --verbose --label python bench_mpo_mpo.py --run src --compare $COMPARE --n_sites $NSITES --chi_out $CHI
elif [ "$RUN_TYPE" = "src" ]; then
  srun --verbose --label python bench_mpo_mpo.py --run src --compare $COMPARE --n_sites $NSITES --chi_out $CHI
elif [ "$RUN_TYPE" = "quimb" ]; then
  srun --verbose --label python bench_mpo_mpo.py --run quimb --compare $COMPARE --n_sites $NSITES --chi_out $CHI
else
  echo "Unknown run type: $RUN_TYPE"
  exit 1
fi

# Report stats
echo -e "\n\n#------------------------#"
echo "Task and CPU usage stats:"
sacct --format=JobID,JobName,NCPUS,NNodes,NTasks,AveCPU,MinCPU,MinCPUNode,MinCPUTask,Elapsed,ExitCode --jobs="${SLURM_JOBID}"

echo "Memory usage stats:"
sacct --format=JobID,JobName,AveRSS%-50,MaxRSS%-50,MaxRSSNode,MaxRSSTask,AvePages,MaxPages,MaxPagesNode,MaxPagesTask --units=G --jobs="${SLURM_JOBID}"

echo "Disk usage stats:"
sacct --format=JobID,JobName,AveDiskRead,MaxDiskRead,MaxDiskReadNode,MaxDiskReadTask,AveDiskWrite,MaxDiskWrite,MaxDiskWriteNode,MaxDiskWriteTask --units=G --jobs="${SLURM_JOBID}"

echo "Trackable resources usage stats:"
sacct --format=JobID,JobName,AllocNodes,AllocCPUS,AllocTRES%-100 --units=G --jobs="${SLURM_JOBID}"

echo "Trackable resources (ingress) usage stats:"
sacct --format=JobID,JobName,TRESUsageInTot%-70,TRESUsageInAve%-70,TRESUsageInMax%-70,TRESUsageInMaxNode%-70,TRESUsageInMaxTask%-70 --units=G --jobs="${SLURM_JOBID}"

echo "Trackable resources (egress) usage stats:"
sacct --format=JobID,JobName,TRESUsageOutTot%-70,TRESUsageOutAve%-70,TRESUsageOutMax%-70,TRESUsageOutMaxNode%-70,TRESUsageOutMaxTask%-70 --units=G --jobs="${SLURM_JOBID}"
