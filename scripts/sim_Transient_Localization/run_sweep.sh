#!/bin/bash
#SBATCH --job-name=cd_sweep
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --output=cd_sweep_%j.log
#SBATCH --error=cd_sweep_%j.err

# ====================== Configuration ======================
RUN_GENERATOR=no

# ====================== Setup ======================
LOGFILE="cd_sweep_${SLURM_JOB_ID}.log"
ERRFILE="cd_sweep_${SLURM_JOB_ID}.err"

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Output log  : $LOGFILE"
echo "Error log   : $ERRFILE"
echo "To monitor  : tail -f $LOGFILE"

# Allow NumPy/SciPy to use all requested cores
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/miniconda3/etc/profile.d/conda.sh
conda activate phd_env

cd ~/PhD/xcorr-cd-wdm-fusion-localization/scripts/sim_Transient_Localization

# ====================== Step 1 – Generate Jones data (optional) ======================
if [ "$RUN_GENERATOR" = "yes" ]; then
    echo ""
    echo "====================== Stage 1: Jones matrix generation ======================"
    python3 generate_jones_data_MF.py
    status=$?
    if [ $status -ne 0 ]; then
        echo "ERROR: Jones generation failed with exit code $status"
        exit 1
    fi
    echo "Jones generation completed successfully."
else
    echo ""
    echo "====================== Skipping Jones generation ======================"
fi

# ====================== Step 2 – Process sweep ======================
echo ""
echo "====================== Stage 2: Post‑processing ======================"
python3 process_sweep_MF.py
status=$?
if [ $status -ne 0 ]; then
    echo "ERROR: Post‑processing failed with exit code $status"
    exit 1
fi
echo "Post‑processing completed successfully."

echo ""
echo "Job finished at $(date)"
echo "Log file: $LOGFILE"