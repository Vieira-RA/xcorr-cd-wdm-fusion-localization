#!/bin/bash
#SBATCH --job-name=gen_fibres_4_9
#SBATCH --time=12:00:00              # adjust as needed (rough estimate: ~2 min per fibre per bandwidth, so 6 fibres × 60 bandwidths ≈ 12 h max)
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --output=gen_fibres_%j.log
#SBATCH --error=gen_fibres_%j.err

# ====================== Setup ======================
LOGFILE="gen_fibres_${SLURM_JOB_ID}.log"
ERRFILE="gen_fibres_${SLURM_JOB_ID}.err"

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

# ====================== Generate extra fibres ======================
echo ""
echo "====================== Stage 1: Jones matrix generation (fibres 4‑9) ======================"
python3 generate_jones_data_MF.py
status=$?
if [ $status -ne 0 ]; then
    echo "ERROR: Jones generation failed with exit code $status"
    exit 1
fi
echo "Jones generation completed successfully."

echo ""
echo "Job finished at $(date)"
echo "Log file: $LOGFILE"