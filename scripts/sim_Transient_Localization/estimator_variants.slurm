#!/bin/bash
#SBATCH --job-name=crb_variants
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=crb_variants_%j.log
#SBATCH --error=crb_variants_%j.err

# ---------------------------------------------------------------------------
# Controlled estimator-variant test (parabolic / phase-slope / GCC-PHAT / signed)
#   Usage:  sbatch estimator_variants.slurm [config] [count]
# ---------------------------------------------------------------------------
CONFIG="${1:-sweep_config_crb_equispaced.yaml}"
COUNT="${2:-15}"

echo "Job started at $(date) on $(hostname)"
echo "config=$CONFIG  count=$COUNT"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate phd_env

cd ~/PhD/xcorr-cd-wdm-fusion-localization/scripts/sim_Transient_Localization
python -u estimator_variants.py "$CONFIG" "$COUNT"
status=$?
if [ $status -ne 0 ]; then
    echo "ERROR: estimator_variants failed with exit code $status"
    exit 1
fi
echo "Job finished at $(date)"
