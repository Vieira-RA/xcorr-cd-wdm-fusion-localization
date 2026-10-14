#!/bin/bash
#SBATCH --job-name=crb_figs
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=crb_figs_%j.log
#SBATCH --error=crb_figs_%j.err

# ---------------------------------------------------------------------------
# Build the estimator-comparison figures (F1..F6) from per-pair results.
#   Usage:  sbatch estimator_comparison.slurm [run_dir]
#   Default run_dir: results_crb_full_pairs
# Needs the pair_Z data produced by sweep_config_crb_pairs.yaml.
# ---------------------------------------------------------------------------
RUN_DIR="${1:-results_crb_full_pairs}"

echo "Job started at $(date) on $(hostname)"
echo "RUN_DIR = $RUN_DIR"

# BLAS threading (the script is memory/CPU bound in the median/percentile step)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/miniconda3/etc/profile.d/conda.sh
conda activate phd_env

cd ~/PhD/xcorr-cd-wdm-fusion-localization/scripts/sim_Transient_Localization

python -u estimator_comparison.py "$RUN_DIR"
status=$?
if [ $status -ne 0 ]; then
    echo "ERROR: estimator_comparison failed with exit code $status"
    exit 1
fi

echo "Figures written to $RUN_DIR/figs_estimators/"
echo "Derived data:   $RUN_DIR/estcomp_<n>ch.npz"
echo "Job finished at $(date)"
