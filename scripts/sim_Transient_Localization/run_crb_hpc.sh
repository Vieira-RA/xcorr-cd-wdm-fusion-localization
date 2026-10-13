#!/bin/bash
#SBATCH --job-name=crb_sweep
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --output=crb_sweep_%j.log
#SBATCH --error=crb_sweep_%j.err

# ====================== Configuration ======================
CONFIG=sweep_config_crb.yaml
RUN_GENERATOR=yes      # yes: generate Jones data first (now ~30x faster)
RUN_PROCESSOR=yes      # yes: run the Monte-Carlo CRB sweep (checkpointed per fibre)
RUN_ANALYSIS=yes       # yes: build heatmaps / summaries from the results

# ====================== Setup ======================
LOGFILE="crb_sweep_${SLURM_JOB_ID}.log"
ERRFILE="crb_sweep_${SLURM_JOB_ID}.err"

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Config      : $CONFIG"
echo "Output log  : $LOGFILE"
echo "Error log   : $ERRFILE"
echo "To monitor  : tail -f $LOGFILE"

# BLAS threading and joblib workers (set per stage below).
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRB_N_JOBS=$SLURM_CPUS_PER_TASK     # joblib workers for the Monte-Carlo stage

source ~/miniconda3/etc/profile.d/conda.sh
conda activate phd_env

cd ~/PhD/xcorr-cd-wdm-fusion-localization/scripts/sim_Transient_Localization

# Sanity check - the shared library and dependencies must be importable
python3 -c "import numpy, scipy, matplotlib, yaml, tqdm, joblib; import crb, channel_selection, fiber_propagation, quaternion, rotations, signal_processing, estimation, chromatic_dispersion, pmd_model" || {
    echo "ERROR: environment/imports failed. Is sop-simulation-lib installed in phd_env?"
    exit 1
}

# ====================== Stage 1 - Jones generation ======================
if [ "$RUN_GENERATOR" = "yes" ]; then
    echo ""
    echo "====================== Stage 1: Jones matrix generation ======================"
    # single process, multithreaded BLAS (batched tree-product)
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
    export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
    export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
    python3 generate_jones_crb.py "$CONFIG"
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

# ====================== Stage 2 - Monte-Carlo CRB sweep ======================
if [ "$RUN_PROCESSOR" = "yes" ]; then
    echo ""
    echo "====================== Stage 2: Monte-Carlo CRB sweep (checkpointed) ======================"
    # many joblib processes -> single-threaded BLAS each (avoid oversubscription)
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export CRB_N_JOBS=$SLURM_CPUS_PER_TASK
    python3 process_crb_sweep.py "$CONFIG"
    status=$?
    if [ $status -ne 0 ]; then
        echo "ERROR: Sweep processing failed with exit code $status"
        exit 1
    fi
    echo "Sweep processing completed successfully."
else
    echo ""
    echo "====================== Skipping sweep processing ======================"
fi

# ====================== Stage 3 - Analysis ======================
if [ "$RUN_ANALYSIS" = "yes" ]; then
    echo ""
    echo "====================== Stage 3: Analysis and figures ======================"
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
    export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
    export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
    python3 analyse_crb_sweep.py "$CONFIG"
    status=$?
    if [ $status -ne 0 ]; then
        echo "ERROR: Analysis failed with exit code $status"
        exit 1
    fi
    echo "Analysis completed successfully."
else
    echo ""
    echo "====================== Skipping analysis ======================"
fi

echo ""
echo "Job finished at $(date)"
echo "Log file: $LOGFILE"
