#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# CRB study - HPC pipeline driver
#
#   ./run_crb_hpc.sh sweep_config_crb.yaml
#
# Stages (Jones -> Monte-Carlo -> analysis) run sequentially; the Monte-Carlo
# stage parallelises over noise realisations via joblib (run.n_jobs, set it to
# the number of allocated cores).  It checkpoints after every fibre, so a
# resubmission resumes where it stopped.
# ---------------------------------------------------------------------------
set -euo pipefail

CFG="${1:-sweep_config_crb.yaml}"

# --- environment: adapt to your HPC -----------------------------------------
# module load python/3.11
# conda activate <env>
# export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# The shared library (sop-simulation-lib) must be importable, e.g.:
#   pip install -e /path/to/sop-simulation-lib
# or: export PYTHONPATH=/path/to/sop-simulation-lib/src:$PYTHONPATH

echo "=== [1/3] generating Jones matrices ==="
python -u generate_jones_crb.py "$CFG"

echo "=== [2/3] Monte-Carlo CRB sweep (checkpointed per fibre) ==="
python -u process_crb_sweep.py "$CFG"

echo "=== [3/3] analysis and figures ==="
python -u analyse_crb_sweep.py "$CFG"

echo "=== done ==="
