#!/bin/bash
#SBATCH --job-name=cd_sim           # name that appears in squeue
#SBATCH --time=00:05:00             # wall‑clock limit (HH:MM:SS)
#SBATCH --ntasks=1                  # one main process
#SBATCH --cpus-per-task=2           # CPU cores
#SBATCH --mem=4G                    # memory
#SBATCH --output=cd_sim_%j.out      # output file (%j = job ID)

# Activate your Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate phd_env

# Go to the script location and run
cd ~/PhD/xcorr-cd-wdm-fusion-localization/scripts/sim_Transient_Localization
python -u cd_transient_localization_GCC-PHAT.py