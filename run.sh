#!/bin/bash
#SBATCH --job-name=time-series-l2d
#SBATCH --output=time-series-l2d-%A_%a.out
#SBATCH --error=time-series-l2d-%A_%a.err
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --mail-user=yuletian@u.nus.edu
#SBATCH --mail-type=START,END,FAIL
#
# There are 5 learning rates and 8 beta (cost) settings, making 40 combinations.
# Limit to 6 concurrent jobs with the %6 notation.
#SBATCH --array=0-9

# Change to the directory from which the job was submitted
cd "${SLURM_SUBMIT_DIR}"

python3 slds_imm_router.py -c config/config_etth1.yaml > output_${SLURM_ARRAY_TASK_ID}.log 2>&1
