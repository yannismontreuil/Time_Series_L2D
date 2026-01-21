#!/bin/bash
#SBATCH --job-name=time-series-l2d
#SBATCH --output=time-series-l2d-%A.out
#SBATCH --error=time-series-l2d-%A.err
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --mail-user=yuletian@u.nus.edu
#SBATCH --mail-type=START,END,FAIL

# Change to the directory from which the job was submitted
cd "${SLURM_SUBMIT_DIR}"

python3 -u slds_imm_router.py -c config/config_etth1.yaml > stdout_${SLURM_JOB_ID}.log 2> stderr_${SLURM_JOB_ID}.log
