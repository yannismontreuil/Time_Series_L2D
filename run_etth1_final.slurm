#!/bin/bash
#SBATCH --job-name=l2d_etth1
#SBATCH --output=logs/etth1_%j.out
#SBATCH --error=logs/etth1_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mail-user=yuletian@u.nus.edu
#SBATCH --mail-type=START,END,FAIL

# Create logs directory if it doesn't exist
mkdir -p logs

echo "================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "================================================="

# Change to the directory from which the job was submitted
cd "${SLURM_SUBMIT_DIR}"

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate Time_Series_L2D

# Print environment info
echo "Python version:"
python --version
echo "Conda environment:"
conda info --envs | grep '*'

# Run the ETTh1 experiment
python slds_imm_router.py --config config/config_etth1_exp.yaml

echo "================================================="
echo "End time: $(date)"
echo "Job completed successfully"
echo "================================================="
