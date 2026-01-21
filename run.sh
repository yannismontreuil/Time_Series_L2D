#!/bin/bash
#SBATCH --job-name=time-series-l2d
#SBATCH --output=time-series-l2d-%A.out
#SBATCH --error=time-series-l2d-%A.err

#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100-40:1 #h100-47:1
#SBATCH --mem=32G


#SBATCH --mail-user=yuletian@u.nus.edu
#SBATCH --mail-type=START,END,FAIL

# Method 1: Direct activation (try this first)
source /mnt/scratch/y/yuletian/adv_l2d/envs/h100/bin/activate

# Method 2: If Method 1 fails, try conda (uncomment these lines)
# eval "$(/mnt/scratch/y/yuletian/miniconda3/bin/conda shell.bash hook)"
# conda activate /mnt/scratch/y/yuletian/adv_l2d/envs/h100

# Verify setup
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

# Test PyTorch
python -c "import torch; print('PyTorch version:', torch.__version__)" || echo "PyTorch not found!"

# Navigate to your code directory
cd /home/y/yuletian/Time_Series_L2D

# Run your code
python3 -u slds_imm_router.py -c config/config_etth1.yaml > stdout_${SLURM_JOB_ID}.log 2> stderr_${SLURM_JOB_ID}.log
echo "Job completed."
