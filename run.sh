#!/bin/bash
#SBATCH --job-name=time-series-l2d
#SBATCH --output=time-series-l2d-%A.out
#SBATCH --error=time-series-l2d-%A.err

#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100-40:1 #h100-47:1
#SBATCH --mem=32G


#SBATCH --mail-user=yuletian@u.nus.edu
#SBATCH --mail-type=START,END,FAIL

# Initialize conda in non-interactive Slurm shell
source /home/y/yuletian/miniconda3/etc/profile.d/conda.sh

# Activate the environment
conda activate Time_Series_L2D

# Verify setup
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

# Test PyTorch & CUDA
python - << 'EOF'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
EOF

# Navigate to your code directory
cd /home/y/yuletian/Time_Series_L2D

# Run your code
python -u slds_imm_router.py -c config/config_etth1.yaml \
  > stdout_${SLURM_JOB_ID}.log \
  2> stderr_${SLURM_JOB_ID}.log

echo "Job completed."
