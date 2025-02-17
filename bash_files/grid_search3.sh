#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3
#SBATCH --mem=32000M

# Print initial state
echo "Initial PATH: $PATH"
echo "Initial PYTHONPATH: $PYTHONPATH"

# Source micromamba initialization with correct path
export MAMBA_ROOT_PREFIX=/lustre/scratch/WUR/ESG/xu116/micromamba
eval "$(/lustre/scratch/WUR/ESG/xu116/micromamba/micromamba shell hook --shell=bash)"

# Activate the environment with correct path
/lustre/scratch/WUR/ESG/xu116/micromamba/micromamba activate land_cover_fraction


# Print environment state after activation
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo "Updated PATH: $PATH"
echo "Updated PYTHONPATH: $PYTHONPATH"

# Add your project root to PYTHONPATH
export PYTHONPATH="/lustre/scratch/WUR/ESG/xu116/LCF-ViT_new:$PYTHONPATH"

# Load CUDA modules
module load GPU
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

cd /lustre/scratch/WUR/ESG/xu116/LCF-ViT_new/utils

# Verify torch and tensorboard are available
python -c "import torch, tensorboard; print(f'PyTorch version: {torch.__version__}')" || echo "Failed to import dependencies"

# Run your script
python "grid_search3.py"

# Deactivate environment at the end
/lustre/scratch/WUR/ESG/xu116/micromamba/micromamba deactivate

