#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3
#SBATCH --mem=32000M

# Initialize micromamba without relying on profile.d
export MAMBA_ROOT_PREFIX=/lustre/scratch/WUR/ESG/xu116/micromamba
export MAMBA_EXE="/lustre/scratch/WUR/ESG/xu116/micromamba/micromamba"

# Direct shell hook evaluation
eval "$($MAMBA_EXE shell hook --shell bash)"

# Activate environment
micromamba activate land_cover_fraction

# Add your project root to PYTHONPATH
export PYTHONPATH="/lustre/scratch/WUR/ESG/xu116/LCF-ViT_new:$PYTHONPATH"

# Load CUDA modules
module load GPU
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

cd /lustre/scratch/WUR/ESG/xu116/LCF-ViT_new/utils

# Print debug information
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo "NumPy version:"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
echo "Torch version and CUDA status:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Run your script
python "grid_search3.py"