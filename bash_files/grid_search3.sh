#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3
#SBATCH --mem=32000M

# Initialize micromamba
export MAMBA_ROOT_PREFIX=/lustre/scratch/WUR/ESG/xu116/micromamba
source $MAMBA_ROOT_PREFIX/etc/profile.d/micromamba.sh

# Initialize shell and activate environment
micromamba shell init --shell=bash --root-prefix=$MAMBA_ROOT_PREFIX
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
echo "Torch version and CUDA status:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Run your script
python "grid_search3.py"
