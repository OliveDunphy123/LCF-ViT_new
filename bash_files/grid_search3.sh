#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3
#SBATCH --mem=32000M

# Add pip to PATH
export PATH="$HOME/.local/bin:$PATH"

# Load CUDA modules
module load GPU
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

cd /lustre/scratch/WUR/ESG/xu116/LCF-ViT_new/utils

python "grid_search3.py"