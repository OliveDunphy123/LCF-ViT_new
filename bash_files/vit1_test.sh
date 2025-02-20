#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=32000M
#SBATCH --time=72:00:00

# Load CUDA modules
module load GPU
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

# Initialize micromamba without relying on profile.d
export MAMBA_ROOT_PREFIX=/lustre/scratch/WUR/ESG/xu116/micromamba
export MAMBA_EXE="/lustre/scratch/WUR/ESG/xu116/micromamba/micromamba"

# Direct shell hook evaluation
eval "$($MAMBA_EXE shell hook --shell bash)"

# Activate environment
micromamba activate land_cover_fraction

# Add your project root to PYTHONPATH
export PYTHONPATH="/lustre/scratch/WUR/ESG/xu116/LCF-ViT_new:$PYTHONPATH"


cd /lustre/scratch/WUR/ESG/xu116/LCF-ViT_new/inference

# Print debug information to confirm environment
echo "Starting job with environment:"
echo "Node: $(hostname)"
echo "GPU information:"
nvidia-smi
python -c "
import sys
import torch
import torchgeo
import numpy as np
print('Python version:', sys.version)
print('PyTorch version:', torch.__version__)
print('TorchGeo version:', torchgeo.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU device:', torch.cuda.get_device_name(0))
"

# Run your script
python "vit1_test.py"

