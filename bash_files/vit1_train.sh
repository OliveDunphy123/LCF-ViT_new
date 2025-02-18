#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3
#SBATCH --mem=32000M
#SBATCH --time=72:00:00

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

# Set CUDA environment variables explicitly
export CUDA_VISIBLE_DEVICES=0,1,2  # Make 3 GPUs visible to the process
export CUDA_HOME=/usr/local/cuda-11.8  # Adjust this path based on your system
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Change directory to training folder
cd /lustre/scratch/WUR/ESG/xu116/LCF-ViT_new/training

# Print debug information to confirm environment
echo "Starting job with environment:"
echo "Current directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Print Python environment details
python -c "
import sys
import torch
import torchgeo
import numpy as np
print('\nPython environment details:')
print('Python version:', sys.version)
print('PyTorch version:', torch.__version__)
print('TorchGeo version:', torchgeo.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('Number of GPUs:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB')
else:
    print('WARNING: CUDA is not available! Check your environment setup.')
"

# If CUDA is not available, exit
python -c "
import torch
if not torch.cuda.is_available():
    print('ERROR: CUDA is not available. Exiting...')
    exit(1)
"

# If we get here, CUDA is available, so run the training script
echo "\nStarting training script..."
python "vit1_train.py"