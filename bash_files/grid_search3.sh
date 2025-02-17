#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3
#SBATCH --mem=32000M

export NUMEXPR_MAX_THREADS=8
export MAMBA_ROOT_PREFIX="/lustre/scratch/WUR/ESG/xu116/micromamba"
export PATH="/lustre/scratch/WUR/ESG/xu116/micromamba:$PATH"

eval "$(micromamba shell hook -s bash)"
micromamba activate land_cover_fraction

cd /lustre/scratch/WUR/ESG/xu116/LCF-ViT_new/utils

python "grid_search3.py"