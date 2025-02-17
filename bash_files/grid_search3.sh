#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3        # Request 3 GPUs without specifying type
#SBATCH --mem=32000M

export NUMEXPR_MAX_THREADS=8

cd /lustre/scratch/WUR/ESG/xu116/LCF-ViT_new/utils

micromamba activate /lustre/scratch/WUR/ESG/xu116/land_cover_fraction

python "grid_search3.py"