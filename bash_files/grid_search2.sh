#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=18000

cd /mnt/guanabana/raid/home/qinxu/Python/LCF-ViT/utils

source activate $HOME/land_cover_fraction

python "grid_search2.py"


