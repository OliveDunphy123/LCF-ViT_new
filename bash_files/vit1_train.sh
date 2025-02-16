#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --mem=48000M

export NUMEXPR_MAX_THREADS=12

cd /mnt/guanabana/raid/hdd1/qinxu/Python/LCF-ViT

micromamba activate $HOME/land_cover_fraction/

python training/vit1_train.py