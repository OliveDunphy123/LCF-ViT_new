#!/bin/bash
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:gtx_1080_ti:2
#SBATCH --mem=32000M

export NUMEXPR_MAX_THREADS=8

cd /mnt/guanabana/raid/hdd1/qinxu/Python/LCF-ViT

source activate $HOME/land_cover_fraction/

python training/vit2_train.py