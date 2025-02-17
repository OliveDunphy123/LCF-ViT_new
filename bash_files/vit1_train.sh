#!/bin/bash
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:gtx_1080_ti:3
#SBATCH --mem=32000M

export NUMEXPR_MAX_THREADS=8

cd /mnt/guanabana/raid/hdd1/qinxu/Python/LCF-ViT

source activate /mnt/guanabana/raid/hdd1/qinxu/land_cover_fraction

python training/vit1_train.py