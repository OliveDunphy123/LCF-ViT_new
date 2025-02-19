#!/bin/bash
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:gtx_1080_ti:3
#SBATCH --mem-per-gpu=32000

export NUMEXPR_MAX_THREADS=8
export CUDA_VISIBLE_DEVICES="2,3,1"  # Explicitly set GPU order

cd /mnt/guanabana/raid/hdd1/qinxu/Python/LCF-ViT

source activate /mnt/guanabana/raid/hdd1/qinxu/land_cover_fraction

python training/vit2_train.py

