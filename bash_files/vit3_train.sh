#!/bin/bash
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:gtx_1080_ti:1
#SBATCH --mem=18000M

export CUDA_VISIBLE_DEVICES=3
export NUMEXPR_MAX_THREADS=8

cd /mnt/guanabana/raid/hdd1/qinxu/Python/LCF-ViT

source activate $HOME/land_cover_fraction/

python training/vit3_train.py