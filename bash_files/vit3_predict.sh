#!/bin/bash
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:gtx_1080_ti:1
#SBATCH --mem=11000M

# Set GPU device
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=512:expandable_segments=True

cd /mnt/guanabana/raid/home/qinxu/Python/LCF-ViT

source activate $HOME/land_cover_fraction

python "vit3_predict.py"