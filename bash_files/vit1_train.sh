#!/bin/bash
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:gtx_1080_ti:1
#SBATCH --mem=18000M

export CUDA_VISIBLE_DEVICES=3
export NUMEXPR_MAX_THREADS=8

# Reduce batch size to avoid OOM
BATCH_SIZE=8 python vit1_train.py