#!/bin/bash
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=8000

cd /mnt/guanabana/raid/hdd1/qinxu/Python/LCF-ViT/data

source activate $HOME/land_cover_fraction

python "my_whole_datset.py"