#!/bin/bash
#SBATCH -c 3
#SBATCH --mem=64G


Rscript "Mask_training_africa_subpixel.R"
