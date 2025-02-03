#!/bin/bash
#SBATCH -c 3
#SBATCH --mem=32G


Rscript "Mask_validation_africa_subpixel.R"
