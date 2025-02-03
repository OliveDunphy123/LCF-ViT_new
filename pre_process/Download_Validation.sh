#!/bin/bash
#SBATCH -c 15
#SBATCH --mem=20G

umask g+w
Rscript "../Download_Validation.R"
