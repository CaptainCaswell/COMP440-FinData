#!/bin/bash
#SBATCH --job-name=classify_combine
#SBATCH --output=logs/slurm/classify_combine_%j.out
#SBATCH --time=00:10:00

source .venv/bin/activate

python -u combine_classification.py