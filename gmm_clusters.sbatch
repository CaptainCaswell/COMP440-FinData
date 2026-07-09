#!/bin/bash
#SBATCH --job-name=gmm
#SBATCH --output=logs/gmm_clusters_%j.out
#SBATCH --time=03:00:00

source .venv/bin/activate
python -u gmm_gen1.py