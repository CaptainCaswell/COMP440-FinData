#!/bin/bash
#SBATCH --job-name=hdbscan
#SBATCH --output=logs/slurm/hdbscan_clusters_%j.out
#SBATCH --time=03:00:00

source .venv/bin/activate
python -u hdbscan_gen1.py