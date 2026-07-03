#!/bin/bash
#SBATCH --job-name=get_clusters
#SBATCH --output=logs/get_clusters_%j.out
#SBATCH --time=02:00:00

source .venv/bin/activate
python -u clusters.py