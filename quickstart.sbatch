#!/bin/bash
#SBATCH --job-name=findata-dataset
#SBATCH --output=logs/slurm/%x_%j.out

set -e

source .venv/bin/activate

mkdir -p logs/slurm

set -e

echo "=== get_raw_data.py started at $(date) ==="
python -u get_raw_data.py

echo "=== get_raw_info.py started at $(date) ==="
python -u get_raw_info.py

echo "=== build_data.py started at $(date) ==="
python -u build_data.py

echo "=== Pipeline complete at $(date) ==="