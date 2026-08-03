#!/bin/bash
#SBATCH --job-name=findata-predict
#SBATCH --output=logs/slurm/%x_%j.out

set -e

source .venv/bin/activate

mkdir -p logs/slurm

echo "=== get_snapshot.py started at $(date) ==="
python -u get_snapshot.py

echo "=== get_raw_info.py --snapshot started at $(date) ==="
python -u get_raw_info.py --snapshot

echo "=== build_snapshot_data.py started at $(date) ==="
python -u build_snapshot_data.py

echo "=== predict_snapshot.py started at $(date) ==="
python -u predict_snapshot.py

echo "=== Pipeline complete at $(date) ==="