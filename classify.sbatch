#!/bin/bash
#SBATCH --job-name=classify
#SBATCH --output=logs/slurm/classify_%A_%a.out
#SBATCH --array=0-6
#SBATCH --time=01:00:00

HORIZONS=("1d" "1w" "1m" "6m" "1y" "3y" "5y")

source .venv/bin/activate

python -u simple_classify.py --horizon ${HORIZONS[$SLURM_ARRAY_TASK_ID]}