#!/bin/bash
#SBATCH --job-name=build_dataset
#SBATCH --time=01:00:00

module load python3
source .venv/bin/activate
python build_data.py