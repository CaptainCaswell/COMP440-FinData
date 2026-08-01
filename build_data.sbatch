#!/bin/bash
#SBATCH --job-name=build_dataset
#SBATCH --output=logs/build_dataset_%j.out
#SBATCH --time=01:00:00

source .venv/bin/activate
python -u build_data.py