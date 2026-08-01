# COMP 440 - FinData

## Program Overview

The purpose of this program is to download stock data, prepare features, and then train 4 different models on that data.

## Setup

### Python Virtual Environment

This program requires Python to be installed on the machine. It can be [downloaded here](https://www.python.org/downloads/).

In order to setup the virtual enviroment and the required libraries, simply run setup_venv.bat for Windows machines or setup_venv.sh on Linux machines.

### Data Download

Price and company data can be downloaded by running, in order, ```get_raw_data.py``` then ```get_raw_info.py```. These steps do not require the use of Slurm job management as their limiting factor is the API rate limit and not processing power.

These programs will create ```raw_data.parquet``` and ```data/raw_info.parquet```. They are both capable or resuming from existing files.

### Data Processing

#### Desktop Computer

Once the data has been downloaded, running ```build_data.py``` will create ```data/data.parquet``` for model training and testing.

#### Slurm Server

Once the data has been downloaded, running ```build_data.sbatch``` will create a Slurm job to run ```build_data.py```, which will create ```data/data.parquet``` for model training and testing.

### Model Training

#### Desktop Computer

Running ```classify.py``` will run the 4 models across the data set, at all 7 time horizons. If you wish to only run a specific time horizon, you can run ```classify.py --horizon 6m```. ```classify_combine.py``` can be used to compile the csv output of different runs into a single csv.

#### Slurm Server

Running ```submit_classification.sh``` will start an array of Slurm jobs, each running the four models over a single prediction horizon.

## Output

There are two main outputs from running the classification:

### Stock predictions

For easier viewing, separate csv files are available for each model, time horizon, and in different levels of detail. The levels are top 20, top 5%, and all. These can be found in the ```logs/classify/stocks``` folder.

### Model Evaluation Metrics

The importances for Random Forest and coefficients can be found seperatly in the ```logs/classify``` folder.

There are also files for each time horizon that captures the accuracy, precision, recall, f1 score, and ROC-AUC.

If the program was run on a Slurm Server, or if combine.py is run, there will also be a combined versions of these files with all avaiable horizons in one file.
