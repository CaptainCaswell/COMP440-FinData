# COMP 440 - FinData

## Program Overview

The purpose of this program is to download stock data, prepare features, and then train 4 different models accross 7 different return horizons, and generate stock predictions. This id done both from historical test data and from a current single date snapshot of the current market.

## Setup

### Python Virtual Environment

This program requires Python to be installed on the machine. It can be [downloaded here](https://www.python.org/downloads/).

In order to setup the virtual enviroment and the required libraries, simply run ```setup_venv.bat``` for Windows machines or ```setup_venv.sh``` on Linux machines.

## Desktop

### Data Download

Price and company data can be downloaded by running, in order, ```get_raw_data.py``` then ```get_raw_info.py```. These steps do not require the use of Slurm job management as their limiting factor is the API rate limit and not processing power.

These programs will create ```data/raw_data.parquet``` and ```data/raw_info.parquet```. They are both capable or resuming from existing files.

### Data Processing

Once the data has been downloaded, running ```build_data.py``` will create ```data/data.parquet``` for model training and testing.

### Model Training

Running ```classify.py``` will run the 4 models across the data set, at all 7 time horizons. If you wish to only run a specific time horizon, you can run ```classify.py --horizon 6m```. ```classify_combine.py``` can be used to compile the csv output of different runs into a single csv.

### Snapshot Predictions

A parallel pipefule generates predictions on a single recent data, so results can be checked against current prices rather than data that's several years old.

* ```get_snapshot.py``` will pull the last 6 years of price data ending yesterday, creating ```data/raw_snapshot.parquet```.
* ```get_raw_info --snapshot``` fetches comapny info for tickers. This ensures that any tickers not part of the original dataset are added.
* ```build_snapshot_data.py``` builds one feature row per ticker as of the most recent available trading day (maximum 7 days prior) into ```snapshot.parquet```.
* ```predict_snapshot.py``` loads the saved model from ```logs/classification/models```, predicts stocks based on the snapshot data, and creates ```logs/classification/snapshot_predictions.csv```.

## Slurm Server

### Data Download and Processing

```quickstart.sbatch``` chains all three training-data steps (```get_raw_data.py```, ```get_raw_info.py```, and ```build_data.py```) into a single Slurm job, stopping early if any step fails. Alternatively, ```build_data.sbatch``` runs just the ```build_data.py``` step as its own Slurm job, once the raw data has already been downloaded.

### Model Training

Running ```submit_classification.sh``` will start an array of Slurm jobs, each running the four models over a single prediction horizon. Once they are all complete the ```combine.py``` will merge the per-horizon outputs.

### Snapshot Predictions

```quick_portfolio.sbatch``` chains all four snapshot-pipeline steps (```get_snapshot.py```, ```get_raw_info.py --snapshot```, ```build_snapshot_data.py```) and ```predict_snapshot.py``` into a single Slurm job, stopping early if any step fails.

## Output

### Snapshot Predictions

A single consolidated file, ```logs/classification/predictions.csv```, holds predictions for every ticker across all models and horizons as of the most recent snapshot date. Analysis notebooks in the ```notebooks``` folder filter/rank this file without needing to re-run predictions.

### Model Evaluation Metrics

The Random Forest importances and Logistic Regression coefficients can be found separately in the ```logs/classification/ranking``` folder, along with per-horizon files capturing accuracy, precision, recall, F1 score, and ROC-AUC.

If the program was run on a Slurm Server, or if ```combine.py``` is run manually, combined versions of these files (all available horizons in one file) will also be created in the same folder.
