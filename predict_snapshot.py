import os
from pathlib import Path

import joblib
import pandas as pd

FEATURES = [
    "ret_1d",
    "ret_1w",
    "ret_1m",
    "ret_6m",
    "ret_1y",
    "ret_3y",
    "ret_5y",
]

HORIZONS = ["1d", "1w", "1m", "6m", "1y", "3y", "5y"]
MODELS = ["logistic_regression", "random_forest", "gradient_boosting", "mlpclassifier"]

SNAPSHOT_FILE = "data/snapshot.parquet"   # recent feature data, one row per ticker
MODEL_FOLDER = "logs/classification/models"
OUT_FOLDER = "logs/classification/snapshot_predictions"
TOP_N = 20
REMOVE_ETFS = True


def load_snapshot() -> pd.DataFrame:
    if not os.path.isfile( SNAPSHOT_FILE ):
        raise FileNotFoundError( f"{SNAPSHOT_FILE} not found. Build a recent snapshot dataset first." )

    df = pd.read_parquet( SNAPSHOT_FILE )

    if REMOVE_ETFS:
        df = df[df["quote_type"] == "EQUITY"].copy()

    missing = set( FEATURES ) - set( df.columns )
    if missing:
        raise ValueError( f"Snapshot is missing required features: {sorted(missing)}" )

    before = len( df )
    df = df.dropna( subset=FEATURES ).reset_index( drop=True )
    dropped = before - len( df )
    if dropped:
        print(f"Dropped {dropped} rows with missing features")

    # If the snapshot has more than one date, keep only the most recent row per ticker
    if df["date"].nunique() > 1:
        df = df.sort_values( "date" ).groupby( "ticker", as_index=False ).tail( 1 ).reset_index( drop=True )

    return df


def predict_horizon(horizon: str, snapshot: pd.DataFrame) -> None:
    Path( OUT_FOLDER ).mkdir( parents=True, exist_ok=True )

    for model_name in MODELS:
        model_path = f"{MODEL_FOLDER}/{horizon}_{model_name}.joblib"
        if not os.path.isfile( model_path ):
            print( f"Skipping {horizon} - {model_name}: no saved model at {model_path}" )
            continue

        pipeline = joblib.load( model_path )

        X = snapshot[FEATURES]
        proba = pipeline.predict_proba(X)[:, 1]
        pred = pipeline.predict(X)

        scored = snapshot[["ticker", "date"]].copy()
        scored["pred_proba"] = proba
        scored["predicted_label"] = pred
        scored = scored.sort_values( "pred_proba", ascending=False ).reset_index( drop=True )

        all_path = f"{OUT_FOLDER}/{horizon}_{model_name}_snapshot_all.csv"
        scored.head( TOP_N ).to_csv( all_path, index=False)

        top_path = f"{OUT_FOLDER}/{horizon}_{model_name}_snapshot_top{TOP_N}.csv"
        scored.head( TOP_N ).to_csv( top_path, index=False)
        
        print( f"Saved for {model_name} @ {horizon}" )


def main():
    snapshot = load_snapshot()
    print( f"Snapshot: {len(snapshot)} tickers as of {snapshot['date'].max()}" )

    for horizon in HORIZONS:
        predict_horizon( horizon, snapshot )


if __name__ == "__main__":
    main()