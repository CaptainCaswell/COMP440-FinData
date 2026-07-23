import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import sys
from classify_features import FEATURES

# Files
STOCK_FILE = "data/data.parquet"
LOG_FOLDER = "logs/regression"

# Columns
TARGET_COL = "future_excess_5y"

RANDOM_STATE = 42
TEST_FRACTION = 0.2
GAP_DAYS = 252

REMOVE_ETFS = True

GEN = 1

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()


def ensure_log_directory():
    Path( LOG_FOLDER ).mkdir( parents=True, exist_ok=True )


def load_data() -> pd.DataFrame:
    if not os.path.isfile( STOCK_FILE ):
        raise FileNotFoundError(f"{STOCK_FILE} not found. Run build_dataset.py first.")

    df = pd.read_parquet( STOCK_FILE )

    if REMOVE_ETFS:
        df = df[df["quote_type"] == "EQUITY"].copy()
    
    return df


def select_feature_columns( df: pd.DataFrame ) -> list:
    missing = set(FEATURES) - set(df.columns)

    if missing:
        print("Warning: requested features missing from dataset:")
        for col in sorted(missing):
            print(f"  {col}")

    feature_cols = [
        col for col in FEATURES
        if col in df.columns
    ]

    return feature_cols


def prepare_features( df: pd.DataFrame, feature_cols: list ) -> tuple:
    features = df[feature_cols].copy()
    medians = {}

    for col in features.columns:
        median_val = features[col].median()
        medians[col] = median_val
        n_missing = features[col].isna().sum()
        if n_missing > 0:
            print( f"Imputing {n_missing} missing values in '{col}' with median ({median_val:.4f})" )
        features[col] = features[col].fillna( median_val )

    return features, medians


def split_train_test( df: pd.DataFrame ) -> tuple:
    # Same time-based split (with gap) as the clustering script, so results
    # are comparable and the same regime-shift caveat applies.
    dates_sorted = np.sort( df["date"].unique() )
    n_dates = len( dates_sorted )

    test_start_idx = int( n_dates * ( 1 - TEST_FRACTION ) )
    test_start_date = pd.Timestamp( dates_sorted[test_start_idx] )
    train_cutoff_date = test_start_date - pd.Timedelta( days=GAP_DAYS )

    train_df = df[df["date"] < train_cutoff_date].reset_index( drop=True )
    test_df = df[df["date"] >= test_start_date].reset_index( drop=True )

    print( f"Train: {len(train_df):,} rows through {train_cutoff_date}" )
    print( f"Gap:   excluded rows between {train_cutoff_date} and {test_start_date}" )
    print( f"Test:  {len(test_df):,} rows from {test_start_date}\n" )

    return train_df, test_df

def main():
    ensure_log_directory()
    log_file = open( f"{LOG_FOLDER}/regression_log.txt", "w" )
    sys.stdout = Tee( sys.__stdout__, log_file )

    print( "Loading feature dataset..." )
    df = load_data()
    print( f"Load {len(df):,} rows, {df['ticker'].nunique()} tickers\n" )

    before = len( df )
    df = df.dropna( subset=TARGET_COL ).reset_index( drop=True )
    dropped = before - len( df )
    print( f"Dropped {dropped} rows missing future returns\n" )

    print( f"Target: predicting {TARGET_COL}" )
    print( f"Mean future excess return: {df[TARGET_COL].mean():.2%}\n" )

    train_df, test_df = split_train_test( df )
    print( f"Train mean excess return: {train_df[TARGET_COL].mean():.2%}" )
    print( f"Test mean excess return: {test_df[TARGET_COL].mean():.2%}\n" )

    feature_cols = select_feature_columns( train_df )
    print( f"Using {len(feature_cols)} features:" )
    print( f"  {feature_cols}\n" )

    print( "Preparing training features (imputing with train medians)..." )
    train_features, medians = prepare_features( train_df, feature_cols )

    print( "Preparing test features (imputing with TRAIN medians, not test's own)..." )
    test_features = test_df.reindex( columns=feature_cols, fill_value=0 ).copy()
    for col in feature_cols:
        n_missing = test_features[col].isna().sum()
        if n_missing > 0:
            print( f"Imputing {n_missing} missing values in '{col}' with TRAIN median ({medians[col]:.4f})" )
        test_features[col] = test_features[col].fillna( medians[col] )

    y_train = train_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    # --- Primary model: Random Forest Regression ---
    print("\nFitting Random Forest Regressor...")

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    rf.fit(
        train_features,
        y_train
    )

    pred = rf.predict( test_features )

    print( "\n" + "="*60 )
    print( "Random Forest Regression — TEST performance" )
    print( "="*60 )

    print( f"MAE: {mean_absolute_error(y_test, pred):.4f}" )

    print( f"RMSE: {mean_squared_error(y_test, pred)**0.5:.4f}" )

    print( f"R²: {r2_score(y_test, pred):.4f}" )

    corr = np.corrcoef(pred, y_test)[0,1]

    print( f"Prediction correlation: {corr:.4f}" )

    # --- Ranking evaluation ---
    scored = test_df.copy()
    scored["predicted_excess_5y"] = pred

    scored.to_parquet(
        f"{LOG_FOLDER}/daily_ranked_predictions_gen{GEN}.parquet"
    )

    ranked = (
        scored
        .groupby("ticker")
        .agg({
            "predicted_excess_5y": "mean",
            "future_excess_5y": "mean",
            "sector": "first"
        })
        .sort_values(
            "predicted_excess_5y",
            ascending=False
        )
    )

    print("\nTop 100 predicted stocks:")
    print(ranked.head(100))

    ranked.to_parquet(
        f"{LOG_FOLDER}/ticker_rankings_gen{GEN}.parquet"
    )

    print("\nRanking performance:")

    for pct in [0.01, 0.05, 0.10, 0.25]:

        n = int(len(scored) * pct)

        top = scored.iloc[:n]

        print(f"\nTop {pct:.0%} ({n:,} stocks)")
        print(
            f"Predicted excess return: "
            f"{top['predicted_excess_5y'].mean():.2%}"
        )
        print(
            f"Actual excess return: "
            f"{top['future_excess_5y'].mean():.2%}"
        )
        print(
            f"Beat market rate: "
            f"{(top['future_excess_5y'] > 0).mean():.2%}"
        )

    scored.to_parquet(
        f"{LOG_FOLDER}/ranked_predictions_gen{GEN}.parquet"
    )


    importance_table = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_
    }).sort_values(
        "importance",
        ascending=False
    )

    print("\nRandom Forest feature importances:")
    print(importance_table.to_string(index=False))

    importance_table.to_csv(
        f"{LOG_FOLDER}/rf_importances_gen{GEN}.csv",
        index=False
    )


    test_df["prediction"] = pred

    test_df.to_parquet( f"{LOG_FOLDER}/test_scored_gen{GEN}.parquet" )

    # Metrics
    metrics = {
        "mae": mean_absolute_error(y_test, pred),
        "rmse": mean_squared_error(y_test,pred)**0.5,
        "r2": r2_score(y_test,pred),
        "correlation": corr
    }

    pd.DataFrame([metrics]).to_csv(
        f"{LOG_FOLDER}/regression_metrics_gen{GEN}.csv",
        index=False
    )

    print( "\n" + "=" * 60 )
    print( "REGRESSION RUN COMPLETE" )
    print( "=" * 60 )


if __name__ == "__main__":
    main()