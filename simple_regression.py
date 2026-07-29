import os
import sys
import io
from pathlib import Path
from typing import Optional

from classify_features import SIMPLE_FEATURES as FEATURES

import pandas as pd
import numpy as np
import joblib
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Files
STOCK_FILE = "data/data.parquet"
LOG_FOLDER = "logs/simple_regression"
MODEL_FOLDER = "models"

# Columns
TARGET_COL = "future_excess_1y"   # beats-market label is derived from this

RANDOM_STATE = 42
TEST_FRACTION = 0.2
GAP_DAYS = 252

TOP_DECILE = 0.10   # fraction of highest-confidence test predictions to profile

REMOVE_ETFS = True

class Tee:
    def __init__( self, *streams ):
        self.streams = streams

    def write( self, data ):
        for s in self.streams:
            try:
                s.write( data )
            except ( OSError, ValueError, io.UnsupportedOperation ):
                pass

    def flush( self ):
        for s in self.streams:
            try:
                s.flush()
            except (OSError, ValueError, io.UnsupportedOperation):
                pass


def ensure_log_directory():
    Path( LOG_FOLDER ).mkdir( parents=True, exist_ok=True )

def ensure_model_directory():
    Path( MODEL_FOLDER ).mkdir( parents=True, exist_ok=True )

def save_model( name: str, estimator, use_scaled: bool ) -> None:
    # Save a fitted model for later use
    ensure_model_directory()
    joblib.dump( estimator, f"{MODEL_FOLDER}/{clean_name( name )}.joblib" )
    print( f"  Saved model to {MODEL_FOLDER}/{clean_name( name )}.joblib (use_scaled={use_scaled})")

def setup_logging() -> None:
    # Split stdout to both the console and the run's log file
    ensure_log_directory()
    log_file = open( f"{LOG_FOLDER}/simple_classification_log.txt" )
    sys.stdout = Tee( sys.__stdout__, log_file )

def load_data() -> pd.DataFrame:
    # Loads dataset
    if not os.path.isfile( STOCK_FILE ):
        raise FileNotFoundError( f"{STOCK_FILE} not found. Run build_dataset.py first." )

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

    return [col for col in FEATURES if col in df.columns]


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

# def preapare_test_features( test_df: pd.DataFrame, feature_cols: list, medians: dict ) -> pd.DataFrame:
#     test_features = test_df.reindex( columns=feature_cols, fill_value=0 ).copy()
#     for col in feature_cols:
#         n_missing = test_features[col].isna().sum()
#         if n_missing > 0:
#             print( f"Imputing {n_missing} missing values in '{col}' with TRAIN median ({medians[col]:.4f})" )
#         test_features[col] = test_features[col].fillna( medians[col] )
#     return test_features

def split_train_test( df: pd.DataFrame ) -> tuple:
    # Same time-based split (with gap) as the clustering script, so results are comparable and the same regime-shift caveat applies.
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

def evaluate_model( name: str, y_true: np.ndarray, y_pred: np.ndarray ) -> dict:
    #Regression metrics: Error magnitude plus rank correlation
    rmse = np.sqrt( mean_squared_error( y_true, y_pred ) )
    mae = mean_absolute_error( y_true, y_pred )
    r2 = r2_score( y_true, y_pred )
    spearman_corr, _ = spearmanr( y_true, y_pred )

    print( f"\n{'='*60}\n{name} — TEST performance\n{'='*60}" )
    print( f"RMSE:              {rmse:.4f}" )
    print( f"MAE:               {mae:.4f}" )
    print( f"R²:                {r2:.4f}" )
    print( f"Spearman rank corr: {spearman_corr:.4f}" )

    return {
        "model": name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "spearman_corr": spearman_corr,
    }

def profile_top_decile( name: str, test_df: pd.DataFrame, y_pred: np.ndarray, top_frac: float = TOP_DECILE ) -> pd.DataFrame:
    # For the stocks ranked highest, what did they actually return?
    scored = test_df.copy()
    scored["pred_excess"] = y_pred
    scored = scored.sort_values( "pred_excess", ascending=False ).reset_index( drop=True )

    n_top = max( 1, int( len( scored ) * top_frac ) )
    top = scored.iloc[:n_top]
    rest = scored.iloc[n_top:]

    print( f"\n{name} — top {top_frac:.0%} highest ranked test predictions (n={n_top}):" )
    print( f"  Mean actual {TARGET_COL}: {top[TARGET_COL].mean():.2%}" )
    print( f"  Median actual {TARGET_COL}: {top[TARGET_COL].median():.2%}" )
    print( f"  (vs. rest of test — mean: {rest[TARGET_COL].mean():.2%},\n"
           f"                    median: {rest[TARGET_COL].median():.2%})" )

    return top

def clean_name( name: str ) -> str:
    # cleans name of model for file/column use
    return name.lower().replace( " ", "_" )

def get_feature_ranking( model, feature_cols: list ) -> Optional[pd.DataFrame]:
    # Returns a feature ranking table for models that expose coefficients or importances
    if hasattr( model, "coef_" ):
        return pd.DataFrame( {"features": feature_cols, "coefficent": model.coef_[0]} ).sort_values( "coefficent", ascending=False )

    if hasattr( model, "feature_importances_" ):
        return pd.DataFrame( {"features": feature_cols, "importance": model.feature_importances_} ).sort_values( "importance", ascending=False )

    return None

def run_model(
        name: str,
        estimator,
        X_train,
        X_test,
        y_train: np.ndarray,
        y_test: np.ndarray,
        test_df: pd.DataFrame,
        feature_cols: list,
        use_scaled: bool
) -> dict:
    # Fits, evaluated, profiles, and saves one model
    print( f"\nFitting {name}..." )
    estimator.fit( X_train, y_train )
    save_model( name, estimator, use_scaled )

    pred = estimator.predict( X_test )

    result = evaluate_model( name, y_test, pred )

    ci = bootstrap_ci( y_test, pred )
    print(f"\n{name} — 95% bootstrap confidence intervals:")
    print(f"  RMSE:      [{ci['rmse_ci'][0]:.4f}, {ci['rmse_ci'][1]:.4f}]")
    print(f"  MAE:       [{ci['mae_ci'][0]:.4f}, {ci['mae_ci'][1]:.4f}]")
    print(f"  R²:        [{ci['r2_ci'][0]:.4f}, {ci['r2_ci'][1]:.4f}]")
    print(f"  Spearman:  [{ci['spearman_ci'][0]:.4f}, {ci['spearman_ci'][1]:.4f}]")

    profile_top_decile( name, test_df, pred )
    print_hit_rate( name, test_df, pred )
    print_top_stocks_by_freq( name, test_df, pred )

    ranking = get_feature_ranking( estimator, feature_cols )

    if ranking is not None:
        ranking_kind = "coefficient" if "coefficient" in ranking.columns else "importance"
        print( f"\n{name} feature {ranking_kind}s:" )
        print( ranking.to_string( index=False ) )
        ranking.to_csv( f"{LOG_FOLDER}/{clean_name( name )}_{ranking_kind}s.csv" )

    return {"result": result, "pred": pred}


def print_top_stocks_by_freq( name: str, test_df: pd.DataFrame, y_pred: np.ndarray, n: int = 10, min_appearances: int = 10 ) -> None:
    # Print the n tickers most often rated in the top decile of predictide probabilities

    scored = test_df.copy()
    scored["pred_excess"] = y_pred

    ticker_stats = scored.groupby( "ticker" ).agg(
        appearances=( "pred_excess", "size" ),
        avg_pred_excess=( "pred_excess", "mean"),
        median_pred_excess=( "pred_excess", "median"),
        avg_actual_excess=( TARGET_COL, "mean")
    )

    eligable = ticker_stats[ticker_stats["appearances"] >= min_appearances]
    top = eligable.sort_values( ["median_pred_excess"], ascending=False ).head(n)

    print( f"\n{name} - top {n} stocks by median predicted excess return "
           f"(min {min_appearances} appearances)")

    print( top.to_string() )

def print_summary( results_df ) -> None:
    print(f"\n{'='*60}\nMODEL COMPARISON\n{'='*60}")
    print(results_df.to_string(index=False))
    results_df.to_csv(f"{LOG_FOLDER}/model_comparison.csv", index=False)

    print("\n" + "=" * 60)
    print("REGRESSION RUN COMPLETE")
    print("=" * 60)


def bootstrap_ci( y_true: np.ndarray, y_pred: np.ndarray, n_boot:int = 1000, seed: int = RANDOM_STATE ) -> dict:
    # Bootstrap confidence intervals for metrics

    rng = np.random.default_rng( seed )
    n = len( y_true )

    rmses, maes, r2s, corrs = [], [], [], []

    for _ in range( n_boot ):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]

        rmses.append( np.sqrt( mean_squared_error(yt, yp) ) )
        maes.append( mean_absolute_error(yt, yp) )
        r2s.append ( r2_score( yt, yp ) )
        corr, _ = spearmanr( yt, yp )
        if not np.isnan( corr ):
            corrs.append( corr )

    def ci( values ):
        return np.percentile( values, [2.5, 97.5] )

    return {
        "rmse_ci": ci( rmses ),
        "mae_ci": ci( maes ),
        "r2_ci": ci( r2s ),
        "spearman_ci": ci( corrs ),
    }


def bootstrap_hit_rate( values: np.ndarray, n_boot: int = 1000, seed: int = RANDOM_STATE ) -> np.ndarray:
    # Bootstrap CI for hit rate
    rng = np.random.default_rng( seed )
    n = len( values )
    rates = []

    for _ in range( n_boot ):
        idx = rng.integers( 0, n, n )
        sample = values[idx]
        rates.append( (sample >= 0 ).mean() )

    return np.percentile( rates, [2.5, 97.5] )

def print_hit_rate(name: str, test_df: pd.DataFrame, y_pred: np.ndarray, top_frac: float = TOP_DECILE) -> None:
    # Prints fraction of stocks that outperformed compared to baseline
    scored = test_df.copy()
    scored["pred_excess"] = y_pred
    scored = scored.sort_values( "pred_excess", ascending=False ).reset_index( drop=True )

    n_top = max( 1, int( len( scored ) * top_frac ))
    top = scored.iloc[:n_top]
    rest = scored.iloc[n_top:]

    top_hit_rate = ( top[TARGET_COL] >= 0 ).mean()
    rest_hit_rate = ( rest[TARGET_COL] >= 0 ).mean()
    overall_hit_rate = ( scored[TARGET_COL] >= 0 ).mean()

    top_ci = bootstrap_hit_rate( top[TARGET_COL].values )
    rest_ci = bootstrap_hit_rate( rest[TARGET_COL].values )
    overall_ci = bootstrap_hit_rate( scored[TARGET_COL].values )

    print( f"\n{name} — hit rate (beat or tied market), with 95% bootstrap CI:" )
    print( f"  Top {top_frac:.0%} picks:  {top_hit_rate:.2%}  [{top_ci[0]:.2%}, {top_ci[1]:.2%}]" )
    print( f"  Rest of test:      {rest_hit_rate:.2%}  [{rest_ci[0]:.2%}, {rest_ci[1]:.2%}]" )
    print( f"  Overall baseline:  {overall_hit_rate:.2%}  [{overall_ci[0]:.2%}, {overall_ci[1]:.2%}]" )


def main():
    # setup_logging()

    df = load_data()
    print( f"Load {len(df):,} rows, {df['ticker'].nunique()} tickers\n" )

    before = len( df )
    df = df.dropna( subset=TARGET_COL ).reset_index( drop=True )
    dropped = before - len( df )
    print( f"Dropped {dropped} rows missing future returns\n" )

    print(f"Overall mean {TARGET_COL}: {df[TARGET_COL].mean():.2%}")
    print(f"Overall median {TARGET_COL}: {df[TARGET_COL].median():.2%}\n")

    train_df, test_df = split_train_test( df )

    feature_cols = select_feature_columns( train_df )
    print( f"Using {len(feature_cols)} features:" )
    print( f"  {feature_cols}\n" )

    print( "Preparing training features (imputing with train medians)..." )
    train_features, medians = prepare_features( train_df, feature_cols )

    print( "Preparing test features (imputing with TRAIN medians, not test's own)..." )
    test_features = test_df.reindex( columns=feature_cols, fill_value=0 ).copy()

    print( "\nScaling features (scaler fit on train only)..." )
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform( train_features )
    test_scaled = scaler.transform( test_features )

    ensure_model_directory()
    joblib.dump( scaler, f"{MODEL_FOLDER}/scaler.joblib" )

    y_train = train_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    model_specs = [
        (
            "Linear Regression",
            LinearRegression(),
            True,
        ),
        (
            "Random Forest",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=50,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
        ),
        (
            "Gradient Boosting",
            HistGradientBoostingRegressor(
                max_iter=300,
                max_depth=8,
                min_samples_leaf=50,
                random_state=RANDOM_STATE,
            ),
            False,
        ),
        (
            "Neural Network",
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=1e-3,
                early_stopping=True,
                n_iter_no_change=10,
                max_iter=200,
                random_state=RANDOM_STATE,
            ),
            True,
        ),
    ]

    results = []

    for name, estimator, use_scaled in model_specs:
        X_train = train_scaled if use_scaled else train_features
        X_test = test_scaled if use_scaled else test_features

        outcome = run_model( name, estimator, X_train, X_test, y_train, y_test, test_df, feature_cols, use_scaled )
        results.append( outcome["result"] )

    results_df = pd.DataFrame(results)

    print_summary( results_df )

if __name__ == "__main__":
    main()