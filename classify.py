import os
import sys
import io
import argparse
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score
)

FEATURES = [
    "ret_1d",
    "ret_1w",
    "ret_1m",
    "ret_6m",
    "ret_1y",
    "ret_3y",
    "ret_5y",
]

# Files
STOCK_FILE = "data/data.parquet"
LOG_FOLDER = "logs/classification"
MODEL_FOLDER = f"{LOG_FOLDER}/models"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--horizon",
    default=None,
    help="Run only this horizon (ex: '5y'). If omitted, run all horizons."
)

DEFAULT_HORIZONS = [
        "1d",
        "1w",
        "1m",
        "6m",
        "1y",
        "3y",
        "5y"
    ]

# Parser for single horizons (Slurm)
args = parser.parse_args()

if args.horizon is not None:
    if args.horizon not in DEFAULT_HORIZONS:
        parser.error( f"Invalid horizon '{args.horizon}" 
                      f"Choose from: {DEFAULT_HORIZONS}"
        )

    HORIZONS = [args.horizon]
else:
    HORIZONS = DEFAULT_HORIZONS

RANDOM_STATE = 42
TEST_FRACTION = 0.2
GAP_DAYS = 252

TOP_DECILE = 0.05   # Percent of highest-confidence test predictions to profile
TOP_N = 20          # Number of top stocks to save

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


def prepare_features( df: pd.DataFrame, feature_cols: list ) -> pd.DataFrame:
    features = df[feature_cols].copy()

    n_missing = features.isna().sum()

    if n_missing.any():
        raise ValueError( f"Missing feature values found (no imputation performed)" )

    return features

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

def add_label( df: pd.DataFrame, label_source_col:str , label_col: str ) -> pd.DataFrame:
    df = df.copy()
    df[label_col] = ( df[label_source_col] > 0 ).astype( int )
    return df

def evaluate_model(
        horizon: str,
        name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> dict:
    acc = accuracy_score( y_true, y_pred )
    prec = precision_score( y_true, y_pred, zero_division=0 )
    rec = recall_score( y_true, y_pred, zero_division=0 )
    auc = roc_auc_score( y_true, y_proba )
    cm = confusion_matrix( y_true, y_pred )
    f1 = f1_score( y_true, y_pred, zero_division=0 )

    print( f"\n{'='*60}\n{horizon} - {name} — TEST performance\n{'='*60}" )
    print( f"Predicted class distribution: {np.bincount(y_pred)}" )
    print( f"Accuracy:  {acc:.4f}" )
    print( f"Precision: {prec:.4f}" )
    print( f"Recall:    {rec:.4f}" )
    print( f"ROC-AUC:   {auc:.4f}" )
    print( f"F1-score:  {f1:.4f}")
    print( "\nConfusion matrix (rows=actual, cols=predicted):" )
    print( pd.DataFrame(
        cm,
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"]
    ) )
    print( "\nFull classification report:" )
    print( classification_report( y_true, y_pred, target_names=["underperforms", "beats_market"], zero_division=0 ) )

    return {
        "horizon": horizon,
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
    }


def profile_top_decile( name: str, test_df: pd.DataFrame, y_proba: np.ndarray, label_source_col: str, label_col: str, top_frac: float = TOP_DECILE ) -> pd.DataFrame:
    # For the stocks the model was MOST confident would beat the market
    scored = test_df.copy()
    scored["pred_proba"] = y_proba
    scored = scored.sort_values( "pred_proba", ascending=False ).reset_index( drop=True )

    n_top = max( 1, int( len( scored ) * top_frac ) )
    top = scored.iloc[:n_top]
    rest = scored.iloc[n_top:]

    print( f"\n{name} — top {top_frac:.0%} most-confident test predictions (n={n_top}):" )
    print( f"  Mean  {label_source_col}: {top[label_source_col].mean():.2%}" )
    print( f"  Median {label_source_col}: {top[label_source_col].median():.2%}" )
    print( f"  Actual beat-market rate: {top[label_col].mean():.2%}" )
    print( f"  (vs. rest of test — mean: {rest[label_source_col].mean():.2%}, "
           f"beat-market rate: {rest[label_col].mean():.2%})" )

    return top


def save_stock_selections(
    horizon: str,
    name: str,
    test_df: pd.DataFrame,
    pred: np.ndarray,
    proba: np.ndarray,
    label_source_col: str,
    top_n: int = TOP_N,
    top_frac: float = TOP_DECILE
) -> None:
    # Saves stock selection information

    scored = test_df.copy()
    scored["pred_proba"] = proba
    scored["predicted_label"] = pred
    scored = scored.sort_values( "pred_proba", ascending=False ).reset_index( drop=True)

    cols = [c for c in ["ticker", "date", "pred_proba", "predicted_label", label_source_col] if c in scored.columns]

    base_filename = f"{LOG_FOLDER}/stocks/{horizon}_{clean_name( name )}"

    n_top = min( top_n, len( scored ) )
    top_df = scored.iloc[:n_top]
    top_df[cols].to_csv( f"{base_filename}_top{top_n}.csv", index=False )

    n_decile = max( 1, int( len(scored) * top_frac ) )
    decile_df = scored.iloc[:n_decile]
    decile_df[cols].to_csv( f"{base_filename}_top{top_frac:.0%}.csv", index=False)

    winners_df = scored[scored["predicted_label"] == 1]
    winners_df[cols].to_csv( f"{base_filename}_all.csv", index=False )

    print( f"\nSaved selections for {name} @ {horizon}: "
           f"top {len( top_df )}, top {top_frac:.0%} ({len( decile_df )}), "
           f"all predicted winners ({len( winners_df )})" )


def clean_name( name: str ) -> str:
    # cleans name of model for file/column use
    return name.lower().replace( " ", "_" )

def get_feature_ranking( model, feature_cols: list ) -> Optional[pd.DataFrame]:
    # Returns a feature ranking table for models that expose coefficients or importances
    if hasattr( model, "coef_" ):
        return pd.DataFrame( {"features": feature_cols, "coefficient": model.coef_[0]} ).sort_values( "coefficient", ascending=False )

    if hasattr( model, "feature_importances_" ):
        return pd.DataFrame( {"features": feature_cols, "importance": model.feature_importances_} ).sort_values( "importance", ascending=False )

    return None

def run_model(
        horizon: str,
        name: str,
        estimator,
        X_train,
        X_test,
        y_train: np.ndarray,
        y_test: np.ndarray,
        test_df: pd.DataFrame,
        feature_cols: list,
        label_source_col,
        label_col,
        scaler,
        use_scaled: bool
) -> dict:
    # Fits, evaluated, profiles, and saves one model
    print( f"\nFitting {name}..." )
    estimator.fit( X_train, y_train )

    save_model( horizon, name, estimator, scaler, use_scaled )

    pred = estimator.predict( X_test )
    proba = estimator.predict_proba( X_test )[:, 1]

    result = evaluate_model( horizon, name, y_test, pred, proba )

    ci = bootstrap_ci( y_test, pred, proba )
    print(f"\n{horizon} - {name} — 95% bootstrap confidence intervals:")
    print(f"  Accuracy:  [{ci['accuracy_ci'][0]:.4f}, {ci['accuracy_ci'][1]:.4f}]")
    print(f"  Precision: [{ci['precision_ci'][0]:.4f}, {ci['precision_ci'][1]:.4f}]")
    print(f"  Recall:    [{ci['recall_ci'][0]:.4f}, {ci['recall_ci'][1]:.4f}]")
    print(f"  F1 Score:  [{ci['f1_ci'][0]:.4f}, {ci['f1_ci'][1]:.4f}]")
    print(f"  ROC-AUC:   [{ci['auc_ci'][0]:.4f}, {ci['auc_ci'][1]:.4f}]")

    profile_top_decile( name, test_df, proba, label_source_col, label_col )
    save_stock_selections( horizon, name, test_df, pred, proba, label_source_col )

    ranking = get_feature_ranking( estimator, feature_cols )

    if ranking is not None:
        ranking_kind = "coefficient" if "coefficient" in ranking.columns else "importance"
        print( f"\n{name} feature {ranking_kind}s:" )
        print( ranking.to_string( index=False ) )
        ranking.to_csv( f"{LOG_FOLDER}/{horizon}_{clean_name( name )}_{ranking_kind}s.csv" )

    return {"result": result, "pred": pred, "proba": proba }

# def print_top_predictions( name: str, test_df: pd.DataFrame, proba: np.ndarray, label_source_col: str, horizon: str, n: int = 10 ) -> None:
#     # Print the top n stocks the model is most confident will beat the market
#     scored = test_df.copy()
#     scored["pred_proba"] = proba
#     scored = scored.sort_values( "pred_proba", ascending=False )
#     scored = scored.drop_duplicates(subset="ticker", keep="first" ).reset_index( drop=True )

#     top = scored.head( n )
#     print( f"\n{horizon} - {name} - top {n} predicted stocks:" )
#     print( top[["ticker", "date", "pred_proba", label_source_col]].to_string( index=False ) )

def print_top_stocks_by_freq(
        name: str,
        test_df: pd.DataFrame,
        proba: np.ndarray,
        label_source_col: str,
        top_frac: float = TOP_DECILE,
        n: int = 10,
        min_appearances: int = 10 ) -> None:
    # Print the n tickers most often rated in the top decile of predicted probabilities

    scored = test_df.copy()
    scored["pred_proba"] = proba

    threshold = scored["pred_proba"].quantile( 1 - top_frac )
    scored["rated_highly"] = scored["pred_proba"] >= threshold
    ticker_stats = scored.groupby( "ticker" ).agg(
        appearances=("rated_highly", "size" ),
        times_rated_highly=( "rated_highly", "sum" ),
        mean_pred_proba=( "pred_proba", "mean"),
        median_pred_proba=( "pred_proba", "median"),
        avg_future_excess=( label_source_col, "mean")
    )

    ticker_stats["highly_rated_rate"] = ticker_stats["times_rated_highly"] / ticker_stats["appearances"]

    eligeble = ticker_stats[ticker_stats["appearances"] >= min_appearances]
    top = eligeble.sort_values( ["median_pred_proba"], ascending=False ).head(n)

    print( f"\n{name} - top {n} stocks by frequency rated in top {top_frac:.0%} "
           f"(min {min_appearances} appearances)")

    print( top.to_string() )

# def print_summary( results, scored_df ) -> None:
#     results_df = pd.DataFrame(results)
#     print(f"\n{'='*60}\nMODEL COMPARISON\n{'='*60}")
#     print(results_df.to_string(index=False))

#     results_df.to_csv(f"{LOG_FOLDER}/model_comparison.csv", index=False)
#     scored_df.to_parquet(f"{LOG_FOLDER}/test_scored.parquet", index=False)

#     print("\n" + "=" * 60)
#     print("CLASSIFICATION RUN COMPLETE")
#     print("=" * 60)

def bootstrap_ci( y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, n_boot:int = 1000, seed: int = RANDOM_STATE ) -> dict:
    # Bootstrap confidence intervals for accuary, precision, recall, f1 score, and AUC

    rng = np.random.default_rng( seed )
    n = len( y_true )

    accs, precs, recs, f1, aucs = [], [], [], [], []

    for _ in range( n_boot ):
        idx = rng.integers( 0, n, n ) # sample n indices with replacement
        yt, yp, ypr = y_true[idx], y_pred[idx], y_proba[idx]

        accs.append( accuracy_score( yt, yp ) )
        precs.append( precision_score( yt, yp, zero_division=0 ) )
        recs.append( recall_score( yt, yp, zero_division=0 ) )
        f1.append( f1_score( yt, yp, zero_division=0 ))

        if len( np.unique( yt ) ) > 1:
            aucs.append( roc_auc_score( yt, ypr) )

    def ci( values ):
        return np.percentile( values, [2.5, 97.5] )

    return {
        "accuracy_ci": ci( accs ),
        "precision_ci": ci( precs ),
        "recall_ci": ci( recs ),
        "f1_ci": ci( f1 ),
        "auc_ci": ci( aucs )
    }


def save_model( horizon: str, name: str, estimator, scaler, use_scaled: bool ) -> None:
    Path( MODEL_FOLDER ).mkdir( parents=True, exist_ok=True )

    pipeline = Pipeline( [
        ( "scaler", scaler if use_scaled else "passthrough" ),
        ( "clf", estimator )
    ])

    path = f"{MODEL_FOLDER}/{horizon}_{clean_name( name )}.joblib"
    joblib.dump( pipeline, path )
    print( f"Saved model pipeline: {path}")


def main():
    # setup_logging()

    df_all = load_data()
    print( f"Load {len( df_all ):,} rows, {df_all['ticker'].nunique()} tickers\n" )

    all_results = []

    model_specs = [
        (
            "Logistic Regression",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight="balanced"
            ),
            True
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=50,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1
            ),
            False
        ),
        (
            "Gradient Boosting",
            HistGradientBoostingClassifier(
                max_iter=300,
                max_depth=8,
                min_samples_leaf=50,
                random_state=RANDOM_STATE,
                class_weight="balanced"
            ),
            False,
        ),
        (
            "MLPClassifier",
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=1e-3,
                early_stopping=True,
                n_iter_no_change=10,
                max_iter=200,
                random_state=RANDOM_STATE,
            ),
            True,
        )
    ]

    for horizon in HORIZONS:
        print( "\n" + "=" * 80 )
        print( f"Running Horizon: {horizon}")
        print( "\n" + "=" * 80 )

        label_source_col = f"future_excess_{horizon}"
        label_col = f"beats_market_{horizon}"

        df = df_all.copy()

        before = len( df )
        df = df.dropna( subset=label_source_col ).reset_index( drop=True )
        dropped = before - len( df )
        print( f"Dropped {dropped} rows missing future returns\n" )

        df = add_label( df, label_source_col, label_col )
        print( f"Label '{label_col}' derived from {label_source_col} > 0" )
        print( f"Overall beat-market rate: {df[label_col].mean():.2%}\n" )

        train_df, test_df = split_train_test( df )
        print( f"Train beat-market rate: {train_df[label_col].mean():.2%}" )
        print( f"Test  beat-market rate: {test_df[label_col].mean():.2%}\n" )

        feature_cols = select_feature_columns( train_df )

        train_features = prepare_features( train_df, feature_cols )
        test_features = prepare_features( test_df, feature_cols )

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform( train_features )
        test_scaled = scaler.transform( test_features )

        y_train = train_df[label_col].values
        y_test = test_df[label_col].values

        horizon_results = []

        for name, estimator, use_scaled in model_specs:
            X_train = train_scaled if use_scaled else train_features
            X_test = test_scaled if use_scaled else test_features

            outcome = run_model( horizon, name, estimator, X_train, X_test, y_train, y_test, test_df, feature_cols, label_source_col, label_col, scaler, use_scaled )

            horizon_results.append( outcome["result"] )

        results_df = pd.DataFrame( all_results )
        results_df.to_csv( f"{LOG_FOLDER}/{horizon}_model_comparison.csv", index=False )

    print( "\n" + "=" * 80 )
    print( "FINAL HORIZON COMPARISON" )
    print( "=" * 80 )   


if __name__ == "__main__":
    main()