import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import sys
from classify_features import FEATURES

# Files
STOCK_FILE = "data/data.parquet"
LOG_FOLDER = "logs/classification"

# Columns
LABEL_SOURCE_COL = "future_excess_5y"   # beats-market label is derived from this
LABEL_COL = "beats_market_5y"

RANDOM_STATE = 42
TEST_FRACTION = 0.2
GAP_DAYS = 252

TOP_DECILE = 0.10   # fraction of highest-confidence test predictions to profile

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


def add_label( df: pd.DataFrame ) -> pd.DataFrame:
    df = df.copy()
    df[LABEL_COL] = ( df[LABEL_SOURCE_COL] > 0 ).astype( int )
    return df


def evaluate_model( name: str, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray ) -> dict:
    acc = accuracy_score( y_true, y_pred )
    prec = precision_score( y_true, y_pred )
    rec = recall_score( y_true, y_pred )
    auc = roc_auc_score( y_true, y_proba )
    cm = confusion_matrix( y_true, y_pred )

    print( f"\n{'='*60}\n{name} — TEST performance\n{'='*60}" )
    print( f"Accuracy:  {acc:.4f}" )
    print( f"Precision: {prec:.4f}" )
    print( f"Recall:    {rec:.4f}" )
    print( f"ROC-AUC:   {auc:.4f}" )
    print( "\nConfusion matrix (rows=actual, cols=predicted):" )
    print( pd.DataFrame(
        cm,
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"]
    ) )
    print( "\nFull classification report:" )
    print( classification_report( y_true, y_pred, target_names=["underperforms", "beats_market"] ) )

    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "roc_auc": auc,
    }


def profile_top_decile( name: str, test_df: pd.DataFrame, y_proba: np.ndarray, top_frac: float = TOP_DECILE ) -> pd.DataFrame:
    # For the stocks the model was MOST confident would beat the market,
    # what did they actually do? This ties the classifier's output back to
    # the original "find good stocks" goal, beyond a bare accuracy number.
    scored = test_df.copy()
    scored["pred_proba"] = y_proba
    scored = scored.sort_values( "pred_proba", ascending=False ).reset_index( drop=True )

    n_top = max( 1, int( len( scored ) * top_frac ) )
    top = scored.iloc[:n_top]
    rest = scored.iloc[n_top:]

    print( f"\n{name} — top {top_frac:.0%} most-confident test predictions (n={n_top}):" )
    print( f"  Mean  future_excess_5y: {top['future_excess_5y'].mean():.2%}" )
    print( f"  Median future_excess_5y: {top['future_excess_5y'].median():.2%}" )
    print( f"  Actual beat-market rate: {top[LABEL_COL].mean():.2%}" )
    print( f"  (vs. rest of test — mean: {rest['future_excess_5y'].mean():.2%}, "
           f"beat-market rate: {rest[LABEL_COL].mean():.2%})" )

    return top


def main():
    ensure_log_directory()
    log_file = open( f"{LOG_FOLDER}/classification_log.txt", "w" )
    sys.stdout = Tee( sys.__stdout__, log_file )

    print( "Loading feature dataset..." )
    df = load_data()
    print( f"Load {len(df):,} rows, {df['ticker'].nunique()} tickers\n" )

    before = len( df )
    df = df.dropna( subset=LABEL_SOURCE_COL ).reset_index( drop=True )
    dropped = before - len( df )
    print( f"Dropped {dropped} rows missing future returns\n" )

    df = add_label( df )
    print( f"Label '{LABEL_COL}' derived from {LABEL_SOURCE_COL} > 0" )
    print( f"Overall beat-market rate: {df[LABEL_COL].mean():.2%}\n" )

    train_df, test_df = split_train_test( df )
    print( f"Train beat-market rate: {train_df[LABEL_COL].mean():.2%}" )
    print( f"Test  beat-market rate: {test_df[LABEL_COL].mean():.2%}\n" )

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

    print( "\nScaling features (scaler fit on train only)..." )
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform( train_features )
    test_scaled = scaler.transform( test_features )

    y_train = train_df[LABEL_COL].values
    y_test = test_df[LABEL_COL].values

    results = []

    # --- Primary model: Logistic Regression ---
    print( "\nFitting Logistic Regression..." )
    logreg = LogisticRegression( max_iter=1000, random_state=RANDOM_STATE )
    logreg.fit( train_scaled, y_train )

    logreg_pred = logreg.predict( test_scaled )
    logreg_proba = logreg.predict_proba( test_scaled )[:, 1]
    results.append( evaluate_model( "Logistic Regression", y_test, logreg_pred, logreg_proba ) )
    profile_top_decile( "Logistic Regression", test_df, logreg_proba )

    coef_table = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": logreg.coef_[0]
    }).sort_values( "coefficient", ascending=False )
    print( "\nLogistic Regression coefficients (positive = associated with beating market):" )
    print( coef_table.to_string( index=False ) )
    coef_table.to_csv( f"{LOG_FOLDER}/logreg_coefficients_gen{GEN}.csv", index=False )

    # --- Comparison baseline: Random Forest ---
    print( "\nFitting Random Forest (comparison baseline)..." )
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=50,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )
    rf.fit( train_features, y_train )

    rf_pred = rf.predict( test_features )
    rf_proba = rf.predict_proba( test_features )[:, 1]
    results.append( evaluate_model( "Random Forest", y_test, rf_pred, rf_proba ) )
    profile_top_decile( "Random Forest", test_df, rf_proba )

    importance_table = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_
    }).sort_values( "importance", ascending=False )
    print( "\nRandom Forest feature importances:" )
    print( importance_table.to_string( index=False ) )
    importance_table.to_csv( f"{LOG_FOLDER}/rf_importances_gen{GEN}.csv", index=False )

    # --- Summary ---
    results_df = pd.DataFrame( results )
    print( f"\n{'='*60}\nMODEL COMPARISON\n{'='*60}" )
    print( results_df.to_string( index=False ) )
    results_df.to_csv( f"{LOG_FOLDER}/model_comparison_gen{GEN}.csv", index=False )

    test_df.to_parquet( f"{LOG_FOLDER}/test_scored_gen{GEN}.parquet" )

    print( "\n" + "=" * 60 )
    print( "CLASSIFICATION RUN COMPLETE" )
    print( "=" * 60 )


if __name__ == "__main__":
    main()