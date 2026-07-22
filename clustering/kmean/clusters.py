import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from scipy.stats import trim_mean
import sys

# Files
STOCK_FILE = "data/data.parquet"
LOG_FOLDER = "logs/kmean"

# Columns
TARGET_COLS = [
    # Absolute returns
    "future_ret_1d",
    "future_ret_1w",
    "future_ret_1m",
    "future_ret_6m",
    "future_ret_1y",
    "future_ret_3y",
    "future_ret_5y",

    # Relative returns
    "future_excess_1d",
    "future_excess_1w",
    "future_excess_1m",
    "future_excess_6m",
    "future_excess_1y",
    "future_excess_3y",
    "future_excess_5y"
]
ID_COLS = ["ticker", "date", "sector"]

# Clustering config
K_RANGE = range( 2, 11 )
MIN_CLUSTER_SIZE = 200
RANDOM_STATE = 42

MAX_DEPTH = 5
MIN_ROWS = 5000
TEST_FRACTION = 0.2
GAP_DAYS = 252

JUNK_EXCLUDE = {
    "price",
    "sector_size",
    "rows",
    "sector_size_market",
    "rows_market"
}

SEC_EXCLUDE = {
    "sector_is_trending",

    "sec_avg_ret_1w",
    "sec_avg_ret_1m",
    "sec_avg_ret_6m",
    "sec_avg_ret_1y",
    "sec_avg_ret_3y",
    "sec_avg_ret_5y",

    "sec_avg_ret_1w_market",
    "sec_avg_ret_1m_market",
    "sec_avg_ret_6m_market",
    "sec_avg_ret_1y_market",
    "sec_avg_ret_3y_market",
    "sec_avg_ret_5y_market",

    "sec_positive_1y_trend_pct",
    "sec_positive_5y_trend_pct",
    "sec_breadth_positive_1y",
    "sec_breadth_positive_5y",

    "sec_ret_1y_dispersion",
    "sec_ret_5y_dispersion",
    "sec_ret_1y_dispersion_market",
    "sec_ret_5y_dispersion_market",

    "quality_score",
    "sector_Unknown",
}

OTHER_EXCLUDE = {
    "pe"
}

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

def ensure_data_directory():
    # Create path if it doesn't exist
    Path( "data" ).mkdir( exist_ok=True )

def load_data() -> pd.DataFrame:
    # Load saved data
    if not os.path.isfile( STOCK_FILE ):
        raise FileNotFoundError(f"{STOCK_FILE} not found. Run build_dataset.py first.")
    return pd.read_parquet( STOCK_FILE )

def select_feature_columns( df: pd.DataFrame ) -> list:
    # Create list of columns for ML
    numeric_cols = df.select_dtypes( include=[np.number] ).columns.tolist()
    exclude = set( ID_COLS ) | set( TARGET_COLS ) | JUNK_EXCLUDE | SEC_EXCLUDE | {c for c in df.columns if c.startswith("future_")} | OTHER_EXCLUDE
    return [c for c in numeric_cols if c not in exclude]

def prepare_features( df: pd.DataFrame, feature_cols: list ) -> tuple:
    # Replace any missing cells with median value
    features = df[feature_cols].copy()
    medians = {}

    for col in features.columns:
        median_val = features[col].median()
        medians[col] = median_val
        n_missing = features[col].isna().sum()
        if n_missing > 0:
            print( f"Imputing {n_missing} missing values in '{col}' with median ({median_val:.4f})")
        features[col] = features[col].fillna(median_val)

    return features, medians

def split_train_test( df: pd.DataFrame ) -> tuple:
    # Time based split with a gap

    dates_sorted = np.sort( df["date"].unique() )
    n_dates = len( dates_sorted )

    test_start_idx = int( n_dates * ( 1 - TEST_FRACTION ) )
    test_start_date = pd.Timestamp(dates_sorted[test_start_idx])

    train_cutoff_date = test_start_date - pd.Timedelta(days=GAP_DAYS)

    train_df = df[df["date"] < train_cutoff_date].reset_index( drop=True )
    test_df = df[df["date"] >= test_start_date].reset_index( drop=True )

    print( f"Train: {len(train_df):,} rows through {train_cutoff_date}" )
    print( f"Gap:   excluded rows between {train_cutoff_date} and {test_start_date}" )
    print( f"Test:  {len(test_df):,} rows from {test_start_date}\n" )

    return train_df, test_df

def  fit_and_evaluate( train_df: pd.DataFrame, test_df: pd.DataFrame ) -> tuple:
    train_df = train_df.copy()
    test_df = test_df.copy()

    feature_cols = select_feature_columns( train_df )
    print( f"Using {len(feature_cols)} features for clustering:" )
    print( f"  {feature_cols}\n" )

    print( "Preparing training features (imputing with train medians)..." )
    train_features, medians = prepare_features( train_df, feature_cols )

    print( "Preparing test features (imputing with TRAIN medians, not test's own)..." )
    test_features = test_df.reindex( columns = feature_cols, fill_value=0 ).copy()

    for col in feature_cols:
        n_missing = test_features[col].isna().sum()
        if n_missing > 0:
            print( f"Imputing {n_missing} missing values in '{col}' with TRAIN median ({medians[col]:.4f})" )
        test_features[col] = test_features[col].fillna( medians[col] )

    print( "\nScaling features (scaler fit on train only)..." )
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform( train_features )
    test_scaled = scaler.transform( test_features )

    print( "\nSearching for best k via silhouette score (train only)..." )
    best_k, scores = choose_k( train_scaled, K_RANGE )
    print( f"\nBest k = {best_k} (silhouette={scores[best_k]:.4f})\n" )

    print( f"Fitting final MiniBatchKMeans with k={best_k} on train only..." )
    km = MiniBatchKMeans( n_clusters=best_k, random_state=RANDOM_STATE, n_init=10 )
    train_df["cluster"] = km.fit_predict( train_scaled )

    print( "Assigning test rows using the train-fitted model (no refitting on test)..." )
    test_df["cluster"] = km.predict( test_scaled )

    print( "\nSummarizing TRAIN clusters by future returns...\n" )
    train_summary = summarize_clusters( train_df )

    print( train_summary[[
        "cluster",
        "count",
        "overall_score",
        "future_ret_1y_median",
        "future_ret_5y_median",
        "future_excess_1y_median",
        "future_excess_5y_median"
    ]].to_string( index=False ) )

    reliable = train_summary[train_summary["reliable"]]
    if reliable.empty:
        print( "\nNo reliable cluster found on train — stopping." )
        return train_df, test_df, train_summary, None, None

    best_cluster_id = int( reliable.iloc[0]["cluster"] )
    best_train_row = train_summary[train_summary["cluster"] == best_cluster_id].iloc[0]
    print( f"\nBest TRAIN cluster: #{best_cluster_id} (n={int(best_train_row['count'])}, "
           f"overall score={best_train_row['overall_score']:.2%})" )

    print( "\nSummarizing TEST clusters by future returns (same assignments, unseen data)...\n" )
    test_summary = summarize_clusters( test_df )
    print( test_summary[[
        "cluster",
        "count",
        "overall_score",
        "future_ret_1y_median",
        "future_ret_5y_median",
        "future_excess_1y_median",
        "future_excess_5y_median"
    ]].to_string( index=False ) )

    test_match = test_summary[test_summary["cluster"] == best_cluster_id]
    if test_match.empty:
        print( f"\nCluster #{best_cluster_id} has no test rows — can't validate." )
        return train_df, test_df, train_summary, test_summary, best_cluster_id

    test_row = test_match.iloc[0]
    print( f"\n{'='*60}\nOUT-OF-SAMPLE CHECK\n{'='*60}" )
    print( f"Cluster #{best_cluster_id} was picked as 'best' using TRAIN future returns." )
    print( f"  Train overall_score: {best_train_row['overall_score']:.2%}  (n={int(best_train_row['count'])})" )
    print( f"  Test  overall_score: {test_row['overall_score']:.2%}  (n={int(test_row['count'])})" )
    print( f"  Train 5y median: {best_train_row['future_ret_5y_median']:.2%}   "
           f"Test 5y median: {test_row['future_ret_5y_median']:.2%}" )

    return train_df, test_df, train_summary, test_summary, best_cluster_id

def choose_k( scaled_features: np.ndarray, k_range: range, sample_size: int = 20000 ) -> tuple:
    # 
    rng = np.random.RandomState( RANDOM_STATE )

    if len( scaled_features ) > sample_size:
        idx = rng.choice( len( scaled_features ), sample_size, replace=False )
        sample = scaled_features[idx]
    else:
        sample = scaled_features

    scores = {}

    for k in k_range:
        km = MiniBatchKMeans( n_clusters=k, random_state=RANDOM_STATE, n_init=10 )
        labels = km.fit_predict( sample )
        score = silhouette_score( sample, labels )
        scores[k] = score
        print( f"k={k}: silhouette={score:.4f}" )

    best_k = max( scores, key=scores.get )

    return best_k, scores

def summarize_clusters( df: pd.DataFrame, cluster_col: str = "cluster" ) -> pd.DataFrame:
    #
    overall_means = {
        col: df[col].mean() for col in TARGET_COLS
    }

    rows = []

    for cluster_id, group in df.groupby( cluster_col ):

        target_stats = {}

        for col in TARGET_COLS:
            target = group[col]

            q25, q75 = target.quantile(0.25), target.quantile(0.75)
            mean_val = target.mean()
            median_val = target.median()

            target_stats[f"{col}_mean"] = mean_val
            target_stats[f"{col}_median"] = median_val
            target_stats[f"{col}_trimmed_mean"] = trim_mean( target, proportiontocut=0.10 )
            target_stats[f"{col}_std"] = target.std()
            target_stats[f"{col}_iqr"] = q75 - q25
            target_stats[f"{col}_vs_overall"] = mean_val - overall_means[col]

        # Weighted mean of medians
        target_stats["overall_score"] = (
            target_stats["future_ret_1d_median"] * 0.05 +
            target_stats["future_ret_1w_median"] * 0.05 +
            target_stats["future_ret_1m_median"] * 0.10 +
            target_stats["future_ret_6m_median"] * 0.10 +
            target_stats["future_ret_1y_median"] * 0.20 +
            target_stats["future_ret_3y_median"] * 0.25 +
            target_stats["future_ret_5y_median"] * 0.25
        )

        rows.append({
            "cluster": cluster_id,
            "count": len(group),

            **target_stats,

            

            "avg_beta_1y": group["beta_1y"].mean(),
            "avg_1y_trend": group["1y_trend"].mean(),
            "avg_5y_drawdown": group["5y_drawdown"].mean(),

            "reliable": len(group) >= MIN_CLUSTER_SIZE
        })

    summary = pd.DataFrame( rows ).sort_values( "overall_score", ascending=False ).reset_index( drop=True )

    return summary

def run_clustering( df: pd.DataFrame, depth: int ) -> tuple:
    # Runs one round of clustering

    df = df.copy()

    print( "\n\n" + "=" * 60 )
    print( f"Cluster Depth {depth}" )
    print( "=" * 60 )
    
    feature_cols = select_feature_columns( df )
    if depth == 0:
        print( f"Using {len(feature_cols)} features for clustering:" )
        print( f"  {feature_cols}\n" )

    print( "Preparing features (imputing missing values)..." )
    features = prepare_features(df, feature_cols)

    print( "\nScaling features..." )
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    print("\nSearching for best k via silhouette score...")
    best_k, scores = choose_k(scaled, K_RANGE)
    print(f"\nBest k = {best_k} (silhouette={scores[best_k]:.4f})\n")

    print(f"Fitting final MiniBatchKMeans with k={best_k} on full dataset...")
    km = MiniBatchKMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    df["cluster"] = km.fit_predict(scaled)
 
    print("\nSummarizing clusters by future returns...\n")

    summary = summarize_clusters(df)

    summary_print = summary[
        [
            "cluster",
            "count",
            "overall_score",
            "future_ret_1y_median",
            "future_ret_3y_median",
            "future_ret_5y_median",
            "avg_beta_1y",
            "avg_1y_trend",
            "avg_5y_drawdown"
        ]
    ]

    print(summary_print.to_string(index=False))

    reliable = summary[summary["reliable"]]
    if reliable.empty:
        return df, summary, None
    
    best_cluster_id = int( reliable.iloc[0]["cluster"] )
    return df, summary, best_cluster_id
    
def main():
    log_file = open(f"{LOG_FOLDER}/cluster_log.txt", "w")
    sys.stdout = Tee(sys.__stdout__, log_file)

    print( "Loading feature dataset..." )
    df = load_data()
    print( f"Load {len(df):,} rows, {df['ticker'].nunique()} tickers\n")

    # Clean any rows with no target
    before = len( df )
    df = df.dropna( subset=TARGET_COLS ).reset_index( drop=True )
    dropped = before - len( df )
    
    print( f"Dropped {dropped} rows missing future returns\n")

    train_df, test_df = split_train_test( df )

    train_df, test_df, train_summary, test_summary, best_cluster_id = fit_and_evaluate( train_df, test_df )

    train_df.to_parquet( f"{LOG_FOLDER}/train_clustered_gen{GEN}.parquet" )
    test_df.to_parquet( f"{LOG_FOLDER}/test_clustered_gen{GEN}.parquet" )
    if train_summary is not None:
        train_summary.to_csv( f"{LOG_FOLDER}/train_summary_gen{GEN}.csv", index=False )
    if test_summary is not None:
        test_summary.to_csv( f"{LOG_FOLDER}/test_summary_gen{GEN}.csv", index=False )

    print( "\n" + "=" * 60 )
    print( "WALK-FORWARD VALIDATION COMPLETE" )
    print( "=" * 60 )
        
if __name__ == "__main__":
    main()