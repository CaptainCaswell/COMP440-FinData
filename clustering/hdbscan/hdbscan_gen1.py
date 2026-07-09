import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from scipy.stats import trim_mean
import sys
import hdbscan

# Files
STOCK_FILE = "data/data.parquet"
CLUSTERED_FILE = "data/clustered_data.parquet"
SUMMARY_FILE = "data/cluster_summary.csv"

# Columns
TARGET_COL = "future_5y_return"
ID_COLS = ["ticker", "date", "sector"]

# Clustering config
K_RANGE = range( 2, 11 )
MIN_CLUSTER_SIZE = 200
RANDOM_STATE = 42

MAX_DEPTH = 5
MIN_ROWS = 5000

PRIM_EXCLUDE = {
    "price",
    "sector_size",
    "rows",
    "sector_size_market",
    "rows_market"
}

SEC_EXCLUDE = {
    "excess_ret_1y", "excess_ret_5y",
    "trend_vs_sector_1y", "trend_vs_sector_5y",
    "drawdown_rel_1y", "drawdown_rel_5y",
    "excess_vs_market_1y", "trend_vs_market_1y",
    "risk_adjusted_1y", "risk_adjusted_5y",
    "quality_score",
    "sector_Unknown"
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
    exclude = set( ID_COLS ) | { TARGET_COL } | PRIM_EXCLUDE | SEC_EXCLUDE
    # return [c for c in numeric_cols if c not in exclude]
    return [c for c in numeric_cols if c not in exclude and not c.endswith( "_market" ) and not c.startswith("sec_") and not c.startswith("sector_")]

def prepare_features( df: pd.DataFrame, feature_cols: list ) -> pd.DataFrame:
    # Replace any missing cells with median value
    features = df[feature_cols].copy()

    for col in features.columns:
        median_val = features[col].median()
        n_missing = features[col].isna().sum()
        if n_missing > 0:
            print( f"Imputing {n_missing} missing values in '{col}' with median ({median_val:.4f})")
        features[col] = features[col].fillna(median_val)

    return features

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
    overall_mean = df[TARGET_COL].mean()

    rows = []

    for cluster_id, group in df.groupby( cluster_col ):

        target = group[TARGET_COL]
        q25, q75 = target.quantile(0.25), target.quantile(0.75)
        mean_val = target.mean()
        median_val = target.median()

        rows.append({
            "cluster": cluster_id,
            "count": len(group),
            "mean_future_5y_return": mean_val,
            "median_future_5y_return": median_val,
            "trimmed_mean_future_5y_return": trim_mean(target, proportiontocut=0.10),
            "std_future_5y_return": target.std(),
            "iqr_future_5y_return": q75 - q25,
            "skew_ratio": mean_val / median_val if median_val != 0 else np.nan,
            "vs_overall_mean": mean_val - overall_mean,
            "avg_beta_1y": group["beta_1y"].mean(),
            "avg_1y_trend": group["1y_trend"].mean(),
            "avg_5y_drawdown": group["5y_drawdown"].mean(),
            "reliable": len(group) >= MIN_CLUSTER_SIZE
        })

    summary = pd.DataFrame( rows ).sort_values( "trimmed_mean_future_5y_return", ascending=False ).reset_index( drop=True )
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
 
    print("\nSummarizing clusters by mean future_5y_return...\n")
    summary = summarize_clusters(df)
    print(summary.to_string(index=False))

    reliable = summary[summary["reliable"]]
    if reliable.empty:
        return df, summary, None
    
    best_cluster_id = int( reliable.iloc[0]["cluster"] )
    return df, summary, best_cluster_id

import hdbscan

def run_hdbscan_clustering( df: pd.DataFrame, min_cluster_size: int = 500, min_samples: int = 50 ) -> tuple:
    feature_cols = select_feature_columns( df )
    print(f"Using {len(feature_cols)} features for clustering:")
    print(f"  {feature_cols}\n")

    print("Preparing features (imputing missing values)...")
    features = prepare_features( df, feature_cols )

    print("\nScaling features...")
    scaler = StandardScaler()
    scaled = scaler.fit_transform( features )

    print(f"\nRunning HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples})...")
    clusterer = hdbscan.HDBSCAN( min_cluster_size=min_cluster_size, min_samples=min_samples, core_dist_n_jobs=-1 )
    labels = clusterer.fit_predict( scaled )

    df = df.copy()
    df["cluster"] = labels

    n_noise = (labels == -1).sum()
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"\nFound {n_clusters} clusters, {n_noise} noise points ({n_noise/len(labels):.1%} of data)")

    summary = summarize_clusters( df )
    print("\nCluster summary (cluster -1 = noise/outliers, excluded from 'reliable'):")
    print(summary.to_string(index=False))

    # Noise points are never a valid "best cluster" candidate
    reliable = summary[(summary["reliable"]) & (summary["cluster"] != -1)]
    if reliable.empty:
        return df, summary, None

    best_cluster_id = int( reliable.iloc[0]["cluster"] )
    return df, summary, best_cluster_id

def main():
    log_file = open(f"data/cluster_log_gen{GEN}.txt", "w")
    sys.stdout = Tee(sys.__stdout__, log_file)
    
    print("Loading feature dataset...")
    df = load_data()
    print(f"Load {len(df):,} rows, {df['ticker'].nunique()} tickers\n")

    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    clustered_df, summary, best_cluster_id = run_hdbscan_clustering(df)

    clustered_df.to_parquet( f"data/hdbscan_clustered_data_gen{GEN}.parquet")
    summary.to_csv( f"data/hdbscan_cluster_summary_gen{GEN}.csv", index=False)

    if best_cluster_id is not None:
        best_row = summary[summary["cluster"] == best_cluster_id].iloc[0]
        print(f"\nBest cluster: #{best_cluster_id} (n={int(best_row['count'])}, "
              f"trimmed mean={best_row['trimmed_mean_future_5y_return']:.2%})")
    else:
        print("\nNo reliable cluster found.")

if __name__ == "__main__":
    main()