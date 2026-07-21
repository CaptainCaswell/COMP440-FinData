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
    "future_ret_1d",
    "future_ret_1w",
    "future_ret_1m",
    "future_ret_6m",
    "future_ret_1y",
    "future_ret_3y",
    "future_ret_5y"
]
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
    "excess_ret_1y",
    "excess_ret_5y",
    "trend_vs_sector_1y",
    "trend_vs_sector_5y",
    "drawdown_rel_1y",
    "drawdown_rel_5y",
    "excess_vs_market_1y",
    "trend_vs_market_1y",
    "risk_adjusted_1y",
    "risk_adjusted_5y",
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
    exclude = set( ID_COLS ) | set( TARGET_COLS ) | PRIM_EXCLUDE | SEC_EXCLUDE
    return [c for c in numeric_cols if c not in exclude]

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
            target_stats["future_ret_6m_median"] * 0.10 +
            target_stats["future_ret_1y_median"] * 0.20 +
            target_stats["future_ret_3y_median"] * 0.25 +
            target_stats["future_ret_5y_median"] * 0.35
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
    log_file = open(f"{LOG_FOLDER}/cluster_log_gen{GEN}.txt", "w")
    sys.stdout = Tee(sys.__stdout__, log_file)

    print( "Loading feature dataset..." )
    df = load_data()
    print( f"Load {len(df):,} rows, {df['ticker'].nunique()} tickers\n")

    # Clean any rows with no target
    before = len( df )

    df = df.dropna( subset=TARGET_COLS ).reset_index( drop=True )
    
    after = len( df )
    dropped = before - after
    
    print( f"Dropped {dropped} rows missing future returns\n")

    depth = 0
    current_df = df

    while depth < MAX_DEPTH:
        clustered_df, summary, best_cluster_id = run_clustering( current_df, depth )
        clustered_df.to_parquet( f"{LOG_FOLDER}/clustered_data_gen{GEN}_depth{depth}.parquet" )
        summary.to_csv( f"{LOG_FOLDER}/cluster_summary_gen{GEN}_depth{depth}.csv", index=False)

        if best_cluster_id is None:
            print(f"[Depth {depth}] No reliable cluster found — stopping.")
            break

        best_row = summary[summary["cluster"] == best_cluster_id].iloc[0]
        print( f"\n[depth {depth}] Best cluster: #{best_cluster_id} "
               f"(n={int(best_row['count'])}, overall score={best_row['overall_score']:.2%}, "
               f"5y median={best_row['future_ret_5y_median']:.2%})" )

        next_df = clustered_df[clustered_df["cluster"] == best_cluster_id].drop(columns=["cluster"])

        if len(next_df) < MIN_ROWS:
            print(f"[depth {depth}] Best cluster too small to continue ({len(next_df)} rows) — stopping.")
            break

        current_df = next_df
        depth += 1
    
    print( "\n" + "=" * 60 )
    print( "DRILL-DOWN COMPLETE" )
    print( "=" * 60 )
    print( f"Reached depth {depth}, final subset size: {len(current_df):,} rows" )
        
if __name__ == "__main__":
    main()