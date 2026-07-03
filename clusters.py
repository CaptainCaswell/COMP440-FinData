import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

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
    exclude = set( ID_COLS ) | { TARGET_COL }
    return [c for c in numeric_cols if c not in exclude]

def prepare_features( df: pd.DataFrame, feature_cols: list ) -> pd.DataFrame:
    # Replace any missing cells with median value
    features = df[feature_cols].copy()

    for col in features.columns:
        median_val = features[col].median()
        n_missing = features[col].isna().sum()
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
        rows.append({
            "cluster": cluster_id,
            "count": len(group),
            "mean_future_5y_return": group[TARGET_COL].mean(),
            "median_future_5y_return": group[TARGET_COL].median(),
            "std_future_5y_return": group[TARGET_COL].std(),
            "vs_overall_mean": group[TARGET_COL].mean() - overall_mean,
            "avg_beta_1y": group["beta_1y"].mean(),
            "avg_1y_trend": group["1y_trend"].mean(),
            "avg_5y_drawdown": group["5y_drawdown"].mean(),
            "reliable": len(group) >= MIN_CLUSTER_SIZE
        })

    summary = pd.DataFrame( rows ).sort_values( "mean_future_5y_return", ascending=False ).reset_index( drop=True )
    return summary
    
def main():
    print( "Loading feature dataset..." )
    df = load_data()
    print( f"Load {len(df):,} rows, {df['ticker'].nunique()} tickers\n")

    # Clean any rows with no target
    before = len( df )
    df = df.dropna( subset=[TARGET_COL] ).reset_index( drop=True )
    after = len( df )
    dropped = before - after
    print( f"Dropped {dropped} rows missing {TARGET_COL}\n")

    feature_cols = select_feature_columns( df )
    print(f"Using {len(feature_cols)} features for clustering:")
    print(f"  {feature_cols}\n")

    print("Preparing features (imputing missing values)...")
    features = prepare_features(df, feature_cols)

    print("\nScaling features...")
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
 
    ensure_data_directory()
    df.to_parquet(CLUSTERED_FILE)
    summary.to_csv(SUMMARY_FILE, index=False)
 
    print(f"\nSaved clustered dataset to {CLUSTERED_FILE}")
    print(f"Saved cluster summary to {SUMMARY_FILE}")

    reliable = summary[summary["reliable"]]
    if not reliable.empty:
        best_cluster = reliable.iloc[0]
        print(f"\nBest reliable cluster: #{int(best_cluster['cluster'])} "
              f"(n={int(best_cluster['count'])}, "
              f"mean 5y return={best_cluster['mean_future_5y_return']:.2%})")
    unreliable = summary[~summary["reliable"]]
    if not unreliable.empty:
        print(f"\nNote: {len(unreliable)} cluster(s) have fewer than {MIN_CLUSTER_SIZE} rows "
              f"and are flagged unreliable regardless of their mean return.")
        
if __name__ == "__main__":
    main()