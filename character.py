import pandas as pd
import numpy as np

df = pd.read_parquet("data/clustered_data_gen2_depth4.parquet")
best_cluster_id = 1

# === 1. Concentration checks — rule out regime/sector/ticker artifacts ===
best = df[df["cluster"] == best_cluster_id].copy()
other = df[df["cluster"] != best_cluster_id]

print(f"Rows: {len(best)}, unique tickers: {best['ticker'].nunique()}")
print(f"Avg rows per ticker: {len(best) / best['ticker'].nunique():.1f}")

print("\n=== Sector breakdown ===")
print((best["sector"].value_counts(normalize=True) * 100).round(1))

print("\n=== Date distribution ===")
best["year"] = pd.to_datetime(best["date"]).dt.year
print(best["year"].value_counts().sort_index())

print("\n=== Ticker concentration ===")
top10 = best["ticker"].value_counts().head(10)
print(f"Top 10 tickers = {top10.sum() / len(best):.1%} of cluster")

# === 2. What actually defines this cluster — standardized feature differences ===
exclude = {"ticker", "date", "sector", "future_5y_return", "cluster", "year"}
feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

cluster_means = df.groupby("cluster")[feature_cols].mean().T
overall_std = df[feature_cols].std()

cluster_means["diff"] = cluster_means[best_cluster_id] - cluster_means.drop(columns=best_cluster_id).mean(axis=1)
cluster_means["diff_in_stds"] = cluster_means["diff"] / overall_std

result = cluster_means.sort_values("diff_in_stds", key=abs, ascending=False)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 150)
print("\n=== Feature characterization (sorted by standardized difference) ===")
print(result.head(20).to_string())

# === 3. Return profile vs. rest of dataset ===
print("\n=== Return comparison ===")
print(f"Best cluster — mean: {best['future_5y_return'].mean():.2%}, median: {best['future_5y_return'].median():.2%}")
print(f"Rest of data — mean: {other['future_5y_return'].mean():.2%}, median: {other['future_5y_return'].median():.2%}")