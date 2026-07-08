import yfinance as yf
import pandas as pd

df = pd.read_parquet("data/clustered_data_gen2_depth4.parquet")
best_cluster = df[df["cluster"] == 1]

spy = yf.download("SPY", period="15y", auto_adjust=True, progress=False)["Close"].squeeze()
spy = spy.dropna()

FUTURE_DAYS = 1254
spy_future_5y_return = (spy.shift(-FUTURE_DAYS) / spy - 1).dropna()

print(f"SPY rolling 5yr return — mean: {spy_future_5y_return.mean():.2%}")
print(f"SPY rolling 5yr return — median: {spy_future_5y_return.median():.2%}")

cluster_dates = pd.to_datetime(best_cluster["date"])
spy_in_range = spy_future_5y_return[
    (spy_future_5y_return.index >= cluster_dates.min()) &
    (spy_future_5y_return.index <= cluster_dates.max())
]

print(f"\nSPY over cluster's date range ({cluster_dates.min().date()} to {cluster_dates.max().date()}):")
print(f"  Mean: {spy_in_range.mean():.2%}")
print(f"  Median: {spy_in_range.median():.2%}")

print(f"\nYour cluster's actual numbers:")
print(f"  Mean: {best_cluster['future_5y_return'].mean():.2%}")
print(f"  Median: {best_cluster['future_5y_return'].median():.2%}")