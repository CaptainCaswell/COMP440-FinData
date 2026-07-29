import pandas as pd
import os
from pathlib import Path
from typing import Optional
from datetime import datetime

# Files
RAW_FILE = "data/recent_data.parquet"
FEATURE_FILE = "data/recent_features.parquet"

TIME_WINDOWS = {
    "1d": 1,
    "1w": 5,
    "1m": 21,
    "6m": 126,
    "1y": 251,
    "3y": 753,
    "5y": 1254
}

LOOKBACK_DAYS = max( TIME_WINDOWS.values() ) # Longest time window
FUTURE_DAYS = 251 # 1 year
MIN_PRICE = 1.00
MAX_PRICE = 5000

SAMPLE_DATE = "2025-07-01"

def ensure_data_directory():
    # Create bath if it doesn't exist
    Path( "data" ).mkdir( exist_ok=True )


def load_raw_data() -> tuple:
    if not os.path.isfile( RAW_FILE ):
        raise FileNotFoundError(f"{RAW_FILE} not found. Run the raw data download step first.")
    
    # Split into prices and volume
    raw_df = pd.read_parquet( RAW_FILE )
    return raw_df["close"], raw_df["volume"]


def process_ticker( ticker: str, series: pd.Series ) -> Optional[dict]:
    series = series.dropna().sort_index()

    if SAMPLE_DATE is None:
        sample_idx = len( series ) - 1
    else:
        sample_idx = series.index.searchsorted( pd.Timestamp( SAMPLE_DATE ), side="right" ) - 1

    if sample_idx < LOOKBACK_DAYS:
        return None

    current_price = series.iloc[sample_idx]

    if current_price < MIN_PRICE or current_price > MAX_PRICE:
        return None
    
    row = {
        "ticker": ticker,
        "date": series.index[sample_idx],
        "price": current_price
    }

    for label, span in TIME_WINDOWS.items():
        row[f"ret_{label}"] = current_price / series.iloc[sample_idx - span] - 1

    if SAMPLE_DATE is not None and sample_idx + FUTURE_DAYS < len ( series ):
        row["future_ret_1y"] = series.iloc[sample_idx + FUTURE_DAYS] / current_price - 1

    return row


def build_features( close: pd.DataFrame ) -> pd.DataFrame:
    # Build data feature matrix from raw price data
    rows = []

    for ticker in close.columns:
        row = process_ticker( ticker, close[ticker] )

        if row is not None:
            rows.append( row )

    return pd.DataFrame( rows )


def main():
    # Get starting time
    start_time = datetime.now()
    print( f" Starting data at {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    ensure_data_directory()

    # Get Raw Data
    print(f"Loading raw price data from {RAW_FILE}...")
    raw_df, volume_df = load_raw_data()

    if raw_df is None or raw_df.empty:
        print( "Raw data error. No data found..." )
        return

    print(f"  Raw data fo loaded for {len(raw_df.columns)} tickers, {len(raw_df)} rows\n")

    # Calculate returns from raw data
    print( "=" * 60 )
    print( "Building feature data")
    print( "=" * 60 )
    data = build_features( raw_df )
    # data = data.reset_index()
    data = data.rename( columns={"Date": "date"} )
    print( "" )

    # Save data
    data.to_parquet( FEATURE_FILE )
    
    # Summary statistics
    end_time = datetime.now()
    duration = ( end_time - start_time ).total_seconds()
    
    print( "\n" + "=" * 60 )
    print( "BUILD COMPLETE" )
    print( "=" * 60 )
    print( f"Output file: {FEATURE_FILE}" )
    print( f"Total rows: {len( data ):,}" )
    print( f"Unique tickers: {data['ticker'].nunique()}" )
    print( f"Date range: {data['date'].min()} to {data['date'].max()}" )
    print( f"Duration: {duration:.1f} seconds" )
    print( f"Memory usage: {data.memory_usage( deep=True ).sum() / 1024**2:.1f} MB" )


if __name__ == "__main__":
    main()