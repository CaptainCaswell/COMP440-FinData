import pandas as pd
import os
from pathlib import Path
from typing import Optional
from datetime import datetime

# Files
RAW_FILE = "data/raw_snapshot.parquet"
INFO_FILE = "data/raw_info.parquet"
SNAPSHOT_FILE = "data/snapshot.parquet"

TIME_WINDOWS = {
    "1d": 1,
    "1w": 5,
    "1m": 21,
    "6m": 126,
    "1y": 251,
    "3y": 753,
    "5y": 1254
}

# Data quality filters (same thresholds as build_data.py)
MIN_PRICE = 1.00
MAX_PRICE = 5000
MIN_VOLUME = 10000
MAX_CHANGE = 1.00

INFO_MAP = None


def ensure_data_directory():
    # Create path if it doesn't exist
    Path( "data" ).mkdir( exist_ok=True )


def load_raw_data() -> tuple:
    # Loads raw snapshot price data
    if not os.path.isfile( RAW_FILE ):
        raise FileNotFoundError(f"{RAW_FILE} not found. Run get_raw_data.py first.")

    raw_df = pd.read_parquet( RAW_FILE )
    return raw_df["close"], raw_df["volume"]


def load_info_map() -> dict:
    # Loads current ticker information
    if not os.path.isfile( INFO_FILE ):
        raise FileNotFoundError(f"{INFO_FILE} not found. Run get_info.py --snapshot first.")

    info_df = pd.read_parquet( INFO_FILE )
    info_map = {}

    for _, row in info_df.iterrows():
        info_map[row["ticker"]] = {
            "sector": row["sector"] if pd.notna(row["sector"]) else "Unknown",
            "quote_type": row["quote_type"] if pd.notna( row["quote_type"] ) else "UNKNOWN",
        }

    return info_map


def get_info( ticker: str, info_map: dict ) -> dict:
    return info_map.get(ticker, {
        "sector": "Unknown",
        "quote_type": "UNKNOWN",
    } )


def process_ticker( ticker: str, series: pd.Series, volume: pd.Series, target_date: pd.Timestamp ) -> Optional[pd.DataFrame]:
    # Same price-quality checks as build_data.py's process_ticker, but keeps only
    # the most recent valid row and skips everything future-return related.

    series = series.dropna().sort_index()

    df = pd.DataFrame( index=series.index )
    df["ticker"] = ticker
    df["price"] = series

    # Remove improbable prices
    df = df[( df["price"] >= MIN_PRICE ) & ( df["price"] <= MAX_PRICE )]

    if df.empty:
        return None

    # Check for anomalous price changes
    daily_ret = df["price"].pct_change()
    above_floor = df["price"] >= MIN_PRICE
    big_moves = ( daily_ret.abs() > MAX_CHANGE ) & above_floor

    if big_moves.any():
        vol_aligned = volume.reindex( df.index ) if volume is not None else None

        if vol_aligned is not None:
            suspicious_jump = big_moves & ( vol_aligned.fillna(0) < MIN_VOLUME )
            if suspicious_jump.any():
                print( f"    **** {ticker} rejected due to suspicious jump in price, change of {daily_ret[suspicious_jump].abs().max():.2%} with less than {MIN_VOLUME} trades ****")
                return None
        else:
            print( f"    **** {ticker} rejected due to suspicious jump in price, change of {daily_ret[big_moves].abs().max():.2%}, no volume data available to verify ****")
            return None

    # Info
    info = get_info( ticker, INFO_MAP )
    df["quote_type"] = info["quote_type"]

    # Trailing returns
    for label, span in TIME_WINDOWS.items():
        df[f"ret_{label}"] = df["price"].pct_change( span )

    # Keep only the most recent row - the one we're predicting on
    latest = df.loc[[target_date]]

    # Drop it if any trailing window couldn't be computed (not enough history yet)
    required_cols = [f"ret_{label}" for label in TIME_WINDOWS]
    latest = latest.dropna( subset=required_cols )

    if latest.empty:
        return None

    return latest


def build_snapshot( close: pd.DataFrame, volume: pd.DataFrame, info_map: dict ) -> pd.DataFrame:
    global INFO_MAP
    INFO_MAP = info_map

    target_date = close.index.max()
    print( f'Snapshot date: {target_date.date()}' )

    df_list = []
    total = len( close.columns )

    print( f"Processing {total} tickers..." )

    for i, ticker in enumerate( close.columns, 1 ):
        try:
            df = process_ticker( ticker, close[ticker], volume[ticker], target_date )
        except Exception as e:
            print( f"    [{i}/{total}] {ticker}: FAILED ({e})" )
            continue

        if df is None or df.empty:
            print( f"    [{i}/{total}] Skipping {ticker}: insufficient data" )
            continue

        df_list.append( df )
        print( f"    [{i}/{total}] {ticker}: latest row as of {df.index[0].date()}" )

    result = pd.concat( df_list, axis=0 )
    return result


def main():
    start_time = datetime.now()

    ensure_data_directory()

    print(f"Loading raw price data from {RAW_FILE}...")
    close_df, volume_df = load_raw_data()

    if close_df is None or close_df.empty:
        print( "Raw data error. No data found..." )
        return

    print(f"  Raw data loaded for {len(close_df.columns)} tickers, {len(close_df)} rows\n")

    print(f"Loading cached info from {INFO_FILE}...")
    info_map = load_info_map()
    print(f"  Loaded info for {len(info_map)} tickers\n")

    print( "Building snapshot features..." )
    data = build_snapshot( close_df, volume_df, info_map )
    data = data.reset_index()

    if "Date" in data.columns:
        data = data.rename( columns={"Date": "date"} )
    elif "index" in data.columns:
        data = data.rename( columns={"index": "date"} )

    print( "" )

    data.to_parquet( SNAPSHOT_FILE )

    end_time = datetime.now()
    duration = ( end_time - start_time ).total_seconds()

    print( "\n" + "=" * 60 )
    print( "SNAPSHOT BUILD COMPLETE" )
    print( "=" * 60 )
    print( f"Output file: {SNAPSHOT_FILE}" )
    print( f"Total tickers: {len( data ):,}" )
    print( f"Date range: {data['date'].min()} to {data['date'].max()}" )
    print( f"Duration: {duration:.1f} seconds" )

if __name__ == "__main__":
    main()