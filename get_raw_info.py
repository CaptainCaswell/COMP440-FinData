import yfinance as yf
import pandas as pd
import time
import os
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Set
from datetime import datetime

# Files
RAW_FILE = "data/raw_data.parquet"
SNAPSHOT_RAW_FILE = "data/raw_snapshot.parquet"
INFO_FILE = "data/raw_info.parquet"
STOCK_FILE = "data/data.parquet"
SECTOR_FILE = "data/sector.parquet"

TIME_WINDOWS = {
    "1d": 1,
    "1w": 5,
    "1m": 21,
    "6m": 126,
    "1y": 251,
    "3y": 753,
    "5y": 1254
}

WINDOW_STRIDE = 5
BATCH_SIZE = 20 # How many tickers to download at one time
PERIOD = "15y" # Total length of data downloaded
TICKER_COUNT = 10 # Number of symbols loaded (random sampling)
MIN_ROWS = 3574 # How much data a ticker must have after removing bad values (5% loss)
LOOKBACK_DAYS = max( TIME_WINDOWS.values() ) # Longest time window
FUTURE_DAYS = 1254 # 5 years
MIN_WINDOWS = 500 # Minimum number of rolling windows for a ticker
REQUEST_DELAY = 0.5
CHECKPOINT_EVERY = 20

parser = argparse.ArgumentParser()
parser.add_argument(
    "--snapshot",
    action="store_true",
    help="Fetch info for tickers in the snapshot raw file instead of training raw file"
)
args = parser.parse_args()

def ensure_data_directory() -> None:
    # Create bath if it doesn't exist
    Path( "data" ).mkdir( exist_ok=True )

def load_existing_info() -> pd.DataFrame:
    if os.path.isfile( INFO_FILE ):
        df = pd.read_parquet( INFO_FILE )
        print( f"Loaded existing info for {len(df)} tickers from {INFO_FILE}")
        return df
    return pd.DataFrame( columns=["ticker", "sector", "quote_type", "shares_outstanding", "trailing_pe", "fetched_at"] )

def is_complete( row: pd.Series ) -> bool:
    # Check if row complete (No flag, )
    if not bool( row.get( "fetch_success", False ) ):
        return False
    if pd.isna( row.get( "shares_outstanding" ) ):
        return False
    return True

def get_tickers( raw_file: str ) -> list:
    # Load the ticker list from the raw price data
    if not os.path.isfile( raw_file ):
        raise FileNotFoundError( f"{raw_file} not found. Run the raw data download step first." )
    raw_df = pd.read_parquet( raw_file )
    return list( raw_df["close"].columns )

def finite_float( value ) -> Optional[float]:
    # Cleans infinite values
    try:
        num = float( value )
    except ( TypeError, ValueError ):
        return None
    if num != num or num in ( float( "inf" ), float( "-inf" ) ):
        return None
    return num

def get_info( ticker:str ) -> dict:
    # Fetch info for a single ticker from yfiance
    try:
        info = yf.Ticker( ticker ).info

        return {
            "ticker": ticker,
            "sector": info.get("sector") or "Unknown",
            "quote_type": info.get( "quoteType" ) or "UNKNOWN",
            "shares_outstanding": finite_float( info.get( "sharesOutstanding", None ) ),
            "trailing_pe": finite_float( info.get( "trailingPE", None ) ),
            "fetch_success": True,
            "fetched_at": datetime.now()
        }

    except Exception as e:
        print( f"    Failed to fetch {ticker}: {e}" )
        return {
            "ticker": ticker,
            "sector": "Unknown",
            "quote_type": info.get( "quoteType" ),
            "shares_outstanding": None,
            "trailing_pe": None,
            "fetch_success": False,
            "fetched_at": datetime.now()
        }
    
def save_info( existing: pd.DataFrame, new_rows: list, refetch_tickers: set ) -> pd.DataFrame:
    # Merge newly fetched rows into the existing info cache and save to disk
    if not existing.empty and refetch_tickers:
        existing = existing[~existing["ticker"].isin( refetch_tickers )]

    if new_rows:
        new_df = pd.DataFrame( new_rows )
        combined = pd.concat( [existing, new_df], axis=0, ignore_index=True )
    else:
        combined = existing

    combined = combined.drop_duplicates( subset="ticker", keep="last" ).reset_index( drop=True )
    combined.to_parquet( INFO_FILE )
    return combined

def split_complete_incomplete( tickers: List[str], existing: pd.DataFrame ) -> tuple:
    # Split tickers into complete_tickers and to_fetch based on existing cache
    complete_tickers: Set[str] = set()

    if not existing.empty:
        for _, row in existing.iterrows():
            if is_complete( row ):
                complete_tickers.add( row["ticker"] )

    to_fetch = [t for t in tickers if t not in complete_tickers]
    return complete_tickers, to_fetch

def fetch_all( to_fetch: List[str], existing: pd.DataFrame ) -> tuple:
    # Fetch info for all pending tickers, checkpointing periodically. Returns new_rows and combined_df.
    new_rows = []
    refetch_tickers = set( to_fetch )
    combined = existing
    
    try:
        for i, ticker in enumerate( to_fetch, 1 ):
            print( f"  [{i}/{len(to_fetch)}] {ticker}" )
            new_rows.append( get_info( ticker ) )
            time.sleep( REQUEST_DELAY )

            if i % CHECKPOINT_EVERY == 0:
                combined = save_info( existing, new_rows, refetch_tickers )
                print( f"    -- checkpoint saved ({len(new_rows)}/{len(to_fetch)} fetched) --" )
    
    finally:
        combined = save_info( existing, new_rows, refetch_tickers )

    return new_rows, combined

def print_summary(
        combined: pd.DataFrame,
        new_rows: list,
        to_fetch: list,
        start_time: datetime
) -> None:
    duration = ( datetime.now() - start_time ).total_seconds()
    fetched_ok = sum( 1 for r in new_rows if r["fetch_success"] )
    
    print( "\n" + "=" * 60 )
    print( "INFO FETCH COMPLETE" )
    print( "=" * 60 )
    print( f"Output file: {INFO_FILE}" )
    print( f"Total tickers cached: {len(combined)}" )
    print( f"Fetched this run: {len(new_rows)} ({fetched_ok} succeeded, {len(new_rows) - fetched_ok} failed)" )
    print( f"Duration: {duration:.1f} seconds" )
    if len (new_rows ) < len( to_fetch ):
        print( f"Run interrupted: {len( to_fetch ) - len(new_rows)} tickers not attempted. "
               f"Re-run the script to continue." )

def main():
    # Get starting time
    start_time = datetime.now()
    print( f" Getting info data at {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    ensure_data_directory()

    raw_file = SNAPSHOT_RAW_FILE if args.snapshot else RAW_FILE

    # Get tickers from raw data
    tickers = get_tickers( raw_file )
    print(f"Found {len(tickers)} tickers in {raw_file}")

    existing = load_existing_info()

    # Get complete and incomplete tickers
    complete_tickers, to_fetch = split_complete_incomplete( tickers, existing )
    incomplete_count = len( existing ) - len( complete_tickers ) if not existing.empty else 0
    print( f"{len(complete_tickers)} already complete, {incomplete_count} incomplete/failed, "
           f"{len(to_fetch)} to fetch this run\n")
    
    new_rows, combined = fetch_all( to_fetch, existing )
    print_summary( combined, new_rows, to_fetch, start_time )

if __name__ == "__main__":
    main()