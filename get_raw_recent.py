import yfinance as yf
import pandas as pd
import random
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Files
RAW_FILE = "data/recent_data.parquet"
TICKERS_FILE = "data/company_tickers.json"

BATCH_SIZE = 20 # How many tickers to download at one time
PERIOD = "11y" # Total length of data downloaded
MIN_ROWS = 1600 # How much data a ticker must have after removing bad values (5% loss)
CHECKPOINT_EVERY_BATCHES = 10 # Number of batches to fetch between saving

def ensure_data_directory() -> None:
    # Create path if it doesn't exist
    Path( "data" ).mkdir( exist_ok=True )

def get_symbols() -> List[str]:
    # Load ticker symbols from file tickers JSON file
    try:
        with open( TICKERS_FILE, "r" ) as file:
            data = json.load( file )

        symbols = [stock["ticker"] for stock in data.values() ]
        print( f"Loaded {len(symbols)} symbols from {TICKERS_FILE}" )

        return symbols
    
    except FileNotFoundError:
        print( f"Error: {TICKERS_FILE} not found")
        return []

    except json.JSONDecodeError:
        print( f"Error: Invalid JSON in {TICKERS_FILE}" )
        return []

def load_raw_data() -> pd.DataFrame:
    # Load existing raw data from file, or return empty frame if none exists.
    if os.path.isfile( RAW_FILE ):
        print(f"Loading existing raw data from {RAW_FILE}...")
        raw_df = pd.read_parquet(RAW_FILE)
        print(f"  Loaded {len(raw_df['close'].columns)} tickers, {len(raw_df)} rows\n")
        return raw_df
    
    print("No existing raw data found. New files created...")
    return pd.DataFrame()

def get_missing_symbols( all_symbols: List[str], existing_df: pd.DataFrame ) -> List[str]:
    # Return sybols not already present in existing_df
    if not existing_df.empty:
        existing_tickers = set( existing_df["close"].columns )
    else:
        existing_tickers = set() # TODO just return all_symbols here?

    symbols = [s for s in all_symbols if s not in existing_tickers]
    return symbols

def download_batch( batch: List[str] ) -> Optional[tuple[pd.DataFrame, pd.DataFrame]]:
    # Download one batch of tickers. Returns close_df and volume_df, or None
    try:
        # Download batch
        full_df = yf.download( batch, period=PERIOD, auto_adjust=True, progress=False )

        # Guard for empty download
        if full_df.empty:
            print( f"Batch returned empty data" )
            return None

        close_df = full_df["Close"]
        volume_df = full_df["Volume"]

        # Guard for single series return instead of dataframe (only one result)
        if isinstance( close_df, pd.Series ):
            close_df = close_df.to_frame( name=batch[0])
            volume_df = volume_df.to_frame( name=batch[0])

        return close_df, volume_df

    except Exception as e:
        print( f"Batch failed: {e}" )
        return None

def filter_batch_tickers( close_df: pd.DataFrame, volume_df: pd.DataFrame, close_data: Dict[str, pd.Series], volume_data: Dict[str, pd.Series] ) -> int:
    # Add tickers with enough rows from a downloaded batch into close_data/volume_data. Return number of tickers added from batch.

    added = 0

    for ticker in close_df.columns:             
        series = close_df[ticker].dropna()

        # Skip tickers without enough data
        if len( series ) < MIN_ROWS:
            print( f"    Skipping {ticker}: only {len(series)} rows ({MIN_ROWS} required)" )
            continue

        close_data[ticker] = series
        volume_data[ticker] = volume_df[ticker].reindex( series.index )

        added += 1
        print( f"    + {ticker}: {len(series)} rows added ({len(close_data)} total)" )

    return added


def get_data( all_symbols: List[str], existing_df: pd.DataFrame ) -> Optional[pd.DataFrame]:
    # Download stick data for symbols not already in existing_df

    symbol_queue = get_missing_symbols( all_symbols, existing_df )
    print( f"Starting download for the remaining {len(symbol_queue)} tickers..." )
    
    close_data: Dict[str, pd.Series] = {}
    volume_data: Dict[str, pd.Series] = {}
    raw_df = existing_df
    batch_num = 0

    # Loop through all available tickers
    while symbol_queue:
        # Seperate off current batch
        batch = symbol_queue[:BATCH_SIZE]
        symbol_queue = symbol_queue[BATCH_SIZE:]
        batch_num += 1

        result = download_batch( batch )
        added = 0

        if result is not None:
            close_df, volume_df = result
            added = filter_batch_tickers( close_df, volume_df, close_data, volume_data )

        print( f"    {added} of {len( batch )} tickers added. Current total is {len(close_data)}.")

        if close_data and batch_num % CHECKPOINT_EVERY_BATCHES == 0:
            raw_df = merge_and_save( existing_df, close_data, volume_data )
            print( f"    -- Checkpoint saved ({len(close_data)} new tickers so far) --" )

    # Final save, whether or not we hit a checkpoint boundary
    if close_data:
        raw_df = merge_and_save( existing_df, close_data, volume_data )
        print( f"{len(close_data)} new tickers added this run." )
    elif existing_df.empty:
        print( "No data collected." )
        return None
    
    return raw_df

def merge_and_save( existing_df: pd.DataFrame, close_data: dict, volume_data: dict ) -> pd.DataFrame:
    # Create dataframe for new data, then merge with existing
    close_new = pd.concat( close_data, axis=1 )
    volume_new = pd.concat( volume_data, axis=1 )
    new_df = pd.concat( {"close": close_new, "volume": volume_new}, axis=1 )

    if not existing_df.empty:
        merged = pd.concat( [existing_df, new_df], axis=1 )
    else:
        merged = new_df
        
    # Turn list of df into single df
    merged.to_parquet( RAW_FILE )

    return merged

def print_summary( raw_df: pd.DataFrame, start_time: datetime ) -> None:
    # Print summary statistics for the completed run
    duration = ( datetime.now() - start_time ).total_seconds()

    print( "\n" + "=" * 60 )
    print( "RAW DATA COMPLETE" )
    print( "=" * 60 )
    print( f"Output file: {RAW_FILE}" )
    print( f"Total tickers: {len( raw_df['close'].columns ) - 1}" )
    print( f"Duration: {duration:.1f} seconds" )
    print( f"Memory usage: {raw_df.memory_usage( deep=True ).sum() / 1024**2:.1f} MB" )

def main():
    # Get starting time
    start_time = datetime.now()
    print( f" Getting recent (7y) raw data at {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    ensure_data_directory()

    symbols = get_symbols()

    # Guard for no symbols
    if not symbols:
        print("No symbols available. Exiting.")
        return

    # Download data for new symbols
    raw_df = load_raw_data()
    raw_df = get_data( symbols, raw_df )

    # Make sure data was added
    if raw_df is None or raw_df.empty:
        print( "Raw data download unsucessful. No data found..." )
        return
    
    print_summary( raw_df, start_time )

if __name__ == "__main__":
    main()