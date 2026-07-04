import yfinance as yf
import pandas as pd
import numpy as np
import random
import time
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Files
RAW_FILE = "data/raw_data.parquet"
STOCK_FILE = "data/data.parquet"
SECTOR_FILE = "data/sector.parquet"
TICKERS_FILE = "data/company_tickers.json"

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
CHECKPOINT_EVERY_BATCHES = 10

def ensure_data_directory():
    # Create bath if it doesn't exist
    Path( "data" ).mkdir( exist_ok=True )

def get_symbols() -> List[str]:
    # Load tickers from file
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

def get_data( all_symbols: List[str], existing_df: pd.DataFrame ) -> Optional[pd.DataFrame]:
    # Download stock data from random sample of symbols
    # Args:
    #     symbols: List of ticker symbols to sample from
    #     existing_df: DataFrame with already downloaded raw data
    # Returns: Dataframe with stock prices or None if download fails

    # Create list of symbols already downloaded
    if not existing_df.empty:
        existing_tickers = set( existing_df["close"].columns )
    else:
        existing_tickers = set()

    # Create list of symbols that need to be added
    symbols = [s for s in all_symbols if s not in existing_tickers]

    random.shuffle( symbols )
    symbol_queue = list( symbols )
    
    close_data = {}
    volume_data = {}

    raw_df = existing_df
    batch_num = 0
    
    print( f"Starting download for the remaining { len(symbols) } tickers..." )

    # Loop through all available tickers
    while symbol_queue:
        # Seperate off current batch
        batch = symbol_queue[:BATCH_SIZE]
        symbol_queue = symbol_queue[BATCH_SIZE:]

        added = 0
        batch_num += 1

        try:
            # Download batch
            full_df = yf.download( batch, period=PERIOD, auto_adjust=True, progress=False )

            # Guard for empty download
            if full_df.empty:
                print( f"Batch returned empty data" )
                continue

            close_df = full_df["Close"]
            volume_df = full_df["Volume"]

            # Guard for single series return instead of dataframe (only one result)
            if isinstance( close_df, pd.Series ):
                close_df = close_df.to_frame( name=batch[0])
                volume_df = volume_df.to_frame( name=batch[0])

            for ticker in close_df.columns:             
                series = close_df[ticker].dropna()

                # Skip tickers without enough data
                if len( series ) < MIN_ROWS:
                    print( f"    Skipping {ticker}: only {len(series)} rows ({MIN_ROWS} required)" )
                    continue

                close_data[ticker] = series
                volume_data[ticker] = volume_df[ticker].reindex( series.index )

                added += 1
                print( f"    + {ticker}: {len(series)} rows added ({len(close_data)} total" )

        except Exception as e:
            print( f"Batch failed: {e}" )

        print( f"    {added} of {len( batch )} tickers added. Current total is {len(close_data)}.")

        if close_data and batch_num % CHECKPOINT_EVERY_BATCHES == 0:
            raw_df = merge_and_save( existing_df, close_data, volume_data )
            print( f"    -- checkpoint saved ({len(close_data)} new tickers so far) --" )

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

def main():
    # Get starting time
    start_time = datetime.now()
    print( f" Getting raw data at {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    ensure_data_directory()

    symbols = get_symbols()

    # Get raw data
    if os.path.isfile( RAW_FILE ):
        print(f"Loading existing raw data from {RAW_FILE}...")
        raw_df = pd.read_parquet(RAW_FILE)
        print(f"  Loaded {len(raw_df['close'].columns)} tickers, {len(raw_df)} rows\n")
    else:
        print("No existing raw data found. New files created...")

        # Download data
        raw_df = pd.DataFrame()

        # Confirm dataframe
        if raw_df is None:
            print( "Raw data download failed. Existing." )
        
        print( "Raw data download successful!\n" )

    # Guard for no symbols
    if not symbols:
        print("No symbols available. Exiting.")
        return

    # Download data for new symbols
    raw_df = get_data( symbols, raw_df )

    # Make sure data was added
    if raw_df is None or raw_df.empty:
        print( "Raw data download unsucessful. No data found..." )
        return
    
    # Summary statistics
    end_time = datetime.now()
    duration = ( end_time - start_time ).total_seconds()

    print( "\n" + "=" * 60 )
    print( "RAW DATA COMPLETE" )
    print( "=" * 60 )
    print( f"Output file: {RAW_FILE}" )
    print( f"Total tickers: {len( raw_df['close'].columns ) - 1}" )
    print( f"Duration: {duration:.1f} seconds" )
    print( f"Memory usage: {raw_df.memory_usage( deep=True ).sum() / 1024**2:.1f} MB" )

if __name__ == "__main__":
    main()