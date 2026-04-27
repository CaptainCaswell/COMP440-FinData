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
DATA_FILE = "data/data.parquet"
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

BATCH_SIZE = 20 # How many tickers to download at one time
PERIOD = "15y" # Total length of data downloaded
TICKER_COUNT = 100 # Number of symbols loaded (random sampling)
MIN_ROWS = 3574 # How much data a ticker must have after removing bad values (5% loss)
LOOKBACK_DAYS = max( TIME_WINDOWS.values() ) # Longest time window
FUTURE_DAYS = 1254 # 5 years
MIN_WINDOWS = 500 # Minimum number of rolling windows for a ticker

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

def get_data( symbols: List[str] ) -> Optional[pd.DataFrame]:
    # Download stock data from random sample of symbols
    # Args:
    #     symbols: List of ticker symbols to sample from
    # Returns: Dataframe with stock prices or None if download fails

    random.shuffle( symbols )
    symbol_queue = list( symbols )
    data = {}
    
    print( f"Starting download for up to {TICKER_COUNT} tickers..." )

    # Loop through all available tickers
    while len( data ) < TICKER_COUNT and symbol_queue:
        # Seperate off current batch
        batch = symbol_queue[:BATCH_SIZE]
        symbol_queue = symbol_queue[BATCH_SIZE:]

        added = 0

        try:
            # Download batch
            df = yf.download( batch, period=PERIOD, auto_adjust=True, progress=False )

            # Guard for empty download
            if df.empty:
                print( f"Batch returned empty data" )
                continue

            df = df["Close"]

            # Guard for single series return instead of dataframe (only one result)
            if isinstance( df, pd.Series ):
                df = df.to_frame( name=batch[0])

            for ticker in df.columns:
                if len( data ) >= TICKER_COUNT:
                    break
                
                series = df[ticker].dropna()

                # Skip tickers without enough data
                if len( series ) < MIN_ROWS:
                    print( f"    Skipping {ticker}: only {len(series)} rows ({MIN_ROWS} required)" )
                    continue

                data[ticker] = series
                added += 1

        except Exception as e:
            print(f"Batch failed: {e}")

        print( f"    {added} of {len( batch )} tickers added. Current total is {len(data)}.")

    if not data:
        print("No data collected.")
        return None
        
    # Turn list of df into single df
    raw_df = pd.DataFrame( data )
    raw_df.to_parquet( RAW_FILE )
    print( f"{len( data )} tickers saved to file." )

    return raw_df  

def add_returns( series: pd.Series, i: int ) -> Optional[Dict]:
    # Get multiple historical returns for a single ticker
    # Args:
    #     series: Price series for a ticker
    #     i: Current index for ticker
    # Returns: Dictionary with return calculations or None if data insufficient

    if i + FUTURE_DAYS >= len( series ):
        return None

    future_price = series.iloc[i + FUTURE_DAYS]
    current_price = series.iloc[i]

    # Guard for very values
    if pd.isna( current_price ) or current_price <= 0:
        return None
    
    if pd.isna( future_price ) or future_price <= 0:
        return None

    # Get target value
    row = { "future_5y_return": ( future_price / current_price ) - 1 }

    # Calculate historical returns for each time window
    for label, span in TIME_WINDOWS.items():
            if i - span < 0:
                row[f"ret_{label}"] = None
                continue
            
            past_price = series.iloc[i - span]

            if pd.isna( past_price ) or past_price <= 0:
                row[f"ret_{label}"] = None
                continue
            
            row[f"ret_{label}"] = ( current_price / past_price ) - 1

    return row

def add_monotonic( row_data: Dict ) -> float:
    # Calculate monotonic score (how consistently returns increase for each time window)
    # Args:
    #     row_data: Dictionary containing ticker return values for each window
    # Returns: Score between 0 and 1 representing monotonic score
    
    values = []

    # Get return values
    for label in TIME_WINDOWS.keys():
        key = f"ret_{label}"
        if key in row_data and row_data[key] is not None:
            values.append( row_data[key] )

    if len( values ) < 2:
        return 0.0
    
    score = 0

    # Add a point for each monotonic
    for i in range( 1, len( values ) ):
        if values[i] >= values[i - 1]:
            score += 1

    # Normalize score
    normal_score = score / ( len( values ) - 1 )
    
    # Guard against bad score
    if 0 <= normal_score <= 1:
        return normal_score
    
    return 0.0

def add_monotonic_daily( series: pd.Series ) -> float:
    # Calculate monotonic score based on day to day changes
    # Args:
    #     series: Price series
    # Returns: Score between 0 and 1 representing monotonic score
    
    series = series.dropna()

    if len( series ) < 2:
        return None
    
    # Calculate faily returns
    daily_returns = series.pct_change().dropna()

    if len( daily_returns ) == 0:
        return 0.0
    
    # Count days where price increased (monotonic)
    up_days = ( daily_returns > 0 ).sum()
    total_days = len( daily_returns )
    
    return up_days / total_days

def add_drawdown( series: pd.Series ) -> Optional[float]:
    # Calculate maximum drawdown for a price series
    # Args:
    #     series: Price series
    # Returns: Maximum drawdown as negative percentage or None if insufficient data

    series = series.dropna()

    if len( series ) < 2:
        return None

    # Get list of maximum value seen
    running_max = series.cummax()

    # Get list of differences between each running max and current
    drawdown = ( series / running_max ) - 1

    # Smallest value is largest drawdown
    max_drawdown = drawdown.min()

    return max_drawdown

def add_trend( series ):
    # Calculate log trend slope for a price seriues
    # Args:
    #     series: Price series
    # Returns:Trend slope or None if insufficient data

    series = series.dropna()

    if len( series ) < 2:
        return None
    
    x = np.arange( len( series ) )
    y = np.log( series.values )

    slope, _ = np.polyfit( x, y, 1 )

    return slope



def build_features( close: pd.DataFrame ) -> pd.DataFrame:
    # Build data feature matrix from raw price data
    # Args:
    #     close: DataFrame with closeing prices
    # Returns: Dataframe with calculated features

    results = []

    min_length = LOOKBACK_DAYS + FUTURE_DAYS + MIN_WINDOWS

    for ticker in close.columns:
        series = close[ticker].dropna()

        # Make sure there is enough data for minimum 500 windows
        if len( series ) < min_length:
            print( f"    Skipping {ticker}: Only {len(series)} rows ({min_length} needed)" )
            continue

        # Starting with enough data to look back on, iterate until only enough left to check future
        for i in range( LOOKBACK_DAYS, len( series ) - FUTURE_DAYS ):
            row = {
                "ticker": ticker,
                "date": series.index[i],
                "price": series.iloc[i]
            }

            row_data = add_returns( series, i )

            # Skip row if no data
            if row_data is None:
                continue

            row_data["monotonic_score"] = add_monotonic( row_data )
            row_data["monotonic_score_daily"] = add_monotonic_daily( series[i - 1254: i] )
            row_data["1y_drawdown"] = add_drawdown( series[i - 252: i] )
            row_data["5y_drawdown"] = add_drawdown( series[i - 1254: i] )
            row_data["1y_trend"] = add_trend( series[i - 252 : i] )
            row_data["5y_trend"] = add_trend( series[i - 1254 : i] )

            # Calculate returns
            row.update( row_data )
            results.append( row )

        print( f"    {ticker}: {len(series) - LOOKBACK_DAYS - FUTURE_DAYS:,} rows processed" ) # TODO Use len(row) instead?

    return pd.DataFrame( results )

def main():
    # Get starting time
    start_time = datetime.now()
    print( f" Starting data at {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    ensure_data_directory()

    # Get raw data
    if os.path.isfile( RAW_FILE ):
        print(f"Loading existing raw data from {RAW_FILE}...")
        raw_df = pd.read_parquet(RAW_FILE)
        print(f"  Loaded {len(raw_df.columns)} tickers, {len(raw_df)} rows\n")
    else:
        print("No existing raw data found. Downloading...")

        symbols = get_symbols()

        # Guard for no symbols
        if not symbols:
            print("No symbols available. Exiting.")
            return

        # Download data
        raw_df = get_data( symbols )

        # Confirm dataframe
        if raw_df is None:
            print( "Raw data download failed. Existing." )
        
        print( "Raw data download successful!\n" )
        
    # Calculate returns from raw data
    print( "=" * 60 )
    data = build_features( raw_df )

    data.to_parquet( DATA_FILE )
    
    # Summary statistics
    end_time = datetime.now()
    duration = ( end_time - start_time ).total_seconds()
    
    print( "\n" + "=" * 60 )
    print( "PIPELINE COMPLETE" )
    print( "=" * 60 )
    print( f"Output file: {DATA_FILE}" )
    print( f"Total rows: {len(data):,}" )
    print( f"Unique tickers: {data['ticker'].nunique()}" )
    print( f"Date range: {data['date'].min()} to {data['date'].max()}" )
    print( f"Duration: {duration:.1f} seconds" )
    print( f"Memory usage: {data.memory_usage( deep=True ).sum() / 1024**2:.1f} MB" )
    
    # Show sample of null values
    null_counts = data.isnull().sum()
    if null_counts.any():
        print( "\nNull value counts:" )
        print( null_counts[null_counts > 0] )

if __name__ == "__main__":
    main()