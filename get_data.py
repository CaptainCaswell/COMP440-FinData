import yfinance as yf
import pandas as pd
import pickle
import random
import time
import json
import os

RAW_FILE = "data/raw_data.parquet"
DATA_FILE = "data/data.parquet"
TICKERS = "data/company_tickers.json"

MIN_VALUATION = 10_000_000 # Drops stocks below threashold
BATCH_SIZE = 10
PERIOD = "15y"
TICKER_COUNT = 20 # Number of symbols loaded (random sampling)
TIME_WINDOWS = {
    "1d": 1,
    "1w": 5,
    "1m": 21,
    "6m": 126,
    "1y": 251,
    "3y": 753,
    "5y": 1254
}

def get_data( symbols ):
    try:
        all_data = []

        # Add each df from each batch to a list
        for batch in chunk_list( symbols, BATCH_SIZE):
            df = yf.download( batch, period=PERIOD, auto_adjust=True )
            df = df["Close"]

            all_data.append( df )
        
        # Turn list of df into single df
        raw_df = pd.concat( all_data, axis=1 )

        raw_df.to_parquet( RAW_FILE )
        return raw_df

    except Exception as e:
        print(f"Failed to get Data: {e}")
        return None

def add_returns( series, i ):
    row = {}

    current_price = series.iloc[i]

    if pd.isna( current_price ) or current_price == 0:
        return None

    for label, span in TIME_WINDOWS.items():
            if i - span < 0:
                row[f"ret_{label}"] = None
                continue
            
            past_price = series.iloc[i - span]

            if pd.isna( past_price ) or past_price == 0:
                row[f"ret_{label}"] = None
                continue
            
            row[f"ret_{label}"] = ( current_price / past_price ) - 1

    return row

def add_monotonic( df ):
    return df


def get_symbols():
    # For testing, small sample
    return ["AAPL", "MSFT"]
    '''
    # Load file
    with open( TICKERS, "r" ) as file:
        data = json.load( file )

    # Return list of all tickers
    return [stock["ticker"] for stock in data.values() ]
    '''

def chunk_list( lst, size ):
    # Generator to get subsets of a list
    for i in range( 0, len(lst), size ):
        yield lst[i:i + size]

def build_features( close ):
    # Convert from wide (tickers as columns) → long format
    results = []

    lookback = max( TIME_WINDOWS.values() )
    future = 1254

    for ticker in close.columns:
        series = close[ticker].dropna()

        if len( series ) < 3000:
            continue

        # Starting with enough data to look back on, iterate until only enough left to check future
        for i in range( lookback, len( series ) - future ):
            row = {
                "ticker": ticker,
                "date": series.index[i],
                "price": series.iloc[i]
            }

            returns = add_returns( series, i )

            if returns is None:
                continue

            # Calculate returns
            row.update( returns )
            results.append( row )

    return pd.DataFrame( results )

def main():
    # Get all tickers
    symbols = get_symbols()
    print(f"{len( symbols )} tickers have been imported.\n")

    # Get random selection from symbols
    # symbols = random.sample( symbols, TICKER_COUNT )

    # Get raw data
    if os.path.isfile( RAW_FILE ):
        # Data from file
        raw_df = pd.read_parquet( RAW_FILE )
        print( "Existing raw data found..." )
    else:
        # Download data
        raw_df = get_data( symbols )

        # Confirm dataframe
        if raw_df is not None:
            print( "Raw data download sucessfull..." )
        else:
            print( "Raw data download failed." )
            return
        
    # Calculate returns from raw data
    data = build_features( raw_df )

    data.to_parquet( DATA_FILE )

if __name__ == "__main__":
    main()