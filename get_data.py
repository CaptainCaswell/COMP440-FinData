import yfinance as yf
import pandas as pd
import numpy as np
import random
import time
import json
import os

RAW_FILE = "data/raw_data.parquet"
DATA_FILE = "data/data.parquet"
TICKERS = "data/company_tickers.json"

BATCH_SIZE = 20
PERIOD = "15y"
TICKER_COUNT = 100 # Number of symbols loaded (random sampling)
MIN_ROWS = 3000

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
    # Get a randomized list of symbols
    random.shuffle( symbols )
    symbol_queue = list( symbols )
    data = {}

    # Loop through all available tickers
    while len( data ) < TICKER_COUNT and symbol_queue:
        # Seperate off current batch
        batch = symbol_queue[:BATCH_SIZE]
        symbol_queue = symbol_queue[BATCH_SIZE:]

        added = 0

        try:
            df = yf.download( batch, period=PERIOD, auto_adjust=True )
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
                    continue

                data[ticker] = series
                added += 1

        except Exception as e:
            print(f"Batch failed: {e}")

        print( f"{added} of {len( batch )} tickers added. Current total is {len(data)}.")

    if not data:
        print("No data collected.")
        return None
        
    # Turn list of df into single df
    raw_df = pd.DataFrame( data )
    raw_df.to_parquet( RAW_FILE )
    print( f"{len( collected )} tickers saved to file." )

    return raw_df  

def add_returns( series, i ):
    row = {}

    future_price = series.iloc[i + 1254]
    current_price = series.iloc[i]

    row["future_5y_return"] = ( future_price / current_price ) - 1

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

def add_monotonic( row_data ):
    values = []

    # Get return values
    for label in TIME_WINDOWS.keys():
        key = f"ret_{label}"
        if key in row_data and row_data[key] is not None:
            values.append( row_data[key] )

    if len( values ) < 2:
        return 0
    
    score = 0

    # Add a point for each monotonic
    for i in range( 1, len( values ) ):
        if values[i] >= values[i - 1]:
            score += 1

    # Normalize score
    normal_score = score / ( len( values ) - 1 )
    
    # Guard against bad score
    if normal_score >= 0 and normal_score <= 1:
        return normal_score
    
    return 0

def add_drawdown( series ):
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
    series = series.dropna()

    if len( series ) < 2:
        return None
    
    x = np.arange( len( series ) )
    y = np.log( series.values )

    slope = np.polyfit( x, y, 1 )[0]

    return slope

def get_symbols():
    # Load file
    with open( TICKERS, "r" ) as file:
        data = json.load( file )

    # Return list of all tickers
    return [stock["ticker"] for stock in data.values() ]

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

        # Make sure there is enough data for minimum 500 windows
        if len( series ) < lookback + future + 500:
            continue

        # Starting with enough data to look back on, iterate until only enough left to check future
        for i in range( lookback, len( series ) - future ):
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
            row_data["1y_drawdown"] = add_drawdown( series[i - 252: i] )
            row_data["5y_drawdown"] = add_drawdown( series[i - 1254: i] )
            row_data["1y_trend"] = add_trend( series[i - 252 : i] )
            row_data["5y_trend"] = add_trend( series[i - 1254 : i] )

            # Calculate returns
            row.update( row_data )
            results.append( row )

        print( f"{ticker} feature build complete...")

    return pd.DataFrame( results )

def main():
    # Get raw data
    if os.path.isfile( RAW_FILE ):
        # Data from file
        raw_df = pd.read_parquet( RAW_FILE )
        print( "Existing raw data found. Loading data from file..." )
    else:
        symbols = get_symbols()

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
    print( f"Features complete and saved. {len(data):,} rows ready for training." )

if __name__ == "__main__":
    main()