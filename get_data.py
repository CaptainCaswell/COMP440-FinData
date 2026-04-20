import yfinance as yf
import pandas as pd
import random
import time
import json
import os

OUTPUT = "stock_metrics.parquet"
TICKERS = "company_tickers.json"
MIN_VALUATION = 10_000_000 # Drops stocks below threashold
BATCH_SIZE = 10
PERIOD = "5y"

TESTING = True # Limits symbols, debugging messages
TICKER_COUNT = 20 # Number of symbols loaded (random sampling)
WAIT = 0.25 # Check actual API limits

def get_data( batch ):
    try:
        df = yfinance.download( batch, period=PERIOD )

        for symbol in batch:
            data = df[symbol]
            
            for time_span in 



        return {
            "ticker": symbol,
            "beta": beta,
            "ps_ratio": ps,
            "pe_ratio": pe,
            "market_cap": market_cap,
            "eps_ttm": eps,
            "avg_volume": vol,
            "target_1y_mean": target,
        }
    except Exception as e:
        print(f"Failed {symbol}: {e}")
        return None

def clean_float( x ):
    try:
        # Cleans infinity
        if x in [None, "Infinity"]:
            return None # Change to max float?
        
        # Ensures float
        return float( x )
    
    except:
        return None

def get_symbols():
    # Load file
    with open( TICKERS, "r" ) as file:
        data = json.load( file )

    # Return list of all tickers
    return [stock["ticker"] for stock in data.values() ]

def chunk_list( lst, size ):
    for i in range( 0, len(lst), size ):
        yield lst[i:i + size]

def main():
    # Get all tickers
    symbols = get_symbols()
    print(f"{len( symbols )} tickers have been imported.")

    data = []

    # Reduce to limited number of symbols if testing
    if ( TESTING ):
        symbols = random.sample( symbols, TICKER_COUNT )
        print(f"Tickers reduced to {TICKER_COUNT} for testing")

    # Get batches of ticker symbols
    for batch in chunk_list(symbols, BATCH_SIZE):
        print(f"{len( batch) } tickers in batch: {batch}")

        rates_of_return = get_data( batch )

        # Don't append if None
        if rates_of_return:
            data.append( rates_of_return )
        
        time.sleep( WAIT )  # API ban prevention

        if ( TESTING ):
            print(f"{batch} imported")

    # Create data frame
    df = pd.DataFrame( data )
    df = df.set_index( "ticker" )

    # Save to Parquet
    df.to_parquet( OUTPUT, engine="pyarrow" )

    print(f"{ len( data ) } tickers imported.")
    print(f"Saved to {OUTPUT}")

if __name__ == "__main__":
    main()