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
SECTOR_MAP_FILE = "data/sector_map.json"

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

def ensure_data_directory():
    # Create bath if it doesn't exist
    Path( "data" ).mkdir( exist_ok=True )

def save_sector_map( sector_map: dict ):
    with open( SECTOR_FILE, "w" ) as f:
        json.dump( sector_map, f )

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
    
def load_sector_map() -> dict:
    if os.path.isfile( SECTOR_FILE ):
        try:
            with open( SECTOR_FILE, "r" ) as f:
                return json.load( f )
        except Exception:
            return {}
    return {}

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
            print( f"Batch failed: {e}" )

        print( f"    {added} of {len( batch )} tickers added. Current total is {len(data)}.")

    if not data:
        print("No data collected.")
        return None
        
    # Turn list of df into single df
    raw_df = pd.DataFrame( data )
    raw_df.to_parquet( RAW_FILE )
    print( f"{len( data )} tickers saved to file." )

    return raw_df  

def add_monotonic( row ):
    # TODO Update
    # Calculate monotonic score (how consistently returns increase for each time window)
    # Args:
    #     row_data: Dictionary containing ticker return values for each window
    # Returns: Score between 0 and 1 representing monotonic score
    
    values = []

    # Get return values
    for label in TIME_WINDOWS.keys():
        val = row.get( f"ret_{label}" )
        if pd.notna( val ):
            values.append( val )

    if len( values ) < 2:
        return 0.0
    
    score = sum(
        values[i] >= values[i-1]
        for i in range( 1, len(values) )
    )
    
    return score / ( len( values ) - 1 )

def build_features( close: pd.DataFrame, sector_map: dict ) -> pd.DataFrame:
    # Build data feature matrix from raw price data
    # Args:
    #     close: DataFrame with closeing prices
    # Returns: Dataframe with calculated features

    df_list = []
    min_length = LOOKBACK_DAYS + FUTURE_DAYS + MIN_WINDOWS

    for ticker in close.columns:
        # Create copy of that ticker
        series = close[ticker].copy()

        # Remove empty
        series = series.dropna()

        # Sort
        series = series.sort_index()

        # Make sure there is enough data for minimum 500 windows
        if len( series ) < min_length:
            print( f"    Skipping {ticker}: Only {len(series)} rows ({min_length} needed)" )
            continue

        # Base DataFrame
        df = pd.DataFrame( index=series.index )
        df["ticker"] = ticker
        df["price"] = series
        df["sector"] = get_sector( ticker, sector_map )

        # Future Price
        df["future_5y_return"] = ( df["price"].shift( -TIME_WINDOWS["5y"] ) / df["price"] - 1 )

        # Returns
        for label, span in TIME_WINDOWS.items():
            df[f"ret_{label}"] = df["price"].pct_change( span )

        # Monotonic
        df["monotonic_score_daily"] = (
            df["price"].pct_change().gt( 0 )
            .rolling( TIME_WINDOWS["1y"] )
            .mean()
        )

        df["monotonic_score"] = df.apply( add_monotonic, axis=1 )

        # Drawdown
        rolling_max_1y = df["price"].rolling( TIME_WINDOWS["1y"] ).max()
        df["1y_drawdown"] = df["price"] / rolling_max_1y - 1

        rolling_max_5y = df["price"].rolling( TIME_WINDOWS["5y"] ).max()
        df["5y_drawdown"] = df["price"] / rolling_max_5y - 1

        # Trend
        log_price = np.log( df["price"] )

        def slope(x):
            return np.polyfit( np.arange(len(x)), x, 1 )[0]

        df["1y_trend"] = log_price.rolling( TIME_WINDOWS["1y"] ).apply( slope, raw=True )
        df["5y_trend"] = log_price.rolling( TIME_WINDOWS["5y"] ).apply( slope, raw=True )

        # Switch sectors to numeric dummies
        sector_dummies = pd.get_dummies( df["sector"], prefix="sector" ).astype( "int8" )
        df = pd.concat( [df, sector_dummies], axis=1 )

        # Clean
        df = df.iloc[LOOKBACK_DAYS:-FUTURE_DAYS]

        # Reduce rows
        df = df.iloc[::WINDOW_STRIDE]

        df_list.append( df )

        print( f"    {ticker}: {len(df)} rows processed" )

    return pd.concat( df_list, axis=0 )

def get_sector( ticker: str, sector_map: dict ) -> str | None:
    if ticker in sector_map:
        return sector_map[ticker]
    
    try:
        info = yf.Ticker( ticker ).info
        sector = info.get( "sector", None )
    except Exception:
        sector = None
    
    sector_map[ticker] = sector
    return sector

def build_sector( data: pd.DataFrame) -> pd.DataFrame:
    # Get list of sectors
    sector_columns = [col for col in data.columns if col.startswith( "sector_" )]
    
    # Add MARKET sector
    sectors = [col.removeprefix("sector_") for col in sector_columns]
    sectors.append("MARKET")

    sector_rows = []

    # Iterate through sectors
    for sector in sectors:
        # Get rows for given sector
        if sector == "MARKET":
            sector_df = data
        else:
            sector_df = data[data[f"sector_{sector}"] == 1].copy()

        # Skip empty sectors
        if sector_df.empty:
            continue

        for date, date_df in sector_df.groupby("date"):
            row = {
                # Indentity
                "date": date,
                "sector": sector,

                # Size
                "sector_size": date_df["ticker"].nunique(),
                "rows": len( date_df ),

                # Average returns
                "sec_avg_ret_1d": date_df["ret_1d"].mean(),
                "sec_avg_ret_1w": date_df["ret_1w"].mean(),
                "sec_avg_ret_1m": date_df["ret_1m"].mean(),
                "sec_avg_ret_6m": date_df["ret_6m"].mean(),
                "sec_avg_ret_1y": date_df["ret_1y"].mean(),
                "sec_avg_ret_3y": date_df["ret_3y"].mean(),
                "sec_avg_ret_5y": date_df["ret_5y"].mean(),

                # Breadth
                "sec_breadth_positive_1y": ( date_df["ret_1y"] > 0 ).mean(),
                "sec_breadth_positive_5y": ( date_df["ret_5y"] > 0 ).mean(),

                # Average Trend
                "sec_avg_1y_trend": date_df["1y_trend"].mean(),
                "sec_avg_5y_trend": date_df["5y_trend"].mean(),

                # Percentage of positive trends
                "sec_positive_1y_trend_pct": ( date_df["1y_trend"] > 0 ).mean(),
                "sec_positive_5y_trend_pct": ( date_df["5y_trend"] > 0 ).mean(),

                # Average Drawdown
                "sec_avg_1y_drawdown": date_df["1y_drawdown"].mean(),
                "sec_avg_5y_drawdown": date_df["5y_drawdown"].mean(),

                # Percentage without large Drawdown
                "sec_strong_drawdown_resilience": ( date_df["1y_drawdown"] > -0.2 ).mean(),

                # Average Monotonicity
                "sec_avg_monotonic_score": date_df["monotonic_score"].mean(),
                "sec_avg_monotonic_score_daily": date_df["monotonic_score_daily"].mean(),

                # Percentage with high Monotonicity
                "sec_high_monotonic_pct": ( date_df["monotonic_score"] > 0.8 ).mean(),

                # Spread
                "sec_ret_1y_dispersion": date_df["ret_1y"].std(),
                "sec_ret_5y_dispersion": date_df["ret_5y"].std(),
            }

            sector_rows.append( row )

        print( f"    {sector} sector added.")

    return pd.DataFrame( sector_rows )

def build_comparisons( stock_data: pd.DataFrame, sector_data: pd.DataFrame ) -> pd.Dataframe:
    # Split out market data and rename column heading
    market_data = sector_data[ sector_data["sector"] == "MARKET"].rename( columns=lambda c: c + "_market" if c not in ["date"] else c )
    sector_data = sector_data[sector_data["sector"] != "MARKET"]

    # Merge sector and market data with stocks
    df = stock_data.merge( sector_data, on=["date", "sector"], how="left" )
    df = df.merge( market_data, on=["date"], how="left" )

    # Sector relative features
    df["excess_ret_1y"] = df["ret_1y"] - df["sec_avg_ret_1y"]
    df["excess_ret_5y"] = df["ret_5y"] - df["sec_avg_ret_5y"]

    df["trend_vs_sector_1y"] = df["1y_trend"] - df["sec_avg_1y_trend"]
    df["trend_vs_sector_5y"] = df["5y_trend"] - df["sec_avg_5y_trend"]

    df["drawdown_rel_1y"] = df["1y_drawdown"] - df["sec_avg_1y_drawdown"]
    df["drawdown_rel_5y"] = df["5y_drawdown"] - df["sec_avg_5y_drawdown"]

    # Market relative features
    df["excess_vs_market_1y"] = df["ret_1y"] - df["sec_avg_ret_1y_market"]
    df["trend_vs_market_1y"] = df["1y_trend"] - df["sec_avg_1y_trend_market"]

    # Risk adjusted strength
    df["risk_adjusted_1y"] = df["excess_ret_1y"] / df["sec_ret_1y_dispersion"].replace(0, np.nan)
    df["risk_adjusted_5y"] = df["excess_ret_5y"] / df["sec_ret_5y_dispersion"].replace(0, np.nan)

    # Sector signals
    df["sector_is_strong"] = df["sec_avg_ret_1y"] > 0
    df["sector_is_trending"] = df["sec_avg_1y_trend"] > 0
    df["sector_high_breadth"] = df["sec_breadth_positive_1y"] > 0.6

    # Combined quality score
    df["quality_score"] = (df["excess_ret_1y"] > 0).astype(int) + (df["trend_vs_sector_1y"] > 0).astype(int) + (df["drawdown_rel_1y"] > 0).astype(int)

    return df

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
        # TODO Check for quantity, prompt user if different
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
    
    # Load Sector Map
    sector_map = load_sector_map()

    # Calculate returns from raw data
    print( "=" * 60 )
    print( "Building feature data")
    print( "=" * 60 )
    stock_data = build_features( raw_df, sector_map )
    stock_data = stock_data.reset_index()
    stock_data = stock_data.rename( columns={"Date": "date"} )
    print( "" )

    # Build market/sector data
    print( "=" * 60 )
    print( "Building sector data")
    print( "=" * 60 )
    sector_data = build_sector( stock_data )
    print( "" )

    # Build comparison data
    data = build_comparisons( stock_data, sector_data )

    # Save data
    data.to_parquet( STOCK_FILE )
    sector_data.to_parquet( SECTOR_FILE )
    save_sector_map( sector_map )
    
    # Summary statistics
    end_time = datetime.now()
    duration = ( end_time - start_time ).total_seconds()
    
    print( "\n" + "=" * 60 )
    print( "PIPELINE COMPLETE" )
    print( "=" * 60 )
    print( f"Output file: {STOCK_FILE}" )
    print( f"Total rows: {len( data ):,}" )
    print( f"Unique tickers: {data['ticker'].nunique()}" )
    print( f"Date range: {data['date'].min()} to {data['date'].max()}" )
    print( f"Duration: {duration:.1f} seconds" )
    print( f"Memory usage: {data.memory_usage( deep=True ).sum() / 1024**2:.1f} MB" )

if __name__ == "__main__":
    main()