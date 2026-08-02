import pandas as pd
import os
from pathlib import Path
from typing import Optional
from datetime import datetime

# Files
RAW_FILE = "data/raw_data.parquet"
INFO_FILE = "data/raw_info.parquet"
STOCK_FILE = "data/data.parquet"

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
LOOKBACK_DAYS = max( TIME_WINDOWS.values() ) # Longest time window
FUTURE_DAYS = 1254 # 5 years
MIN_WINDOWS = 500 # Minimum number of rolling windows for a ticker

# Data quality filters
MIN_PRICE = 1.00
MAX_PRICE = 5000
MIN_VOLUME = 10000
MAX_CHANGE = 1.00

INFO_MAP = None
MARKET_DAILY_RET = None
SPY_FUTURE_RETURNS = None
SPY_RETURNS = None


def ensure_data_directory():
    # Create bath if it doesn't exist
    Path( "data" ).mkdir( exist_ok=True )


def load_raw_data() -> tuple:
    # Loads raw historical price data
    if not os.path.isfile( RAW_FILE ):
        raise FileNotFoundError(f"{RAW_FILE} not found. Run the raw data download step first.")
    
    # Split into prices and volume
    raw_df = pd.read_parquet( RAW_FILE )
    return raw_df["close"], raw_df["volume"]


def load_info_map() -> dict:
    # Loads current ticker information
    if not os.path.isfile( INFO_FILE ):
        raise FileNotFoundError(f"{INFO_FILE} not found. Run get_raw_info.py first.")
    
    info_df = pd.read_parquet( INFO_FILE )
    info_map = {}

    for _, row in info_df.iterrows():
        info_map[row["ticker"]] = {
            "sector": row["sector"] if pd.notna(row["sector"]) else "Unknown",
            "quote_type": row["quote_type"] if pd.notna( row["quote_type"] ) else "UNKNOWN",
        }

    return info_map


def get_info( ticker: str, info_map: dict ) -> dict:
    # Gets information for specific stock
    return info_map.get(ticker, {
        "sector": "Unknown",
        "quote_type": "UNKNOWN",
    } )


def process_ticker( ticker: str, series: pd.Series, volume: pd.Series ) -> Optional[pd.DataFrame]:

    series = series.dropna().sort_index()

    df = pd.DataFrame( index=series.index )
    df["ticker"] = ticker
    df["price"] = series

    # Remove improbable prices
    df = df[( df["price"] >= MIN_PRICE ) & ( df["price"] <= MAX_PRICE )]

    # Check for anomolous price changes
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
    df["sector"] = info["sector"]

    # Returns
    for label, span in TIME_WINDOWS.items():
        df[f"ret_{label}"] = df["price"].pct_change( span )

    # Future Price
    for label, span in TIME_WINDOWS.items():
        # Return windows
        future_ret = df["price"].shift( -span ) / df["price"] - 1

        # Market return
        spy_future = SPY_FUTURE_RETURNS[label].reindex(df.index)

        # Excess return vs SPY
        df[f"future_excess_{label}"] = future_ret - spy_future

    # Remove rows without required features/targets
    required_cols = [f"ret_{label}" for label in TIME_WINDOWS] + [f"future_excess_{label}" for label in TIME_WINDOWS]

    df = df.dropna( subset=required_cols )

    # Drop if core stats not available
    core_cols = ["price"]
    df = df.dropna( subset=core_cols )

    return df


def build_features( close: pd.DataFrame, volume: pd.DataFrame, info_map: dict ) -> pd.DataFrame:
    # Build data feature matrix from raw price data
    global INFO_MAP, SPY_FUTURE_RETURNS

    INFO_MAP = info_map

    spy_price = close["SPY"]

    spy_future_returns = {}

    for label, span in TIME_WINDOWS.items():
        spy_future_returns[label] = ( spy_price.shift( -span ) / spy_price - 1 )

    SPY_FUTURE_RETURNS = spy_future_returns

    stride_dates = set( close.index[::WINDOW_STRIDE] )

    df_list = []
    total = len( close.columns )
    
    print( f"Processing {total} tickers..." )
    

    for i, ticker in enumerate( close.columns, 1 ):
        try:
            df = process_ticker( ticker, close[ticker], volume[ticker] )
        except Exception as e:
            print( f"    [{i}/{total}] {ticker}: FAILED ({e})" )
            continue

        if df is None or df.empty:
            print( f"    [{i}/{total}] Skipping {ticker}: insufficient data" )
            continue

        df = df[df.index.isin( stride_dates )]
        df_list.append( df )

        print( f"    [{i}/{total}] {ticker}: {len(df)} rows processed" )

    result = pd.concat( df_list, axis=0 )

    # Switch sectors to numeric dummies
    sector_dummies = pd.get_dummies( result["sector"], prefix="sector" ).astype( "int8" )
    result = pd.concat( [result, sector_dummies], axis=1 )

    return result


def main():
    start_time = datetime.now()

    ensure_data_directory()

    # Get Raw Data
    print(f"Loading raw price data from {RAW_FILE}...")
    raw_df, volume_df = load_raw_data()

    if raw_df is None or raw_df.empty:
        print( "Raw data error. No data found..." )
        return

    print(f"  Raw data fo loaded for {len(raw_df.columns)} tickers, {len(raw_df)} rows\n")

    # Get Info Data
    print(f"Loading cached info from {INFO_FILE}...")
    info_map = load_info_map()
    print(f"  Loaded info for {len(info_map)} tickers\n")

    # Calculate returns from raw data
    print( "Building feature data...")
    data = build_features( raw_df, volume_df, info_map )
    data = data.reset_index()
    data = data.rename( columns={"Date": "date"} )
    print( "" )

    # Save data
    data.to_parquet( STOCK_FILE )
    
    # Summary statistics
    end_time = datetime.now()
    duration = ( end_time - start_time ).total_seconds()
    
    print( "\n" + "=" * 60 )
    print( "BUILD COMPLETE" )
    print( "=" * 60 )
    print( f"Output file: {STOCK_FILE}" )
    print( f"Total rows: {len( data ):,}" )
    print( f"Unique tickers: {data['ticker'].nunique()}" )
    print( f"Date range: {data['date'].min()} to {data['date'].max()}" )
    print( f"Duration: {duration:.1f} seconds" )
    print( f"Memory usage: {data.memory_usage( deep=True ).sum() / 1024**2:.1f} MB" )

if __name__ == "__main__":
    main()