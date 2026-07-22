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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Files
RAW_FILE = "data/raw_data.parquet"
INFO_FILE = "data/raw_info.parquet"
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
MIN_PRICE = 1.00
MAX_PRICE = 5000

INFO_MAP = None
MARKET_DAILY_RET = None
SPY_FUTURE_RETURNS = None
SPY_RETURNS = None
VOLUME_MAP = None
MIN_VOLUME = 10000
MAX_CHANGE = 1.00

def init_worker(info_map, market_daily_ret, spy_future_returns, spy_returns):
    global INFO_MAP, MARKET_DAILY_RET, SPY_FUTURE_RETURNS, SPY_RETURNS, VOLUME_MAP
    
    INFO_MAP = info_map
    MARKET_DAILY_RET = market_daily_ret
    SPY_FUTURE_RETURNS = spy_future_returns
    SPY_RETURNS = spy_returns

def ensure_data_directory():
    # Create bath if it doesn't exist
    Path( "data" ).mkdir( exist_ok=True )

def load_raw_data() -> tuple:
    if not os.path.isfile( RAW_FILE ):
        raise FileNotFoundError(f"{RAW_FILE} not found. Run the raw data download step first.")
    
    # Split into prices and volume
    raw_df = pd.read_parquet( RAW_FILE )
    return raw_df["close"], raw_df["volume"]

def load_info_map() -> dict:
    if not os.path.isfile( INFO_FILE ):
        raise FileNotFoundError(f"{INFO_FILE} not found. Run get_raw_info.py first.")
    info_df = pd.read_parquet( INFO_FILE )
    info_map = {}
    for _, row in info_df.iterrows():
        info_map[row["ticker"]] = {
            "sector": row["sector"] if pd.notna(row["sector"]) else "Unknown",
            "shares_outstanding": row["shares_outstanding"],
            "trailing_pe": row["trailing_pe"],
        }
    return info_map

def get_info( ticker: str, info_map: dict ) -> dict:
    return info_map.get(ticker, {
        "sector": "Unknown",
        "shares_outstanding": None,
        "trailing_pe": None,
    } )

def rolling_slope(y: pd.Series, window: int) -> pd.Series:
    idx = y.index

    y = y.values.astype(float)

    x = np.arange(window)
    x_mean = x.mean()
    x2 = np.sum((x - x_mean) ** 2)

    # Precompute rolling sums
    y_sum = np.convolve(y, np.ones(window), mode="valid")
    y_x_sum = np.convolve(y, x[::-1], mode="valid")

    # slope formula:
    # (n * Σ(xy) - Σx Σy) / (n * Σ(x^2) - (Σx)^2)
    n = window
    sum_x = np.sum(x)

    slope = (n * y_x_sum - sum_x * y_sum) / ( n* x2 )

    # pad to match original length
    return pd.Series(np.concatenate([np.full(window - 1, np.nan), slope]), index=idx)

def process_ticker( ticker: str, series: pd.Series, volume: pd.Series ) -> Optional[pd.DataFrame]:
    min_length = LOOKBACK_DAYS + FUTURE_DAYS + MIN_WINDOWS

    series = series.dropna().sort_index()

    if len( series ) < min_length:
        return None

    df = pd.DataFrame( index=series.index )
    df["ticker"] = ticker
    df["price"] = series

    # Remove improbable prices
    df = df[( df["price"] >= MIN_PRICE ) & ( df["price"] <= MAX_PRICE )]

    # Remove tickers without enough data
    if len( df ) < min_length:
        return None

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

    # Alpha/Beta
    stock_ret = df["price"].pct_change()
    market_ret_aligned = MARKET_DAILY_RET.reindex( df.index )

    rolling_cov = stock_ret.rolling(TIME_WINDOWS["1y"], min_periods=50).cov(market_ret_aligned)
    rolling_var = market_ret_aligned.rolling(TIME_WINDOWS["1y"], min_periods=50).var()

    df["beta_1y"] = rolling_cov / rolling_var
    df["alpha_1y"] = stock_ret - df["beta_1y"] * market_ret_aligned

    # Info
    info = get_info( ticker, INFO_MAP )
    df["sector"] = info["sector"]

    # Market Cap
    shares = info["shares_outstanding"]

    if shares and shares > 0:
        market_cap = df["price"] * shares
        df["log_market_cap"] = np.log( market_cap )
    else:
        None

    # PE Ratio
    trailing_pe = info["trailing_pe"]
    current_price = series.iloc[-1]

    if trailing_pe and trailing_pe > 0:
        eps_now = current_price / trailing_pe
        df["pe"] = df["price"] / eps_now
    else:
        df["pe"] = np.nan

    # Future Price
    for label, span in TIME_WINDOWS.items():
        # Return windows
        df[f"future_ret_{label}"] = df["price"].shift( -span ) / df["price"] - 1

        # Market return
        spy_future = SPY_FUTURE_RETURNS[label].reindex(df.index)

        # Excess return vs SPY
        df[f"future_excess_{label}"] = ( df[f"future_ret_{label}"] - spy_future )

    # Returns
    for label, span in TIME_WINDOWS.items():
        df[f"ret_{label}"] = df["price"].pct_change( span )

    # SPY benchmark returns
    for label in TIME_WINDOWS.keys():
        df[f"spy_ret_{label}"] = SPY_RETURNS[label].reindex(df.index)

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

    # Faster
    df["1y_trend"] = rolling_slope( log_price, TIME_WINDOWS["1y"] )
    df["5y_trend"] = rolling_slope( log_price, TIME_WINDOWS["5y"] )

    # Clean
    df = df.iloc[LOOKBACK_DAYS:-FUTURE_DAYS]

    # Drop if core stats not available
    core_cols = ["price", "beta_1y", "alpha_1y", "future_ret_1y", "future_ret_5y", "1y_trend", "5y_trend", "monotonic_score_daily"]
    df = df.dropna( subset=core_cols )

    return df

def add_monotonic( row: dict ) -> float:
    # Calculate monotonic score (how consistently returns increase for each time window)
    # Args:
    #     row: Dictionary containing ticker return values for each window
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

def build_features( close: pd.DataFrame, volume: pd.DataFrame, info_map: dict ) -> pd.DataFrame:
    # Build data feature matrix from raw price data
    # Args:
    #     close: DataFrame with closeing prices
    # Returns: Dataframe with calculated features

    # Daily market return
    daily_returns = close.pct_change()
    market_daily_ret = daily_returns.median(axis=1)

    spy_price = close["SPY"]

    spy_future_returns = {}

    for label, span in TIME_WINDOWS.items():
        spy_future_returns[label] = ( spy_price.shift( -span ) / spy_price - 1 )

    spy_returns = {}

    for label, span in TIME_WINDOWS.items():
        spy_returns[label] = spy_price.pct_change( span )

    stride_dates = set( close.index[::WINDOW_STRIDE] )

    df_list = []
    total = len( close.columns )
    
    n_workers = max( mp.cpu_count() - 1, 1 )
    print( f"Processing {total} tickers using {n_workers} processes..." )
    
    # min_length = LOOKBACK_DAYS + FUTURE_DAYS + MIN_WINDOWS
    
    with ProcessPoolExecutor( max_workers=n_workers, initializer=init_worker, initargs=(info_map, market_daily_ret, spy_future_returns, spy_returns ) ) as executor:
        futures = {
            executor.submit( process_ticker, ticker, close[ticker], volume[ticker] ): ticker
            for ticker in close.columns
        }
    

        for i, future in enumerate( as_completed( futures ), 1 ):
            ticker = futures[future]

            try:
                df = future.result()
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

def build_comparisons( stock_data: pd.DataFrame, sector_data: pd.DataFrame ) -> pd.DataFrame:
    # Split out market data and rename column heading
    market_data = sector_data[ sector_data["sector"] == "MARKET"].rename( columns=lambda c: c + "_market" if c not in ["date"] else c )
    sector_data = sector_data[sector_data["sector"] != "MARKET"]

    # Merge sector and market data with stocks
    df = stock_data.merge( sector_data, on=["date", "sector"], how="left" )
    df = df.merge( market_data, on=["date"], how="left" )

    # Sector relative features
    df["excess_ret_1y"] = df["ret_1y"] - df["spy_ret_1y"]
    df["excess_ret_5y"] = df["ret_5y"] - df["spy_ret_5y"]

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

# TODO remove if not repurposed
def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure sorted (VERY important for rolling calculations)
    df = df.sort_values(["ticker", "date"]).copy()

    # Market return = cross-sectional mean per date
    market_ret = df.groupby("date")["ret_1d"].mean()

    output = []

    for ticker, stock_df in df.groupby("ticker"):
        stock_df = stock_df.copy()

        # Stock return
        stock_ret = np.log( stock_df["price"] ).diff()

        # Align market return to this ticker's dates
        market_aligned = market_ret.reindex(stock_df["date"].values).values

        aligned = pd.DataFrame({
            "stock": stock_ret.values,
            "market": market_aligned
        })

        # Rolling beta
        rolling_cov = ( aligned["stock"].rolling(251, min_periods=50).cov(aligned["market"]) )

        rolling_var = ( aligned["market"].rolling(251, min_periods=50).var() )

        beta = rolling_cov / rolling_var

        stock_df["beta_1y"] = beta.values

        # Alpha
        stock_df["alpha_1y"] = ( aligned["stock"].values - stock_df["beta_1y"].values * aligned["market"].values )

        output.append(stock_df)

    return pd.concat(output, axis=0)

def main():
    # Get starting time
    start_time = datetime.now()
    print( f" Starting data at {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

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
    print( "=" * 60 )
    print( "Building feature data")
    print( "=" * 60 )
    stock_data = build_features( raw_df, volume_df, info_map )
    stock_data = stock_data.reset_index()
    stock_data = stock_data.rename( columns={"Date": "date"} )
    print( "" )

    # Add market features
    # Moved to build features
    # stock_data = add_market_features( stock_data )

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