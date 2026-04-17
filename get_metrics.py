import yfinance as yf
import pandas as pd
import random
import time
import json
import os

OUTPUT = "stock_metrics.parquet"
TICKERS = "company_tickers.json"
MIN_VALUATION = 10_000_000 # Drops stocks below threashold

TESTING = True # Limits symbols, debugging messages
TICKER_COUNT = 20 # Number of symbols loaded (random sampling)
WAIT = 0.25 # Check actual API limits

def get_metrics( symbol ):
    try:
        ticker = yf.Ticker( symbol )
        info = ticker.get_info()

        market_cap = clean_float(info.get("marketCap"))

        # Drop if under valuation minimum
        if market_cap is None or market_cap < MIN_VALUATION:
            return None

        beta = clean_float(info.get("beta"))
        ps = clean_float(info.get("priceToSalesTrailing12Months"))
        pe = clean_float(info.get("trailingPE"))
        eps = clean_float(info.get("trailingEps"))
        vol = clean_float(info.get("averageVolume"))
        target = clean_float(info.get("targetMeanPrice"))

        metrics = [beta, ps, pe, eps, vol, target]

        # Drop if no metrics returned
        if all( metric is None for metric in metrics ):
            return None

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
        if x in [None, "Infinity"]:
            return None # Change to max float?
        return float( x )
    except:
        return None

def get_symbols():
    with open( TICKERS, "r" ) as file:
        data = json.load( file )

    return [stock["ticker"] for stock in data.values() ]

def main():

    symbols = get_symbols()
    print(f"{len( symbols )} tickers have been imported.")

    data = []

    # Reduce to limited number of symbols if testing
    if ( TESTING ):
        symbols = random.sample( symbols, TICKER_COUNT )
        print(f"Tickers reduced to {TICKER_COUNT} for testing")

    for symbol in symbols:
        metrics = get_metrics( symbol )
        
        # Don't append if None
        if metrics:
            data.append( metrics )
        
        time.sleep( WAIT )  # API ban prevention
        if ( TESTING ):
            print(f"{symbol} imported")

    df = pd.DataFrame( data )
    df = df.set_index( "ticker" )

    # Save to Parquet
    df.to_parquet( OUTPUT, engine="pyarrow" )

    print(f"{ len( data ) } tickers imported.")
    print(f"Saved to {OUTPUT}")

if __name__ == "__main__":
    main()