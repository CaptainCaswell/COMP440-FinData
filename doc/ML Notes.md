# Notes

## Overall

Best code doesn't need comments.

Simple is better. Even extra steps/code to make things more readable are worth a little bit of overhead. Can be simplified after initial testing if efficiency is required.

Don't plan to clean code later, it is way more difficult than cleaning now.

## Clustering

### Outliers

Noticed mean and median VERY different, found outliers caused by cheap stocks gaining relatively huge %. Removed stocks before $1, also filtered gains of more than 100% and less than 10,000 trades.

Discovered notebooks

### Mean to Median, Beta

Looking at clusters showed a small cluster, found to all be 2020 and 2021 stocks (COVID). Thought it was due to bad data, but clustering has simply found the rebounding stocks from covid crash.

Switched from Mean Market Data to Median - This exposed there also being large outliers. Massive increases (+1,000,000%) with no trade volume.

Started checking time ranges for results.

### Feature numbers

Reducing features actually improved grouping. Come back and test this later.

### Heavy on Sector/Market

First clusters were based almost entirely on market and sector aggregate features, not anything about the individual stock.

"stocks measured near a market bottom subsequently had strong 5-year returns."

### Known data-quality notes (for reference)

- `1y_trend`/`5y_trend` had a sign-flip + scaling bug (fixed) — verify any historical exports predate the fix.
- `MIN_PRICE` filter currently 1.00 (was regressed to 0.10, restored).
- Split-artifact rejection via `MAX_CHANGE`/`MIN_VOLUME` — tickers with a large single-day move at low volume above `MIN_PRICE` are dropped entirely (see `process_ticker` skip-reason logging).
- Chronic penny stocks (≥50% of history below `MIN_PRICE`) are not currently auto-excluded — decide if still desired.
- `market_daily_ret` computed via **median** (not mean) across all tickers per date — fixed a COVID-2020 regime-artifact cluster.
- Best cluster driven mostly by `_market`/`sec_*` aggregate features is largely a **mean-reversion-after-drawdown** effect, concentrated in 2020 — treat with caution as an "algorithm-ready" signal.
- Best cluster from **pure stock-level features only** (17 features, no sector/market aggregates) shows a smaller effect (~61% trimmed mean) but is well-distributed across 2016–2021 and 1,807+ tickers — more likely to generalize.
