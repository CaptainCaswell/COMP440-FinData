# COMP440-FinData — Data Dictionary

Covers every column produced by `build_data.py` and stored in `data.parquet`.

## Identity / metadata columns

| Column | Description |
| --- | --- |
| `ticker` | Stock symbol |
| `date` | Observation date (strided every `WINDOW_STRIDE`=5 trading days) |
| `sector` | Sector name from yfinance `.info` (or "Unknown" if unavailable) |

## Price / size / valuation

| Column | Description |
| --- | --- |
| `price` | Raw closing price on this date |
| `log_market_cap` | `log(price × shares_outstanding)` or NaN if shares_outstanding unavailable. Imputed as median value if NaN. |
| `pe` | `Price / (current_price ÷ trailing_pe)` A flat-EPS PE proxy applied across history. NaN if trailing_pe missing/≤0 |

## Returns

| Column | Description |
| --- | --- |
| `ret_1d`, `ret_1w`, `ret_1m`, `ret_6m`, `ret_1y`, `ret_3y`, `ret_5y` | Trailing % price change over each window (1 day / week / month / 6mo / 1yr / 3yr / 5yr) |
| `future_5y_return` | Forward-looking 5-year return. **the clustering target**, never a feature |

## Risk / trend / behavior

| Column | Description |
| --- | --- |
| `beta_1y` | Rolling 1-year beta vs. median-based market daily return |
| `alpha_1y` | Rolling 1-year alpha (stock return minus beta-scaled market return) |
| `monotonic_score_daily` | Fraction of up-days over trailing 1 year |
| `monotonic_score` | Consistency of return acceleration across the 7 return windows (0–1) |
| `1y_drawdown`, `5y_drawdown` | Current price vs. trailing 1yr/5yr max, as a % |
| `1y_trend`, `5y_trend` | Rolling slope of log(price) over 1yr/5yr window (per-day log-price slope) |

## Sector one-hot dummies

| Column | Description | Used in clustering |
| --- | --- |
| `sector_Basic Materials` … `sector_Utilities` (11 real sectors) | 1 if stock belongs to this sector, else 0 |
| `sector_Unknown` | 1 if sector could not be determined (missing/failed info fetch) |

## Dataset-size proxies (per date, per sector / market)

| Column | Description |
| --- | --- |
| `sector_size`, `rows` | Number of unique tickers / rows in this sector on this date |
| `sector_size_market`, `rows_market` | Same, but market-wide |

## Sector aggregates (`sec_*`, computed per sector per date)

| Column | Description |
| --- | --- |
| `sec_avg_ret_1d/1w/1m/6m/1y/3y/5y` | Mean of that return window across all stocks in the sector |
| `sec_breadth_positive_1y/5y` | % of sector stocks with positive 1yr/5yr return |
| `sec_avg_1y_trend`, `sec_avg_5y_trend` | Mean trend across sector |
| `sec_positive_1y_trend_pct`, `sec_positive_5y_trend_pct` | % of sector stocks with positive trend |
| `sec_avg_1y_drawdown`, `sec_avg_5y_drawdown` | Mean drawdown across sector |
| `sec_strong_drawdown_resilience` | % of sector stocks with 1yr drawdown better than -20% |
| `sec_avg_monotonic_score`, `sec_avg_monotonic_score_daily` | Mean monotonicity across sector |
| `sec_high_monotonic_pct` | % of sector stocks with monotonic_score > 0.8 |
| `sec_ret_1y_dispersion`, `sec_ret_5y_dispersion` | Std dev of sector's 1yr/5yr returns (spread) |

## Market aggregates (`sec_*_market`, same as above but market-wide, not sector-limited)

| Column | Description |
| --- | --- |
| `sec_avg_ret_*_market`, `sec_breadth_positive_*_market`, `sec_avg_*_trend_market`, `sec_positive_*_trend_pct_market`, `sec_avg_*_drawdown_market`, `sec_strong_drawdown_resilience_market`, `sec_avg_monotonic_score*_market`, `sec_high_monotonic_pct_market`, `sec_ret_*_dispersion_market` | Same definitions as sector aggregates, computed market-wide instead of per-sector |

## Relative / composite features

| Column | Description |
| --- | --- |
| `excess_ret_1y`, `excess_ret_5y` | `ret_1y`/`ret_5y` minus sector average |
| `trend_vs_sector_1y`, `trend_vs_sector_5y` | Trend minus sector average trend |
| `drawdown_rel_1y`, `drawdown_rel_5y` | Drawdown minus sector average drawdown |
| `excess_vs_market_1y` | `ret_1y` minus market average |
| `trend_vs_market_1y` | Trend minus market average trend |
| `risk_adjusted_1y`, `risk_adjusted_5y` | Excess return divided by sector return dispersion |
| `sector_is_strong`, `sector_is_trending`, `sector_high_breadth` | Boolean flags derived from sector aggregates |
| `quality_score` | Sum of 3 boolean composite checks (0-3) |
