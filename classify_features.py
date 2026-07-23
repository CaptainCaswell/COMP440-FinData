FEATURES = [
    # Basic Info
    "beta_1y",
    "alpha_1y",
    # "log_market_cap",
    # "pe",

    # Price behavior
    "monotonic_score_daily",
    "monotonic_score",
    "1y_drawdown",
    "5y_drawdown",
    "1y_trend",
    "5y_trend",

    # Historical stock returns
    "ret_1d",
    "ret_1w",
    "ret_1m",
    "ret_6m",
    "ret_1y",
    "ret_3y",
    "ret_5y",

    # Return relative to SPY
    "excess_ret_1y",
    "excess_ret_5y",

    # Market Comparisons
    "excess_vs_market_1y",
    "trend_vs_market_1y",

    # Sector Identity (one hot)
    "sector_Basic Materials",
    "sector_Communication Services",
    "sector_Consumer Cyclical",
    "sector_Consumer Defensive",
    "sector_Energy",
    "sector_Financial Services",
    "sector_Healthcare",
    "sector_Industrials",
    "sector_Real Estate",
    "sector_Technology",
    "sector_Utilities",

    # Sector comparison
    "sec_avg_1y_trend",
    "sec_avg_5y_trend",
    "sec_avg_1y_drawdown",
    "sec_avg_5y_drawdown",

    # Sector Returns
    "sec_avg_ret_1d",
    "sec_avg_ret_1w",
    "sec_avg_ret_1m",
    "sec_avg_ret_6m",
    "sec_avg_ret_1y",
    "sec_avg_ret_3y",
    "sec_avg_ret_5y",

    # Sector participation
    "sec_breadth_positive_1y",
    "sec_breadth_positive_5y",

    # Sector trend
    "sec_positive_1y_trend_pct",
    "sec_positive_5y_trend_pct",

    # Sector stability
    "sec_strong_drawdown_resilience",

    # Sector monotonics
    "sec_avg_monotonic_score",
    "sec_avg_monotonic_score_daily",
    "sec_high_monotonic_pct",

    # Sector dispersion
    "sec_ret_1y_dispersion",
    "sec_ret_5y_dispersion",

    # Stock relative to sector
    "trend_vs_sector_1y",
    "trend_vs_sector_5y",
    "drawdown_rel_1y",
    "drawdown_rel_5y",

    # Risk Adjusted
    "risk_adjusted_1y",
    "risk_adjusted_5y",

    # Sector flags
    "sector_is_strong",
    "sector_is_trending",
    "sector_high_breadth",

    # Composite
    "quality_score"
    
    # # All Features below are not intended for use

    # # Misc
    # "date",
    # "ticker",
    # "price",
    # "sector",
    # "quote_type",
    # "sector_size",
    # "rows",
    # "sector_Unknown",
    
    # # Future
    # "future_ret_1d",
    # "future_excess_1d",
    # "future_ret_1w",
    # "future_excess_1w",
    # "future_ret_1m",
    # "future_excess_1m",
    # "future_ret_6m",
    # "future_excess_6m",
    # "future_ret_1y",
    # "future_excess_1y",
    # "future_ret_3y",
    # "future_excess_3y",
    # "future_ret_5y",
    # "future_excess_5y",

    # # SPY (S&P500 ETF)
    # "spy_ret_1d",
    # "spy_ret_1w",
    # "spy_ret_1m",
    # "spy_ret_6m",
    # "spy_ret_1y",
    # "spy_ret_3y",
    # "spy_ret_5y",

    # # Market Sector (all sectors)
    # "sec_breadth_positive_1y_market",
    # "sec_breadth_positive_5y_market",
    # "sec_avg_1y_trend_market",
    # "sec_avg_5y_trend_market",
    # "sec_positive_1y_trend_pct_market",
    # "sec_positive_5y_trend_pct_market",
    # "sec_avg_1y_drawdown_market",
    # "sec_avg_5y_drawdown_market",
    # "sec_strong_drawdown_resilience_market",
    # "sec_avg_monotonic_score_market",
    # "sec_avg_monotonic_score_daily_market",
    # "sec_high_monotonic_pct_market",
    # "sec_ret_1y_dispersion_market",
    # "sec_ret_5y_dispersion_market",
    # "sec_avg_ret_1d_market",
    # "sec_avg_ret_1w_market",
    # "sec_avg_ret_1m_market",
    # "sec_avg_ret_6m_market",
    # "sec_avg_ret_1y_market",
    # "sec_avg_ret_3y_market",
    # "sec_avg_ret_5y_market",
    # "sector_market",
    # "sector_size_market",
    # "rows_market",
]