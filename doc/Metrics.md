# Metrics Documentation

This document describes all calculated metrics used in the stock feature pipeline.

---

## Base Features

### future_5y_return

#### Description

Target variable representing the stock's forward 5-year return.

#### Code

```python
df["future_5y_return"] = (
    df["price"].shift( -TIME_WINDOWS["5y"] )
    / df["price"] - 1
)
```

---

## Return Metrics

### ret_1d, ret_1w, ret_1m, ret_6m, ret_1y, ret_3y, ret_5y

#### Description

Percentage return over multiple historical lookback periods.

#### Code

```python
for label, span in TIME_WINDOWS.items():
    df[f"ret_{label}"] = df["price"].pct_change( span )
```

#### Time Windows

| Label | Trading Days |
|-------|-------|
| 1d | 1 |
| 1w | 5 |
| 1m | 21 |
| 6m | 126 |
| 1y | 251 |
| 3y | 753 |
| 5y | 1254 |

---

## Monotonicity Metrics

### monotonic_score_daily

#### Description

Measures how consistently a stock posts positive daily returns during the last year.

#### Code

```python
df["monotonic_score_daily"] = (
    df["price"]
    .pct_change()
    .gt(0)
    .rolling( TIME_WINDOWS["1y"] )
    .mean()
)
```

---

### monotonic_score

#### Description

Measures whether returns increase consistently across progressively longer time horizons.

#### Code

```python
df["monotonic_score"] = df.apply(
    add_monotonic,
    axis=1
)
```

#### Calculation Function

```python
def add_monotonic( row ):

    values = []

    for label in TIME_WINDOWS.keys():
        val = row.get( f"ret_{label}" )

        if pd.notna( val ):
            values.append( val )

    if len( values ) < 2:
        return 0.0

    score = sum(
        values[i] >= values[i-1]
        for i in range(1, len(values))
    )

    return score / ( len(values) - 1 )
```

---

## Drawdown Metrics

### 1y_drawdown

#### Description

Measures decline from the highest price reached within the past year.

#### Code

```python
rolling_max_1y = (
    df["price"]
    .rolling( TIME_WINDOWS["1y"] )
    .max()
)

df["1y_drawdown"] = (
    df["price"]
    / rolling_max_1y
    - 1
)
```

---

### 5y_drawdown

#### Description

Measures decline from the highest price reached within the past 5 years.

#### Code

```python
rolling_max_5y = (
    df["price"]
    .rolling( TIME_WINDOWS["5y"] )
    .max()
)

df["5y_drawdown"] = (
    df["price"]
    / rolling_max_5y
    - 1
)
```

---

## Trend Metrics

### 1y_trend

#### Description

Linear trend slope of log price over a rolling 1-year window.

#### Code

```python
log_price = np.log( df["price"] )

def slope(x):
    return np.polyfit(
        np.arange(len(x)),
        x,
        1
    )[0]

df["1y_trend"] = (
    log_price
    .rolling( TIME_WINDOWS["1y"] )
    .apply( slope, raw=True )
)
```

---

### 5y_trend

#### Description

Linear trend slope of log price over a rolling 5-year window.

#### Code

```python
df["5y_trend"] = (
    log_price
    .rolling( TIME_WINDOWS["5y"] )
    .apply( slope, raw=True )
)
```

---

## Sector Metrics

### sec_avg_ret_*

#### Description

Average sector return for all stocks within a sector on a given date.

#### Code

```python
"sec_avg_ret_1d": date_df["ret_1d"].mean(),
"sec_avg_ret_1w": date_df["ret_1w"].mean(),
"sec_avg_ret_1m": date_df["ret_1m"].mean(),
"sec_avg_ret_6m": date_df["ret_6m"].mean(),
"sec_avg_ret_1y": date_df["ret_1y"].mean(),
"sec_avg_ret_3y": date_df["ret_3y"].mean(),
"sec_avg_ret_5y": date_df["ret_5y"].mean(),
```

---

### sec_breadth_positive_1y

#### Description

Fraction of sector members with positive 1-year returns.

#### Code

```python
"sec_breadth_positive_1y":
    ( date_df["ret_1y"] > 0 ).mean()
```

---

### sec_breadth_positive_5y

#### Description

Fraction of sector members with positive 5-year returns.

#### Code

```python
"sec_breadth_positive_5y":
    ( date_df["ret_5y"] > 0 ).mean()
```

---

### sec_avg_1y_trend / sec_avg_5y_trend

#### Description

Average trend slope across sector members.

#### Code

```python
"sec_avg_1y_trend":
    date_df["1y_trend"].mean(),

"sec_avg_5y_trend":
    date_df["5y_trend"].mean(),
```

---

### sec_positive_1y_trend_pct

#### Description

Percentage of stocks with positive 1-year trends.

#### Code

```python
"sec_positive_1y_trend_pct":
    ( date_df["1y_trend"] > 0 ).mean()
```

---

### sec_avg_1y_drawdown / sec_avg_5y_drawdown

#### Description

Average drawdown across sector members.

#### Code

```python
"sec_avg_1y_drawdown":
    date_df["1y_drawdown"].mean()

"sec_avg_5y_drawdown":
    date_df["5y_drawdown"].mean()
```

---

### sec_strong_drawdown_resilience

#### Description

Percentage of stocks with less than 20% 1-year drawdown.

#### Code

```python
"sec_strong_drawdown_resilience":
    ( date_df["1y_drawdown"] > -0.2 ).mean()
```

---

### sec_avg_monotonic_score

#### Description

Average monotonic score across sector members.

#### Code

```python
"sec_avg_monotonic_score":
    date_df["monotonic_score"].mean()
```

---

### sec_high_monotonic_pct

#### Description

Percentage of stocks with monotonic score above 0.8.

#### Code

```python
"sec_high_monotonic_pct":
    ( date_df["monotonic_score"] > 0.8 ).mean()
```

---

### sec_ret_1y_dispersion / sec_ret_5y_dispersion

#### Description

Cross-sectional standard deviation of sector returns.

#### Code

```python
"sec_ret_1y_dispersion":
    date_df["ret_1y"].std()

"sec_ret_5y_dispersion":
    date_df["ret_5y"].std()
```

---

## Relative Comparison Metrics

### excess_ret_1y / excess_ret_5y

#### Description

Stock return minus sector average return.

#### Code

```python
df["excess_ret_1y"] = (
    df["ret_1y"]
    - df["sec_avg_ret_1y"]
)

df["excess_ret_5y"] = (
    df["ret_5y"]
    - df["sec_avg_ret_5y"]
)
```

---

### trend_vs_sector_1y / trend_vs_sector_5y

#### Description

Stock trend relative to sector average trend.

#### Code

```python
df["trend_vs_sector_1y"] = (
    df["1y_trend"]
    - df["sec_avg_1y_trend"]
)
```

---

### drawdown_rel_1y / drawdown_rel_5y

#### Description

Stock drawdown relative to sector average drawdown.

#### Code

```python
df["drawdown_rel_1y"] = (
    df["1y_drawdown"]
    - df["sec_avg_1y_drawdown"]
)
```

---

### excess_vs_market_1y

#### Description

Stock return relative to market average return.

#### Code

```python
df["excess_vs_market_1y"] = (
    df["ret_1y"]
    - df["sec_avg_ret_1y_market"]
)
```

---

### risk_adjusted_1y / risk_adjusted_5y

#### Description

Sector-relative excess return normalized by sector dispersion.

#### Code

```python
df["risk_adjusted_1y"] = (
    df["excess_ret_1y"]
    / df["sec_ret_1y_dispersion"]
        .replace(0, np.nan)
)
```

---

## Boolean Sector Signals

### sector_is_strong

#### Description

True if sector average 1-year return is positive.

#### Code

```python
df["sector_is_strong"] =
    df["sec_avg_ret_1y"] > 0
```

---

### sector_is_trending

#### Description

True if sector average trend is positive.

#### Code

```python
df["sector_is_trending"] =
    df["sec_avg_1y_trend"] > 0
```

---

### sector_high_breadth

#### Description

True if more than 60% of the sector has positive 1-year returns.

#### Code

```python
df["sector_high_breadth"] =
    df["sec_breadth_positive_1y"] > 0.6
```

---

## Composite Score

### quality_score

#### Description
Simple additive quality score combining return strength, trend strength, and drawdown strength.

#### Code

```python
df["quality_score"] = (
    (df["excess_ret_1y"] > 0).astype(int)
    + (df["trend_vs_sector_1y"] > 0).astype(int)
    + (df["drawdown_rel_1y"] > 0).astype(int)
)
```
