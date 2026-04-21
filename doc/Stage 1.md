# COMP 440 - FinData

## Stage 1 - Data Collection

### Training Data

The source for all ticker symbols is a JSON file from the SEC that contains all (10,368) publicly traded US companies. Not all of these are current or listed on Yahoo Finance, so some will have to be removed.

In order to account for events effecting the stock market, data will be taken from a range of periods over the last 30 years.

### Test Data

Selected stocks will be chosen to be used as test data.

### Comparison Data

The SP500 will be used to benchmark any algorithms created. The goal is to have returns consistently higher than the SP500.

## Stage 2 - Data Creation

### Ideas

In order to evaluate stocks, several metrics will be calculated from the downloaded data.

#### Returns

* 1 day
* 5 day
* 1 month
* 6 month
* 1 year
* 3 years
* 5 years

#### Monotonic

Point per period where it's monotonic?

% return higher than last returns
Raw returns higher than all other returns

#### Max Drawdown

A measure of the stocks worst crash.

```python
rolling_max = df["Close"].cummax()
drawdown = df["Close"] / rolling_max - 1
max_drawdown = drawdown.min()
```

#### % above 200 day moving average

Measures how often a stock is beating it's long term trend.

```python
ma200 = df["Close"].rolling(200).mean()
above = df["Close"] > ma200
percent_above_200ma = above.mean()
```

#### Trend Slope

```python
import numpy as np

x = np.arange(len(df))
y = df["Close"].values

slope = np.polyfit(x, y, 1)[0]
```

#### Volume Stability

```python
vol_stability = df["Volume"].std() / df["Volume"].mean()
```
