# COMP 440 - FinData

## Notes

### Ideas

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



```python
rolling_max = df["Close"].cummax()
drawdown = df["Close"] / rolling_max - 1
max_drawdown = drawdown.min()
```

#### % above 200 day moving average

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
