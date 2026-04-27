# COMP 440 - FinData

## Data Collection

### Training Data

The source for all ticker symbols is a JSON file from the SEC that contains all (10,368) publicly traded US companies. Not all of these are current or listed on Yahoo Finance, so some will have to be removed.

In order to account for events effecting the stock market, data will be taken from a range of periods over the last 30 years.

### Test Data

Selected stocks will be chosen to be used as test data.

### Comparison Data

The SP500 will be used to benchmark any algorithms created. The goal is to have returns consistently higher than the SP500.

## Data Creation

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
* 5 years after

#### Monotonic

Comparing each window, this checks to see if the returns were higher than the last window. A score of 1 is given for each period, then normalized to a range of 0 to 1.

As this is only looking at each window, there is a lot of "noise" in these results. It may make more sense to switch to looking at day to day prices, comparing how often there is growth versus how often there is loss (1 =  always increasing, 0 =  always dropping).

#### Max Drawdown

This is calculated on a 1 year and 5 year window for each row. It compares each days price to the stocks maximum price that has been seen at that point. This finds the largest drop in stock price for that window. This is normalized by using a percentage.

#### Trend Slope

Again with a 1 year and 5 year window, this uses linear regression to find the best fit slope. Logarithmic prices were used to normalize the data.

## Data Structure

Each ticker ticker symbol has multiple records, each with different dates. The date signifies the start of the window for each row. Each window contains data for each of the above data creation metrics. Below is a table with an example record:

| ticker | date | price | future_5_y_return | ret_1d | ret_1w | ret_1m | ret_6m | ret_1y | ret_3y | ret_5y | monotonic_score | monotonic_score_daily | 1y_drawdown | 5y_drawdown | 1y_drawdown | 1y_trend | 5y_trend |
| ------ | ---- | ----- | ----------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | --------------- | --------------------- | ----------- | ----------- | ----------- | -------- | -------- |
| AIRI   | 2016-04-19T00:00.000Z | 55.5 | -0.765765765765765 | -0.034782608695652195 | -0.07499999999999996 | -0.05932203389830504 | -0.3629188901462078 | -0.43912647244320324 | 0.07268248035527947 | 1.1874399023146447 | 0.5 | 0.2753391859537111 | -0.44596720280862545 | -0.678423176328976 | -0.0022368426433517054 | 0.0011264210263270099 |

| Column | Description |
| ------ | ----------- |
| ticker | The ticker symbol of the stock. Repeated for each date window. |
| date | The start date of each window. Dates repeated for different stocks |
| price | The price of a single share |
| future_5y_return | The percentage return for a stock in a 5 year window after main 5 year window |
| ret_?? | The percentage returns for a given time period |
| monotonic_score | Percentage of periods that are increasing (monotonic) |
| monotonic_score_daily | Percentage of days that are increasing (monotonic) |
| 1y_drawdown | The max drawdown in a 1 year window |
| 5y_drawdown | The max drawdown in a 5 year window |
| 1y_trend | The best fit slope in a 1 year window |
| 5y_trend | The best fit slope in a 5 year window |
