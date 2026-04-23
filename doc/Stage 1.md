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

Each ticker symbol has multiple records, each with different dates. The date signifies the start of the window for each row. Each window contains data for each of the above data creation metrics.
