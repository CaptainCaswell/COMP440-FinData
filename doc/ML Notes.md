# Notes

## Overall

Best code doesn't need comments.

Simple is better. Even extra steps/code to make things more readable are worth a little bit of overhead. Can be simplified after initial testing if efficiency is required.

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
