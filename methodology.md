# Methodology

## 1. Research Objective

This project studies whether medium-term cross-sectional momentum provides a stable and economically meaningful predictive signal across U.S. equities.

The analysis focuses on three predefined momentum specifications:

- 6-1 momentum
- 9-1 momentum
- 12-1 momentum

The objective is not to maximize historical backtest performance, but to evaluate whether momentum signals are robust across different time periods and portfolio constructions.


## 2. Stock Universe

The initial universe contains 200 U.S. large-cap equities.

A data-quality filter is applied before the factor research begins.

Stocks are excluded if:

- More than 5% of monthly price observations are missing
- Internal price gaps exist after the stock first appears in the dataset
- No valid price exists for the latest completed month

After filtering, the final research universe contains 193 stocks.

The universe is fixed throughout the study.


## 3. Data

Monthly adjusted price data are obtained from Yahoo Finance using the `yfinance` Python package.

The study uses approximately ten years of monthly observations.

The current unfinished month is excluded from the analysis.

Missing prices are not forward-filled.

This avoids artificially creating zero-return observations when a market price is unavailable.


## 4. Momentum Signal Construction

For a formation period \(k\), momentum is defined as:

\[
Momentum_t = \frac{P_{t-1}}{P_{t-k}} - 1
\]

where:

- \(P_{t-1}\) is the price one month before the ranking date
- \(P_{t-k}\) is the price at the beginning of the formation window

The most recent month is skipped in order to reduce exposure to short-term reversal effects.

Three specifications are evaluated:

- 6-1
- 9-1
- 12-1


## 5. Portfolio Construction

At each monthly rebalance date, stocks are ranked by their momentum score.

The baseline long-short portfolio:

- Longs the top 20% of stocks
- Shorts the bottom 20% of stocks
- Uses equal weights within each leg
- Rebalances monthly

The long leg has a total weight of +100%.

The short leg has a total weight of -100%.

The portfolio therefore has approximately zero net exposure and 200% gross exposure before transaction costs.


## 6. Return Timing

The signal is constructed using information available at time \(t\).

Portfolio weights are determined at time \(t\).

Performance is then measured using the realized return during \(t+1\).

Future return availability is not used when selecting stocks.

This is intended to avoid look-ahead bias in portfolio formation.


## 7. Transaction Costs

Transaction costs are modeled as:

\[
Cost_t =
Turnover_t
\times
\frac{Cost\ in\ bps}{10000}
\]

Turnover is calculated as the absolute change in portfolio weights between two consecutive rebalancing dates.

The baseline assumption is:

- 10 bps per unit of turnover

Sensitivity tests are also conducted at:

- 0 bps
- 5 bps
- 10 bps
- 25 bps


## 8. Chronological Research Split

The sample is divided chronologically rather than randomly.

The split is:

- Development: 50%
- Validation: 25%
- Historical Evaluation: 25%

The development period is used to study the strategy and understand its behavior.

The validation period is used to compare the predefined 6-1, 9-1 and 12-1 specifications.

The specification with the highest validation Sharpe ratio is selected.

The final historical evaluation period is used to study how the selected specification behaves in a later market period.

Because this evaluation period was already observed during earlier iterations of the project, it is not described as a fresh untouched out-of-sample test.


## 9. Performance Metrics

The main portfolio metrics include:

- Mean monthly return
- Annualized return
- Annualized volatility
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Total return
- Win rate
- Average turnover


## 10. Rank IC

Cross-sectional predictive power is evaluated using Spearman Rank Information Coefficient.

For each month:

\[
IC_t =
Corr_{Spearman}
(
Momentum_t,
Return_{t+1}
)
\]

A positive Rank IC indicates that stocks with higher momentum scores tend to rank higher in subsequent returns.

The study reports:

- Mean Rank IC
- Median Rank IC
- Rank IC volatility
- Annualized ICIR
- Positive IC rate


## 11. Quintile Portfolio Analysis

Each month, stocks are divided into five portfolios based on momentum rank:

- Q1: weakest momentum
- Q2
- Q3
- Q4
- Q5: strongest momentum

The analysis evaluates whether future returns increase monotonically from Q1 to Q5.

A stable momentum factor would generally be expected to show higher subsequent returns in stronger momentum quintiles.

The Q5 minus Q1 spread is also evaluated.


## 12. Long / Short Leg Decomposition

The long-short portfolio is decomposed into:

- Long winner portfolio
- Short loser portfolio

The raw future returns of loser stocks are also measured.

This helps determine whether strategy performance is driven by:

- Strong winner continuation
- Weak loser continuation
- Loser-stock rebounds
- Or both portfolio legs


## 13. Benchmark Comparison

The selected momentum strategy is compared with:

### Long-Short Momentum

Long top 20% and short bottom 20%.

### Long-Only Momentum

Long top 20% only.

### Equal-Weight Universe

Equal-weight allocation across the eligible stock universe.

The benchmark comparison helps distinguish momentum-specific performance from broad equity-market performance.


## 14. Bootstrap Inference

A centered circular-block bootstrap is used to evaluate statistical uncertainty.

Returns are first centered by subtracting the observed sample mean, producing a zero-mean null distribution.

Contiguous blocks of monthly returns are then resampled.

Circular sampling allows blocks near the end of the sample to wrap around to the beginning.

The baseline setup uses:

- 5,000 bootstrap simulations
- Block size of 3 months

The analysis reports:

- Observed mean return
- Observed Sharpe ratio
- One-sided p-value for positive mean return
- One-sided p-value for positive Sharpe
- 95% bootstrap confidence intervals


## 15. Multiple-Testing Adjustment

Three momentum specifications are compared:

- 6-1
- 9-1
- 12-1

Because multiple candidate specifications are evaluated, Bonferroni adjustment is applied to bootstrap p-values.

This reduces the risk of interpreting the strongest result among several tested specifications as statistically meaningful purely because of parameter search.


## 16. Main Research Findings

The 6-1 specification achieved the highest validation Sharpe ratio.

However, its historical evaluation performance was negative.

Further diagnostics showed:

- The long winner portfolio remained profitable
- The short loser leg generated substantial losses
- Past loser stocks experienced strong subsequent rebounds
- Rank IC was unstable across research periods
- Quintile ordering was not consistently monotonic
- Long-only momentum was profitable but did not outperform the equal-weight universe
- Transaction costs worsened performance but were not the main cause of long-short underperformance

These results suggest that the momentum signal was weak and regime-dependent over the studied sample.


## 17. Limitations

This research has several important limitations.

### Survivorship and Selection Bias

The stock universe is based on a fixed set of current equities rather than a true historical point-in-time constituent database.

This can introduce survivorship and selection bias.

### Data Source

Yahoo Finance is convenient for research and education but is not an institutional point-in-time market database.

### Delistings

Delisting returns are not explicitly modeled.

### Short-Selling Frictions

The analysis does not explicitly model:

- Borrow availability
- Stock-specific borrow fees
- Short-sale constraints

### Execution

Monthly closing prices simplify:

- Bid-ask spreads
- Market impact
- Execution timing
- Intramonth price movement

### Evaluation Reuse

The historical evaluation period has already been observed during the research process.

It should therefore be interpreted as exploratory historical evidence rather than a fresh confirmatory out-of-sample test.


## 18. Research Philosophy

The project prioritizes robustness and transparency over producing an artificially optimized backtest.

Negative or unstable results are treated as research findings rather than hidden through repeated parameter tuning.
