"""
momentum_research.py
--------------------
Cross-sectional momentum factor research and validation.

Research design:
  - Fixed 200-stock U.S. equity universe with strict data-quality filtering
  - 6-1, 9-1, and 12-1 cross-sectional momentum signals
  - Chronological development / validation / historical evaluation split
  - Rank IC and quintile diagnostics
  - Long/short leg decomposition and benchmark comparison
  - Transaction-cost sensitivity and centered circular-block bootstrap

Important:
  The historical evaluation period has already been observed in earlier
  research iterations, so it is not described as an untouched OOS test.

The calculations in this file are unchanged from the V2 research notebook;
this version only reorganizes formatting and comments for GitHub readability.
"""


# ── Imports ─────────────────────────────────────────────────────────────────
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
warnings.filterwarnings('ignore')

# ── Universe — fixed 200-stock sample ───────────────────────────────────────
UNIVERSE = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'AVGO', 'TSLA', 'BRK-B', 'LLY',
    'JPM', 'V', 'WMT', 'MA', 'XOM', 'COST', 'ORCL', 'NFLX', 'JNJ', 'HD',
    'PG', 'ABBV', 'BAC', 'KO', 'PLTR', 'CRM', 'PM', 'CVX', 'CSCO', 'IBM',
    'GE', 'WFC', 'ABT', 'UNH', 'MCD', 'GS', 'MRK', 'AXP', 'CAT', 'NOW',
    'ISRG', 'PEP', 'TMO', 'QCOM', 'ACN', 'INTU', 'TXN', 'AMGN', 'AMD', 'LIN',
    'MS', 'RTX', 'SPGI', 'DHR', 'BKNG', 'BLK', 'AMAT', 'PGR', 'NEE', 'TJX',
    'HON', 'LOW', 'SYK', 'SCHW', 'VRTX', 'PANW', 'C', 'ADBE', 'GILD', 'ETN',
    'ANET', 'DE', 'BSX', 'PLD', 'ADI', 'CB', 'MMC', 'MDT', 'LRCX', 'FI',
    'AMT', 'SO', 'ICE', 'WM', 'SHW', 'KLAC', 'SBUX', 'APH', 'MO', 'DUK',
    'CI', 'UPS', 'ELV', 'EQIX', 'MCO', 'CME', 'INTC', 'TT', 'PH', 'CMG',
    'GD', 'SNPS', 'CDNS', 'AON', 'NOC', 'MCK', 'PNC', 'USB', 'CL', 'ECL',
    'MAR', 'ITW', 'ORLY', 'WELL', 'AJG', 'ZTS', 'MMM', 'EMR', 'RSG', 'CTAS',
    'FDX', 'REGN', 'APD', 'CSX', 'NSC', 'GM', 'F', 'PCAR', 'NXPI', 'MPC',
    'PSX', 'OXY', 'EOG', 'SLB', 'COP', 'KMI', 'OKE', 'VLO', 'BKR', 'WMB',
    'SPG', 'O', 'PSA', 'DLR', 'CCI', 'VICI', 'CBRE', 'AVB', 'EQR', 'EXR',
    'AWK', 'AEP', 'SRE', 'D', 'EXC', 'XEL', 'ED', 'PEG', 'WEC', 'DTE',
    'CARR', 'JCI', 'URI', 'AEE', 'CMS', 'ES', 'ETR', 'FE', 'PPL', 'LNT',
    'MDLZ', 'KHC', 'GIS', 'ADM', 'HSY', 'SJM', 'CPB', 'CAG', 'HRL', 'MKC',
    'EL', 'CLX', 'CHD', 'KMB', 'STZ', 'MNST', 'TAP', 'DG', 'DLTR', 'ROST',
    'NKE', 'LULU', 'YUM', 'DPZ', 'DRI', 'BBY', 'EBAY', 'ETSY', 'APTV', 'PHM',
]
print('Original universe:', len(UNIVERSE))
print('Unique tickers:', len(set(UNIVERSE)))
assert len(UNIVERSE) == 200
assert len(set(UNIVERSE)) == 200

# ── Configuration ───────────────────────────────────────────────────────────
PERIOD = '10y'
CANDIDATES = [6, 9, 12]
SKIP_MONTHS = 1
TOP_PCT = 0.2
BOTTOM_PCT = 0.2
BASE_COST_BPS = 10
DEV_FRAC = 0.5
VAL_FRAC = 0.25
MAX_MISSING_RATIO = 0.05
MIN_STOCKS = 100
COST_LEVELS = [0, 5, 10, 25]
N_BOOTSTRAP = 5000
BLOCK_SIZE = 3
RESULT_DIR = Path('momentum_v2_results')
RESULT_DIR.mkdir(exist_ok=True)


# ── Data download ───────────────────────────────────────────────────────────
def download_prices(tickers, period='10y', batch_size=50):
    print('\n' + '=' * 70)
    print('DOWNLOADING RAW DATA')
    print('=' * 70)
    batches = []
    for start in range(0, len(tickers), batch_size):
        batch = tickers[start:start + batch_size]
        print(f'Downloading {start + 1}-{min(start + batch_size, len(tickers))} / {len(tickers)}')
        raw = yf.download(batch, period=period, interval='1mo', auto_adjust=True, progress=False, threads=True)
        if raw.empty:
            print('Warning: empty batch.')
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw['Close'].copy()
        else:
            close = raw[['Close']].copy()
            if len(batch) == 1:
                close.columns = batch
        batches.append(close)
        time.sleep(1)
    if len(batches) == 0:
        raise RuntimeError('No price data downloaded.')
    prices = pd.concat(batches, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()
    prices = prices.dropna(how='all')
    # Remove an unfinished current month before research begins.
    today = pd.Timestamp.today().tz_localize(None)
    if len(prices) > 0:
        last_date = prices.index[-1]
        if last_date.year == today.year and last_date.month == today.month:
            print('\nRemoving unfinished current month:', last_date.strftime('%Y-%m'))
            prices = prices.iloc[:-1]
    print('\nRaw data:', len(prices), 'months ×', len(prices.columns), 'stocks')
    return prices

# ── Download raw monthly prices ─────────────────────────────────────────────
raw_prices = download_prices(UNIVERSE, PERIOD)


# ── Strict data-quality filter ──────────────────────────────────────────────
def filter_data_quality(prices, max_missing_ratio=0.05):
    keep = []
    records = []
    final_date = prices.index[-1]
    for ticker in prices.columns:
        s = prices[ticker]
        total_missing = s.isna().sum()
        missing_ratio = s.isna().mean()
        valid = s.dropna()
        if valid.empty:
            records.append({'Ticker': ticker, 'Missing Ratio': 1.0, 'Internal Gaps': np.nan, 'Reason': 'No usable data', 'Keep': False})
            continue
        first_valid = valid.index[0]
        last_valid = valid.index[-1]
        internal_series = s.loc[first_valid:final_date]
        internal_gaps = internal_series.isna().sum()
        reason = 'Pass'
        acceptable = True
        if missing_ratio > max_missing_ratio:
            acceptable = False
            reason = f'Missing > {max_missing_ratio:.0%}'
        elif internal_gaps > 0:
            acceptable = False
            reason = 'Internal/trailing missing data'
        elif last_valid != final_date:
            acceptable = False
            reason = 'No latest completed-month price'
        if acceptable:
            keep.append(ticker)
        records.append({'Ticker': ticker, 'Missing Ratio': missing_ratio, 'Internal Gaps': internal_gaps, 'First Valid': first_valid, 'Last Valid': last_valid, 'Reason': reason, 'Keep': acceptable})
    report = pd.DataFrame(records).set_index('Ticker')
    filtered = prices[keep].copy()
    print('\n' + '=' * 70)
    print('DATA QUALITY FILTER')
    print('=' * 70)
    print('Original downloaded stocks:', len(prices.columns))
    print('Maximum allowed missing ratio:', f'{max_missing_ratio:.1%}')
    print('Stocks retained:', len(filtered.columns))
    print('Stocks removed:', len(prices.columns) - len(filtered.columns))
    print('\nNo forward filling is used.')
    return (filtered, report)

# ── Apply data-quality filter ───────────────────────────────────────────────
prices, data_quality_report = filter_data_quality(raw_prices, MAX_MISSING_RATIO)
print('\nRemoved stocks:')
display(data_quality_report[~data_quality_report['Keep']][['Missing Ratio', 'Internal Gaps', 'Reason']])
print('\nClean price table:')
display(prices.tail())
print('\nFinal clean universe:', len(prices.columns), 'stocks')


# ── Signal ──────────────────────────────────────────────────────────────────
def compute_momentum(prices, formation_months, skip_months=1):
    end_price = prices.shift(skip_months)
    start_price = prices.shift(formation_months)
    momentum = end_price / start_price - 1
    return momentum


# ── Monthly returns ─────────────────────────────────────────────────────────
def compute_monthly_returns(prices):
    return prices.pct_change(fill_method=None)


# ── Performance statistics ──────────────────────────────────────────────────
def sharpe_ratio(returns):
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    vol = r.std()
    if vol <= 0:
        return np.nan
    return r.mean() / vol * np.sqrt(12)

def compute_stats(returns, turnover=None):
    r = returns.dropna()
    n = len(r)
    if n == 0:
        return {}
    total_growth = (1 + r).prod()
    if total_growth > 0:
        annual_return = total_growth ** (12 / n) - 1
    else:
        annual_return = np.nan
    annual_vol = r.std() * np.sqrt(12)
    sharpe = sharpe_ratio(r)
    downside = r[r < 0].std() * np.sqrt(12)
    arithmetic_annual_return = r.mean() * 12
    if pd.notna(downside) and downside > 0:
        sortino = arithmetic_annual_return / downside
    else:
        sortino = np.nan
    cumulative = (1 + r).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1
    max_drawdown = drawdown.min()
    total_return = cumulative.iloc[-1] - 1
    win_rate = (r > 0).mean()
    stats = {'Months': n, 'Mean Monthly': r.mean(), 'Annual Return': annual_return, 'Annual Vol': annual_vol, 'Sharpe': sharpe, 'Sortino': sortino, 'Max Drawdown': max_drawdown, 'Total Return': total_return, 'Win Rate': win_rate}
    if turnover is not None:
        stats['Average Turnover'] = turnover.dropna().mean()
    return stats


# ── Momentum backtest ───────────────────────────────────────────────────────
def run_backtest(prices, formation_months, skip_months=1, top_pct=0.2, bottom_pct=0.2, transaction_cost_bps=10, mode='long_short'):
    momentum = compute_momentum(prices, formation_months, skip_months)
    monthly_returns = compute_monthly_returns(prices)
    rows = []
    previous_weights = pd.Series(dtype=float)
    skipped_missing_returns = 0
    for i in range(formation_months, len(prices) - 1):
        # Build the signal using information available at time t only.
        scores = momentum.iloc[i].dropna()
        if len(scores) < MIN_STOCKS:
            continue
        n_top = max(1, int(len(scores) * top_pct))
        winners = scores.nlargest(n_top).index
        weights = pd.Series(0.0, index=scores.index)
        weights.loc[winners] = 1.0 / len(winners)
        losers = pd.Index([])
        if mode == 'long_short':
            n_bottom = max(1, int(len(scores) * bottom_pct))
            losers = scores.nsmallest(n_bottom).index
            weights.loc[losers] = -1.0 / len(losers)
        elif mode == 'long_only':
            pass
        else:
            raise ValueError("mode must be 'long_short' or 'long_only'")
        # Observe t+1 returns only after the portfolio has been selected.
        next_returns = monthly_returns.iloc[i + 1]
        selected_assets = winners.union(losers)
        selected_next_returns = next_returns.reindex(selected_assets)
        if selected_next_returns.isna().any():
            skipped_missing_returns += 1
            continue
        # Long/short leg decomposition.
        long_leg_return = next_returns.reindex(winners).mean()
        loser_stock_return = np.nan
        short_leg_pnl = 0.0
        if mode == 'long_short':
            loser_stock_return = next_returns.reindex(losers).mean()
            short_leg_pnl = -loser_stock_return
        # One-way turnover from changes in portfolio weights.
        all_assets = weights.index.union(previous_weights.index)
        current = weights.reindex(all_assets, fill_value=0.0)
        previous = previous_weights.reindex(all_assets, fill_value=0.0)
        turnover = current.sub(previous).abs().sum()
        aligned_returns = next_returns.reindex(weights.index)
        # Portfolio return and transaction costs.
        gross_return = (weights.loc[selected_assets] * next_returns.loc[selected_assets]).sum()
        cost = turnover * transaction_cost_bps / 10000
        net_return = gross_return - cost
        rows.append({'date': prices.index[i + 1], 'gross_return': gross_return, 'net_return': net_return, 'long_leg_return': long_leg_return, 'loser_stock_return': loser_stock_return, 'short_leg_pnl': short_leg_pnl, 'turnover': turnover, 'cost': cost, 'n_stocks': len(scores), 'n_long': len(winners), 'n_short': len(losers)})
        previous_weights = weights.copy()
    result = pd.DataFrame(rows).set_index('date')
    if skipped_missing_returns > 0:
        print(f'Warning: {formation_months}-{skip_months} skipped {skipped_missing_returns} months because a selected position had no next-month return.')
    return result


# ── Equal-weight benchmark ──────────────────────────────────────────────────
def run_equal_weight(prices, start_months=12, transaction_cost_bps=10):
    monthly_returns = compute_monthly_returns(prices)
    rows = []
    previous_weights = pd.Series(dtype=float)
    for i in range(start_months, len(prices) - 1):
        eligible = prices.iloc[i].dropna().index
        if len(eligible) < MIN_STOCKS:
            continue
        weights = pd.Series(1.0 / len(eligible), index=eligible)
        # Observe t+1 returns only after the portfolio has been selected.
        next_returns = monthly_returns.iloc[i + 1].reindex(eligible)
        if next_returns.isna().any():
            continue
        # One-way turnover from changes in portfolio weights.
        all_assets = weights.index.union(previous_weights.index)
        current = weights.reindex(all_assets, fill_value=0.0)
        previous = previous_weights.reindex(all_assets, fill_value=0.0)
        turnover = current.sub(previous).abs().sum()
        # Portfolio return and transaction costs.
        gross_return = (weights * next_returns).sum()
        cost = turnover * transaction_cost_bps / 10000
        rows.append({'date': prices.index[i + 1], 'gross_return': gross_return, 'net_return': gross_return - cost, 'turnover': turnover, 'cost': cost, 'n_stocks': len(eligible)})
        previous_weights = weights.copy()
    return pd.DataFrame(rows).set_index('date')


# ── Chronological split ─────────────────────────────────────────────────────
def make_split_dates(prices, max_formation_months, dev_frac=0.5, val_frac=0.25):
    dates = prices.index[max_formation_months + 1:]
    n = len(dates)
    dev_end = int(n * dev_frac)
    val_end = int(n * (dev_frac + val_frac))
    development = dates[:dev_end]
    validation = dates[dev_end:val_end]
    evaluation = dates[val_end:]
    return (development, validation, evaluation)
development_dates, validation_dates, evaluation_dates = make_split_dates(prices, max(CANDIDATES), DEV_FRAC, VAL_FRAC)
print('\n' + '=' * 70)
print('DATA SPLIT')
print('=' * 70)
print('Development:', development_dates[0].date(), '→', development_dates[-1].date(), f'({len(development_dates)} months)')
print('Validation :', validation_dates[0].date(), '→', validation_dates[-1].date(), f'({len(validation_dates)} months)')
print('Evaluation :', evaluation_dates[0].date(), '→', evaluation_dates[-1].date(), f'({len(evaluation_dates)} months)')
print('\nHistorical Evaluation is NOT described as untouched OOS because we have already seen it.')


# ── Rank IC ─────────────────────────────────────────────────────────────────
def compute_rank_ic(prices, formation_months, skip_months=1):
    momentum = compute_momentum(prices, formation_months, skip_months)
    monthly_returns = compute_monthly_returns(prices)
    rows = []
    for i in range(formation_months, len(prices) - 1):
        score = momentum.iloc[i].rename('score')
        future = monthly_returns.iloc[i + 1].rename('future_return')
        data = pd.concat([score, future], axis=1).dropna()
        if len(data) < MIN_STOCKS:
            continue
        score_rank = data['score'].rank()
        return_rank = data['future_return'].rank()
        ic = score_rank.corr(return_rank)
        rows.append({'date': prices.index[i + 1], 'rank_ic': ic, 'n_stocks': len(data)})
    return pd.DataFrame(rows).set_index('date')

def ic_summary(ic_series):
    ic = ic_series.dropna()
    if len(ic) < 2:
        return {}
    std = ic.std()
    if std > 0:
        icir = ic.mean() / std * np.sqrt(12)
    else:
        icir = np.nan
    return {'Months': len(ic), 'Mean Rank IC': ic.mean(), 'Median Rank IC': ic.median(), 'Rank IC Std': std, 'Annualized ICIR': icir, 'Positive IC Rate': (ic > 0).mean()}


# ── Quintile portfolios ─────────────────────────────────────────────────────
def compute_quintile_returns(prices, formation_months, skip_months=1, n_quantiles=5):
    momentum = compute_momentum(prices, formation_months, skip_months)
    monthly_returns = compute_monthly_returns(prices)
    rows = []
    for i in range(formation_months, len(prices) - 1):
        scores = momentum.iloc[i].rename('score')
        future = monthly_returns.iloc[i + 1].rename('future_return')
        data = pd.concat([scores, future], axis=1).dropna()
        if len(data) < MIN_STOCKS:
            continue
        ranked = data['score'].rank(method='first')
        data['quantile'] = pd.qcut(ranked, q=n_quantiles, labels=False) + 1
        grouped = data.groupby('quantile')['future_return'].mean()
        row = {'date': prices.index[i + 1]}
        for q in range(1, n_quantiles + 1):
            row[f'Q{q}'] = grouped.get(q, np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index('date')

def quintile_summary(quintile_returns):
    rows = []
    for col in quintile_returns.columns:
        r = quintile_returns[col].dropna()
        if len(r) == 0:
            continue
        growth = (1 + r).prod()
        if growth > 0:
            ann_return = growth ** (12 / len(r)) - 1
        else:
            ann_return = np.nan
        rows.append({'Portfolio': col, 'Months': len(r), 'Mean Monthly Return': r.mean(), 'Annual Return': ann_return, 'Win Rate': (r > 0).mean()})
    return pd.DataFrame(rows).set_index('Portfolio')


# ── Centered circular-block bootstrap ───────────────────────────────────────
def circular_block_sample(values, block_size, rng):
    values = np.asarray(values, dtype=float)
    n = len(values)
    output = []
    while len(output) < n:
        start = rng.integers(0, n)
        indices = (start + np.arange(block_size)) % n
        output.extend(values[indices])
    return np.asarray(output[:n])

def bootstrap_significance(returns, n_sim=5000, block_size=3, seed=42):
    clean = returns.dropna().values.astype(float)
    n = len(clean)
    if n < 6:
        return {'months': n, 'observed_mean': np.nan, 'observed_sharpe': np.nan, 'p_mean': np.nan, 'p_sharpe': np.nan, 'mean_ci_low': np.nan, 'mean_ci_high': np.nan, 'sharpe_ci_low': np.nan, 'sharpe_ci_high': np.nan}
    rng = np.random.default_rng(seed)
    observed_mean = clean.mean()
    observed_std = clean.std(ddof=1)
    observed_sharpe = observed_mean / observed_std * np.sqrt(12)
    # Zero-mean null used for significance testing.
    centered = clean - clean.mean()
    null_means = np.empty(n_sim)
    null_sharpes = np.empty(n_sim)
    raw_means = np.empty(n_sim)
    raw_sharpes = np.empty(n_sim)
    for sim in range(n_sim):
        # Null bootstrap preserves short-run dependence through circular blocks.
        sample_null = circular_block_sample(centered, block_size, rng)
        null_means[sim] = sample_null.mean()
        null_std = sample_null.std(ddof=1)
        if null_std > 0:
            null_sharpes[sim] = sample_null.mean() / null_std * np.sqrt(12)
        else:
            null_sharpes[sim] = 0.0
        # Raw bootstrap is used for confidence intervals.
        sample_raw = circular_block_sample(clean, block_size, rng)
        raw_means[sim] = sample_raw.mean()
        raw_std = sample_raw.std(ddof=1)
        if raw_std > 0:
            raw_sharpes[sim] = sample_raw.mean() / raw_std * np.sqrt(12)
        else:
            raw_sharpes[sim] = np.nan
    p_mean = (1 + np.sum(null_means >= observed_mean)) / (n_sim + 1)
    p_sharpe = (1 + np.sum(null_sharpes >= observed_sharpe)) / (n_sim + 1)
    mean_ci_low, mean_ci_high = np.quantile(raw_means, [0.025, 0.975])
    sharpe_ci_low, sharpe_ci_high = np.nanquantile(raw_sharpes, [0.025, 0.975])
    return {'months': n, 'observed_mean': observed_mean, 'observed_sharpe': observed_sharpe, 'p_mean': p_mean, 'p_sharpe': p_sharpe, 'mean_ci_low': mean_ci_low, 'mean_ci_high': mean_ci_high, 'sharpe_ci_low': sharpe_ci_low, 'sharpe_ci_high': sharpe_ci_high}

# ── Run momentum candidates ─────────────────────────────────────────────────
backtests = {}
rank_ics = {}
quintiles = {}
print('\n' + '=' * 70)
print('RUNNING FACTOR RESEARCH')
print('=' * 70)
for formation in CANDIDATES:
    print(f'\nRunning {formation}-{SKIP_MONTHS}...')
    backtests[formation] = run_backtest(prices=prices, formation_months=formation, skip_months=SKIP_MONTHS, top_pct=TOP_PCT, bottom_pct=BOTTOM_PCT, transaction_cost_bps=BASE_COST_BPS, mode='long_short')
    rank_ics[formation] = compute_rank_ic(prices, formation, SKIP_MONTHS)
    quintiles[formation] = compute_quintile_returns(prices, formation, SKIP_MONTHS, n_quantiles=5)

# ── Development performance ─────────────────────────────────────────────────
development_rows = []
for formation in CANDIDATES:
    result = backtests[formation]
    dev = result.loc[result.index.isin(development_dates)]
    stats = compute_stats(dev['net_return'], dev['turnover'])
    development_rows.append({'Momentum': f'{formation}-1', **stats})
development_table = pd.DataFrame(development_rows).set_index('Momentum')
print('\n' + '=' * 70)
print('DEVELOPMENT PERFORMANCE')
print('=' * 70)
display(development_table.style.format({'Mean Monthly': '{:.3%}', 'Annual Return': '{:.2%}', 'Annual Vol': '{:.2%}', 'Sharpe': '{:.3f}', 'Sortino': '{:.3f}', 'Max Drawdown': '{:.2%}', 'Total Return': '{:.2%}', 'Win Rate': '{:.2%}', 'Average Turnover': '{:.2%}'}))

# ── Rank IC analysis ────────────────────────────────────────────────────────
ic_rows = []
for formation in CANDIDATES:
    ic_data = rank_ics[formation]
    for period_name, period_dates in [('Development', development_dates), ('Validation', validation_dates), ('Evaluation', evaluation_dates)]:
        period_ic = ic_data.loc[ic_data.index.isin(period_dates), 'rank_ic']
        summary = ic_summary(period_ic)
        ic_rows.append({'Momentum': f'{formation}-1', 'Period': period_name, **summary})
ic_summary_table = pd.DataFrame(ic_rows)
print('\n' + '=' * 70)
print('RANK IC ANALYSIS')
print('=' * 70)
display(ic_summary_table.style.format({'Mean Rank IC': '{:.4f}', 'Median Rank IC': '{:.4f}', 'Rank IC Std': '{:.4f}', 'Annualized ICIR': '{:.3f}', 'Positive IC Rate': '{:.2%}'}))

# ── Validation and multiple testing ─────────────────────────────────────────
validation_rows = []
for idx, formation in enumerate(CANDIDATES):
    result = backtests[formation]
    validation = result.loc[result.index.isin(validation_dates)]
    stats = compute_stats(validation['net_return'], validation['turnover'])
    boot = bootstrap_significance(validation['net_return'], n_sim=N_BOOTSTRAP, block_size=BLOCK_SIZE, seed=100 + idx)
    validation_rows.append({'formation_months': formation, 'momentum': f'{formation}-1', 'validation_sharpe': stats['Sharpe'], 'annual_return': stats['Annual Return'], 'max_drawdown': stats['Max Drawdown'], 'win_rate': stats['Win Rate'], 'bootstrap_p_mean': boot['p_mean'], 'bootstrap_p_sharpe': boot['p_sharpe']})
validation_table = pd.DataFrame(validation_rows)
validation_table['bonferroni_p_mean'] = np.minimum(validation_table['bootstrap_p_mean'] * len(CANDIDATES), 1.0)
print('\n' + '=' * 70)
print('VALIDATION + MULTIPLE TESTING')
print('=' * 70)
display(validation_table.style.format({'validation_sharpe': '{:.3f}', 'annual_return': '{:.2%}', 'max_drawdown': '{:.2%}', 'win_rate': '{:.2%}', 'bootstrap_p_mean': '{:.4f}', 'bootstrap_p_sharpe': '{:.4f}', 'bonferroni_p_mean': '{:.4f}'}))

# ── Select parameter using validation only ──────────────────────────────────
best_row = validation_table.loc[validation_table['validation_sharpe'].idxmax()]
best_formation = int(best_row['formation_months'])
print('\n' + '=' * 70)
print('Selected signal:', f'{best_formation}-1 Momentum')
print('Selection rule:', 'highest Validation Sharpe')
print('=' * 70)

# ── Historical evaluation ───────────────────────────────────────────────────
selected = backtests[best_formation]
evaluation = selected.loc[selected.index.isin(evaluation_dates)]
evaluation_stats = compute_stats(evaluation['net_return'], evaluation['turnover'])
print('\n' + '=' * 70)
print('HISTORICAL EVALUATION')
print('=' * 70)
print('Strategy:', f'{best_formation}-1 Momentum')
print('Period:', evaluation.index[0].date(), '→', evaluation.index[-1].date())
print('Months:', len(evaluation))
print('Average usable stocks:', f"{evaluation['n_stocks'].mean():.1f}")
print('Average Long positions:', f"{evaluation['n_long'].mean():.1f}")
print('Average Short positions:', f"{evaluation['n_short'].mean():.1f}")
print('-' * 70)
for key, value in evaluation_stats.items():
    if key == 'Months':
        print(f'{key:22s}: {value}')
    elif key in ['Sharpe', 'Sortino']:
        print(f'{key:22s}: {value:.3f}')
    else:
        print(f'{key:22s}: {value:.2%}')

# ── Long / short leg decomposition ──────────────────────────────────────────
long_stats = compute_stats(evaluation['long_leg_return'])
short_stats = compute_stats(evaluation['short_leg_pnl'])
loser_raw_stats = compute_stats(evaluation['loser_stock_return'])
leg_table = pd.DataFrame([{'Leg': 'Long winners', **long_stats}, {'Leg': 'Short losers P&L', **short_stats}, {'Leg': 'Loser stocks raw return', **loser_raw_stats}]).set_index('Leg')
print('\n' + '=' * 70)
print('LONG / SHORT LEG DECOMPOSITION')
print('=' * 70)
display(leg_table.style.format({'Mean Monthly': '{:.3%}', 'Annual Return': '{:.2%}', 'Annual Vol': '{:.2%}', 'Sharpe': '{:.3f}', 'Sortino': '{:.3f}', 'Max Drawdown': '{:.2%}', 'Total Return': '{:.2%}', 'Win Rate': '{:.2%}'}))

# ── Quintile analysis ───────────────────────────────────────────────────────
selected_quintiles = quintiles[best_formation]
quintile_tables = {}
for period_name, period_dates in [('Development', development_dates), ('Validation', validation_dates), ('Evaluation', evaluation_dates)]:
    q_data = selected_quintiles.loc[selected_quintiles.index.isin(period_dates)]
    summary = quintile_summary(q_data)
    quintile_tables[period_name] = summary
    print('\n' + '=' * 70)
    print(period_name.upper(), 'QUINTILE RETURNS')
    print('=' * 70)
    display(summary.style.format({'Mean Monthly Return': '{:.3%}', 'Annual Return': '{:.2%}', 'Win Rate': '{:.2%}'}))
    if 'Q5' in q_data.columns and 'Q1' in q_data.columns:
        spread = q_data['Q5'] - q_data['Q1']
        spread_stats = compute_stats(spread)
        print('Q5 - Q1 Annual Return:', f"{spread_stats['Annual Return']:.2%}")
        print('Q5 - Q1 Sharpe:', f"{spread_stats['Sharpe']:.3f}")
for period_name, summary in quintile_tables.items():
    plt.figure(figsize=(8, 4))
    plt.bar(summary.index, summary['Annual Return'].values)
    plt.axhline(0, linewidth=1)
    plt.title(f'{best_formation}-1 Momentum — {period_name} Quintiles')
    plt.ylabel('Annualized Return')
    plt.xlabel('Momentum Quintile')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / f'quintiles_{period_name.lower()}.png', dpi=150, bbox_inches='tight')
    plt.show()

# ── Rank IC plot ────────────────────────────────────────────────────────────
selected_ic = rank_ics[best_formation]
rolling_ic = selected_ic['rank_ic'].rolling(12).mean()
plt.figure(figsize=(12, 5))
plt.plot(selected_ic.index, selected_ic['rank_ic'], alpha=0.45, label='Monthly Rank IC')
plt.plot(rolling_ic.index, rolling_ic, linewidth=2, label='12-month rolling mean')
plt.axhline(0, linewidth=1)
plt.title(f'{best_formation}-1 Momentum — Rank IC')
plt.ylabel('Spearman Rank IC')
plt.xlabel('Date')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULT_DIR / 'rank_ic.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Benchmarks ──────────────────────────────────────────────────────────────
long_only = run_backtest(prices=prices, formation_months=best_formation, skip_months=SKIP_MONTHS, top_pct=TOP_PCT, bottom_pct=BOTTOM_PCT, transaction_cost_bps=BASE_COST_BPS, mode='long_only')
equal_weight = run_equal_weight(prices, start_months=max(CANDIDATES), transaction_cost_bps=BASE_COST_BPS)
benchmark_data = {'Long-Short Momentum': selected, 'Long-Only Momentum': long_only, 'Equal-Weight Universe': equal_weight}
benchmark_rows = []
for name, data in benchmark_data.items():
    period_data = data.loc[data.index.isin(evaluation_dates)]
    stats = compute_stats(period_data['net_return'], period_data['turnover'])
    benchmark_rows.append({'Strategy': name, **stats})
benchmark_table = pd.DataFrame(benchmark_rows).set_index('Strategy')
print('\n' + '=' * 70)
print('BENCHMARK COMPARISON')
print('=' * 70)
display(benchmark_table.style.format({'Mean Monthly': '{:.3%}', 'Annual Return': '{:.2%}', 'Annual Vol': '{:.2%}', 'Sharpe': '{:.3f}', 'Sortino': '{:.3f}', 'Max Drawdown': '{:.2%}', 'Total Return': '{:.2%}', 'Win Rate': '{:.2%}', 'Average Turnover': '{:.2%}'}))
plt.figure(figsize=(12, 5))
for name, data in benchmark_data.items():
    period_data = data.loc[data.index.isin(evaluation_dates)]
    cumulative = (1 + period_data['net_return']).cumprod()
    plt.plot(cumulative.index, cumulative.values, label=name)
plt.axhline(1, linewidth=1)
plt.title('Historical Evaluation — Strategy Comparison')
plt.ylabel('Growth of $1')
plt.xlabel('Date')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULT_DIR / 'benchmark_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Transaction-cost sensitivity ────────────────────────────────────────────
cost_rows = []
for cost_bps in COST_LEVELS:
    result = run_backtest(prices=prices, formation_months=best_formation, skip_months=SKIP_MONTHS, top_pct=TOP_PCT, bottom_pct=BOTTOM_PCT, transaction_cost_bps=cost_bps, mode='long_short')
    period_result = result.loc[result.index.isin(evaluation_dates)]
    stats = compute_stats(period_result['net_return'], period_result['turnover'])
    cost_rows.append({'Cost_bps': cost_bps, 'Annual Return': stats['Annual Return'], 'Sharpe': stats['Sharpe'], 'Max Drawdown': stats['Max Drawdown'], 'Total Return': stats['Total Return']})
cost_table = pd.DataFrame(cost_rows).set_index('Cost_bps')
print('\n' + '=' * 70)
print('TRANSACTION-COST SENSITIVITY')
print('=' * 70)
display(cost_table.style.format({'Annual Return': '{:.2%}', 'Sharpe': '{:.3f}', 'Max Drawdown': '{:.2%}', 'Total Return': '{:.2%}'}))

# ── Gross vs net ────────────────────────────────────────────────────────────
gross_stats = compute_stats(evaluation['gross_return'])
net_stats = compute_stats(evaluation['net_return'])
gross_net_table = pd.DataFrame([{'Return Type': 'Gross', **gross_stats}, {'Return Type': 'Net of 10 bps', **net_stats}]).set_index('Return Type')
print('\n' + '=' * 70)
print('GROSS VS NET')
print('=' * 70)
display(gross_net_table.style.format({'Mean Monthly': '{:.3%}', 'Annual Return': '{:.2%}', 'Annual Vol': '{:.2%}', 'Sharpe': '{:.3f}', 'Sortino': '{:.3f}', 'Max Drawdown': '{:.2%}', 'Total Return': '{:.2%}', 'Win Rate': '{:.2%}'}))

# ── Bootstrap — historical evaluation ───────────────────────────────────────
evaluation_bootstrap = bootstrap_significance(evaluation['net_return'], n_sim=N_BOOTSTRAP, block_size=BLOCK_SIZE, seed=2026)
print('\n' + '=' * 70)
print('CENTERED CIRCULAR-BLOCK BOOTSTRAP')
print('=' * 70)
print('Months:', evaluation_bootstrap['months'])
print('Observed mean monthly return:', f"{evaluation_bootstrap['observed_mean']:.3%}")
print('Observed Sharpe:', f"{evaluation_bootstrap['observed_sharpe']:.3f}")
print('One-sided p-value (mean > 0):', f"{evaluation_bootstrap['p_mean']:.4f}")
print('One-sided p-value (Sharpe > 0):', f"{evaluation_bootstrap['p_sharpe']:.4f}")
print('95% bootstrap CI — monthly mean:', f"[{evaluation_bootstrap['mean_ci_low']:.3%}, {evaluation_bootstrap['mean_ci_high']:.3%}]")
print('95% bootstrap CI — Sharpe:', f"[{evaluation_bootstrap['sharpe_ci_low']:.3f}, {evaluation_bootstrap['sharpe_ci_high']:.3f}]")

# ── Selected strategy across three periods ──────────────────────────────────
period_rows = []
for period_name, period_dates in [('Development', development_dates), ('Validation', validation_dates), ('Historical Evaluation', evaluation_dates)]:
    period_data = selected.loc[selected.index.isin(period_dates)]
    stats = compute_stats(period_data['net_return'], period_data['turnover'])
    period_rows.append({'Period': period_name, **stats})
period_table = pd.DataFrame(period_rows).set_index('Period')
print('\n' + '=' * 70)
print(f'{best_formation}-1 MOMENTUM ACROSS ALL PERIODS')
print('=' * 70)
display(period_table.style.format({'Mean Monthly': '{:.3%}', 'Annual Return': '{:.2%}', 'Annual Vol': '{:.2%}', 'Sharpe': '{:.3f}', 'Sortino': '{:.3f}', 'Max Drawdown': '{:.2%}', 'Total Return': '{:.2%}', 'Win Rate': '{:.2%}', 'Average Turnover': '{:.2%}'}))

# ── Evaluation cumulative return ────────────────────────────────────────────
evaluation_cumulative = (1 + evaluation['net_return']).cumprod()
plt.figure(figsize=(12, 5))
plt.plot(evaluation_cumulative.index, evaluation_cumulative.values, linewidth=2)
plt.axhline(1, linewidth=1)
plt.title(f'{best_formation}-1 Momentum — Historical Evaluation')
plt.xlabel('Date')
plt.ylabel('Growth of $1')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULT_DIR / 'evaluation_cumulative_return.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Evaluation drawdown ─────────────────────────────────────────────────────
evaluation_drawdown = evaluation_cumulative / evaluation_cumulative.cummax() - 1
plt.figure(figsize=(12, 4))
plt.fill_between(evaluation_drawdown.index, evaluation_drawdown.values, 0, alpha=0.4)
plt.axhline(0, linewidth=1)
plt.title(f'{best_formation}-1 Momentum — Drawdown')
plt.ylabel('Drawdown')
plt.xlabel('Date')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULT_DIR / 'evaluation_drawdown.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Save results ────────────────────────────────────────────────────────────
pd.DataFrame({'Ticker': UNIVERSE}).to_csv(RESULT_DIR / 'original_200_stock_universe.csv', index=False)
prices.to_csv(RESULT_DIR / 'clean_monthly_prices.csv')
data_quality_report.to_csv(RESULT_DIR / 'data_quality_report.csv')
development_table.to_csv(RESULT_DIR / 'development_performance.csv')
validation_table.to_csv(RESULT_DIR / 'validation_multiple_testing.csv', index=False)
ic_summary_table.to_csv(RESULT_DIR / 'rank_ic_summary.csv', index=False)
selected_ic.to_csv(RESULT_DIR / 'selected_signal_rank_ic.csv')
selected_quintiles.to_csv(RESULT_DIR / 'selected_signal_quintile_returns.csv')
leg_table.to_csv(RESULT_DIR / 'long_short_leg_analysis.csv')
benchmark_table.to_csv(RESULT_DIR / 'benchmark_comparison.csv')
cost_table.to_csv(RESULT_DIR / 'cost_sensitivity.csv')
gross_net_table.to_csv(RESULT_DIR / 'gross_vs_net.csv')
period_table.to_csv(RESULT_DIR / 'period_comparison.csv')
evaluation.to_csv(RESULT_DIR / 'historical_evaluation_returns.csv')
pd.DataFrame([evaluation_bootstrap]).to_csv(RESULT_DIR / 'evaluation_bootstrap.csv', index=False)

# ── Save methodology and limitations ────────────────────────────────────────
notes = f"""
Cross-Sectional Momentum Research V2
====================================

Universe
--------
Initial fixed universe: 200 current large-cap US equities.

Data quality
------------
Maximum missing ratio: {MAX_MISSING_RATIO:.1%}

Stocks are removed if:
1. More than {MAX_MISSING_RATIO:.1%} of monthly prices are missing.
2. Any internal/trailing price gap remains after the stock's first
   valid monthly observation.
3. No price exists for the latest completed month.

Missing prices are NOT forward-filled.

Momentum signals
----------------
6-1
9-1
12-1

Signal definition:
P(t-1) / P(t-k) - 1

Portfolio
---------
Long top 20%
Short bottom 20%
Equal weight within each leg
Monthly rebalance

Base transaction cost:
{BASE_COST_BPS} bps per unit of turnover.

Time split
----------
Development: 50%
Validation: 25%
Historical Evaluation: 25%

Parameter selection
-------------------
Highest Validation Sharpe among the three predefined candidates.

Credibility checks
------------------
- Strict missing-data screening
- No forward filling
- No future return availability used to choose positions
- Rank IC
- Quintile portfolios
- Long / Short leg decomposition
- Long-only benchmark
- Equal-weight benchmark
- Gross vs Net
- Cost sensitivity
- Centered circular-block bootstrap
- Bonferroni multiple-testing correction

Limitations
-----------
1. The historical evaluation period was already observed during
   earlier research, so it is NOT a fresh untouched OOS test.

2. The universe is a fixed list of current stocks. This introduces
   survivorship / selection bias.

3. The full-sample data-quality screen uses historical availability
   information and is not a true point-in-time membership database.

4. Yahoo Finance is convenient research data, not institutional
   point-in-time market data.

5. Delisting returns are not explicitly modeled.

6. Short borrow availability and stock-specific borrow fees are not
   modeled.

7. Monthly closing data simplify execution, market impact, spreads
   and intramonth dynamics.

8. The results should be interpreted as exploratory factor research,
   not proof of a persistent trading edge.
"""
with open(RESULT_DIR / 'methodology_and_limitations.txt', 'w') as f:
    f.write(notes)

# ── Final summary ───────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('V2 COMPLETE')
print('=' * 70)
print('Original universe:', len(UNIVERSE))
print('Clean universe:', len(prices.columns))
print('Selected signal:', f'{best_formation}-1 Momentum')
print('Results saved in:', RESULT_DIR)
print('\nCredibility upgrades:\n✓ No forward filling\n✓ <=5% missing-price threshold\n✓ No internal data gaps\n✓ No t+1 availability used for stock selection\n✓ Rank IC\n✓ Quintile analysis\n✓ Long / Short decomposition\n✓ Benchmarks\n✓ Gross vs Net\n✓ Cost sensitivity\n✓ Block bootstrap\n✓ Multiple-testing correction\n✓ Explicit limitations\n')
