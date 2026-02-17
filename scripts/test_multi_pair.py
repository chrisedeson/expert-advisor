#!/usr/bin/env python3
"""
Multi-Pair Portfolio Backtest

Runs independent backtests for EURUSD, GBPUSD, USDJPY, AUDUSD and combines
equity curves to calculate portfolio-level metrics.

Theory: 4 correlated pairs (rho ~0.50) should give portfolio Sharpe improvement
of ~sqrt(4) / sqrt(1 + 3*0.5) = 1.26x. With single-pair Sharpe of 0.90,
portfolio Sharpe target is ~1.14.

Capital is split equally across pairs ($125 each from $500 total).
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.backtesting.protected_grid_engine import (
    ProtectedGridBacktester,
    print_backtest_summary,
    BacktestResult,
)


# Pair-specific parameters
PAIR_CONFIGS = {
    'EURUSD': {
        'pip_size': 0.0001,
        'pip_value_per_lot': 10.0,
        'spread_pips': 0.7,
        'slippage_pips': 0.2,
        'base_lot_size': 0.0228,  # Optimized from single-pair testing
    },
    'GBPUSD': {
        'pip_size': 0.0001,
        'pip_value_per_lot': 10.0,
        'spread_pips': 0.9,
        'slippage_pips': 0.3,
        'base_lot_size': 0.0228,
    },
    'USDJPY': {
        'pip_size': 0.01,
        'pip_value_per_lot': 6.67,  # ~$6.67 per pip per standard lot (at ~150 USDJPY)
        'spread_pips': 0.8,
        'slippage_pips': 0.2,
        'base_lot_size': 0.0228,
    },
    'AUDUSD': {
        'pip_size': 0.0001,
        'pip_value_per_lot': 10.0,
        'spread_pips': 0.9,
        'slippage_pips': 0.3,
        'base_lot_size': 0.0228,
    },
}


def load_config(base_lot_size: float = 0.0228) -> dict:
    """Load grid-optimized configuration for backtesting."""
    return {
        'grid_strategy': {
            'base_lot_size': base_lot_size,
            'lot_multiplier': 1.5,
            'max_grid_levels': 5,
        },
        'volatility_filter': {
            'atr_period': 14,
            'avg_period': 50,
            'normal_threshold': 10.0,
            'crisis_threshold': 20.0,
            'cooldown_days': 0,
        },
        'circuit_breaker': {
            'daily_limit': 0.20,
            'weekly_limit': 0.35,
            'monthly_limit': 0.50,
        },
        'crisis_detector': {
            'volatility_spike_threshold': 6.0,
            'rapid_drawdown_threshold': 0.50,
            'rapid_drawdown_days': 3,
            'consecutive_stops_threshold': 15,
        },
        'recovery_manager': {
            'drawdown_threshold': 0.40,
        },
        'profit_protector': {
            'profit_threshold': 1.0,
        },
    }


def load_historical_data(pair: str, timeframe: str = 'H1') -> pd.DataFrame:
    """Load historical data with ATR calculated."""
    data_path = project_root / 'data' / 'processed' / f'{pair}_{timeframe}.parquet'

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return None

    logger.info(f"Loading {pair} {timeframe} data from {data_path}")
    df = pd.read_parquet(data_path)

    if not isinstance(df.index, pd.DatetimeIndex):
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

    if 'atr' not in df.columns:
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()
        df.drop('tr', axis=1, inplace=True)

    logger.success(f"  {pair}: {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
    return df


def run_single_pair(pair: str, data: pd.DataFrame, capital: float) -> BacktestResult:
    """Run backtest for a single pair with pair-specific parameters."""
    pair_cfg = PAIR_CONFIGS[pair]
    config = load_config(base_lot_size=pair_cfg['base_lot_size'])

    # Scale lot size proportionally to capital allocation
    # Original lot was sized for $500, scale for allocated capital
    capital_ratio = capital / 500.0
    config['grid_strategy']['base_lot_size'] = pair_cfg['base_lot_size'] * capital_ratio

    backtester = ProtectedGridBacktester(
        initial_balance=capital,
        config=config,
        spread_pips=pair_cfg['spread_pips'],
        commission_per_lot=0.0,
        slippage_pips=pair_cfg['slippage_pips'],
        pip_size=pair_cfg['pip_size'],
        pip_value_per_lot=pair_cfg['pip_value_per_lot'],
    )

    result = backtester.run_backtest(data)
    return result


def calculate_portfolio_metrics(
    results: dict,
    total_capital: float,
) -> dict:
    """
    Combine individual pair equity curves into portfolio-level metrics.

    Each pair's equity curve represents its share of the portfolio.
    Portfolio equity = sum of all pair equities at each timestamp.
    """
    # Collect all daily equity series
    daily_equities = {}
    for pair, result in results.items():
        eq = result.equity_curve['equity'].resample('D').last().dropna()
        daily_equities[pair] = eq

    # Combine into portfolio DataFrame, forward-fill missing days
    portfolio_df = pd.DataFrame(daily_equities)
    portfolio_df = portfolio_df.ffill().dropna()

    # Portfolio equity = sum of pair equities
    portfolio_df['portfolio'] = portfolio_df.sum(axis=1)

    # Portfolio daily returns
    portfolio_returns = portfolio_df['portfolio'].pct_change().dropna()

    # CAGR
    start_equity = portfolio_df['portfolio'].iloc[0]
    end_equity = portfolio_df['portfolio'].iloc[-1]
    n_days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
    years = n_days / 365.25
    total_return = (end_equity - start_equity) / start_equity
    cagr = (end_equity / start_equity) ** (1 / years) - 1 if years > 0 else 0

    # Max Drawdown
    peak = portfolio_df['portfolio'].cummax()
    drawdown = (portfolio_df['portfolio'] - peak) / peak
    max_drawdown = abs(drawdown.min())

    # Sharpe
    if len(portfolio_returns) > 10 and portfolio_returns.std() > 0:
        sharpe = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0

    # Sortino
    neg_returns = portfolio_returns[portfolio_returns < 0]
    if len(neg_returns) > 5 and neg_returns.std() > 0:
        sortino = (portfolio_returns.mean() / neg_returns.std()) * np.sqrt(252)
    else:
        sortino = 0

    # Calmar
    calmar = cagr / max_drawdown if max_drawdown > 0 else 0

    # Drawdown duration
    in_dd = drawdown < 0
    dd_durations = []
    dd_start = None
    for idx, is_dd in in_dd.items():
        if is_dd and dd_start is None:
            dd_start = idx
        elif not is_dd and dd_start is not None:
            dd_durations.append((idx - dd_start).days)
            dd_start = None
    max_dd_duration = max(dd_durations) if dd_durations else 0

    # Aggregate trade stats
    total_trades = sum(r.total_trades for r in results.values())
    winning_trades = sum(r.winning_trades for r in results.values())
    losing_trades = sum(r.losing_trades for r in results.values())
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    gross_profit = sum(r.gross_profit for r in results.values())
    gross_loss = sum(r.gross_loss for r in results.values())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Pair correlations (daily returns)
    pair_returns = pd.DataFrame({
        pair: daily_equities[pair].pct_change().dropna()
        for pair in daily_equities
    }).dropna()
    correlation_matrix = pair_returns.corr() if len(pair_returns) > 10 else pd.DataFrame()

    return {
        'start_date': portfolio_df.index[0],
        'end_date': portfolio_df.index[-1],
        'total_capital': total_capital,
        'final_equity': end_equity,
        'total_return': total_return,
        'cagr': cagr,
        'max_drawdown': max_drawdown,
        'max_dd_duration_days': max_dd_duration,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'profit_factor': profit_factor,
        'correlation_matrix': correlation_matrix,
        'portfolio_equity': portfolio_df['portfolio'],
        'pair_equities': portfolio_df.drop(columns='portfolio'),
    }


def print_portfolio_summary(metrics: dict, pair_results: dict):
    """Print formatted portfolio results."""
    print("\n" + "=" * 80)
    print("MULTI-PAIR PORTFOLIO BACKTEST RESULTS")
    print("=" * 80)

    print(f"\n  PERIOD: {metrics['start_date'].date()} to {metrics['end_date'].date()}")
    n_days = (metrics['end_date'] - metrics['start_date']).days
    print(f"  Duration: {n_days} days ({n_days/365.25:.1f} years)")

    print(f"\n  PORTFOLIO RETURNS:")
    print(f"    Initial Capital:  ${metrics['total_capital']:,.2f}")
    print(f"    Final Equity:     ${metrics['final_equity']:,.2f}")
    print(f"    Total Return:     {metrics['total_return']*100:,.2f}%")
    print(f"    CAGR:             {metrics['cagr']*100:.2f}%")

    print(f"\n  PORTFOLIO RISK:")
    print(f"    Max Drawdown:     {metrics['max_drawdown']*100:.2f}%")
    print(f"    Max DD Duration:  {metrics['max_dd_duration_days']} days")
    print(f"    Sharpe Ratio:     {metrics['sharpe']:.2f}")
    print(f"    Sortino Ratio:    {metrics['sortino']:.2f}")
    print(f"    Calmar Ratio:     {metrics['calmar']:.2f}")

    print(f"\n  AGGREGATE TRADING:")
    print(f"    Total Trades:     {metrics['total_trades']}")
    print(f"    Win Rate:         {metrics['win_rate']*100:.1f}%")
    print(f"    Profit Factor:    {metrics['profit_factor']:.2f}")

    # Per-pair breakdown
    print(f"\n  PER-PAIR BREAKDOWN:")
    print(f"  {'Pair':<10} {'Capital':<10} {'Final':<12} {'CAGR':<10} {'DD':<10} {'Sharpe':<10} {'PF':<8} {'Trades':<8}")
    print(f"  {'-'*76}")
    for pair, result in pair_results.items():
        print(
            f"  {pair:<10} "
            f"${result.initial_balance:<9,.0f} "
            f"${result.final_balance:<11,.2f} "
            f"{result.cagr*100:<9.2f}% "
            f"{result.max_drawdown*100:<9.2f}% "
            f"{result.sharpe_ratio:<9.2f} "
            f"{result.profit_factor:<7.2f} "
            f"{result.total_trades:<8}"
        )

    # Correlation matrix
    if not metrics['correlation_matrix'].empty:
        print(f"\n  PAIR CORRELATIONS (daily returns):")
        corr = metrics['correlation_matrix']
        pairs = list(corr.columns)
        print(f"  {'':>10}", end="")
        for p in pairs:
            print(f"  {p:>8}", end="")
        print()
        for p1 in pairs:
            print(f"  {p1:>10}", end="")
            for p2 in pairs:
                print(f"  {corr.loc[p1, p2]:>8.3f}", end="")
            print()

        # Average correlation
        n = len(pairs)
        if n > 1:
            off_diag = []
            for i in range(n):
                for j in range(i + 1, n):
                    off_diag.append(corr.iloc[i, j])
            avg_corr = np.mean(off_diag)
            print(f"\n  Average pairwise correlation: {avg_corr:.3f}")
            theoretical_improvement = np.sqrt(n) / np.sqrt(1 + (n - 1) * avg_corr)
            print(f"  Theoretical Sharpe multiplier: {theoretical_improvement:.2f}x")

    print("\n" + "=" * 80)


def print_go_no_go(metrics: dict, pair_results: dict):
    """Print GO/NO-GO criteria for portfolio."""
    print("\n" + "=" * 80)
    print("PORTFOLIO GO/NO-GO DECISION CRITERIA")
    print("=" * 80)

    go = []
    nogo = []

    if metrics['cagr'] >= 0.20:
        go.append(f"  CAGR: {metrics['cagr']*100:.1f}% (target: >20%)")
    else:
        nogo.append(f"  CAGR: {metrics['cagr']*100:.1f}% (target: >20%)")

    if metrics['max_drawdown'] <= 0.20:
        go.append(f"  Max DD: {metrics['max_drawdown']*100:.1f}% (target: <20%)")
    else:
        nogo.append(f"  Max DD: {metrics['max_drawdown']*100:.1f}% (target: <20%)")

    if metrics['sharpe'] >= 1.0:
        go.append(f"  Sharpe: {metrics['sharpe']:.2f} (target: >1.0)")
    else:
        nogo.append(f"  Sharpe: {metrics['sharpe']:.2f} (target: >1.0)")

    if metrics['win_rate'] >= 0.45:
        go.append(f"  Win Rate: {metrics['win_rate']*100:.1f}% (target: >45%)")
    else:
        nogo.append(f"  Win Rate: {metrics['win_rate']*100:.1f}% (target: >45%)")

    if metrics['profit_factor'] >= 1.3:
        go.append(f"  Profit Factor: {metrics['profit_factor']:.2f} (target: >1.3)")
    else:
        nogo.append(f"  Profit Factor: {metrics['profit_factor']:.2f} (target: >1.3)")

    print("\nMET CRITERIA:")
    for c in go:
        print(f"  [GO] {c}")

    if nogo:
        print("\nFAILED CRITERIA:")
        for c in nogo:
            print(f"  [!!] {c}")

    go_count = len(go)
    total = len(go) + len(nogo)
    print(f"\n  Score: {go_count}/{total} criteria met")

    if go_count == total:
        print("\n  VERDICT: GO FOR PAPER TRADING - All criteria met!")
    elif go_count >= total * 0.8:
        print("\n  VERDICT: CONDITIONAL GO - Most criteria met")
    else:
        print(f"\n  VERDICT: NEEDS WORK - {total - go_count} criteria still failing")

    print("=" * 80)


def run_allocation_scheme(name: str, allocations: dict, pair_data: dict) -> dict:
    """
    Run a specific allocation scheme and return portfolio metrics.

    Args:
        name: Scheme name for display
        allocations: Dict of {pair: capital_amount}
        pair_data: Dict of {pair: DataFrame}
    """
    pair_results = {}
    for pair, capital in allocations.items():
        if pair not in pair_data:
            continue
        if capital <= 0:
            continue
        result = run_single_pair(pair, pair_data[pair], capital)
        pair_results[pair] = result

    total_capital = sum(allocations.values())
    metrics = calculate_portfolio_metrics(pair_results, total_capital)
    return metrics, pair_results


def run_lot_sweep(pair_data: dict, eur_pct: float, gbp_pct: float,
                   lot_sizes: list, total_capital: float = 500.0):
    """Sweep lot sizes for a given EUR/GBP allocation."""
    eur_capital = total_capital * eur_pct
    gbp_capital = total_capital * gbp_pct

    print(f"\n  LOT SIZE SWEEP: EUR {eur_pct*100:.0f}% (${eur_capital:.0f}) + GBP {gbp_pct*100:.0f}% (${gbp_capital:.0f})")
    print(f"  {'Lot':>8} {'CAGR':>8} {'DD':>8} {'Sharpe':>8} {'Calmar':>8} {'PF':>8} {'GO':>6}")
    print(f"  {'-'*54}")

    for lot in lot_sizes:
        # Override lot size for both pairs
        PAIR_CONFIGS['EURUSD']['base_lot_size'] = lot
        PAIR_CONFIGS['GBPUSD']['base_lot_size'] = lot

        pair_results = {}
        for pair, capital in [('EURUSD', eur_capital), ('GBPUSD', gbp_capital)]:
            if capital > 0 and pair in pair_data:
                pair_results[pair] = run_single_pair(pair, pair_data[pair], capital)

        m = calculate_portfolio_metrics(pair_results, total_capital)

        go_count = sum([
            m['cagr'] >= 0.20,
            m['max_drawdown'] <= 0.20,
            m['sharpe'] >= 1.0,
            m['profit_factor'] >= 1.3,
            m['win_rate'] >= 0.45,
        ])

        marker = " ***" if go_count >= 4 and m['max_drawdown'] <= 0.20 else ""
        print(
            f"  {lot:>8.4f} "
            f"{m['cagr']*100:>7.2f}% "
            f"{m['max_drawdown']*100:>7.2f}% "
            f"{m['sharpe']:>8.2f} "
            f"{m['calmar']:>8.2f} "
            f"{m['profit_factor']:>8.2f} "
            f"{go_count:>4}/5{marker}"
        )

    # Reset lot sizes
    PAIR_CONFIGS['EURUSD']['base_lot_size'] = 0.0228
    PAIR_CONFIGS['GBPUSD']['base_lot_size'] = 0.0228


def make_aggressive_config(lot, mult, use_trend=True, compound_equity=False,
                           bb_mult=2.0, grid_spacing=0.75, sl_mult=1.5):
    """Create aggressive config with selective protections."""
    config = load_config(base_lot_size=lot)
    config['grid_strategy']['lot_multiplier'] = mult
    config['grid_strategy']['use_trend_filter'] = use_trend
    config['grid_strategy']['compound_on_equity'] = compound_equity
    config['grid_strategy']['bb_entry_mult'] = bb_mult
    config['grid_strategy']['grid_spacing_atr'] = grid_spacing
    config['grid_strategy']['sl_atr_mult'] = sl_mult
    # Disable recovery manager (chokes at high lots)
    config['recovery_manager'] = {'drawdown_threshold': 1.0}
    config['profit_protector'] = {'profit_threshold': 100.0}
    # Keep crisis detector + CB (they help)
    return config


def main():
    """Frequency and compounding experiments for max returns."""
    logger.info("=" * 80)
    logger.info("FREQUENCY + COMPOUNDING EXPERIMENTS")
    logger.info("=" * 80)

    total_capital = 500.0

    # Load EURUSD only (GBP too slow with aggressive settings)
    logger.info("\nLoading data...")
    data = load_historical_data('EURUSD', 'H1')

    # Base config: lot=0.06, mult=2.5, crisis+CB (best from phase 4: $36K, 73% CAGR)
    # Now test what happens when we boost frequency

    print("\n" + "=" * 90)
    print("FREQUENCY BOOSTERS (base: lot=0.06, mult=2.5, crisis+CB)")
    print("=" * 90)
    print(f"\n  {'Config':>40} {'CAGR':>8} {'DD':>8} {'Final$':>12} {'PF':>8} {'Sharpe':>8} {'Trades':>8}")
    print(f"  {'-'*90}")

    experiments = [
        # (label, lot, mult, use_trend, compound_equity, bb_mult, grid_spacing, sl_mult)
        # Baseline
        ("Baseline trend+BB2.0",                  0.06, 2.5, True,  False, 2.0, 0.75, 1.5),
        # Trend filter ON + frequency boosters
        ("Trend + BB(1.5)",                       0.06, 2.5, True,  False, 1.5, 0.75, 1.5),
        ("Trend + BB(1.75)",                      0.06, 2.5, True,  False, 1.75, 0.75, 1.5),
        ("Trend + tight grid(0.50)",              0.06, 2.5, True,  False, 2.0, 0.50, 1.5),
        ("Trend + tight grid(0.60)",              0.06, 2.5, True,  False, 2.0, 0.60, 1.5),
        ("Trend + BB1.5 + grid0.50",              0.06, 2.5, True,  False, 1.5, 0.50, 1.5),
        # Equity compounding variants
        ("Trend + equity compound",               0.06, 2.5, True,  True,  2.0, 0.75, 1.5),
        ("Trend + BB1.5 + equity",                0.06, 2.5, True,  True,  1.5, 0.75, 1.5),
        ("Trend + BB1.5 + eq + grid0.5",          0.06, 2.5, True,  True,  1.5, 0.50, 1.5),
        # Scale up lot with best frequency config
        ("Trend + BB1.5 lot=0.08",                0.08, 2.5, True,  False, 1.5, 0.75, 1.5),
        ("Trend + BB1.5 + eq lot=0.08",           0.08, 2.5, True,  True,  1.5, 0.75, 1.5),
        ("Trend + BB1.5 lot=0.10",                0.10, 2.5, True,  False, 1.5, 0.75, 1.5),
        ("Trend + BB1.5 + eq lot=0.10",           0.10, 2.5, True,  True,  1.5, 0.75, 1.5),
        # Grid levels boost
        ("Trend + BB1.5 + eq l=0.08 mult3",      0.08, 3.0, True,  True,  1.5, 0.75, 1.5),
        ("Trend + BB1.5 + eq l=0.10 mult3",      0.10, 3.0, True,  True,  1.5, 0.75, 1.5),
    ]

    for label, lot, mult, use_trend, comp_eq, bb, gs, sl in experiments:
        config = make_aggressive_config(lot, mult, use_trend, comp_eq, bb, gs, sl)
        bt = ProtectedGridBacktester(
            initial_balance=total_capital, config=config,
            spread_pips=0.7, slippage_pips=0.2,
            pip_size=0.0001, pip_value_per_lot=10.0,
        )
        r = bt.run_backtest(data)
        print(
            f"  {label:>40} "
            f"{r.cagr*100:>7.2f}% "
            f"{r.max_drawdown*100:>7.2f}% "
            f"${r.final_balance:>11.2f} "
            f"{r.profit_factor:>8.2f} "
            f"{r.sharpe_ratio:>8.2f} "
            f"{r.total_trades:>8}"
        )


if __name__ == '__main__':
    main()
