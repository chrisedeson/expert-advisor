#!/usr/bin/env python3
"""
Comprehensive Protected System Test

Tests the full Smart Grid EA with all 8 protection layers against historical data.

Tests include:
1. Full period backtest (2019-2025)
2. COVID crash stress test (Feb-Apr 2020)
3. Protection vs No-Protection comparison
4. Monte Carlo robustness analysis (future)

This validates that protections actually save capital and improve risk metrics.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.backtesting.protected_grid_engine import (
    ProtectedGridBacktester,
    print_backtest_summary,
    BacktestResult,
)


def load_config() -> dict:
    """Load grid-optimized configuration for backtesting."""
    # Always use grid-tuned defaults (config.yaml has different structure)
    return {
            'grid_strategy': {
                'base_lot_size': 0.0228,
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
                'profit_threshold': 1.0,  # Disable for compounding strategy
            },
        }


def load_historical_data(pair: str = 'EURUSD', timeframe: str = 'H1') -> pd.DataFrame:
    """
    Load historical data with ATR calculated.

    Args:
        pair: Currency pair (e.g., 'EURUSD', 'GBPUSD')
        timeframe: Timeframe (H1, H4, D)

    Returns:
        DataFrame with OHLCV + ATR
    """
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / f'{pair}_{timeframe}.parquet'

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please run download script first:")
        logger.info("  python scripts/download_oanda_data.py")
        sys.exit(1)

    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

    # Calculate ATR if not present
    if 'atr' not in df.columns:
        logger.info("Calculating ATR...")
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()
        df.drop('tr', axis=1, inplace=True)

    logger.success(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    return df


def resample_to_h4(h1_data: pd.DataFrame) -> pd.DataFrame:
    """Resample H1 data to H4 for larger moves and reduced cost impact."""
    h4 = h1_data.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna()

    # Recalculate ATR on H4 bars
    h4['tr'] = np.maximum(
        h4['high'] - h4['low'],
        np.maximum(
            abs(h4['high'] - h4['close'].shift(1)),
            abs(h4['low'] - h4['close'].shift(1))
        )
    )
    h4['atr'] = h4['tr'].rolling(14).mean()
    h4.drop('tr', axis=1, inplace=True)
    h4.dropna(inplace=True)

    logger.info(f"Resampled to H4: {len(h4)} bars (from {len(h1_data)} H1 bars)")
    return h4


def test_full_period(data: pd.DataFrame, config: dict, initial_balance: float = 500.0):
    """Test full historical period with protections"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: FULL PERIOD BACKTEST WITH PROTECTIONS")
    logger.info("=" * 80)

    backtester = ProtectedGridBacktester(
        initial_balance=initial_balance,
        config=config,
        spread_pips=1.2,
        commission_per_lot=0.0,
        slippage_pips=0.2,
    )

    result = backtester.run_backtest(data)

    print_backtest_summary(result)

    return result


def test_covid_crash(data: pd.DataFrame, config: dict, initial_balance: float = 500.0):
    """Test COVID crash period specifically (Feb-Apr 2020)"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: COVID CRASH STRESS TEST (Feb-Apr 2020)")
    logger.info("=" * 80)

    covid_start = datetime(2020, 2, 1)
    covid_end = datetime(2020, 4, 30)

    backtester = ProtectedGridBacktester(
        initial_balance=initial_balance,
        config=config,
        spread_pips=1.2,
        commission_per_lot=0.0,
        slippage_pips=0.2,
    )

    result = backtester.run_backtest(data, covid_start, covid_end)

    print_backtest_summary(result)

    logger.info("\n🧪 COVID CRASH ANALYSIS:")
    logger.info(f"   Starting: Feb 2020 with ${initial_balance:.2f}")
    logger.info(f"   Ending:   Apr 2020 with ${result.final_balance:.2f}")
    logger.info(f"   Drawdown: {result.max_drawdown*100:.2f}%")
    logger.info(f"   Capital Saved by Protections: ${result.capital_saved_estimate:.2f}")

    if result.max_drawdown < 0.10:
        logger.success("   ✅ SURVIVED with <10% loss (EXCELLENT)")
    elif result.max_drawdown < 0.15:
        logger.success("   ✅ SURVIVED with <15% loss (GOOD)")
    elif result.max_drawdown < 0.25:
        logger.warning("   ⚠️  Survived but >15% loss (ACCEPTABLE)")
    else:
        logger.error("   ❌ Large loss >25% (NEEDS IMPROVEMENT)")

    return result


def test_without_protections(data: pd.DataFrame, config: dict, initial_balance: float = 500.0):
    """
    Test WITHOUT protections (for comparison).

    Creates a config with all protections disabled.
    """
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: BACKTEST WITHOUT PROTECTIONS (Comparison)")
    logger.info("=" * 80)

    # Disable ALL protections by setting impossible thresholds
    unprotected_config = config.copy()
    unprotected_config['volatility_filter'] = {
        'normal_threshold': 100.0,
        'crisis_threshold': 100.0,
        'cooldown_days': 0,
    }
    unprotected_config['circuit_breaker'] = {
        'daily_limit': 1.0,
        'weekly_limit': 1.0,
        'monthly_limit': 1.0,
    }
    unprotected_config['crisis_detector'] = {
        'volatility_spike_threshold': 100.0,
        'rapid_drawdown_threshold': 1.0,
        'consecutive_stops_threshold': 999,
        'gap_threshold_pips': 99999.0,
    }
    unprotected_config['recovery_manager'] = {
        'drawdown_threshold': 1.0,
    }
    unprotected_config['profit_protector'] = {
        'profit_threshold': 100.0,
    }

    backtester = ProtectedGridBacktester(
        initial_balance=initial_balance,
        config=unprotected_config,
        spread_pips=1.2,
        commission_per_lot=0.0,
        slippage_pips=0.2,
    )

    result = backtester.run_backtest(data)

    print_backtest_summary(result)

    return result


def compare_results(protected: BacktestResult, unprotected: BacktestResult):
    """Compare protected vs unprotected results"""
    logger.info("\n" + "=" * 80)
    logger.info("PROTECTED vs UNPROTECTED COMPARISON")
    logger.info("=" * 80)

    def calc_improvement(protected_val, unprotected_val, lower_is_better=False):
        """Calculate improvement percentage"""
        if unprotected_val == 0:
            return 0
        if lower_is_better:
            return ((unprotected_val - protected_val) / unprotected_val) * 100
        else:
            return ((protected_val - unprotected_val) / abs(unprotected_val)) * 100

    print(f"\n{'Metric':<30} {'Protected':<15} {'Unprotected':<15} {'Improvement':<15}")
    print("-" * 75)

    # Returns
    print(f"{'Final Balance':<30} ${protected.final_balance:>13,.2f} ${unprotected.final_balance:>13,.2f} "
          f"{calc_improvement(protected.final_balance, unprotected.final_balance):>13.1f}%")

    print(f"{'Total Return':<30} {protected.total_return*100:>13.2f}% {unprotected.total_return*100:>13.2f}% "
          f"{calc_improvement(protected.total_return, unprotected.total_return):>13.1f}%")

    print(f"{'CAGR':<30} {protected.cagr*100:>13.2f}% {unprotected.cagr*100:>13.2f}% "
          f"{calc_improvement(protected.cagr, unprotected.cagr):>13.1f}%")

    # Risk
    print(f"{'Max Drawdown':<30} {protected.max_drawdown*100:>13.2f}% {unprotected.max_drawdown*100:>13.2f}% "
          f"{calc_improvement(protected.max_drawdown, unprotected.max_drawdown, True):>13.1f}%")

    print(f"{'Sharpe Ratio':<30} {protected.sharpe_ratio:>13.2f} {unprotected.sharpe_ratio:>13.2f} "
          f"{calc_improvement(protected.sharpe_ratio, unprotected.sharpe_ratio):>13.1f}%")

    print(f"{'Sortino Ratio':<30} {protected.sortino_ratio:>13.2f} {unprotected.sortino_ratio:>13.2f} "
          f"{calc_improvement(protected.sortino_ratio, unprotected.sortino_ratio):>13.1f}%")

    print(f"{'Calmar Ratio':<30} {protected.calmar_ratio:>13.2f} {unprotected.calmar_ratio:>13.2f} "
          f"{calc_improvement(protected.calmar_ratio, unprotected.calmar_ratio):>13.1f}%")

    # Trading
    print(f"{'Win Rate':<30} {protected.win_rate*100:>13.1f}% {unprotected.win_rate*100:>13.1f}% "
          f"{calc_improvement(protected.win_rate, unprotected.win_rate):>13.1f}%")

    print(f"{'Profit Factor':<30} {protected.profit_factor:>13.2f} {unprotected.profit_factor:>13.2f} "
          f"{calc_improvement(protected.profit_factor, unprotected.profit_factor):>13.1f}%")

    print("\n" + "=" * 80)
    logger.success("PROTECTION SYSTEM VERDICT:")

    # Calculate overall benefit
    dd_improvement = calc_improvement(protected.max_drawdown, unprotected.max_drawdown, True)
    sharpe_improvement = calc_improvement(protected.sharpe_ratio, unprotected.sharpe_ratio)

    if dd_improvement > 30 and sharpe_improvement > 20:
        logger.success("✅ EXCELLENT - Protections significantly improve risk-adjusted returns")
    elif dd_improvement > 20 or sharpe_improvement > 10:
        logger.success("✅ GOOD - Protections provide meaningful benefit")
    elif dd_improvement > 10:
        logger.info("⚠️  MODERATE - Some benefit, but could be improved")
    else:
        logger.warning("❌ LIMITED - Protections not providing significant value")

    logger.info(f"\n💰 Capital saved by protections: ${protected.capital_saved_estimate:.2f}")
    logger.info("=" * 80)


def main():
    """Run comprehensive protected system tests"""

    logger.info("=" * 80)
    logger.info("SMART GRID EA - COMPREHENSIVE PROTECTION SYSTEM TEST")
    logger.info("=" * 80)
    logger.info("Testing all 8 protection layers against historical data")
    logger.info("=" * 80)

    # Load config
    config = load_config()

    # Load data
    pair = 'EURUSD'
    timeframe = 'H1'
    initial_balance = 500.0

    logger.info(f"\nTest Configuration:")
    logger.info(f"  Pair: {pair}")
    logger.info(f"  Timeframe: {timeframe} (Asian session filter)")
    logger.info(f"  Initial Balance: ${initial_balance:.2f}")

    data = load_historical_data(pair, timeframe)

    # Run tests
    results = {}

    # Test 1: Full period with protections
    results['protected_full'] = test_full_period(data, config, initial_balance)

    # Test 2: COVID crash with protections
    results['protected_covid'] = test_covid_crash(data, config, initial_balance)

    # Test 3: Full period WITHOUT protections
    results['unprotected_full'] = test_without_protections(data, config, initial_balance)

    # Compare protected vs unprotected
    compare_results(results['protected_full'], results['unprotected_full'])

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.success("ALL TESTS COMPLETE")
    logger.info("=" * 80)

    protected = results['protected_full']
    logger.info(f"\n📊 PROTECTED SYSTEM FINAL RESULTS:")
    logger.info(f"   Starting Balance:  ${initial_balance:.2f}")
    logger.info(f"   Final Balance:     ${protected.final_balance:.2f}")
    logger.info(f"   Total Return:      {protected.total_return*100:.2f}%")
    logger.info(f"   CAGR:              {protected.cagr*100:.2f}%")
    logger.info(f"   Max Drawdown:      {protected.max_drawdown*100:.2f}%")
    logger.info(f"   Sharpe Ratio:      {protected.sharpe_ratio:.2f}")
    logger.info(f"   Win Rate:          {protected.win_rate*100:.1f}%")
    logger.info(f"   Profit Factor:     {protected.profit_factor:.2f}")

    logger.info(f"\n🛡️  PROTECTION SYSTEM STATS:")
    logger.info(f"   Protection Events:      {len(protected.protection_events)}")
    logger.info(f"   Circuit Breaker:        {protected.circuit_breaker_triggers} triggers")
    logger.info(f"   Crisis Mode:            {protected.crisis_mode_activations} activations")
    logger.info(f"   Volatility Pauses:      {protected.volatility_pauses}")
    logger.info(f"   Capital Saved:          ${protected.capital_saved_estimate:.2f}")

    # Go/No-Go decision criteria
    logger.info("\n" + "=" * 80)
    logger.info("GO/NO-GO DECISION CRITERIA")
    logger.info("=" * 80)

    go_criteria = []
    no_go_criteria = []

    # Check each criterion
    if protected.cagr >= 0.20:
        go_criteria.append(f"✅ CAGR: {protected.cagr*100:.1f}% (target: >20%)")
    else:
        no_go_criteria.append(f"❌ CAGR: {protected.cagr*100:.1f}% (target: >20%)")

    if protected.max_drawdown <= 0.20:
        go_criteria.append(f"✅ Max DD: {protected.max_drawdown*100:.1f}% (target: <20%)")
    else:
        no_go_criteria.append(f"❌ Max DD: {protected.max_drawdown*100:.1f}% (target: <20%)")

    if protected.sharpe_ratio >= 1.0:
        go_criteria.append(f"✅ Sharpe: {protected.sharpe_ratio:.2f} (target: >1.0)")
    else:
        no_go_criteria.append(f"❌ Sharpe: {protected.sharpe_ratio:.2f} (target: >1.0)")

    if protected.win_rate >= 0.45:
        go_criteria.append(f"✅ Win Rate: {protected.win_rate*100:.1f}% (target: >45%)")
    else:
        no_go_criteria.append(f"❌ Win Rate: {protected.win_rate*100:.1f}% (target: >45%)")

    if protected.profit_factor >= 1.3:
        go_criteria.append(f"✅ Profit Factor: {protected.profit_factor:.2f} (target: >1.3)")
    else:
        no_go_criteria.append(f"❌ Profit Factor: {protected.profit_factor:.2f} (target: >1.3)")

    covid_result = results['protected_covid']
    if covid_result.max_drawdown <= 0.15:
        go_criteria.append(f"✅ COVID Survival: {covid_result.max_drawdown*100:.1f}% DD (target: <15%)")
    else:
        no_go_criteria.append(f"❌ COVID Survival: {covid_result.max_drawdown*100:.1f}% DD (target: <15%)")

    # Print results
    logger.info("\nMET CRITERIA:")
    for criterion in go_criteria:
        logger.success(f"  {criterion}")

    if no_go_criteria:
        logger.info("\nFAILED CRITERIA:")
        for criterion in no_go_criteria:
            logger.error(f"  {criterion}")

    # Final verdict
    go_count = len(go_criteria)
    total_count = len(go_criteria) + len(no_go_criteria)

    logger.info(f"\nScore: {go_count}/{total_count} criteria met")

    if go_count == total_count:
        logger.success("\n🎉 VERDICT: GO FOR PAPER TRADING")
        logger.success("   All criteria met! System is ready for next phase.")
    elif go_count >= total_count * 0.75:
        logger.info("\n⚠️  VERDICT: CONDITIONAL GO")
        logger.info("   Most criteria met. Review failed criteria and decide.")
    else:
        logger.error("\n❌ VERDICT: NO-GO")
        logger.error("   Too many criteria failed. Needs optimization/improvement.")

    logger.info("=" * 80)


if __name__ == '__main__':
    main()
