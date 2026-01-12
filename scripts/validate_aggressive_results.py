#!/usr/bin/env python3
"""Validate aggressive optimization results."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from src.strategies import EnhancedTrendFollowingStrategy
from src.optimization import StrategyValidator

def main():
    """Validate the best parameters from aggressive optimization."""

    logger.info("=" * 70)
    logger.info("VALIDATING AGGRESSIVE OPTIMIZATION RESULTS")
    logger.info("=" * 70)

    # Load H1 data
    data_file = Path(__file__).parent.parent / "data" / "raw" / "EURUSDm_H1.parquet"
    logger.info(f"\n📁 Loading H1 data...")
    data = pd.read_parquet(data_file)
    logger.info(f"Loaded {len(data):,} H1 candles: {data.index[0].date()} to {data.index[-1].date()}")

    # Split data
    n = len(data)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]

    logger.info(f"\n📊 Data splits:")
    logger.info(f"  Training:   {len(train_data):,} candles ({train_data.index[0].date()} to {train_data.index[-1].date()})")
    logger.info(f"  Validation: {len(val_data):,} candles ({val_data.index[0].date()} to {val_data.index[-1].date()})")
    logger.info(f"  Test:       {len(test_data):,} candles ({test_data.index[0].date()} to {test_data.index[-1].date()})")

    # Best parameters from optimization
    best_params = {
        'fast_ma_period': 49,
        'slow_ma_period': 48,
        'adx_period': 13,
        'adx_threshold': 32.45317045529314,
        'atr_period': 20,
        'volatility_lookback': 139,
        'volatility_threshold': 3.2587646604955034,
        'use_session_filter': False,
        'ma_type': 'ema',
        'enable_regime_filter': True,
    }

    logger.info(f"\n🏆 Best Parameters (Trial 333, CAGR=5.69%):")
    for param, value in best_params.items():
        if isinstance(value, float):
            logger.info(f"  {param}: {value:.2f}")
        else:
            logger.info(f"  {param}: {value}")

    # Create strategy with best parameters
    strategy = EnhancedTrendFollowingStrategy(**best_params)

    # Create validator
    validator = StrategyValidator(
        strategy=strategy,
        initial_capital=100.0,
        risk_per_trade=0.01,
        max_drawdown=0.18,
    )

    # Validate on all three sets
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING VALIDATION ON ALL DATASETS")
    logger.info("=" * 70)

    train_results = validator.validate_on_dataset(train_data, 'Training', symbol='EURUSD')
    val_results = validator.validate_on_dataset(val_data, 'Validation', symbol='EURUSD')
    test_results = validator.validate_on_dataset(test_data, 'Test', symbol='EURUSD')

    # Display results
    logger.info("\n" + "=" * 70)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("=" * 70)

    logger.info(f"\n{'Metric':<20} {'Training':<12} {'Validation':<12} {'Test':<12}")
    logger.info("-" * 70)

    train_m = train_results['metrics']
    val_m = val_results['metrics']
    test_m = test_results['metrics']

    for metric in ['total_return', 'cagr', 'sharpe_ratio', 'max_drawdown', 'num_trades', 'win_rate']:
        train_val = train_m.get(metric, 0)
        val_val = val_m.get(metric, 0)
        test_val = test_m.get(metric, 0)

        if metric in ['total_return', 'cagr', 'max_drawdown', 'win_rate']:
            train_str = f"{train_val*100:>6.2f}%"
            val_str = f"{val_val*100:>6.2f}%"
            test_str = f"{test_val*100:>6.2f}%"
        elif metric == 'sharpe_ratio':
            train_str = f"{train_val:>6.2f}"
            val_str = f"{val_val:>6.2f}"
            test_str = f"{test_val:>6.2f}"
        else:
            train_str = f"{train_val:>6.0f}"
            val_str = f"{val_val:>6.0f}"
            test_str = f"{test_val:>6.0f}"

        logger.info(f"{metric:<20} {train_str:<12} {val_str:<12} {test_str:<12}")

    # Check for overfitting
    logger.info("\n🔍 Overfitting Check:")
    val_degradation = ((val_m['cagr'] - train_m['cagr']) / abs(train_m['cagr'])) * 100
    test_degradation = ((test_m['cagr'] - train_m['cagr']) / abs(train_m['cagr'])) * 100

    logger.info(f"  Validation degradation: {val_degradation:+.1f}%")
    logger.info(f"  Test degradation: {test_degradation:+.1f}%")

    if abs(val_degradation) < 30:
        logger.info("  ✅ Validation looks good (< 30% degradation)")
    else:
        logger.warning(f"  ⚠️  High degradation on validation set")

    # Full 6-year backtest
    logger.info("\n" + "=" * 70)
    logger.info("FULL 6-YEAR BACKTEST (2019-2025)")
    logger.info("=" * 70)

    full_results = validator.validate_on_dataset(data, 'Full Dataset', symbol='EURUSD')
    full_m = full_results['metrics']

    logger.info(f"\n📈 Full Period Performance:")
    logger.info(f"  Total Return:   {full_m['total_return']*100:>6.2f}%")
    logger.info(f"  CAGR:           {full_m['cagr']*100:>6.2f}%")
    logger.info(f"  Sharpe Ratio:   {full_m['sharpe_ratio']:>6.2f}")
    logger.info(f"  Max Drawdown:   {full_m['max_drawdown']*100:>6.2f}%")
    logger.info(f"  Win Rate:       {full_m.get('win_rate', 0)*100:>6.2f}%")
    logger.info(f"  Profit Factor:  {full_m.get('profit_factor', 0):>6.2f}")
    logger.info(f"  Total Trades:   {full_m['num_trades']:>6.0f}")

    # Go/No-Go assessment
    logger.info("\n" + "=" * 70)
    logger.info("GO/NO-GO ASSESSMENT (Based on Full 6-Year Period)")
    logger.info("=" * 70)

    criteria_met = 0
    total_criteria = 5

    logger.info(f"\n{'Criterion':<25} {'Target':<12} {'Actual':<12} {'Status':<10}")
    logger.info("-" * 70)

    # CAGR
    cagr_target = 0.12
    cagr_actual = full_m['cagr']
    cagr_pass = cagr_actual > cagr_target
    criteria_met += int(cagr_pass)
    logger.info(f"{'CAGR':<25} {'>12%':<12} {cagr_actual*100:>6.2f}% {('✅ PASS' if cagr_pass else '❌ MISS'):<10}")

    # Sharpe
    sharpe_target = 1.0
    sharpe_actual = full_m['sharpe_ratio']
    sharpe_pass = sharpe_actual > sharpe_target
    criteria_met += int(sharpe_pass)
    logger.info(f"{'Sharpe Ratio':<25} {'>1.0':<12} {sharpe_actual:>6.2f}  {('✅ PASS' if sharpe_pass else '❌ MISS'):<10}")

    # Max DD
    dd_target = -0.18
    dd_actual = full_m['max_drawdown']
    dd_pass = dd_actual > dd_target
    criteria_met += int(dd_pass)
    logger.info(f"{'Max Drawdown':<25} {'>-18%':<12} {dd_actual*100:>6.2f}% {('✅ PASS' if dd_pass else '❌ MISS'):<10}")

    # Win Rate
    wr_target = 0.45
    wr_actual = full_m.get('win_rate', 0)
    wr_pass = wr_actual > wr_target
    criteria_met += int(wr_pass)
    logger.info(f"{'Win Rate':<25} {'>45%':<12} {wr_actual*100:>6.2f}% {('✅ PASS' if wr_pass else '❌ MISS'):<10}")

    # Min Trades
    trades_target = 200
    trades_actual = full_m['num_trades']
    trades_pass = trades_actual > trades_target
    criteria_met += int(trades_pass)
    logger.info(f"{'Min Trades':<25} {'>200':<12} {trades_actual:>6.0f}    {('✅ PASS' if trades_pass else '❌ MISS'):<10}")

    logger.info("\n" + "=" * 70)
    logger.info(f"📊 VERDICT: {criteria_met}/{total_criteria} Criteria Met")
    logger.info("=" * 70)

    if criteria_met >= 4:
        logger.info("\n🎉 ✅ STRATEGY READY FOR PAPER TRADING!")
        logger.info("   Strategy meets minimum requirements. Proceed to Phase 8.")
    elif criteria_met >= 3:
        logger.info("\n⚠️  STRATEGY SHOWS PROMISE")
        logger.info("   Consider running Phase 6 robustness tests before proceeding.")
    else:
        logger.info("\n❌ STRATEGY NEEDS MORE WORK")
        logger.info(f"   Only {criteria_met}/5 criteria met. Consider:")
        logger.info("   1. Test on other pairs (GBPUSD, USDJPY, AUDUSD)")
        logger.info("   2. Try different strategy approach (mean reversion?)")
        logger.info("   3. Combine multiple strategies")

    # Save results
    output_dir = Path(__file__).parent.parent / "reports"

    # Save best parameters
    params_file = output_dir / "best_parameters_h1_aggressive.txt"
    with open(params_file, 'w') as f:
        f.write("Best Parameters from Aggressive H1 Optimization\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Objective: Maximize CAGR\n")
        f.write(f"Training CAGR: {train_m['cagr']*100:.2f}%\n")
        f.write(f"Validation CAGR: {val_m['cagr']*100:.2f}%\n")
        f.write(f"Full Period CAGR: {full_m['cagr']*100:.2f}%\n\n")
        f.write("Parameters:\n")
        for param, value in best_params.items():
            if isinstance(value, float):
                f.write(f"  {param}: {value:.4f}\n")
            else:
                f.write(f"  {param}: {value}\n")
        f.write(f"\nFull Period Results:\n")
        f.write(f"  Total Return: {full_m['total_return']*100:.2f}%\n")
        f.write(f"  CAGR: {full_m['cagr']*100:.2f}%\n")
        f.write(f"  Sharpe: {full_m['sharpe_ratio']:.2f}\n")
        f.write(f"  Max DD: {full_m['max_drawdown']*100:.2f}%\n")
        f.write(f"  Win Rate: {full_m.get('win_rate', 0)*100:.2f}%\n")
        f.write(f"  Trades: {full_m['num_trades']:.0f}\n")
        f.write(f"\nGo/No-Go: {criteria_met}/5 criteria met\n")

    logger.info(f"\n💾 Saved results: {params_file}")

    logger.info("\n" + "=" * 70)


if __name__ == "__main__":
    main()
