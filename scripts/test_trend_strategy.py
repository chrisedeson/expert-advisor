#!/usr/bin/env python3
"""
Test Trend-Following Strategy

Run the trend-following strategy on historical data to verify it works.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from src.strategies import TrendFollowingStrategy
from src.backtesting import BacktestEngine


def main():
    """Test trend-following strategy."""

    logger.info("=" * 60)
    logger.info("Testing Trend-Following Strategy")
    logger.info("=" * 60)

    # Load data
    data_file = Path(__file__).parent.parent / "data" / "raw" / "EURUSDm_H4.parquet"

    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        return

    logger.info(f"Loading data from {data_file}")
    data = pd.read_parquet(data_file)
    logger.info(f"Loaded {len(data)} candles from {data.index[0]} to {data.index[-1]}")

    # Split data: Train on 2019-2022, Test on 2023-2024
    train_end = pd.Timestamp('2022-12-31')
    train_data = data[data.index <= train_end]
    test_data = data[data.index > train_end]

    logger.info(f"\nTrain period: {train_data.index[0]} to {train_data.index[-1]} ({len(train_data)} candles)")
    logger.info(f"Test period: {test_data.index[0]} to {test_data.index[-1]} ({len(test_data)} candles)")

    # Initialize strategy with default parameters
    logger.info("\nInitializing Trend-Following Strategy...")
    strategy = TrendFollowingStrategy(
        fast_ma_period=20,
        slow_ma_period=50,
        adx_period=14,
        adx_threshold=25.0,
        atr_period=14,
        use_session_filter=False,  # Disable for H4 data (includes all sessions)
        ma_type='ema',
    )

    logger.info("\n" + strategy.describe())

    # Generate signals
    logger.info("Generating signals on training data...")
    train_signals = strategy.generate_signals(train_data)

    long_signals = (train_signals == 1).sum()
    short_signals = (train_signals == -1).sum()
    flat_signals = (train_signals == 0).sum()

    logger.info(f"Train signals: {long_signals} long, {short_signals} short, {flat_signals} flat")

    # Run backtest on training data
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING DATA BACKTEST (2019-2022)")
    logger.info("=" * 60)

    engine = BacktestEngine(
        initial_capital=100.0,
        risk_per_trade=0.01,
        max_drawdown=0.15,
    )

    train_results = engine.run(
        data=train_data,
        signals=train_signals,
        symbol='EURUSD',
    )

    # Display training results
    train_metrics = train_results['metrics']
    logger.info("\n📊 Training Performance:")
    logger.info(f"  Total Return:       {train_metrics['total_return']*100:.2f}%")
    logger.info(f"  CAGR:               {train_metrics['cagr']*100:.2f}%")
    logger.info(f"  Sharpe Ratio:       {train_metrics['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown:       {train_metrics['max_drawdown']*100:.2f}%")
    logger.info(f"  Number of Trades:   {train_metrics['num_trades']}")
    logger.info(f"  Win Rate:           {train_metrics['win_rate']*100:.2f}%")
    logger.info(f"  Profit Factor:      {train_metrics['profit_factor']:.2f}")

    # Run backtest on test data (out-of-sample)
    logger.info("\n" + "=" * 60)
    logger.info("TEST DATA BACKTEST (2023-2024) - OUT OF SAMPLE")
    logger.info("=" * 60)

    test_signals = strategy.generate_signals(test_data)

    test_results = engine.run(
        data=test_data,
        signals=test_signals,
        symbol='EURUSD',
    )

    # Display test results
    test_metrics = test_results['metrics']
    logger.info("\n📊 Test Performance (Out-of-Sample):")
    logger.info(f"  Total Return:       {test_metrics['total_return']*100:.2f}%")
    logger.info(f"  CAGR:               {test_metrics['cagr']*100:.2f}%")
    logger.info(f"  Sharpe Ratio:       {test_metrics['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown:       {test_metrics['max_drawdown']*100:.2f}%")
    logger.info(f"  Number of Trades:   {test_metrics['num_trades']}")
    logger.info(f"  Win Rate:           {test_metrics['win_rate']*100:.2f}%")
    logger.info(f"  Profit Factor:      {test_metrics['profit_factor']:.2f}")

    # Compare train vs test
    logger.info("\n" + "=" * 60)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("=" * 60)

    logger.info(f"\n{'Metric':<25} {'Training':<15} {'Test':<15} {'Degradation':<15}")
    logger.info("-" * 70)

    metrics_to_compare = [
        ('Total Return', 'total_return', '%'),
        ('CAGR', 'cagr', '%'),
        ('Sharpe Ratio', 'sharpe_ratio', ''),
        ('Max Drawdown', 'max_drawdown', '%'),
        ('Win Rate', 'win_rate', '%'),
        ('Profit Factor', 'profit_factor', ''),
    ]

    for label, key, unit in metrics_to_compare:
        train_val = train_metrics[key]
        test_val = test_metrics[key]

        if unit == '%':
            train_str = f"{train_val*100:.2f}%"
            test_str = f"{test_val*100:.2f}%"
        else:
            train_str = f"{train_val:.2f}"
            test_str = f"{test_val:.2f}"

        # Calculate degradation
        if train_val != 0:
            degradation = ((test_val - train_val) / abs(train_val)) * 100
            deg_str = f"{degradation:+.1f}%"
        else:
            deg_str = "N/A"

        logger.info(f"{label:<25} {train_str:<15} {test_str:<15} {deg_str:<15}")

    # Verdict
    logger.info("\n" + "=" * 60)
    logger.info("VERDICT")
    logger.info("=" * 60)

    # Check go/no-go criteria
    passed = []
    failed = []

    if test_metrics['sharpe_ratio'] > 0.5:
        passed.append("✅ Test Sharpe > 0.5")
    else:
        failed.append(f"❌ Test Sharpe < 0.5 ({test_metrics['sharpe_ratio']:.2f})")

    if test_metrics['num_trades'] >= 20:
        passed.append("✅ Sufficient trades for statistical significance")
    else:
        failed.append(f"❌ Too few trades ({test_metrics['num_trades']})")

    if test_metrics['total_return'] > 0:
        passed.append("✅ Positive returns on test data")
    else:
        failed.append(f"❌ Negative returns ({test_metrics['total_return']*100:.2f}%)")

    if test_metrics['max_drawdown'] > -0.20:
        passed.append("✅ Drawdown within acceptable range")
    else:
        failed.append(f"❌ Excessive drawdown ({test_metrics['max_drawdown']*100:.2f}%)")

    logger.info("\nPassed Criteria:")
    for item in passed:
        logger.info(f"  {item}")

    if failed:
        logger.info("\nFailed Criteria:")
        for item in failed:
            logger.info(f"  {item}")

    logger.info("\n" + "-" * 60)
    if len(failed) == 0:
        logger.info("🎉 STRATEGY LOOKS PROMISING - Ready for optimization!")
    elif len(failed) <= 2:
        logger.info("⚠️  STRATEGY SHOWS POTENTIAL - Needs parameter tuning")
    else:
        logger.info("❌ STRATEGY NEEDS WORK - Consider revising logic")

    logger.info("=" * 60)

    # Save results
    output_dir = Path(__file__).parent.parent / "reports"
    output_dir.mkdir(exist_ok=True, parents=True)

    train_file = output_dir / "trend_strategy_train_results.csv"
    test_file = output_dir / "trend_strategy_test_results.csv"

    train_results['equity_curve'].to_csv(train_file)
    test_results['equity_curve'].to_csv(test_file)

    logger.info(f"\n💾 Results saved to:")
    logger.info(f"  Training: {train_file}")
    logger.info(f"  Test: {test_file}")


if __name__ == "__main__":
    main()
