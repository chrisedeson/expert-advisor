#!/usr/bin/env python3
"""
Test Backtesting Engine

Quick smoke test to verify the backtesting engine works with our data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger

from src.backtesting import BacktestEngine, TransactionCostModel


def create_simple_signals(data: pd.DataFrame, fast_period: int = 20, slow_period: int = 50) -> pd.Series:
    """
    Create simple moving average crossover signals.

    Args:
        data: OHLCV DataFrame
        fast_period: Fast MA period
        slow_period: Slow MA period

    Returns:
        Series with signals {-1, 0, 1}
    """
    # Calculate moving averages
    fast_ma = data['close'].rolling(fast_period).mean()
    slow_ma = data['close'].rolling(slow_period).mean()

    # Generate signals
    signals = pd.Series(0, index=data.index)
    signals[fast_ma > slow_ma] = 1   # Long when fast > slow
    signals[fast_ma < slow_ma] = -1  # Short when fast < slow

    return signals


def main():
    """Test backtesting engine with real data."""

    logger.info("=" * 60)
    logger.info("Testing Backtesting Engine")
    logger.info("=" * 60)

    # Load data
    data_file = Path(__file__).parent.parent / "data" / "raw" / "EURUSDm_H4.parquet"

    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        logger.error("Run export_mt5_data_windows.py first!")
        return

    logger.info(f"Loading data from {data_file}")
    data = pd.read_parquet(data_file)

    logger.info(f"Loaded {len(data)} candles from {data.index[0]} to {data.index[-1]}")

    # Create simple MA crossover signals
    logger.info("Generating signals (20/50 MA crossover)...")
    signals = create_simple_signals(data, fast_period=20, slow_period=50)

    # Count signals
    long_signals = (signals == 1).sum()
    short_signals = (signals == -1).sum()
    flat_signals = (signals == 0).sum()

    logger.info(
        f"Signals: {long_signals} long, {short_signals} short, {flat_signals} flat"
    )

    # Initialize backtesting engine
    logger.info("\nInitializing backtesting engine...")
    engine = BacktestEngine(
        initial_capital=100.0,  # $100 starting capital
        risk_per_trade=0.01,     # 1% risk per trade
        max_drawdown=0.15,       # 15% max drawdown
    )

    # Run backtest
    logger.info("\nRunning backtest...")
    results = engine.run(
        data=data,
        signals=signals,
        symbol='EURUSD',
    )

    # Display results
    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)

    metrics = results['metrics']

    logger.info("\n📊 Performance Metrics:")
    logger.info(f"  Initial Capital:    ${engine.initial_capital:.2f}")
    logger.info(f"  Final Equity:       ${metrics['final_equity']:.2f}")
    logger.info(f"  Total Return:       {metrics['total_return']*100:.2f}%")
    logger.info(f"  CAGR:               {metrics['cagr']*100:.2f}%")
    logger.info(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown:       {metrics['max_drawdown']*100:.2f}%")

    logger.info("\n📈 Trade Statistics:")
    logger.info(f"  Number of Trades:   {metrics['num_trades']}")
    logger.info(f"  Win Rate:           {metrics['win_rate']*100:.2f}%")
    logger.info(f"  Average Win:        {metrics['avg_win']*100:.2f}%")
    logger.info(f"  Average Loss:       {metrics['avg_loss']*100:.2f}%")
    logger.info(f"  Profit Factor:      {metrics['profit_factor']:.2f}")

    # Show sample trades
    trades = results['trades']
    if len(trades) > 0:
        logger.info("\n📋 Sample Trades (first 5):")
        logger.info(trades.head(5).to_string())

    # Show equity curve sample
    equity_curve = results['equity_curve']
    logger.info("\n💰 Equity Curve (first 5 rows):")
    logger.info(equity_curve.head(5).to_string())

    logger.info("\n" + "=" * 60)
    logger.info("✅ Backtesting engine test complete!")
    logger.info("=" * 60)

    # Save results for inspection
    output_dir = Path(__file__).parent.parent / "reports"
    output_dir.mkdir(exist_ok=True, parents=True)

    output_file = output_dir / "test_backtest_results.csv"
    equity_curve.to_csv(output_file)
    logger.info(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
