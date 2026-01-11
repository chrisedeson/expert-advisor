#!/usr/bin/env python3
"""
Generate Comprehensive Performance Report

Creates detailed HTML and text reports with full analytics.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from src.strategies import EnhancedTrendFollowingStrategy
from src.backtesting import BacktestEngine
from src.analytics import PerformanceMetrics, ReportGenerator


def main():
    """Generate comprehensive performance report."""

    logger.info("=" * 70)
    logger.info("GENERATING COMPREHENSIVE PERFORMANCE REPORT")
    logger.info("=" * 70)

    # Load data
    data_file = Path(__file__).parent.parent / "data" / "raw" / "EURUSDm_H4.parquet"

    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        return

    logger.info(f"\n📁 Loading data from {data_file.name}")
    data = pd.read_parquet(data_file)
    logger.info(f"Loaded {len(data)} candles: {data.index[0].date()} to {data.index[-1].date()}")

    # Initialize strategy with optimized parameters
    logger.info("\n🔧 Initializing Enhanced Strategy...")
    strategy = EnhancedTrendFollowingStrategy(
        fast_ma_period=40,
        slow_ma_period=186,
        adx_period=15,
        adx_threshold=20.55,
        atr_period=15,
        use_session_filter=False,
        ma_type='ema',
        volatility_lookback=100,
        volatility_threshold=2.5,
        enable_regime_filter=True,
    )

    logger.info(f"Strategy: {strategy.name}")

    # Generate signals
    logger.info("\n📊 Generating signals...")
    signals = strategy.generate_signals(data)

    long_signals = (signals == 1).sum()
    short_signals = (signals == -1).sum()
    flat_signals = (signals == 0).sum()

    logger.info(f"Signals: {long_signals} long, {short_signals} short, {flat_signals} flat")

    # Run backtest
    logger.info("\n🚀 Running backtest...")
    engine = BacktestEngine(
        initial_capital=100.0,
        risk_per_trade=0.01,
        max_drawdown=0.15,
    )

    results = engine.run(
        data=data,
        signals=signals,
        symbol='EURUSD',
    )

    logger.info(f"Backtest complete: {results['metrics']['num_trades']} trades")

    # Calculate comprehensive metrics
    logger.info("\n📈 Calculating performance metrics...")

    perf_metrics = PerformanceMetrics(
        equity_curve=results['equity_curve'],
        trades=results['trades'],
    )

    all_metrics = perf_metrics.calculate_all_metrics()

    # Display summary
    summary = perf_metrics.generate_summary()
    logger.info("\n" + summary)

    # Generate reports
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING REPORTS")
    logger.info("=" * 70)

    output_dir = Path(__file__).parent.parent / "reports"
    output_dir.mkdir(exist_ok=True, parents=True)

    # 1. HTML Report
    logger.info("\n📄 Generating HTML report...")
    report_gen = ReportGenerator()

    html_file = output_dir / "performance_report.html"
    report_gen.generate_html_report(
        metrics=all_metrics,
        equity_curve=results['equity_curve'],
        trades=results['trades'],
        strategy_name=strategy.name,
        output_path=html_file,
    )

    logger.info(f"✅ HTML report: {html_file}")
    logger.info(f"   Open in browser: file://{html_file.absolute()}")

    # 2. Text Report
    logger.info("\n📄 Generating text report...")
    text_file = output_dir / "performance_report.txt"
    with open(text_file, 'w') as f:
        f.write(summary)
        f.write("\n\n")
        f.write(strategy.describe())

    logger.info(f"✅ Text report: {text_file}")

    # 3. Metrics CSV
    logger.info("\n📄 Exporting metrics to CSV...")
    metrics_file = output_dir / "performance_metrics.csv"

    metrics_df = pd.DataFrame([all_metrics])
    metrics_df.to_csv(metrics_file, index=False)

    logger.info(f"✅ Metrics CSV: {metrics_file}")

    # 4. Trades CSV
    logger.info("\n📄 Exporting trades to CSV...")
    trades_file = output_dir / "all_trades.csv"
    results['trades'].to_csv(trades_file, index=False)

    logger.info(f"✅ Trades CSV: {trades_file}")

    # 5. Equity Curve CSV
    logger.info("\n📄 Exporting equity curve to CSV...")
    equity_file = output_dir / "equity_curve.csv"
    results['equity_curve'].to_csv(equity_file)

    logger.info(f"✅ Equity curve CSV: {equity_file}")

    # Final Summary
    logger.info("\n" + "=" * 70)
    logger.info("REPORT GENERATION COMPLETE")
    logger.info("=" * 70)

    logger.info(f"\n📊 Key Metrics:")
    logger.info(f"  Total Return:   {all_metrics['total_return']*100:>8.2f}%")
    logger.info(f"  CAGR:           {all_metrics['cagr']*100:>8.2f}%")
    logger.info(f"  Sharpe Ratio:   {all_metrics['sharpe_ratio']:>8.2f}")
    logger.info(f"  Max Drawdown:   {all_metrics['max_drawdown']*100:>8.2f}%")
    logger.info(f"  Win Rate:       {all_metrics['win_rate']*100:>8.2f}%")
    logger.info(f"  Profit Factor:  {all_metrics['profit_factor']:>8.2f}")

    logger.info(f"\n📁 Generated Files:")
    logger.info(f"  1. {html_file.name} (Open in browser)")
    logger.info(f"  2. {text_file.name}")
    logger.info(f"  3. {metrics_file.name}")
    logger.info(f"  4. {trades_file.name}")
    logger.info(f"  5. {equity_file.name}")

    logger.info("\n" + "=" * 70)


if __name__ == "__main__":
    main()
