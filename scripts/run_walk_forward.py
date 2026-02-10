#!/usr/bin/env python3
"""
Run Walk-Forward Analysis

Test strategy robustness with rolling optimization windows.
This is the PRIMARY defense against overfitting.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from src.strategies import EnhancedTrendFollowingStrategy
from src.optimization import WalkForwardAnalyzer


def main():
    """Run walk-forward analysis on enhanced strategy."""

    logger.info("=" * 70)
    logger.info("WALK-FORWARD ANALYSIS WITH ENHANCED STRATEGY")
    logger.info("=" * 70)

    # Load data
    data_file = Path(__file__).parent.parent / "data" / "raw" / "EURUSDm_H4.parquet"

    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        return

    logger.info(f"\n📁 Loading data from {data_file.name}")
    data = pd.read_parquet(data_file)
    logger.info(
        f"Loaded {len(data)} candles: {data.index[0].date()} to {data.index[-1].date()}"
    )

    # Use optimized parameters from Phase 4 as starting point
    logger.info("\n🔧 Initializing Enhanced Strategy...")
    strategy = EnhancedTrendFollowingStrategy(
        fast_ma_period=40,
        slow_ma_period=186,
        adx_period=15,
        adx_threshold=20.55,
        atr_period=15,
        use_session_filter=False,
        ma_type='ema',
        # Volatility regime parameters
        volatility_lookback=100,
        volatility_threshold=2.5,
        enable_regime_filter=True,
    )

    logger.info(f"Strategy: {strategy}")
    logger.info(strategy.describe())

    # Check regime statistics first
    logger.info("\n📊 Volatility Regime Analysis...")
    regime_stats = strategy.get_regime_stats(data)
    logger.info(f"Total Bars: {regime_stats['total_bars']}")
    logger.info(f"Normal:   {regime_stats['normal_bars']:>6} bars ({regime_stats['normal_pct']:.1f}%)")
    logger.info(f"Elevated: {regime_stats['elevated_bars']:>6} bars ({regime_stats['elevated_pct']:.1f}%)")
    logger.info(f"Crisis:   {regime_stats['crisis_bars']:>6} bars ({regime_stats['crisis_pct']:.1f}%)")

    if regime_stats['crisis_pct'] > 20:
        logger.warning(
            f"⚠️  High crisis percentage ({regime_stats['crisis_pct']:.1f}%). "
            f"Consider adjusting volatility_threshold."
        )

    # Initialize walk-forward analyzer
    logger.info("\n" + "=" * 70)
    logger.info("CONFIGURING WALK-FORWARD ANALYSIS")
    logger.info("=" * 70)

    analyzer = WalkForwardAnalyzer(
        strategy=strategy,
        data=data,
        train_period_months=6,   # 6 months optimization
        test_period_months=3,    # 3 months testing
        initial_capital=100.0,
        risk_per_trade=0.01,
        max_drawdown=0.15,
        optimization_trials=50,  # Reduced for speed (use 100-200 for production)
    )

    # Create windows
    logger.info(f"\n📅 Creating rolling windows...")
    windows = analyzer.create_windows()

    logger.info(f"\nWindows Created: {len(windows)}")
    for i, window in enumerate(windows[:3]):  # Show first 3
        logger.info(
            f"  Window {i}: "
            f"{window['train_start'].date()} → {window['test_end'].date()} "
            f"(train={window['train_bars']}, test={window['test_bars']})"
        )
    if len(windows) > 3:
        logger.info(f"  ... and {len(windows)-3} more windows")

    # Run walk-forward analysis
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING WALK-FORWARD ANALYSIS")
    logger.info("=" * 70)
    logger.info("\nThis will take several minutes...")
    logger.info("Each window: optimize (50 trials) → test → repeat\n")

    results = analyzer.run_analysis(symbol='EURUSD')

    # Generate report
    report = analyzer.generate_report()
    logger.info("\n" + report)

    # Save results
    logger.info("\n" + "=" * 70)
    logger.info("SAVING RESULTS")
    logger.info("=" * 70)

    output_dir = Path(__file__).parent.parent / "reports"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save detailed results
    results_file = output_dir / "walk_forward_results.csv"
    results_df = pd.DataFrame([
        {
            'window_id': r['window_id'],
            'test_start': r['test_start'],
            'test_end': r['test_end'],
            'train_sharpe': r['train_sharpe'],
            'test_sharpe': r['test_sharpe'],
            'test_return': r['test_metrics']['total_return'],
            'test_drawdown': r['test_metrics']['max_drawdown'],
            'test_trades': r['test_metrics']['num_trades'],
            'degradation_pct': r['degradation_pct'],
        }
        for r in results['windows']
    ])
    results_df.to_csv(results_file, index=False)
    logger.info(f"💾 Detailed results: {results_file}")

    # Save report
    report_file = output_dir / "walk_forward_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    logger.info(f"💾 Full report: {report_file}")

    # Save combined equity curve
    equity_file = output_dir / "walk_forward_equity.csv"
    combined_equity = analyzer.get_combined_equity_curve()
    combined_equity.to_csv(equity_file)
    logger.info(f"💾 Combined equity: {equity_file}")

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 70)

    agg = results['aggregate']
    logger.info(f"\n📊 Performance Across {agg['n_windows']} Windows:")
    logger.info(f"  Avg Test Sharpe:    {agg['avg_test_sharpe']:.4f}")
    logger.info(f"  Avg Test Return:    {agg['avg_test_return']*100:.2f}%")
    logger.info(f"  Positive Windows:   {agg['positive_windows']}/{agg['n_windows']} ({agg['positive_pct']:.1f}%)")
    logger.info(f"  Stability Ratio:    {agg['stability_ratio']:.2f}")
    logger.info(f"  Avg Degradation:    {agg['avg_degradation']:+.1f}%")

    # Verdict
    logger.info("\n" + "-" * 70)
    if agg['avg_test_sharpe'] > 0.5 and agg['positive_pct'] >= 60:
        logger.info("🎉 ✅ STRATEGY PASSES WALK-FORWARD ANALYSIS!")
        logger.info("Ready for multi-pair testing (Phase 6)")
    elif agg['avg_test_sharpe'] > 0.3 and agg['positive_pct'] >= 50:
        logger.info("⚠️  STRATEGY SHOWS PROMISE")
        logger.info("Consider further optimization or parameter tuning")
    else:
        logger.info("❌ STRATEGY NEEDS IMPROVEMENT")
        logger.info("Consider strategy redesign or different approach")

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
