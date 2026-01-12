#!/usr/bin/env python3
"""
Test Bollinger Grid Strategy (Medium-Risk)

Tests the grid strategy with dynamic position sizing starting from $100.
Shows the compounding effect and scaling as capital grows.
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from src.strategies import BollingerGridStrategy
from src.backtesting.grid_engine import GridBacktestingEngine
from src.analytics import PerformanceMetrics


def main():
    """Test grid strategy on EUR/USD H1 data."""

    logger.info("=" * 70)
    logger.info("BOLLINGER GRID STRATEGY TEST (Medium-Risk)")
    logger.info("=" * 70)

    # Load H1 data
    data_file = Path(__file__).parent.parent / "data" / "raw" / "EURUSDm_H1.parquet"

    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        logger.info("Please run the data export script first")
        return

    logger.info(f"\n📁 Loading {data_file.name}...")
    data = pd.read_parquet(data_file)
    logger.info(
        f"Loaded {len(data):,} H1 candles: "
        f"{data.index[0].date()} to {data.index[-1].date()}"
    )

    # Split data
    n = len(data)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]

    logger.info(f"\n📊 Data splits:")
    logger.info(
        f"  Training:   {len(train_data):,} candles "
        f"({train_data.index[0].date()} to {train_data.index[-1].date()})"
    )
    logger.info(
        f"  Validation: {len(val_data):,} candles "
        f"({val_data.index[0].date()} to {val_data.index[-1].date()})"
    )
    logger.info(
        f"  Test:       {len(test_data):,} candles "
        f"({test_data.index[0].date()} to {test_data.index[-1].date()})"
    )

    # Initialize strategy (Medium-Risk settings)
    strategy = BollingerGridStrategy(
        bb_period=20,
        bb_deviation=2.0,
        max_grid_levels=5,              # Up to 5 positions
        grid_spacing_atr_mult=1.2,      # Grid spacing
        lot_scaling_factor=1.2,         # 20% increase per level
        risk_per_trade=0.01,            # 1% risk
        max_drawdown=0.20,              # 20% max drawdown
        enable_session_filter=True,     # Trade during active hours
        take_profit_atr_mult=1.5,       # TP at 1.5x ATR
    )

    logger.info(f"\n🎯 Strategy: {strategy}")
    logger.info(f"  BB Period: {strategy.bb_period}")
    logger.info(f"  BB Deviation: {strategy.bb_deviation}")
    logger.info(f"  Max Grid Levels: {strategy.max_grid_levels}")
    logger.info(f"  Lot Scaling: {strategy.lot_scaling_factor}x per level")
    logger.info(f"  Risk per Trade: {strategy.risk_per_trade*100:.1f}%")
    logger.info(f"  Max Drawdown: {strategy.max_drawdown*100:.0f}%")

    # Run backtest on training data
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING SET BACKTEST")
    logger.info("=" * 70)

    engine = GridBacktestingEngine(
        strategy=strategy,
        initial_capital=100.0,          # Start with $100
        risk_per_trade=0.01,
        max_drawdown=0.20,
    )

    train_results = engine.run(train_data, symbol='EURUSD')

    # Calculate metrics
    metrics = PerformanceMetrics(
        train_results['equity_curve'],
        train_results['trades']
    )
    train_metrics = metrics.calculate_all_metrics()

    logger.info(f"\n📈 Training Results:")
    logger.info(f"  Initial Capital:  ${100.0:.2f}")
    logger.info(f"  Final Equity:     ${train_results['final_equity']:.2f}")
    logger.info(f"  Total Return:     {train_metrics['total_return']*100:.2f}%")
    logger.info(f"  CAGR:             {train_metrics['cagr']*100:.2f}%")
    logger.info(f"  Sharpe Ratio:     {train_metrics['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown:     {train_metrics['max_drawdown']*100:.2f}%")
    logger.info(f"  Total Trades:     {train_metrics['num_trades']}")
    logger.info(f"  Win Rate:         {train_metrics.get('win_rate', 0)*100:.2f}%")
    logger.info(f"  Profit Factor:    {train_metrics.get('profit_factor', 0):.2f}")

    # Check if meets minimum criteria
    logger.info(f"\n✅ Success Criteria Check:")
    criteria_met = 0
    total_criteria = 5

    # CAGR > 40%
    cagr_pass = train_metrics['cagr'] > 0.40
    criteria_met += int(cagr_pass)
    logger.info(
        f"  CAGR > 40%:         {train_metrics['cagr']*100:.2f}% "
        f"{'✅ PASS' if cagr_pass else '❌ MISS'}"
    )

    # Sharpe > 0.8
    sharpe_pass = train_metrics['sharpe_ratio'] > 0.8
    criteria_met += int(sharpe_pass)
    logger.info(
        f"  Sharpe > 0.8:       {train_metrics['sharpe_ratio']:.2f} "
        f"{'✅ PASS' if sharpe_pass else '❌ MISS'}"
    )

    # Max DD < 20%
    dd_pass = train_metrics['max_drawdown'] > -0.20
    criteria_met += int(dd_pass)
    logger.info(
        f"  Max DD < 20%:       {train_metrics['max_drawdown']*100:.2f}% "
        f"{'✅ PASS' if dd_pass else '❌ MISS'}"
    )

    # Win Rate > 60%
    wr_pass = train_metrics.get('win_rate', 0) > 0.60
    criteria_met += int(wr_pass)
    logger.info(
        f"  Win Rate > 60%:     {train_metrics.get('win_rate', 0)*100:.2f}% "
        f"{'✅ PASS' if wr_pass else '❌ MISS'}"
    )

    # Min Trades > 50
    trades_pass = train_metrics['num_trades'] > 50
    criteria_met += int(trades_pass)
    logger.info(
        f"  Trades > 50:        {train_metrics['num_trades']} "
        f"{'✅ PASS' if trades_pass else '❌ MISS'}"
    )

    logger.info(f"\n📊 Total: {criteria_met}/{total_criteria} criteria met")

    # Run on full dataset
    logger.info("\n" + "=" * 70)
    logger.info("FULL DATASET BACKTEST (2019-2025)")
    logger.info("=" * 70)

    engine_full = GridBacktestingEngine(
        strategy=strategy,
        initial_capital=100.0,
        risk_per_trade=0.01,
        max_drawdown=0.20,
    )

    full_results = engine_full.run(data, symbol='EURUSD')
    full_metrics_calc = PerformanceMetrics(
        full_results['equity_curve'],
        full_results['trades']
    )
    full_metrics = full_metrics_calc.calculate_all_metrics()

    logger.info(f"\n📈 Full Period Performance:")
    logger.info(f"  Initial Capital:  ${100.0:.2f}")
    logger.info(f"  Final Equity:     ${full_results['final_equity']:.2f}")
    logger.info(f"  Total Return:     {full_metrics['total_return']*100:.2f}%")
    logger.info(f"  CAGR:             {full_metrics['cagr']*100:.2f}%")
    logger.info(f"  Sharpe Ratio:     {full_metrics['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown:     {full_metrics['max_drawdown']*100:.2f}%")
    logger.info(f"  Win Rate:         {full_metrics.get('win_rate', 0)*100:.2f}%")
    logger.info(f"  Profit Factor:    {full_metrics.get('profit_factor', 0):.2f}")
    logger.info(f"  Total Trades:     {full_metrics['num_trades']}")

    # Calculate profit projections
    logger.info(f"\n💰 Profit Projections (Based on {full_metrics['cagr']*100:.2f}% CAGR):")

    starting_capital = 100.0
    annual_return = full_metrics['cagr']

    logger.info(f"\n  Starting with ${starting_capital:.0f}:")
    for year in [1, 2, 3, 5, 10]:
        future_balance = starting_capital * ((1 + annual_return) ** year)
        annual_profit = future_balance - starting_capital if year == 1 else future_balance * annual_return
        weekly_profit = annual_profit / 52
        daily_profit = annual_profit / 365

        logger.info(
            f"    Year {year:2d}: ${future_balance:>10,.2f} "
            f"(${daily_profit:>6.2f}/day, ${weekly_profit:>7.2f}/week)"
        )

    # Calculate capital needed for $100/day target
    logger.info(f"\n🎯 Capital Required for $100/day:")
    target_daily = 100.0
    target_annual = target_daily * 365
    required_capital = target_annual / annual_return if annual_return > 0 else float('inf')

    logger.info(f"  Target: $100/day = ${target_annual:,.0f}/year")
    logger.info(f"  At {annual_return*100:.1f}% CAGR: Need ${required_capital:,.0f} capital")

    # Calculate compounding timeline from $100 to target
    if annual_return > 0.10:  # Only if positive return
        years_to_target = np.log(required_capital / starting_capital) / np.log(1 + annual_return)
        logger.info(
            f"  From ${starting_capital:.0f} → ${required_capital:,.0f}: "
            f"~{years_to_target:.1f} years (compounding)"
        )

    # Display sample trades
    logger.info(f"\n📋 Sample Trades (last 10):")
    if len(full_results['trades']) > 0:
        trades_df = full_results['trades']
        sample_trades = trades_df.tail(10)

        for _, trade in sample_trades.iterrows():
            pnl_sign = "+" if trade['net_pnl'] > 0 else ""
            logger.info(
                f"  {trade['direction']:5s} L{trade['grid_level']} | "
                f"Entry: {trade['entry_price']:.5f} | "
                f"Exit: {trade['exit_price']:.5f} | "
                f"Size: {trade['size']:.2f} | "
                f"P&L: {pnl_sign}${trade['net_pnl']:.2f} | "
                f"{trade['exit_reason']}"
            )
    else:
        logger.warning("  No trades executed!")

    # Final verdict
    logger.info("\n" + "=" * 70)
    logger.info("VERDICT")
    logger.info("=" * 70)

    if full_metrics['cagr'] > 0.35 and full_metrics['max_drawdown'] > -0.20:
        logger.info("\n✅ STRATEGY LOOKS PROMISING!")
        logger.info("   Grid strategy shows viable performance.")
        logger.info("   Next steps: Optimize parameters and paper trade.")
    elif full_metrics['cagr'] > 0.15:
        logger.info("\n⚠️  STRATEGY SHOWS POTENTIAL")
        logger.info("   Performance is moderate. Consider optimization.")
    else:
        logger.info("\n❌ STRATEGY NEEDS IMPROVEMENT")
        logger.info("   Returns below expectations for grid strategy.")
        logger.info("   Consider different parameters or timeframes.")

    logger.info("\n" + "=" * 70)


if __name__ == "__main__":
    main()
