#!/usr/bin/env python3
"""Test optimized strategy on all 4 currency pairs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from src.strategies import EnhancedTrendFollowingStrategy
from src.optimization import StrategyValidator

def main():
    """Test best parameters on all pairs."""

    logger.info("=" * 70)
    logger.info("TESTING OPTIMIZED STRATEGY ON ALL 4 PAIRS")
    logger.info("=" * 70)

    # Best parameters from EURUSD optimization
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

    pairs = ['EURUSDm', 'GBPUSDm', 'USDJPYm', 'AUDUSDm']
    pair_names = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']

    data_dir = Path(__file__).parent.parent / "data" / "raw"

    results = []

    for pair, pair_name in zip(pairs, pair_names):
        logger.info("\n" + "=" * 70)
        logger.info(f"TESTING {pair_name}")
        logger.info("=" * 70)

        # Load H1 data
        data_file = data_dir / f"{pair}_H1.parquet"

        if not data_file.exists():
            logger.warning(f"Skipping {pair_name}: Data file not found")
            continue

        logger.info(f"📁 Loading {pair}_H1.parquet...")
        data = pd.read_parquet(data_file)
        logger.info(f"Loaded {len(data):,} candles: {data.index[0].date()} to {data.index[-1].date()}")

        # Create strategy
        strategy = EnhancedTrendFollowingStrategy(**best_params)

        # Create validator
        validator = StrategyValidator(
            strategy=strategy,
            initial_capital=100.0,
            risk_per_trade=0.01,
            max_drawdown=0.18,
        )

        # Backtest full period
        logger.info(f"🚀 Running backtest on {pair_name}...")
        result = validator.validate_on_dataset(data, pair_name, symbol=pair_name.replace('m', ''))

        metrics = result['metrics']

        logger.info(f"\n📈 {pair_name} Results:")
        logger.info(f"  Total Return:   {metrics['total_return']*100:>7.2f}%")
        logger.info(f"  CAGR:           {metrics['cagr']*100:>7.2f}%")
        logger.info(f"  Sharpe Ratio:   {metrics['sharpe_ratio']:>7.2f}")
        logger.info(f"  Max Drawdown:   {metrics['max_drawdown']*100:>7.2f}%")
        logger.info(f"  Win Rate:       {metrics.get('win_rate', 0)*100:>7.2f}%")
        logger.info(f"  Profit Factor:  {metrics.get('profit_factor', 0):>7.2f}")
        logger.info(f"  Total Trades:   {metrics['num_trades']:>7.0f}")

        results.append({
            'pair': pair_name,
            'total_return': metrics['total_return'],
            'cagr': metrics['cagr'],
            'sharpe': metrics['sharpe_ratio'],
            'max_dd': metrics['max_drawdown'],
            'win_rate': metrics.get('win_rate', 0),
            'profit_factor': metrics.get('profit_factor', 0),
            'trades': metrics['num_trades'],
        })

    # Summary comparison
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY: ALL PAIRS COMPARISON")
    logger.info("=" * 70)

    logger.info(f"\n{'Pair':<10} {'CAGR':<10} {'Sharpe':<10} {'Max DD':<10} {'Win Rate':<10} {'Trades':<10}")
    logger.info("-" * 70)

    for r in results:
        logger.info(
            f"{r['pair']:<10} "
            f"{r['cagr']*100:>6.2f}%   "
            f"{r['sharpe']:>6.2f}    "
            f"{r['max_dd']*100:>6.2f}%   "
            f"{r['win_rate']*100:>6.2f}%   "
            f"{r['trades']:>6.0f}"
        )

    # Find best pair
    logger.info("\n" + "=" * 70)
    logger.info("BEST PERFORMING PAIR")
    logger.info("=" * 70)

    best_cagr = max(results, key=lambda x: x['cagr'])
    best_sharpe = max(results, key=lambda x: x['sharpe'])

    logger.info(f"\n🏆 Best CAGR:   {best_cagr['pair']} ({best_cagr['cagr']*100:.2f}%)")
    logger.info(f"🏆 Best Sharpe: {best_sharpe['pair']} ({best_sharpe['sharpe']:.2f})")

    # Check if any pair meets criteria
    logger.info("\n📊 GO/NO-GO Assessment:")

    for r in results:
        criteria_met = 0
        if r['cagr'] > 0.12: criteria_met += 1
        if r['sharpe'] > 1.0: criteria_met += 1
        if r['max_dd'] > -0.18: criteria_met += 1
        if r['win_rate'] > 0.45: criteria_met += 1
        if r['trades'] > 200: criteria_met += 1

        status = "✅ READY" if criteria_met >= 4 else "⚠️ CLOSE" if criteria_met >= 3 else "❌ NEEDS WORK"
        logger.info(f"  {r['pair']:<10} {criteria_met}/5 criteria met - {status}")

    # Portfolio approach
    logger.info("\n" + "=" * 70)
    logger.info("PORTFOLIO APPROACH (ALL 4 PAIRS)")
    logger.info("=" * 70)

    avg_cagr = sum(r['cagr'] for r in results) / len(results)
    avg_sharpe = sum(r['sharpe'] for r in results) / len(results)
    avg_dd = sum(r['max_dd'] for r in results) / len(results)

    logger.info(f"\nAverage Performance:")
    logger.info(f"  Avg CAGR:       {avg_cagr*100:>7.2f}%")
    logger.info(f"  Avg Sharpe:     {avg_sharpe:>7.2f}")
    logger.info(f"  Avg Max DD:     {avg_dd*100:>7.2f}%")

    logger.info(f"\nPortfolio potential:")
    logger.info(f"  4 pairs × ${100/4:.2f}/pair = $100 total")
    logger.info(f"  If uncorrelated: Sharpe could improve by √4 = 2x")
    logger.info(f"  Estimated portfolio Sharpe: {avg_sharpe * 1.5:.2f} (conservative)")

    logger.info("\n" + "=" * 70)


if __name__ == "__main__":
    main()
