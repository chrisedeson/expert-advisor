#!/usr/bin/env python3
"""
OVERNIGHT OPTIMIZER - Exhaustive Parameter Search

Runs extensive optimization across multiple timeframes and parameter ranges.
Designed to run overnight (8-12 hours) to find the absolute best configuration.

Strategy:
1. Test H1, H4, D timeframes
2. Run 1000 trials per timeframe
3. Test wide parameter ranges
4. Save best results periodically
5. Find the best overall configuration
"""

import sys
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from src.strategies import EnhancedTrendFollowingStrategy
from src.optimization import ParameterOptimizer, StrategyValidator

def test_timeframe(timeframe: str, n_trials: int = 1000) -> dict:
    """
    Run optimization on a specific timeframe.

    Args:
        timeframe: 'H1', 'H4', or 'D'
        n_trials: Number of optimization trials

    Returns:
        dict with best parameters and metrics
    """
    logger.info("=" * 80)
    logger.info(f"OPTIMIZING {timeframe} TIMEFRAME ({n_trials} trials)")
    logger.info("=" * 80)

    # Load data
    data_file = Path(__file__).parent.parent / "data" / "raw" / f"EURUSDm_{timeframe}.parquet"

    if not data_file.exists():
        logger.warning(f"Data file not found: {data_file}")
        return None

    logger.info(f"📁 Loading {data_file.name}...")
    data = pd.read_parquet(data_file)
    logger.info(f"Loaded {len(data):,} candles: {data.index[0].date()} to {data.index[-1].date()}")

    # Split data
    n = len(data)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]

    logger.info(f"Training: {len(train_data):,} candles")
    logger.info(f"Validation: {len(val_data):,} candles")
    logger.info(f"Test: {len(test_data):,} candles")

    # Initialize strategy with wide parameter ranges
    strategy = EnhancedTrendFollowingStrategy()

    # VERY WIDE parameter space for exhaustive search
    if timeframe == 'H1':
        # H1: Faster parameters for intraday
        param_space = {
            'fast_ma_period': (5, 50, 'int'),
            'slow_ma_period': (20, 200, 'int'),
            'adx_period': (8, 25, 'int'),
            'adx_threshold': (10.0, 35.0, 'float'),
            'atr_period': (8, 25, 'int'),
            'volatility_lookback': (50, 200, 'int'),
            'volatility_threshold': (2.0, 4.0, 'float'),
        }
    elif timeframe == 'H4':
        # H4: Medium parameters for swing
        param_space = {
            'fast_ma_period': (10, 80, 'int'),
            'slow_ma_period': (30, 250, 'int'),
            'adx_period': (10, 30, 'int'),
            'adx_threshold': (15.0, 35.0, 'float'),
            'atr_period': (10, 25, 'int'),
            'volatility_lookback': (80, 250, 'int'),
            'volatility_threshold': (2.0, 4.0, 'float'),
        }
    else:  # D
        # D: Slower parameters for position trading
        param_space = {
            'fast_ma_period': (20, 100, 'int'),
            'slow_ma_period': (50, 300, 'int'),
            'adx_period': (12, 35, 'int'),
            'adx_threshold': (18.0, 35.0, 'float'),
            'atr_period': (12, 30, 'int'),
            'volatility_lookback': (100, 300, 'int'),
            'volatility_threshold': (2.0, 4.0, 'float'),
        }

    strategy.param_space = param_space

    logger.info("\n🎯 Parameter ranges:")
    for param, (min_val, max_val, _) in param_space.items():
        logger.info(f"  {param}: {min_val} - {max_val}")

    # Run optimization
    logger.info(f"\n🔍 Running {n_trials} trials (this will take ~{n_trials * 0.3 / 60:.0f} minutes)...")

    optimizer = ParameterOptimizer(
        strategy=strategy,
        data=train_data,
        initial_capital=100.0,
        risk_per_trade=0.01,
        max_drawdown=0.20,  # Relaxed to 20% for exploration
        objective_metric='cagr',
    )

    results = optimizer.optimize(
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=True,
    )

    best_params = results['best_params']
    best_train_cagr = results['best_value']

    logger.info(f"\n🏆 Best Training CAGR: {best_train_cagr*100:.2f}%")
    logger.info(f"📊 Best Parameters:")
    for param, value in best_params.items():
        if isinstance(value, float):
            logger.info(f"  {param}: {value:.2f}")
        else:
            logger.info(f"  {param}: {value}")

    # Validate on all datasets
    strategy.set_parameters(**best_params)

    validator = StrategyValidator(
        strategy=strategy,
        initial_capital=100.0,
        risk_per_trade=0.01,
        max_drawdown=0.20,
    )

    logger.info("\n📈 Validating...")
    full_results = validator.validate_on_dataset(data, 'Full Period', symbol='EURUSD')
    val_results = validator.validate_on_dataset(val_data, 'Validation', symbol='EURUSD')

    full_m = full_results['metrics']
    val_m = val_results['metrics']

    logger.info(f"\n✅ {timeframe} Results:")
    logger.info(f"  Full CAGR:      {full_m['cagr']*100:>7.2f}%")
    logger.info(f"  Val CAGR:       {val_m['cagr']*100:>7.2f}%")
    logger.info(f"  Sharpe:         {full_m['sharpe_ratio']:>7.2f}")
    logger.info(f"  Max DD:         {full_m['max_drawdown']*100:>7.2f}%")
    logger.info(f"  Win Rate:       {full_m.get('win_rate', 0)*100:>7.2f}%")
    logger.info(f"  Profit Factor:  {full_m.get('profit_factor', 0):>7.2f}")
    logger.info(f"  Trades:         {full_m['num_trades']:>7.0f}")

    # Calculate criteria score
    criteria_met = 0
    if full_m['cagr'] > 0.12: criteria_met += 1
    if full_m['sharpe_ratio'] > 1.0: criteria_met += 1
    if full_m['max_drawdown'] > -0.20: criteria_met += 1
    if full_m.get('win_rate', 0) > 0.45: criteria_met += 1
    if full_m['num_trades'] > 200: criteria_met += 1

    logger.info(f"\n📊 Criteria Met: {criteria_met}/5")

    return {
        'timeframe': timeframe,
        'best_params': best_params,
        'train_cagr': best_train_cagr,
        'full_cagr': full_m['cagr'],
        'val_cagr': val_m['cagr'],
        'sharpe': full_m['sharpe_ratio'],
        'max_dd': full_m['max_drawdown'],
        'win_rate': full_m.get('win_rate', 0),
        'profit_factor': full_m.get('profit_factor', 0),
        'trades': full_m['num_trades'],
        'criteria_met': criteria_met,
        'full_metrics': full_m,
    }


def main():
    """Run overnight optimization across all timeframes."""

    start_time = datetime.now()

    logger.info("=" * 80)
    logger.info("OVERNIGHT EXHAUSTIVE OPTIMIZER")
    logger.info("=" * 80)
    logger.info(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Expected Duration: 6-10 hours")
    logger.info("Testing: H1, H4, D timeframes")
    logger.info("Trials per timeframe: 1000")
    logger.info("=" * 80)

    # Test all timeframes
    timeframes = ['H1', 'H4', 'D']
    results = []

    for tf in timeframes:
        logger.info(f"\n\n{'='*80}")
        logger.info(f"STARTING {tf} OPTIMIZATION")
        logger.info(f"{'='*80}\n")

        try:
            result = test_timeframe(tf, n_trials=1000)
            if result:
                results.append(result)

                # Save intermediate results
                output_dir = Path(__file__).parent.parent / "reports"
                progress_file = output_dir / f"overnight_progress_{tf}.json"

                with open(progress_file, 'w') as f:
                    # Convert numpy types to native Python types for JSON
                    result_copy = result.copy()
                    if 'full_metrics' in result_copy:
                        del result_copy['full_metrics']  # Too large for JSON
                    json.dump(result_copy, f, indent=2, default=float)

                logger.info(f"\n💾 Saved progress: {progress_file}")

        except Exception as e:
            logger.error(f"Error optimizing {tf}: {e}")
            continue

    # Find best overall
    logger.info("\n\n" + "=" * 80)
    logger.info("OPTIMIZATION COMPLETE - FINAL RESULTS")
    logger.info("=" * 80)

    if not results:
        logger.error("No successful optimizations!")
        return

    # Display comparison table
    logger.info(f"\n{'Timeframe':<12} {'CAGR':<10} {'Sharpe':<10} {'Max DD':<10} {'Win Rate':<10} {'Trades':<10} {'Score':<10}")
    logger.info("-" * 80)

    for r in results:
        logger.info(
            f"{r['timeframe']:<12} "
            f"{r['full_cagr']*100:>6.2f}%   "
            f"{r['sharpe']:>6.2f}    "
            f"{r['max_dd']*100:>6.2f}%   "
            f"{r['win_rate']*100:>6.2f}%   "
            f"{r['trades']:>6.0f}      "
            f"{r['criteria_met']}/5"
        )

    # Find best by CAGR
    best_cagr = max(results, key=lambda x: x['full_cagr'])
    # Find best by criteria met
    best_criteria = max(results, key=lambda x: x['criteria_met'])
    # Find best Sharpe
    best_sharpe = max(results, key=lambda x: x['sharpe'])

    logger.info(f"\n🏆 WINNERS:")
    logger.info(f"  Best CAGR:     {best_cagr['timeframe']} ({best_cagr['full_cagr']*100:.2f}%)")
    logger.info(f"  Best Sharpe:   {best_sharpe['timeframe']} ({best_sharpe['sharpe']:.2f})")
    logger.info(f"  Best Overall:  {best_criteria['timeframe']} ({best_criteria['criteria_met']}/5 criteria)")

    # Save final results
    output_dir = Path(__file__).parent.parent / "reports"
    final_file = output_dir / "overnight_optimization_final.json"

    final_results = {
        'start_time': start_time.isoformat(),
        'end_time': datetime.now().isoformat(),
        'duration_hours': (datetime.now() - start_time).total_seconds() / 3600,
        'timeframes_tested': timeframes,
        'trials_per_timeframe': 1000,
        'results': [
            {k: v for k, v in r.items() if k != 'full_metrics'}
            for r in results
        ],
        'best_cagr': {
            'timeframe': best_cagr['timeframe'],
            'cagr': float(best_cagr['full_cagr']),
            'params': best_cagr['best_params'],
        },
        'best_sharpe': {
            'timeframe': best_sharpe['timeframe'],
            'sharpe': float(best_sharpe['sharpe']),
            'params': best_sharpe['best_params'],
        },
        'best_overall': {
            'timeframe': best_criteria['timeframe'],
            'criteria_met': best_criteria['criteria_met'],
            'params': best_criteria['best_params'],
        },
    }

    with open(final_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=float)

    logger.info(f"\n💾 Saved final results: {final_file}")

    # Recommendation
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDATION")
    logger.info("=" * 80)

    if best_criteria['criteria_met'] >= 4:
        logger.info(f"\n🎉 EXCELLENT! Use {best_criteria['timeframe']} timeframe:")
        logger.info(f"   CAGR: {best_criteria['full_cagr']*100:.2f}%")
        logger.info(f"   Sharpe: {best_criteria['sharpe']:.2f}")
        logger.info(f"   Criteria: {best_criteria['criteria_met']}/5")
        logger.info(f"\n   ✅ READY FOR PAPER TRADING")
    elif best_criteria['criteria_met'] >= 3:
        logger.info(f"\n⚠️  SHOWS PROMISE. Use {best_criteria['timeframe']} timeframe:")
        logger.info(f"   CAGR: {best_criteria['full_cagr']*100:.2f}%")
        logger.info(f"   Sharpe: {best_criteria['sharpe']:.2f}")
        logger.info(f"   Criteria: {best_criteria['criteria_met']}/5")
        logger.info(f"\n   Consider Phase 6 robustness tests before paper trading")
    else:
        logger.info(f"\n❌ NOT READY. Best we found:")
        logger.info(f"   Timeframe: {best_criteria['timeframe']}")
        logger.info(f"   CAGR: {best_criteria['full_cagr']*100:.2f}%")
        logger.info(f"   Sharpe: {best_criteria['sharpe']:.2f}")
        logger.info(f"   Criteria: {best_criteria['criteria_met']}/5")
        logger.info(f"\n   Consider:")
        logger.info(f"   1. Try other pairs (GBPUSD, USDJPY, AUDUSD)")
        logger.info(f"   2. Different strategy approach")
        logger.info(f"   3. Accept lower returns and go live for learning")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 3600

    logger.info(f"\n⏱️  Total Duration: {duration:.1f} hours")
    logger.info(f"📅 Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()
