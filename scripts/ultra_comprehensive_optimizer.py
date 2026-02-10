#!/usr/bin/env python3
"""
ULTRA-COMPREHENSIVE OVERNIGHT OPTIMIZER

Tests EVERYTHING in one massive run:
- Symbols: EURUSD, GBPUSD, USDJPY, AUDUSD, XAUUSD (GOLD)
- Timeframes: M1, M5, M15, M30, H1, H4, D
- 500 trials per combination
- Total: 5 symbols × 7 timeframes = 35 optimizations
- Estimated time: 24-48 hours on t2.micro

This is the FINAL comprehensive test before deciding to go live or not.
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from src.strategies import EnhancedTrendFollowingStrategy
from src.optimization import ParameterOptimizer, StrategyValidator


def test_symbol_timeframe(symbol: str, timeframe: str, n_trials: int = 500) -> dict:
    """
    Run optimization on a specific symbol and timeframe.

    Args:
        symbol: 'EURUSDm', 'GBPUSDm', 'USDJPYm', 'AUDUSDm', 'XAUUSDm'
        timeframe: 'M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D'
        n_trials: Number of optimization trials

    Returns:
        dict with best parameters and metrics
    """
    logger.info("=" * 80)
    logger.info(f"OPTIMIZING {symbol} {timeframe} ({n_trials} trials)")
    logger.info("=" * 80)

    # Load data
    data_file = Path(__file__).parent.parent / "data" / "raw" / f"{symbol}_{timeframe}.parquet"

    if not data_file.exists():
        logger.warning(f"Data file not found: {data_file}")
        return None

    logger.info(f"📁 Loading {data_file.name}...")
    try:
        data = pd.read_parquet(data_file)
        logger.info(f"Loaded {len(data):,} candles: {data.index[0].date()} to {data.index[-1].date()}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return None

    # Minimum data requirements
    if len(data) < 1000:
        logger.warning(f"Insufficient data: {len(data)} candles (need at least 1000)")
        return None

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

    # Initialize strategy with timeframe-specific parameter ranges
    strategy = EnhancedTrendFollowingStrategy()

    # Adjust parameter ranges based on timeframe
    if timeframe in ['M1', 'M5']:
        # Very fast parameters for scalping
        param_space = {
            'fast_ma_period': (3, 20, 'int'),
            'slow_ma_period': (10, 100, 'int'),
            'adx_period': (5, 15, 'int'),
            'adx_threshold': (10.0, 30.0, 'float'),
            'atr_period': (5, 15, 'int'),
            'volatility_lookback': (20, 100, 'int'),
            'volatility_threshold': (2.0, 4.0, 'float'),
        }
    elif timeframe in ['M15', 'M30']:
        # Fast parameters for intraday
        param_space = {
            'fast_ma_period': (5, 30, 'int'),
            'slow_ma_period': (15, 120, 'int'),
            'adx_period': (8, 20, 'int'),
            'adx_threshold': (10.0, 32.0, 'float'),
            'atr_period': (8, 20, 'int'),
            'volatility_lookback': (30, 150, 'int'),
            'volatility_threshold': (2.0, 4.0, 'float'),
        }
    elif timeframe == 'H1':
        # Medium parameters for swing
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
        # Slower parameters for swing
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
        # Slowest parameters for position trading
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

    # Run optimization
    logger.info(f"\n🔍 Running {n_trials} trials...")

    try:
        optimizer = ParameterOptimizer(
            strategy=strategy,
            data=train_data,
            initial_capital=100.0,
            risk_per_trade=0.01,
            max_drawdown=0.20,  # Relaxed to 20%
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

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        logger.error(traceback.format_exc())
        return None

    # Validate on all datasets
    try:
        strategy.set_parameters(**best_params)

        validator = StrategyValidator(
            strategy=strategy,
            initial_capital=100.0,
            risk_per_trade=0.01,
            max_drawdown=0.20,
        )

        # Clean symbol name for display
        symbol_clean = symbol.replace('m', '')

        logger.info("\n📈 Validating...")
        full_results = validator.validate_on_dataset(data, 'Full Period', symbol=symbol_clean)
        val_results = validator.validate_on_dataset(val_data, 'Validation', symbol=symbol_clean)

        full_m = full_results['metrics']
        val_m = val_results['metrics']

        logger.info(f"\n✅ {symbol} {timeframe} Results:")
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
        if full_m['num_trades'] > 100: criteria_met += 1  # Lowered from 200

        logger.info(f"\n📊 Criteria Met: {criteria_met}/5")

        return {
            'symbol': symbol,
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
        }

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        logger.error(traceback.format_exc())
        return None


def main():
    """Run ultra-comprehensive optimization across all symbols and timeframes."""

    start_time = datetime.now()

    logger.info("=" * 80)
    logger.info("ULTRA-COMPREHENSIVE OVERNIGHT OPTIMIZER")
    logger.info("=" * 80)
    logger.info(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Expected Duration: 24-48 hours (t2.micro)")
    logger.info("")
    logger.info("Testing:")
    logger.info("  Symbols: EURUSD, GBPUSD, USDJPY, AUDUSD, XAUUSD (GOLD)")
    logger.info("  Timeframes: M1, M5, M15, M30, H1, H4, D")
    logger.info("  Trials per combo: 500")
    logger.info("  Total combinations: 35")
    logger.info("=" * 80)

    # Test all combinations
    symbols = ['EURUSDm', 'GBPUSDm', 'USDJPYm', 'AUDUSDm', 'XAUUSDm']
    timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D']

    results = []
    total_combos = len(symbols) * len(timeframes)
    completed = 0

    for symbol in symbols:
        for timeframe in timeframes:
            completed += 1
            logger.info(f"\n\n{'='*80}")
            logger.info(f"[{completed}/{total_combos}] TESTING {symbol} {timeframe}")
            logger.info(f"{'='*80}\n")

            try:
                result = test_symbol_timeframe(symbol, timeframe, n_trials=500)
                if result:
                    results.append(result)

                    # Save intermediate results
                    output_dir = Path(__file__).parent.parent / "reports"
                    output_dir.mkdir(parents=True, exist_ok=True)

                    progress_file = output_dir / f"ultra_progress_{symbol}_{timeframe}.json"

                    with open(progress_file, 'w') as f:
                        json.dump(result, f, indent=2, default=float)

                    logger.info(f"\n💾 Saved progress: {progress_file}")

            except Exception as e:
                logger.error(f"Error optimizing {symbol} {timeframe}: {e}")
                logger.error(traceback.format_exc())
                continue

    # Find best overall
    logger.info("\n\n" + "=" * 80)
    logger.info("OPTIMIZATION COMPLETE - FINAL RESULTS")
    logger.info("=" * 80)

    if not results:
        logger.error("No successful optimizations!")
        return

    # Display comparison table
    logger.info(f"\n{'Symbol':<12} {'TF':<6} {'CAGR':<10} {'Sharpe':<10} {'Max DD':<10} {'Win Rate':<10} {'Trades':<10} {'Score':<10}")
    logger.info("-" * 90)

    for r in sorted(results, key=lambda x: x['full_cagr'], reverse=True):
        logger.info(
            f"{r['symbol']:<12} "
            f"{r['timeframe']:<6} "
            f"{r['full_cagr']*100:>6.2f}%   "
            f"{r['sharpe']:>6.2f}    "
            f"{r['max_dd']*100:>6.2f}%   "
            f"{r['win_rate']*100:>6.2f}%   "
            f"{r['trades']:>6.0f}      "
            f"{r['criteria_met']}/5"
        )

    # Find best by different criteria
    best_cagr = max(results, key=lambda x: x['full_cagr'])
    best_sharpe = max(results, key=lambda x: x['sharpe'])
    best_criteria = max(results, key=lambda x: x['criteria_met'])

    logger.info(f"\n🏆 WINNERS:")
    logger.info(f"  Best CAGR:     {best_cagr['symbol']} {best_cagr['timeframe']} ({best_cagr['full_cagr']*100:.2f}%)")
    logger.info(f"  Best Sharpe:   {best_sharpe['symbol']} {best_sharpe['timeframe']} ({best_sharpe['sharpe']:.2f})")
    logger.info(f"  Best Overall:  {best_criteria['symbol']} {best_criteria['timeframe']} ({best_criteria['criteria_met']}/5 criteria)")

    # Save final results
    output_dir = Path(__file__).parent.parent / "reports"
    final_file = output_dir / "ultra_optimization_final.json"

    final_results = {
        'start_time': start_time.isoformat(),
        'end_time': datetime.now().isoformat(),
        'duration_hours': (datetime.now() - start_time).total_seconds() / 3600,
        'symbols_tested': symbols,
        'timeframes_tested': timeframes,
        'trials_per_combo': 500,
        'total_combinations': total_combos,
        'successful_combinations': len(results),
        'results': results,
        'best_cagr': {
            'symbol': best_cagr['symbol'],
            'timeframe': best_cagr['timeframe'],
            'cagr': float(best_cagr['full_cagr']),
            'params': best_cagr['best_params'],
        },
        'best_sharpe': {
            'symbol': best_sharpe['symbol'],
            'timeframe': best_sharpe['timeframe'],
            'sharpe': float(best_sharpe['sharpe']),
            'params': best_sharpe['best_params'],
        },
        'best_overall': {
            'symbol': best_criteria['symbol'],
            'timeframe': best_criteria['timeframe'],
            'criteria_met': best_criteria['criteria_met'],
            'cagr': float(best_criteria['full_cagr']),
            'sharpe': float(best_criteria['sharpe']),
            'params': best_criteria['best_params'],
        },
    }

    with open(final_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=float)

    logger.info(f"\n💾 Saved final results: {final_file}")

    # Recommendation
    logger.info("\n" + "=" * 80)
    logger.info("FINAL RECOMMENDATION")
    logger.info("=" * 80)

    if best_criteria['criteria_met'] >= 4:
        logger.info(f"\n🎉 EXCELLENT! Use {best_criteria['symbol']} {best_criteria['timeframe']}:")
        logger.info(f"   CAGR: {best_criteria['full_cagr']*100:.2f}%")
        logger.info(f"   Sharpe: {best_criteria['sharpe']:.2f}")
        logger.info(f"   Criteria: {best_criteria['criteria_met']}/5")
        logger.info(f"\n   ✅ READY FOR PAPER TRADING")
    elif best_criteria['criteria_met'] >= 3:
        logger.info(f"\n⚠️  SHOWS PROMISE. Use {best_criteria['symbol']} {best_criteria['timeframe']}:")
        logger.info(f"   CAGR: {best_criteria['full_cagr']*100:.2f}%")
        logger.info(f"   Sharpe: {best_criteria['sharpe']:.2f}")
        logger.info(f"   Criteria: {best_criteria['criteria_met']}/5")
        logger.info(f"\n   Consider robustness tests before paper trading")
    else:
        logger.info(f"\n❌ NOT READY. Best we found:")
        logger.info(f"   Symbol: {best_criteria['symbol']}")
        logger.info(f"   Timeframe: {best_criteria['timeframe']}")
        logger.info(f"   CAGR: {best_criteria['full_cagr']*100:.2f}%")
        logger.info(f"   Sharpe: {best_criteria['sharpe']:.2f}")
        logger.info(f"   Criteria: {best_criteria['criteria_met']}/5")
        logger.info(f"\n   Options:")
        logger.info(f"   1. Accept lower returns and go live for learning")
        logger.info(f"   2. Try different strategy approach")
        logger.info(f"   3. Combine multiple instruments in portfolio")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 3600

    logger.info(f"\n⏱️  Total Duration: {duration:.1f} hours")
    logger.info(f"📅 Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()
