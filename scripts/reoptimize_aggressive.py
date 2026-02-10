#!/usr/bin/env python3
"""
Re-optimize Strategy with Aggressive Parameters

Changes from Phase 4:
1. Use H1 timeframe (4x more trades)
2. Broader parameter ranges (faster signals)
3. More trials (500 instead of 100)
4. Optimize for CAGR (not just Sharpe)
5. Relax drawdown to 18% temporarily
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from src.strategies import EnhancedTrendFollowingStrategy
from src.optimization import ParameterOptimizer, StrategyValidator

def main():
    """Re-optimize with aggressive parameters."""

    logger.info("=" * 70)
    logger.info("AGGRESSIVE RE-OPTIMIZATION")
    logger.info("=" * 70)

    # Load H1 data (4x more granular than H4)
    data_file = Path(__file__).parent.parent / "data" / "raw" / "EURUSDm_H1.parquet"

    if not data_file.exists():
        logger.error(f"H1 data file not found: {data_file}")
        logger.info("Need to export H1 data from MT5 first")
        return

    logger.info(f"\n📁 Loading H1 data from {data_file.name}")
    data = pd.read_parquet(data_file)
    logger.info(f"Loaded {len(data)} H1 candles: {data.index[0].date()} to {data.index[-1].date()}")

    # Initialize strategy
    logger.info("\n🔧 Initializing Enhanced Strategy...")
    strategy = EnhancedTrendFollowingStrategy(
        fast_ma_period=20,  # Placeholder, will be optimized
        slow_ma_period=100,
        adx_period=14,
        adx_threshold=20.0,
        atr_period=14,
        use_session_filter=False,
        ma_type='ema',
        volatility_lookback=100,
        volatility_threshold=2.5,
        enable_regime_filter=True,
    )

    # Split data manually (60/20/20 split)
    n = len(data)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]

    logger.info(f"\n📊 Data splits:")
    logger.info(f"  Training:   {len(train_data)} candles ({train_data.index[0].date()} to {train_data.index[-1].date()})")
    logger.info(f"  Validation: {len(val_data)} candles ({val_data.index[0].date()} to {val_data.index[-1].date()})")
    logger.info(f"  Test:       {len(test_data)} candles ({test_data.index[0].date()} to {test_data.index[-1].date()})")

    # Define AGGRESSIVE parameter space on strategy
    logger.info("\n🎯 Defining aggressive parameter space...")
    param_space = {
        # Faster MAs for more trades
        'fast_ma_period': (5, 30, 'int'),      # Was 10-50, now 5-30 (faster)
        'slow_ma_period': (30, 150, 'int'),    # Was 30-200, now 30-150 (faster)

        # ADX settings
        'adx_period': (10, 20, 'int'),         # Same
        'adx_threshold': (10.0, 30.0, 'float'),  # Lower minimum (15→10) for more trades

        # ATR settings
        'atr_period': (10, 20, 'int'),         # Same
    }

    # Set parameter space on strategy
    strategy.param_space = param_space

    logger.info("Parameter ranges:")
    for param, (min_val, max_val, _) in param_space.items():
        logger.info(f"  {param}: {min_val} - {max_val}")

    # Run optimization with CAGR objective
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING AGGRESSIVE OPTIMIZATION (500 TRIALS)")
    logger.info("=" * 70)
    logger.info("Objective: Maximize CAGR (not just Sharpe)")
    logger.info("Constraint: Max Drawdown < 18% (relaxed from 15%)")
    logger.info("Expected: ~30 mins on modern CPU")

    optimizer = ParameterOptimizer(
        strategy=strategy,
        data=train_data,
        initial_capital=100.0,
        risk_per_trade=0.01,
        max_drawdown=0.18,  # Relaxed from 0.15 to 0.18
        objective_metric='cagr',  # Changed from 'sharpe_ratio' to 'cagr'
    )

    results = optimizer.optimize(
        n_trials=500,  # Increased from 100 to 500
        n_jobs=1,
        show_progress_bar=True,
    )

    best_params = results['best_params']
    best_score = results['best_value']

    logger.info("\n" + "=" * 70)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 70)

    logger.info(f"\n🏆 Best CAGR: {best_score*100:.2f}%")
    logger.info(f"\n📊 Best Parameters:")
    for param, value in best_params.items():
        logger.info(f"  {param}: {value}")

    # Validate on validation set
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION (OUT-OF-SAMPLE)")
    logger.info("=" * 70)

    # Update strategy with best parameters
    strategy.set_parameters(**best_params)

    # Create validator instance
    validator = StrategyValidator(
        strategy=strategy,
        initial_capital=100.0,
        risk_per_trade=0.01,
        max_drawdown=0.18,
    )

    # Run validation on train and val data
    train_results = validator.validate_on_dataset(train_data, 'Training', symbol='EURUSD')
    val_results_raw = validator.validate_on_dataset(val_data, 'Validation', symbol='EURUSD')

    # Package results
    val_results = {
        'train_metrics': train_results['metrics'],
        'val_metrics': val_results_raw['metrics'],
        'degradation': (val_results_raw['metrics']['sharpe_ratio'] - train_results['metrics']['sharpe_ratio']) / abs(train_results['metrics']['sharpe_ratio'])
    }

    # Display results
    logger.info("\n📈 Performance Comparison:")
    logger.info(f"{'Metric':<20} {'Training':<12} {'Validation':<12} {'Change':<10}")
    logger.info("-" * 70)

    train_m = val_results['train_metrics']
    val_m = val_results['val_metrics']

    for metric in ['total_return', 'cagr', 'sharpe_ratio', 'max_drawdown', 'num_trades']:
        train_val = train_m.get(metric, 0)
        val_val = val_m.get(metric, 0)

        if metric in ['total_return', 'cagr', 'max_drawdown']:
            train_str = f"{train_val*100:>6.2f}%"
            val_str = f"{val_val*100:>6.2f}%"
        elif metric == 'sharpe_ratio':
            train_str = f"{train_val:>6.2f}"
            val_str = f"{val_val:>6.2f}"
        else:
            train_str = f"{train_val:>6.0f}"
            val_str = f"{val_val:>6.0f}"

        # Calculate change
        if train_val != 0:
            change_pct = ((val_val - train_val) / abs(train_val)) * 100
            change_str = f"{change_pct:+.1f}%"
        else:
            change_str = "N/A"

        logger.info(f"{metric:<20} {train_str:<12} {val_str:<12} {change_str:<10}")

    # Check for overfitting
    logger.info("\n🔍 Overfitting Check:")
    if val_results['degradation'] < 0.30:  # Less than 30% degradation
        logger.info(f"✅ Validation degradation: {val_results['degradation']*100:.1f}% (acceptable)")
    else:
        logger.warning(f"⚠️  Validation degradation: {val_results['degradation']*100:.1f}% (possible overfitting)")

    # Save results
    output_dir = Path(__file__).parent.parent / "reports"

    # Save best parameters
    params_file = output_dir / "best_parameters_aggressive.txt"
    with open(params_file, 'w') as f:
        f.write("Best Parameters from Aggressive Re-optimization\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Objective: Maximize CAGR\n")
        f.write(f"Best CAGR: {best_score*100:.2f}%\n")
        f.write(f"Trials Run: 500\n\n")
        f.write("Parameters:\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
        f.write(f"\nValidation Results:\n")
        f.write(f"  Validation CAGR: {val_m['cagr']*100:.2f}%\n")
        f.write(f"  Validation Sharpe: {val_m['sharpe_ratio']:.2f}\n")
        f.write(f"  Degradation: {val_results['degradation']*100:+.1f}%\n")

    logger.info(f"\n💾 Saved best parameters: {params_file}")

    # Save optimization history
    history_file = output_dir / "optimization_history_aggressive.csv"
    optimizer.results.to_csv(history_file, index=False)
    logger.info(f"💾 Saved optimization history: {history_file}")

    logger.info("\n" + "=" * 70)
    logger.info("AGGRESSIVE RE-OPTIMIZATION COMPLETE")
    logger.info("=" * 70)

    # Final assessment
    logger.info("\n🎯 Performance Assessment:")
    logger.info(f"  CAGR:           {val_m['cagr']*100:>6.2f}% (target: > 12%)")
    logger.info(f"  Sharpe Ratio:   {val_m['sharpe_ratio']:>6.2f} (target: > 1.0)")
    logger.info(f"  Max Drawdown:   {val_m['max_drawdown']*100:>6.2f}% (target: < 18%)")
    logger.info(f"  Win Rate:       {val_m.get('win_rate', 0)*100:>6.2f}% (target: > 45%)")
    logger.info(f"  Trades:         {val_m['num_trades']:>6.0f} (target: > 200)")

    # Go/No-Go decision
    criteria_met = 0
    total_criteria = 5

    if val_m['cagr'] > 0.12:
        criteria_met += 1
        logger.info("\n✅ CAGR criterion MET")
    else:
        logger.info(f"\n❌ CAGR criterion MISSED (need {(0.12 - val_m['cagr'])*100:.2f}% more)")

    if val_m['sharpe_ratio'] > 1.0:
        criteria_met += 1
        logger.info("✅ Sharpe criterion MET")
    else:
        logger.info(f"❌ Sharpe criterion MISSED (need {1.0 - val_m['sharpe_ratio']:.2f} more)")

    if val_m['max_drawdown'] < -0.18:
        logger.info(f"❌ Max Drawdown criterion MISSED ({val_m['max_drawdown']*100:.2f}% > 18%)")
    else:
        criteria_met += 1
        logger.info("✅ Max Drawdown criterion MET")

    if val_m.get('win_rate', 0) > 0.45:
        criteria_met += 1
        logger.info("✅ Win Rate criterion MET")
    else:
        logger.info(f"❌ Win Rate criterion MISSED")

    if val_m['num_trades'] > 200:
        criteria_met += 1
        logger.info("✅ Minimum Trades criterion MET")
    else:
        logger.info(f"❌ Minimum Trades criterion MISSED (need {200 - val_m['num_trades']:.0f} more)")

    logger.info(f"\n📊 VERDICT: {criteria_met}/{total_criteria} criteria met")

    if criteria_met >= 4:
        logger.info("🎉 ✅ STRATEGY READY FOR PAPER TRADING!")
    elif criteria_met >= 3:
        logger.info("⚠️  STRATEGY SHOWS PROMISE - Consider Phase 6 robustness tests")
    else:
        logger.info("❌ STRATEGY NEEDS MORE WORK - Try H1 timeframe or different approach")

    logger.info("\n" + "=" * 70)


if __name__ == "__main__":
    main()
