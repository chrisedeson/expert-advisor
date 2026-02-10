#!/usr/bin/env python3
"""
Optimize Strategy Parameters

Run Bayesian optimization to find optimal parameters for the trend-following strategy.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from src.strategies import TrendFollowingStrategy
from src.optimization import ParameterOptimizer, StrategyValidator


def split_data(
    data: pd.DataFrame, train_pct: float = 0.6, val_pct: float = 0.2
) -> tuple:
    """
    Split data into train/validation/test sets.

    Args:
        data: Full dataset
        train_pct: Percentage for training (default: 60%)
        val_pct: Percentage for validation (default: 20%)

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    n = len(data)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]

    return train_data, val_data, test_data


def main():
    """Run parameter optimization."""

    logger.info("=" * 70)
    logger.info("BAYESIAN PARAMETER OPTIMIZATION")
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

    # Split data
    logger.info("\n📊 Splitting data...")
    train_data, val_data, test_data = split_data(data, train_pct=0.6, val_pct=0.2)

    logger.info(f"Training:   {len(train_data)} candles ({train_data.index[0].date()} to {train_data.index[-1].date()})")
    logger.info(f"Validation: {len(val_data)} candles ({val_data.index[0].date()} to {val_data.index[-1].date()})")
    logger.info(f"Test:       {len(test_data)} candles ({test_data.index[0].date()} to {test_data.index[-1].date()})")

    # Initialize strategy with default parameters
    logger.info("\n🔧 Initializing strategy...")
    strategy = TrendFollowingStrategy(
        fast_ma_period=20,
        slow_ma_period=50,
        adx_period=14,
        adx_threshold=25.0,
        atr_period=14,
        use_session_filter=False,
        ma_type='ema',
    )

    logger.info(f"Strategy: {strategy}")
    logger.info(f"Parameter space: {strategy.get_parameter_space()}")

    # Run optimization
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: PARAMETER OPTIMIZATION (Training Data)")
    logger.info("=" * 70)

    optimizer = ParameterOptimizer(
        strategy=strategy,
        data=train_data,
        initial_capital=100.0,
        risk_per_trade=0.01,
        max_drawdown=0.15,
        objective_metric='sharpe_ratio',
        constraint_metric='max_drawdown',
        constraint_threshold=-0.15,  # Max drawdown must be > -15%
    )

    # Run optimization with fewer trials for speed (increase for production)
    n_trials = 100  # Increase to 500+ for production
    logger.info(f"\n🚀 Running optimization: {n_trials} trials")
    logger.info("Objective: Maximize Sharpe Ratio")
    logger.info("Constraint: Max Drawdown < 15%\n")

    opt_results = optimizer.optimize(
        n_trials=n_trials,
        n_jobs=1,  # Use -1 for parallel (all cores)
        show_progress_bar=True,
    )

    # Show optimization results
    logger.info("\n" + "=" * 70)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("=" * 70)

    logger.info(f"\n✅ Best Sharpe Ratio: {opt_results['best_value']:.4f}")
    logger.info(f"Best Trial: #{opt_results['best_trial_number']}")
    logger.info(f"Total Trials: {opt_results['n_trials']}")

    logger.info("\n📋 Best Parameters:")
    for param, value in opt_results['best_params'].items():
        logger.info(f"  {param}: {value}")

    # Get top 5 trials
    logger.info("\n🏆 Top 5 Trials:")
    top_trials = optimizer.get_top_trials(n=5)
    for i, trial in enumerate(top_trials, 1):
        logger.info(f"\n  {i}. Trial #{trial['trial_number']}: Sharpe = {trial['value']:.4f}")
        logger.info(f"     Params: {trial['params']}")

    # Validate on validation data
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: VALIDATION (Out-of-Sample)")
    logger.info("=" * 70)

    val_results = optimizer.validate_best_params(val_data, symbol='EURUSD')

    logger.info("\n📊 Validation Metrics:")
    val_metrics = val_results['metrics']
    logger.info(f"  Return:       {val_metrics['total_return']*100:.2f}%")
    logger.info(f"  CAGR:         {val_metrics['cagr']*100:.2f}%")
    logger.info(f"  Sharpe:       {val_metrics['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown: {val_metrics['max_drawdown']*100:.2f}%")
    logger.info(f"  Trades:       {val_metrics['num_trades']}")
    logger.info(f"  Win Rate:     {val_metrics.get('win_rate', 0)*100:.2f}%")

    # Calculate degradation
    train_sharpe = opt_results['best_value']
    val_sharpe = val_metrics['sharpe_ratio']
    degradation_pct = ((val_sharpe - train_sharpe) / abs(train_sharpe)) * 100 if train_sharpe != 0 else 0

    logger.info(f"\n📉 Performance Degradation:")
    logger.info(f"  Training Sharpe:   {train_sharpe:.4f}")
    logger.info(f"  Validation Sharpe: {val_sharpe:.4f}")
    logger.info(f"  Degradation:       {degradation_pct:+.1f}%")

    if degradation_pct > -30:
        logger.info("  ✅ Degradation acceptable (< 30%)")
    else:
        logger.warning("  ⚠️  High degradation - possible overfitting")

    # Full validation with train/val/test
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: COMPREHENSIVE VALIDATION")
    logger.info("=" * 70)

    validator = StrategyValidator(
        strategy=strategy,
        initial_capital=100.0,
        risk_per_trade=0.01,
        max_drawdown=0.15,
    )

    # Update strategy with best parameters
    strategy.set_parameters(**opt_results['best_params'])

    # Validate on all datasets
    full_results = validator.validate_train_val_test(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        symbol='EURUSD',
    )

    # Generate report
    report = validator.generate_validation_report(full_results)
    logger.info("\n" + report)

    # Save results
    logger.info("\n" + "=" * 70)
    logger.info("SAVING RESULTS")
    logger.info("=" * 70)

    output_dir = Path(__file__).parent.parent / "reports"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save optimization history
    history_file = output_dir / "optimization_history.csv"
    opt_history = optimizer.get_optimization_history()
    opt_history.to_csv(history_file, index=False)
    logger.info(f"\n💾 Optimization history: {history_file}")

    # Save best parameters
    params_file = output_dir / "best_parameters.txt"
    with open(params_file, 'w') as f:
        f.write("Best Parameters from Bayesian Optimization\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Objective: Maximize Sharpe Ratio\n")
        f.write(f"Best Sharpe: {opt_results['best_value']:.4f}\n")
        f.write(f"Trials Run: {opt_results['n_trials']}\n\n")
        f.write("Parameters:\n")
        for param, value in opt_results['best_params'].items():
            f.write(f"  {param}: {value}\n")
        f.write("\n")
        f.write("Validation Results:\n")
        f.write(f"  Validation Sharpe: {val_sharpe:.4f}\n")
        f.write(f"  Degradation: {degradation_pct:+.1f}%\n")

    logger.info(f"💾 Best parameters: {params_file}")

    # Save equity curves
    for dataset_name in ['train', 'validation', 'test']:
        equity_file = output_dir / f"optimized_strategy_{dataset_name}.csv"
        equity_curve = full_results[dataset_name]['results']['equity_curve']
        equity_curve.to_csv(equity_file)
        logger.info(f"💾 {dataset_name.capitalize()} equity: {equity_file}")

    # Final verdict
    logger.info("\n" + "=" * 70)
    logger.info("FINAL VERDICT")
    logger.info("=" * 70)

    go_no_go = full_results['go_no_go']

    if go_no_go.get('overall_pass', False):
        logger.info("\n🎉 ✅ STRATEGY APPROVED!")
        logger.info("Ready for next phase: Walk-Forward Analysis")
    elif go_no_go.get('positive_return', False) and go_no_go.get('sharpe_gt_1', False):
        logger.info("\n⚠️  STRATEGY SHOWS PROMISE")
        logger.info("Consider further optimization or different parameters")
    else:
        logger.info("\n❌ STRATEGY NEEDS IMPROVEMENT")
        logger.info("Consider revising strategy logic or trying different approach")

    logger.info("\n" + "=" * 70)


if __name__ == "__main__":
    main()
