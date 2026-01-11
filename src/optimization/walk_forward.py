"""
Walk-Forward Analysis

Primary defense against overfitting. Tests if optimized parameters
remain profitable on unseen data using rolling windows.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger
from datetime import timedelta

from ..backtesting import BacktestEngine
from ..strategies import BaseStrategy
from .parameter_optimizer import ParameterOptimizer


class WalkForwardAnalyzer:
    """
    Walk-forward analysis for strategy validation.

    Process:
    1. Split data into multiple windows
    2. For each window:
       - Train: Optimize parameters
       - Test: Apply parameters to next period
    3. Combine results to assess real-world performance
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        train_period_months: int = 6,
        test_period_months: int = 3,
        initial_capital: float = 100.0,
        risk_per_trade: float = 0.01,
        max_drawdown: float = 0.15,
        optimization_trials: int = 100,
    ):
        """
        Initialize walk-forward analyzer.

        Args:
            strategy: Strategy to analyze
            data: Full dataset
            train_period_months: Training window size (default: 6 months)
            test_period_months: Test window size (default: 3 months)
            initial_capital: Starting capital
            risk_per_trade: Risk per trade
            max_drawdown: Maximum drawdown threshold
            optimization_trials: Trials per optimization
        """
        self.strategy = strategy
        self.data = data
        self.train_period = train_period_months
        self.test_period = test_period_months
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_drawdown = max_drawdown
        self.optimization_trials = optimization_trials

        self.windows = []
        self.results = []

        logger.info(
            f"Initialized walk-forward analyzer: "
            f"{train_period_months}M train / {test_period_months}M test"
        )

    def create_windows(self) -> List[Dict]:
        """
        Create rolling train/test windows.

        Returns:
            List of window dictionaries with train/test data
        """
        # Calculate window size in days (approximate)
        train_days = self.train_period * 30
        test_days = self.test_period * 30
        total_window = train_days + test_days

        # Get date range
        start_date = self.data.index[0]
        end_date = self.data.index[-1]
        total_days = (end_date - start_date).days

        # Calculate number of windows
        n_windows = max(1, int((total_days - train_days) / test_days))

        logger.info(
            f"Creating {n_windows} walk-forward windows "
            f"(train={train_days}d, test={test_days}d)"
        )

        windows = []

        for i in range(n_windows):
            # Calculate window boundaries
            window_start = start_date + timedelta(days=i * test_days)
            train_end = window_start + timedelta(days=train_days)
            test_end = train_end + timedelta(days=test_days)

            # Extract data
            train_data = self.data[
                (self.data.index >= window_start) & (self.data.index < train_end)
            ]
            test_data = self.data[
                (self.data.index >= train_end) & (self.data.index < test_end)
            ]

            # Skip if insufficient data
            if len(train_data) < 100 or len(test_data) < 20:
                continue

            windows.append({
                'window_id': i,
                'train_start': window_start,
                'train_end': train_end,
                'test_start': train_end,
                'test_end': test_end,
                'train_data': train_data,
                'test_data': test_data,
                'train_bars': len(train_data),
                'test_bars': len(test_data),
            })

        self.windows = windows
        logger.info(f"Created {len(windows)} valid windows")

        return windows

    def run_window(self, window: Dict, symbol: str = 'EURUSD') -> Dict:
        """
        Run optimization and testing for a single window.

        Args:
            window: Window dictionary with train/test data
            symbol: Trading symbol

        Returns:
            Dictionary with window results
        """
        window_id = window['window_id']
        logger.info(
            f"\n{'='*70}\n"
            f"Window {window_id}: "
            f"{window['train_start'].date()} to {window['test_end'].date()}\n"
            f"{'='*70}"
        )

        # Phase 1: Optimize on training data
        logger.info(f"Phase 1: Optimizing on training data...")

        optimizer = ParameterOptimizer(
            strategy=self.strategy,
            data=window['train_data'],
            initial_capital=self.initial_capital,
            risk_per_trade=self.risk_per_trade,
            max_drawdown=self.max_drawdown,
            objective_metric='sharpe_ratio',
            constraint_metric='max_drawdown',
            constraint_threshold=-self.max_drawdown,
        )

        opt_results = optimizer.optimize(
            n_trials=self.optimization_trials,
            show_progress_bar=False,
        )

        train_sharpe = opt_results['best_value']
        best_params = opt_results['best_params']

        logger.info(f"Training Sharpe: {train_sharpe:.4f}")
        logger.info(f"Best params: {best_params}")

        # Phase 2: Test on out-of-sample data
        logger.info(f"\nPhase 2: Testing on out-of-sample data...")

        self.strategy.set_parameters(**best_params)
        test_signals = self.strategy.generate_signals(window['test_data'])

        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            risk_per_trade=self.risk_per_trade,
            max_drawdown=self.max_drawdown,
        )

        test_results = engine.run(
            data=window['test_data'],
            signals=test_signals,
            symbol=symbol,
        )

        test_metrics = test_results['metrics']
        test_sharpe = test_metrics['sharpe_ratio']

        logger.info(f"Test Sharpe: {test_sharpe:.4f}")
        logger.info(f"Test Return: {test_metrics['total_return']*100:.2f}%")
        logger.info(f"Test Trades: {test_metrics['num_trades']}")

        # Calculate degradation
        if train_sharpe != 0:
            degradation = ((test_sharpe - train_sharpe) / abs(train_sharpe)) * 100
        else:
            degradation = 0

        logger.info(f"Degradation: {degradation:+.1f}%")

        return {
            'window_id': window_id,
            'train_start': window['train_start'],
            'train_end': window['train_end'],
            'test_start': window['test_start'],
            'test_end': window['test_end'],
            'train_bars': window['train_bars'],
            'test_bars': window['test_bars'],
            'optimized_params': best_params,
            'train_sharpe': train_sharpe,
            'test_sharpe': test_sharpe,
            'test_metrics': test_metrics,
            'degradation_pct': degradation,
            'equity_curve': test_results['equity_curve'],
            'trades': test_results['trades'],
        }

    def run_analysis(self, symbol: str = 'EURUSD') -> Dict:
        """
        Run complete walk-forward analysis.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with analysis results
        """
        logger.info("=" * 70)
        logger.info("WALK-FORWARD ANALYSIS")
        logger.info("=" * 70)

        # Create windows if not already done
        if not self.windows:
            self.create_windows()

        if len(self.windows) == 0:
            raise ValueError("No valid windows created. Data may be too short.")

        # Run each window
        results = []
        for window in self.windows:
            try:
                result = self.run_window(window, symbol)
                results.append(result)
            except Exception as e:
                logger.error(f"Window {window['window_id']} failed: {e}")
                continue

        self.results = results

        # Aggregate results
        aggregate_stats = self._aggregate_results()

        logger.info("\n" + "=" * 70)
        logger.info("WALK-FORWARD ANALYSIS COMPLETE")
        logger.info("=" * 70)

        return {
            'windows': results,
            'aggregate': aggregate_stats,
            'n_windows': len(results),
        }

    def _aggregate_results(self) -> Dict:
        """
        Aggregate results across all windows.

        Returns:
            Dictionary with aggregate statistics
        """
        if not self.results:
            return {}

        # Extract metrics
        train_sharpes = [r['train_sharpe'] for r in self.results]
        test_sharpes = [r['test_sharpe'] for r in self.results]
        test_returns = [r['test_metrics']['total_return'] for r in self.results]
        degradations = [r['degradation_pct'] for r in self.results]

        # Calculate statistics
        stats = {
            'n_windows': len(self.results),
            'avg_train_sharpe': np.mean(train_sharpes),
            'avg_test_sharpe': np.mean(test_sharpes),
            'std_test_sharpe': np.std(test_sharpes),
            'min_test_sharpe': np.min(test_sharpes),
            'max_test_sharpe': np.max(test_sharpes),
            'avg_test_return': np.mean(test_returns),
            'positive_windows': sum(1 for r in test_returns if r > 0),
            'positive_pct': (sum(1 for r in test_returns if r > 0) / len(test_returns)) * 100,
            'avg_degradation': np.mean(degradations),
            'stability_ratio': np.mean(test_sharpes) / np.mean(train_sharpes) if np.mean(train_sharpes) != 0 else 0,
        }

        return stats

    def get_combined_equity_curve(self) -> pd.DataFrame:
        """
        Get combined equity curve across all test windows.

        Returns:
            DataFrame with combined equity curve
        """
        if not self.results:
            return pd.DataFrame()

        # Concatenate all test equity curves
        curves = []
        cumulative_capital = self.initial_capital

        for result in self.results:
            equity_curve = result['equity_curve'].copy()

            # Adjust equity to start from previous window's end
            start_equity = equity_curve['equity'].iloc[0]
            equity_curve['equity_adjusted'] = (
                cumulative_capital + (equity_curve['equity'] - start_equity)
            )

            curves.append(equity_curve[['equity_adjusted']])

            # Update cumulative capital
            cumulative_capital = equity_curve['equity_adjusted'].iloc[-1]

        combined = pd.concat(curves)
        combined.columns = ['equity']

        return combined

    def generate_report(self) -> str:
        """
        Generate walk-forward analysis report.

        Returns:
            Formatted report string
        """
        if not self.results:
            return "No results available. Run analysis first."

        report = []
        report.append("=" * 70)
        report.append("WALK-FORWARD ANALYSIS REPORT")
        report.append("=" * 70)
        report.append("")

        # Strategy info
        report.append(f"Strategy: {self.strategy.name}")
        report.append(f"Train Period: {self.train_period} months")
        report.append(f"Test Period: {self.test_period} months")
        report.append(f"Number of Windows: {len(self.results)}")
        report.append("")

        # Aggregate stats
        stats = self._aggregate_results()

        report.append("AGGREGATE PERFORMANCE")
        report.append("-" * 70)
        report.append(f"Average Train Sharpe:     {stats['avg_train_sharpe']:.4f}")
        report.append(f"Average Test Sharpe:      {stats['avg_test_sharpe']:.4f}")
        report.append(f"Std Dev Test Sharpe:      {stats['std_test_sharpe']:.4f}")
        report.append(f"Min Test Sharpe:          {stats['min_test_sharpe']:.4f}")
        report.append(f"Max Test Sharpe:          {stats['max_test_sharpe']:.4f}")
        report.append(f"Average Test Return:      {stats['avg_test_return']*100:.2f}%")
        report.append(f"Positive Windows:         {stats['positive_windows']}/{stats['n_windows']} ({stats['positive_pct']:.1f}%)")
        report.append(f"Average Degradation:      {stats['avg_degradation']:+.1f}%")
        report.append(f"Stability Ratio:          {stats['stability_ratio']:.2f}")
        report.append("")

        # Individual windows
        report.append("INDIVIDUAL WINDOWS")
        report.append("-" * 70)

        for result in self.results:
            report.append(
                f"Window {result['window_id']}: "
                f"{result['test_start'].date()} to {result['test_end'].date()}"
            )
            report.append(f"  Train Sharpe: {result['train_sharpe']:.4f}")
            report.append(f"  Test Sharpe:  {result['test_sharpe']:.4f}")
            report.append(f"  Return:       {result['test_metrics']['total_return']*100:+.2f}%")
            report.append(f"  Trades:       {result['test_metrics']['num_trades']}")
            report.append(f"  Degradation:  {result['degradation_pct']:+.1f}%")
            report.append("")

        # Verdict
        report.append("=" * 70)
        report.append("VERDICT")
        report.append("=" * 70)

        # Check criteria
        passed = []
        failed = []

        if stats['avg_test_sharpe'] > 0.3:
            passed.append(f"✅ Avg test Sharpe > 0.3 ({stats['avg_test_sharpe']:.4f})")
        else:
            failed.append(f"❌ Avg test Sharpe < 0.3 ({stats['avg_test_sharpe']:.4f})")

        if stats['positive_pct'] >= 60:
            passed.append(f"✅ Positive windows ≥ 60% ({stats['positive_pct']:.1f}%)")
        else:
            failed.append(f"❌ Positive windows < 60% ({stats['positive_pct']:.1f}%)")

        if stats['stability_ratio'] > 0.7:
            passed.append(f"✅ Stability ratio > 0.7 ({stats['stability_ratio']:.2f})")
        else:
            failed.append(f"⚠️  Stability ratio < 0.7 ({stats['stability_ratio']:.2f})")

        if abs(stats['avg_degradation']) < 30:
            passed.append(f"✅ Degradation < 30% ({stats['avg_degradation']:+.1f}%)")
        else:
            failed.append(f"❌ Degradation > 30% ({stats['avg_degradation']:+.1f}%)")

        report.append("")
        for p in passed:
            report.append(p)
        for f in failed:
            report.append(f)

        report.append("")
        report.append("-" * 70)
        if len(failed) == 0:
            report.append("🎉 STRATEGY PASSES WALK-FORWARD ANALYSIS")
        elif len(failed) <= 1:
            report.append("⚠️  STRATEGY SHOWS PROMISE - Minor improvements needed")
        else:
            report.append("❌ STRATEGY NEEDS SIGNIFICANT IMPROVEMENT")

        report.append("=" * 70)

        return "\n".join(report)

    def __repr__(self) -> str:
        return (
            f"WalkForwardAnalyzer({self.strategy.name}, "
            f"windows={len(self.windows)}, "
            f"train={self.train_period}M, test={self.test_period}M)"
        )
