"""
Validation Framework

Tools for validating optimized strategies on out-of-sample data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from loguru import logger

from ..backtesting import BacktestEngine
from ..strategies import BaseStrategy


class StrategyValidator:
    """
    Validate strategy performance on multiple datasets.

    Features:
    - Train/validation/test split validation
    - Multi-pair validation
    - Performance degradation analysis
    - Statistical significance testing
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 100.0,
        risk_per_trade: float = 0.01,
        max_drawdown: float = 0.15,
    ):
        """
        Initialize validator.

        Args:
            strategy: Strategy to validate
            initial_capital: Starting capital
            risk_per_trade: Risk per trade
            max_drawdown: Maximum drawdown threshold
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_drawdown = max_drawdown

        logger.info(f"Initialized validator for {strategy.name}")

    def validate_on_dataset(
        self, data: pd.DataFrame, dataset_name: str, symbol: str = 'EURUSD'
    ) -> Dict:
        """
        Validate strategy on a single dataset.

        Args:
            data: OHLCV data
            dataset_name: Name of dataset (e.g., 'train', 'validation', 'test')
            symbol: Trading symbol

        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating on {dataset_name} data...")

        # Generate signals
        signals = self.strategy.generate_signals(data)

        # Run backtest
        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            risk_per_trade=self.risk_per_trade,
            max_drawdown=self.max_drawdown,
        )

        results = engine.run(data=data, signals=signals, symbol=symbol)

        metrics = results['metrics']

        logger.info(
            f"{dataset_name.capitalize()}: "
            f"Return={metrics['total_return']*100:.2f}%, "
            f"Sharpe={metrics['sharpe_ratio']:.2f}, "
            f"Trades={metrics['num_trades']}"
        )

        return {
            'dataset': dataset_name,
            'metrics': metrics,
            'results': results,
            'data_range': (data.index[0], data.index[-1]),
            'n_candles': len(data),
        }

    def validate_train_val_test(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        symbol: str = 'EURUSD',
    ) -> Dict:
        """
        Validate on train/validation/test split.

        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data (out-of-sample)
            symbol: Trading symbol

        Returns:
            Dictionary with results for all datasets
        """
        logger.info("Running train/validation/test validation...")

        results = {
            'train': self.validate_on_dataset(train_data, 'train', symbol),
            'validation': self.validate_on_dataset(val_data, 'validation', symbol),
            'test': self.validate_on_dataset(test_data, 'test', symbol),
        }

        # Calculate degradation
        results['degradation'] = self._calculate_degradation(
            results['train']['metrics'],
            results['validation']['metrics'],
            results['test']['metrics'],
        )

        # Check go/no-go criteria
        results['go_no_go'] = self._check_go_no_go_criteria(results)

        return results

    def validate_multi_pair(
        self, data_dict: Dict[str, pd.DataFrame], dataset_name: str = 'test'
    ) -> Dict:
        """
        Validate strategy on multiple currency pairs.

        Args:
            data_dict: Dictionary of {symbol: data}
            dataset_name: Name of dataset

        Returns:
            Dictionary with results for each pair
        """
        logger.info(f"Validating on {len(data_dict)} pairs...")

        results = {}
        for symbol, data in data_dict.items():
            logger.info(f"Testing {symbol}...")
            results[symbol] = self.validate_on_dataset(data, dataset_name, symbol)

        # Calculate aggregate metrics
        results['aggregate'] = self._aggregate_multi_pair_results(results)

        return results

    def _calculate_degradation(
        self, train_metrics: Dict, val_metrics: Dict, test_metrics: Dict
    ) -> Dict:
        """
        Calculate performance degradation across datasets.

        Args:
            train_metrics: Training metrics
            val_metrics: Validation metrics
            test_metrics: Test metrics

        Returns:
            Dictionary with degradation percentages
        """
        degradation = {}

        metrics_to_compare = ['total_return', 'sharpe_ratio', 'max_drawdown']

        for metric in metrics_to_compare:
            train_val = train_metrics.get(metric, 0)
            val_val = val_metrics.get(metric, 0)
            test_val = test_metrics.get(metric, 0)

            if train_val != 0:
                val_deg = ((val_val - train_val) / abs(train_val)) * 100
                test_deg = ((test_val - train_val) / abs(train_val)) * 100
            else:
                val_deg = 0
                test_deg = 0

            degradation[metric] = {
                'train_to_val': val_deg,
                'train_to_test': test_deg,
                'val_to_test': ((test_val - val_val) / abs(val_val) * 100) if val_val != 0 else 0,
            }

        return degradation

    def _check_go_no_go_criteria(self, results: Dict) -> Dict:
        """
        Check if strategy meets go/no-go criteria.

        Args:
            results: Validation results

        Returns:
            Dictionary with pass/fail for each criterion
        """
        test_metrics = results['test']['metrics']
        degradation = results['degradation']

        criteria = {}

        # Returns criteria
        criteria['cagr_gt_12pct'] = test_metrics['cagr'] > 0.12
        criteria['sharpe_gt_1'] = test_metrics['sharpe_ratio'] > 1.0
        criteria['positive_return'] = test_metrics['total_return'] > 0

        # Risk criteria
        criteria['drawdown_lt_15pct'] = test_metrics['max_drawdown'] > -0.15
        criteria['drawdown_lt_20pct'] = test_metrics['max_drawdown'] > -0.20

        # Trade criteria
        criteria['min_trades'] = test_metrics['num_trades'] >= 20
        criteria['win_rate_gt_40pct'] = test_metrics.get('win_rate', 0) > 0.40

        # Degradation criteria (out-of-sample should be > 70% of in-sample)
        train_sharpe = results['train']['metrics']['sharpe_ratio']
        test_sharpe = test_metrics['sharpe_ratio']

        if train_sharpe > 0:
            sharpe_ratio = test_sharpe / train_sharpe
            criteria['sharpe_degradation_acceptable'] = sharpe_ratio > 0.70
        else:
            criteria['sharpe_degradation_acceptable'] = False

        # Overall pass
        critical_criteria = [
            'positive_return',
            'drawdown_lt_20pct',
            'min_trades',
        ]

        criteria['overall_pass'] = all(
            criteria.get(c, False) for c in critical_criteria
        )

        return criteria

    def _aggregate_multi_pair_results(self, results: Dict) -> Dict:
        """
        Aggregate results across multiple pairs.

        Args:
            results: Dictionary of results per pair

        Returns:
            Aggregated metrics
        """
        # Filter out non-pair keys (like 'aggregate')
        pair_results = {k: v for k, v in results.items() if k != 'aggregate'}

        if not pair_results:
            return {}

        # Calculate averages
        metrics_list = [r['metrics'] for r in pair_results.values()]

        aggregate = {
            'n_pairs': len(pair_results),
            'avg_return': np.mean([m['total_return'] for m in metrics_list]),
            'avg_sharpe': np.mean([m['sharpe_ratio'] for m in metrics_list]),
            'avg_drawdown': np.mean([m['max_drawdown'] for m in metrics_list]),
            'avg_trades': np.mean([m['num_trades'] for m in metrics_list]),
            'positive_pairs': sum(
                1 for m in metrics_list if m['total_return'] > 0
            ),
            'sharpe_gt_1_pairs': sum(
                1 for m in metrics_list if m['sharpe_ratio'] > 1.0
            ),
        }

        # Calculate success rate
        aggregate['pair_success_rate'] = (
            aggregate['positive_pairs'] / aggregate['n_pairs']
        )

        return aggregate

    def generate_validation_report(self, results: Dict) -> str:
        """
        Generate human-readable validation report.

        Args:
            results: Validation results

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append("STRATEGY VALIDATION REPORT")
        report.append("=" * 70)
        report.append("")

        # Strategy info
        report.append(f"Strategy: {self.strategy.name}")
        report.append(f"Parameters: {self.strategy.get_parameters()}")
        report.append("")

        # Results for each dataset
        for dataset in ['train', 'validation', 'test']:
            if dataset not in results:
                continue

            metrics = results[dataset]['metrics']
            data_range = results[dataset]['data_range']

            report.append(f"{dataset.upper()} RESULTS")
            report.append("-" * 70)
            report.append(f"Period: {data_range[0].date()} to {data_range[1].date()}")
            report.append(f"Return: {metrics['total_return']*100:.2f}%")
            report.append(f"CAGR: {metrics['cagr']*100:.2f}%")
            report.append(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
            report.append(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
            report.append(f"Trades: {metrics['num_trades']}")
            report.append(f"Win Rate: {metrics.get('win_rate', 0)*100:.2f}%")
            report.append("")

        # Go/No-Go criteria
        if 'go_no_go' in results:
            report.append("GO/NO-GO CRITERIA")
            report.append("-" * 70)

            criteria = results['go_no_go']
            for criterion, passed in criteria.items():
                if criterion == 'overall_pass':
                    continue
                status = "✅ PASS" if passed else "❌ FAIL"
                report.append(f"{criterion:40s} {status}")

            report.append("")
            report.append("-" * 70)

            if criteria.get('overall_pass', False):
                report.append("🎉 VERDICT: STRATEGY APPROVED FOR NEXT PHASE")
            else:
                report.append("❌ VERDICT: STRATEGY NEEDS IMPROVEMENT")

        report.append("=" * 70)

        return "\n".join(report)

    def __repr__(self) -> str:
        return f"StrategyValidator({self.strategy.name})"
