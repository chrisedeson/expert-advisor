"""
Robustness Testing

Test strategy performance under various stress scenarios:
- Multiple currency pairs
- Higher transaction costs
- Different market conditions
- Parameter sensitivity
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger

from ..backtesting import BacktestEngine, TransactionCostModel
from ..strategies import BaseStrategy


class RobustnessAnalyzer:
    """
    Test strategy robustness across multiple dimensions.

    Tests:
    1. Multi-pair validation
    2. Transaction cost stress testing
    3. Parameter sensitivity analysis
    4. Market regime performance
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 100.0,
        risk_per_trade: float = 0.01,
        max_drawdown: float = 0.15,
    ):
        """
        Initialize robustness analyzer.

        Args:
            strategy: Strategy to test
            initial_capital: Starting capital
            risk_per_trade: Risk per trade
            max_drawdown: Maximum drawdown threshold
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_drawdown = max_drawdown

        self.results = {}

        logger.info(f"Initialized robustness analyzer for {strategy.name}")

    def test_multi_pair(self, data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """
        Test strategy on multiple currency pairs.

        Args:
            data_dict: Dictionary of {symbol: data}

        Returns:
            Dictionary with multi-pair results
        """
        logger.info(f"\n{'='*70}")
        logger.info("MULTI-PAIR ROBUSTNESS TEST")
        logger.info(f"{'='*70}")
        logger.info(f"Testing on {len(data_dict)} pairs...")

        results = {}

        for symbol, data in data_dict.items():
            logger.info(f"\nTesting {symbol}...")

            # Generate signals
            signals = self.strategy.generate_signals(data)

            # Run backtest
            engine = BacktestEngine(
                initial_capital=self.initial_capital,
                risk_per_trade=self.risk_per_trade,
                max_drawdown=self.max_drawdown,
            )

            backtest_results = engine.run(data=data, signals=signals, symbol=symbol)

            metrics = backtest_results['metrics']

            logger.info(
                f"  Return: {metrics['total_return']*100:>6.2f}%, "
                f"Sharpe: {metrics['sharpe_ratio']:>5.2f}, "
                f"Trades: {metrics['num_trades']:>3}"
            )

            results[symbol] = {
                'metrics': metrics,
                'results': backtest_results,
            }

        # Calculate aggregate stats
        aggregate = self._aggregate_pair_results(results)

        self.results['multi_pair'] = {
            'pairs': results,
            'aggregate': aggregate,
        }

        logger.info(f"\n{'='*70}")
        logger.info("MULTI-PAIR SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Avg Return:    {aggregate['avg_return']*100:.2f}%")
        logger.info(f"Avg Sharpe:    {aggregate['avg_sharpe']:.2f}")
        logger.info(f"Positive Pairs: {aggregate['positive_pairs']}/{aggregate['n_pairs']}")

        return results

    def test_cost_sensitivity(
        self, data: pd.DataFrame, symbol: str = 'EURUSD'
    ) -> Dict:
        """
        Test strategy with increased transaction costs.

        Args:
            data: Price data
            symbol: Trading symbol

        Returns:
            Dictionary with cost sensitivity results
        """
        logger.info(f"\n{'='*70}")
        logger.info("TRANSACTION COST STRESS TEST")
        logger.info(f"{'='*70}")

        # Test with different cost multipliers
        cost_multipliers = [1.0, 1.5, 2.0, 3.0]

        results = {}

        for multiplier in cost_multipliers:
            logger.info(f"\nTesting with {multiplier}x costs...")

            # Create cost model with multiplier
            cost_model = TransactionCostModel(cost_multiplier=multiplier)

            # Generate signals
            signals = self.strategy.generate_signals(data)

            # Run backtest
            engine = BacktestEngine(
                initial_capital=self.initial_capital,
                risk_per_trade=self.risk_per_trade,
                max_drawdown=self.max_drawdown,
                transaction_cost_model=cost_model,
            )

            backtest_results = engine.run(data=data, signals=signals, symbol=symbol)

            metrics = backtest_results['metrics']

            logger.info(
                f"  Return: {metrics['total_return']*100:>6.2f}%, "
                f"Sharpe: {metrics['sharpe_ratio']:>5.2f}"
            )

            results[f'{multiplier}x'] = {
                'multiplier': multiplier,
                'metrics': metrics,
            }

        self.results['cost_sensitivity'] = results

        # Calculate degradation
        base_return = results['1.0x']['metrics']['total_return']
        high_cost_return = results['3.0x']['metrics']['total_return']
        degradation = ((high_cost_return - base_return) / abs(base_return)) * 100 if base_return != 0 else 0

        logger.info(f"\nPerformance with 3x costs: {degradation:+.1f}% degradation")

        return results

    def test_parameter_sensitivity(
        self, data: pd.DataFrame, param_name: str, param_range: List, symbol: str = 'EURUSD'
    ) -> Dict:
        """
        Test sensitivity to a specific parameter.

        Args:
            data: Price data
            param_name: Parameter to vary
            param_range: List of parameter values to test
            symbol: Trading symbol

        Returns:
            Dictionary with sensitivity results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"PARAMETER SENSITIVITY: {param_name}")
        logger.info(f"{'='*70}")

        results = {}

        for value in param_range:
            logger.info(f"\nTesting {param_name}={value}...")

            # Update parameter
            self.strategy.set_parameters(**{param_name: value})

            # Generate signals
            signals = self.strategy.generate_signals(data)

            # Run backtest
            engine = BacktestEngine(
                initial_capital=self.initial_capital,
                risk_per_trade=self.risk_per_trade,
                max_drawdown=self.max_drawdown,
            )

            backtest_results = engine.run(data=data, signals=signals, symbol=symbol)

            metrics = backtest_results['metrics']

            logger.info(
                f"  Return: {metrics['total_return']*100:>6.2f}%, "
                f"Sharpe: {metrics['sharpe_ratio']:>5.2f}"
            )

            results[value] = {
                'param_value': value,
                'metrics': metrics,
            }

        self.results[f'sensitivity_{param_name}'] = results

        return results

    def _aggregate_pair_results(self, results: Dict) -> Dict:
        """Aggregate results across multiple pairs."""
        metrics_list = [r['metrics'] for r in results.values()]

        aggregate = {
            'n_pairs': len(results),
            'avg_return': np.mean([m['total_return'] for m in metrics_list]),
            'avg_sharpe': np.mean([m['sharpe_ratio'] for m in metrics_list]),
            'avg_drawdown': np.mean([m['max_drawdown'] for m in metrics_list]),
            'positive_pairs': sum(1 for m in metrics_list if m['total_return'] > 0),
            'min_return': min(m['total_return'] for m in metrics_list),
            'max_return': max(m['total_return'] for m in metrics_list),
        }

        return aggregate

    def generate_report(self) -> str:
        """Generate comprehensive robustness report."""
        report = []
        report.append("=" * 70)
        report.append("ROBUSTNESS ANALYSIS REPORT")
        report.append("=" * 70)
        report.append("")

        # Multi-pair results
        if 'multi_pair' in self.results:
            report.append("MULTI-PAIR PERFORMANCE")
            report.append("-" * 70)

            agg = self.results['multi_pair']['aggregate']
            report.append(f"Pairs Tested:      {agg['n_pairs']}")
            report.append(f"Avg Return:        {agg['avg_return']*100:.2f}%")
            report.append(f"Avg Sharpe:        {agg['avg_sharpe']:.2f}")
            report.append(f"Positive Pairs:    {agg['positive_pairs']}/{agg['n_pairs']}")
            report.append(f"Min Return:        {agg['min_return']*100:.2f}%")
            report.append(f"Max Return:        {agg['max_return']*100:.2f}%")
            report.append("")

        # Cost sensitivity
        if 'cost_sensitivity' in self.results:
            report.append("TRANSACTION COST SENSITIVITY")
            report.append("-" * 70)

            for key, result in self.results['cost_sensitivity'].items():
                multiplier = result['multiplier']
                ret = result['metrics']['total_return']
                report.append(f"{multiplier}x costs:     Return = {ret*100:>6.2f}%")

            report.append("")

        report.append("=" * 70)

        return "\n".join(report)

    def __repr__(self) -> str:
        return f"RobustnessAnalyzer({self.strategy.name})"
