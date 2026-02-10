"""
Monte Carlo Simulation

Test strategy robustness by randomizing trade sequences
and analyzing distribution of outcomes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger


class MonteCarloSimulator:
    """
    Monte Carlo simulation for strategy validation.

    Methods:
    1. Trade Randomization: Shuffle trade sequence
    2. Bootstrap Resampling: Resample with replacement
    3. Parameter Perturbation: Test parameter sensitivity

    Provides confidence intervals and risk metrics.
    """

    def __init__(self, trades: pd.DataFrame, initial_capital: float = 100.0):
        """
        Initialize Monte Carlo simulator.

        Args:
            trades: DataFrame with individual trades
            initial_capital: Starting capital
        """
        self.trades = trades.copy()
        self.initial_capital = initial_capital
        self.simulations = []

        logger.info(f"Initialized Monte Carlo: {len(trades)} trades, ${initial_capital} capital")

    def run_simulations(
        self,
        n_simulations: int = 10000,
        method: str = 'shuffle',
        random_seed: Optional[int] = None,
    ) -> Dict:
        """
        Run Monte Carlo simulations.

        Args:
            n_simulations: Number of simulations (default: 10,000)
            method: 'shuffle' or 'bootstrap'
            random_seed: Random seed for reproducibility

        Returns:
            Dictionary with simulation results
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        logger.info(f"Running {n_simulations} Monte Carlo simulations (method: {method})...")

        results = []

        for i in range(n_simulations):
            if method == 'shuffle':
                sim_trades = self._shuffle_trades()
            elif method == 'bootstrap':
                sim_trades = self._bootstrap_trades()
            else:
                raise ValueError(f"Unknown method: {method}")

            # Calculate metrics for this simulation
            metrics = self._calculate_sim_metrics(sim_trades)
            results.append(metrics)

            if (i + 1) % 1000 == 0:
                logger.debug(f"Completed {i+1}/{n_simulations} simulations")

        self.simulations = results

        # Calculate statistics
        stats = self._calculate_statistics(results)

        logger.info("Monte Carlo simulations complete")

        return stats

    def _shuffle_trades(self) -> pd.DataFrame:
        """Shuffle trade sequence randomly."""
        return self.trades.sample(frac=1.0).reset_index(drop=True)

    def _bootstrap_trades(self) -> pd.DataFrame:
        """Bootstrap resample trades (with replacement)."""
        n = len(self.trades)
        indices = np.random.choice(n, size=n, replace=True)
        return self.trades.iloc[indices].reset_index(drop=True)

    def _calculate_sim_metrics(self, trades: pd.DataFrame) -> Dict:
        """Calculate metrics for a single simulation."""
        # Calculate equity curve
        cumulative_returns = (1 + trades['pnl_pct']).cumprod()
        equity = self.initial_capital * cumulative_returns

        # Total return
        total_return = (equity.iloc[-1] / self.initial_capital) - 1

        # Drawdown
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        max_drawdown = drawdown.min()

        # Sharpe (simplified)
        returns = trades['pnl_pct']
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'final_equity': equity.iloc[-1],
        }

    def _calculate_statistics(self, results: List[Dict]) -> Dict:
        """Calculate statistics across all simulations."""
        returns = np.array([r['total_return'] for r in results])
        drawdowns = np.array([r['max_drawdown'] for r in results])
        sharpes = np.array([r['sharpe_ratio'] for r in results])

        return {
            'mean_return': returns.mean(),
            'median_return': np.median(returns),
            'std_return': returns.std(),
            'return_5th': np.percentile(returns, 5),
            'return_95th': np.percentile(returns, 95),
            'mean_drawdown': drawdowns.mean(),
            'worst_drawdown': drawdowns.min(),
            'mean_sharpe': sharpes.mean(),
            'prob_profit': (returns > 0).mean(),
            'prob_drawdown_gt_15pct': (drawdowns < -0.15).mean(),
        }

    def generate_report(self) -> str:
        """Generate Monte Carlo analysis report."""
        if not self.simulations:
            return "No simulations run yet"

        stats = self._calculate_statistics(self.simulations)

        report = []
        report.append("=" * 70)
        report.append("MONTE CARLO SIMULATION RESULTS")
        report.append("=" * 70)
        report.append(f"\nSimulations: {len(self.simulations):,}")
        report.append(f"Original Trades: {len(self.trades)}\n")
        report.append("RETURN DISTRIBUTION")
        report.append("-" * 70)
        report.append(f"Mean Return:          {stats['mean_return']*100:>8.2f}%")
        report.append(f"Median Return:        {stats['median_return']*100:>8.2f}%")
        report.append(f"5th Percentile:       {stats['return_5th']*100:>8.2f}%")
        report.append(f"95th Percentile:      {stats['return_95th']*100:>8.2f}%\n")
        report.append("DRAWDOWN DISTRIBUTION")
        report.append("-" * 70)
        report.append(f"Mean Drawdown:        {stats['mean_drawdown']*100:>8.2f}%")
        report.append(f"Worst Drawdown:       {stats['worst_drawdown']*100:>8.2f}%\n")
        report.append("PROBABILITY ANALYSIS")
        report.append("-" * 70)
        report.append(f"P(Profit):            {stats['prob_profit']*100:>7.1f}%")
        report.append(f"P(Drawdown > 15%):    {stats['prob_drawdown_gt_15pct']*100:>7.1f}%")
        report.append("=" * 70)

        return "\n".join(report)
