"""
Parameter Optimizer using Bayesian Optimization

Uses Optuna for intelligent parameter search.
"""

import optuna
import pandas as pd
import numpy as np
from typing import Dict, Callable, Optional, List
from loguru import logger

from ..backtesting import BacktestEngine
from ..strategies import BaseStrategy


class ParameterOptimizer:
    """
    Bayesian parameter optimizer using Optuna.

    Features:
    - Intelligent search (learns from previous trials)
    - Constraint handling (max drawdown, min trades)
    - Multi-objective optimization support
    - Progress tracking and early stopping
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        initial_capital: float = 100.0,
        risk_per_trade: float = 0.01,
        max_drawdown: float = 0.15,
        objective_metric: str = 'sharpe_ratio',
        constraint_metric: Optional[str] = None,
        constraint_threshold: Optional[float] = None,
    ):
        """
        Initialize parameter optimizer.

        Args:
            strategy: Strategy instance to optimize
            data: Training data for optimization
            initial_capital: Starting capital
            risk_per_trade: Risk per trade (default: 1%)
            max_drawdown: Maximum drawdown (default: 15%)
            objective_metric: Metric to maximize (default: 'sharpe_ratio')
            constraint_metric: Optional constraint metric (e.g., 'max_drawdown')
            constraint_threshold: Threshold for constraint (e.g., -0.15)
        """
        self.strategy = strategy
        self.data = data
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_drawdown = max_drawdown
        self.objective_metric = objective_metric
        self.constraint_metric = constraint_metric
        self.constraint_threshold = constraint_threshold

        self.best_params = None
        self.best_value = None
        self.study = None

        logger.info(
            f"Initialized optimizer for {strategy.name}: "
            f"objective={objective_metric}"
        )

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna.

        Args:
            trial: Optuna trial object

        Returns:
            Objective value to maximize
        """
        # Get parameter space from strategy
        param_space = self.strategy.get_parameter_space()

        # Sample parameters
        params = {}
        for param_name, (min_val, max_val, param_type) in param_space.items():
            if param_type == 'int':
                params[param_name] = trial.suggest_int(param_name, min_val, max_val)
            elif param_type == 'float':
                params[param_name] = trial.suggest_float(param_name, min_val, max_val)
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")

        # Update strategy parameters
        self.strategy.set_parameters(**params)

        # Generate signals
        try:
            signals = self.strategy.generate_signals(self.data)
        except Exception as e:
            logger.warning(f"Signal generation failed: {e}")
            return -np.inf

        # Run backtest
        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            risk_per_trade=self.risk_per_trade,
            max_drawdown=self.max_drawdown,
        )

        try:
            results = engine.run(
                data=self.data,
                signals=signals,
                symbol='EURUSD',  # TODO: Make configurable
            )
        except Exception as e:
            logger.warning(f"Backtest failed: {e}")
            return -np.inf

        metrics = results['metrics']

        # Check constraints
        if self.constraint_metric and self.constraint_threshold:
            constraint_value = metrics.get(self.constraint_metric, 0)
            if self.constraint_metric == 'max_drawdown':
                # For drawdown, constraint is upper bound (e.g., < -0.15)
                if constraint_value < self.constraint_threshold:
                    logger.debug(
                        f"Trial {trial.number}: Constraint violated "
                        f"({self.constraint_metric}={constraint_value:.4f})"
                    )
                    return -np.inf
            else:
                # For other metrics, constraint is lower bound
                if constraint_value < self.constraint_threshold:
                    return -np.inf

        # Check minimum number of trades
        if metrics['num_trades'] < 20:
            logger.debug(f"Trial {trial.number}: Too few trades ({metrics['num_trades']})")
            return -np.inf

        # Get objective value
        objective_value = metrics.get(self.objective_metric, -np.inf)

        # Log trial results
        logger.debug(
            f"Trial {trial.number}: {self.objective_metric}={objective_value:.4f}, "
            f"params={params}"
        )

        return objective_value

    def optimize(
        self,
        n_trials: int = 500,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        show_progress_bar: bool = True,
    ) -> Dict:
        """
        Run optimization.

        Args:
            n_trials: Number of trials (default: 500)
            timeout: Timeout in seconds (optional)
            n_jobs: Number of parallel jobs (default: 1, use -1 for all cores)
            show_progress_bar: Show progress bar

        Returns:
            Dictionary with best parameters and results
        """
        logger.info(
            f"Starting optimization: {n_trials} trials, "
            f"objective={self.objective_metric}"
        )

        # Create study
        self.study = optuna.create_study(
            direction='maximize',
            study_name=f"{self.strategy.name}_optimization",
            sampler=optuna.samplers.TPESampler(seed=42),  # Reproducible
        )

        # Run optimization
        self.study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=show_progress_bar,
            catch=(Exception,),  # Catch exceptions and continue
        )

        # Get best results
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value

        logger.info(
            f"Optimization complete: Best {self.objective_metric}={self.best_value:.4f}"
        )
        logger.info(f"Best parameters: {self.best_params}")

        # Get best trial details
        best_trial = self.study.best_trial

        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'n_trials': len(self.study.trials),
            'best_trial_number': best_trial.number,
            'study': self.study,
        }

    def get_best_parameters(self) -> Dict:
        """
        Get best parameters found.

        Returns:
            Dictionary of best parameters
        """
        if self.best_params is None:
            raise ValueError("No optimization has been run yet")

        return self.best_params.copy()

    def get_optimization_history(self) -> pd.DataFrame:
        """
        Get optimization history as DataFrame.

        Returns:
            DataFrame with trial history
        """
        if self.study is None:
            raise ValueError("No optimization has been run yet")

        # Convert trials to DataFrame
        trials_df = self.study.trials_dataframe()

        return trials_df

    def plot_optimization_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot optimization history.

        Args:
            save_path: Optional path to save plot
        """
        if self.study is None:
            raise ValueError("No optimization has been run yet")

        try:
            import matplotlib.pyplot as plt

            # Plot optimization history
            fig = optuna.visualization.plot_optimization_history(self.study)

            if save_path:
                fig.write_image(save_path)
                logger.info(f"Saved optimization history to {save_path}")
            else:
                fig.show()

        except ImportError:
            logger.warning("Plotly not available for visualization")

    def plot_param_importances(self, save_path: Optional[str] = None) -> None:
        """
        Plot parameter importances.

        Args:
            save_path: Optional path to save plot
        """
        if self.study is None:
            raise ValueError("No optimization has been run yet")

        try:
            import matplotlib.pyplot as plt

            # Plot parameter importances
            fig = optuna.visualization.plot_param_importances(self.study)

            if save_path:
                fig.write_image(save_path)
                logger.info(f"Saved parameter importances to {save_path}")
            else:
                fig.show()

        except ImportError:
            logger.warning("Plotly not available for visualization")

    def get_top_trials(self, n: int = 10) -> List[Dict]:
        """
        Get top N trials.

        Args:
            n: Number of top trials to return

        Returns:
            List of trial dictionaries
        """
        if self.study is None:
            raise ValueError("No optimization has been run yet")

        # Sort trials by value
        sorted_trials = sorted(
            self.study.trials,
            key=lambda t: t.value if t.value is not None else -np.inf,
            reverse=True,
        )

        # Get top N
        top_trials = []
        for trial in sorted_trials[:n]:
            if trial.value is not None:
                top_trials.append({
                    'trial_number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                })

        return top_trials

    def validate_best_params(
        self, validation_data: pd.DataFrame, symbol: str = 'EURUSD'
    ) -> Dict:
        """
        Validate best parameters on validation data.

        Args:
            validation_data: Validation dataset
            symbol: Trading symbol

        Returns:
            Dictionary with validation metrics
        """
        if self.best_params is None:
            raise ValueError("No optimization has been run yet")

        logger.info("Validating best parameters on validation data...")

        # Set best parameters
        self.strategy.set_parameters(**self.best_params)

        # Generate signals
        signals = self.strategy.generate_signals(validation_data)

        # Run backtest
        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            risk_per_trade=self.risk_per_trade,
            max_drawdown=self.max_drawdown,
        )

        results = engine.run(
            data=validation_data,
            signals=signals,
            symbol=symbol,
        )

        metrics = results['metrics']

        logger.info(
            f"Validation complete: {self.objective_metric}={metrics[self.objective_metric]:.4f}"
        )

        return {
            'metrics': metrics,
            'results': results,
            'degradation': self._calculate_degradation(metrics),
        }

    def _calculate_degradation(self, validation_metrics: Dict) -> Dict:
        """
        Calculate performance degradation from training to validation.

        Args:
            validation_metrics: Validation metrics

        Returns:
            Dictionary with degradation percentages
        """
        if self.best_value is None:
            return {}

        degradation = {}
        val_objective = validation_metrics.get(self.objective_metric, 0)

        if self.best_value != 0:
            deg_pct = ((val_objective - self.best_value) / abs(self.best_value)) * 100
            degradation[self.objective_metric] = deg_pct

        return degradation

    def __repr__(self) -> str:
        status = "optimized" if self.best_params else "not optimized"
        return (
            f"ParameterOptimizer({self.strategy.name}, "
            f"objective={self.objective_metric}, {status})"
        )
