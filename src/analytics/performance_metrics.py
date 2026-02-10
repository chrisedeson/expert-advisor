"""
Performance Metrics Calculator

Comprehensive performance metrics for trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from loguru import logger


class PerformanceMetrics:
    """
    Calculate comprehensive performance metrics for trading strategies.

    Metrics include:
    - Returns (total, CAGR, monthly, yearly)
    - Risk (volatility, drawdown, VaR)
    - Risk-adjusted (Sharpe, Sortino, Calmar, Omega)
    - Trading (win rate, profit factor, expectancy)
    - Time-based (by hour, day, month)
    """

    def __init__(self, equity_curve: pd.DataFrame, trades: pd.DataFrame):
        """
        Initialize metrics calculator.

        Args:
            equity_curve: DataFrame with equity over time
            trades: DataFrame with individual trades
        """
        self.equity_curve = equity_curve
        self.trades = trades

        logger.info(f"Initialized performance metrics: {len(trades)} trades")

    def calculate_all_metrics(self) -> Dict:
        """
        Calculate all performance metrics.

        Returns:
            Dictionary with comprehensive metrics
        """
        metrics = {}

        # Returns
        metrics.update(self.calculate_return_metrics())

        # Risk
        metrics.update(self.calculate_risk_metrics())

        # Risk-adjusted
        metrics.update(self.calculate_risk_adjusted_metrics())

        # Trading
        metrics.update(self.calculate_trading_metrics())

        # Time-based
        metrics.update(self.calculate_time_metrics())

        return metrics

    def calculate_return_metrics(self) -> Dict:
        """Calculate return-based metrics."""
        equity = self.equity_curve['equity']
        initial = equity.iloc[0]
        final = equity.iloc[-1]

        # Total return
        total_return = (final / initial) - 1 if initial > 0 else 0.0

        # Time period
        start_date = self.equity_curve.index[0]
        end_date = self.equity_curve.index[-1]
        years = (end_date - start_date).days / 365.25

        # CAGR (Compound Annual Growth Rate)
        if years > 0 and initial > 0 and final > 0:
            cagr = ((final / initial) ** (1 / years)) - 1
        else:
            cagr = 0.0

        # Monthly returns (use 'ME' for month-end instead of deprecated 'M')
        monthly_returns = equity.resample('ME').last().pct_change().dropna()

        return {
            'total_return': total_return,
            'cagr': cagr,
            'avg_monthly_return': monthly_returns.mean(),
            'best_month': monthly_returns.max(),
            'worst_month': monthly_returns.min(),
            'positive_months': (monthly_returns > 0).sum(),
            'total_months': len(monthly_returns),
        }

    def calculate_risk_metrics(self) -> Dict:
        """Calculate risk-based metrics."""
        equity = self.equity_curve['equity']

        # Volatility (annualized)
        returns = equity.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Assuming daily data

        # Drawdown
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        max_drawdown = drawdown.min()

        # Drawdown duration
        drawdown_start = None
        max_duration = 0
        current_duration = 0

        for i, dd in enumerate(drawdown):
            if dd < 0:
                if drawdown_start is None:
                    drawdown_start = i
                current_duration = i - drawdown_start
                max_duration = max(max_duration, current_duration)
            else:
                drawdown_start = None
                current_duration = 0

        # Value at Risk (5%)
        var_5 = np.percentile(returns, 5)

        return {
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'avg_drawdown': drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0,
            'max_drawdown_duration': max_duration,
            'value_at_risk_5': var_5,
        }

    def calculate_risk_adjusted_metrics(self) -> Dict:
        """Calculate risk-adjusted performance metrics."""
        equity = self.equity_curve['equity']
        returns = equity.pct_change().dropna()

        # Sharpe Ratio
        mean_return = returns.mean()
        std_return = returns.std()
        sharpe = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0

        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        sortino = (mean_return / downside_std * np.sqrt(252)) if downside_std > 0 else 0

        # Calmar Ratio (CAGR / Max Drawdown)
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        max_dd = abs(drawdown.min())

        start_date = self.equity_curve.index[0]
        end_date = self.equity_curve.index[-1]
        years = (end_date - start_date).days / 365.25

        cagr = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / years)) - 1 if years > 0 else 0
        calmar = cagr / max_dd if max_dd > 0 else 0

        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
        }

    def calculate_trading_metrics(self) -> Dict:
        """Calculate trading-specific metrics."""
        if len(self.trades) == 0:
            return {
                'num_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'expectancy': 0,
                'avg_trade_duration': 0,
            }

        # Win rate
        winning_trades = self.trades[self.trades['pnl_pct'] > 0]
        losing_trades = self.trades[self.trades['pnl_pct'] <= 0]

        win_rate = len(winning_trades) / len(self.trades) if len(self.trades) > 0 else 0

        # Average win/loss
        avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0

        # Profit factor
        gross_profit = winning_trades['pnl_pct'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl_pct'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # Trade duration
        avg_duration = self.trades['duration'].mean() if 'duration' in self.trades.columns else 0

        # Consecutive wins/losses
        pnl_sign = (self.trades['pnl_pct'] > 0).astype(int)
        consecutive_wins = self._max_consecutive(pnl_sign, 1)
        consecutive_losses = self._max_consecutive(pnl_sign, 0)

        return {
            'num_trades': len(self.trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'avg_trade_duration': avg_duration,
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses,
            'largest_win': self.trades['pnl_pct'].max(),
            'largest_loss': self.trades['pnl_pct'].min(),
        }

    def calculate_time_metrics(self) -> Dict:
        """Calculate time-based performance metrics."""
        if 'entry_time' not in self.trades.columns:
            return {}

        # By day of week
        self.trades['day_of_week'] = pd.to_datetime(self.trades['entry_time']).dt.dayofweek
        day_performance = self.trades.groupby('day_of_week')['pnl_pct'].agg(['mean', 'count'])

        # By hour
        self.trades['hour'] = pd.to_datetime(self.trades['entry_time']).dt.hour
        hour_performance = self.trades.groupby('hour')['pnl_pct'].agg(['mean', 'count'])

        return {
            'best_day': day_performance['mean'].idxmax() if len(day_performance) > 0 else None,
            'worst_day': day_performance['mean'].idxmin() if len(day_performance) > 0 else None,
            'best_hour': hour_performance['mean'].idxmax() if len(hour_performance) > 0 else None,
            'worst_hour': hour_performance['mean'].idxmin() if len(hour_performance) > 0 else None,
        }

    def _max_consecutive(self, series: pd.Series, value: int) -> int:
        """Find maximum consecutive occurrences of a value."""
        max_count = 0
        current_count = 0

        for v in series:
            if v == value:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count

    def generate_summary(self) -> str:
        """Generate human-readable summary."""
        metrics = self.calculate_all_metrics()

        summary = []
        summary.append("=" * 70)
        summary.append("PERFORMANCE SUMMARY")
        summary.append("=" * 70)
        summary.append("")

        # Returns
        summary.append("RETURNS")
        summary.append("-" * 70)
        summary.append(f"Total Return:         {metrics['total_return']*100:>8.2f}%")
        summary.append(f"CAGR:                 {metrics['cagr']*100:>8.2f}%")
        summary.append(f"Avg Monthly Return:   {metrics['avg_monthly_return']*100:>8.2f}%")
        summary.append(f"Best Month:           {metrics['best_month']*100:>8.2f}%")
        summary.append(f"Worst Month:          {metrics['worst_month']*100:>8.2f}%")
        summary.append(f"Positive Months:      {metrics['positive_months']}/{metrics['total_months']}")
        summary.append("")

        # Risk
        summary.append("RISK")
        summary.append("-" * 70)
        summary.append(f"Volatility:           {metrics['volatility']*100:>8.2f}%")
        summary.append(f"Max Drawdown:         {metrics['max_drawdown']*100:>8.2f}%")
        summary.append(f"Avg Drawdown:         {metrics['avg_drawdown']*100:>8.2f}%")
        summary.append(f"VaR (5%):             {metrics['value_at_risk_5']*100:>8.2f}%")
        summary.append("")

        # Risk-Adjusted
        summary.append("RISK-ADJUSTED RETURNS")
        summary.append("-" * 70)
        summary.append(f"Sharpe Ratio:         {metrics['sharpe_ratio']:>8.2f}")
        summary.append(f"Sortino Ratio:        {metrics['sortino_ratio']:>8.2f}")
        summary.append(f"Calmar Ratio:         {metrics['calmar_ratio']:>8.2f}")
        summary.append("")

        # Trading
        summary.append("TRADING STATISTICS")
        summary.append("-" * 70)
        summary.append(f"Total Trades:         {metrics['num_trades']:>8}")
        summary.append(f"Win Rate:             {metrics['win_rate']*100:>8.2f}%")
        summary.append(f"Avg Win:              {metrics['avg_win']*100:>8.2f}%")
        summary.append(f"Avg Loss:             {metrics['avg_loss']*100:>8.2f}%")
        summary.append(f"Profit Factor:        {metrics['profit_factor']:>8.2f}")
        summary.append(f"Expectancy:           {metrics['expectancy']*100:>8.2f}%")
        summary.append(f"Largest Win:          {metrics['largest_win']*100:>8.2f}%")
        summary.append(f"Largest Loss:         {metrics['largest_loss']*100:>8.2f}%")
        summary.append("")

        summary.append("=" * 70)

        return "\n".join(summary)

    def __repr__(self) -> str:
        return f"PerformanceMetrics(trades={len(self.trades)})"
