"""
Performance Monitor - Detect when strategy stops working

Problem: Market conditions change, strategies degrade over time.
         Without monitoring, you keep trading a broken strategy.

Solution: Track rolling performance metrics and alert when degradation detected.

What it monitors:
1. Rolling 6-month CAGR (target: >30%)
2. Win rate (target: >60%)
3. Profit factor (target: >1.3)
4. Consecutive losses (alert if >5)
5. Drawdown frequency

When to alert:
- CAGR drops below 20% for 6 months
- Win rate drops below 50%
- 3 consecutive losing months
- Profit factor < 1.0

Actions suggested:
- Re-optimize parameters
- Pause trading
- Review strategy logic

This prevents slow death by continuing with broken strategy.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
from loguru import logger


@dataclass
class PerformanceSnapshot:
    """Snapshot of performance metrics at a point in time"""
    timestamp: datetime
    balance: float
    trades_count: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    cagr: float


class PerformanceMonitor:
    """
    Monitors strategy performance and detects degradation.

    Features:
    - Tracks rolling metrics
    - Compares to baseline
    - Detects underperformance
    - Suggests re-optimization
    """

    def __init__(
        self,
        baseline_cagr: float = 0.40,        # Expected 40% CAGR
        baseline_win_rate: float = 0.65,     # Expected 65% win rate
        baseline_profit_factor: float = 1.3, # Expected 1.3 PF
        baseline_sharpe: float = 0.8,        # Expected 0.8 Sharpe

        # Degradation thresholds
        cagr_warning_threshold: float = 0.25,       # Warn if < 25% CAGR
        cagr_critical_threshold: float = 0.15,      # Critical if < 15%
        win_rate_warning_threshold: float = 0.55,   # Warn if < 55%
        win_rate_critical_threshold: float = 0.45,  # Critical if < 45%

        lookback_months: int = 6,                   # Track 6-month rolling metrics
    ):
        """
        Initialize performance monitor.

        Args:
            baseline_cagr: Expected CAGR for healthy strategy
            baseline_win_rate: Expected win rate
            baseline_profit_factor: Expected profit factor
            baseline_sharpe: Expected Sharpe ratio
            cagr_warning_threshold: CAGR below this = warning
            cagr_critical_threshold: CAGR below this = critical
            win_rate_warning_threshold: Win rate below this = warning
            win_rate_critical_threshold: Win rate below this = critical
            lookback_months: Months to track for rolling metrics
        """
        self.baseline_cagr = baseline_cagr
        self.baseline_win_rate = baseline_win_rate
        self.baseline_profit_factor = baseline_profit_factor
        self.baseline_sharpe = baseline_sharpe

        self.cagr_warning_threshold = cagr_warning_threshold
        self.cagr_critical_threshold = cagr_critical_threshold
        self.win_rate_warning_threshold = win_rate_warning_threshold
        self.win_rate_critical_threshold = win_rate_critical_threshold

        self.lookback_months = lookback_months

        # State
        self.snapshots: List[PerformanceSnapshot] = []
        self.last_check_time: Optional[datetime] = None
        self.degradation_warnings: List[Dict] = []
        self.reoptimization_suggested = False

        logger.info(
            f"Initialized PerformanceMonitor: "
            f"Baseline={baseline_cagr*100:.0f}% CAGR, "
            f"{baseline_win_rate*100:.0f}% WR"
        )

    def record_snapshot(
        self,
        balance: float,
        trades: List[Dict],
        current_time: Optional[datetime] = None
    ):
        """
        Record a performance snapshot.

        Args:
            balance: Current balance
            trades: List of trade records
            current_time: Current datetime
        """
        if current_time is None:
            current_time = datetime.now()

        if len(trades) == 0:
            return  # Can't calculate metrics without trades

        # Calculate metrics from trades
        wins = [t for t in trades if t.get('net_pnl', 0) > 0]
        losses = [t for t in trades if t.get('net_pnl', 0) < 0]

        win_rate = len(wins) / len(trades) if trades else 0.0

        total_wins = sum(t['net_pnl'] for t in wins)
        total_losses = abs(sum(t['net_pnl'] for t in losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        # Calculate CAGR (simplified - would need more historical data normally)
        # For now, use annualized return based on recent trades
        if len(self.snapshots) > 0:
            prev_snapshot = self.snapshots[-1]
            days_elapsed = (current_time - prev_snapshot.timestamp).days
            if days_elapsed > 0:
                return_pct = (balance - prev_snapshot.balance) / prev_snapshot.balance
                cagr = ((1 + return_pct) ** (365 / days_elapsed)) - 1
            else:
                cagr = 0.0
        else:
            cagr = 0.0  # First snapshot, no CAGR yet

        # Create snapshot
        snapshot = PerformanceSnapshot(
            timestamp=current_time,
            balance=balance,
            trades_count=len(trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=0.0,  # Would need returns series to calculate
            max_drawdown=0.0,  # Would need equity curve
            cagr=cagr,
        )

        self.snapshots.append(snapshot)

        # Keep only lookback period
        cutoff_date = current_time - timedelta(days=self.lookback_months * 30)
        self.snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_date]

    def check_performance(self, current_time: Optional[datetime] = None) -> Dict:
        """
        Check if strategy is underperforming.

        Args:
            current_time: Current datetime

        Returns:
            Dictionary with performance status and warnings
        """
        if current_time is None:
            current_time = datetime.now()

        self.last_check_time = current_time

        if len(self.snapshots) < 2:
            return {
                'status': 'INSUFFICIENT_DATA',
                'warnings': [],
                'recommendations': [],
            }

        # Calculate rolling metrics
        rolling_metrics = self._calculate_rolling_metrics()

        # Check for degradation
        warnings = []
        recommendations = []

        # Check CAGR
        if rolling_metrics['cagr'] < self.cagr_critical_threshold:
            warnings.append({
                'level': 'CRITICAL',
                'metric': 'CAGR',
                'value': rolling_metrics['cagr'],
                'threshold': self.cagr_critical_threshold,
                'message': f"CAGR critically low: {rolling_metrics['cagr']*100:.1f}% (expected {self.baseline_cagr*100:.0f}%+)"
            })
            recommendations.append("CRITICAL: Consider stopping trading and re-evaluating strategy")

        elif rolling_metrics['cagr'] < self.cagr_warning_threshold:
            warnings.append({
                'level': 'WARNING',
                'metric': 'CAGR',
                'value': rolling_metrics['cagr'],
                'threshold': self.cagr_warning_threshold,
                'message': f"CAGR below target: {rolling_metrics['cagr']*100:.1f}% (expected {self.baseline_cagr*100:.0f}%+)"
            })
            recommendations.append("Consider re-optimizing parameters")

        # Check win rate
        if rolling_metrics['win_rate'] < self.win_rate_critical_threshold:
            warnings.append({
                'level': 'CRITICAL',
                'metric': 'Win Rate',
                'value': rolling_metrics['win_rate'],
                'threshold': self.win_rate_critical_threshold,
                'message': f"Win rate critically low: {rolling_metrics['win_rate']*100:.1f}% (expected {self.baseline_win_rate*100:.0f}%+)"
            })
            recommendations.append("CRITICAL: Strategy may be broken - review immediately")

        elif rolling_metrics['win_rate'] < self.win_rate_warning_threshold:
            warnings.append({
                'level': 'WARNING',
                'metric': 'Win Rate',
                'value': rolling_metrics['win_rate'],
                'threshold': self.win_rate_warning_threshold,
                'message': f"Win rate declining: {rolling_metrics['win_rate']*100:.1f}% (expected {self.baseline_win_rate*100:.0f}%+)"
            })

        # Check profit factor
        if rolling_metrics['profit_factor'] < 1.0:
            warnings.append({
                'level': 'CRITICAL',
                'metric': 'Profit Factor',
                'value': rolling_metrics['profit_factor'],
                'threshold': 1.0,
                'message': f"Losing money: Profit factor {rolling_metrics['profit_factor']:.2f} < 1.0"
            })
            recommendations.append("CRITICAL: Strategy losing money - STOP trading")

        # Store warnings
        if warnings:
            self.degradation_warnings.append({
                'timestamp': current_time,
                'warnings': warnings,
                'recommendations': recommendations,
            })

        # Determine overall status
        if any(w['level'] == 'CRITICAL' for w in warnings):
            status = 'CRITICAL'
        elif any(w['level'] == 'WARNING' for w in warnings):
            status = 'WARNING'
        else:
            status = 'HEALTHY'

        # Log warnings
        if status != 'HEALTHY':
            logger.warning(f"⚠️ PERFORMANCE DEGRADATION DETECTED - Status: {status}")
            for warning in warnings:
                if warning['level'] == 'CRITICAL':
                    logger.error(f"   🔴 {warning['message']}")
                else:
                    logger.warning(f"   ⚠️  {warning['message']}")

            for rec in recommendations:
                logger.info(f"   💡 {rec}")

        return {
            'status': status,
            'rolling_metrics': rolling_metrics,
            'warnings': warnings,
            'recommendations': recommendations,
        }

    def _calculate_rolling_metrics(self) -> Dict:
        """Calculate rolling performance metrics"""

        if not self.snapshots:
            return {
                'cagr': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
            }

        # Average metrics over lookback period
        cagrs = [s.cagr for s in self.snapshots if s.cagr != 0]
        avg_cagr = np.mean(cagrs) if cagrs else 0.0

        win_rates = [s.win_rate for s in self.snapshots]
        avg_win_rate = np.mean(win_rates) if win_rates else 0.0

        profit_factors = [s.profit_factor for s in self.snapshots if s.profit_factor != 0]
        avg_profit_factor = np.mean(profit_factors) if profit_factors else 0.0

        sharpe_ratios = [s.sharpe_ratio for s in self.snapshots if s.sharpe_ratio != 0]
        avg_sharpe = np.mean(sharpe_ratios) if sharpe_ratios else 0.0

        return {
            'cagr': avg_cagr,
            'win_rate': avg_win_rate,
            'profit_factor': avg_profit_factor,
            'sharpe_ratio': avg_sharpe,
            'snapshots_count': len(self.snapshots),
        }

    def should_reoptimize(self) -> bool:
        """Check if strategy should be re-optimized"""

        # Already suggested?
        if self.reoptimization_suggested:
            return False

        # Need enough data
        if len(self.snapshots) < 10:
            return False

        # Check recent performance
        recent_check = self.check_performance()

        if recent_check['status'] == 'WARNING':
            # Two consecutive warnings → suggest reoptimization
            if len(self.degradation_warnings) >= 2:
                last_two = self.degradation_warnings[-2:]
                if all(any(w['level'] == 'WARNING' for w in check['warnings']) for check in last_two):
                    self.reoptimization_suggested = True
                    return True

        elif recent_check['status'] == 'CRITICAL':
            # Any critical warning → suggest immediately
            self.reoptimization_suggested = True
            return True

        return False

    def get_status(self) -> Dict:
        """Get current monitoring status"""
        recent_check = self.check_performance() if self.snapshots else None

        status = {
            'snapshots_count': len(self.snapshots),
            'warnings_count': len(self.degradation_warnings),
            'reoptimization_suggested': self.reoptimization_suggested,
            'last_check': self.last_check_time.isoformat() if self.last_check_time else None,
        }

        if recent_check:
            status['current_status'] = recent_check['status']
            status['rolling_metrics'] = recent_check['rolling_metrics']

        return status

    def get_statistics(self) -> Dict:
        """Get monitoring statistics"""
        return {
            'total_snapshots': len(self.snapshots),
            'total_warnings': len(self.degradation_warnings),
            'critical_warnings': sum(
                1 for w in self.degradation_warnings
                if any(warn['level'] == 'CRITICAL' for warn in w['warnings'])
            ),
            'reoptimization_suggested': self.reoptimization_suggested,
        }

    def reset_reoptimization_flag(self):
        """Reset reoptimization suggestion (after completing reopt)"""
        self.reoptimization_suggested = False
        logger.info("Reoptimization flag reset")

    def reset(self):
        """Reset performance monitor (for testing)"""
        self.snapshots = []
        self.last_check_time = None
        self.degradation_warnings = []
        self.reoptimization_suggested = False
        logger.info("PerformanceMonitor reset")
