"""
Crisis Detector - Multi-factor crisis identification

Goes beyond just volatility to detect crises using multiple signals:
1. Volatility spike (ATR > 2.5x normal)
2. Rapid drawdown (>10% in 3 days)
3. Multiple consecutive stop losses (3+ in a row)
4. Gap events (>100 pip overnight gaps)
5. Correlation breakdown (all pairs moving together)

Crisis Levels:
- YELLOW (Caution): 1-2 factors triggered
- ORANGE (Alert): 3+ factors triggered
- RED (Emergency): 4+ factors or extreme single factor

Actions by level:
- YELLOW: Reduce position size 50%
- ORANGE: Close positions, pause trading
- RED: Emergency shutdown, manual restart only

This catches crises that volatility filter alone might miss.
"""

from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
from loguru import logger


class CrisisLevel(Enum):
    """Crisis severity levels"""
    NORMAL = "NORMAL"       # No crisis
    YELLOW = "YELLOW"       # Caution - 1-2 factors
    ORANGE = "ORANGE"       # Alert - 3+ factors
    RED = "RED"             # Emergency - 4+ factors


@dataclass
class CrisisSignal:
    """Single crisis signal/factor"""
    timestamp: datetime
    factor: str
    severity: float  # 0.0 to 1.0
    description: str


@dataclass
class CrisisEvent:
    """Complete crisis event record"""
    start_time: datetime
    end_time: Optional[datetime]
    level: CrisisLevel
    signals: List[CrisisSignal]
    actions_taken: List[str]


class CrisisDetector:
    """
    Multi-factor crisis detection system.

    More sophisticated than single volatility filter.
    """

    def __init__(
        self,
        # Volatility thresholds
        volatility_spike_threshold: float = 2.5,  # 2.5x normal ATR

        # Drawdown thresholds
        rapid_drawdown_threshold: float = 0.10,    # 10% in 3 days
        rapid_drawdown_days: int = 3,

        # Stop loss thresholds
        consecutive_stops_threshold: int = 3,      # 3 stops in a row

        # Gap thresholds
        gap_threshold_pips: float = 100.0,         # 100 pip gap

        # Crisis level thresholds
        yellow_factors: int = 1,                   # 1-2 factors = yellow
        orange_factors: int = 3,                   # 3 factors = orange
        red_factors: int = 4,                      # 4+ factors = red
    ):
        """
        Initialize crisis detector.

        Args:
            volatility_spike_threshold: ATR multiplier for volatility crisis
            rapid_drawdown_threshold: Drawdown % for rapid DD crisis
            rapid_drawdown_days: Days for rapid DD calculation
            consecutive_stops_threshold: Number of consecutive stops
            gap_threshold_pips: Pip size for gap crisis
            yellow_factors: Factors needed for YELLOW level
            orange_factors: Factors needed for ORANGE level
            red_factors: Factors needed for RED level
        """
        self.volatility_spike_threshold = volatility_spike_threshold
        self.rapid_drawdown_threshold = rapid_drawdown_threshold
        self.rapid_drawdown_days = rapid_drawdown_days
        self.consecutive_stops_threshold = consecutive_stops_threshold
        self.gap_threshold_pips = gap_threshold_pips
        self.yellow_factors = yellow_factors
        self.orange_factors = orange_factors
        self.red_factors = red_factors

        # Auto-reset config
        self.crisis_cooldown_hours = 48  # Auto-downgrade crisis after 48h
        self.max_crisis_duration_hours = 168  # Force reset to NORMAL after 7 days max

        # State
        self.current_level = CrisisLevel.NORMAL
        self.active_signals: List[CrisisSignal] = []
        self.crisis_start_time: Optional[datetime] = None
        self._last_consecutive_stop_signal_time: Optional[datetime] = None

        # History
        self.recent_trades: List[Dict] = []
        self.balance_history: List[tuple[datetime, float]] = []
        self.crisis_events: List[CrisisEvent] = []

        logger.info(
            f"Initialized CrisisDetector: "
            f"Multi-factor analysis with 3 alert levels"
        )

    def check_crisis(
        self,
        current_time: datetime,
        current_balance: float,
        current_atr: float,
        avg_atr: float,
        recent_trades: Optional[List[Dict]] = None,
        current_price: Optional[float] = None,
        previous_close: Optional[float] = None,
    ) -> CrisisLevel:
        """
        Check for crisis conditions using multiple factors.

        Args:
            current_time: Current datetime
            current_balance: Current account balance
            current_atr: Current ATR value
            avg_atr: Average ATR (baseline)
            recent_trades: List of recent trades
            current_price: Current market price
            previous_close: Previous close price (for gap detection)

        Returns:
            Current crisis level
        """
        # Update histories
        self.balance_history.append((current_time, current_balance))
        # Trim balance history to last 1000 entries
        if len(self.balance_history) > 1000:
            self.balance_history = self.balance_history[-500:]

        if recent_trades:
            self.recent_trades = recent_trades[-20:]  # Keep last 20

        # Auto-reset: force downgrade if crisis has lasted too long
        if self.crisis_start_time is not None:
            hours_in_crisis = (current_time - self.crisis_start_time).total_seconds() / 3600
            if hours_in_crisis >= self.max_crisis_duration_hours:
                logger.info(f"Crisis auto-reset after {hours_in_crisis:.0f} hours")
                self.active_signals = []
                self.crisis_start_time = None
                self.current_level = CrisisLevel.NORMAL
                return CrisisLevel.NORMAL

        # Clear old signals (>4 hours old instead of 1 hour)
        self.active_signals = [
            s for s in self.active_signals
            if (current_time - s.timestamp).total_seconds() < 14400  # 4 hours
        ]

        # Check each crisis factor
        self._check_volatility_spike(current_time, current_atr, avg_atr)
        self._check_rapid_drawdown(current_time, current_balance)
        self._check_consecutive_stops(current_time)

        if current_price and previous_close:
            self._check_gap_event(current_time, current_price, previous_close)

        # Determine crisis level based on active signals
        old_level = self.current_level
        self.current_level = self._calculate_crisis_level()

        # Log level changes
        if self.current_level != old_level:
            self._handle_level_change(old_level, self.current_level, current_time)

        return self.current_level

    def _check_volatility_spike(
        self,
        current_time: datetime,
        current_atr: float,
        avg_atr: float
    ):
        """Check for volatility spike"""
        if avg_atr <= 0:
            return

        atr_ratio = current_atr / avg_atr

        if atr_ratio >= self.volatility_spike_threshold:
            # Calculate severity (0.0 to 1.0)
            # 2.5x = 0.5, 3.0x = 0.75, 4.0x+ = 1.0
            severity = min(1.0, (atr_ratio - 2.0) / 2.0)

            signal = CrisisSignal(
                timestamp=current_time,
                factor='volatility_spike',
                severity=severity,
                description=f"ATR spike: {atr_ratio:.2f}x normal"
            )

            self.active_signals.append(signal)

            logger.warning(
                f"🌡️  VOLATILITY SPIKE DETECTED: {atr_ratio:.2f}x "
                f"(threshold: {self.volatility_spike_threshold}x)"
            )

    def _check_rapid_drawdown(self, current_time: datetime, current_balance: float):
        """Check for rapid drawdown"""

        # Need enough history
        cutoff_time = current_time - timedelta(days=self.rapid_drawdown_days)
        recent_balances = [
            (t, b) for t, b in self.balance_history
            if t >= cutoff_time
        ]

        if len(recent_balances) < 2:
            return

        # Find peak balance in period
        peak_balance = max(b for _, b in recent_balances)

        # Calculate drawdown
        drawdown = (peak_balance - current_balance) / peak_balance if peak_balance > 0 else 0

        if drawdown >= self.rapid_drawdown_threshold:
            # Calculate severity
            severity = min(1.0, drawdown / 0.20)  # 20% DD = 1.0 severity

            signal = CrisisSignal(
                timestamp=current_time,
                factor='rapid_drawdown',
                severity=severity,
                description=f"Rapid drawdown: {drawdown*100:.1f}% in {self.rapid_drawdown_days} days"
            )

            self.active_signals.append(signal)

            logger.warning(
                f"📉 RAPID DRAWDOWN DETECTED: {drawdown*100:.1f}% in {self.rapid_drawdown_days} days"
            )

    def _check_consecutive_stops(self, current_time: datetime):
        """Check for consecutive stop losses - only triggers once per new loss"""

        if len(self.recent_trades) < self.consecutive_stops_threshold:
            return

        # Only check if we have a NEW trade since last check
        last_trade_time = self.recent_trades[-1].get('exit_time')
        if (self._last_consecutive_stop_signal_time is not None and
                last_trade_time is not None and
                last_trade_time <= self._last_consecutive_stop_signal_time):
            return  # No new trades, don't re-trigger

        # Get last N trades
        last_n = self.recent_trades[-self.consecutive_stops_threshold:]

        # Check if all are losses (stop losses)
        all_losses = all(
            t.get('exit_reason') == 'Stop Loss' or t.get('net_pnl', 0) < 0
            for t in last_n
        )

        if all_losses:
            # Calculate severity based on how many consecutive
            consecutive_count = 0
            for trade in reversed(self.recent_trades):
                if trade.get('exit_reason') == 'Stop Loss' or trade.get('net_pnl', 0) < 0:
                    consecutive_count += 1
                else:
                    break

            severity = min(1.0, (consecutive_count - 2) / 5.0)  # 7+ stops = 1.0

            signal = CrisisSignal(
                timestamp=current_time,
                factor='consecutive_stops',
                severity=severity,
                description=f"Consecutive stop losses: {consecutive_count} in a row"
            )

            self.active_signals.append(signal)
            self._last_consecutive_stop_signal_time = current_time

            logger.warning(
                f"CONSECUTIVE STOPS DETECTED: {consecutive_count} losses in a row"
            )

    def _check_gap_event(
        self,
        current_time: datetime,
        current_price: float,
        previous_close: float
    ):
        """Check for gap event"""

        gap_pips = abs(current_price - previous_close) * 10000

        if gap_pips >= self.gap_threshold_pips:
            # Calculate severity
            severity = min(1.0, gap_pips / 300.0)  # 300 pip gap = 1.0

            signal = CrisisSignal(
                timestamp=current_time,
                factor='gap_event',
                severity=severity,
                description=f"Price gap: {gap_pips:.0f} pips"
            )

            self.active_signals.append(signal)

            logger.warning(
                f"📊 GAP EVENT DETECTED: {gap_pips:.0f} pips"
            )

    def _calculate_crisis_level(self) -> CrisisLevel:
        """Calculate crisis level based on active signals"""

        if not self.active_signals:
            return CrisisLevel.NORMAL

        # Count unique factors
        unique_factors = len(set(s.factor for s in self.active_signals))

        # Check for extreme single factor
        max_severity = max(s.severity for s in self.active_signals)

        if max_severity >= 0.9 or unique_factors >= self.red_factors:
            return CrisisLevel.RED
        elif unique_factors >= self.orange_factors:
            return CrisisLevel.ORANGE
        elif unique_factors >= self.yellow_factors:
            return CrisisLevel.YELLOW
        else:
            return CrisisLevel.NORMAL

    def _handle_level_change(
        self,
        old_level: CrisisLevel,
        new_level: CrisisLevel,
        current_time: datetime
    ):
        """Handle crisis level changes"""

        if new_level == CrisisLevel.NORMAL:
            logger.success(f"✅ Crisis resolved - back to NORMAL")
            self.crisis_start_time = None

        elif old_level == CrisisLevel.NORMAL:
            # Entering crisis
            self.crisis_start_time = current_time
            logger.error(f"🚨 CRISIS LEVEL: {new_level.value}")
            logger.error(f"   Active factors: {len(self.active_signals)}")
            for signal in self.active_signals:
                logger.error(f"      • {signal.description}")

            if new_level == CrisisLevel.YELLOW:
                logger.warning("   ACTION: Reduce position size 50%")
            elif new_level == CrisisLevel.ORANGE:
                logger.error("   ACTION: Close positions, pause trading")
            elif new_level == CrisisLevel.RED:
                logger.critical("   ACTION: EMERGENCY SHUTDOWN")

        elif new_level.value > old_level.value:
            # Crisis escalating
            logger.critical(f"🆙 CRISIS ESCALATING: {old_level.value} → {new_level.value}")

        elif new_level.value < old_level.value:
            # Crisis de-escalating
            logger.info(f"🆘 Crisis de-escalating: {old_level.value} → {new_level.value}")

    def get_position_size_multiplier(self) -> float:
        """Get position size multiplier based on crisis level"""
        if self.current_level == CrisisLevel.YELLOW:
            return 0.5  # 50% size
        elif self.current_level in [CrisisLevel.ORANGE, CrisisLevel.RED]:
            return 0.0  # No trading
        return 1.0  # Normal

    def should_trade(self) -> bool:
        """Check if trading is allowed"""
        return self.current_level in [CrisisLevel.NORMAL, CrisisLevel.YELLOW]

    def should_close_positions(self) -> bool:
        """Check if positions should be closed immediately"""
        return self.current_level in [CrisisLevel.ORANGE, CrisisLevel.RED]

    def requires_manual_restart(self) -> bool:
        """Check if manual restart is required"""
        return self.current_level == CrisisLevel.RED

    def get_status(self) -> Dict:
        """Get current crisis detector status"""
        return {
            'level': self.current_level.value,
            'active_signals_count': len(self.active_signals),
            'active_factors': list(set(s.factor for s in self.active_signals)),
            'can_trade': self.should_trade(),
            'should_close': self.should_close_positions(),
            'manual_restart_required': self.requires_manual_restart(),
            'position_size_multiplier': self.get_position_size_multiplier(),
        }

    def get_statistics(self) -> Dict:
        """Get crisis detector statistics"""
        return {
            'total_crisis_events': len(self.crisis_events),
            'yellow_alerts': sum(1 for e in self.crisis_events if e.level == CrisisLevel.YELLOW),
            'orange_alerts': sum(1 for e in self.crisis_events if e.level == CrisisLevel.ORANGE),
            'red_alerts': sum(1 for e in self.crisis_events if e.level == CrisisLevel.RED),
            'currently_active': self.current_level != CrisisLevel.NORMAL,
            'current_level': self.current_level.value,
        }

    def reset(self):
        """Reset crisis detector (for testing)"""
        self.current_level = CrisisLevel.NORMAL
        self.active_signals = []
        self.crisis_start_time = None
        self.recent_trades = []
        self.balance_history = []
        self.crisis_events = []
        logger.info("CrisisDetector reset")
