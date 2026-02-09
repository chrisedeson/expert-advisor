"""
Circuit Breaker - Stop loss spirals before they become disasters

Prevents scenarios like:
- Bad day → bad week → catastrophic month
- One bad trade → revenge trading → account blown

Three layers of protection:
1. Daily limit (5% loss → pause 24 hours)
2. Weekly limit (10% loss → pause 7 days)
3. Monthly limit (15% loss → require manual restart)

This gives you time to breathe and assess, prevents emotional decisions.
"""

from enum import Enum
from typing import Optional, Dict
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    ACTIVE = "ACTIVE"           # Trading allowed
    DAILY_PAUSE = "DAILY_PAUSE"         # 24-hour pause
    WEEKLY_PAUSE = "WEEKLY_PAUSE"       # 7-day pause
    MONTHLY_STOP = "MONTHLY_STOP"       # Manual restart required


@dataclass
class LossEvent:
    """Record of a loss event that triggered circuit breaker"""
    timestamp: datetime
    loss_pct: float
    period: str  # 'daily', 'weekly', 'monthly'
    balance_before: float
    balance_after: float
    action_taken: str


class CircuitBreaker:
    """
    Automatic trading pauses based on loss thresholds.

    How it works:
    - Tracks daily/weekly/monthly P&L
    - If loss exceeds threshold → pause trading
    - Countdown timer for auto-resume
    - Manual override required for monthly stop

    Example scenario:
    Monday: -2% (continue)
    Tuesday: -3% (total -5% weekly, continue)
    Wednesday: -2% more (total -7% weekly, continue)
    Thursday: -4% more (total -11% weekly)
            → WEEKLY CIRCUIT BREAKER TRIGGERED
            → Trading paused for 7 days
            → Alert sent to user
            → Prevents turning -11% into -20%
    """

    def __init__(
        self,
        initial_balance: float,
        daily_limit: float = 0.05,      # 5% daily loss
        weekly_limit: float = 0.10,     # 10% weekly loss
        monthly_limit: float = 0.15,    # 15% monthly loss
        daily_pause_hours: int = 24,
        weekly_pause_days: int = 7,
    ):
        """
        Initialize circuit breaker.

        Args:
            initial_balance: Starting account balance
            daily_limit: Daily loss threshold (0.05 = 5%)
            weekly_limit: Weekly loss threshold (0.10 = 10%)
            monthly_limit: Monthly loss threshold (0.15 = 15%)
            daily_pause_hours: Hours to pause after daily limit
            weekly_pause_days: Days to pause after weekly limit
        """
        self.initial_balance = initial_balance
        self.daily_limit = daily_limit
        self.weekly_limit = weekly_limit
        self.monthly_limit = monthly_limit
        self.daily_pause_hours = daily_pause_hours
        self.weekly_pause_days = weekly_pause_days

        # Current state
        self.state = CircuitBreakerState.ACTIVE
        self.pause_until: Optional[datetime] = None

        # Period tracking (reset at intervals)
        self.period_start_balance = {
            'daily': initial_balance,
            'weekly': initial_balance,
            'monthly': initial_balance,
        }

        self.period_start_time = {
            'daily': datetime.now(),
            'weekly': datetime.now(),
            'monthly': datetime.now(),
        }

        # Track last known balance for period resets
        self._last_known_balance = initial_balance

        # History
        self.loss_events: list[LossEvent] = []
        self.manual_resume_required = False

        logger.info(
            f"Initialized CircuitBreaker: "
            f"Limits={daily_limit*100:.0f}%/{weekly_limit*100:.0f}%/{monthly_limit*100:.0f}%"
        )

    def check(
        self,
        current_balance: float,
        current_time: Optional[datetime] = None
    ) -> CircuitBreakerState:
        """
        Check if circuit breaker should trigger.

        Args:
            current_balance: Current account balance
            current_time: Current datetime (for testing, else uses now)

        Returns:
            Current CircuitBreakerState
        """
        if current_time is None:
            current_time = datetime.now()

        # Track balance for period resets
        self._last_known_balance = current_balance

        # Check if we're in a pause period
        if self.pause_until is not None:
            if current_time < self.pause_until:
                # Still paused
                return self.state
            else:
                # Pause period ended - auto-resume
                self._resume_from_pause(current_time)

        # Check if manual resume is required
        if self.manual_resume_required:
            return CircuitBreakerState.MONTHLY_STOP

        # Reset periods if needed
        self._check_period_resets(current_time)

        # Calculate period losses
        daily_loss_pct = self._calculate_period_loss(current_balance, 'daily')
        weekly_loss_pct = self._calculate_period_loss(current_balance, 'weekly')
        monthly_loss_pct = self._calculate_period_loss(current_balance, 'monthly')

        # Check thresholds (check most severe first)
        if monthly_loss_pct >= self.monthly_limit:
            self._trigger_monthly_stop(
                current_balance, monthly_loss_pct, current_time
            )
            return CircuitBreakerState.MONTHLY_STOP

        if weekly_loss_pct >= self.weekly_limit:
            self._trigger_weekly_pause(
                current_balance, weekly_loss_pct, current_time
            )
            return CircuitBreakerState.WEEKLY_PAUSE

        if daily_loss_pct >= self.daily_limit:
            self._trigger_daily_pause(
                current_balance, daily_loss_pct, current_time
            )
            return CircuitBreakerState.DAILY_PAUSE

        # All clear - trading allowed
        self.state = CircuitBreakerState.ACTIVE
        return self.state

    def _calculate_period_loss(self, current_balance: float, period: str) -> float:
        """Calculate loss percentage for a given period."""
        start_balance = self.period_start_balance[period]

        if start_balance <= 0:
            return 0.0

        loss_pct = (start_balance - current_balance) / start_balance

        # Only count losses (not gains)
        return max(0.0, loss_pct)

    def _check_period_resets(self, current_time: datetime):
        """Reset period tracking when new period begins."""

        # Daily reset (at midnight)
        daily_start = self.period_start_time['daily']
        if current_time.date() > daily_start.date():
            # New day - reset daily tracking AND balance
            self.period_start_time['daily'] = current_time.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            self.period_start_balance['daily'] = self._last_known_balance

        # Weekly reset (every 7 days)
        weekly_start = self.period_start_time['weekly']
        days_since_weekly_start = (current_time - weekly_start).days
        if days_since_weekly_start >= 7:
            self.period_start_time['weekly'] = current_time
            self.period_start_balance['weekly'] = self._last_known_balance

        # Monthly reset (on 1st of month)
        monthly_start = self.period_start_time['monthly']
        if current_time.month != monthly_start.month or current_time.year != monthly_start.year:
            self.period_start_time['monthly'] = current_time.replace(day=1, hour=0, minute=0, second=0)
            self.period_start_balance['monthly'] = self._last_known_balance

    def _trigger_daily_pause(
        self,
        current_balance: float,
        loss_pct: float,
        current_time: datetime
    ):
        """Trigger daily circuit breaker."""
        self.state = CircuitBreakerState.DAILY_PAUSE
        self.pause_until = current_time + timedelta(hours=self.daily_pause_hours)

        event = LossEvent(
            timestamp=current_time,
            loss_pct=loss_pct,
            period='daily',
            balance_before=self.period_start_balance['daily'],
            balance_after=current_balance,
            action_taken=f"Pause {self.daily_pause_hours}h"
        )
        self.loss_events.append(event)

        logger.warning(
            f"🔴 DAILY CIRCUIT BREAKER TRIGGERED - "
            f"Loss: {loss_pct*100:.2f}% (limit: {self.daily_limit*100:.0f}%)"
        )
        logger.warning(
            f"⏸️  Trading PAUSED for {self.daily_pause_hours} hours"
        )
        logger.warning(
            f"⏰ Resume at: {self.pause_until.strftime('%Y-%m-%d %H:%M')}"
        )
        logger.info(f"💡 This gives you time to assess what went wrong")

    def _trigger_weekly_pause(
        self,
        current_balance: float,
        loss_pct: float,
        current_time: datetime
    ):
        """Trigger weekly circuit breaker."""
        self.state = CircuitBreakerState.WEEKLY_PAUSE
        self.pause_until = current_time + timedelta(days=self.weekly_pause_days)

        event = LossEvent(
            timestamp=current_time,
            loss_pct=loss_pct,
            period='weekly',
            balance_before=self.period_start_balance['weekly'],
            balance_after=current_balance,
            action_taken=f"Pause {self.weekly_pause_days}d"
        )
        self.loss_events.append(event)

        logger.warning(
            f"🔴 WEEKLY CIRCUIT BREAKER TRIGGERED - "
            f"Loss: {loss_pct*100:.2f}% (limit: {self.weekly_limit*100:.0f}%)"
        )
        logger.warning(
            f"⏸️  Trading PAUSED for {self.weekly_pause_days} days"
        )
        logger.warning(
            f"⏰ Resume at: {self.pause_until.strftime('%Y-%m-%d')}"
        )
        logger.info(f"💡 Take a break, markets will be here next week")

    def _trigger_monthly_stop(
        self,
        current_balance: float,
        loss_pct: float,
        current_time: datetime
    ):
        """Trigger monthly circuit breaker (manual restart required)."""
        self.state = CircuitBreakerState.MONTHLY_STOP
        self.manual_resume_required = True

        event = LossEvent(
            timestamp=current_time,
            loss_pct=loss_pct,
            period='monthly',
            balance_before=self.period_start_balance['monthly'],
            balance_after=current_balance,
            action_taken="STOP - Manual restart required"
        )
        self.loss_events.append(event)

        logger.error(
            f"🛑 MONTHLY CIRCUIT BREAKER TRIGGERED - "
            f"Loss: {loss_pct*100:.2f}% (limit: {self.monthly_limit*100:.0f}%)"
        )
        logger.error(
            f"⛔ Trading STOPPED - Manual review REQUIRED"
        )
        logger.error(
            f"📧 Check your email for alert"
        )
        logger.info(
            f"💡 This is serious - take time to review what happened"
        )

    def _resume_from_pause(self, current_time: datetime):
        """Auto-resume after pause period ends."""
        old_state = self.state
        self.state = CircuitBreakerState.ACTIVE
        self.pause_until = None

        logger.info(
            f"✅ Circuit breaker pause ended - Resuming from {old_state.value}"
        )
        logger.info(f"✅ Trading ACTIVE at {current_time.strftime('%Y-%m-%d %H:%M')}")

    def manual_resume(self, current_balance: float, current_time: Optional[datetime] = None):
        """
        Manually resume trading after monthly stop.

        Args:
            current_balance: Current balance (resets period tracking)
            current_time: Current datetime
        """
        if current_time is None:
            current_time = datetime.now()

        if not self.manual_resume_required:
            logger.warning("Manual resume called but not required")
            return

        self.manual_resume_required = False
        self.state = CircuitBreakerState.ACTIVE
        self.pause_until = None

        # Reset all period tracking with current balance
        self.period_start_balance = {
            'daily': current_balance,
            'weekly': current_balance,
            'monthly': current_balance,
        }

        self.period_start_time = {
            'daily': current_time,
            'weekly': current_time,
            'monthly': current_time,
        }

        logger.info(
            f"✅ Manual resume completed - Trading ACTIVE"
        )
        logger.info(f"📊 Period tracking reset at balance: ${current_balance:.2f}")

    def update_period_balance(self, current_balance: float, current_time: Optional[datetime] = None):
        """
        Update period start balances (call at period boundaries).

        Args:
            current_balance: Current balance
            current_time: Current datetime
        """
        if current_time is None:
            current_time = datetime.now()

        # Update period start balances when periods reset
        self._check_period_resets(current_time)

        # Update balances for any newly reset periods
        for period in ['daily', 'weekly', 'monthly']:
            if self.period_start_time[period] == current_time.replace(
                hour=0, minute=0, second=0, microsecond=0
            ):
                self.period_start_balance[period] = current_balance

    def should_trade(self) -> bool:
        """Check if trading is allowed."""
        return self.state == CircuitBreakerState.ACTIVE

    def should_close_positions(self) -> bool:
        """Check if positions should be closed."""
        # Close positions when any circuit breaker triggers
        return self.state != CircuitBreakerState.ACTIVE

    def get_time_until_resume(self) -> Optional[timedelta]:
        """Get time remaining until auto-resume."""
        if self.pause_until is None:
            return None

        remaining = self.pause_until - datetime.now()
        return remaining if remaining.total_seconds() > 0 else None

    def get_status(self) -> Dict:
        """Get current circuit breaker status."""
        time_until_resume = self.get_time_until_resume()

        status = {
            'state': self.state.value,
            'can_trade': self.should_trade(),
            'manual_resume_required': self.manual_resume_required,
            'pause_until': self.pause_until.isoformat() if self.pause_until else None,
            'hours_until_resume': time_until_resume.total_seconds() / 3600 if time_until_resume else None,
            'total_loss_events': len(self.loss_events),
        }

        # Add period losses if we have balance data
        if hasattr(self, 'last_check_balance'):
            status.update({
                'daily_loss_pct': self._calculate_period_loss(self.last_check_balance, 'daily') * 100,
                'weekly_loss_pct': self._calculate_period_loss(self.last_check_balance, 'weekly') * 100,
                'monthly_loss_pct': self._calculate_period_loss(self.last_check_balance, 'monthly') * 100,
            })

        return status

    def get_statistics(self) -> Dict:
        """Get circuit breaker statistics."""
        if len(self.loss_events) == 0:
            return {
                'total_triggers': 0,
                'daily_triggers': 0,
                'weekly_triggers': 0,
                'monthly_triggers': 0,
            }

        daily_triggers = sum(1 for e in self.loss_events if e.period == 'daily')
        weekly_triggers = sum(1 for e in self.loss_events if e.period == 'weekly')
        monthly_triggers = sum(1 for e in self.loss_events if e.period == 'monthly')

        return {
            'total_triggers': len(self.loss_events),
            'daily_triggers': daily_triggers,
            'weekly_triggers': weekly_triggers,
            'monthly_triggers': monthly_triggers,
            'worst_daily_loss': max([e.loss_pct for e in self.loss_events if e.period == 'daily'], default=0) * 100,
            'worst_weekly_loss': max([e.loss_pct for e in self.loss_events if e.period == 'weekly'], default=0) * 100,
            'worst_monthly_loss': max([e.loss_pct for e in self.loss_events if e.period == 'monthly'], default=0) * 100,
        }

    def reset(self, initial_balance: float):
        """Reset circuit breaker (for testing)."""
        self.initial_balance = initial_balance
        self._last_known_balance = initial_balance
        self.state = CircuitBreakerState.ACTIVE
        self.pause_until = None

        self.period_start_balance = {
            'daily': initial_balance,
            'weekly': initial_balance,
            'monthly': initial_balance,
        }

        self.period_start_time = {
            'daily': datetime.now(),
            'weekly': datetime.now(),
            'monthly': datetime.now(),
        }

        self.loss_events = []
        self.manual_resume_required = False

        logger.info(f"CircuitBreaker reset with balance ${initial_balance:.2f}")
