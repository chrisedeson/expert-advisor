"""
Profit Protection - Lock in gains during winning streaks

Problem: After big wins, traders often give it all back by:
- Getting overconfident
- Taking bigger risks
- Not securing profits

Solution: Automatically protect gains when on a winning streak.

How it works:
1. Detect strong profit run (>20% gain from recent low)
2. Reduce position size by 25% (protect capital)
3. Tighten take profit targets (exit faster)
4. Tighten stop losses (less giveback)
5. Suggest monthly profit withdrawals

Example:
Balance grows from $500 → $625 (+25% in 2 weeks)
  → Profit protection activates
  → Position size: 0.06 → 0.045 lots (25% smaller)
  → Take profit: 50 pips → 40 pips (tighter)
  → Stop loss: 30 pips → 25 pips (tighter)
  → Alert: "Consider withdrawing $62 (50% of profit)"

Result: Lock in $600+ instead of giving it back to $520
"""

from typing import Optional, Dict
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger


@dataclass
class ProfitRun:
    """Record of a profit run"""
    start_time: datetime
    start_balance: float
    peak_balance: float
    peak_time: datetime
    gain_pct: float


class ProfitProtector:
    """
    Protects profits during winning streaks.

    Features:
    - Detects strong profit runs
    - Reduces risk after big gains
    - Suggests profit withdrawals
    - Tightens exit parameters
    """

    def __init__(
        self,
        profit_threshold: float = 0.20,          # 20% gain triggers protection
        protection_size_reduction: float = 0.25,  # Reduce size by 25%
        tp_tightening_factor: float = 0.8,        # Tighter TP (80% of normal)
        sl_tightening_factor: float = 0.85,       # Tighter SL (85% of normal)
        lookback_days: int = 30,                  # Track gains over 30 days
        withdrawal_suggestion_pct: float = 0.50,  # Suggest withdrawing 50% of profit
    ):
        """
        Initialize profit protector.

        Args:
            profit_threshold: Gain % that triggers protection
            protection_size_reduction: How much to reduce position size
            tp_tightening_factor: Multiplier for take profit (< 1.0 = tighter)
            sl_tightening_factor: Multiplier for stop loss (< 1.0 = tighter)
            lookback_days: Days to track for profit runs
            withdrawal_suggestion_pct: % of profit to suggest withdrawing
        """
        self.profit_threshold = profit_threshold
        self.protection_size_reduction = protection_size_reduction
        self.tp_tightening_factor = tp_tightening_factor
        self.sl_tightening_factor = sl_tightening_factor
        self.lookback_days = lookback_days
        self.withdrawal_suggestion_pct = withdrawal_suggestion_pct

        # State
        self.protection_active = False
        self.current_run: Optional[ProfitRun] = None
        self.recent_low_balance: Optional[float] = None
        self.recent_low_time: Optional[datetime] = None

        # History
        self.profit_runs: list[ProfitRun] = []
        self.withdrawal_suggestions: list[Dict] = []

        logger.info(
            f"Initialized ProfitProtector: "
            f"Threshold={profit_threshold*100:.0f}%, "
            f"Size reduction={protection_size_reduction*100:.0f}%"
        )

    def check_and_update(
        self,
        current_balance: float,
        current_time: Optional[datetime] = None
    ) -> bool:
        """
        Check if profit protection should be activated/deactivated.

        Args:
            current_balance: Current account balance
            current_time: Current datetime

        Returns:
            True if protection is active
        """
        if current_time is None:
            current_time = datetime.now()

        # Track recent low (over lookback period)
        self._update_recent_low(current_balance, current_time)

        if self.recent_low_balance is None:
            return False

        # Calculate gain from recent low
        gain = current_balance - self.recent_low_balance
        gain_pct = gain / self.recent_low_balance if self.recent_low_balance > 0 else 0

        # Check if we should activate protection
        if not self.protection_active:
            if gain_pct >= self.profit_threshold:
                self._activate_protection(current_balance, gain_pct, current_time)
        else:
            # Already active - check if we should deactivate
            if gain_pct < (self.profit_threshold * 0.5):  # 50% retracement
                self._deactivate_protection(current_balance, gain_pct, current_time)

        # Update current run if active
        if self.protection_active and self.current_run:
            if current_balance > self.current_run.peak_balance:
                self.current_run.peak_balance = current_balance
                self.current_run.peak_time = current_time
                self.current_run.gain_pct = gain_pct

        return self.protection_active

    def _update_recent_low(self, current_balance: float, current_time: datetime):
        """Track recent low balance over lookback period"""

        # Initialize if first time
        if self.recent_low_balance is None:
            self.recent_low_balance = current_balance
            self.recent_low_time = current_time
            return

        # Update if new low
        if current_balance < self.recent_low_balance:
            self.recent_low_balance = current_balance
            self.recent_low_time = current_time

        # Reset if lookback period expired
        if self.recent_low_time:
            days_since_low = (current_time - self.recent_low_time).days
            if days_since_low > self.lookback_days:
                # Reset to current balance
                self.recent_low_balance = current_balance
                self.recent_low_time = current_time

    def _activate_protection(
        self,
        current_balance: float,
        gain_pct: float,
        current_time: datetime
    ):
        """Activate profit protection"""

        if self.recent_low_balance is None or self.recent_low_time is None:
            return

        self.protection_active = True

        # Create profit run record
        self.current_run = ProfitRun(
            start_time=self.recent_low_time,
            start_balance=self.recent_low_balance,
            peak_balance=current_balance,
            peak_time=current_time,
            gain_pct=gain_pct,
        )

        profit_amount = current_balance - self.recent_low_balance

        logger.success(
            f"🛡️ PROFIT PROTECTION ACTIVATED"
        )
        logger.success(
            f"   Gain: ${profit_amount:.2f} (+{gain_pct*100:.1f}%)"
        )
        logger.success(
            f"   From: ${self.recent_low_balance:.2f} → ${current_balance:.2f}"
        )
        logger.info(
            f"   Protection measures:"
        )
        logger.info(
            f"      • Position size reduced by {self.protection_size_reduction*100:.0f}%"
        )
        logger.info(
            f"      • Take profit {(1-self.tp_tightening_factor)*100:.0f}% tighter"
        )
        logger.info(
            f"      • Stop loss {(1-self.sl_tightening_factor)*100:.0f}% tighter"
        )

        # Generate withdrawal suggestion
        self._suggest_withdrawal(current_balance, profit_amount, current_time)

    def _deactivate_protection(
        self,
        current_balance: float,
        gain_pct: float,
        current_time: datetime
    ):
        """Deactivate profit protection"""

        logger.info(
            f"📊 Profit protection deactivated - Gain retraced to {gain_pct*100:.1f}%"
        )
        logger.info(
            f"   Back to normal position sizing"
        )

        # Save profit run to history
        if self.current_run:
            self.profit_runs.append(self.current_run)

        self.protection_active = False
        self.current_run = None

    def _suggest_withdrawal(
        self,
        current_balance: float,
        profit_amount: float,
        current_time: datetime
    ):
        """Suggest profit withdrawal"""

        withdrawal_amount = profit_amount * self.withdrawal_suggestion_pct

        suggestion = {
            'timestamp': current_time,
            'balance': current_balance,
            'profit': profit_amount,
            'suggested_withdrawal': withdrawal_amount,
            'keep_trading': current_balance - withdrawal_amount,
        }

        self.withdrawal_suggestions.append(suggestion)

        logger.info(
            f"💰 PROFIT WITHDRAWAL SUGGESTION"
        )
        logger.info(
            f"   Current balance: ${current_balance:.2f}"
        )
        logger.info(
            f"   Profit made: ${profit_amount:.2f}"
        )
        logger.success(
            f"   💡 Consider withdrawing: ${withdrawal_amount:.2f} "
            f"({self.withdrawal_suggestion_pct*100:.0f}% of profit)"
        )
        logger.info(
            f"   Keep trading with: ${current_balance - withdrawal_amount:.2f}"
        )
        logger.info(
            f"   This locks in gains while keeping capital working!"
        )

    def get_position_size_multiplier(self) -> float:
        """
        Get position size multiplier based on protection status.

        Returns:
            Multiplier (0.75 if protection active, 1.0 if not)
        """
        if self.protection_active:
            return 1.0 - self.protection_size_reduction
        return 1.0

    def get_tp_multiplier(self) -> float:
        """
        Get take profit multiplier.

        Returns:
            Multiplier (< 1.0 if protection active = tighter TP)
        """
        if self.protection_active:
            return self.tp_tightening_factor
        return 1.0

    def get_sl_multiplier(self) -> float:
        """
        Get stop loss multiplier.

        Returns:
            Multiplier (< 1.0 if protection active = tighter SL)
        """
        if self.protection_active:
            return self.sl_tightening_factor
        return 1.0

    def is_active(self) -> bool:
        """Check if profit protection is currently active"""
        return self.protection_active

    def get_status(self) -> Dict:
        """Get current profit protection status"""
        status = {
            'active': self.protection_active,
            'position_size_multiplier': self.get_position_size_multiplier(),
            'tp_multiplier': self.get_tp_multiplier(),
            'sl_multiplier': self.get_sl_multiplier(),
        }

        if self.protection_active and self.current_run:
            status['current_run'] = {
                'start_balance': self.current_run.start_balance,
                'peak_balance': self.current_run.peak_balance,
                'gain_pct': self.current_run.gain_pct * 100,
                'days_running': (datetime.now() - self.current_run.start_time).days,
            }

        if self.recent_low_balance:
            status['recent_low'] = self.recent_low_balance

        return status

    def get_statistics(self) -> Dict:
        """Get profit protection statistics"""
        total_runs = len(self.profit_runs)

        if total_runs > 0:
            avg_gain = sum(run.gain_pct for run in self.profit_runs) / total_runs
            max_gain = max(run.gain_pct for run in self.profit_runs)
            total_protected_profit = sum(
                run.peak_balance - run.start_balance
                for run in self.profit_runs
            )
        else:
            avg_gain = 0.0
            max_gain = 0.0
            total_protected_profit = 0.0

        return {
            'total_profit_runs': total_runs,
            'average_gain_pct': avg_gain * 100,
            'max_gain_pct': max_gain * 100,
            'total_protected_profit': total_protected_profit,
            'withdrawal_suggestions': len(self.withdrawal_suggestions),
            'currently_active': self.protection_active,
        }

    def reset(self):
        """Reset profit protector (for testing)"""
        self.protection_active = False
        self.current_run = None
        self.recent_low_balance = None
        self.recent_low_time = None
        self.profit_runs = []
        self.withdrawal_suggestions = []
        logger.info("ProfitProtector reset")
