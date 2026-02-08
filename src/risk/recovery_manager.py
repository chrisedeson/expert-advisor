"""
Recovery Mode - Gradual restart after drawdowns

Problem: After a big loss, jumping back in aggressively often leads to more losses.

Solution: Phase-based recovery with gradually increasing position sizes.

How it works:
1. Detect drawdown event (>10% loss)
2. Enter Phase 1: Trade at 30% normal size for 1-2 weeks
3. If profitable → Phase 2: Trade at 60% size for 1-2 weeks
4. If still profitable → Phase 3: Back to 100% size
5. If unprofitable in any phase → Stay in phase longer or restart

This prevents:
- Revenge trading (trying to recover quickly)
- Doubling down after losses
- Emotional decisions

Example:
Account hits 12% drawdown → Circuit breaker triggers
After 7-day pause:
  Week 1-2 (Phase 1): Trade 0.003 lots instead of 0.01
                      Make +$2-3 slowly, rebuild confidence
  Week 3-4 (Phase 2): Trade 0.006 lots
                      Make +$5-8, getting back on track
  Week 5+ (Phase 3):  Trade 0.01 lots (full size)
                      Normal operations resumed

This gives the strategy and you time to verify it's working again.
"""

from enum import Enum
from typing import Optional, Dict
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger


class RecoveryPhase(Enum):
    """Recovery phase states"""
    NORMAL = "NORMAL"           # No recovery needed (0% - 10% DD)
    PHASE_1 = "PHASE_1"        # 30% size (after >10% DD)
    PHASE_2 = "PHASE_2"        # 60% size (after Phase 1 success)
    PHASE_3 = "PHASE_3"        # 100% size (final recovery)


@dataclass
class PhaseConfig:
    """Configuration for a recovery phase"""
    position_size_multiplier: float
    minimum_duration_days: int
    profit_threshold_pct: float  # Required profit to advance


class RecoveryManager:
    """
    Manages gradual recovery after significant drawdowns.

    Prevents aggressive re-entry after losses by using phased approach.
    """

    def __init__(
        self,
        drawdown_threshold: float = 0.10,  # 10% DD triggers recovery
        phase_configs: Optional[Dict[RecoveryPhase, PhaseConfig]] = None,
    ):
        """
        Initialize recovery manager.

        Args:
            drawdown_threshold: Drawdown % that triggers recovery mode
            phase_configs: Custom phase configurations (or use defaults)
        """
        self.drawdown_threshold = drawdown_threshold

        # Default phase configurations
        if phase_configs is None:
            self.phase_configs = {
                RecoveryPhase.PHASE_1: PhaseConfig(
                    position_size_multiplier=0.3,
                    minimum_duration_days=7,     # At least 1 week
                    profit_threshold_pct=2.0,     # Need 2% profit to advance
                ),
                RecoveryPhase.PHASE_2: PhaseConfig(
                    position_size_multiplier=0.6,
                    minimum_duration_days=7,     # At least 1 week
                    profit_threshold_pct=3.0,     # Need 3% profit to advance
                ),
                RecoveryPhase.PHASE_3: PhaseConfig(
                    position_size_multiplier=1.0,
                    minimum_duration_days=0,      # No minimum, this is full recovery
                    profit_threshold_pct=0.0,     # No threshold needed
                ),
            }
        else:
            self.phase_configs = phase_configs

        # Current state
        self.current_phase = RecoveryPhase.NORMAL
        self.phase_start_time: Optional[datetime] = None
        self.phase_start_balance: Optional[float] = None
        self.recovery_trigger_balance: Optional[float] = None
        self.peak_balance_before_dd: Optional[float] = None

        # History
        self.phase_history = []

        logger.info(
            f"Initialized RecoveryManager: "
            f"DD threshold={drawdown_threshold*100:.0f}%, "
            f"Phases=3"
        )

    def check_and_update(
        self,
        current_balance: float,
        peak_balance: float,
        current_time: Optional[datetime] = None
    ) -> RecoveryPhase:
        """
        Check if recovery mode should be entered/advanced.

        Args:
            current_balance: Current account balance
            peak_balance: Historical peak balance
            current_time: Current datetime

        Returns:
            Current recovery phase
        """
        if current_time is None:
            current_time = datetime.now()

        # Calculate current drawdown
        if peak_balance > 0:
            drawdown = (peak_balance - current_balance) / peak_balance
        else:
            drawdown = 0.0

        # Check if we should enter recovery mode
        if self.current_phase == RecoveryPhase.NORMAL:
            if drawdown >= self.drawdown_threshold:
                self._enter_recovery_mode(
                    current_balance, peak_balance, drawdown, current_time
                )
                return self.current_phase
            else:
                # Normal operation
                return RecoveryPhase.NORMAL

        # We're in recovery mode - check if we should advance phases
        if self.current_phase in [RecoveryPhase.PHASE_1, RecoveryPhase.PHASE_2]:
            should_advance, reason = self._should_advance_phase(
                current_balance, current_time
            )

            if should_advance:
                self._advance_phase(current_balance, current_time, reason)

        # Check if fully recovered
        if self.current_phase == RecoveryPhase.PHASE_3:
            if self._is_fully_recovered(current_balance):
                self._exit_recovery_mode(current_balance, current_time)

        return self.current_phase

    def _enter_recovery_mode(
        self,
        current_balance: float,
        peak_balance: float,
        drawdown: float,
        current_time: datetime
    ):
        """Enter recovery mode (Phase 1)"""
        self.current_phase = RecoveryPhase.PHASE_1
        self.phase_start_time = current_time
        self.phase_start_balance = current_balance
        self.recovery_trigger_balance = current_balance
        self.peak_balance_before_dd = peak_balance

        self.phase_history.append({
            'phase': RecoveryPhase.PHASE_1,
            'start_time': current_time,
            'start_balance': current_balance,
            'trigger_drawdown': drawdown,
        })

        logger.warning(
            f"🔄 RECOVERY MODE ACTIVATED - Drawdown: {drawdown*100:.1f}%"
        )
        logger.warning(
            f"   Entering PHASE 1: Trading at 30% position size"
        )
        logger.warning(
            f"   Goal: Make 2% profit over next 7+ days"
        )
        logger.info(
            f"   Balance: ${current_balance:.2f} (down from ${peak_balance:.2f})"
        )

    def _should_advance_phase(
        self,
        current_balance: float,
        current_time: datetime
    ) -> tuple[bool, str]:
        """Check if we should advance to next phase"""

        if self.phase_start_time is None or self.phase_start_balance is None:
            return False, "Missing phase data"

        config = self.phase_configs[self.current_phase]

        # Check minimum duration
        days_in_phase = (current_time - self.phase_start_time).days
        if days_in_phase < config.minimum_duration_days:
            return False, f"Only {days_in_phase}/{config.minimum_duration_days} days in phase"

        # Check profit threshold
        profit_pct = ((current_balance - self.phase_start_balance) / self.phase_start_balance) * 100

        if profit_pct >= config.profit_threshold_pct:
            return True, f"Profit goal met: {profit_pct:.1f}% (target: {config.profit_threshold_pct:.1f}%)"
        else:
            return False, f"Profit {profit_pct:.1f}% below target {config.profit_threshold_pct:.1f}%"

    def _advance_phase(
        self,
        current_balance: float,
        current_time: datetime,
        reason: str
    ):
        """Advance to next recovery phase"""

        old_phase = self.current_phase

        if self.current_phase == RecoveryPhase.PHASE_1:
            self.current_phase = RecoveryPhase.PHASE_2
        elif self.current_phase == RecoveryPhase.PHASE_2:
            self.current_phase = RecoveryPhase.PHASE_3
        else:
            return  # Already in final phase

        self.phase_start_time = current_time
        self.phase_start_balance = current_balance

        self.phase_history.append({
            'phase': self.current_phase,
            'start_time': current_time,
            'start_balance': current_balance,
            'previous_phase': old_phase,
            'advancement_reason': reason,
        })

        config = self.phase_configs[self.current_phase]

        logger.success(
            f"✅ ADVANCING TO {self.current_phase.value}"
        )
        logger.success(f"   Reason: {reason}")
        logger.success(
            f"   New position size: {config.position_size_multiplier*100:.0f}%"
        )

        if self.current_phase == RecoveryPhase.PHASE_2:
            logger.info(
                f"   Goal: Make {config.profit_threshold_pct:.0f}% profit over "
                f"{config.minimum_duration_days}+ days"
            )
        elif self.current_phase == RecoveryPhase.PHASE_3:
            logger.info(
                f"   Final phase - full position size restored"
            )
            logger.info(
                f"   Continue until fully recovered to previous peak"
            )

    def _is_fully_recovered(self, current_balance: float) -> bool:
        """Check if account has fully recovered"""
        if self.peak_balance_before_dd is None:
            return True  # No peak recorded, consider recovered

        # Fully recovered = back to within 95% of previous peak
        recovery_pct = current_balance / self.peak_balance_before_dd

        return recovery_pct >= 0.95

    def _exit_recovery_mode(self, current_balance: float, current_time: datetime):
        """Exit recovery mode - back to normal"""

        days_in_recovery = 0
        if self.phase_history:
            first_phase = self.phase_history[0]
            days_in_recovery = (current_time - first_phase['start_time']).days

        logger.success(
            f"🎉 FULL RECOVERY COMPLETE!"
        )
        logger.success(
            f"   Balance: ${current_balance:.2f} (recovered to peak)"
        )
        logger.success(
            f"   Recovery duration: {days_in_recovery} days"
        )
        logger.success(
            f"   Back to NORMAL operations with 100% position sizing"
        )

        # Reset state
        self.current_phase = RecoveryPhase.NORMAL
        self.phase_start_time = None
        self.phase_start_balance = None
        self.recovery_trigger_balance = None
        self.peak_balance_before_dd = None

    def get_position_size_multiplier(self) -> float:
        """
        Get position size multiplier for current phase.

        Returns:
            Multiplier (0.3 for Phase 1, 0.6 for Phase 2, 1.0 for normal/Phase 3)
        """
        if self.current_phase == RecoveryPhase.NORMAL:
            return 1.0

        config = self.phase_configs.get(self.current_phase)
        if config:
            return config.position_size_multiplier

        return 1.0  # Fallback

    def is_in_recovery(self) -> bool:
        """Check if currently in recovery mode"""
        return self.current_phase != RecoveryPhase.NORMAL

    def get_status(self) -> Dict:
        """Get current recovery status"""
        status = {
            'phase': self.current_phase.value,
            'in_recovery': self.is_in_recovery(),
            'position_size_multiplier': self.get_position_size_multiplier(),
        }

        if self.is_in_recovery():
            if self.phase_start_time:
                days_in_phase = (datetime.now() - self.phase_start_time).days
                status['days_in_current_phase'] = days_in_phase

            if self.phase_start_balance:
                status['phase_start_balance'] = self.phase_start_balance

            if self.recovery_trigger_balance:
                status['recovery_trigger_balance'] = self.recovery_trigger_balance

            if self.peak_balance_before_dd:
                status['peak_before_drawdown'] = self.peak_balance_before_dd

            config = self.phase_configs.get(self.current_phase)
            if config:
                status['minimum_days'] = config.minimum_duration_days
                status['profit_threshold'] = config.profit_threshold_pct

        return status

    def get_statistics(self) -> Dict:
        """Get recovery mode statistics"""
        total_recoveries = sum(
            1 for h in self.phase_history
            if h['phase'] == RecoveryPhase.PHASE_1
        )

        successful_recoveries = 0
        if self.phase_history:
            # Count how many times we got to Phase 3 or exited recovery
            phase_3_count = sum(
                1 for h in self.phase_history
                if h['phase'] == RecoveryPhase.PHASE_3
            )
            successful_recoveries = phase_3_count

        return {
            'total_recovery_events': total_recoveries,
            'successful_recoveries': successful_recoveries,
            'currently_in_recovery': self.is_in_recovery(),
            'current_phase': self.current_phase.value,
            'phase_transitions': len(self.phase_history),
        }

    def reset(self):
        """Reset recovery manager (for testing)"""
        self.current_phase = RecoveryPhase.NORMAL
        self.phase_start_time = None
        self.phase_start_balance = None
        self.recovery_trigger_balance = None
        self.peak_balance_before_dd = None
        self.phase_history = []
        logger.info("RecoveryManager reset")
