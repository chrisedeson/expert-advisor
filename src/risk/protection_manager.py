"""
Protection Manager - The Orchestrator

Coordinates all 8 protection systems to work together seamlessly.

This is the "brain" that:
1. Initializes all protection layers
2. Coordinates their interactions
3. Resolves conflicts (e.g., multiple systems want different actions)
4. Makes final decisions on position sizing and trading permissions
5. Manages state and logging

Single point of contact for the trading engine - ask the manager,
it consults all systems and gives you the answer.

Example flow:
    Trading Engine: "Can I trade? What size?"
    Protection Manager:
        → Asks Volatility Filter: NORMAL (OK)
        → Asks Circuit Breaker: ACTIVE (OK)
        → Asks Crisis Detector: NORMAL (OK)
        → Asks Recovery Manager: PHASE_2 (60% size)
        → Asks Profit Protector: ACTIVE (75% size)
        → Calculates: min(1.0, 1.0, 1.0, 0.6, 0.75) = 0.6
        → Returns: "Yes, trade at 60% size"
"""

from typing import Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass
import pandas as pd
from loguru import logger

from .volatility_filter import VolatilityFilter, MarketCondition
from .circuit_breaker import CircuitBreaker, CircuitBreakerState
from .recovery_manager import RecoveryManager, RecoveryPhase
from .profit_protector import ProfitProtector
from .crisis_detector import CrisisDetector, CrisisLevel
from ..notifications import AlertManager


@dataclass
class ProtectionDecision:
    """Decision from protection manager"""
    can_trade: bool
    position_size_multiplier: float
    should_close_positions: bool
    requires_manual_action: bool
    active_protections: List[str]
    reasoning: str


class ProtectionManager:
    """
    Orchestrates all protection systems.

    The single source of truth for trading permissions and risk management.
    """

    def __init__(
        self,
        initial_balance: float,
        config: Optional[Dict] = None,
        alert_manager: Optional[AlertManager] = None,
    ):
        """
        Initialize protection manager.

        Args:
            initial_balance: Starting account balance
            config: Configuration dictionary (from config.yaml)
            alert_manager: Alert manager instance
        """
        self.initial_balance = initial_balance
        self.config = config or {}
        self.alert_manager = alert_manager

        # Initialize all protection systems

        # Layer 1: Real-time Monitoring
        vol_config = self.config.get('volatility_filter', {})
        self.volatility_filter = VolatilityFilter(
            atr_period=vol_config.get('atr_period', 14),
            avg_period=vol_config.get('avg_period', 50),
            normal_threshold=vol_config.get('normal_threshold', 1.5),
            crisis_threshold=vol_config.get('crisis_threshold', 2.5),
            crisis_cooldown_days=vol_config.get('cooldown_days', 3),
        )

        crisis_config = self.config.get('crisis_detector', {})
        self.crisis_detector = CrisisDetector(
            volatility_spike_threshold=crisis_config.get('volatility_spike_threshold', 2.5),
            rapid_drawdown_threshold=crisis_config.get('rapid_drawdown_threshold', 0.15),
            rapid_drawdown_days=crisis_config.get('rapid_drawdown_days', 3),
            consecutive_stops_threshold=crisis_config.get('consecutive_stops_threshold', 4),
            gap_threshold_pips=crisis_config.get('gap_threshold_pips', 100.0),
        )

        # Layer 2: Loss Protection
        cb_config = self.config.get('circuit_breaker', {})
        self.circuit_breaker = CircuitBreaker(
            initial_balance=initial_balance,
            daily_limit=cb_config.get('daily_limit', 0.08),
            weekly_limit=cb_config.get('weekly_limit', 0.15),
            monthly_limit=cb_config.get('monthly_limit', 0.20),
        )

        recovery_config = self.config.get('recovery_manager', {})
        self.recovery_manager = RecoveryManager(
            drawdown_threshold=recovery_config.get('drawdown_threshold', 0.12),
        )

        # Layer 3: Profit Protection
        profit_config = self.config.get('profit_protector', {})
        self.profit_protector = ProfitProtector(
            profit_threshold=profit_config.get('profit_threshold', 0.20),
        )

        # Track state
        self.peak_balance = initial_balance
        self.last_check_time = datetime.now()
        self.decision_history: List[ProtectionDecision] = []

        # State change tracking (only log when state changes)
        self._prev_protection_set: frozenset = frozenset()
        self._prev_can_trade: bool = True

        logger.info("All protection systems initialized")

    def check_trading_permission(
        self,
        current_balance: float,
        current_data: pd.DataFrame,
        recent_trades: Optional[List[Dict]] = None,
        current_time: Optional[datetime] = None,
    ) -> ProtectionDecision:
        """
        Check if trading is allowed and at what size.

        This is THE method the trading engine calls.

        Args:
            current_balance: Current account balance
            current_data: DataFrame with recent OHLCV + ATR data
            recent_trades: List of recent trades (for crisis detector)
            current_time: Current datetime

        Returns:
            ProtectionDecision with all necessary information
        """
        if current_time is None:
            current_time = datetime.now()

        self.last_check_time = current_time

        # Update peak balance
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance

        # Collect input data
        current_atr = current_data['atr'].iloc[-1] if 'atr' in current_data.columns else 0.001
        avg_atr = current_data['atr'].rolling(50).mean().iloc[-1] if len(current_data) >= 50 else current_atr

        # === CHECK ALL PROTECTION SYSTEMS ===

        # 1. Volatility Filter
        market_condition = self.volatility_filter.check_market_condition(
            current_data, current_time
        )
        vol_can_trade = self.volatility_filter.should_trade()
        vol_multiplier = self.volatility_filter.get_position_size_multiplier()
        vol_should_close = self.volatility_filter.should_close_positions()

        # 2. Crisis Detector
        crisis_level = self.crisis_detector.check_crisis(
            current_time=current_time,
            current_balance=current_balance,
            current_atr=current_atr,
            avg_atr=avg_atr,
            recent_trades=recent_trades,
        )
        crisis_can_trade = self.crisis_detector.should_trade()
        crisis_multiplier = self.crisis_detector.get_position_size_multiplier()
        crisis_should_close = self.crisis_detector.should_close_positions()

        # 3. Circuit Breaker
        cb_state = self.circuit_breaker.check(current_balance, current_time)
        cb_can_trade = self.circuit_breaker.should_trade()
        cb_should_close = self.circuit_breaker.should_close_positions()

        # 4. Recovery Manager
        recovery_phase = self.recovery_manager.check_and_update(
            current_balance, self.peak_balance, current_time
        )
        recovery_multiplier = self.recovery_manager.get_position_size_multiplier()

        # 5. Profit Protector
        profit_protection_active = self.profit_protector.check_and_update(
            current_balance, current_time
        )
        profit_multiplier = self.profit_protector.get_position_size_multiplier()

        # === MAKE DECISION ===

        # Can trade? (ALL must agree)
        can_trade = vol_can_trade and crisis_can_trade and cb_can_trade

        # Should close? (ANY saying yes means close)
        should_close = vol_should_close or crisis_should_close or cb_should_close

        # Position size multiplier (use MOST CONSERVATIVE)
        position_size_multiplier = min(
            vol_multiplier,
            crisis_multiplier,
            recovery_multiplier,
            profit_multiplier,
        )

        # Requires manual action?
        requires_manual = (
            cb_state == CircuitBreakerState.MONTHLY_STOP or
            self.crisis_detector.requires_manual_restart()
        )

        # Active protections (for logging/display)
        active_protections = []
        if market_condition != MarketCondition.NORMAL:
            active_protections.append(f"Volatility: {market_condition.value}")
        if crisis_level != CrisisLevel.NORMAL:
            active_protections.append(f"Crisis: {crisis_level.value}")
        if cb_state != CircuitBreakerState.ACTIVE:
            active_protections.append(f"Circuit Breaker: {cb_state.value}")
        if recovery_phase != RecoveryPhase.NORMAL:
            active_protections.append(f"Recovery: {recovery_phase.value}")
        if profit_protection_active:
            active_protections.append("Profit Protection")

        # Build reasoning
        reasoning_parts = []
        if not can_trade:
            reasons = []
            if not vol_can_trade:
                reasons.append("Volatility crisis")
            if not crisis_can_trade:
                reasons.append("Multi-factor crisis")
            if not cb_can_trade:
                reasons.append("Circuit breaker pause")
            reasoning_parts.append(f"TRADING PAUSED: {', '.join(reasons)}")

        if position_size_multiplier < 1.0:
            factors = []
            if vol_multiplier < 1.0:
                factors.append(f"Volatility ({vol_multiplier*100:.0f}%)")
            if crisis_multiplier < 1.0:
                factors.append(f"Crisis ({crisis_multiplier*100:.0f}%)")
            if recovery_multiplier < 1.0:
                factors.append(f"Recovery ({recovery_multiplier*100:.0f}%)")
            if profit_multiplier < 1.0:
                factors.append(f"Profit protection ({profit_multiplier*100:.0f}%)")
            reasoning_parts.append(f"Position size reduced by {factors}")

        if should_close:
            reasoning_parts.append("Immediate position closure recommended")

        reasoning = " | ".join(reasoning_parts) if reasoning_parts else "All systems normal"

        # Create decision
        decision = ProtectionDecision(
            can_trade=can_trade,
            position_size_multiplier=position_size_multiplier,
            should_close_positions=should_close,
            requires_manual_action=requires_manual,
            active_protections=active_protections,
            reasoning=reasoning,
        )

        # Store in history
        self.decision_history.append(decision)

        # Keep only recent history
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-500:]

        # Log only on state CHANGES (not every bar)
        current_prot_set = frozenset(active_protections)
        if current_prot_set != self._prev_protection_set or can_trade != self._prev_can_trade:
            if active_protections:
                logger.info(
                    f"Protection change: {', '.join(active_protections)} | "
                    f"Trade={can_trade}, Size={position_size_multiplier*100:.0f}%"
                )
            elif self._prev_protection_set:
                logger.info("All protections cleared - normal trading")
            self._prev_protection_set = current_prot_set
            self._prev_can_trade = can_trade

        # Send alerts if needed
        if self.alert_manager:
            self._send_alerts_if_needed(decision, current_balance)

        return decision

    def _send_alerts_if_needed(self, decision: ProtectionDecision, current_balance: float):
        """Send alerts based on protection decision"""
        if not self.alert_manager:
            return

        # Circuit breaker triggered
        cb_state = self.circuit_breaker.state
        if cb_state != CircuitBreakerState.ACTIVE:
            # Check if this is a new trigger (not already alerted)
            # Implementation would check against last alert sent
            pass

        # Crisis mode activated
        crisis_level = self.crisis_detector.current_level
        if crisis_level in [CrisisLevel.ORANGE, CrisisLevel.RED]:
            # Send crisis alert
            pass

        # Manual action required
        if decision.requires_manual_action:
            self.alert_manager.alert_manual_restart_required(
                reason="Circuit breaker or crisis requires manual restart",
                balance=current_balance
            )

    def get_comprehensive_status(self) -> Dict:
        """Get status from all protection systems"""
        return {
            'volatility_filter': self.volatility_filter.get_status(),
            'crisis_detector': self.crisis_detector.get_status(),
            'circuit_breaker': self.circuit_breaker.get_status(),
            'recovery_manager': self.recovery_manager.get_status(),
            'profit_protector': self.profit_protector.get_status(),
            'peak_balance': self.peak_balance,
            'last_check': self.last_check_time.isoformat(),
        }

    def get_statistics(self) -> Dict:
        """Get statistics from all protection systems"""
        return {
            'volatility_filter': self.volatility_filter.get_statistics(),
            'crisis_detector': self.crisis_detector.get_statistics(),
            'circuit_breaker': self.circuit_breaker.get_statistics(),
            'recovery_manager': self.recovery_manager.get_statistics(),
            'profit_protector': self.profit_protector.get_statistics(),
            'total_decisions': len(self.decision_history),
            'protections_active_now': len(self.decision_history[-1].active_protections) if self.decision_history else 0,
        }

    def manual_resume(self, current_balance: float):
        """
        Manually resume trading after requiring manual action.

        Args:
            current_balance: Current balance
        """
        # Reset systems that require manual restart
        if self.circuit_breaker.manual_resume_required:
            self.circuit_breaker.manual_resume(current_balance, datetime.now())

        # Reset crisis detector if in RED
        if self.crisis_detector.current_level == CrisisLevel.RED:
            self.crisis_detector.current_level = CrisisLevel.NORMAL
            self.crisis_detector.active_signals = []

        logger.success("✅ Manual resume completed - All systems reset")
        logger.info("   Trading can now resume")

    def force_pause(self, reason: str):
        """Force pause all trading (emergency stop)"""
        logger.critical(f"🛑 FORCE PAUSE: {reason}")

        # Trigger circuit breaker monthly stop
        self.circuit_breaker.manual_resume_required = True
        self.circuit_breaker.state = CircuitBreakerState.MONTHLY_STOP

        if self.alert_manager:
            self.alert_manager.alert_manual_restart_required(
                reason=f"Force pause: {reason}",
                balance=0.0  # Will be updated
            )

    def reset_all(self):
        """Reset all protection systems (for testing)"""
        self.volatility_filter.reset()
        self.crisis_detector.reset()
        self.circuit_breaker.reset(self.initial_balance)
        self.recovery_manager.reset()
        self.profit_protector.reset()
        self.peak_balance = self.initial_balance
        self.decision_history = []
        self._prev_protection_set = frozenset()
        self._prev_can_trade = True
