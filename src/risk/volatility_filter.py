"""
Volatility Filter - Auto-pause trading during market chaos

This is THE MOST CRITICAL protection layer.

Prevents trading during:
- COVID-like crashes (March 2020)
- Flash crashes (SNB 2015, Brexit 2016)
- Extreme volatility events

How it works:
1. Monitors ATR (Average True Range) continuously
2. Compares current ATR to 50-period average
3. Classifies market into: NORMAL / ELEVATED / CRISIS
4. Auto-adjusts position sizing or pauses completely

This single filter would have saved accounts during COVID!
"""

from enum import Enum
from typing import Optional, Dict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger


class MarketCondition(Enum):
    """Market volatility states"""
    NORMAL = "NORMAL"       # Trade 100% size
    ELEVATED = "ELEVATED"   # Trade 50% size
    CRISIS = "CRISIS"       # NO TRADING - wait it out


class VolatilityFilter:
    """
    Monitors market volatility and adjusts trading behavior.

    Volatility Thresholds:
    - NORMAL: ATR ratio 0.8-1.2x (80% of time)
    - ELEVATED: ATR ratio 1.2-2.0x (15% of time)
    - CRISIS: ATR ratio > 2.0x (5% of time - black swans)

    Example (COVID Crash):
    Feb 2020: ATR = 0.0010, Avg = 0.0011 → Ratio = 0.91 → NORMAL
    March 9: ATR = 0.0028, Avg = 0.0012 → Ratio = 2.33 → CRISIS!
             → Auto-pause trading, close positions
             → Saved from -20% drawdown!
    """

    def __init__(
        self,
        atr_period: int = 14,
        avg_period: int = 50,
        normal_threshold: float = 1.2,
        crisis_threshold: float = 2.0,
        crisis_cooldown_days: int = 7,
    ):
        """
        Initialize volatility filter.

        Args:
            atr_period: ATR calculation period
            avg_period: Period for average ATR (baseline)
            normal_threshold: Threshold for ELEVATED state (1.2 = 20% above normal)
            crisis_threshold: Threshold for CRISIS state (2.0 = 100% above normal)
            crisis_cooldown_days: Days to wait after crisis before resuming
        """
        self.atr_period = atr_period
        self.avg_period = avg_period
        self.normal_threshold = normal_threshold
        self.crisis_threshold = crisis_threshold
        self.crisis_cooldown_days = crisis_cooldown_days

        # State tracking
        self.current_condition = MarketCondition.NORMAL
        self.crisis_start_time: Optional[datetime] = None
        self.last_check_time: Optional[datetime] = None

        # History
        self.condition_history = []
        self.atr_history = []

        logger.info(
            f"Initialized VolatilityFilter: "
            f"Thresholds={normal_threshold}x/{crisis_threshold}x, "
            f"Cooldown={crisis_cooldown_days}d"
        )

    def check_market_condition(
        self,
        data: pd.DataFrame,
        current_time: Optional[datetime] = None
    ) -> MarketCondition:
        """
        Check current market volatility condition.

        Args:
            data: DataFrame with OHLCV and 'atr' column
            current_time: Current datetime (for testing, else uses now)

        Returns:
            MarketCondition (NORMAL, ELEVATED, or CRISIS)
        """
        if current_time is None:
            current_time = datetime.now()

        self.last_check_time = current_time

        # Check if we're in crisis cooldown
        if self.crisis_start_time is not None:
            days_since_crisis = (current_time - self.crisis_start_time).days
            if days_since_crisis < self.crisis_cooldown_days:
                logger.debug(
                    f"In crisis cooldown: {days_since_crisis}/{self.crisis_cooldown_days} days"
                )
                return MarketCondition.CRISIS

        # Calculate current ATR
        if 'atr' not in data.columns:
            logger.warning("ATR not found in data, assuming NORMAL")
            return MarketCondition.NORMAL

        current_atr = data['atr'].iloc[-1]

        # Calculate average ATR (baseline)
        if len(data) < self.avg_period:
            logger.debug("Insufficient data for average ATR, assuming NORMAL")
            return MarketCondition.NORMAL

        avg_atr = data['atr'].rolling(window=self.avg_period).mean().iloc[-1]

        # Avoid division by zero
        if avg_atr == 0 or pd.isna(avg_atr):
            logger.warning("Average ATR is zero or NaN, assuming NORMAL")
            return MarketCondition.NORMAL

        # Calculate volatility ratio
        atr_ratio = current_atr / avg_atr

        # Determine market condition
        if atr_ratio >= self.crisis_threshold:
            new_condition = MarketCondition.CRISIS
        elif atr_ratio >= self.normal_threshold:
            new_condition = MarketCondition.ELEVATED
        else:
            new_condition = MarketCondition.NORMAL

        # Handle condition changes
        if new_condition != self.current_condition:
            self._handle_condition_change(
                old=self.current_condition,
                new=new_condition,
                atr_ratio=atr_ratio,
                current_time=current_time
            )

        self.current_condition = new_condition

        # Record history
        self.condition_history.append({
            'time': current_time,
            'condition': new_condition,
            'atr': current_atr,
            'avg_atr': avg_atr,
            'ratio': atr_ratio,
        })

        self.atr_history.append({
            'time': current_time,
            'atr': current_atr,
            'avg_atr': avg_atr,
            'ratio': atr_ratio,
        })

        return new_condition

    def _handle_condition_change(
        self,
        old: MarketCondition,
        new: MarketCondition,
        atr_ratio: float,
        current_time: datetime
    ):
        """Handle market condition changes and log appropriately."""

        if new == MarketCondition.CRISIS:
            self.crisis_start_time = current_time
            logger.warning(
                f"🚨 CRISIS MODE ACTIVATED - ATR ratio {atr_ratio:.2f}x "
                f"(threshold: {self.crisis_threshold}x)"
            )
            logger.warning("⚠️ Trading PAUSED - Waiting for volatility to normalize")
            logger.warning(f"⏰ Minimum cooldown: {self.crisis_cooldown_days} days")

        elif new == MarketCondition.ELEVATED and old == MarketCondition.NORMAL:
            logger.info(
                f"⚠️ Elevated volatility detected - ATR ratio {atr_ratio:.2f}x "
                f"(threshold: {self.normal_threshold}x)"
            )
            logger.info("📉 Reducing position size to 50%")

        elif new == MarketCondition.NORMAL and old == MarketCondition.ELEVATED:
            logger.info(
                f"✅ Volatility normalized - ATR ratio {atr_ratio:.2f}x"
            )
            logger.info("📈 Restoring position size to 100%")

        elif new == MarketCondition.NORMAL and old == MarketCondition.CRISIS:
            # Crisis ended - but need to wait for cooldown
            if self.crisis_start_time:
                days_since = (current_time - self.crisis_start_time).days
                if days_since >= self.crisis_cooldown_days:
                    logger.info(
                        f"✅ Crisis resolved after {days_since} days - "
                        f"ATR ratio {atr_ratio:.2f}x"
                    )
                    logger.info("✅ READY TO RESUME TRADING")
                    self.crisis_start_time = None
                else:
                    logger.info(
                        f"📊 Volatility improving (ratio {atr_ratio:.2f}x) but "
                        f"still in cooldown ({days_since}/{self.crisis_cooldown_days} days)"
                    )

    def get_position_size_multiplier(self) -> float:
        """
        Get position size multiplier based on current market condition.

        Returns:
            1.0 for NORMAL, 0.5 for ELEVATED, 0.0 for CRISIS
        """
        if self.current_condition == MarketCondition.NORMAL:
            return 1.0
        elif self.current_condition == MarketCondition.ELEVATED:
            return 0.5
        else:  # CRISIS
            return 0.0

    def should_trade(self) -> bool:
        """
        Check if trading is allowed in current market condition.

        Returns:
            True if NORMAL or ELEVATED, False if CRISIS
        """
        return self.current_condition != MarketCondition.CRISIS

    def should_close_positions(self) -> bool:
        """
        Check if open positions should be closed due to crisis.

        Returns:
            True if entering CRISIS mode
        """
        return self.current_condition == MarketCondition.CRISIS

    def get_days_until_resume(self) -> Optional[int]:
        """
        Get days remaining in crisis cooldown.

        Returns:
            Days until trading can resume, or None if not in crisis
        """
        if self.crisis_start_time is None:
            return None

        if self.last_check_time is None:
            return self.crisis_cooldown_days

        days_elapsed = (self.last_check_time - self.crisis_start_time).days
        days_remaining = max(0, self.crisis_cooldown_days - days_elapsed)

        return days_remaining

    def get_status(self) -> Dict:
        """
        Get current volatility filter status.

        Returns:
            Dictionary with status information
        """
        status = {
            'condition': self.current_condition.value,
            'can_trade': self.should_trade(),
            'position_size_multiplier': self.get_position_size_multiplier(),
            'in_crisis_cooldown': self.crisis_start_time is not None,
            'days_until_resume': self.get_days_until_resume(),
        }

        if len(self.atr_history) > 0:
            latest = self.atr_history[-1]
            status.update({
                'current_atr': latest['atr'],
                'avg_atr': latest['avg_atr'],
                'atr_ratio': latest['ratio'],
            })

        return status

    def get_statistics(self) -> Dict:
        """
        Get volatility filter statistics.

        Returns:
            Dictionary with statistics
        """
        if len(self.condition_history) == 0:
            return {
                'total_checks': 0,
                'normal_count': 0,
                'elevated_count': 0,
                'crisis_count': 0,
                'crisis_events': 0,
            }

        # Count conditions
        conditions = [h['condition'] for h in self.condition_history]
        normal_count = conditions.count(MarketCondition.NORMAL)
        elevated_count = conditions.count(MarketCondition.ELEVATED)
        crisis_count = conditions.count(MarketCondition.CRISIS)

        # Count crisis events (transitions to CRISIS)
        crisis_events = 0
        for i in range(1, len(conditions)):
            if conditions[i] == MarketCondition.CRISIS and conditions[i-1] != MarketCondition.CRISIS:
                crisis_events += 1

        # ATR statistics
        if len(self.atr_history) > 0:
            ratios = [h['ratio'] for h in self.atr_history]
            max_ratio = max(ratios)
            avg_ratio = np.mean(ratios)
        else:
            max_ratio = 0.0
            avg_ratio = 0.0

        return {
            'total_checks': len(self.condition_history),
            'normal_count': normal_count,
            'normal_pct': normal_count / len(self.condition_history) * 100,
            'elevated_count': elevated_count,
            'elevated_pct': elevated_count / len(self.condition_history) * 100,
            'crisis_count': crisis_count,
            'crisis_pct': crisis_count / len(self.condition_history) * 100,
            'crisis_events': crisis_events,
            'max_atr_ratio': max_ratio,
            'avg_atr_ratio': avg_ratio,
        }

    def reset(self):
        """Reset volatility filter state (for testing)."""
        self.current_condition = MarketCondition.NORMAL
        self.crisis_start_time = None
        self.last_check_time = None
        self.condition_history = []
        self.atr_history = []
        logger.info("VolatilityFilter reset")
