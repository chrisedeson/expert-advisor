"""
Risk Manager

Handles position sizing, risk limits, and drawdown control.
"""

import numpy as np
import pandas as pd
from typing import Optional
from loguru import logger


class RiskManager:
    """
    Risk management for trading strategies.

    Key features:
    - Fixed fractional position sizing (risk % per trade)
    - ATR-based stop-loss placement
    - Maximum drawdown circuit breaker
    - Exposure limits
    """

    def __init__(
        self,
        initial_capital: float,
        risk_per_trade: float = 0.01,
        max_drawdown: float = 0.15,
        max_positions: int = 1,
    ):
        """
        Initialize risk manager.

        Args:
            initial_capital: Starting capital in USD
            risk_per_trade: Risk per trade as fraction (default: 1%)
            max_drawdown: Maximum drawdown before stopping (default: 15%)
            max_positions: Maximum concurrent positions (default: 1)
        """
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_drawdown = max_drawdown
        self.max_positions = max_positions

        self.current_capital = initial_capital
        self.peak_capital = initial_capital

        logger.info(
            f"Initialized risk manager: "
            f"Risk={risk_per_trade*100}%, MaxDD={max_drawdown*100}%, "
            f"MaxPos={max_positions}"
        )

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        current_capital: Optional[float] = None,
    ) -> float:
        """
        Calculate position size using fixed fractional method.

        Position size = (Risk amount) / (Stop loss distance in dollars)

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss_price: Stop-loss price
            current_capital: Current capital (defaults to initial)

        Returns:
            Position size in lots (can be fractional for micro-lots)
        """
        capital = current_capital or self.current_capital
        risk_amount = capital * self.risk_per_trade

        # Calculate stop distance in price
        stop_distance = abs(entry_price - stop_loss_price)

        if stop_distance == 0:
            logger.warning("Stop distance is zero, using minimum position size")
            return 0.01  # Minimum micro-lot

        # Calculate pip value
        if 'JPY' in symbol:
            # JPY pairs: 1 pip = 0.01
            pip_value = 10.0  # $10 per pip per standard lot (approximate)
            stop_distance_pips = stop_distance / 0.01
        else:
            # Other pairs: 1 pip = 0.0001
            pip_value = 10.0
            stop_distance_pips = stop_distance / 0.0001

        # Position size in lots
        # Risk amount = Stop distance (pips) * Pip value * Position size
        # Position size = Risk amount / (Stop distance pips * Pip value)
        position_size = risk_amount / (stop_distance_pips * pip_value)

        # Ensure minimum size (0.01 micro-lot on Exness)
        position_size = max(position_size, 0.01)

        logger.debug(
            f"Position sizing: Risk=${risk_amount:.2f}, "
            f"Stop={stop_distance_pips:.1f} pips, Size={position_size:.4f} lots"
        )

        return position_size

    def calculate_stop_loss(
        self,
        data: pd.DataFrame,
        atr_multiplier: float = 2.5,
        min_stop_pips: float = 10.0,
    ) -> pd.Series:
        """
        Calculate ATR-based stop-loss distance.

        Args:
            data: DataFrame with 'high', 'low', 'close'
            atr_multiplier: Multiplier for ATR (default: 2.5x)
            min_stop_pips: Minimum stop distance in pips (default: 10)

        Returns:
            Series with stop-loss distances (as fraction of price)
        """
        # Calculate ATR if not present
        if 'atr' not in data.columns:
            atr = self._calculate_atr(data, period=14)
        else:
            atr = data['atr']

        # Stop distance = ATR * multiplier
        stop_distance = atr * atr_multiplier

        # Convert to fraction of price
        stop_distance_pct = stop_distance / data['close']

        # Apply minimum stop (in pips)
        min_stop_fraction = min_stop_pips * 0.0001 / data['close']
        stop_distance_pct = stop_distance_pct.clip(lower=min_stop_fraction)

        return stop_distance_pct

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).

        Args:
            data: DataFrame with 'high', 'low', 'close'
            period: ATR period (default: 14)

        Returns:
            Series with ATR values
        """
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)

        # True Range
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR = EMA of TR
        atr = tr.ewm(span=period, adjust=False).mean()

        return atr

    def check_drawdown(self, current_equity: float) -> bool:
        """
        Check if current drawdown exceeds maximum limit.

        Args:
            current_equity: Current equity value

        Returns:
            True if drawdown limit breached, False otherwise
        """
        # Update peak
        if current_equity > self.peak_capital:
            self.peak_capital = current_equity

        # Calculate drawdown
        drawdown = (current_equity - self.peak_capital) / self.peak_capital

        if drawdown < -self.max_drawdown:
            logger.error(
                f"DRAWDOWN LIMIT BREACHED: {drawdown*100:.2f}% "
                f"(max: {self.max_drawdown*100:.2f}%)"
            )
            return True

        return False

    def update_capital(self, new_capital: float) -> None:
        """
        Update current capital (after trades).

        Args:
            new_capital: New capital value
        """
        self.current_capital = new_capital

        # Update peak if needed
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital

    def get_current_drawdown(self) -> float:
        """
        Get current drawdown as fraction.

        Returns:
            Drawdown as negative fraction (e.g., -0.05 = 5% drawdown)
        """
        return (self.current_capital - self.peak_capital) / self.peak_capital

    def reset(self) -> None:
        """Reset risk manager to initial state."""
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        logger.info("Risk manager reset")

    def __repr__(self) -> str:
        return (
            f"RiskManager(capital=${self.current_capital:.2f}, "
            f"risk={self.risk_per_trade*100}%, "
            f"drawdown={self.get_current_drawdown()*100:.2f}%)"
        )
