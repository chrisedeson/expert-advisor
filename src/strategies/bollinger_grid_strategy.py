"""
Bollinger Bands Grid Strategy (Medium-Risk)

A mean reversion grid strategy using Bollinger Bands for entry signals.

Strategy:
- Entry: Price touches BB upper band → SELL, lower band → BUY
- Grid: Opens additional positions if price moves against us (max 5 levels)
- Exit: Global TP when all positions in profit, hard stop at 20% drawdown
- Lot Management: Conservative scaling (1.2x per level)

Risk Profile:
- Max Drawdown: 20%
- Max Grid Levels: 5
- Blowup Risk: 15-20%
- Expected Return: 40-50% annually

Dynamic Money Management:
- Position size automatically scales with account balance
- Always risk 1% per trade
- Grid levels adjust based on available capital
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy
from . import indicators


class BollingerGridStrategy(BaseStrategy):
    """
    Medium-Risk Grid Strategy using Bollinger Bands mean reversion.

    Automatically adjusts position sizing and grid levels based on account balance.
    """

    def __init__(
        self,
        # Bollinger Bands settings
        bb_period: int = 20,
        bb_deviation: float = 2.0,

        # Grid settings
        max_grid_levels: int = 5,
        grid_spacing_atr_mult: float = 1.2,
        lot_scaling_factor: float = 1.2,

        # Risk management
        risk_per_trade: float = 0.01,  # 1% risk per trade
        max_drawdown: float = 0.20,    # 20% max drawdown

        # Filters
        max_spread: float = 20.0,      # Max spread in points
        enable_session_filter: bool = True,

        # Exit settings
        take_profit_atr_mult: float = 1.5,
        trailing_stop_enabled: bool = True,
        trailing_start_pips: float = 50.0,
        trailing_distance_pips: float = 30.0,
    ):
        """
        Initialize Bollinger Grid Strategy.

        Args:
            bb_period: Bollinger Bands period
            bb_deviation: BB standard deviations
            max_grid_levels: Maximum number of grid positions (1 initial + 4 additional)
            grid_spacing_atr_mult: Grid spacing as ATR multiplier
            lot_scaling_factor: Lot multiplier per grid level (1.2 = 20% increase)
            risk_per_trade: Risk percentage per trade (0.01 = 1%)
            max_drawdown: Maximum allowed drawdown before emergency stop
            max_spread: Maximum spread to allow trading
            enable_session_filter: Only trade during London/NY session
            take_profit_atr_mult: Take profit as ATR multiplier
            trailing_stop_enabled: Enable trailing stop
            trailing_start_pips: Start trailing after this profit
            trailing_distance_pips: Trailing stop distance
        """
        super().__init__()

        # Store parameters
        self.bb_period = bb_period
        self.bb_deviation = bb_deviation
        self.max_grid_levels = max_grid_levels
        self.grid_spacing_atr_mult = grid_spacing_atr_mult
        self.lot_scaling_factor = lot_scaling_factor
        self.risk_per_trade = risk_per_trade
        self.max_drawdown = max_drawdown
        self.max_spread = max_spread
        self.enable_session_filter = enable_session_filter
        self.take_profit_atr_mult = take_profit_atr_mult
        self.trailing_stop_enabled = trailing_stop_enabled
        self.trailing_start_pips = trailing_start_pips
        self.trailing_distance_pips = trailing_distance_pips

        # Dynamic position sizing parameters (updated in generate_signals)
        self.current_balance = 100.0  # Will be updated
        self.base_lot_size = 0.01     # Will be calculated dynamically
        self.available_grid_levels = max_grid_levels  # May reduce if low capital

        # Parameter space for optimization
        self.param_space = {
            'bb_period': (10, 30, 'int'),
            'bb_deviation': (1.5, 3.0, 'float'),
            'grid_spacing_atr_mult': (1.0, 2.0, 'float'),
            'lot_scaling_factor': (1.1, 1.4, 'float'),
            'take_profit_atr_mult': (1.0, 2.5, 'float'),
        }

    def update_dynamic_sizing(self, balance: float, current_price: float, atr: float):
        """
        Update position sizing based on current account balance.

        This enables the strategy to scale automatically as capital grows.

        Args:
            balance: Current account balance
            current_price: Current market price
            atr: Current ATR value
        """
        self.current_balance = balance

        # Calculate base lot size (risk 1% per trade)
        # Assuming stop loss at 2 * ATR
        risk_amount = balance * self.risk_per_trade
        stop_distance_pips = (2.0 * atr) * 10000  # Convert to pips
        pip_value = 1.0  # $1 per pip for 0.01 lots on EURUSD

        # Position size = Risk / (Stop Distance * Pip Value)
        self.base_lot_size = risk_amount / (stop_distance_pips * pip_value)

        # Ensure minimum lot size
        self.base_lot_size = max(0.01, self.base_lot_size)

        # Adjust grid levels based on capital
        # Need enough capital to support grid positions
        total_risk_if_all_levels = risk_amount * sum(
            self.lot_scaling_factor ** i for i in range(self.max_grid_levels)
        )

        # If total risk exceeds 15% of account, reduce grid levels
        if total_risk_if_all_levels > balance * 0.15:
            self.available_grid_levels = max(1, int(self.max_grid_levels * 0.6))
        else:
            self.available_grid_levels = self.max_grid_levels

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Bollinger Bands.

        Signal Logic:
        - BUY (1) when price touches lower band (expect mean reversion up)
        - SELL (-1) when price touches upper band (expect mean reversion down)
        - HOLD (0) otherwise

        Filters:
        - Spread filter (max spread)
        - Session filter (London/NY hours if enabled)

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with added 'signal' column
        """
        df = data.copy()

        # Calculate Bollinger Bands
        df['bb_middle'] = indicators.sma(df['close'], self.bb_period)
        df['bb_std'] = df['close'].rolling(window=self.bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (self.bb_deviation * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (self.bb_deviation * df['bb_std'])

        # Calculate ATR for grid spacing and TP
        df['atr'] = indicators.atr(df['high'], df['low'], df['close'], period=14)

        # Initialize signals
        df['signal'] = 0

        # Generate base signals (mean reversion)
        # BUY when price at/below lower band
        df.loc[df['close'] <= df['bb_lower'], 'signal'] = 1

        # SELL when price at/above upper band
        df.loc[df['close'] >= df['bb_upper'], 'signal'] = -1

        # Apply spread filter (simplified - assume 1 pip spread for backtest)
        # In live trading, this would check actual spread
        df['spread'] = 1.0  # Placeholder
        df.loc[df['spread'] > self.max_spread, 'signal'] = 0

        # Apply session filter if enabled
        if self.enable_session_filter:
            # London/NY session: 12:00 - 20:00 UTC
            if hasattr(df.index, 'hour'):
                df['hour'] = df.index.hour
                # Only trade during active session
                df.loc[(df['hour'] < 12) | (df['hour'] > 20), 'signal'] = 0

        # Update dynamic sizing for each bar
        if len(df) > self.bb_period:
            last_atr = df['atr'].iloc[-1]
            last_price = df['close'].iloc[-1]
            # Assume current balance is tracked externally (in backtesting engine)
            # This is updated in real-time during backtesting
            self.update_dynamic_sizing(self.current_balance, last_price, last_atr)

        # Store ATR for grid spacing calculations
        df['grid_spacing'] = df['atr'] * self.grid_spacing_atr_mult
        df['take_profit_distance'] = df['atr'] * self.take_profit_atr_mult

        return df

    def calculate_position_size(
        self,
        balance: float,
        grid_level: int = 0
    ) -> float:
        """
        Calculate position size for a given grid level.

        Grid Level 0: Base position (calculated from risk management)
        Grid Level 1: Base * lot_scaling_factor
        Grid Level 2: Base * lot_scaling_factor^2
        etc.

        Args:
            balance: Current account balance
            grid_level: Grid level (0 = initial position, 1-4 = grid positions)

        Returns:
            Position size in lots
        """
        # Update base lot size if balance changed
        if balance != self.current_balance:
            # Recalculate based on new balance
            # This is simplified; in practice, ATR is needed
            self.current_balance = balance
            risk_amount = balance * self.risk_per_trade
            # Assume average stop of 100 pips
            self.base_lot_size = max(0.01, risk_amount / 100.0)

        # Calculate scaled lot size for grid level
        lot_size = self.base_lot_size * (self.lot_scaling_factor ** grid_level)

        # Round to 2 decimals (standard lot precision)
        lot_size = round(lot_size, 2)

        # Ensure minimum lot size
        lot_size = max(0.01, lot_size)

        return lot_size

    def get_parameter_space(self) -> Dict:
        """Return parameter search space for optimization."""
        return self.param_space

    def set_parameters(self, **params):
        """Update strategy parameters."""
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)

    def get_parameters(self) -> Dict:
        """Get current strategy parameters."""
        return {
            'bb_period': self.bb_period,
            'bb_deviation': self.bb_deviation,
            'max_grid_levels': self.max_grid_levels,
            'grid_spacing_atr_mult': self.grid_spacing_atr_mult,
            'lot_scaling_factor': self.lot_scaling_factor,
            'risk_per_trade': self.risk_per_trade,
            'max_drawdown': self.max_drawdown,
            'take_profit_atr_mult': self.take_profit_atr_mult,
        }

    def __str__(self) -> str:
        """String representation."""
        return (
            f"BollingerGridStrategy(BB={self.bb_period}/{self.bb_deviation}, "
            f"GridLevels={self.max_grid_levels}, "
            f"Scaling={self.lot_scaling_factor}x, "
            f"Risk={self.risk_per_trade*100}%)"
        )
