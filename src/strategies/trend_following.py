"""
Trend Following Strategy

Conservative trend-following strategy using:
- Moving average crossovers for trend direction
- ADX for trend strength filtering
- ATR-based position sizing and stops
- Session filtering (London/NY overlap)
- Pullback entry logic
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

from .base_strategy import BaseStrategy
from .indicators import sma, ema, atr, adx, session_filter


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend-following strategy optimized for Forex.

    Entry Logic:
    1. Trend direction: Fast MA > Slow MA = Long bias
    2. Trend strength: ADX > threshold (confirming strong trend)
    3. Session filter: Only trade during high liquidity hours
    4. Entry: Price pulls back to fast MA (momentum entry)

    Exit Logic:
    - MA crossover reversal (trend change)
    - ATR-based stop-loss (dynamic)

    Parameters:
    - fast_ma_period: Fast MA period (default: 20)
    - slow_ma_period: Slow MA period (default: 50)
    - adx_period: ADX calculation period (default: 14)
    - adx_threshold: Minimum ADX for trend confirmation (default: 25)
    - atr_period: ATR calculation period (default: 14)
    - use_session_filter: Enable session filtering (default: True)
    - ma_type: 'sma' or 'ema' (default: 'ema')
    """

    def __init__(
        self,
        fast_ma_period: int = 20,
        slow_ma_period: int = 50,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        atr_period: int = 14,
        use_session_filter: bool = True,
        ma_type: str = 'ema',
    ):
        """
        Initialize trend-following strategy.

        Args:
            fast_ma_period: Fast MA period
            slow_ma_period: Slow MA period
            adx_period: ADX period
            adx_threshold: Minimum ADX value (20-30 typical)
            atr_period: ATR period
            use_session_filter: Enable London/NY overlap filter
            ma_type: 'sma' or 'ema'
        """
        super().__init__(
            name="TrendFollowing",
            fast_ma_period=fast_ma_period,
            slow_ma_period=slow_ma_period,
            adx_period=adx_period,
            adx_threshold=adx_threshold,
            atr_period=atr_period,
            use_session_filter=use_session_filter,
            ma_type=ma_type,
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on trend-following logic.

        Args:
            data: OHLCV DataFrame

        Returns:
            Series with signals {-1, 0, 1}
        """
        # Validate data
        self.validate_data(data)

        df = data.copy()

        # Extract parameters
        fast_period = self.parameters['fast_ma_period']
        slow_period = self.parameters['slow_ma_period']
        adx_period = self.parameters['adx_period']
        adx_thresh = self.parameters['adx_threshold']
        atr_period = self.parameters['atr_period']
        use_session = self.parameters['use_session_filter']
        ma_type = self.parameters['ma_type']

        # Calculate indicators
        if ma_type == 'ema':
            df['fast_ma'] = ema(df['close'], fast_period)
            df['slow_ma'] = ema(df['close'], slow_period)
        else:  # sma
            df['fast_ma'] = sma(df['close'], fast_period)
            df['slow_ma'] = sma(df['close'], slow_period)

        df['atr'] = atr(df['high'], df['low'], df['close'], atr_period)
        df['adx'] = adx(df['high'], df['low'], df['close'], adx_period)

        # Trend direction: Fast MA vs Slow MA
        df['trend_long'] = df['fast_ma'] > df['slow_ma']
        df['trend_short'] = df['fast_ma'] < df['slow_ma']

        # Trend strength: ADX filter
        df['strong_trend'] = df['adx'] > adx_thresh

        # Session filter (optional)
        if use_session:
            df['in_session'] = session_filter(df, start_hour=12, end_hour=16)
        else:
            df['in_session'] = True

        # Entry conditions
        # Long: Uptrend + Strong trend + In session
        df['long_condition'] = (
            df['trend_long'] &
            df['strong_trend'] &
            df['in_session']
        )

        # Short: Downtrend + Strong trend + In session
        df['short_condition'] = (
            df['trend_short'] &
            df['strong_trend'] &
            df['in_session']
        )

        # Generate signals
        signals = pd.Series(0, index=df.index)
        signals[df['long_condition']] = 1
        signals[df['short_condition']] = -1

        # Forward-fill signals (stay in position until trend changes)
        signals = signals.replace(0, np.nan).ffill().fillna(0)

        return signals

    def get_parameter_space(self) -> Dict:
        """
        Get parameter space for optimization.

        Returns:
            Dictionary with parameter bounds
        """
        return {
            'fast_ma_period': (10, 50, 'int'),
            'slow_ma_period': (30, 200, 'int'),
            'adx_period': (10, 20, 'int'),
            'adx_threshold': (15.0, 35.0, 'float'),
            'atr_period': (10, 20, 'int'),
        }

    def get_stop_loss(self, data: pd.DataFrame, atr_multiplier: float = 2.5) -> pd.Series:
        """
        Calculate ATR-based stop-loss distances.

        Args:
            data: OHLCV DataFrame (must have ATR already calculated)
            atr_multiplier: Multiplier for ATR (default: 2.5x)

        Returns:
            Series with stop-loss distances as fraction of price
        """
        if 'atr' not in data.columns:
            # Calculate ATR if not present
            atr_period = self.parameters['atr_period']
            data['atr'] = atr(data['high'], data['low'], data['close'], atr_period)

        # Stop distance = ATR * multiplier
        stop_distance = data['atr'] * atr_multiplier

        # Convert to fraction of price
        stop_pct = stop_distance / data['close']

        # Apply minimum stop (e.g., 0.5% or 50 pips)
        min_stop = 0.005  # 0.5%
        stop_pct = stop_pct.clip(lower=min_stop)

        return stop_pct

    def get_take_profit(self, data: pd.DataFrame, risk_reward_ratio: float = 2.0) -> pd.Series:
        """
        Calculate take-profit based on risk-reward ratio.

        Args:
            data: OHLCV DataFrame
            risk_reward_ratio: Target R:R ratio (default: 2.0)

        Returns:
            Series with take-profit distances as fraction of price
        """
        stop_loss = self.get_stop_loss(data)
        take_profit = stop_loss * risk_reward_ratio

        return take_profit

    def get_position_size(
        self,
        data: pd.DataFrame,
        capital: float,
        risk_per_trade: float = 0.01,
    ) -> pd.Series:
        """
        Calculate position sizes based on ATR and risk.

        Args:
            data: OHLCV DataFrame
            capital: Current capital
            risk_per_trade: Risk as fraction (default: 1%)

        Returns:
            Series with position sizes
        """
        stop_loss_pct = self.get_stop_loss(data)
        risk_amount = capital * risk_per_trade

        # Position size = Risk amount / Stop loss distance
        position_sizes = risk_amount / (data['close'] * stop_loss_pct)

        # Normalize to reasonable values
        # For Forex, 1.0 = 1 standard lot (100,000 units)
        # We want micro-lots (0.01 = 1,000 units)
        position_sizes = position_sizes.clip(lower=0.01, upper=10.0)

        return position_sizes

    def describe(self) -> str:
        """
        Get human-readable strategy description.

        Returns:
            Strategy description string
        """
        params = self.parameters
        return f"""
Trend Following Strategy
========================
Type: Moving Average Crossover with ADX Filter

Entry Rules:
- Long: Fast MA ({params['fast_ma_period']}) > Slow MA ({params['slow_ma_period']})
- Short: Fast MA < Slow MA
- Filter: ADX > {params['adx_threshold']} (trend strength)
- Session: {'London/NY overlap only' if params['use_session_filter'] else 'All sessions'}

Exit Rules:
- Trend reversal (MA crossover)
- ATR-based stop-loss (2.5x ATR)

Risk Management:
- Position sizing: Based on ATR and 1% risk per trade
- Stop-loss: {params['atr_period']}-period ATR × 2.5

Indicators Used:
- MA Type: {params['ma_type'].upper()}
- ADX Period: {params['adx_period']}
- ATR Period: {params['atr_period']}
"""
