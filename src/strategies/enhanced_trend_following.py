"""
Enhanced Trend Following Strategy with Volatility Regime Filter

Adds volatility regime detection to avoid trading during crisis periods.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

from .trend_following import TrendFollowingStrategy
from .indicators import atr


class EnhancedTrendFollowingStrategy(TrendFollowingStrategy):
    """
    Enhanced trend-following strategy with volatility regime filter.

    Additions to base strategy:
    - Volatility regime detection (normal vs crisis)
    - Dynamic position sizing based on regime
    - Crisis mode circuit breaker
    - Correlation spike detection (optional)
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
        # New parameters for regime detection
        volatility_lookback: int = 100,
        volatility_threshold: float = 2.5,
        enable_regime_filter: bool = True,
    ):
        """
        Initialize enhanced strategy.

        Args:
            fast_ma_period: Fast MA period
            slow_ma_period: Slow MA period
            adx_period: ADX period
            adx_threshold: Minimum ADX value
            atr_period: ATR period
            use_session_filter: Enable session filter
            ma_type: 'sma' or 'ema'
            volatility_lookback: Period for volatility baseline (default: 100)
            volatility_threshold: Multiplier for crisis detection (default: 2.5x)
            enable_regime_filter: Enable volatility regime filtering
        """
        # Initialize base strategy
        super().__init__(
            fast_ma_period=fast_ma_period,
            slow_ma_period=slow_ma_period,
            adx_period=adx_period,
            adx_threshold=adx_threshold,
            atr_period=atr_period,
            use_session_filter=use_session_filter,
            ma_type=ma_type,
        )

        # Update name and parameters
        self.name = "EnhancedTrendFollowing"
        self.parameters.update({
            'volatility_lookback': volatility_lookback,
            'volatility_threshold': volatility_threshold,
            'enable_regime_filter': enable_regime_filter,
        })

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate signals with volatility regime filtering.

        Args:
            data: OHLCV DataFrame

        Returns:
            Series with signals {-1, 0, 1}
        """
        # Get base strategy signals
        base_signals = super().generate_signals(data)

        # Apply regime filter if enabled
        if not self.parameters['enable_regime_filter']:
            return base_signals

        # Calculate volatility regime
        df = data.copy()
        regime = self._detect_volatility_regime(df)

        # Filter signals: only trade in normal regime
        filtered_signals = base_signals.copy()
        filtered_signals[regime == 'crisis'] = 0  # Flat during crisis

        return filtered_signals

    def _detect_volatility_regime(self, data: pd.DataFrame) -> pd.Series:
        """
        Detect volatility regime: 'normal', 'elevated', or 'crisis'.

        Uses ATR relative to its moving average to detect abnormal volatility.

        Args:
            data: OHLCV DataFrame

        Returns:
            Series with regime labels
        """
        atr_period = self.parameters['atr_period']
        lookback = self.parameters['volatility_lookback']
        threshold = self.parameters['volatility_threshold']

        # Calculate ATR
        current_atr = atr(data['high'], data['low'], data['close'], atr_period)

        # Calculate baseline (rolling mean)
        baseline_atr = current_atr.rolling(window=lookback, min_periods=lookback//2).mean()

        # Calculate volatility ratio
        volatility_ratio = current_atr / baseline_atr

        # Classify regime
        regime = pd.Series('normal', index=data.index)
        regime[volatility_ratio > threshold] = 'crisis'
        regime[(volatility_ratio > (threshold * 0.7)) & (volatility_ratio <= threshold)] = 'elevated'

        return regime

    def get_position_size_adjusted(
        self,
        data: pd.DataFrame,
        capital: float,
        risk_per_trade: float = 0.01,
    ) -> pd.Series:
        """
        Calculate position sizes with regime-based adjustment.

        Reduces position size during elevated volatility.

        Args:
            data: OHLCV DataFrame
            capital: Current capital
            risk_per_trade: Base risk per trade

        Returns:
            Series with adjusted position sizes
        """
        # Get base position sizes
        base_sizes = self.get_position_size(data, capital, risk_per_trade)

        # Get regime
        regime = self._detect_volatility_regime(data)

        # Adjust based on regime
        adjusted_sizes = base_sizes.copy()
        adjusted_sizes[regime == 'elevated'] *= 0.5  # Half size in elevated regime
        adjusted_sizes[regime == 'crisis'] = 0.0      # No positions in crisis

        return adjusted_sizes

    def get_regime_stats(self, data: pd.DataFrame) -> Dict:
        """
        Get statistics about volatility regimes in the data.

        Args:
            data: OHLCV DataFrame

        Returns:
            Dictionary with regime statistics
        """
        regime = self._detect_volatility_regime(data)

        stats = {
            'total_bars': len(regime),
            'normal_bars': (regime == 'normal').sum(),
            'elevated_bars': (regime == 'elevated').sum(),
            'crisis_bars': (regime == 'crisis').sum(),
            'normal_pct': (regime == 'normal').sum() / len(regime) * 100,
            'elevated_pct': (regime == 'elevated').sum() / len(regime) * 100,
            'crisis_pct': (regime == 'crisis').sum() / len(regime) * 100,
        }

        return stats

    def get_parameter_space(self) -> Dict:
        """
        Get parameter space including new regime parameters.

        Returns:
            Dictionary with parameter bounds
        """
        base_space = super().get_parameter_space()

        # Add regime parameters
        base_space.update({
            'volatility_lookback': (50, 200, 'int'),
            'volatility_threshold': (2.0, 4.0, 'float'),
        })

        return base_space

    def describe(self) -> str:
        """
        Get human-readable strategy description.

        Returns:
            Strategy description string
        """
        params = self.parameters
        base_desc = super().describe()

        regime_desc = f"""

REGIME FILTER (NEW):
- Lookback: {params['volatility_lookback']} bars
- Crisis Threshold: {params['volatility_threshold']}x normal ATR
- Action: Stop trading during crisis periods

Regime Classification:
- Normal: ATR < {params['volatility_threshold']}x baseline
- Elevated: ATR {params['volatility_threshold']*0.7:.1f}x - {params['volatility_threshold']}x baseline (50% position size)
- Crisis: ATR > {params['volatility_threshold']}x baseline (NO trading)
"""

        return base_desc + regime_desc
