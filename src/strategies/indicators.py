"""
Technical Indicators

Vectorized technical indicator calculations for strategy use.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple


def sma(data: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average.

    Args:
        data: Price series
        period: MA period

    Returns:
        SMA series
    """
    return data.rolling(window=period).mean()


def ema(data: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average.

    Args:
        data: Price series
        period: EMA period

    Returns:
        EMA series
    """
    return data.ewm(span=period, adjust=False).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average True Range (ATR).

    Measures volatility.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default: 14)

    Returns:
        ATR series
    """
    # True Range components
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    # True Range = max of the three
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR = EMA of True Range
    atr_values = tr.ewm(span=period, adjust=False).mean()

    return atr_values


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average Directional Index (ADX).

    Measures trend strength (0-100).
    - ADX < 20: Weak trend (ranging)
    - ADX 20-25: Developing trend
    - ADX 25-50: Strong trend
    - ADX > 50: Very strong trend

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period (default: 14)

    Returns:
        ADX series
    """
    # Calculate +DM and -DM
    high_diff = high.diff()
    low_diff = -low.diff()

    plus_dm = high_diff.copy()
    plus_dm[~((high_diff > low_diff) & (high_diff > 0))] = 0

    minus_dm = low_diff.copy()
    minus_dm[~((low_diff > high_diff) & (low_diff > 0))] = 0

    # Calculate ATR for normalization
    atr_values = atr(high, low, close, period)

    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_values)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_values)

    # Calculate DX (Directional Index)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)

    # ADX = smoothed DX
    adx_values = dx.ewm(span=period, adjust=False).mean()

    return adx_values


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI).

    Momentum oscillator (0-100).
    - RSI < 30: Oversold
    - RSI > 70: Overbought

    Args:
        close: Close prices
        period: RSI period (default: 14)

    Returns:
        RSI series
    """
    delta = close.diff()

    gain = delta.copy()
    gain[gain < 0] = 0

    loss = -delta.copy()
    loss[loss < 0] = 0

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi_values = 100 - (100 / (1 + rs))

    return rsi_values


def bollinger_bands(
    close: pd.Series, period: int = 20, std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.

    Args:
        close: Close prices
        period: MA period (default: 20)
        std_dev: Standard deviation multiplier (default: 2.0)

    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle = sma(close, period)
    std = close.rolling(window=period).std()

    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    return upper, middle, lower


def macd(
    close: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Moving Average Convergence Divergence (MACD).

    Args:
        close: Close prices
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    fast_ema = ema(close, fast_period)
    slow_ema = ema(close, slow_period)

    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Look-back period (default: 14)

    Returns:
        Tuple of (%K, %D)
    """
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()

    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=3).mean()

    return k, d


def session_filter(
    data: pd.DataFrame,
    start_hour: int = 12,
    start_minute: int = 0,
    end_hour: int = 16,
    end_minute: int = 0,
) -> pd.Series:
    """
    Filter for trading session (UTC time).

    Default: London/NY overlap (12:00-16:00 UTC)

    Args:
        data: DataFrame with DatetimeIndex
        start_hour: Session start hour (UTC)
        start_minute: Session start minute
        end_hour: Session end hour (UTC)
        end_minute: Session end minute

    Returns:
        Boolean series (True during session)
    """
    hour = data.index.hour
    minute = data.index.minute

    # Convert to minutes since midnight
    current_time = hour * 60 + minute
    start_time = start_hour * 60 + start_minute
    end_time = end_hour * 60 + end_minute

    in_session = (current_time >= start_time) & (current_time < end_time)

    return in_session


def add_all_indicators(
    data: pd.DataFrame,
    ma_periods: Optional[list] = None,
    atr_period: int = 14,
    adx_period: int = 14,
) -> pd.DataFrame:
    """
    Add commonly used indicators to data.

    Args:
        data: OHLCV DataFrame
        ma_periods: List of MA periods (default: [20, 50, 200])
        atr_period: ATR period
        adx_period: ADX period

    Returns:
        DataFrame with indicators added
    """
    if ma_periods is None:
        ma_periods = [20, 50, 200]

    df = data.copy()

    # Moving averages
    for period in ma_periods:
        df[f'sma_{period}'] = sma(df['close'], period)
        df[f'ema_{period}'] = ema(df['close'], period)

    # Volatility and trend
    df['atr'] = atr(df['high'], df['low'], df['close'], atr_period)
    df['adx'] = adx(df['high'], df['low'], df['close'], adx_period)

    # RSI
    df['rsi'] = rsi(df['close'])

    # Bollinger Bands
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = bollinger_bands(df['close'])

    # MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = macd(df['close'])

    return df
