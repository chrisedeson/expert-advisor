"""Signal engine: extracted from backtest engine for live trading.

Produces identical signals to the backtester. Takes OHLCV+ATR data,
returns entry/exit signals without managing positions or balance.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime


@dataclass
class Signal:
    """A trading signal."""
    direction: str  # 'BUY' or 'SELL'
    grid_level: int
    entry_price: float
    take_profit: float
    stop_loss: float
    symbol: str
    timestamp: datetime
    atr: float


@dataclass
class Position:
    """A tracked open position."""
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    entry_price: float
    entry_time: datetime
    lot_size: float
    grid_level: int
    stop_loss: float
    take_profit: float
    order_id: Optional[str] = None

    def unrealized_pnl(self, current_price: float, pip_size: float, pip_value: float) -> float:
        if self.direction == 'BUY':
            pips = (current_price - self.entry_price) / pip_size
        else:
            pips = (self.entry_price - current_price) / pip_size
        return pips * pip_value * self.lot_size


class SignalEngine:
    """Generates signals using the same logic as the backtest engine.

    This is a stateless signal generator. It takes recent candle data
    and current positions, and returns signals.

    Parameters match the golden config discovered in optimization.
    """

    def __init__(
        self,
        bb_entry_mult: float = 2.0,
        grid_spacing_atr: float = 0.75,
        sl_atr_mult: float = 1.5,
        use_trend_filter: bool = True,
        max_grid_levels: int = 5,
        min_tp_pips: float = 6.0,
        pip_size: float = 0.0001,
    ):
        self.bb_entry_mult = bb_entry_mult
        self.grid_spacing_atr = grid_spacing_atr
        self.sl_atr_mult = sl_atr_mult
        self.use_trend_filter = use_trend_filter
        self.max_grid_levels = max_grid_levels
        self.min_tp_pips = min_tp_pips
        self.pip_size = pip_size

    def check_entry(
        self,
        symbol: str,
        recent_data: pd.DataFrame,
        open_positions: List[Position],
        current_time: datetime,
    ) -> Optional[Signal]:
        """Check for entry signal. Requires at least 200 bars of data.

        Args:
            symbol: Instrument symbol
            recent_data: DataFrame with columns [open, high, low, close, volume, atr]
                         Must have at least 200 rows for SMA(200)
            open_positions: Currently open positions for this symbol
            current_time: Current timestamp

        Returns:
            Signal if entry conditions met, None otherwise
        """
        if len(recent_data) < 200:
            return None

        current_bar = recent_data.iloc[-1]
        close_prices = recent_data['close']
        current_price = current_bar['close']
        current_atr = current_bar['atr']

        if current_atr <= 0 or pd.isna(current_atr):
            return None

        # Volatility band filter
        atr_series = recent_data['atr']
        avg_atr = atr_series.rolling(50).mean().iloc[-1]
        if not pd.isna(avg_atr) and avg_atr > 0:
            atr_ratio = current_atr / avg_atr
            if atr_ratio < 0.8 or atr_ratio > 3.0:
                return None

        sma50 = close_prices.rolling(50).mean().iloc[-1]
        sma200 = close_prices.rolling(200).mean().iloc[-1]
        std50 = close_prices.rolling(50).std().iloc[-1]

        if pd.isna(sma50) or pd.isna(std50) or std50 <= 0 or pd.isna(sma200):
            return None

        upper_bb = sma50 + self.bb_entry_mult * std50
        lower_bb = sma50 - self.bb_entry_mult * std50

        is_uptrend = sma50 > sma200
        is_downtrend = sma50 < sma200

        allow_buy = is_uptrend if self.use_trend_filter else True
        allow_sell = is_downtrend if self.use_trend_filter else True

        buy_positions = [p for p in open_positions if p.direction == 'BUY' and p.symbol == symbol]
        sell_positions = [p for p in open_positions if p.direction == 'SELL' and p.symbol == symbol]

        # BUY SIGNAL
        if allow_buy and len(buy_positions) < self.max_grid_levels:
            buy_signal = False
            grid_level = 0

            if len(buy_positions) == 0:
                if current_price <= lower_bb:
                    buy_signal = True
                    grid_level = 0
            else:
                lowest_buy = min(p.entry_price for p in buy_positions)
                if current_price < lowest_buy - self.grid_spacing_atr * current_atr:
                    buy_signal = True
                    grid_level = len(buy_positions)

            if buy_signal:
                tp_price = upper_bb
                tp_distance_pips = (tp_price - current_price) / self.pip_size
                if tp_distance_pips >= self.min_tp_pips:
                    sl_price = current_price - self.sl_atr_mult * current_atr
                    return Signal(
                        direction='BUY', grid_level=grid_level,
                        entry_price=current_price, take_profit=tp_price,
                        stop_loss=sl_price, symbol=symbol,
                        timestamp=current_time, atr=current_atr,
                    )

        # SELL SIGNAL
        if allow_sell and len(sell_positions) < self.max_grid_levels:
            sell_signal = False
            grid_level = 0

            if len(sell_positions) == 0:
                if current_price >= upper_bb:
                    sell_signal = True
                    grid_level = 0
            else:
                highest_sell = max(p.entry_price for p in sell_positions)
                if current_price > highest_sell + self.grid_spacing_atr * current_atr:
                    sell_signal = True
                    grid_level = len(sell_positions)

            if sell_signal:
                tp_price = lower_bb
                tp_distance_pips = (current_price - tp_price) / self.pip_size
                if tp_distance_pips >= self.min_tp_pips:
                    sl_price = current_price + self.sl_atr_mult * current_atr
                    return Signal(
                        direction='SELL', grid_level=grid_level,
                        entry_price=current_price, take_profit=tp_price,
                        stop_loss=sl_price, symbol=symbol,
                        timestamp=current_time, atr=current_atr,
                    )

        return None

    def check_exits(
        self,
        positions: List[Position],
        current_bar: pd.Series,
        pip_size: float = None,
    ) -> List[Dict]:
        """Check for exit conditions on open positions.

        Returns list of dicts: {'position': pos, 'reason': str, 'exit_price': float}
        """
        if pip_size is None:
            pip_size = self.pip_size
        exits = []
        current_atr = current_bar.get('atr', 0)

        for pos in positions:
            if pos.direction == 'BUY':
                # TP hit
                if current_bar['high'] >= pos.take_profit:
                    exits.append({'position': pos, 'reason': 'Take Profit', 'exit_price': pos.take_profit})
                    continue
                # SL hit
                if current_bar['low'] <= pos.stop_loss:
                    exits.append({'position': pos, 'reason': 'Stop Loss', 'exit_price': pos.stop_loss})
                    continue
            else:
                if current_bar['low'] <= pos.take_profit:
                    exits.append({'position': pos, 'reason': 'Take Profit', 'exit_price': pos.take_profit})
                    continue
                if current_bar['high'] >= pos.stop_loss:
                    exits.append({'position': pos, 'reason': 'Stop Loss', 'exit_price': pos.stop_loss})
                    continue

        return exits

    def update_trailing_stops(
        self,
        positions: List[Position],
        current_bar: pd.Series,
    ):
        """Update two-stage trailing stops on positions. Modifies positions in-place."""
        current_atr = current_bar.get('atr', 0)
        if current_atr <= 0:
            return

        for pos in positions:
            if pos.direction == 'BUY':
                max_favorable = current_bar['high'] - pos.entry_price
                if max_favorable >= 2.0 * current_atr:
                    trail_sl = current_bar['high'] - 1.5 * current_atr
                    pos.stop_loss = max(pos.stop_loss, trail_sl)
                elif max_favorable >= 1.0 * current_atr:
                    pos.stop_loss = max(pos.stop_loss, pos.entry_price)
            else:
                max_favorable = pos.entry_price - current_bar['low']
                if max_favorable >= 2.0 * current_atr:
                    trail_sl = current_bar['low'] + 1.5 * current_atr
                    pos.stop_loss = min(pos.stop_loss, trail_sl)
                elif max_favorable >= 1.0 * current_atr:
                    pos.stop_loss = min(pos.stop_loss, pos.entry_price)
