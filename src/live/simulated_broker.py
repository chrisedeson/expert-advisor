"""Simulated broker for paper trading. No real orders placed.

Tracks virtual positions and P&L using live price data from another
source (e.g., a data feed). Useful for validating the engine before
going live.
"""
import uuid
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional, List, Dict
from pathlib import Path

from .broker_interface import BrokerInterface, OrderResult, AccountInfo


class SimulatedBroker(BrokerInterface):
    """Paper trading broker. Simulates order execution with configurable slippage."""

    def __init__(
        self,
        initial_balance: float = 500.0,
        data_dir: str = "data/processed",
        slippage_pips: float = 0.5,
    ):
        self.initial_balance = initial_balance
        self.cash_balance = initial_balance
        self.data_dir = Path(data_dir)
        self.slippage_pips = slippage_pips
        self.positions: List[Dict] = []
        self.closed_trades: List[Dict] = []
        self.connected = False

        # Cache for candle data
        self._candle_cache: Dict[str, pd.DataFrame] = {}

    def connect(self) -> bool:
        self.connected = True
        return True

    def disconnect(self):
        self.connected = False

    def get_account_info(self) -> AccountInfo:
        equity = self.cash_balance
        for pos in self.positions:
            equity += pos.get('unrealized_pnl', 0)
        return AccountInfo(
            balance=self.cash_balance,
            equity=equity,
            margin_used=0,
            margin_free=equity,
        )

    def place_market_order(
        self,
        symbol: str,
        direction: str,
        lot_size: float,
        stop_loss: float,
        take_profit: float,
        comment: str = "",
    ) -> OrderResult:
        # Get current price
        price_info = self.get_current_price(symbol)
        if price_info is None:
            return OrderResult(success=False, error=f"No price data for {symbol}")

        # Apply slippage
        pip_size = self._get_pip_size(symbol)
        slip = self.slippage_pips * pip_size
        if direction == 'BUY':
            fill_price = price_info['ask'] + slip
        else:
            fill_price = price_info['bid'] - slip

        order_id = str(uuid.uuid4())[:8]
        position = {
            'order_id': order_id,
            'symbol': symbol,
            'direction': direction,
            'lot_size': lot_size,
            'entry_price': fill_price,
            'entry_time': datetime.now(timezone.utc),
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'comment': comment,
            'unrealized_pnl': 0,
        }
        self.positions.append(position)
        return OrderResult(success=True, order_id=order_id, fill_price=fill_price)

    def close_position(self, order_id: str) -> OrderResult:
        pos = next((p for p in self.positions if p['order_id'] == order_id), None)
        if pos is None:
            return OrderResult(success=False, error=f"Position {order_id} not found")

        price_info = self.get_current_price(pos['symbol'])
        if price_info is None:
            return OrderResult(success=False, error=f"No price for {pos['symbol']}")

        pip_size = self._get_pip_size(pos['symbol'])
        pip_value = self._get_pip_value(pos['symbol'])
        slip = self.slippage_pips * pip_size

        if pos['direction'] == 'BUY':
            exit_price = price_info['bid'] - slip
            pips = (exit_price - pos['entry_price']) / pip_size
        else:
            exit_price = price_info['ask'] + slip
            pips = (pos['entry_price'] - exit_price) / pip_size

        net_pnl = pips * pip_value * pos['lot_size']
        self.cash_balance += net_pnl

        trade = {**pos, 'exit_price': exit_price, 'exit_time': datetime.now(timezone.utc),
                 'pips': pips, 'net_pnl': net_pnl, 'exit_reason': 'Manual close'}
        self.closed_trades.append(trade)
        self.positions.remove(pos)

        return OrderResult(success=True, order_id=order_id, fill_price=exit_price)

    def modify_position(
        self, order_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> OrderResult:
        pos = next((p for p in self.positions if p['order_id'] == order_id), None)
        if pos is None:
            return OrderResult(success=False, error=f"Position {order_id} not found")
        if stop_loss is not None:
            pos['stop_loss'] = stop_loss
        if take_profit is not None:
            pos['take_profit'] = take_profit
        return OrderResult(success=True, order_id=order_id)

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        if symbol:
            return [p for p in self.positions if p['symbol'] == symbol]
        return list(self.positions)

    def get_candles(
        self, symbol: str, timeframe: str = "H1", count: int = 250,
    ) -> Optional[pd.DataFrame]:
        """Load candles from parquet files, returning the most recent `count` bars."""
        cache_key = f"{symbol}_{timeframe}"
        if cache_key not in self._candle_cache:
            path = self.data_dir / f"{symbol}_{timeframe}.parquet"
            if not path.exists():
                return None
            df = pd.read_parquet(path)
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    df.set_index('time', inplace=True)
            # Compute ATR if missing
            if 'atr' not in df.columns:
                tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                               abs(df['low'] - df['close'].shift(1))))
                df['atr'] = tr.rolling(14).mean()
            self._candle_cache[cache_key] = df

        df = self._candle_cache[cache_key]
        # Return last `count` bars up to current time
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        available = df[df.index <= now]
        if len(available) == 0:
            available = df  # Use all data if we're backtesting
        return available.tail(count)

    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """Get latest price from candle data. Simulates bid/ask with spread."""
        candles = self.get_candles(symbol, "H1", count=1)
        if candles is None or len(candles) == 0:
            return None
        last = candles.iloc[-1]
        pip_size = self._get_pip_size(symbol)
        spread = self._get_spread(symbol) * pip_size
        mid = last['close']
        return {'bid': mid - spread / 2, 'ask': mid + spread / 2, 'mid': mid}

    def check_sl_tp(self, current_bars: Dict[str, pd.Series]):
        """Check if any positions hit SL or TP based on current bar data.

        Args:
            current_bars: dict of {symbol: latest_bar_series}
        """
        to_close = []
        for pos in self.positions:
            symbol = pos['symbol']
            if symbol not in current_bars:
                continue
            bar = current_bars[symbol]
            pip_size = self._get_pip_size(symbol)
            pip_value = self._get_pip_value(symbol)

            if pos['direction'] == 'BUY':
                if bar['high'] >= pos['take_profit']:
                    exit_price = pos['take_profit']
                    reason = 'Take Profit'
                elif bar['low'] <= pos['stop_loss']:
                    exit_price = pos['stop_loss']
                    reason = 'Stop Loss'
                else:
                    # Update unrealized P&L
                    pips = (bar['close'] - pos['entry_price']) / pip_size
                    pos['unrealized_pnl'] = pips * pip_value * pos['lot_size']
                    continue
            else:
                if bar['low'] <= pos['take_profit']:
                    exit_price = pos['take_profit']
                    reason = 'Take Profit'
                elif bar['high'] >= pos['stop_loss']:
                    exit_price = pos['stop_loss']
                    reason = 'Stop Loss'
                else:
                    pips = (pos['entry_price'] - bar['close']) / pip_size
                    pos['unrealized_pnl'] = pips * pip_value * pos['lot_size']
                    continue

            # Close the position
            if pos['direction'] == 'BUY':
                pips = (exit_price - pos['entry_price']) / pip_size
            else:
                pips = (pos['entry_price'] - exit_price) / pip_size

            net_pnl = pips * pip_value * pos['lot_size']
            self.cash_balance += net_pnl

            trade = {**pos, 'exit_price': exit_price, 'exit_time': datetime.now(timezone.utc),
                     'pips': pips, 'net_pnl': net_pnl, 'exit_reason': reason}
            self.closed_trades.append(trade)
            to_close.append(pos)

        for pos in to_close:
            self.positions.remove(pos)

        return to_close

    def _get_pip_size(self, symbol: str) -> float:
        specs = {
            'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'EURJPY': 0.01,
            'XAGUSD': 0.001, 'US500': 0.01,
        }
        return specs.get(symbol, 0.0001)

    def _get_pip_value(self, symbol: str) -> float:
        specs = {
            'EURUSD': 10.0, 'GBPUSD': 10.0, 'EURJPY': 6.67,
            'XAGUSD': 5.0, 'US500': 1.0,
        }
        return specs.get(symbol, 10.0)

    def _get_spread(self, symbol: str) -> float:
        specs = {
            'EURUSD': 0.7, 'GBPUSD': 0.9, 'EURJPY': 1.0,
            'XAGUSD': 3.0, 'US500': 0.5,
        }
        return specs.get(symbol, 1.0)
