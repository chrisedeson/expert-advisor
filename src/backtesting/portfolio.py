"""
Portfolio Tracker

Tracks positions, equity, and performance metrics.
"""

import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class Position:
    """Represents an open trading position."""

    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    position_size: float  # in lots
    entry_time: pd.Timestamp
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def get_unrealized_pnl(self, current_price: float) -> float:
        """
        Calculate unrealized P&L in price units.

        Args:
            current_price: Current market price

        Returns:
            P&L as price difference
        """
        if self.direction == "LONG":
            return (current_price - self.entry_price) * self.position_size
        else:  # SHORT
            return (self.entry_price - current_price) * self.position_size

    def get_unrealized_pnl_pct(self, current_price: float) -> float:
        """
        Calculate unrealized P&L as percentage.

        Args:
            current_price: Current market price

        Returns:
            P&L as percentage
        """
        if self.direction == "LONG":
            return (current_price - self.entry_price) / self.entry_price
        else:  # SHORT
            return (self.entry_price - current_price) / self.entry_price


class Portfolio:
    """
    Portfolio tracker for backtesting.

    Tracks:
    - Current positions
    - Equity over time
    - Trade history
    - Performance metrics
    """

    def __init__(self, initial_capital: float):
        """
        Initialize portfolio.

        Args:
            initial_capital: Starting capital in USD
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.equity = initial_capital

        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Dict] = []

        self.equity_history: List[Dict] = []

        logger.info(f"Initialized portfolio: Capital=${initial_capital:.2f}")

    def open_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        position_size: float,
        entry_time: pd.Timestamp,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> None:
        """
        Open a new position.

        Args:
            symbol: Trading symbol
            direction: "LONG" or "SHORT"
            entry_price: Entry price
            position_size: Position size in lots
            entry_time: Entry timestamp
            stop_loss: Stop-loss price (optional)
            take_profit: Take-profit price (optional)
        """
        if symbol in self.positions:
            logger.warning(f"Position already exists for {symbol}, closing first")
            self.close_position(symbol, entry_price, entry_time)

        position = Position(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            position_size=position_size,
            entry_time=entry_time,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        self.positions[symbol] = position

        logger.debug(
            f"Opened {direction} position: {symbol} @ {entry_price:.5f}, "
            f"size={position_size:.4f} lots"
        )

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_time: pd.Timestamp,
        exit_reason: str = "SIGNAL",
    ) -> Optional[Dict]:
        """
        Close an existing position.

        Args:
            symbol: Trading symbol
            exit_price: Exit price
            exit_time: Exit timestamp
            exit_reason: Reason for exit (SIGNAL, STOP_LOSS, TAKE_PROFIT)

        Returns:
            Trade dictionary with P&L info, or None if no position
        """
        if symbol not in self.positions:
            logger.warning(f"No position to close for {symbol}")
            return None

        position = self.positions[symbol]

        # Calculate P&L
        pnl_price = position.get_unrealized_pnl(exit_price)
        pnl_pct = position.get_unrealized_pnl_pct(exit_price)

        # Duration
        duration = (exit_time - position.entry_time).total_seconds() / 3600  # hours

        # Create trade record
        trade = {
            'symbol': symbol,
            'direction': position.direction,
            'entry_time': position.entry_time,
            'exit_time': exit_time,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'position_size': position.position_size,
            'pnl_price': pnl_price,
            'pnl_pct': pnl_pct,
            'duration_hours': duration,
            'exit_reason': exit_reason,
        }

        self.closed_trades.append(trade)

        # Update cash (simplified - actual calculation would include pip values)
        self.cash += pnl_price

        # Remove position
        del self.positions[symbol]

        logger.debug(
            f"Closed {position.direction} position: {symbol} @ {exit_price:.5f}, "
            f"P&L={pnl_pct*100:.2f}%"
        )

        return trade

    def update_equity(self, current_prices: Dict[str, float], timestamp: pd.Timestamp) -> None:
        """
        Update portfolio equity based on current prices.

        Args:
            current_prices: Dictionary of {symbol: price}
            timestamp: Current timestamp
        """
        # Calculate unrealized P&L
        unrealized_pnl = 0.0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                unrealized_pnl += position.get_unrealized_pnl(current_prices[symbol])

        # Total equity = cash + unrealized P&L
        self.equity = self.cash + unrealized_pnl

        # Record equity
        self.equity_history.append({
            'timestamp': timestamp,
            'equity': self.equity,
            'cash': self.cash,
            'unrealized_pnl': unrealized_pnl,
            'num_positions': len(self.positions),
        })

    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get equity curve as DataFrame.

        Returns:
            DataFrame with timestamp, equity, cash, unrealized_pnl
        """
        if not self.equity_history:
            return pd.DataFrame()

        df = pd.DataFrame(self.equity_history)
        df.set_index('timestamp', inplace=True)
        return df

    def get_trades(self) -> pd.DataFrame:
        """
        Get all closed trades as DataFrame.

        Returns:
            DataFrame with trade details
        """
        if not self.closed_trades:
            return pd.DataFrame()

        return pd.DataFrame(self.closed_trades)

    def get_total_return(self) -> float:
        """
        Get total portfolio return.

        Returns:
            Return as fraction (e.g., 0.15 = 15%)
        """
        return (self.equity - self.initial_capital) / self.initial_capital

    def get_num_trades(self) -> int:
        """Get number of closed trades."""
        return len(self.closed_trades)

    def reset(self, initial_capital: Optional[float] = None) -> None:
        """
        Reset portfolio to initial state.

        Args:
            initial_capital: New initial capital (optional, defaults to original)
        """
        if initial_capital is not None:
            self.initial_capital = initial_capital

        self.cash = self.initial_capital
        self.equity = self.initial_capital
        self.positions = {}
        self.closed_trades = []
        self.equity_history = []

        logger.info(f"Portfolio reset: Capital=${self.initial_capital:.2f}")

    def __repr__(self) -> str:
        return (
            f"Portfolio(equity=${self.equity:.2f}, "
            f"positions={len(self.positions)}, "
            f"trades={len(self.closed_trades)}, "
            f"return={self.get_total_return()*100:.2f}%)"
        )
