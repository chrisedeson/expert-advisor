"""
Grid Backtesting Engine

Specialized backtesting engine for grid trading strategies.

Handles:
- Multiple concurrent positions per direction
- Grid level tracking and management
- Dynamic position sizing per grid level
- Global take profit / stop loss across all positions
- Trailing stops on aggregate position
- Risk management with drawdown limits
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger

from .transaction_costs import TransactionCostModel
from .risk_manager import RiskManager


@dataclass
class GridPosition:
    """Represents a single grid position."""
    entry_time: datetime
    entry_price: float
    size: float  # Lot size
    direction: int  # 1 for long, -1 for short
    grid_level: int  # 0 = initial, 1-4 = grid levels
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class GridBacktestingEngine:
    """
    Backtesting engine specifically designed for grid trading strategies.

    Features:
    - Tracks multiple open positions per direction
    - Implements grid logic (spacing, levels, lot scaling)
    - Global TP/SL across all positions in one direction
    - Dynamic position sizing
    - Comprehensive risk management
    """

    def __init__(
        self,
        strategy,
        initial_capital: float = 100.0,
        risk_per_trade: float = 0.01,
        max_drawdown: float = 0.20,
        cost_model: Optional[TransactionCostModel] = None,
    ):
        """
        Initialize grid backtesting engine.

        Args:
            strategy: Grid trading strategy instance
            initial_capital: Starting capital
            risk_per_trade: Risk per trade (0.01 = 1%)
            max_drawdown: Maximum allowed drawdown
            cost_model: Transaction cost model (if None, creates default)
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_drawdown = max_drawdown

        # Initialize cost model and risk manager
        self.cost_model = cost_model or TransactionCostModel()
        self.risk_manager = RiskManager(
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade,
            max_drawdown=max_drawdown,
        )

        # Trading state
        self.balance = initial_capital
        self.equity = initial_capital
        self.peak_equity = initial_capital

        # Grid positions
        self.long_positions: List[GridPosition] = []
        self.short_positions: List[GridPosition] = []

        # Trade history
        self.trades: List[Dict] = []
        self.equity_curve: List[Tuple[datetime, float]] = []

        logger.info(
            f"Initialized grid backtesting engine: Capital=${initial_capital:.2f}, "
            f"Risk={risk_per_trade*100:.1f}%, MaxDD={max_drawdown*100:.0f}%"
        )

    def run(self, data: pd.DataFrame, symbol: str = 'EURUSD') -> Dict:
        """
        Run backtest on historical data.

        Args:
            data: DataFrame with OHLCV data and 'signal' column
            symbol: Trading symbol

        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Running grid backtest on {symbol}: {len(data):,} candles")

        # Generate signals
        df = self.strategy.generate_signals(data)

        # Reset state
        self.balance = self.initial_capital
        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital
        self.long_positions = []
        self.short_positions = []
        self.trades = []
        self.equity_curve = [(df.index[0], self.equity)]

        # Iterate through bars
        for i in range(len(df)):
            current_time = df.index[i]
            current_price = df['close'].iloc[i]
            atr = df['atr'].iloc[i] if 'atr' in df.columns else 0.0010
            signal = df['signal'].iloc[i]

            # Update strategy with current balance
            self.strategy.update_dynamic_sizing(self.balance, current_price, atr)

            # Check if we should open initial position
            if signal != 0:
                self._process_signal(
                    signal, current_time, current_price, atr, df.iloc[i]
                )

            # Check grid levels (add positions if price moved against us)
            self._check_grid_levels(current_time, current_price, atr, df.iloc[i])

            # Check exit conditions (TP/SL)
            self._check_exits(current_time, current_price, df.iloc[i])

            # Update equity
            self._update_equity(current_price)

            # Record equity
            self.equity_curve.append((current_time, self.equity))

            # Check drawdown limit
            drawdown = (self.peak_equity - self.equity) / self.peak_equity
            if drawdown > self.max_drawdown:
                logger.warning(
                    f"Max drawdown exceeded at {current_time}: "
                    f"{drawdown*100:.2f}% > {self.max_drawdown*100:.0f}%"
                )
                # Close all positions
                self._close_all_positions(current_time, current_price, "Max DD Hit")
                break

        # Close any remaining positions at end
        if self.long_positions or self.short_positions:
            final_price = df['close'].iloc[-1]
            final_time = df.index[-1]
            self._close_all_positions(final_time, final_price, "End of Backtest")

        # Calculate final metrics
        total_return = (self.equity - self.initial_capital) / self.initial_capital
        num_trades = len(self.trades)

        logger.info(
            f"Backtest complete: {num_trades} trades, "
            f"Return={total_return*100:.2f}%, Final Equity=${self.equity:.2f}"
        )

        return {
            'equity_curve': pd.DataFrame(
                self.equity_curve, columns=['time', 'equity']
            ).set_index('time'),
            'trades': pd.DataFrame(self.trades),
            'final_equity': self.equity,
            'total_return': total_return,
            'num_trades': num_trades,
        }

    def _process_signal(
        self,
        signal: int,
        time: datetime,
        price: float,
        atr: float,
        bar: pd.Series
    ):
        """Process a new trading signal."""
        direction = signal  # 1 for long, -1 for short

        # Check if we already have positions in this direction
        existing_positions = (
            self.long_positions if direction == 1 else self.short_positions
        )

        # Only open initial position if we don't have any
        if len(existing_positions) == 0:
            self._open_position(direction, time, price, atr, grid_level=0)

    def _check_grid_levels(
        self,
        time: datetime,
        price: float,
        atr: float,
        bar: pd.Series
    ):
        """Check if we should add grid positions."""
        grid_spacing = atr * self.strategy.grid_spacing_atr_mult

        # Check long positions
        if self.long_positions:
            last_long = self.long_positions[-1]
            # If price moved down (against us), add grid position
            if (
                price < last_long.entry_price - grid_spacing
                and len(self.long_positions) < self.strategy.available_grid_levels
            ):
                self._open_position(1, time, price, atr, len(self.long_positions))

        # Check short positions
        if self.short_positions:
            last_short = self.short_positions[-1]
            # If price moved up (against us), add grid position
            if (
                price > last_short.entry_price + grid_spacing
                and len(self.short_positions) < self.strategy.available_grid_levels
            ):
                self._open_position(-1, time, price, atr, len(self.short_positions))

    def _open_position(
        self,
        direction: int,
        time: datetime,
        price: float,
        atr: float,
        grid_level: int
    ):
        """Open a new grid position."""
        # Calculate position size for this grid level
        lot_size = self.strategy.calculate_position_size(self.balance, grid_level)

        # Create position
        position = GridPosition(
            entry_time=time,
            entry_price=price,
            size=lot_size,
            direction=direction,
            grid_level=grid_level,
        )

        # Add to appropriate list
        if direction == 1:
            self.long_positions.append(position)
            logger.debug(
                f"Opened LONG grid level {grid_level} @ {price:.5f}, "
                f"size={lot_size:.2f}, total_longs={len(self.long_positions)}"
            )
        else:
            self.short_positions.append(position)
            logger.debug(
                f"Opened SHORT grid level {grid_level} @ {price:.5f}, "
                f"size={lot_size:.2f}, total_shorts={len(self.short_positions)}"
            )

    def _check_exits(self, time: datetime, price: float, bar: pd.Series):
        """Check if any positions should be closed."""
        # Check long positions
        if self.long_positions:
            self._check_long_exit(time, price, bar)

        # Check short positions
        if self.short_positions:
            self._check_short_exit(time, price, bar)

    def _check_long_exit(self, time: datetime, price: float, bar: pd.Series):
        """Check if long positions should be closed."""
        if not self.long_positions:
            return

        # Calculate weighted average entry price
        total_lots = sum(p.size for p in self.long_positions)
        avg_entry = sum(p.entry_price * p.size for p in self.long_positions) / total_lots

        # Calculate take profit level
        atr = bar['atr'] if 'atr' in bar else 0.0010
        take_profit = avg_entry + (atr * self.strategy.take_profit_atr_mult)

        # Calculate stop loss level (from first position)
        stop_loss = self.long_positions[0].entry_price - (atr * 2.0)

        # Check TP
        if price >= take_profit:
            self._close_all_longs(time, price, "Take Profit")

        # Check SL
        elif price <= stop_loss:
            self._close_all_longs(time, price, "Stop Loss")

    def _check_short_exit(self, time: datetime, price: float, bar: pd.Series):
        """Check if short positions should be closed."""
        if not self.short_positions:
            return

        # Calculate weighted average entry price
        total_lots = sum(p.size for p in self.short_positions)
        avg_entry = sum(p.entry_price * p.size for p in self.short_positions) / total_lots

        # Calculate take profit level
        atr = bar['atr'] if 'atr' in bar else 0.0010
        take_profit = avg_entry - (atr * self.strategy.take_profit_atr_mult)

        # Calculate stop loss level (from first position)
        stop_loss = self.short_positions[0].entry_price + (atr * 2.0)

        # Check TP
        if price <= take_profit:
            self._close_all_shorts(time, price, "Take Profit")

        # Check SL
        elif price >= stop_loss:
            self._close_all_shorts(time, price, "Stop Loss")

    def _close_all_longs(self, time: datetime, price: float, reason: str):
        """Close all long positions."""
        total_pnl = 0.0
        total_cost = 0.0

        for position in self.long_positions:
            # Calculate P&L
            pnl = (price - position.entry_price) * position.size * 100000  # Convert to dollars
            cost = self.cost_model.calculate_total_cost(position.size, price)
            net_pnl = pnl - cost

            total_pnl += pnl
            total_cost += cost

            # Record trade
            self.trades.append({
                'entry_time': position.entry_time,
                'exit_time': time,
                'direction': 'LONG',
                'entry_price': position.entry_price,
                'exit_price': price,
                'size': position.size,
                'grid_level': position.grid_level,
                'pnl': pnl,
                'cost': cost,
                'net_pnl': net_pnl,
                'exit_reason': reason,
            })

        # Update balance
        net_total_pnl = total_pnl - total_cost
        self.balance += net_total_pnl

        logger.debug(
            f"Closed {len(self.long_positions)} LONG positions @ {price:.5f}: "
            f"Net P&L=${net_total_pnl:.2f}, Reason={reason}"
        )

        # Clear positions
        self.long_positions = []

    def _close_all_shorts(self, time: datetime, price: float, reason: str):
        """Close all short positions."""
        total_pnl = 0.0
        total_cost = 0.0

        for position in self.short_positions:
            # Calculate P&L
            pnl = (position.entry_price - price) * position.size * 100000  # Convert to dollars
            cost = self.cost_model.calculate_total_cost(position.size, price)
            net_pnl = pnl - cost

            total_pnl += pnl
            total_cost += cost

            # Record trade
            self.trades.append({
                'entry_time': position.entry_time,
                'exit_time': time,
                'direction': 'SHORT',
                'entry_price': position.entry_price,
                'exit_price': price,
                'size': position.size,
                'grid_level': position.grid_level,
                'pnl': pnl,
                'cost': cost,
                'net_pnl': net_pnl,
                'exit_reason': reason,
            })

        # Update balance
        net_total_pnl = total_pnl - total_cost
        self.balance += net_total_pnl

        logger.debug(
            f"Closed {len(self.short_positions)} SHORT positions @ {price:.5f}: "
            f"Net P&L=${net_total_pnl:.2f}, Reason={reason}"
        )

        # Clear positions
        self.short_positions = []

    def _close_all_positions(self, time: datetime, price: float, reason: str):
        """Close all open positions."""
        if self.long_positions:
            self._close_all_longs(time, price, reason)
        if self.short_positions:
            self._close_all_shorts(time, price, reason)

    def _update_equity(self, current_price: float):
        """Update current equity based on open positions."""
        # Start with balance
        equity = self.balance

        # Add unrealized P&L from long positions
        for position in self.long_positions:
            unrealized_pnl = (current_price - position.entry_price) * position.size * 100000
            equity += unrealized_pnl

        # Add unrealized P&L from short positions
        for position in self.short_positions:
            unrealized_pnl = (position.entry_price - current_price) * position.size * 100000
            equity += unrealized_pnl

        self.equity = equity

        # Update peak equity
        if equity > self.peak_equity:
            self.peak_equity = equity
