"""
Vectorized Backtesting Engine

Fast, vectorized backtesting using NumPy/Pandas for Forex trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from loguru import logger

from .transaction_costs import TransactionCostModel
from .risk_manager import RiskManager
from .portfolio import Portfolio


class BacktestEngine:
    """
    Vectorized backtesting engine for Forex strategies.

    Key features:
    - Vectorized operations (no loops) for speed
    - Realistic transaction costs (spreads, slippage, swaps)
    - Risk management (position sizing, stops, drawdown limits)
    - Trade-by-trade tracking
    """

    def __init__(
        self,
        initial_capital: float,
        risk_per_trade: float = 0.01,
        max_drawdown: float = 0.15,
        transaction_cost_model: Optional[TransactionCostModel] = None,
    ):
        """
        Initialize backtesting engine.

        Args:
            initial_capital: Starting capital in USD
            risk_per_trade: Risk per trade as fraction of capital (default: 1%)
            max_drawdown: Maximum drawdown before stopping (default: 15%)
            transaction_cost_model: Custom transaction cost model (optional)
        """
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_drawdown = max_drawdown

        # Initialize components
        self.cost_model = transaction_cost_model or TransactionCostModel()
        self.risk_manager = RiskManager(
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade,
            max_drawdown=max_drawdown,
        )
        self.portfolio = Portfolio(initial_capital=initial_capital)

        logger.info(
            f"Initialized backtesting engine: "
            f"Capital=${initial_capital}, Risk={risk_per_trade*100}%, "
            f"MaxDD={max_drawdown*100}%"
        )

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        symbol: str,
        stop_loss_pct: Optional[pd.Series] = None,
        take_profit_pct: Optional[pd.Series] = None,
    ) -> Dict:
        """
        Run vectorized backtest on data with given signals.

        Args:
            data: OHLCV DataFrame with DatetimeIndex
            signals: Series with values {-1, 0, 1} for short/flat/long
                    Must have same index as data
            symbol: Trading symbol (e.g., "EURUSD")
            stop_loss_pct: Optional stop-loss as percentage (e.g., 0.02 for 2%)
            take_profit_pct: Optional take-profit as percentage (e.g., 0.03 for 3%)

        Returns:
            Dictionary with backtest results:
                - equity_curve: DataFrame with equity over time
                - trades: DataFrame with all trades
                - metrics: Dictionary of performance metrics
                - portfolio: Final portfolio state
        """
        logger.info(f"Running backtest on {symbol}: {len(data)} candles")

        # Validate inputs
        self._validate_inputs(data, signals)

        # Reset portfolio
        self.portfolio.reset(self.initial_capital)

        # Prepare data
        df = data.copy()
        df['signal'] = signals

        # Detect signal changes (entries/exits)
        df['position'] = df['signal'].fillna(0)
        df['position_change'] = df['position'].diff().fillna(0)

        # Identify entry and exit points
        # IMPORTANT: Entry happens on NEXT bar after signal change (can't enter same bar as exit)
        df['entry'] = (df['position_change'] != 0) & (df['position'] != 0)
        # Shift entry to next bar using iloc to avoid fillna warning
        entry_shifted = pd.Series(False, index=df.index)
        entry_shifted.iloc[:-1] = df['entry'].iloc[1:].values
        df['entry'] = entry_shifted
        df['exit'] = (df['position_change'] != 0) & (df['position'].shift(1) != 0)

        # Calculate position sizes
        df['position_size'] = self._calculate_position_sizes(
            df, symbol, stop_loss_pct
        )

        # Calculate returns per bar
        df['price_return'] = df['close'].pct_change()
        df['strategy_return'] = df['position'].shift(1) * df['price_return']

        # Apply transaction costs
        df = self._apply_transaction_costs(df, symbol)

        # Apply stop-loss and take-profit
        if stop_loss_pct is not None or take_profit_pct is not None:
            df = self._apply_stops(df, stop_loss_pct, take_profit_pct)

        # Calculate equity curve
        df['gross_return'] = df['strategy_return']
        df['net_return'] = df['strategy_return'] - df['transaction_costs']

        # Fill NaN returns on first bar with 0 to initialize equity properly
        df['net_return'] = df['net_return'].fillna(0)
        df['equity'] = self.initial_capital * (1 + df['net_return']).cumprod()

        # Check for drawdown breaches
        df = self._check_drawdown_limit(df)

        # Extract trades
        trades = self._extract_trades(df, symbol)

        # Calculate metrics
        metrics = self._calculate_metrics(df, trades)

        logger.info(
            f"Backtest complete: {len(trades)} trades, "
            f"Return={metrics['total_return']*100:.2f}%, "
            f"Sharpe={metrics['sharpe_ratio']:.2f}"
        )

        return {
            'equity_curve': df[['equity', 'position', 'net_return']],
            'trades': trades,
            'metrics': metrics,
            'portfolio': self.portfolio,
            'full_data': df,  # Include for debugging
        }

    def _validate_inputs(self, data: pd.DataFrame, signals: pd.Series) -> None:
        """Validate input data and signals."""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")

        if len(data) != len(signals):
            raise ValueError(
                f"Data length ({len(data)}) != signals length ({len(signals)})"
            )

        if not signals.isin([-1, 0, 1]).all():
            raise ValueError("Signals must be -1 (short), 0 (flat), or 1 (long)")

    def _calculate_position_sizes(
        self,
        df: pd.DataFrame,
        symbol: str,
        stop_loss_pct: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Calculate position sizes based on risk management rules.

        Uses fixed fractional position sizing:
        Position size = (Risk per trade * Capital) / Stop loss distance
        """
        # For now, use simplified position sizing
        # Full implementation will use ATR-based stops and pip values

        if stop_loss_pct is not None:
            # Calculate position size based on risk and stop distance
            risk_amount = self.initial_capital * self.risk_per_trade
            position_sizes = risk_amount / (df['close'] * stop_loss_pct)
        else:
            # Fixed position size (simplified for now)
            # This will be improved in risk_manager.py
            position_sizes = pd.Series(1.0, index=df.index)

        return position_sizes

    def _apply_transaction_costs(
        self, df: pd.DataFrame, symbol: str
    ) -> pd.DataFrame:
        """Apply transaction costs (spreads, slippage, swaps) to returns."""
        # Calculate cost per trade (as fraction of capital)
        # Cost = (spread + slippage) * position_size / capital

        # Simplified cost calculation
        # Full implementation in transaction_costs.py
        cost_per_trade = self.cost_model.get_trade_cost(symbol)

        # Apply cost on position changes (entries/exits)
        df['transaction_costs'] = 0.0
        df.loc[df['position_change'] != 0, 'transaction_costs'] = cost_per_trade

        return df

    def _apply_stops(
        self,
        df: pd.DataFrame,
        stop_loss_pct: Optional[pd.Series],
        take_profit_pct: Optional[pd.Series],
    ) -> pd.DataFrame:
        """
        Apply stop-loss and take-profit logic.

        This is simplified for vectorized backtesting.
        Full bar-by-bar stop logic would require iteration.
        """
        # For now, we'll implement this in a simplified way
        # Full implementation will be added later if needed
        return df

    def _check_drawdown_limit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check if drawdown limit is breached and stop trading."""
        # Calculate running maximum (peak)
        df['peak'] = df['equity'].expanding().max()

        # Calculate drawdown
        df['drawdown'] = (df['equity'] - df['peak']) / df['peak']

        # Find when drawdown limit is breached
        breach_mask = df['drawdown'] < -self.max_drawdown

        if breach_mask.any():
            breach_idx = breach_mask.idxmax()
            logger.warning(
                f"Drawdown limit breached at {breach_idx}: "
                f"{df.loc[breach_idx, 'drawdown']*100:.2f}%"
            )
            # Zero out returns after breach
            df.loc[breach_idx:, 'net_return'] = 0
            df.loc[breach_idx:, 'position'] = 0

        return df

    def _extract_trades(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Extract individual trades from equity curve."""
        trades = []

        # Find entry points
        entries = df[df['entry']].copy()

        for entry_idx, entry_row in entries.iterrows():
            # Find corresponding exit
            future_df = df.loc[entry_idx:]
            future_exits = future_df[future_df['exit']]

            if len(future_exits) == 0:
                # Position still open at end
                exit_idx = df.index[-1]
                exit_row = df.loc[exit_idx]
            else:
                exit_idx = future_exits.index[0]
                exit_row = df.loc[exit_idx]

            # Calculate trade metrics
            direction = "LONG" if entry_row['position'] > 0 else "SHORT"
            entry_price = entry_row['close']
            exit_price = exit_row['close']

            if direction == "LONG":
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price

            # Get returns for this trade period
            trade_returns = df.loc[entry_idx:exit_idx, 'net_return'].sum()

            trades.append({
                'symbol': symbol,
                'entry_time': entry_idx,
                'exit_time': exit_idx,
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'pnl_gross': trade_returns * self.initial_capital,
                'duration': (exit_idx - entry_idx).total_seconds() / 3600,  # hours
            })

        return pd.DataFrame(trades)

    def _calculate_metrics(
        self, df: pd.DataFrame, trades: pd.DataFrame
    ) -> Dict:
        """Calculate comprehensive performance metrics."""
        # Returns
        total_return = (df['equity'].iloc[-1] / self.initial_capital) - 1

        # Annualized metrics
        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Risk metrics
        returns = df['net_return'].dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        # Drawdown
        max_drawdown = df['drawdown'].min()

        # Trade metrics
        if len(trades) > 0:
            win_rate = (trades['pnl_pct'] > 0).mean()
            avg_win = trades.loc[trades['pnl_pct'] > 0, 'pnl_pct'].mean()
            avg_loss = trades.loc[trades['pnl_pct'] < 0, 'pnl_pct'].mean()
            profit_factor = (
                abs(trades.loc[trades['pnl_pct'] > 0, 'pnl_pct'].sum()) /
                abs(trades.loc[trades['pnl_pct'] < 0, 'pnl_pct'].sum())
                if (trades['pnl_pct'] < 0).any() else float('inf')
            )
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0

        return {
            'total_return': total_return,
            'cagr': cagr,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'final_equity': df['equity'].iloc[-1],
        }
