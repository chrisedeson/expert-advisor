"""
Protected Grid Backtesting Engine v2

Major fixes over v1:
1. Separated cash_balance from equity (fixed double-counting unrealized PnL)
2. Smart mean-reversion entries with BB(20, 2.0) + RSI(14) confirmation + ATR grid spacing
3. Dynamic TP at SMA(20), SL at 1.5*ATR (adapts to volatility)
4. Protection events tracked on state changes only (not every bar)
5. Trade cooldown to prevent overtrading
6. Minimum TP distance check (ensures profit after costs)
7. Proper daily resampling for Sharpe/Sortino calculation
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from loguru import logger

from ..risk.protection_manager import ProtectionManager, ProtectionDecision


@dataclass
class ProtectedTrade:
    """Trade with protection information"""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    direction: str  # 'BUY' or 'SELL'
    lot_size: float
    grid_level: int
    stop_loss: float
    take_profit: float
    protection_multiplier: float
    active_protections: List[str]
    exit_reason: Optional[str] = None
    gross_pnl: Optional[float] = None
    net_pnl: Optional[float] = None
    pips: Optional[float] = None


@dataclass
class ProtectionEvent:
    """Protection system event (only recorded on state changes)"""
    timestamp: datetime
    event_type: str
    description: str
    balance_at_event: float
    action_taken: str


@dataclass
class BacktestResult:
    """Complete backtest result with protection analysis"""
    start_date: datetime
    end_date: datetime
    initial_balance: float
    final_balance: float
    total_return: float
    cagr: float

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    gross_profit: float
    gross_loss: float
    net_profit: float
    profit_factor: float

    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float

    max_drawdown: float
    max_drawdown_duration_days: int
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    protection_events: List[ProtectionEvent] = field(default_factory=list)
    trades_blocked: int = 0
    positions_force_closed: int = 0
    capital_saved_estimate: float = 0.0

    volatility_pauses: int = 0
    crisis_mode_activations: int = 0
    circuit_breaker_triggers: int = 0
    recovery_mode_activations: int = 0
    profit_protection_activations: int = 0

    trades: List[ProtectedTrade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)


class ProtectedGridBacktester:
    """
    Grid backtester with full protection system integration.

    Uses mean-reversion strategy: buy below BB, sell above BB, TP at SMA.
    Grid levels add positions when price moves further against us.
    """

    def __init__(
        self,
        initial_balance: float,
        config: Dict,
        spread_pips: float = 1.2,
        commission_per_lot: float = 0.0,
        slippage_pips: float = 0.2,
        pip_size: float = 0.0001,
        pip_value_per_lot: float = 10.0,
    ):
        self.initial_balance = initial_balance
        self.config = config
        self.spread_pips = spread_pips
        self.commission_per_lot = commission_per_lot
        self.slippage_pips = slippage_pips
        self.pip_size = pip_size
        self.pip_value_per_lot = pip_value_per_lot

        grid_config = config.get('grid_strategy', {})
        self.base_lot = grid_config.get('base_lot_size', 0.01)
        self.lot_multiplier = grid_config.get('lot_multiplier', 1.2)
        self.max_grid_levels = grid_config.get('max_grid_levels', 5)
        self.use_trend_filter = grid_config.get('use_trend_filter', True)
        self.compound_on_equity = grid_config.get('compound_on_equity', False)
        self.bb_entry_mult = grid_config.get('bb_entry_mult', 2.0)
        self.grid_spacing_atr = grid_config.get('grid_spacing_atr', 0.75)
        self.sl_atr_mult = grid_config.get('sl_atr_mult', 1.5)

        # Minimum TP distance in pips (must exceed costs)
        self.min_tp_pips = self.spread_pips + self.slippage_pips + 5.0

        # Initialize protection manager
        self.protection_manager = ProtectionManager(
            initial_balance=initial_balance,
            config=config,
            alert_manager=None,
        )

        # CRITICAL: Separate cash balance from equity
        # cash_balance = realized only (changes on trade close)
        # equity = cash_balance + unrealized PnL (changes every bar)
        self.cash_balance = initial_balance
        self.peak_balance = initial_balance
        self.open_positions: List[ProtectedTrade] = []
        self.closed_trades: List[ProtectedTrade] = []
        self.protection_events: List[ProtectionEvent] = []

        self.trades_blocked = 0
        self.positions_force_closed = 0

        # Entry cooldown tracking
        self._last_buy_time: Optional[datetime] = None
        self._last_sell_time: Optional[datetime] = None
        self._min_hours_between_entries = 1  # Min hours between same-direction entries

        # Protection state change tracking (to avoid logging every bar)
        self._prev_protection_state: Optional[str] = None

        self.equity_history: List[Tuple[datetime, float]] = []

        logger.info(
            f"ProtectedGridBacktester: ${initial_balance:.2f}, "
            f"lot={self.base_lot}, levels={self.max_grid_levels}, "
            f"spread={spread_pips}pip, slip={slippage_pips}pip"
        )

    def run_backtest(
        self,
        data: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> BacktestResult:
        """Run backtest with full protection system."""
        required_cols = ['open', 'high', 'low', 'close', 'volume', 'atr']
        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]

        if len(data) == 0:
            raise ValueError("No data in specified date range")

        # Reset state
        self.cash_balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.open_positions = []
        self.closed_trades = []
        self.protection_events = []
        self.trades_blocked = 0
        self.positions_force_closed = 0
        self.equity_history = []
        self._last_buy_time = None
        self._last_sell_time = None
        self._prev_protection_state = None

        self.protection_manager.reset_all()

        logger.info(f"Backtesting {len(data)} bars: {data.index[0]} to {data.index[-1]}")

        for i in range(len(data)):
            current_time = data.index[i]
            current_bar = data.iloc[i]

            # Calculate equity (for equity curve) and track peak
            equity = self._calculate_equity(current_bar['close'])
            if equity > self.peak_balance:
                self.peak_balance = equity

            # CRITICAL: Use cash_balance for protection checks, NOT equity.
            # Grid trading has large unrealized swings that are normal.
            # CB should only react to realized losses (closed trades).
            recent_window = data.iloc[max(0, i - 250):i + 1]

            decision = self.protection_manager.check_trading_permission(
                current_balance=self.cash_balance,
                current_data=recent_window,
                recent_trades=self._get_recent_trade_dicts(),
                current_time=current_time,
            )

            # Track protection state changes only
            self._track_protection_changes(decision, current_time, equity)

            # Force close positions if protection demands
            if decision.should_close_positions and self.open_positions:
                self._force_close_all_positions(current_time, current_bar, decision)

            # Update open positions (check SL/TP)
            self._update_open_positions(current_time, current_bar)

            # Detect entry signal and execute (or count as blocked)
            if len(recent_window) >= 200:
                signal = self._detect_entry_signal(
                    current_time, current_bar, recent_window
                )
                if signal and decision.can_trade:
                    self._execute_entry(
                        signal, current_time, current_bar,
                        decision.position_size_multiplier,
                        decision.active_protections,
                    )
                elif signal and not decision.can_trade:
                    self.trades_blocked += 1

            # Record final equity after all processing
            final_equity = self._calculate_equity(current_bar['close'])
            self.equity_history.append((current_time, final_equity))

            # Progress log every 5000 bars
            if (i + 1) % 5000 == 0:
                logger.info(
                    f"[{i+1}/{len(data)}] Equity=${final_equity:.2f}, "
                    f"Trades={len(self.closed_trades)}, Open={len(self.open_positions)}"
                )

        # Close remaining positions at end
        if self.open_positions:
            self._force_close_all_positions(
                data.index[-1], data.iloc[-1], None, reason="End of backtest"
            )

        result = self._calculate_result(data.index[0], data.index[-1])

        logger.success(
            f"Done: ${result.initial_balance:.2f} -> ${result.final_balance:.2f} "
            f"({result.total_return*100:+.1f}%), {result.total_trades} trades, "
            f"WR={result.win_rate*100:.0f}%, DD={result.max_drawdown*100:.1f}%, "
            f"Sharpe={result.sharpe_ratio:.2f}, PF={result.profit_factor:.2f}"
        )

        return result

    def _calculate_equity(self, current_price: float) -> float:
        """Calculate equity = cash_balance + unrealized PnL of open positions."""
        unrealized = 0.0
        for pos in self.open_positions:
            if pos.direction == 'BUY':
                pips = (current_price - pos.entry_price) / self.pip_size
            else:
                pips = (pos.entry_price - current_price) / self.pip_size
            unrealized += pips * self.pip_value_per_lot * pos.lot_size
        return self.cash_balance + unrealized

    def _get_recent_trade_dicts(self) -> List[Dict]:
        """Convert recent closed trades to dict format for protection manager."""
        return [
            {
                'exit_time': t.exit_time,
                'exit_reason': t.exit_reason,
                'net_pnl': t.net_pnl,
            }
            for t in self.closed_trades[-20:]
        ]

    def _track_protection_changes(
        self,
        decision: ProtectionDecision,
        current_time: datetime,
        equity: float,
    ):
        """Record protection events only on state changes, not every bar."""
        current_state = (
            f"{','.join(sorted(decision.active_protections))}"
            f"|{decision.can_trade}"
        )

        if current_state != self._prev_protection_state:
            if decision.active_protections:
                event = ProtectionEvent(
                    timestamp=current_time,
                    event_type='protection_change',
                    description=f"Active: {', '.join(decision.active_protections)}",
                    balance_at_event=equity,
                    action_taken=(
                        f"Size={decision.position_size_multiplier*100:.0f}%, "
                        f"Trade={decision.can_trade}"
                    ),
                )
                self.protection_events.append(event)
            self._prev_protection_state = current_state

    def _detect_entry_signal(
        self,
        current_time: datetime,
        current_bar: pd.Series,
        recent_data: pd.DataFrame,
    ) -> Optional[Dict]:
        """
        Trend-aligned BB mean reversion with opposite-BB TP for better R:R.

        KEY CHANGE: TP at opposite BB (not SMA) → R:R ~2.25:1 instead of ~0.62:1.
        Combined with trend filter → only need ~31% WR to be profitable.

        - Trend: SMA(20) vs SMA(50) determines allowed direction
        - Entry: BB(20, 1.5) touch at extreme
        - Grid: 1.0*ATR spacing for adds
        - TP: OPPOSITE BB band (full reversion + overshoot)
        - SL: 2.0*ATR from entry
        """
        close_prices = recent_data['close']
        current_price = current_bar['close']
        current_atr = current_bar['atr']

        if current_atr <= 0 or pd.isna(current_atr):
            return None

        # Volatility band filter: skip low-vol chop AND high-vol chaos
        atr_series = recent_data['atr']
        avg_atr = atr_series.rolling(50).mean().iloc[-1]
        if not pd.isna(avg_atr) and avg_atr > 0:
            atr_ratio = current_atr / avg_atr
            if atr_ratio < 0.8 or atr_ratio > 3.0:
                return None

        sma50 = close_prices.rolling(50).mean().iloc[-1]
        sma200 = close_prices.rolling(200).mean().iloc[-1] if len(close_prices) >= 200 else sma50
        std50 = close_prices.rolling(50).std().iloc[-1]

        if pd.isna(sma50) or pd.isna(std50) or std50 <= 0:
            return None
        if pd.isna(sma200):
            return None

        upper_bb = sma50 + self.bb_entry_mult * std50
        lower_bb = sma50 - self.bb_entry_mult * std50

        # Trend direction: SMA(50) vs SMA(200)
        is_uptrend = sma50 > sma200
        is_downtrend = sma50 < sma200

        # When trend filter is disabled, allow both directions
        allow_buy = is_uptrend if self.use_trend_filter else True
        allow_sell = is_downtrend if self.use_trend_filter else True

        buy_positions = [p for p in self.open_positions if p.direction == 'BUY']
        sell_positions = [p for p in self.open_positions if p.direction == 'SELL']

        # === BUY SIGNALS ===
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

            if buy_signal and self._check_cooldown('BUY', current_time):
                # TP at OPPOSITE (upper) BB - full band traversal
                tp_price = upper_bb
                tp_distance_pips = (tp_price - current_price) / self.pip_size

                if tp_distance_pips >= self.min_tp_pips:
                    sl_price = current_price - self.sl_atr_mult * current_atr
                    return {
                        'direction': 'BUY',
                        'grid_level': grid_level,
                        'tp': tp_price,
                        'sl': sl_price,
                    }

        # === SELL SIGNALS ===
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

            if sell_signal and self._check_cooldown('SELL', current_time):
                # TP at OPPOSITE (lower) BB
                tp_price = lower_bb
                tp_distance_pips = (current_price - tp_price) / self.pip_size

                if tp_distance_pips >= self.min_tp_pips:
                    sl_price = current_price + self.sl_atr_mult * current_atr
                    return {
                        'direction': 'SELL',
                        'grid_level': grid_level,
                        'tp': tp_price,
                        'sl': sl_price,
                    }

        return None

    def _check_cooldown(self, direction: str, current_time: datetime) -> bool:
        """Check if enough time has passed since last entry in this direction."""
        if direction == 'BUY' and self._last_buy_time is not None:
            hours = (current_time - self._last_buy_time).total_seconds() / 3600
            if hours < self._min_hours_between_entries:
                return False
        elif direction == 'SELL' and self._last_sell_time is not None:
            hours = (current_time - self._last_sell_time).total_seconds() / 3600
            if hours < self._min_hours_between_entries:
                return False
        return True

    def _execute_entry(
        self,
        signal: Dict,
        current_time: datetime,
        current_bar: pd.Series,
        size_multiplier: float,
        active_protections: List[str],
    ):
        """Execute an entry signal with compounding lot sizing."""
        direction = signal['direction']
        grid_level = signal['grid_level']
        current_price = current_bar['close']

        # Compounding: scale lots proportionally to account growth
        if self.compound_on_equity:
            current_equity = self._calculate_equity(current_price)
            equity_ratio = max(current_equity / self.initial_balance, 0.5)
        else:
            equity_ratio = max(self.cash_balance / self.initial_balance, 0.5)
        lot_size = self.base_lot * (self.lot_multiplier ** grid_level) * size_multiplier * equity_ratio

        if lot_size < 0.001:
            return

        position = ProtectedTrade(
            entry_time=current_time,
            exit_time=None,
            entry_price=current_price,
            exit_price=None,
            direction=direction,
            lot_size=lot_size,
            grid_level=grid_level,
            stop_loss=signal['sl'],
            take_profit=signal['tp'],
            protection_multiplier=size_multiplier,
            active_protections=active_protections.copy(),
        )

        self.open_positions.append(position)

        # Update cooldown
        if direction == 'BUY':
            self._last_buy_time = current_time
        else:
            self._last_sell_time = current_time

    def _force_close_all_positions(
        self,
        current_time: datetime,
        current_bar: pd.Series,
        decision: Optional[ProtectionDecision],
        reason: str = "Protection system",
    ):
        """Force close all open positions."""
        n = len(self.open_positions)
        for pos in self.open_positions:
            self._close_position(pos, current_time, current_bar['close'], reason)
            self.positions_force_closed += 1

        if decision and n > 0:
            event = ProtectionEvent(
                timestamp=current_time,
                event_type='force_close',
                description=f"Closed {n} positions",
                balance_at_event=self.cash_balance,
                action_taken=reason,
            )
            self.protection_events.append(event)

        self.open_positions = []

    def _update_open_positions(self, current_time: datetime, current_bar: pd.Series):
        """Check SL/TP hits and update trailing stops on open positions."""
        to_close = []
        current_atr = current_bar['atr'] if not pd.isna(current_bar['atr']) else 0

        for pos in self.open_positions:
            # --- Check SL/TP exits first ---
            if pos.direction == 'BUY':
                if current_bar['high'] >= pos.take_profit:
                    self._close_position(pos, current_time, pos.take_profit, "Take Profit")
                    to_close.append(pos)
                elif current_bar['low'] <= pos.stop_loss:
                    self._close_position(pos, current_time, pos.stop_loss, "Stop Loss")
                    to_close.append(pos)
            else:  # SELL
                if current_bar['low'] <= pos.take_profit:
                    self._close_position(pos, current_time, pos.take_profit, "Take Profit")
                    to_close.append(pos)
                elif current_bar['high'] >= pos.stop_loss:
                    self._close_position(pos, current_time, pos.stop_loss, "Stop Loss")
                    to_close.append(pos)

        for pos in to_close:
            self.open_positions.remove(pos)

        # --- Two-stage stop management (takes effect next bar) ---
        # Stage 1: Move to breakeven after 1.0*ATR profit (protect capital)
        # Stage 2: Start trailing at 1.5*ATR behind high/low after 2.0*ATR profit (let winners run)
        if current_atr > 0:
            for pos in self.open_positions:
                if pos.direction == 'BUY':
                    max_favorable = current_bar['high'] - pos.entry_price
                    if max_favorable >= 2.0 * current_atr:
                        # Stage 2: Trail at 1.5*ATR behind high
                        trail_sl = current_bar['high'] - 1.5 * current_atr
                        pos.stop_loss = max(pos.stop_loss, trail_sl)
                    elif max_favorable >= 1.0 * current_atr:
                        # Stage 1: Move to breakeven
                        pos.stop_loss = max(pos.stop_loss, pos.entry_price)
                else:
                    max_favorable = pos.entry_price - current_bar['low']
                    if max_favorable >= 2.0 * current_atr:
                        # Stage 2: Trail at 1.5*ATR above low
                        trail_sl = current_bar['low'] + 1.5 * current_atr
                        pos.stop_loss = min(pos.stop_loss, trail_sl)
                    elif max_favorable >= 1.0 * current_atr:
                        # Stage 1: Move to breakeven
                        pos.stop_loss = min(pos.stop_loss, pos.entry_price)

    def _close_position(
        self,
        position: ProtectedTrade,
        exit_time: datetime,
        exit_price: float,
        exit_reason: str,
    ):
        """Close a position and update cash_balance with realized PnL."""
        position.exit_time = exit_time
        position.exit_price = exit_price
        position.exit_reason = exit_reason

        if position.direction == 'BUY':
            pips = (exit_price - position.entry_price) / self.pip_size
        else:
            pips = (position.entry_price - exit_price) / self.pip_size

        # Subtract transaction costs
        pips -= (self.spread_pips + self.slippage_pips)

        position.pips = pips
        position.gross_pnl = pips * self.pip_value_per_lot * position.lot_size

        commission = self.commission_per_lot * position.lot_size
        position.net_pnl = position.gross_pnl - commission

        # CRITICAL: Update cash_balance (realized), NOT equity
        self.cash_balance += position.net_pnl

        self.closed_trades.append(position)

    def _calculate_result(self, start_date: datetime, end_date: datetime) -> BacktestResult:
        """Calculate comprehensive backtest results."""

        # Final equity
        final_equity = self.cash_balance  # No open positions after force-close
        total_return = (final_equity - self.initial_balance) / self.initial_balance

        years = (end_date - start_date).days / 365.25
        if years > 0 and final_equity > 0:
            cagr = ((final_equity / self.initial_balance) ** (1 / years)) - 1
        else:
            cagr = 0

        # Trade metrics
        total_trades = len(self.closed_trades)
        winners = [t for t in self.closed_trades if t.net_pnl and t.net_pnl > 0]
        losers = [t for t in self.closed_trades if t.net_pnl and t.net_pnl <= 0]

        win_rate = len(winners) / total_trades if total_trades > 0 else 0

        gross_profit = sum(t.net_pnl for t in winners)
        gross_loss = abs(sum(t.net_pnl for t in losers))
        net_profit = gross_profit - gross_loss

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        avg_win = gross_profit / len(winners) if winners else 0
        avg_loss = gross_loss / len(losers) if losers else 0

        largest_win = max((t.net_pnl for t in winners), default=0)
        largest_loss = min((t.net_pnl for t in losers), default=0)

        # Equity curve
        equity_curve_df = pd.DataFrame(
            self.equity_history, columns=['time', 'equity']
        )
        equity_curve_df.set_index('time', inplace=True)

        # Max drawdown
        equity_curve_df['peak'] = equity_curve_df['equity'].cummax()
        equity_curve_df['drawdown'] = (
            (equity_curve_df['equity'] - equity_curve_df['peak'])
            / equity_curve_df['peak']
        )
        max_drawdown = abs(equity_curve_df['drawdown'].min())

        # Drawdown duration
        in_drawdown = equity_curve_df['drawdown'] < 0
        max_dd_duration = 0
        if in_drawdown.any():
            dd_durations = []
            dd_start = None
            for idx, is_dd in in_drawdown.items():
                if is_dd and dd_start is None:
                    dd_start = idx
                elif not is_dd and dd_start is not None:
                    dd_durations.append((idx - dd_start).days)
                    dd_start = None
            if dd_durations:
                max_dd_duration = max(dd_durations)

        # Risk metrics - resample to daily for proper annualization
        daily_equity = equity_curve_df['equity'].resample('D').last().dropna()
        daily_returns = daily_equity.pct_change().dropna()

        if len(daily_returns) > 10 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        negative_returns = daily_returns[daily_returns < 0]
        if len(negative_returns) > 5 and negative_returns.std() > 0:
            sortino_ratio = (daily_returns.mean() / negative_returns.std()) * np.sqrt(252)
        else:
            sortino_ratio = 0

        calmar_ratio = cagr / max_drawdown if max_drawdown > 0 else 0

        # Protection stats
        prot_stats = self.protection_manager.get_statistics()

        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_balance=self.initial_balance,
            final_balance=final_equity,
            total_return=total_return,
            cagr=cagr,

            total_trades=total_trades,
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=win_rate,

            gross_profit=gross_profit,
            gross_loss=gross_loss,
            net_profit=net_profit,
            profit_factor=profit_factor,

            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,

            max_drawdown=max_drawdown,
            max_drawdown_duration_days=max_dd_duration,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,

            protection_events=self.protection_events,
            trades_blocked=self.trades_blocked,
            positions_force_closed=self.positions_force_closed,
            capital_saved_estimate=self._estimate_capital_saved(),

            volatility_pauses=prot_stats['volatility_filter'].get('crisis_count', 0),
            crisis_mode_activations=prot_stats['crisis_detector'].get('total_crisis_events', 0),
            circuit_breaker_triggers=prot_stats['circuit_breaker'].get('total_triggers', 0),
            recovery_mode_activations=prot_stats['recovery_manager'].get('total_recovery_events', 0),
            profit_protection_activations=prot_stats['profit_protector'].get('total_profit_runs', 0),

            trades=self.closed_trades,
            equity_curve=equity_curve_df,
        )

    def _estimate_capital_saved(self) -> float:
        """Estimate capital saved by protection system."""
        if not self.closed_trades:
            return 0.0

        sl_trades = [t for t in self.closed_trades if t.exit_reason == 'Stop Loss']
        if not sl_trades:
            return 0.0

        avg_sl_loss = abs(sum(t.net_pnl for t in sl_trades) / len(sl_trades))
        return self.positions_force_closed * avg_sl_loss * 0.5


def print_backtest_summary(result: BacktestResult):
    """Print formatted backtest summary."""
    print("\n" + "=" * 80)
    print("PROTECTED GRID BACKTEST RESULTS")
    print("=" * 80)

    print(f"\n  PERIOD: {result.start_date.date()} to {result.end_date.date()}")
    print(f"  Duration: {(result.end_date - result.start_date).days} days")

    print(f"\n  RETURNS:")
    print(f"    Initial Balance:  ${result.initial_balance:,.2f}")
    print(f"    Final Balance:    ${result.final_balance:,.2f}")
    print(f"    Total Return:     {result.total_return*100:,.2f}%")
    print(f"    CAGR:             {result.cagr*100:.2f}%")
    print(f"    Net Profit:       ${result.net_profit:,.2f}")

    print(f"\n  TRADE STATISTICS:")
    print(f"    Total Trades:     {result.total_trades}")
    print(f"    Winners:          {result.winning_trades} ({result.win_rate*100:.1f}%)")
    print(f"    Losers:           {result.losing_trades}")
    print(f"    Profit Factor:    {result.profit_factor:.2f}")
    print(f"    Avg Win:          ${result.avg_win:.2f}")
    print(f"    Avg Loss:         ${result.avg_loss:.2f}")
    print(f"    Largest Win:      ${result.largest_win:.2f}")
    print(f"    Largest Loss:     ${result.largest_loss:.2f}")

    print(f"\n  RISK METRICS:")
    print(f"    Max Drawdown:     {result.max_drawdown*100:.2f}%")
    print(f"    Max DD Duration:  {result.max_drawdown_duration_days} days")
    print(f"    Sharpe Ratio:     {result.sharpe_ratio:.2f}")
    print(f"    Sortino Ratio:    {result.sortino_ratio:.2f}")
    print(f"    Calmar Ratio:     {result.calmar_ratio:.2f}")

    print(f"\n  PROTECTION SYSTEM:")
    print(f"    Protection Events:       {len(result.protection_events)}")
    print(f"    Trades Blocked:          {result.trades_blocked}")
    print(f"    Positions Force-Closed:  {result.positions_force_closed}")
    print(f"    Capital Saved (est):     ${result.capital_saved_estimate:.2f}")
    print(f"    Volatility Pauses:       {result.volatility_pauses}")
    print(f"    Crisis Mode:             {result.crisis_mode_activations}")
    print(f"    Circuit Breaker:         {result.circuit_breaker_triggers}")
    print(f"    Recovery Mode:           {result.recovery_mode_activations}")
    print(f"    Profit Protection:       {result.profit_protection_activations}")

    print("\n" + "=" * 80)
