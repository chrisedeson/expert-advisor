"""Live trading engine: main loop that monitors instruments and trades.

Orchestrates:
- Session filter (12-16 UTC only)
- Signal generation per instrument
- Order execution via broker interface
- Position management (trailing stops, SL/TP)
- State persistence
- Logging

Supports multiple risk profiles: conservative, balanced, growth, aggressive.
"""
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd

from .signal_engine import SignalEngine, Position, Signal
from .broker_interface import BrokerInterface
from .session_filter import SessionFilter
from .state_manager import StateManager

logger = logging.getLogger('live_engine')


# Instrument specifications for the 5 MC-validated winners
INSTRUMENTS = {
    'EURUSD': {'pip_size': 0.0001, 'pip_value': 10.0, 'spread': 0.7, 'slip': 0.2, 'lot_scale': 1.0},
    'GBPUSD': {'pip_size': 0.0001, 'pip_value': 10.0, 'spread': 0.9, 'slip': 0.3, 'lot_scale': 1.0},
    'EURJPY': {'pip_size': 0.01,   'pip_value': 6.67, 'spread': 1.0, 'slip': 0.3, 'lot_scale': 1.0},
    'XAGUSD': {'pip_size': 0.001,  'pip_value': 5.0,  'spread': 3.0, 'slip': 0.5, 'lot_scale': 0.1},
    'US500':  {'pip_size': 0.01,   'pip_value': 1.0,  'spread': 0.5, 'slip': 0.2, 'lot_scale': 0.1},
}

RISK_PROFILES = {
    'conservative': {'base_lot': 0.02, 'mult': 1.5, 'description': 'Low risk, steady growth'},
    'balanced':     {'base_lot': 0.03, 'mult': 2.0, 'description': 'Moderate risk/reward'},
    'growth':       {'base_lot': 0.04, 'mult': 2.0, 'description': 'Higher returns, more volatility'},
    'aggressive':   {'base_lot': 0.05, 'mult': 2.5, 'description': 'High risk, high reward'},
}

ALLOCATION_EQUAL = {'EURUSD': 0.20, 'GBPUSD': 0.20, 'EURJPY': 0.20, 'XAGUSD': 0.20, 'US500': 0.20}


class LiveEngine:
    """Main live trading engine."""

    def __init__(
        self,
        broker: BrokerInterface,
        risk_profile: str = 'conservative',
        allocation: Dict[str, float] = None,
        initial_capital: float = 500.0,
        state_dir: str = "state",
        paper_mode: bool = True,
        notifier=None,
    ):
        self.broker = broker
        self.paper_mode = paper_mode
        self.notifier = notifier
        self.initial_capital = initial_capital
        self.state_dir = Path(state_dir)

        # Risk profile
        if risk_profile not in RISK_PROFILES:
            raise ValueError(f"Unknown risk profile: {risk_profile}. Use: {list(RISK_PROFILES.keys())}")
        self.profile = RISK_PROFILES[risk_profile]
        self.risk_profile_name = risk_profile

        # Allocation
        self.allocation = allocation or ALLOCATION_EQUAL

        # Session filter
        self.session = SessionFilter(start_hour=12, end_hour=16)

        # Signal engines (one per instrument with correct pip_size)
        self.signal_engines: Dict[str, SignalEngine] = {}
        for symbol in self.allocation:
            spec = INSTRUMENTS[symbol]
            min_tp = spec['spread'] + spec['slip'] + 5.0
            self.signal_engines[symbol] = SignalEngine(
                pip_size=spec['pip_size'],
                min_tp_pips=min_tp,
            )

        # Position tracking
        self.positions: Dict[str, List[Position]] = {s: [] for s in self.allocation}

        # State
        self.state_manager = StateManager(str(self.state_dir / "live_state.json"))
        self.cash_balance = initial_capital
        self.running = False

        # Cooldown tracking
        self._last_entry_time: Dict[str, Dict[str, Optional[datetime]]] = {
            s: {'BUY': None, 'SELL': None} for s in self.allocation
        }

        # Stats
        self.total_signals = 0
        self.total_trades_opened = 0
        self.total_trades_closed = 0

        # Session notification tracking (send once per day)
        self._session_start_date: Optional[str] = None
        self._session_end_date: Optional[str] = None
        self._last_heartbeat: Optional[datetime] = None

    def _notify(self, method: str, *args, **kwargs):
        """Safely call a notifier method (fail-silent)."""
        if self.notifier is None:
            return
        try:
            getattr(self.notifier, method)(*args, **kwargs)
        except Exception as e:
            logger.debug(f"Notification failed ({method}): {e}")

    def start(self):
        """Start the live trading loop."""
        logger.info(f"Starting live engine: profile={self.risk_profile_name}, "
                     f"capital=${self.initial_capital}, paper={self.paper_mode}")
        logger.info(f"Instruments: {list(self.allocation.keys())}")
        logger.info(f"Base lot: {self.profile['base_lot']}, Mult: {self.profile['mult']}")
        logger.info(f"Session: 12-16 UTC (London/NY overlap)")

        if not self.broker.connect():
            logger.error("Failed to connect to broker")
            self._notify('send_error', "Failed to connect to broker")
            return

        # Try to restore state
        self._restore_state()

        self.running = True
        logger.info("Engine started. Waiting for session...")

        broker_name = type(self.broker).__name__
        self._notify('send_startup', self.risk_profile_name, self.initial_capital,
                     broker_name, list(self.allocation.keys()))

        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.exception(f"Engine error: {e}")
            self._notify('send_error', f"Engine crashed: {e}")
        finally:
            self._save_state()
            self.broker.disconnect()
            logger.info("Engine stopped")

    def stop(self):
        """Signal the engine to stop."""
        self.running = False

    def _main_loop(self):
        """Main trading loop. Runs every 60 seconds."""
        while self.running:
            now = datetime.now(timezone.utc)

            if self.session.is_active(now):
                self._send_session_start(now)
                self._trading_tick(now)
                # Check for SL/TP hits on existing positions
                self._check_position_exits(now)
                # Save state every tick during session
                self._save_state()
                time.sleep(60)
            else:
                self._send_session_end(now)
                # Save state once when session ends
                self._save_state()
                next_session = self.session.next_session_start(now)
                wait_mins = (next_session - now).total_seconds() / 60
                logger.info(f"Outside session. Next: {next_session.strftime('%Y-%m-%d %H:%M UTC')} "
                             f"({wait_mins:.0f} min)")
                time.sleep(min(300, max(60, wait_mins * 30)))

    def _trading_tick(self, now: datetime):
        """One trading tick: check signals and manage positions."""
        # Heartbeat: log prices + BB diagnostics every 15 minutes
        do_heartbeat = (self._last_heartbeat is None or
                        (now - self._last_heartbeat).total_seconds() >= 3600)
        heartbeat_lines = []

        for symbol, weight in self.allocation.items():
            if weight <= 0:
                continue

            spec = INSTRUMENTS[symbol]
            engine = self.signal_engines[symbol]

            # Get recent candle data
            candles = self.broker.get_candles(symbol, "H1", count=250)
            if candles is None or len(candles) < 200:
                logger.warning(f"{symbol}: insufficient data ({len(candles) if candles is not None else 0} bars)")
                if do_heartbeat:
                    heartbeat_lines.append(f"{symbol}: NO DATA")
                continue

            current_bar = candles.iloc[-1]

            # Build diagnostic info at heartbeat intervals
            if do_heartbeat:
                close = current_bar['close']
                atr = current_bar.get('atr', 0)
                close_prices = candles['close']
                sma50 = close_prices.rolling(50).mean().iloc[-1]
                sma200 = close_prices.rolling(200).mean().iloc[-1]
                std50 = close_prices.rolling(50).std().iloc[-1]
                if not pd.isna(sma50) and not pd.isna(std50) and std50 > 0:
                    upper_bb = sma50 + 2.0 * std50
                    lower_bb = sma50 - 2.0 * std50
                    trend = "UP" if sma50 > sma200 else "DN"
                    # Distance to nearest band as % of band width
                    band_width = upper_bb - lower_bb
                    dist_lower = (close - lower_bb) / band_width * 100
                    dist_upper = (upper_bb - close) / band_width * 100
                    # Show which side is actionable
                    if trend == "UP":
                        near = f"{dist_lower:.0f}% from buy"
                    else:
                        near = f"{dist_upper:.0f}% from sell"
                    heartbeat_lines.append(
                        f"{symbol} {close:.5f} [{trend}] {near}"
                    )
                else:
                    heartbeat_lines.append(f"{symbol} {close:.5f} [calc err]")

            # Check exits on existing positions
            exits = engine.check_exits(self.positions[symbol], current_bar, spec['pip_size'])
            for exit_info in exits:
                self._close_position_from_signal(symbol, exit_info, spec)

            # Update trailing stops
            engine.update_trailing_stops(self.positions[symbol], current_bar)
            self._sync_trailing_stops(symbol)

            # Check for new entry signals
            signal = engine.check_entry(
                symbol=symbol,
                recent_data=candles,
                open_positions=self.positions[symbol],
                current_time=now,
            )

            if signal:
                self.total_signals += 1
                logger.info(f"SIGNAL: {signal.direction} {symbol} @ {signal.entry_price:.5f}")
                if self._check_cooldown(symbol, signal.direction, now):
                    self._execute_signal(symbol, signal, spec, weight)
                else:
                    logger.info(f"COOLDOWN: {symbol} {signal.direction} blocked (< 1h since last)")

        # Heartbeat log + Telegram every 15 min
        if do_heartbeat and heartbeat_lines:
            open_count = sum(len(v) for v in self.positions.values())
            log_str = " | ".join(heartbeat_lines)
            logger.info(f"HEARTBEAT: {log_str} | pos={open_count} | sig={self.total_signals}")
            # Telegram: one line per instrument for readability
            tg_lines = "\n".join(heartbeat_lines)
            self._notify('send',
                f"\U0001f493 <b>HEARTBEAT</b> ({now.strftime('%H:%M UTC')})\n"
                f"{tg_lines}\n"
                f"Open: {open_count} | Signals: {self.total_signals}")
            self._last_heartbeat = now

    def _execute_signal(self, symbol: str, signal: Signal, spec: Dict, weight: float):
        """Execute a trading signal."""
        # Calculate lot size with compounding
        allocated_capital = self.cash_balance * weight
        equity_ratio = max(self.cash_balance / self.initial_capital, 0.5)
        base_lot = self.profile['base_lot'] * spec['lot_scale']
        lot_size = base_lot * (self.profile['mult'] ** signal.grid_level) * equity_ratio

        if lot_size < 0.001:
            lot_size = 0.001

        comment = f"EA_{self.risk_profile_name}_{symbol}_L{signal.grid_level}"

        logger.info(f"SIGNAL: {signal.direction} {symbol} @ {signal.entry_price:.5f}, "
                     f"SL={signal.stop_loss:.5f}, TP={signal.take_profit:.5f}, "
                     f"lot={lot_size:.3f}, grid={signal.grid_level}")

        result = self.broker.place_market_order(
            symbol=symbol,
            direction=signal.direction,
            lot_size=lot_size,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            comment=comment,
        )

        if result.success:
            pos = Position(
                symbol=symbol,
                direction=signal.direction,
                entry_price=result.fill_price or signal.entry_price,
                entry_time=datetime.now(timezone.utc),
                lot_size=lot_size,
                grid_level=signal.grid_level,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                order_id=result.order_id,
            )
            self.positions[symbol].append(pos)
            self.total_trades_opened += 1
            self._last_entry_time[symbol][signal.direction] = datetime.now(timezone.utc)

            # Log trade
            self.state_manager.save_trade_log({
                'event': 'OPEN', 'symbol': symbol, 'direction': signal.direction,
                'price': result.fill_price, 'lot': lot_size, 'sl': signal.stop_loss,
                'tp': signal.take_profit, 'grid_level': signal.grid_level,
                'order_id': result.order_id, 'time': datetime.now(timezone.utc).isoformat(),
            }, str(self.state_dir / "trade_log.jsonl"))

            logger.info(f"OPENED: {signal.direction} {symbol} @ {result.fill_price:.5f}, "
                         f"id={result.order_id}")

            self._notify('send_trade_opened', symbol, signal.direction,
                         result.fill_price or signal.entry_price, lot_size,
                         signal.stop_loss, signal.take_profit, signal.grid_level)
        else:
            logger.error(f"ORDER FAILED: {symbol} {signal.direction} - {result.error}")
            self._notify('send_error', f"Order failed: {signal.direction} {symbol} - {result.error}")

    def _close_position_from_signal(self, symbol: str, exit_info: Dict, spec: Dict):
        """Close a position based on signal engine exit."""
        pos = exit_info['position']
        reason = exit_info['reason']

        if pos.order_id:
            result = self.broker.close_position(pos.order_id)
            if result.success:
                # Calculate P&L
                exit_price = result.fill_price or exit_info['exit_price']
                pip_size = spec['pip_size']
                pip_value = spec['pip_value']
                if pos.direction == 'BUY':
                    pips = (exit_price - pos.entry_price) / pip_size
                else:
                    pips = (pos.entry_price - exit_price) / pip_size
                net_pnl = pips * pip_value * pos.lot_size

                logger.info(f"CLOSED: {pos.direction} {symbol} @ {exit_price:.5f}, "
                             f"reason={reason}, pips={pips:.1f}, P&L=${net_pnl:.2f}")

                self._notify('send_trade_closed', symbol, pos.direction,
                             exit_price, reason, pips, net_pnl)

                self.state_manager.save_trade_log({
                    'event': 'CLOSE', 'symbol': symbol, 'direction': pos.direction,
                    'entry_price': pos.entry_price, 'exit_price': exit_price,
                    'reason': reason, 'pips': pips, 'net_pnl': net_pnl,
                    'lot': pos.lot_size, 'order_id': pos.order_id,
                    'time': datetime.now(timezone.utc).isoformat(),
                }, str(self.state_dir / "trade_log.jsonl"))

                self.total_trades_closed += 1
            else:
                logger.error(f"CLOSE FAILED: {symbol} {pos.order_id} - {result.error}")
                self._notify('send_error', f"Close failed: {symbol} {pos.order_id} - {result.error}")
                return

        if pos in self.positions[symbol]:
            self.positions[symbol].remove(pos)

    def _sync_trailing_stops(self, symbol: str):
        """Push updated trailing stop levels to broker."""
        for pos in self.positions[symbol]:
            if pos.order_id:
                self.broker.modify_position(
                    order_id=pos.order_id,
                    stop_loss=pos.stop_loss,
                )

    def _check_position_exits(self, now: datetime):
        """Check all positions for SL/TP hits using current bar data."""
        for symbol in self.allocation:
            if not self.positions[symbol]:
                continue
            candles = self.broker.get_candles(symbol, "H1", count=1)
            if candles is None or len(candles) == 0:
                continue
            current_bar = candles.iloc[-1]
            spec = INSTRUMENTS[symbol]
            engine = self.signal_engines[symbol]

            exits = engine.check_exits(self.positions[symbol], current_bar, spec['pip_size'])
            for exit_info in exits:
                self._close_position_from_signal(symbol, exit_info, spec)

            engine.update_trailing_stops(self.positions[symbol], current_bar)

    def _send_session_start(self, now: datetime):
        """Send session start notification (once per day)."""
        today = now.strftime('%Y-%m-%d')
        if self._session_start_date == today:
            return
        self._session_start_date = today

        account = self.broker.get_account_info()
        open_count = sum(len(v) for v in self.positions.values())
        pos_detail = {s: len(p) for s, p in self.positions.items() if len(p) > 0}

        msg = (
            f"\U0001f514 <b>SESSION OPEN</b> ({today})\n"
            f"Balance: ${account.balance:.2f}\n"
            f"Equity: ${account.equity:.2f}\n"
            f"Open positions: {open_count}"
        )
        if pos_detail:
            msg += "\n" + ", ".join(f"{s}: {n}" for s, n in pos_detail.items())
        self._notify('send', msg)
        logger.info(f"Session start: balance=${account.balance:.2f}, equity=${account.equity:.2f}, "
                     f"{open_count} open positions")

    def _send_session_end(self, now: datetime):
        """Send session end summary (once per day)."""
        today = now.strftime('%Y-%m-%d')
        if self._session_end_date == today:
            return
        # Only send if we had a session start today (avoid sending on first boot outside session)
        if self._session_start_date != today:
            return
        self._session_end_date = today

        account = self.broker.get_account_info()
        open_count = sum(len(v) for v in self.positions.values())
        pos_detail = {s: len(p) for s, p in self.positions.items() if len(p) > 0}

        msg = (
            f"\U0001f4ca <b>SESSION CLOSED</b> ({today})\n"
            f"Balance: ${account.balance:.2f}\n"
            f"Equity: ${account.equity:.2f}\n"
            f"Open positions: {open_count}\n"
            f"Total signals: {self.total_signals}\n"
            f"Trades opened: {self.total_trades_opened}\n"
            f"Trades closed: {self.total_trades_closed}"
        )
        if pos_detail:
            msg += "\nPositions: " + ", ".join(f"{s}: {n}" for s, n in pos_detail.items())
        self._notify('send', msg)
        logger.info(f"Session end: balance=${account.balance:.2f}, equity=${account.equity:.2f}")

    def _check_cooldown(self, symbol: str, direction: str, now: datetime) -> bool:
        """Check if enough time has passed since last entry."""
        last = self._last_entry_time.get(symbol, {}).get(direction)
        if last is None:
            return True
        hours = (now - last).total_seconds() / 3600
        return hours >= 1.0

    def _save_state(self):
        """Save current engine state to disk."""
        state = {
            'risk_profile': self.risk_profile_name,
            'cash_balance': self.cash_balance,
            'initial_capital': self.initial_capital,
            'paper_mode': self.paper_mode,
            'positions': {
                symbol: [
                    {
                        'symbol': p.symbol, 'direction': p.direction,
                        'entry_price': p.entry_price,
                        'entry_time': p.entry_time.isoformat() if p.entry_time else None,
                        'lot_size': p.lot_size, 'grid_level': p.grid_level,
                        'stop_loss': p.stop_loss, 'take_profit': p.take_profit,
                        'order_id': p.order_id,
                    }
                    for p in positions
                ]
                for symbol, positions in self.positions.items()
            },
            'stats': {
                'total_signals': self.total_signals,
                'total_trades_opened': self.total_trades_opened,
                'total_trades_closed': self.total_trades_closed,
            },
        }
        self.state_manager.save(state)

    def _restore_state(self):
        """Restore engine state from disk."""
        state = self.state_manager.load()
        if state is None:
            logger.info("No previous state found. Starting fresh.")
            return

        self.cash_balance = state.get('cash_balance', self.initial_capital)
        self.total_signals = state.get('stats', {}).get('total_signals', 0)
        self.total_trades_opened = state.get('stats', {}).get('total_trades_opened', 0)
        self.total_trades_closed = state.get('stats', {}).get('total_trades_closed', 0)

        # Restore positions
        for symbol, pos_list in state.get('positions', {}).items():
            if symbol not in self.positions:
                continue
            for p in pos_list:
                entry_time = datetime.fromisoformat(p['entry_time']) if p.get('entry_time') else None
                pos = Position(
                    symbol=p['symbol'], direction=p['direction'],
                    entry_price=p['entry_price'], entry_time=entry_time,
                    lot_size=p['lot_size'], grid_level=p['grid_level'],
                    stop_loss=p['stop_loss'], take_profit=p['take_profit'],
                    order_id=p.get('order_id'),
                )
                self.positions[symbol].append(pos)

        total_open = sum(len(v) for v in self.positions.values())
        logger.info(f"State restored: balance=${self.cash_balance:.2f}, "
                     f"{total_open} open positions, "
                     f"{self.total_trades_opened} trades opened, "
                     f"{self.total_trades_closed} closed")

    def status(self) -> Dict:
        """Return current engine status."""
        account = self.broker.get_account_info()
        open_count = sum(len(v) for v in self.positions.values())
        return {
            'running': self.running,
            'paper_mode': self.paper_mode,
            'risk_profile': self.risk_profile_name,
            'session_active': self.session.is_active(),
            'balance': account.balance,
            'equity': account.equity,
            'open_positions': open_count,
            'signals': self.total_signals,
            'trades_opened': self.total_trades_opened,
            'trades_closed': self.total_trades_closed,
            'positions_by_symbol': {s: len(p) for s, p in self.positions.items()},
        }
