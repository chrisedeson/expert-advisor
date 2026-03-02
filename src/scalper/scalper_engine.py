"""Live Supply & Demand Scalper Engine.

Separate from the Grid EA. Uses:
- H1 candles for zone detection (supply/demand)
- M15 candles for entry signals (zone reaction + momentum filter)
- Fixed R:R take profit with breakeven at 1:1
- Telegram notifications prefixed with [SCALPER]

Architecture mirrors live_engine.py but with scalper-specific logic.
"""
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.live.broker_interface import BrokerInterface
from src.live.session_filter import SessionFilter
from src.live.state_manager import StateManager
from .zone_detector import ZoneDetector, SupplyDemandZone

logger = logging.getLogger('scalper_engine')


# Top instruments from MC validation (all 100% profitable)
SCALPER_INSTRUMENTS = {
    'US500':   {'pip_size': 0.01,   'pip_value': 1.0,  'spread': 0.5,  'slip': 0.2,  'lot_scale': 0.1},
    'XAUUSD':  {'pip_size': 0.01,   'pip_value': 1.0,  'spread': 2.5,  'slip': 0.5,  'lot_scale': 1.0},
    'XAGUSD':  {'pip_size': 0.001,  'pip_value': 5.0,  'spread': 3.0,  'slip': 0.5,  'lot_scale': 0.1},
    'USTEC':   {'pip_size': 0.01,   'pip_value': 1.0,  'spread': 1.5,  'slip': 0.3,  'lot_scale': 0.1},
    'EURUSD':  {'pip_size': 0.0001, 'pip_value': 10.0, 'spread': 0.7,  'slip': 0.2,  'lot_scale': 1.0},
    'ETHUSD':  {'pip_size': 0.01,   'pip_value': 1.0,  'spread': 2.0,  'slip': 0.5,  'lot_scale': 0.1},
}

SCALPER_PROFILES = {
    'conservative': {
        'risk_pct': 0.02,
        'max_concurrent': 3,
        'fixed_rr': 2.0,
        'min_score': 5.0,
        'description': 'Low risk: 2% per trade, R:R 2.0, max 3 concurrent',
    },
    'balanced': {
        'risk_pct': 0.03,
        'max_concurrent': 5,
        'fixed_rr': 2.0,
        'min_score': 5.0,
        'description': 'Moderate: 3% per trade, R:R 2.0, max 5 concurrent',
    },
    'aggressive': {
        'risk_pct': 0.05,
        'max_concurrent': 5,
        'fixed_rr': 3.0,
        'min_score': 3.0,
        'description': 'High risk: 5% per trade, R:R 3.0, max 5 concurrent',
    },
}


@dataclass
class ScalperPosition:
    """A tracked open scalper position."""
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    entry_price: float
    entry_time: datetime
    lot_size: float
    stop_loss: float
    take_profit: float
    zone_score: float
    zone_type: str  # 'supply' or 'demand'
    sl_at_breakeven: bool = False
    order_id: Optional[str] = None
    zone_key: Optional[str] = None  # Zone identity for touch tracking


class ScalperEngine:
    """Live Supply & Demand Scalper - runs alongside the Grid EA."""

    def __init__(
        self,
        broker: BrokerInterface,
        risk_profile: str = 'conservative',
        instruments: Dict[str, float] = None,
        initial_capital: float = 100.0,
        state_dir: str = "state_scalper",
        notifier=None,
        session_start: int = -1,
        session_end: int = -1,
        htf: str = 'H1',
        ltf: str = 'M15',
    ):
        self.broker = broker
        self.notifier = notifier
        self.initial_capital = initial_capital
        self.state_dir = Path(state_dir)
        self.htf = htf
        self.ltf = ltf

        # Risk profile
        if risk_profile not in SCALPER_PROFILES:
            raise ValueError(f"Unknown profile: {risk_profile}. Use: {list(SCALPER_PROFILES.keys())}")
        self.profile = SCALPER_PROFILES[risk_profile]
        self.risk_profile_name = risk_profile

        # Instruments and weights (equal weight by default)
        if instruments is None:
            instruments = {sym: 1.0 / len(SCALPER_INSTRUMENTS) for sym in SCALPER_INSTRUMENTS}
        self.instruments = instruments

        # Session filter (-1 = no filter, scalper runs 24h by default)
        if session_start >= 0 and session_end >= 0:
            self.session = SessionFilter(start_hour=session_start, end_hour=session_end)
            self.use_session = True
        else:
            self.session = SessionFilter(start_hour=0, end_hour=24)
            self.use_session = False

        # Zone detectors (one per instrument, fed with H1 data)
        self.zone_detectors: Dict[str, ZoneDetector] = {}
        for symbol in self.instruments:
            self.zone_detectors[symbol] = ZoneDetector(
                min_consecutive=4,
                ema_period=200,
                min_zone_score=self.profile['min_score'],
                max_zone_age_bars=200,
                zone_touch_limit=3,
            )

        # Active zones per instrument
        self.active_zones: Dict[str, List[SupplyDemandZone]] = {s: [] for s in self.instruments}
        self._last_htf_bar_time: Dict[str, Optional[datetime]] = {s: None for s in self.instruments}

        # Position tracking
        self.positions: Dict[str, List[ScalperPosition]] = {s: [] for s in self.instruments}

        # Zone touch tracking: counts trades per zone (matches backtest z_touches)
        # Key: symbol -> {zone_key -> trade_count}
        # zone_key = f"{zone_type}_{top:.6f}_{bottom:.6f}"
        self._zone_trades: Dict[str, Dict[str, int]] = {s: {} for s in self.instruments}
        self._zone_last_seen: Dict[str, Dict[str, datetime]] = {s: {} for s in self.instruments}
        self._zone_touch_limit = 3  # Max trades per zone (matches backtest touch_limit=3)

        # Per-bar entry tracking: prevent multiple entries on same M15 bar
        # Key: symbol -> last M15 bar time when we entered
        self._last_entry_bar_time: Dict[str, Optional[datetime]] = {s: None for s in self.instruments}

        # Market closed tracking: skip symbols whose market is closed
        # Key: symbol -> expiry time (recheck after this time)
        self._market_closed: Dict[str, datetime] = {}
        self._market_closed_recheck_minutes = 15  # recheck every 15 min

        # State
        self.state_manager = StateManager(str(self.state_dir / "scalper_state.json"))
        self.cash_balance = initial_capital
        self.running = False

        # Stats
        self.total_signals = 0
        self.total_trades_opened = 0
        self.total_trades_closed = 0

        # Notification tracking
        self._session_start_date: Optional[str] = None
        self._session_end_date: Optional[str] = None
        self._last_heartbeat: Optional[datetime] = None

    def _notify(self, method: str, *args, **kwargs):
        """Safely call a notifier method with [SCALPER] prefix."""
        if self.notifier is None:
            return
        try:
            if method == 'send':
                # Prepend [SCALPER] to raw messages
                if args:
                    msg = f"[SCALPER] {args[0]}"
                    self.notifier.send(msg)
                return
            elif method == 'send_startup':
                # Custom startup with [SCALPER] prefix
                profile, capital, broker_name, instruments = args[:4]
                msg = (
                    f"[SCALPER] <b>EA Started</b>\n"
                    f"Strategy: Supply & Demand Scalper\n"
                    f"Profile: {profile}\n"
                    f"Capital: ${capital:.0f}\n"
                    f"Broker: {broker_name}\n"
                    f"Instruments: {', '.join(instruments)}\n"
                    f"Session: {'24h' if not self.use_session else f'{self.session.start_hour}-{self.session.end_hour} UTC'}"
                )
                self.notifier.send(msg)
                return
            elif method == 'send_trade_opened':
                symbol, direction, price, lot_size, sl, tp = args[:6]
                zone_score = kwargs.get('zone_score', 0)
                arrow = "\u2197\ufe0f" if direction == "BUY" else "\u2198\ufe0f"
                msg = (
                    f"[SCALPER] {arrow} <b>OPENED {direction} {symbol}</b>\n"
                    f"Price: {price:.5f} | Lot: {lot_size:.3f}\n"
                    f"SL: {sl:.5f} | TP: {tp:.5f}\n"
                    f"Zone score: {zone_score:.1f}"
                )
                self.notifier.send(msg)
                return
            elif method == 'send_trade_closed':
                symbol, direction, exit_price, reason, pips, pnl = args[:6]
                icon = "\u2705" if pnl >= 0 else "\u274c"
                sign = "+" if pnl >= 0 else ""
                msg = (
                    f"[SCALPER] {icon} <b>CLOSED {direction} {symbol}</b>\n"
                    f"Exit: {exit_price:.5f} | {sign}{pips:.1f} pips | {sign}${pnl:.2f}\n"
                    f"Reason: {reason}"
                )
                self.notifier.send(msg)
                return
            elif method == 'send_error':
                msg = f"[SCALPER] \u26a0\ufe0f <b>ERROR</b>\n{args[0]}"
                self.notifier.send(msg)
                return
            # Fallback
            getattr(self.notifier, method)(*args, **kwargs)
        except Exception as e:
            logger.debug(f"Notification failed ({method}): {e}")

    def start(self):
        """Start the scalper trading loop."""
        logger.info(f"Starting scalper: profile={self.risk_profile_name}, "
                     f"capital=${self.initial_capital}")
        logger.info(f"Instruments: {list(self.instruments.keys())}")
        logger.info(f"Risk: {self.profile['risk_pct']*100:.0f}% per trade, "
                     f"R:R {self.profile['fixed_rr']}, "
                     f"max {self.profile['max_concurrent']} concurrent")

        # Wire up auth failure alerts
        if hasattr(self.broker, '_auth_failure_callback'):
            self.broker._auth_failure_callback = lambda msg: self._notify('send_error', f"AUTH FAILURE: {msg}")

        if not self.broker.connect():
            logger.error("Failed to connect to broker")
            self._notify('send_error', "Failed to connect to broker")
            return

        self._restore_state()
        self._reconcile_positions()
        self.running = True
        logger.info("Scalper engine started.")

        broker_name = type(self.broker).__name__
        self._notify('send_startup', self.risk_profile_name, self.initial_capital,
                     broker_name, list(self.instruments.keys()))

        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.exception(f"Scalper engine error: {e}")
            self._notify('send_error', f"Scalper crashed: {e}")
        finally:
            self._save_state()
            self.broker.disconnect()
            logger.info("Scalper engine stopped")

    def stop(self):
        self.running = False

    def _main_loop(self):
        """Main loop. Checks M15 candles every 60 seconds."""
        while self.running:
            now = datetime.now(timezone.utc)

            # Weekend check (always skip weekends)
            if now.weekday() >= 5:
                next_mon = now + timedelta(days=(7 - now.weekday()))
                next_mon = next_mon.replace(hour=0, minute=0, second=0, microsecond=0)
                wait_mins = (next_mon - now).total_seconds() / 60
                logger.info(f"Weekend. Next: {next_mon.strftime('%Y-%m-%d %H:%M UTC')} ({wait_mins:.0f} min)")
                time.sleep(min(300, max(60, wait_mins * 30)))
                continue

            if self.use_session:
                if self.session.is_active(now):
                    self._send_session_start(now)
                    self._trading_tick(now)
                    self._save_state()
                    time.sleep(60)
                else:
                    self._send_session_end(now)
                    self._save_state()
                    next_session = self.session.next_session_start(now)
                    wait_mins = (next_session - now).total_seconds() / 60
                    logger.info(f"Outside session. Next: {next_session.strftime('%Y-%m-%d %H:%M UTC')} "
                                 f"({wait_mins:.0f} min)")
                    time.sleep(min(300, max(60, wait_mins * 30)))
            else:
                # 24h mode: always trade on weekdays
                self._send_session_start(now)
                self._trading_tick(now)
                self._save_state()
                time.sleep(60)

    def _trading_tick(self, now: datetime):
        """One tick: refresh zones, check entries, manage positions."""
        do_heartbeat = (self._last_heartbeat is None or
                        (now - self._last_heartbeat).total_seconds() >= 3600)
        heartbeat_lines = []

        for symbol, weight in self.instruments.items():
            if weight <= 0:
                continue

            # Skip symbols whose market is known to be closed
            if symbol in self._market_closed:
                if now < self._market_closed[symbol]:
                    if do_heartbeat:
                        heartbeat_lines.append(f"{symbol} CLOSED")
                    continue
                else:
                    # Recheck period expired, try again
                    del self._market_closed[symbol]

            spec = SCALPER_INSTRUMENTS[symbol]

            # 1. Get H1 data for zone detection
            htf_candles = self.broker.get_candles(symbol, self.htf, count=250)
            if htf_candles is None or len(htf_candles) < 50:
                logger.warning(f"{symbol}: insufficient {self.htf} data")
                if do_heartbeat:
                    heartbeat_lines.append(f"{symbol}: NO HTF DATA")
                continue

            # 2. Get M15 data for entries
            ltf_candles = self.broker.get_candles(symbol, self.ltf, count=20)
            if ltf_candles is None or len(ltf_candles) < 5:
                logger.warning(f"{symbol}: insufficient {self.ltf} data")
                if do_heartbeat:
                    heartbeat_lines.append(f"{symbol}: NO LTF DATA")
                continue

            # 3. Refresh zones if new H1 bar
            self._refresh_zones(symbol, htf_candles)

            # 4. Current M15 bar
            current_bar = ltf_candles.iloc[-1]
            close = current_bar['close']
            atr = current_bar.get('atr', None)
            if atr is None or pd.isna(atr):
                # Compute ATR from LTF data
                atr = self._compute_atr_single(ltf_candles)

            # 5. Check exits on open positions
            self._check_exits(symbol, current_bar, spec)

            # 6. Update breakeven
            self._update_breakeven(symbol, current_bar, spec)

            # 7. Check for new entries
            if len(self.positions[symbol]) < self.profile['max_concurrent'] and atr > 0:
                self._check_entries(symbol, current_bar, atr, spec, now)

            # Heartbeat info
            if do_heartbeat:
                n_zones = len(self.active_zones[symbol])
                n_pos = len(self.positions[symbol])
                n_zt = sum(self._zone_trades.get(symbol, {}).values())
                heartbeat_lines.append(f"{symbol} {close:.5f} z={n_zones} p={n_pos} t={n_zt}")

        if do_heartbeat and heartbeat_lines:
            open_count = sum(len(v) for v in self.positions.values())
            total_zt = sum(sum(v.values()) for v in self._zone_trades.values())
            log_str = " | ".join(heartbeat_lines)
            logger.info(f"HEARTBEAT: {log_str} | total_pos={open_count} | zt={total_zt} | sig={self.total_signals}")
            tg_lines = "\n".join(heartbeat_lines)
            self._notify('send',
                f"\U0001f493 <b>HEARTBEAT</b> ({now.strftime('%H:%M UTC')})\n"
                f"{tg_lines}\n"
                f"Open: {open_count} | Touches: {total_zt} | Signals: {self.total_signals}")
            self._last_heartbeat = now

    def _refresh_zones(self, symbol: str, htf_candles: pd.DataFrame):
        """Re-detect zones when a new H1 bar appears."""
        latest_time = htf_candles.index[-1]
        if self._last_htf_bar_time[symbol] is not None and latest_time <= self._last_htf_bar_time[symbol]:
            return  # No new bar

        self._last_htf_bar_time[symbol] = latest_time

        # Detect all zones
        htf_candles.columns = [c.lower() for c in htf_candles.columns]
        detector = self.zone_detectors[symbol]
        all_zones = detector.detect_zones(htf_candles)

        # Filter by score and validity
        min_score = self.profile['min_score']
        valid_zones = [z for z in all_zones if z.strength >= min_score and z.is_valid]

        # Only keep zones within reasonable age (last 200 bars)
        max_age = 200
        n_bars = len(htf_candles)
        age_filtered = [
            z for z in valid_zones
            if (n_bars - z.creation_idx) <= max_age
        ]

        # Filter out zones that have exceeded touch limit (matches backtest z_touches > touch_limit)
        now = datetime.now(timezone.utc)
        active = []
        for z in age_filtered:
            zk = self._zone_key(z)
            self._zone_last_seen[symbol][zk] = now
            trade_count = self._zone_trades[symbol].get(zk, 0)
            if trade_count > self._zone_touch_limit:
                continue  # Zone exhausted - same as backtest line 320
            active.append(z)
        self.active_zones[symbol] = active

        # Cleanup stale zone records not seen in 24 hours
        cutoff = now - timedelta(hours=24)
        stale = [zk for zk, ts in self._zone_last_seen[symbol].items() if ts < cutoff]
        for zk in stale:
            self._zone_trades[symbol].pop(zk, None)
            del self._zone_last_seen[symbol][zk]

        logger.debug(f"{symbol}: {len(active)} active zones "
                      f"(from {len(all_zones)} total, "
                      f"{len(self._zone_trades[symbol])} tracked)")

    @staticmethod
    def _zone_key(zone: 'SupplyDemandZone') -> str:
        """Unique key for a zone based on its identity (not entry price)."""
        return f"{zone.zone_type}_{zone.top:.6f}_{zone.bottom:.6f}"

    def _check_entries(self, symbol: str, bar: pd.Series, atr: float,
                       spec: dict, now: datetime):
        """Check if M15 bar reacts to any active zone."""
        close = bar['close']
        open_ = bar['open']
        high = bar['high']
        low = bar['low']

        # Get the current M15 bar time
        bar_time = bar.name if hasattr(bar, 'name') and bar.name is not None else now

        # GUARD: Only allow one entry per M15 bar per symbol
        if self._last_entry_bar_time[symbol] is not None:
            if bar_time == self._last_entry_bar_time[symbol]:
                return  # Already entered on this bar
            # If bar_time is a datetime, also check if within same 15-min window
            if isinstance(bar_time, datetime) and isinstance(self._last_entry_bar_time[symbol], datetime):
                diff = abs((bar_time - self._last_entry_bar_time[symbol]).total_seconds())
                if diff < 900:  # 15 minutes
                    return

        total_cost = (spec['spread'] + spec['slip']) * spec['pip_size']
        sl_buffer = 0.3 * atr  # ATR-based SL buffer

        # Minimum SL distance: at least 3x the spread+slippage cost
        # This prevents entries where SL is so tight the spread alone closes it
        min_sl_distance = 3.0 * total_cost

        # Build set of zone keys with open positions (matches backtest used_zones)
        used_zone_keys = set()
        for pos in self.positions[symbol]:
            if pos.zone_key:
                used_zone_keys.add(pos.zone_key)

        for zone in self.active_zones[symbol]:
            if len(self.positions[symbol]) >= self.profile['max_concurrent']:
                break

            # Skip zones with open positions (matches backtest used_zones check)
            zk = self._zone_key(zone)
            if zk in used_zone_keys:
                continue

            # Skip zones that exceeded touch limit (matches backtest line 320)
            trade_count = self._zone_trades[symbol].get(zk, 0)
            if trade_count > self._zone_touch_limit:
                continue

            direction = None
            entry = None
            sl = None

            if zone.zone_type == 'demand':
                # Price dipped into zone and bounced (bullish candle)
                if low <= zone.top and close > zone.bottom and close > open_:
                    # Momentum filter: reject if candle is too large and too directional
                    # (fast slam through zone = not a clean reaction)
                    bar_range = high - low
                    bar_body = abs(close - open_)
                    if bar_range > 2.0 * atr and bar_body / max(bar_range, 1e-10) > 0.6:
                        continue
                    direction = 'BUY'
                    entry = close + total_cost
                    sl = zone.bottom - sl_buffer

            elif zone.zone_type == 'supply':
                # Price pushed into zone and rejected (bearish candle)
                if high >= zone.bottom and close < zone.top and close < open_:
                    bar_range = high - low
                    bar_body = abs(close - open_)
                    if bar_range > 2.0 * atr and bar_body / max(bar_range, 1e-10) > 0.6:
                        continue
                    direction = 'SELL'
                    entry = close - total_cost
                    sl = zone.top + sl_buffer

            if direction is None:
                continue

            # Calculate risk and TP
            risk = abs(entry - sl)
            if risk <= 0:
                continue

            # GUARD: Reject entries where SL is too tight
            if risk < min_sl_distance:
                logger.debug(f"{symbol}: Skipping {direction} - SL too tight "
                            f"({risk/spec['pip_size']:.1f} pips < min {min_sl_distance/spec['pip_size']:.1f} pips)")
                continue

            tp = entry + risk * self.profile['fixed_rr'] if direction == 'BUY' else entry - risk * self.profile['fixed_rr']

            # Position sizing: risk% of capital
            risk_amt = self.cash_balance * self.profile['risk_pct']
            sl_pips = risk / spec['pip_size']
            lot = risk_amt / (sl_pips * spec['pip_value']) if sl_pips * spec['pip_value'] > 0 else 0
            lot *= spec['lot_scale']
            lot = max(0.01, round(lot, 2))

            # Execute
            self.total_signals += 1
            comment = f"SCALPER_{symbol}_{direction}_Z{zone.strength:.0f}"
            logger.info(f"SIGNAL: {direction} {symbol} @ {entry:.5f}, "
                         f"SL={sl:.5f}, TP={tp:.5f}, lot={lot:.3f}, zone={zone.strength:.1f}")

            result = self.broker.place_market_order(
                symbol=symbol,
                direction=direction,
                lot_size=lot,
                stop_loss=sl,
                take_profit=tp,
                comment=comment,
            )

            if result.success:
                actual_price = result.fill_price or entry
                pos = ScalperPosition(
                    symbol=symbol,
                    direction=direction,
                    entry_price=actual_price,
                    entry_time=now,
                    lot_size=lot,
                    stop_loss=sl,
                    take_profit=tp,
                    zone_score=zone.strength,
                    zone_type=zone.zone_type,
                    order_id=result.order_id,
                    zone_key=zk,
                )
                self.positions[symbol].append(pos)
                self.total_trades_opened += 1

                # Increment zone touch counter (matches backtest z_touches[zi] += 1)
                self._zone_trades[symbol][zk] = self._zone_trades[symbol].get(zk, 0) + 1
                new_count = self._zone_trades[symbol][zk]
                remaining = max(0, self._zone_touch_limit - new_count + 1)
                logger.info(f"Zone touch: {zk} = {new_count}/{self._zone_touch_limit + 1} "
                           f"({remaining} remaining)")

                # Record this bar as "entered" - no more entries on this bar
                self._last_entry_bar_time[symbol] = bar_time

                self.state_manager.save_trade_log({
                    'event': 'OPEN', 'symbol': symbol, 'direction': direction,
                    'price': actual_price, 'lot': lot, 'sl': sl, 'tp': tp,
                    'zone_score': zone.strength, 'zone_type': zone.zone_type,
                    'order_id': result.order_id, 'time': now.isoformat(),
                }, str(self.state_dir / "scalper_trades.jsonl"))

                logger.info(f"OPENED: {direction} {symbol} @ {actual_price:.5f}, id={result.order_id}")
                self._notify('send_trade_opened', symbol, direction,
                             actual_price, lot, sl, tp, zone_score=zone.strength)
            else:
                error_str = str(result.error or "")
                if "MARKET_CLOSED" in error_str:
                    # Market is closed - suppress error, mark symbol for skip
                    recheck_at = now + timedelta(minutes=self._market_closed_recheck_minutes)
                    self._market_closed[symbol] = recheck_at
                    logger.info(f"{symbol}: Market closed, skipping until {recheck_at.strftime('%H:%M UTC')}")
                elif "NOT_ENOUGH_MONEY" in error_str:
                    # Not enough margin - suppress repeated errors, skip for 60 min
                    recheck_at = now + timedelta(minutes=60)
                    self._market_closed[symbol] = recheck_at
                    logger.warning(f"{symbol}: Not enough margin for {lot} lot, skipping until {recheck_at.strftime('%H:%M UTC')}")
                    self._notify('send_error', f"Margin insufficient for {direction} {symbol} {lot} lot. Skipping 60 min.")
                else:
                    logger.error(f"ORDER FAILED: {symbol} {direction} - {result.error}")
                    self._notify('send_error', f"Order failed: {direction} {symbol} - {result.error}")

    def _check_exits(self, symbol: str, bar: pd.Series, spec: dict):
        """Check if any positions hit SL/TP. Broker handles actual SL/TP, but we
        also detect broker-side closes by checking position existence."""
        if not self.positions[symbol]:
            return

        # Get broker positions to detect broker-side closes
        broker_positions = self.broker.get_open_positions(symbol)
        broker_ids = {str(p.get('positionId', p.get('order_id', p.get('id', '')))) for p in broker_positions} if broker_positions else set()

        to_remove = []
        for i, pos in enumerate(self.positions[symbol]):
            # If position has an order_id and it's no longer on the broker, it was closed
            if pos.order_id and str(pos.order_id) not in broker_ids:
                # Broker closed it (SL/TP hit on broker side)
                close_price = bar['close']  # approximate
                pip_size = spec['pip_size']
                pip_value = spec['pip_value']
                if pos.direction == 'BUY':
                    pips = (close_price - pos.entry_price) / pip_size
                else:
                    pips = (pos.entry_price - close_price) / pip_size
                net_pnl = pips * pip_value * pos.lot_size
                reason = "TP" if pips > 0 else ("BE" if pos.sl_at_breakeven and abs(pips) < 1 else "SL")

                logger.info(f"BROKER-CLOSED: {pos.direction} {symbol} ~{close_price:.5f}, "
                             f"reason~{reason}, pips~{pips:.1f}, P&L~${net_pnl:.2f}")

                self._notify('send_trade_closed', symbol, pos.direction,
                             close_price, f"{reason} (broker)", pips, net_pnl)

                self.state_manager.save_trade_log({
                    'event': 'CLOSE', 'symbol': symbol, 'direction': pos.direction,
                    'entry_price': pos.entry_price, 'exit_price': close_price,
                    'reason': f'{reason} (broker-side)', 'pips': pips, 'net_pnl': net_pnl,
                    'lot': pos.lot_size, 'order_id': pos.order_id,
                    'time': datetime.now(timezone.utc).isoformat(),
                }, str(self.state_dir / "scalper_trades.jsonl"))

                self.total_trades_closed += 1
                to_remove.append(i)

        for i in reversed(to_remove):
            self.positions[symbol].pop(i)

    def _update_breakeven(self, symbol: str, bar: pd.Series, spec: dict):
        """Move SL to breakeven when price moves 1:1 in favor."""
        total_cost = (spec['spread'] + spec['slip']) * spec['pip_size']

        for pos in self.positions[symbol]:
            if pos.sl_at_breakeven:
                continue

            if pos.direction == 'BUY':
                risk = pos.entry_price - pos.stop_loss
                if risk > 0 and bar['high'] >= pos.entry_price + risk:
                    new_sl = pos.entry_price + total_cost
                    pos.stop_loss = new_sl
                    pos.sl_at_breakeven = True
                    # Update broker
                    if pos.order_id:
                        self.broker.modify_position(order_id=pos.order_id, stop_loss=new_sl)
                    logger.info(f"BE: {symbol} {pos.direction} SL -> {new_sl:.5f}")
            else:
                risk = pos.stop_loss - pos.entry_price
                if risk > 0 and bar['low'] <= pos.entry_price - risk:
                    new_sl = pos.entry_price - total_cost
                    pos.stop_loss = new_sl
                    pos.sl_at_breakeven = True
                    if pos.order_id:
                        self.broker.modify_position(order_id=pos.order_id, stop_loss=new_sl)
                    logger.info(f"BE: {symbol} {pos.direction} SL -> {new_sl:.5f}")

    def _compute_atr_single(self, df: pd.DataFrame, period: int = 14) -> float:
        """Compute latest ATR from a small dataframe."""
        if len(df) < period + 1:
            return (df['high'] - df['low']).mean()
        h, l, c = df['high'], df['low'], df['close']
        tr = pd.DataFrame({
            'hl': h - l,
            'hc': (h - c.shift(1)).abs(),
            'lc': (l - c.shift(1)).abs(),
        }).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]

    def _send_session_start(self, now: datetime):
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
        logger.info(f"Session start: balance=${account.balance:.2f}, equity=${account.equity:.2f}")

    def _send_session_end(self, now: datetime):
        today = now.strftime('%Y-%m-%d')
        if self._session_end_date == today:
            return
        if self._session_start_date != today:
            return
        self._session_end_date = today

        account = self.broker.get_account_info()
        open_count = sum(len(v) for v in self.positions.values())

        msg = (
            f"\U0001f4ca <b>SESSION CLOSED</b> ({today})\n"
            f"Balance: ${account.balance:.2f}\n"
            f"Equity: ${account.equity:.2f}\n"
            f"Open: {open_count} | Signals: {self.total_signals}\n"
            f"Opened: {self.total_trades_opened} | Closed: {self.total_trades_closed}"
        )
        self._notify('send', msg)
        logger.info(f"Session end: balance=${account.balance:.2f}, equity=${account.equity:.2f}")

    def _reconcile_positions(self):
        """Reconcile internal position tracking with broker's actual positions.

        Handles two cases:
        - Ghost positions: in our state but closed on broker (removed while offline)
        - Orphan positions: on broker but not in our state (state was cleared)
        """
        try:
            broker_positions = self.broker.get_open_positions()
        except Exception as e:
            logger.warning(f"Reconciliation skipped - couldn't fetch broker positions: {e}")
            return

        if broker_positions is None:
            broker_positions = []

        # Build broker position map: positionId -> position dict
        broker_map = {}
        for bp in broker_positions:
            pid = str(bp.get('positionId', bp.get('order_id', bp.get('id', ''))))
            if pid:
                broker_map[pid] = bp

        # Build our tracked position map: order_id -> (symbol, index)
        our_map = {}
        for symbol, positions in self.positions.items():
            for i, pos in enumerate(positions):
                if pos.order_id:
                    our_map[str(pos.order_id)] = (symbol, pos)

        # Case A: Ghost positions (in our state, not on broker)
        ghosts = []
        for order_id, (symbol, pos) in our_map.items():
            if order_id not in broker_map:
                ghosts.append((symbol, pos))

        for symbol, pos in ghosts:
            self.positions[symbol] = [p for p in self.positions[symbol]
                                       if str(p.order_id) != str(pos.order_id)]
            logger.info(f"GHOST: {pos.direction} {symbol} (id={pos.order_id}) "
                        f"was closed while offline, removed from tracking")

        if ghosts:
            ghost_summary = ", ".join(f"{p.direction} {s}" for s, p in ghosts)
            self._notify('send', f"\U0001f47b <b>RECONCILIATION</b>\n"
                        f"Removed {len(ghosts)} ghost position(s) "
                        f"(closed while offline):\n{ghost_summary}")

        # Case B: Orphan positions (on broker, not in our state)
        our_ids = set(our_map.keys())
        orphans = []
        for pid, bp in broker_map.items():
            if pid not in our_ids:
                orphans.append((pid, bp))

        adopted = 0
        for pid, bp in orphans:
            symbol_id = bp.get('symbolId')
            symbol_name = None
            if hasattr(self.broker, 'get_symbol_name'):
                symbol_name = self.broker.get_symbol_name(symbol_id)

            if not symbol_name or symbol_name not in self.instruments:
                logger.warning(f"ORPHAN: Position {pid} on unknown/untracked symbol "
                              f"(id={symbol_id}, name={symbol_name}). Leaving on broker.")
                continue

            direction = bp.get('direction', 'BUY')
            entry_price = bp.get('price', 0)
            volume = bp.get('volume', 0)
            # Use broker's actual lot_size for accurate volume-to-lot conversion
            if hasattr(self.broker, 'get_symbol_lot_size'):
                lot_size_units = self.broker.get_symbol_lot_size(symbol_id)
            else:
                lot_size_units = 100000  # fallback
            lot = volume / lot_size_units if lot_size_units > 0 else volume / 100000.0

            pos = ScalperPosition(
                symbol=symbol_name,
                direction=direction,
                entry_price=entry_price,
                entry_time=datetime.now(timezone.utc),
                lot_size=round(lot, 3),
                stop_loss=0,  # Unknown, broker handles it
                take_profit=0,  # Unknown, broker handles it
                zone_score=0,
                zone_type='unknown',
                order_id=pid,
                zone_key=None,
            )
            self.positions[symbol_name].append(pos)
            adopted += 1
            logger.info(f"ADOPTED: {direction} {symbol_name} @ {entry_price:.5f} "
                        f"(id={pid}, lot~{lot:.3f})")

        if adopted > 0:
            adopted_summary = ", ".join(
                f"{bp.get('direction','')} {self.broker.get_symbol_name(bp.get('symbolId',0)) or '?'}"
                for _, bp in orphans
                if (self.broker.get_symbol_name(bp.get('symbolId', 0)) or '') in self.instruments
            )
            self._notify('send', f"\U0001f50d <b>RECONCILIATION</b>\n"
                        f"Adopted {adopted} orphan position(s) from broker:\n"
                        f"{adopted_summary}")

        total_open = sum(len(v) for v in self.positions.values())
        if ghosts or adopted:
            logger.info(f"Reconciliation complete: {len(ghosts)} ghosts removed, "
                        f"{adopted} orphans adopted. Total tracked: {total_open}")
        else:
            logger.info(f"Reconciliation: all positions match. Tracked: {total_open}")

    def _save_state(self):
        state = {
            'strategy': 'scalper',
            'risk_profile': self.risk_profile_name,
            'cash_balance': self.cash_balance,
            'initial_capital': self.initial_capital,
            'positions': {
                symbol: [
                    {
                        'symbol': p.symbol, 'direction': p.direction,
                        'entry_price': p.entry_price,
                        'entry_time': p.entry_time.isoformat() if p.entry_time else None,
                        'lot_size': p.lot_size,
                        'stop_loss': p.stop_loss, 'take_profit': p.take_profit,
                        'zone_score': p.zone_score, 'zone_type': p.zone_type,
                        'sl_at_breakeven': p.sl_at_breakeven,
                        'order_id': p.order_id,
                        'zone_key': p.zone_key,
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
            'zone_state': {
                symbol: {
                    'last_htf_bar': self._last_htf_bar_time[symbol].isoformat()
                    if self._last_htf_bar_time[symbol] else None,
                    'active_zone_count': len(zones),
                }
                for symbol, zones in self.active_zones.items()
            },
            'zone_trades': {
                symbol: dict(trades)
                for symbol, trades in self._zone_trades.items()
            },
            'zone_last_seen': {
                symbol: {zk: ts.isoformat() for zk, ts in seen.items()}
                for symbol, seen in self._zone_last_seen.items()
            },
        }
        self.state_manager.save(state)

    def _restore_state(self):
        state = self.state_manager.load()
        if state is None:
            logger.info("No previous scalper state. Starting fresh.")
            return

        self.cash_balance = state.get('cash_balance', self.initial_capital)
        self.total_signals = state.get('stats', {}).get('total_signals', 0)
        self.total_trades_opened = state.get('stats', {}).get('total_trades_opened', 0)
        self.total_trades_closed = state.get('stats', {}).get('total_trades_closed', 0)

        for symbol, pos_list in state.get('positions', {}).items():
            if symbol not in self.positions:
                continue
            for p in pos_list:
                entry_time = datetime.fromisoformat(p['entry_time']) if p.get('entry_time') else None
                pos = ScalperPosition(
                    symbol=p['symbol'], direction=p['direction'],
                    entry_price=p['entry_price'], entry_time=entry_time,
                    lot_size=p['lot_size'],
                    stop_loss=p['stop_loss'], take_profit=p['take_profit'],
                    zone_score=p.get('zone_score', 0), zone_type=p.get('zone_type', 'demand'),
                    sl_at_breakeven=p.get('sl_at_breakeven', False),
                    order_id=p.get('order_id'),
                    zone_key=p.get('zone_key'),
                )
                self.positions[symbol].append(pos)

        # Restore zone trade counts
        for symbol, trades in state.get('zone_trades', {}).items():
            if symbol in self._zone_trades:
                self._zone_trades[symbol] = trades

        # Restore zone last seen timestamps
        for symbol, seen in state.get('zone_last_seen', {}).items():
            if symbol not in self._zone_last_seen:
                continue
            for zk, ts_str in seen.items():
                try:
                    self._zone_last_seen[symbol][zk] = datetime.fromisoformat(ts_str)
                except (ValueError, TypeError):
                    pass

        total_open = sum(len(v) for v in self.positions.values())
        total_zt = sum(sum(v.values()) for v in self._zone_trades.values())
        logger.info(f"State restored: balance=${self.cash_balance:.2f}, "
                     f"{total_open} open positions, {total_zt} zone touches, "
                     f"{self.total_trades_opened} opened, {self.total_trades_closed} closed")

    def status(self) -> Dict:
        account = self.broker.get_account_info()
        open_count = sum(len(v) for v in self.positions.values())
        return {
            'strategy': 'scalper',
            'running': self.running,
            'risk_profile': self.risk_profile_name,
            'balance': account.balance,
            'equity': account.equity,
            'open_positions': open_count,
            'signals': self.total_signals,
            'trades_opened': self.total_trades_opened,
            'trades_closed': self.total_trades_closed,
            'active_zones': {s: len(z) for s, z in self.active_zones.items()},
            'positions_by_symbol': {s: len(p) for s, p in self.positions.items()},
        }
