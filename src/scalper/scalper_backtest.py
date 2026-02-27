"""Supply & Demand Scalper Backtesting Engine - OPTIMIZED.

Multi-timeframe backtest:
- Detect zones on HTF (H1/H4)
- Take entries on LTF (M15/M5)
- Simulate entries with zone reaction confirmation
- Track positions with SL/TP and breakeven logic

Optimized for speed: pre-computes zone ranges, uses numpy arrays.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta

from .zone_detector import ZoneDetector, SupplyDemandZone


@dataclass
class ScalperTrade:
    """A completed scalper trade."""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    pnl: float
    pips: float
    rr_ratio: float
    exit_reason: str
    zone_score: float
    zone_type: str


@dataclass
class ScalperBacktestResult:
    """Complete backtest results."""
    symbol: str
    htf: str
    ltf: str
    start_date: object
    end_date: object

    initial_balance: float
    final_balance: float
    total_return_pct: float
    cagr: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    avg_rr: float
    trades_per_day: float

    best_trade: float
    worst_trade: float

    trades: List[ScalperTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)


class _OpenPos:
    """Lightweight open position for speed."""
    __slots__ = ['direction', 'entry_price', 'entry_time', 'sl', 'tp',
                 'lot_size', 'zone_idx', 'sl_at_be', 'entry_bar']

    def __init__(self, direction, entry_price, entry_time, sl, tp, lot_size, zone_idx, entry_bar):
        self.direction = direction
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.sl = sl
        self.tp = tp
        self.lot_size = lot_size
        self.zone_idx = zone_idx
        self.sl_at_be = False
        self.entry_bar = entry_bar


class ScalperBacktester:
    """Backtests the Supply & Demand scalper strategy. Optimized for speed."""

    def __init__(
        self,
        initial_balance: float = 100.0,
        spread_pips: float = 0.7,
        slippage_pips: float = 0.2,
        pip_size: float = 0.0001,
        pip_value_per_lot: float = 10.0,
        lot_size: float = 0.1,
        max_concurrent: int = 3,
        min_rr: float = 1.5,
        min_zone_score: float = 5.0,
        sl_buffer_atr_mult: float = 0.3,
        tp_mode: str = 'fixed_rr',
        fixed_rr_target: float = 2.0,
        use_breakeven: bool = True,
        use_momentum_filter: bool = True,
        session_start: int = -1,
        session_end: int = -1,
        risk_pct_per_trade: float = 0.02,
        compound: bool = True,
        max_zone_age_bars: int = 200,
        zone_touch_limit: int = 3,
        consecutive_candles: int = 4,
        ema_period: int = 200,
    ):
        self.initial_balance = initial_balance
        self.spread_pips = spread_pips
        self.slippage_pips = slippage_pips
        self.pip_size = pip_size
        self.pip_value = pip_value_per_lot
        self.lot_size = lot_size
        self.max_concurrent = max_concurrent
        self.min_rr = min_rr
        self.min_zone_score = min_zone_score
        self.sl_buffer_atr_mult = sl_buffer_atr_mult
        self.tp_mode = tp_mode
        self.fixed_rr_target = fixed_rr_target
        self.use_breakeven = use_breakeven
        self.use_momentum_filter = use_momentum_filter
        self.session_start = session_start
        self.session_end = session_end
        self.risk_pct = risk_pct_per_trade
        self.compound = compound

        self.total_cost = (spread_pips + slippage_pips) * pip_size

        self.detector = ZoneDetector(
            min_consecutive=consecutive_candles,
            ema_period=ema_period,
            min_zone_score=0,
            max_zone_age_bars=max_zone_age_bars,
            zone_touch_limit=zone_touch_limit,
        )

    def run_backtest(
        self,
        htf_data: pd.DataFrame,
        ltf_data: pd.DataFrame,
        symbol: str = 'UNKNOWN',
        htf_name: str = 'H1',
        ltf_name: str = 'M15',
    ) -> ScalperBacktestResult:
        """Run a full backtest."""
        htf_data = htf_data.copy()
        ltf_data = ltf_data.copy()
        htf_data.columns = [c.lower() for c in htf_data.columns]
        ltf_data.columns = [c.lower() for c in ltf_data.columns]

        # Detect all zones on HTF (one-time cost)
        all_zones = self.detector.detect_zones(htf_data)

        # Pre-filter zones by score
        zones = [z for z in all_zones if z.strength >= self.min_zone_score]

        # Build zone arrays for fast lookup
        n_zones = len(zones)
        if n_zones == 0:
            return self._empty_result(symbol, htf_name, ltf_name)

        z_type = np.array([1 if z.zone_type == 'demand' else -1 for z in zones])  # 1=demand, -1=supply
        z_top = np.array([z.top for z in zones])
        z_bot = np.array([z.bottom for z in zones])
        z_score = np.array([z.strength for z in zones])
        z_created = np.array([z.creation_idx for z in zones])
        z_valid = np.ones(n_zones, dtype=bool)
        z_touches = np.zeros(n_zones, dtype=int)
        z_push = np.array([z.push_size for z in zones])

        # HTF-to-LTF index mapping
        htf_times = htf_data.index.values
        ltf_times = ltf_data.index.values
        htf_idx_for_ltf = np.searchsorted(htf_times, ltf_times, side='right') - 1

        # Pre-compute ATR on LTF
        if 'atr' not in ltf_data.columns:
            ltf_data['atr'] = self._compute_atr(ltf_data, 14)

        O = ltf_data['open'].values
        H = ltf_data['high'].values
        L = ltf_data['low'].values
        C = ltf_data['close'].values
        ATR = ltf_data['atr'].values

        # Pre-compute session mask
        n_ltf = len(ltf_data)
        if self.session_start >= 0 and self.session_end >= 0:
            hours = np.array([pd.Timestamp(t).hour for t in ltf_times])
            session_mask = (hours >= self.session_start) & (hours < self.session_end)
        else:
            session_mask = np.ones(n_ltf, dtype=bool)

        # Pre-compute momentum: average range over 20 bars
        avg_range = pd.Series(H - L).rolling(20, min_periods=1).mean().values
        bar_range = H - L
        bar_body = np.abs(C - O)

        max_zone_age = self.detector.max_zone_age_bars
        touch_limit = self.detector.zone_touch_limit

        balance = self.initial_balance
        peak_bal = balance
        max_dd = 0.0
        positions: List[_OpenPos] = []
        trades: List[ScalperTrade] = []
        daily_bal = {}
        eq_curve = []

        # Build active zone set: only zones created before current HTF bar
        # Sorted by creation index for efficient activation
        zone_order = np.argsort(z_created)
        next_zone_ptr = 0  # pointer into zone_order for zones not yet activated
        active_set = set()  # indices of currently active zones

        htf_closes = htf_data['close'].values
        prev_htf_i = -1

        for i in range(1, n_ltf):
            htf_i = htf_idx_for_ltf[i]
            atr = ATR[i]

            # --- Check exits first ---
            if positions:
                to_remove = []
                for pi, pos in enumerate(positions):
                    ex_price = None
                    ex_reason = None

                    if pos.direction == 1:  # BUY
                        if L[i] <= pos.sl:
                            ex_price = pos.sl
                            ex_reason = 'BE' if pos.sl_at_be else 'SL'
                        elif H[i] >= pos.tp:
                            ex_price = pos.tp
                            ex_reason = 'TP'
                    else:  # SELL
                        if H[i] >= pos.sl:
                            ex_price = pos.sl
                            ex_reason = 'BE' if pos.sl_at_be else 'SL'
                        elif L[i] <= pos.tp:
                            ex_price = pos.tp
                            ex_reason = 'TP'

                    if ex_price is None and pos.zone_idx >= 0 and not z_valid[pos.zone_idx]:
                        ex_price = C[i]
                        ex_reason = 'zone_break'

                    if ex_price is not None:
                        pips = ((ex_price - pos.entry_price) if pos.direction == 1 else (pos.entry_price - ex_price)) / self.pip_size
                        pnl = pips * self.pip_value * pos.lot_size
                        balance += pnl
                        risk_pips = abs(pos.entry_price - pos.sl) / self.pip_size
                        rr = pips / risk_pips if risk_pips > 0 else 0

                        trades.append(ScalperTrade(
                            entry_time=pos.entry_time, exit_time=ltf_data.index[i],
                            symbol=symbol, direction='BUY' if pos.direction == 1 else 'SELL',
                            entry_price=pos.entry_price, exit_price=ex_price,
                            stop_loss=pos.sl, take_profit=pos.tp,
                            lot_size=pos.lot_size, pnl=pnl, pips=pips, rr_ratio=rr,
                            exit_reason=ex_reason,
                            zone_score=z_score[pos.zone_idx] if pos.zone_idx >= 0 else 0,
                            zone_type='demand' if pos.direction == 1 else 'supply',
                        ))
                        to_remove.append(pi)

                for pi in reversed(to_remove):
                    positions.pop(pi)

                # Breakeven update
                if self.use_breakeven:
                    for pos in positions:
                        if pos.sl_at_be:
                            continue
                        if pos.direction == 1:
                            risk = pos.entry_price - pos.sl
                            if risk > 0 and H[i] >= pos.entry_price + risk:
                                pos.sl = pos.entry_price + self.total_cost
                                pos.sl_at_be = True
                        else:
                            risk = pos.sl - pos.entry_price
                            if risk > 0 and L[i] <= pos.entry_price - risk:
                                pos.sl = pos.entry_price - self.total_cost
                                pos.sl_at_be = True

            # --- Zone management: only on HTF bar change ---
            if htf_i != prev_htf_i and htf_i >= 0 and htf_i < len(htf_closes):
                prev_htf_i = htf_i
                htf_close = htf_closes[htf_i]

                # Activate new zones
                while next_zone_ptr < n_zones:
                    zi = zone_order[next_zone_ptr]
                    if z_created[zi] < htf_i:
                        if z_valid[zi]:
                            active_set.add(zi)
                        next_zone_ptr += 1
                    else:
                        break

                # Invalidate zones
                to_deactivate = []
                for zi in active_set:
                    if htf_i - z_created[zi] > max_zone_age:
                        z_valid[zi] = False
                        to_deactivate.append(zi)
                    elif touch_limit > 0 and z_touches[zi] > touch_limit:
                        z_valid[zi] = False
                        to_deactivate.append(zi)
                    elif z_type[zi] == 1 and htf_close < z_bot[zi]:
                        z_valid[zi] = False
                        to_deactivate.append(zi)
                    elif z_type[zi] == -1 and htf_close > z_top[zi]:
                        z_valid[zi] = False
                        to_deactivate.append(zi)
                for zi in to_deactivate:
                    active_set.discard(zi)

            # --- Skip if outside session ---
            if not session_mask[i]:
                if positions:
                    equity = balance + sum(
                        ((C[i] - p.entry_price) if p.direction == 1 else (p.entry_price - C[i])) / self.pip_size * self.pip_value * p.lot_size
                        for p in positions
                    )
                else:
                    equity = balance
                eq_curve.append(equity)
                continue

            # --- Check for entries (only check active zones) ---
            if len(positions) < self.max_concurrent and atr > 0 and active_set:
                sl_buf = self.sl_buffer_atr_mult * atr
                used_zones = {p.zone_idx for p in positions}

                for zi in active_set:
                    if len(positions) >= self.max_concurrent:
                        break
                    if zi in used_zones:
                        continue

                    if z_type[zi] == 1:  # DEMAND - BUY
                        if L[i] <= z_top[zi] and C[i] > z_bot[zi] and C[i] > O[i]:
                            if self.use_momentum_filter and avg_range[i] > 0:
                                if bar_range[i] / avg_range[i] > 2.0 and bar_body[i] / max(bar_range[i], 1e-10) > 0.6:
                                    continue

                            entry = C[i] + self.total_cost
                            sl = z_bot[zi] - sl_buf
                            if sl >= entry:
                                continue
                            risk = entry - sl
                            tp = entry + risk * self.fixed_rr_target
                            if risk <= 0:
                                continue

                            risk_amt = balance * self.risk_pct if self.compound else self.initial_balance * self.risk_pct
                            sl_pips = risk / self.pip_size
                            lot = risk_amt / (sl_pips * self.pip_value) if sl_pips * self.pip_value > 0 else 0
                            lot = max(0.01, round(lot, 2))

                            positions.append(_OpenPos(1, entry, ltf_data.index[i], sl, tp, lot, zi, i))
                            z_touches[zi] += 1

                    else:  # SUPPLY - SELL
                        if H[i] >= z_bot[zi] and C[i] < z_top[zi] and C[i] < O[i]:
                            if self.use_momentum_filter and avg_range[i] > 0:
                                if bar_range[i] / avg_range[i] > 2.0 and bar_body[i] / max(bar_range[i], 1e-10) > 0.6:
                                    continue

                            entry = C[i] - self.total_cost
                            sl = z_top[zi] + sl_buf
                            if sl <= entry:
                                continue
                            risk = sl - entry
                            tp = entry - risk * self.fixed_rr_target
                            if risk <= 0:
                                continue

                            risk_amt = balance * self.risk_pct if self.compound else self.initial_balance * self.risk_pct
                            sl_pips = risk / self.pip_size
                            lot = risk_amt / (sl_pips * self.pip_value) if sl_pips * self.pip_value > 0 else 0
                            lot = max(0.01, round(lot, 2))

                            positions.append(_OpenPos(-1, entry, ltf_data.index[i], sl, tp, lot, zi, i))
                            z_touches[zi] += 1

            # Track equity
            if positions:
                equity = balance + sum(
                    ((C[i] - p.entry_price) if p.direction == 1 else (p.entry_price - C[i])) / self.pip_size * self.pip_value * p.lot_size
                    for p in positions
                )
            else:
                equity = balance
            eq_curve.append(equity)
            peak_bal = max(peak_bal, equity)
            dd = (peak_bal - equity) / peak_bal if peak_bal > 0 else 0
            max_dd = max(max_dd, dd)

            day_key = str(ltf_data.index[i])[:10]
            daily_bal[day_key] = equity

        # Close remaining
        for pos in positions:
            ex = C[-1]
            if pos.direction == 1:
                pips = (ex - pos.entry_price) / self.pip_size
            else:
                pips = (pos.entry_price - ex) / self.pip_size
            pnl = pips * self.pip_value * pos.lot_size
            balance += pnl

        return self._compute_results(symbol, htf_name, ltf_name, trades, eq_curve, daily_bal, balance, max_dd)

    def _compute_atr(self, df, period=14):
        h, l, c = df['high'], df['low'], df['close']
        tr = pd.DataFrame({'hl': h - l, 'hc': (h - c.shift(1)).abs(), 'lc': (l - c.shift(1)).abs()}).max(axis=1)
        return tr.rolling(period).mean()

    def _empty_result(self, symbol, htf, ltf):
        return ScalperBacktestResult(
            symbol=symbol, htf=htf, ltf=ltf,
            start_date=None, end_date=None,
            initial_balance=self.initial_balance, final_balance=self.initial_balance,
            total_return_pct=0, cagr=0, max_drawdown=0, sharpe_ratio=0,
            profit_factor=0, total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0, avg_win=0, avg_loss=0, avg_rr=0, trades_per_day=0,
            best_trade=0, worst_trade=0,
        )

    def _compute_results(self, symbol, htf, ltf, trades, eq_curve, daily_bal, final_bal, max_dd):
        if not trades:
            return self._empty_result(symbol, htf, ltf)

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        ret = (final_bal - self.initial_balance) / self.initial_balance * 100
        start = trades[0].entry_time
        end = trades[-1].exit_time
        days = max((pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / 86400, 1)
        years = max(days / 365.25, 0.01)
        ratio = final_bal / self.initial_balance
        cagr = (ratio ** (1 / years) - 1) * 100 if ratio > 0 else -100

        daily_vals = list(daily_bal.values())
        if len(daily_vals) > 2:
            dr = np.diff(daily_vals) / np.array(daily_vals[:-1])
            dr = dr[np.isfinite(dr)]
            sharpe = np.mean(dr) / np.std(dr) * np.sqrt(252) if len(dr) > 0 and np.std(dr) > 0 else 0
        else:
            sharpe = 0

        gp = sum(t.pnl for t in wins) if wins else 0
        gl = abs(sum(t.pnl for t in losses)) if losses else 0
        pf = gp / gl if gl > 0 else (99.0 if gp > 0 else 0)

        avg_rr = np.mean([t.rr_ratio for t in trades])
        tpd = len(trades) / max(len(daily_bal), 1)

        return ScalperBacktestResult(
            symbol=symbol, htf=htf, ltf=ltf,
            start_date=start, end_date=end,
            initial_balance=self.initial_balance, final_balance=final_bal,
            total_return_pct=ret, cagr=cagr, max_drawdown=max_dd * 100,
            sharpe_ratio=sharpe, profit_factor=pf,
            total_trades=len(trades), winning_trades=len(wins), losing_trades=len(losses),
            win_rate=len(wins) / len(trades) * 100, avg_win=np.mean([t.pnl for t in wins]) if wins else 0,
            avg_loss=np.mean([t.pnl for t in losses]) if losses else 0,
            avg_rr=avg_rr, trades_per_day=tpd,
            best_trade=max(t.pnl for t in trades), worst_trade=min(t.pnl for t in trades),
            trades=trades, equity_curve=eq_curve,
        )
