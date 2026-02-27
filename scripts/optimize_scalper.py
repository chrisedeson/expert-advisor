#!/usr/bin/env python3
"""Optimize the Supply & Demand Scalper across instruments and parameters.

Focused on finding the max daily return with realistic position sizing.
"""

import sys
from pathlib import Path
from itertools import product

import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.scalper.scalper_backtest import ScalperBacktester
from scripts.backtest_scalper import INSTRUMENTS, load_data


def run_fixed_lot_test(symbol, htf='H1', ltf='M15', capital=100.0, lot=0.01,
                       min_score=5.0, fixed_rr=2.0, consecutive=4,
                       max_concurrent=3, session_start=-1, session_end=-1,
                       use_momentum=True, use_breakeven=True,
                       max_zone_age=200, zone_touches=3, sl_buffer=0.3):
    """Run with fixed lot size (no compounding) for realistic results."""
    spec = INSTRUMENTS[symbol]
    htf_data = load_data(symbol, htf)
    ltf_data = load_data(symbol, ltf)

    common_start = max(htf_data.index[0], ltf_data.index[0])
    common_end = min(htf_data.index[-1], ltf_data.index[-1])
    htf_data = htf_data[htf_data.index <= common_end]
    ltf_data = ltf_data[(ltf_data.index >= common_start) & (ltf_data.index <= common_end)]

    bt = ScalperBacktester(
        initial_balance=capital,
        spread_pips=spec['spread'],
        slippage_pips=spec['slip'],
        pip_size=spec['pip_size'],
        pip_value_per_lot=spec['pip_value'],
        max_concurrent=max_concurrent,
        min_rr=1.0,  # Low bar, fixed_rr handles target
        min_zone_score=min_score,
        sl_buffer_atr_mult=sl_buffer,
        tp_mode='fixed_rr',
        fixed_rr_target=fixed_rr,
        use_breakeven=use_breakeven,
        use_momentum_filter=use_momentum,
        session_start=session_start,
        session_end=session_end,
        risk_pct_per_trade=0.02,
        compound=False,  # FIXED LOT, no compounding
        max_zone_age_bars=max_zone_age,
        zone_touch_limit=zone_touches,
        consecutive_candles=consecutive,
    )

    return bt.run_backtest(htf_data, ltf_data, symbol=symbol, htf_name=htf, ltf_name=ltf)


def main():
    print("=" * 100)
    print("  SUPPLY & DEMAND SCALPER - OPTIMIZATION")
    print("  Target: MAX daily return on $100 capital")
    print("=" * 100)

    # Phase 1: All instruments with default params, no compounding
    print("\n=== PHASE 1: All instruments, fixed lot, no compounding ===")
    print(f"{'Symbol':<10} {'Trades':>6} {'WR%':>6} {'PF':>6} {'CAGR%':>8} {'DD%':>6} {'Sharpe':>7} "
          f"{'T/day':>6} {'$/trade':>8} {'$/day':>7} {'Daily%':>7}")
    print("-" * 95)

    results = {}
    for sym in INSTRUMENTS:
        try:
            r = run_fixed_lot_test(sym, capital=100.0, lot=0.01)
            if r and r.total_trades > 0:
                results[sym] = r
                pnl_per_trade = (r.final_balance - r.initial_balance) / r.total_trades
                days = len(set(str(t.entry_time)[:10] for t in r.trades))
                pnl_per_day = (r.final_balance - r.initial_balance) / max(days, 1)
                daily_pct = pnl_per_day / 100 * 100

                marker = ' ***' if daily_pct >= 20 else (' **' if daily_pct >= 10 else (' *' if daily_pct >= 5 else ''))
                print(f"{sym:<10} {r.total_trades:>6} {r.win_rate:>5.1f}% {r.profit_factor:>5.2f} "
                      f"{r.cagr:>7.1f}% {r.max_drawdown:>5.1f}% {r.sharpe_ratio:>6.2f} "
                      f"{r.trades_per_day:>5.2f} ${pnl_per_trade:>7.2f} ${pnl_per_day:>6.2f} {daily_pct:>6.1f}%{marker}")
        except Exception as e:
            print(f"{sym:<10} ERROR: {e}")

    # Phase 2: Parameter sweep on top instruments
    if results:
        # Sort by profit factor
        top_5 = sorted(results.items(), key=lambda x: x[1].profit_factor, reverse=True)[:5]
        print(f"\n\n=== PHASE 2: Parameter sweep on top 5 ===")

        for sym, _ in top_5:
            print(f"\n--- {sym} Parameter Sweep ---")
            print(f"{'Score':>6} {'RR':>5} {'Consec':>6} {'Conc':>5} {'Trades':>6} {'WR%':>6} {'PF':>6} "
                  f"{'DD%':>6} {'Sharpe':>7} {'$/day':>7} {'Daily%':>7}")
            print("-" * 85)

            best_daily = 0
            best_params = {}

            for score, rr, consec, conc in product(
                [3.0, 5.0, 7.0],
                [1.5, 2.0, 3.0],
                [3, 4, 5],
                [3, 5],
            ):
                try:
                    r = run_fixed_lot_test(sym, min_score=score, fixed_rr=rr,
                                          consecutive=consec, max_concurrent=conc)
                    if r and r.total_trades > 0:
                        days = len(set(str(t.entry_time)[:10] for t in r.trades))
                        pnl_per_day = (r.final_balance - r.initial_balance) / max(days, 1)
                        daily_pct = pnl_per_day / 100 * 100

                        if daily_pct > best_daily:
                            best_daily = daily_pct
                            best_params = {'score': score, 'rr': rr, 'consec': consec, 'conc': conc}

                        if daily_pct >= 5:
                            marker = ' ***' if daily_pct >= 20 else (' **' if daily_pct >= 10 else ' *')
                            print(f"{score:>6.1f} {rr:>5.1f} {consec:>6} {conc:>5} {r.total_trades:>6} "
                                  f"{r.win_rate:>5.1f}% {r.profit_factor:>5.2f} {r.max_drawdown:>5.1f}% "
                                  f"{r.sharpe_ratio:>6.2f} ${pnl_per_day:>6.2f} {daily_pct:>6.1f}%{marker}")
                except Exception:
                    pass

            print(f"\n  BEST: {best_daily:.1f}% daily | params={best_params}")

    # Phase 3: Session filter test on best instruments
    if results:
        print(f"\n\n=== PHASE 3: Session windows on top performers ===")
        top_3 = sorted(results.items(), key=lambda x: x[1].profit_factor, reverse=True)[:3]

        for sym, _ in top_3:
            print(f"\n--- {sym} Session Windows ---")
            sessions = [
                (-1, -1, "24h"),
                (7, 17, "7-17 London+NY"),
                (8, 20, "8-20 Full day"),
                (12, 16, "12-16 Overlap"),
                (0, 8, "0-8 Asian"),
                (14, 22, "14-22 NY"),
            ]
            for start, end, label in sessions:
                r = run_fixed_lot_test(sym, session_start=start, session_end=end)
                if r and r.total_trades > 0:
                    days = len(set(str(t.entry_time)[:10] for t in r.trades))
                    pnl_per_day = (r.final_balance - r.initial_balance) / max(days, 1)
                    daily_pct = pnl_per_day / 100 * 100
                    print(f"  {label:<20} Trades={r.total_trades:>5} WR={r.win_rate:>5.1f}% PF={r.profit_factor:>5.2f} "
                          f"DD={r.max_drawdown:>5.1f}% Sharpe={r.sharpe_ratio:>6.2f} Daily={daily_pct:>6.1f}%")

    print("\n\nDONE.")


if __name__ == '__main__':
    main()
