#!/usr/bin/env python3
"""Backtest the Supply & Demand Scalper strategy.

Usage:
    python scripts/backtest_scalper.py                          # Default: XAUUSD
    python scripts/backtest_scalper.py --symbol EURUSD          # Single instrument
    python scripts/backtest_scalper.py --all                    # All 20 instruments
    python scripts/backtest_scalper.py --optimize               # Parameter sweep
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.scalper.scalper_backtest import ScalperBacktester, ScalperBacktestResult

# Instrument specs (same as grid EA + extras for scalper)
INSTRUMENTS = {
    'EURUSD':  {'pip_size': 0.0001,  'pip_value': 10.0,  'spread': 0.7,  'slip': 0.2,  'lot_scale': 1.0},
    'GBPUSD':  {'pip_size': 0.0001,  'pip_value': 10.0,  'spread': 0.9,  'slip': 0.3,  'lot_scale': 1.0},
    'EURJPY':  {'pip_size': 0.01,    'pip_value': 6.67,  'spread': 1.0,  'slip': 0.3,  'lot_scale': 1.0},
    'USDJPY':  {'pip_size': 0.01,    'pip_value': 6.67,  'spread': 0.8,  'slip': 0.2,  'lot_scale': 1.0},
    'GBPJPY':  {'pip_size': 0.01,    'pip_value': 6.67,  'spread': 1.5,  'slip': 0.5,  'lot_scale': 1.0},
    'AUDUSD':  {'pip_size': 0.0001,  'pip_value': 10.0,  'spread': 0.8,  'slip': 0.2,  'lot_scale': 1.0},
    'NZDUSD':  {'pip_size': 0.0001,  'pip_value': 10.0,  'spread': 1.0,  'slip': 0.3,  'lot_scale': 1.0},
    'USDCAD':  {'pip_size': 0.0001,  'pip_value': 10.0,  'spread': 1.0,  'slip': 0.3,  'lot_scale': 1.0},
    'USDCHF':  {'pip_size': 0.0001,  'pip_value': 10.0,  'spread': 1.0,  'slip': 0.3,  'lot_scale': 1.0},
    'EURGBP':  {'pip_size': 0.0001,  'pip_value': 10.0,  'spread': 1.0,  'slip': 0.3,  'lot_scale': 1.0},
    'EURAUD':  {'pip_size': 0.0001,  'pip_value': 10.0,  'spread': 1.5,  'slip': 0.5,  'lot_scale': 1.0},
    'EURCHF':  {'pip_size': 0.0001,  'pip_value': 10.0,  'spread': 1.2,  'slip': 0.3,  'lot_scale': 1.0},
    'XAUUSD':  {'pip_size': 0.01,    'pip_value': 1.0,   'spread': 2.5,  'slip': 0.5,  'lot_scale': 1.0},
    'XAGUSD':  {'pip_size': 0.001,   'pip_value': 5.0,   'spread': 3.0,  'slip': 0.5,  'lot_scale': 0.1},
    'US500':   {'pip_size': 0.01,    'pip_value': 1.0,   'spread': 0.5,  'slip': 0.2,  'lot_scale': 0.1},
    'US30':    {'pip_size': 0.01,    'pip_value': 1.0,   'spread': 2.0,  'slip': 0.5,  'lot_scale': 0.1},
    'USTEC':   {'pip_size': 0.01,    'pip_value': 1.0,   'spread': 1.5,  'slip': 0.3,  'lot_scale': 0.1},
    'BTCUSD':  {'pip_size': 1.0,     'pip_value': 1.0,   'spread': 30.0, 'slip': 5.0,  'lot_scale': 0.01},
    'ETHUSD':  {'pip_size': 0.01,    'pip_value': 1.0,   'spread': 2.0,  'slip': 0.5,  'lot_scale': 0.1},
    'XRPUSD':  {'pip_size': 0.0001,  'pip_value': 10.0,  'spread': 5.0,  'slip': 1.0,  'lot_scale': 0.1},
}


def load_data(symbol: str, tf: str) -> pd.DataFrame:
    """Load parquet data for a symbol and timeframe."""
    path = project_root / 'data' / 'processed' / f'{symbol}_{tf}.parquet'
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    return df


def run_single(
    symbol: str,
    htf: str = 'H1',
    ltf: str = 'M15',
    capital: float = 100.0,
    risk_pct: float = 0.02,
    min_rr: float = 1.5,
    min_score: float = 5.0,
    tp_mode: str = 'fixed_rr',
    fixed_rr: float = 2.0,
    session_start: int = -1,
    session_end: int = -1,
    consecutive: int = 4,
    max_concurrent: int = 3,
    use_momentum: bool = True,
    use_breakeven: bool = True,
    max_zone_age: int = 200,
    zone_touches: int = 3,
    ema_period: int = 200,
    sl_buffer: float = 0.3,
    verbose: bool = True,
) -> ScalperBacktestResult:
    """Run backtest for a single instrument."""
    spec = INSTRUMENTS.get(symbol)
    if spec is None:
        print(f"Unknown symbol: {symbol}")
        return None

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {symbol} | HTF={htf} LTF={ltf} | ${capital}")
        print(f"{'='*60}")

    htf_data = load_data(symbol, htf)
    ltf_data = load_data(symbol, ltf)

    # Align dates: only use LTF data within HTF date range
    common_start = max(htf_data.index[0], ltf_data.index[0])
    common_end = min(htf_data.index[-1], ltf_data.index[-1])
    htf_data = htf_data[htf_data.index <= common_end]
    ltf_data = ltf_data[(ltf_data.index >= common_start) & (ltf_data.index <= common_end)]

    if verbose:
        print(f"  HTF: {len(htf_data)} bars ({htf_data.index[0]} to {htf_data.index[-1]})")
        print(f"  LTF: {len(ltf_data)} bars ({ltf_data.index[0]} to {ltf_data.index[-1]})")

    bt = ScalperBacktester(
        initial_balance=capital,
        spread_pips=spec['spread'],
        slippage_pips=spec['slip'],
        pip_size=spec['pip_size'],
        pip_value_per_lot=spec['pip_value'],
        max_concurrent=max_concurrent,
        min_rr=min_rr,
        min_zone_score=min_score,
        sl_buffer_atr_mult=sl_buffer,
        tp_mode=tp_mode,
        fixed_rr_target=fixed_rr,
        use_breakeven=use_breakeven,
        use_momentum_filter=use_momentum,
        session_start=session_start,
        session_end=session_end,
        risk_pct_per_trade=risk_pct,
        compound=True,
        max_zone_age_bars=max_zone_age,
        zone_touch_limit=zone_touches,
        consecutive_candles=consecutive,
        ema_period=ema_period,
        max_risk_pct=0.05,
        max_sl_atr_mult=3.0,
        daily_loss_limit=0.06,
        max_dd_halt=0.15,
    )

    result = bt.run_backtest(htf_data, ltf_data, symbol=symbol, htf_name=htf, ltf_name=ltf)

    if verbose:
        print_result(result)

    return result


def print_result(r: ScalperBacktestResult):
    """Print backtest results."""
    print(f"\n  --- RESULTS: {r.symbol} ({r.htf}+{r.ltf}) ---")
    print(f"  Period: {r.start_date} to {r.end_date}")
    print(f"  Balance: ${r.initial_balance:.0f} -> ${r.final_balance:.2f} ({r.total_return_pct:+.1f}%)")
    print(f"  CAGR: {r.cagr:.1f}%  |  Max DD: {r.max_drawdown:.1f}%  |  Sharpe: {r.sharpe_ratio:.2f}")
    print(f"  PF: {r.profit_factor:.2f}  |  Win Rate: {r.win_rate:.1f}%  |  Avg R:R: {r.avg_rr:.2f}")
    print(f"  Trades: {r.total_trades} ({r.winning_trades}W / {r.losing_trades}L)")
    print(f"  Trades/day: {r.trades_per_day:.2f}")
    if r.winning_trades > 0:
        print(f"  Avg Win: ${r.avg_win:.2f}  |  Avg Loss: ${r.avg_loss:.2f}")
        print(f"  Best: ${r.best_trade:.2f}  |  Worst: ${r.worst_trade:.2f}")

    # Exit reason breakdown
    if r.trades:
        reasons = {}
        for t in r.trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        print(f"  Exits: {reasons}")


def run_all(args):
    """Run backtest on all instruments."""
    results = []
    for symbol in INSTRUMENTS:
        try:
            r = run_single(
                symbol, htf=args.htf, ltf=args.ltf, capital=args.capital,
                risk_pct=args.risk, min_rr=args.min_rr, min_score=args.min_score,
                tp_mode=args.tp_mode, fixed_rr=args.fixed_rr,
                session_start=args.session_start, session_end=args.session_end,
                consecutive=args.consecutive, max_concurrent=args.max_concurrent,
                use_momentum=not args.no_momentum, use_breakeven=not args.no_breakeven,
                max_zone_age=args.max_zone_age, zone_touches=args.zone_touches,
                ema_period=args.ema_period, sl_buffer=args.sl_buffer,
            )
            if r and r.total_trades > 0:
                results.append(r)
        except Exception as e:
            print(f"  {symbol}: ERROR - {e}")

    # Summary table
    if results:
        print(f"\n{'='*100}")
        print(f"  SUMMARY: {len(results)} instruments tested")
        print(f"{'='*100}")
        print(f"  {'Symbol':<10} {'Trades':>6} {'WR%':>6} {'PF':>6} {'CAGR%':>8} {'DD%':>6} {'Sharpe':>7} {'R:R':>5} {'T/day':>6} {'Return%':>9}")
        print(f"  {'-'*85}")

        results.sort(key=lambda r: r.profit_factor, reverse=True)
        for r in results:
            marker = ' *' if r.profit_factor > 1.0 and r.sharpe_ratio > 0.5 else ''
            print(f"  {r.symbol:<10} {r.total_trades:>6} {r.win_rate:>5.1f}% {r.profit_factor:>5.2f} "
                  f"{r.cagr:>7.1f}% {r.max_drawdown:>5.1f}% {r.sharpe_ratio:>6.2f} {r.avg_rr:>5.2f} "
                  f"{r.trades_per_day:>5.2f} {r.total_return_pct:>8.1f}%{marker}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Supply & Demand Scalper Backtester')
    parser.add_argument('--symbol', default='XAUUSD', help='Instrument to test')
    parser.add_argument('--all', action='store_true', help='Test all instruments')
    parser.add_argument('--htf', default='H1', help='Higher timeframe for zones (H1, H4)')
    parser.add_argument('--ltf', default='M15', help='Lower timeframe for entries (M5, M15)')
    parser.add_argument('--capital', type=float, default=100.0, help='Starting capital')
    parser.add_argument('--risk', type=float, default=0.02, help='Risk per trade (0.02=2%)')
    parser.add_argument('--min-rr', type=float, default=1.5, help='Minimum R:R ratio')
    parser.add_argument('--min-score', type=float, default=5.0, help='Minimum zone score')
    parser.add_argument('--tp-mode', default='fixed_rr', choices=['fixed_rr', 'recent_swing', 'fvg_target'])
    parser.add_argument('--fixed-rr', type=float, default=2.0, help='Fixed R:R target')
    parser.add_argument('--session-start', type=int, default=-1, help='Session start hour UTC (-1=off)')
    parser.add_argument('--session-end', type=int, default=-1, help='Session end hour UTC (-1=off)')
    parser.add_argument('--consecutive', type=int, default=4, help='Min consecutive candles')
    parser.add_argument('--max-concurrent', type=int, default=3, help='Max concurrent positions')
    parser.add_argument('--no-momentum', action='store_true', help='Disable momentum filter')
    parser.add_argument('--no-breakeven', action='store_true', help='Disable breakeven stops')
    parser.add_argument('--max-zone-age', type=int, default=200, help='Max zone age in HTF bars')
    parser.add_argument('--zone-touches', type=int, default=3, help='Max zone touches')
    parser.add_argument('--ema-period', type=int, default=200, help='EMA period for trend')
    parser.add_argument('--sl-buffer', type=float, default=0.3, help='SL buffer in ATR multiples')
    args = parser.parse_args()

    if args.all:
        run_all(args)
    else:
        run_single(
            args.symbol, htf=args.htf, ltf=args.ltf, capital=args.capital,
            risk_pct=args.risk, min_rr=args.min_rr, min_score=args.min_score,
            tp_mode=args.tp_mode, fixed_rr=args.fixed_rr,
            session_start=args.session_start, session_end=args.session_end,
            consecutive=args.consecutive, max_concurrent=args.max_concurrent,
            use_momentum=not args.no_momentum, use_breakeven=not args.no_breakeven,
            max_zone_age=args.max_zone_age, zone_touches=args.zone_touches,
            ema_period=args.ema_period, sl_buffer=args.sl_buffer,
        )


if __name__ == '__main__':
    main()
