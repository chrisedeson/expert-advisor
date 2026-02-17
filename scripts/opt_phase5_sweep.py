#!/usr/bin/env python3
"""Phase 5: Parameter sweep with Pareto frontier."""
import sys, itertools
from pathlib import Path
from datetime import datetime
import numpy as np

from loguru import logger
logger.remove()
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.backtesting.protected_grid_engine import ProtectedGridBacktester, BacktestResult
import pandas as pd

def P(msg=""): print(msg, flush=True)

PAIR_PARAMS = {'EURUSD': {'pip_size': 0.0001, 'pip_value_per_lot': 10.0, 'spread_pips': 0.7, 'slippage_pips': 0.2}}

def make_config(lot, mult, bb, grid, sl):
    return {
        'grid_strategy': {
            'base_lot_size': lot, 'lot_multiplier': mult, 'max_grid_levels': 5,
            'use_trend_filter': True, 'compound_on_equity': False,
            'bb_entry_mult': bb, 'grid_spacing_atr': grid, 'sl_atr_mult': sl,
        },
        'volatility_filter': {'atr_period': 14, 'avg_period': 50, 'normal_threshold': 10.0, 'crisis_threshold': 20.0, 'cooldown_days': 0},
        'circuit_breaker': {'daily_limit': 0.20, 'weekly_limit': 0.35, 'monthly_limit': 0.50},
        'crisis_detector': {'volatility_spike_threshold': 6.0, 'rapid_drawdown_threshold': 0.50, 'rapid_drawdown_days': 3, 'consecutive_stops_threshold': 15},
        'recovery_manager': {'drawdown_threshold': 1.0},
        'profit_protector': {'profit_threshold': 100.0},
    }

def load_data():
    p = project_root / 'data' / 'processed' / 'EURUSD_H1.parquet'
    df = pd.read_parquet(p)
    if not isinstance(df.index, pd.DatetimeIndex):
        df['time'] = pd.to_datetime(df['time']); df.set_index('time', inplace=True)
    if 'atr' not in df.columns:
        df['tr'] = np.maximum(df['high']-df['low'], np.maximum(abs(df['high']-df['close'].shift(1)), abs(df['low']-df['close'].shift(1))))
        df['atr'] = df['tr'].rolling(14).mean(); df.drop('tr', axis=1, inplace=True)
    return df

def run_bt(data, config):
    try:
        bt = ProtectedGridBacktester(initial_balance=500.0, config=config,
            spread_pips=0.7, slippage_pips=0.2, pip_size=0.0001, pip_value_per_lot=10.0)
        return bt.run_backtest(data)
    except: return None

def main():
    t0 = datetime.now()
    P(f"{'='*90}")
    P(f"PHASE 5: COMPREHENSIVE PARAMETER SWEEP")
    P(f"Started: {t0}")
    P(f"{'='*90}")

    data = load_data()

    # Focused sweep on most impactful params
    lots = [0.02, 0.03, 0.04, 0.05, 0.06]
    mults = [1.5, 2.0, 2.5]
    bbs = [2.0, 2.25, 2.5]
    grids = [0.65, 0.75, 1.0]
    sls = [1.25, 1.5, 2.0]

    total = len(lots) * len(mults) * len(bbs) * len(grids) * len(sls)
    P(f"  Combos: {total}")

    all_res = []
    count = 0
    viable = 0

    P(f"\n  VIABLE CONFIGS (CAGR>15%, DD<45%, Sharpe>0.7):")
    P(f"  {'#':>4} {'Lot':>5} {'M':>4} {'BB':>5} {'G':>5} {'SL':>4} "
      f"{'CAGR':>7} {'DD':>7} {'Shp':>6} {'PF':>6} {'Trd':>5} {'Scr':>6}")
    P(f"  {'-'*75}")

    for lot, mult, bb, grid, sl in itertools.product(lots, mults, bbs, grids, sls):
        count += 1
        cfg = make_config(lot, mult, bb, grid, sl)
        r = run_bt(data, cfg)
        if r is None: continue

        dd_pen = max(0, r.max_drawdown - 0.40) * 5
        score = (r.cagr*100*0.3 + r.sharpe_ratio*20*0.3 + r.profit_factor*10*0.2 + (1-r.max_drawdown)*30*0.2 - dd_pen*100)

        res = {'lot':lot, 'mult':mult, 'bb':bb, 'grid':grid, 'sl':sl,
               'cagr':r.cagr, 'dd':r.max_drawdown, 'sharpe':r.sharpe_ratio,
               'pf':r.profit_factor, 'trades':r.total_trades, 'final':r.final_balance, 'score':score}
        all_res.append(res)

        if r.cagr > 0.15 and r.max_drawdown < 0.45 and r.sharpe_ratio > 0.7:
            viable += 1
            mk = " ***" if r.max_drawdown < 0.40 and r.sharpe_ratio > 1.0 else ""
            P(f"  {count:>4} {lot:>5.3f} {mult:>4.1f} {bb:>5.2f} {grid:>5.2f} {sl:>4.1f} "
              f"{r.cagr*100:>6.1f}% {r.max_drawdown*100:>6.1f}% {r.sharpe_ratio:>6.2f} "
              f"{r.profit_factor:>6.2f} {r.total_trades:>5} {score:>6.1f}{mk}")

        if count % 50 == 0:
            P(f"  ... [{count}/{total}] {viable} viable ...")

    all_res.sort(key=lambda x: x['score'], reverse=True)

    P(f"\n  TOP 20 (by composite score):")
    P(f"  {'#':>3} {'Lot':>5} {'M':>4} {'BB':>5} {'G':>5} {'SL':>4} {'CAGR':>7} {'DD':>7} {'Shp':>6} {'PF':>6} {'Scr':>6}")
    P(f"  {'-'*65}")
    for i, r in enumerate(all_res[:20]):
        mk = " ***" if r['dd'] < 0.40 and r['sharpe'] > 1.0 else ""
        P(f"  {i+1:>3} {r['lot']:>5.3f} {r['mult']:>4.1f} {r['bb']:>5.2f} {r['grid']:>5.2f} {r['sl']:>4.1f} "
          f"{r['cagr']*100:>6.1f}% {r['dd']*100:>6.1f}% {r['sharpe']:>6.2f} {r['pf']:>6.2f} {r['score']:>6.1f}{mk}")

    # Pareto frontier
    pareto = []
    for i, a in enumerate(all_res):
        dom = False
        for j, b in enumerate(all_res):
            if i != j and b['cagr'] >= a['cagr'] and b['dd'] <= a['dd'] and b['sharpe'] >= a['sharpe'] and \
               (b['cagr'] > a['cagr'] or b['dd'] < a['dd'] or b['sharpe'] > a['sharpe']):
                dom = True; break
        if not dom: pareto.append(a)
    pareto.sort(key=lambda x: x['cagr'])

    P(f"\n  PARETO FRONTIER ({len(pareto)} non-dominated configs):")
    P(f"  {'#':>3} {'Lot':>5} {'M':>4} {'BB':>5} {'G':>5} {'SL':>4} {'CAGR':>7} {'DD':>7} {'Shp':>6} {'PF':>6} {'Profile':>15}")
    P(f"  {'-'*75}")
    for i, r in enumerate(pareto):
        prof = "Conservative" if r['dd']<0.25 else "Moderate" if r['dd']<0.35 else "Aggressive" if r['dd']<0.45 else "High Risk"
        P(f"  {i+1:>3} {r['lot']:>5.3f} {r['mult']:>4.1f} {r['bb']:>5.2f} {r['grid']:>5.2f} {r['sl']:>4.1f} "
          f"{r['cagr']*100:>6.1f}% {r['dd']*100:>6.1f}% {r['sharpe']:>6.2f} {r['pf']:>6.2f} {prof:>15}")

    elapsed = (datetime.now() - t0).total_seconds()
    P(f"\n{'='*80}")
    P(f"PHASE 5 COMPLETE ({elapsed/60:.1f} min) - {len(all_res)} tested, {viable} viable, {len(pareto)} Pareto-optimal")
    P(f"{'='*80}")

    # Save pareto to file for Phase 6
    import json
    with open('/tmp/pareto_configs.json', 'w') as f:
        json.dump(pareto, f)
    P(f"  Pareto configs saved to /tmp/pareto_configs.json")

if __name__ == '__main__':
    main()
