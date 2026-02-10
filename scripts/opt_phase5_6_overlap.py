#!/usr/bin/env python3
"""Phase 5+6: Parameter sweep on overlap-filtered data + MC validation.

Key finding: London/NY overlap (12-16 UTC) dramatically improves quality.
Conservative: Sharpe 1.43, PF 2.61, DD 9.2%
Aggressive: Sharpe 1.42, PF 3.49, DD 25.2%

This script:
1. Sweeps parameters on overlap-filtered data
2. Also sweeps on NY-session data (12-21 UTC)
3. MC validates top configs from both
4. Compares with full-data (no filter) sweep
"""
import sys, copy, itertools, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from loguru import logger
logger.remove()
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))
from src.backtesting.protected_grid_engine import ProtectedGridBacktester, BacktestResult

def P(msg=""): print(msg, flush=True)

def make_config(lot, mult, bb=2.0, grid=0.75, sl=1.5):
    return {
        'grid_strategy': {
            'base_lot_size': lot, 'lot_multiplier': mult, 'max_grid_levels': 5,
            'use_trend_filter': True, 'compound_on_equity': False,
            'bb_entry_mult': bb, 'grid_spacing_atr': grid, 'sl_atr_mult': sl,
        },
        'volatility_filter': {'atr_period':14,'avg_period':50,'normal_threshold':10.0,'crisis_threshold':20.0,'cooldown_days':0},
        'circuit_breaker': {'daily_limit':0.20,'weekly_limit':0.35,'monthly_limit':0.50},
        'crisis_detector': {'volatility_spike_threshold':6.0,'rapid_drawdown_threshold':0.50,'rapid_drawdown_days':3,'consecutive_stops_threshold':15},
        'recovery_manager': {'drawdown_threshold': 1.0},
        'profit_protector': {'profit_threshold': 100.0},
    }

def load_eur_h1():
    p = project_root / 'data' / 'processed' / 'EURUSD_H1.parquet'
    df = pd.read_parquet(p)
    if not isinstance(df.index, pd.DatetimeIndex):
        df['time'] = pd.to_datetime(df['time']); df.set_index('time', inplace=True)
    if 'atr' not in df.columns:
        df['tr'] = np.maximum(df['high']-df['low'], np.maximum(abs(df['high']-df['close'].shift(1)), abs(df['low']-df['close'].shift(1))))
        df['atr'] = df['tr'].rolling(14).mean(); df.drop('tr', axis=1, inplace=True)
    return df

def run_bt(data, config, capital=500.0):
    try:
        bt = ProtectedGridBacktester(initial_balance=capital, config=config,
            spread_pips=0.7, slippage_pips=0.2, pip_size=0.0001, pip_value_per_lot=10.0)
        return bt.run_backtest(data)
    except: return None

def monte_carlo(data, config, n_sims=20, label=""):
    P(f"\n  MC: {label} ({n_sims} sims)")
    np.random.seed(42)
    cagrs, dds, sharpes, pfs = [], [], [], []
    for sim in range(n_sims):
        spread = np.random.uniform(0.5, 2.0)
        slip = np.random.uniform(0.0, 1.0)
        off = np.random.randint(0, min(500, len(data)//10))
        sd = data.iloc[off:].copy()
        sc = copy.deepcopy(config)
        sc['grid_strategy']['base_lot_size'] *= np.random.uniform(0.9, 1.1)
        mask = np.random.random(len(sd)) < 0.05
        sd.loc[mask, 'close'] = sd.loc[mask, 'open']
        try:
            bt = ProtectedGridBacktester(initial_balance=500.0, config=sc,
                spread_pips=spread, slippage_pips=slip, pip_size=0.0001, pip_value_per_lot=10.0)
            r = bt.run_backtest(sd)
            cagrs.append(r.cagr); dds.append(r.max_drawdown); sharpes.append(r.sharpe_ratio); pfs.append(r.profit_factor)
        except:
            cagrs.append(-1.0); dds.append(1.0); sharpes.append(0.0); pfs.append(0.0)
        if (sim+1) % 5 == 0:
            good = sum(1 for c in cagrs if c > 0)
            P(f"    [{sim+1}/{n_sims}] Prof: {good}/{sim+1} ({good/(sim+1)*100:.0f}%)")
    c,d,s,p = [np.array(x) for x in [cagrs,dds,sharpes,pfs]]
    v = c > -0.99; nv = v.sum()
    if nv == 0:
        P("    ALL FAILED")
        return {'passes':False,'profitable_pct':0,'cagr_median':0,'cagr_p5':0,'dd_p95':100,'dd_median':100,'sharpe_median':0,'sharpe_p5':0,'pf_median':0}
    pp=(c[v]>0).mean()*100; cm=np.median(c[v])*100; c5=np.percentile(c[v],5)*100
    c25=np.percentile(c[v],25)*100; c75=np.percentile(c[v],75)*100; c95=np.percentile(c[v],95)*100
    dm=np.median(d[v])*100; d95=np.percentile(d[v],95)*100; dmax=np.max(d[v])*100
    sm=np.median(s[v]); s5=np.percentile(s[v],5); pm=np.median(p[v])
    passes = pp >= 90 and c5 > 0 and d95 < 50 and s5 > 0.3
    P(f"    Prof:{pp:.0f}% CAGR:P5={c5:.1f}% Med={cm:.1f}% DD:P95={d95:.1f}% Shp:P5={s5:.2f} Med={sm:.2f} -> {'PASS' if passes else 'FAIL'}")
    return {'passes':passes,'profitable_pct':pp,'cagr_median':cm,'cagr_p5':c5,'cagr_p25':c25,
            'cagr_p75':c75,'cagr_p95':c95,'dd_median':dm,'dd_p95':d95,'dd_max':dmax,
            'sharpe_median':sm,'sharpe_p5':s5,'pf_median':pm}


def sweep_and_validate(data, filter_name, lots, mults, bbs, grids, sls):
    """Sweep parameters, find Pareto frontier, MC validate top configs."""
    total = len(lots)*len(mults)*len(bbs)*len(grids)*len(sls)
    P(f"\n  Sweep: {total} combos on {filter_name} ({len(data)} bars)")
    P(f"  {'#':>4} {'Lot':>5} {'M':>4} {'BB':>5} {'G':>5} {'SL':>4} {'CAGR':>7} {'DD':>7} {'Shp':>6} {'PF':>6} {'Trd':>5}")
    P(f"  {'-'*65}")

    all_res = []; count = 0; viable = 0
    for lot,mult,bb,grid,sl in itertools.product(lots,mults,bbs,grids,sls):
        count += 1
        cfg = make_config(lot,mult,bb,grid,sl)
        r = run_bt(data, cfg)
        if r is None: continue
        dd_pen = max(0, r.max_drawdown-0.40)*5
        score = r.cagr*100*0.3 + r.sharpe_ratio*20*0.3 + r.profit_factor*10*0.2 + (1-r.max_drawdown)*30*0.2 - dd_pen*100
        res = {'lot':lot,'mult':mult,'bb':bb,'grid':grid,'sl':sl,'cagr':r.cagr,'dd':r.max_drawdown,
               'sharpe':r.sharpe_ratio,'pf':r.profit_factor,'trades':r.total_trades,'final':r.final_balance,'score':score}
        all_res.append(res)
        if r.cagr > 0.10 and r.max_drawdown < 0.40 and r.sharpe_ratio > 0.8:
            viable += 1
            mk = " ***" if r.max_drawdown<0.30 and r.sharpe_ratio>1.0 else ""
            P(f"  {count:>4} {lot:>5.3f} {mult:>4.1f} {bb:>5.2f} {grid:>5.2f} {sl:>4.1f} "
              f"{r.cagr*100:>6.1f}% {r.max_drawdown*100:>6.1f}% {r.sharpe_ratio:>6.2f} {r.profit_factor:>6.2f} {r.total_trades:>5}{mk}")
        if count % 30 == 0:
            P(f"  ... [{count}/{total}] {viable} viable ...")

    if not all_res:
        P("  No results!"); return [], [], []

    all_res.sort(key=lambda x: x['score'], reverse=True)

    # Pareto
    pareto = []
    for i,a in enumerate(all_res):
        dom = False
        for j,b in enumerate(all_res):
            if i!=j and b['cagr']>=a['cagr'] and b['dd']<=a['dd'] and b['sharpe']>=a['sharpe'] and \
               (b['cagr']>a['cagr'] or b['dd']<a['dd'] or b['sharpe']>a['sharpe']):
                dom=True; break
        if not dom: pareto.append(a)
    pareto.sort(key=lambda x: x['cagr'])

    P(f"\n  TOP 10:")
    for i,r in enumerate(all_res[:10]):
        mk = " ***" if r['dd']<0.30 and r['sharpe']>1.0 else ""
        P(f"  {i+1:>3} l={r['lot']:.3f} m={r['mult']:.1f} bb={r['bb']:.2f} g={r['grid']:.2f} sl={r['sl']:.1f} "
          f"CAGR={r['cagr']*100:.1f}% DD={r['dd']*100:.1f}% Shp={r['sharpe']:.2f} PF={r['pf']:.2f}{mk}")

    P(f"\n  PARETO ({len(pareto)} configs):")
    for i,r in enumerate(pareto):
        prof = "Conservative" if r['dd']<0.20 else "Moderate" if r['dd']<0.30 else "Growth" if r['dd']<0.40 else "Aggressive"
        P(f"  {i+1:>3} l={r['lot']:.3f} m={r['mult']:.1f} bb={r['bb']:.2f} g={r['grid']:.2f} sl={r['sl']:.1f} "
          f"CAGR={r['cagr']*100:.1f}% DD={r['dd']*100:.1f}% Shp={r['sharpe']:.2f} PF={r['pf']:.2f} [{prof}]")

    # MC validate top 5 configs with DD < 40%
    P(f"\n  MC VALIDATION of top configs (DD < 40%):")
    candidates = [c for c in pareto if c['dd'] < 0.40]
    if len(candidates) > 6:
        candidates.sort(key=lambda x: x['score'], reverse=True)
        candidates = candidates[:6]

    mc_results = []
    for cfg in candidates:
        config = make_config(cfg['lot'], cfg['mult'], cfg['bb'], cfg['grid'], cfg['sl'])
        label = f"l={cfg['lot']:.3f} m={cfg['mult']:.1f} bb={cfg['bb']:.2f} g={cfg['grid']:.2f} sl={cfg['sl']:.1f}"
        mc = monte_carlo(data, config, 20, label)
        mc['config'] = cfg
        mc_results.append(mc)

    return all_res, pareto, mc_results


def main():
    t0 = datetime.now()
    P(f"{'='*90}")
    P(f"PHASE 5+6: PARAMETER SWEEP + MC ON SESSION-FILTERED DATA")
    P(f"Started: {t0}")
    P(f"{'='*90}")

    eur = load_eur_h1()

    # Create filtered datasets
    overlap = eur[(eur.index.hour >= 12) & (eur.index.hour <= 16)]  # London/NY overlap
    ny_sess = eur[(eur.index.hour >= 12) & (eur.index.hour <= 21)]  # NY session
    full = eur  # Full data for comparison

    P(f"  Data: Full={len(full)} bars, NY={len(ny_sess)} bars, Overlap={len(overlap)} bars")

    # Parameter grid (focused)
    lots = [0.02, 0.03, 0.04, 0.05, 0.06]
    mults = [1.5, 2.0, 2.5]
    bbs = [2.0, 2.25, 2.5]
    grids = [0.75, 1.0]
    sls = [1.25, 1.5, 2.0]

    # === SWEEP 1: Overlap filter (best quality) ===
    P(f"\n{'#'*80}")
    P(f"# SWEEP 1: OVERLAP FILTER (12-16 UTC)")
    P(f"{'#'*80}")
    ol_all, ol_pareto, ol_mc = sweep_and_validate(overlap, "Overlap 12-16", lots, mults, bbs, grids, sls)

    # === SWEEP 2: NY session ===
    P(f"\n{'#'*80}")
    P(f"# SWEEP 2: NY SESSION (12-21 UTC)")
    P(f"{'#'*80}")
    ny_all, ny_pareto, ny_mc = sweep_and_validate(ny_sess, "NY 12-21", lots, mults, bbs, grids, sls)

    # === SWEEP 3: Full data (baseline) ===
    P(f"\n{'#'*80}")
    P(f"# SWEEP 3: FULL DATA (no filter)")
    P(f"{'#'*80}")
    full_all, full_pareto, full_mc = sweep_and_validate(full, "Full", lots, mults, bbs, grids, sls)

    # === FINAL SUMMARY ===
    elapsed = (datetime.now() - t0).total_seconds()
    P(f"\n\n{'='*90}")
    P(f"FINAL SUMMARY ({elapsed/60:.1f} min)")
    P(f"{'='*90}")

    for name, mc_list in [("Overlap", ol_mc), ("NY Session", ny_mc), ("Full Data", full_mc)]:
        P(f"\n  {name}:")
        passed = [m for m in mc_list if m['passes']]
        P(f"    MC Passed: {len(passed)}/{len(mc_list)}")
        if passed:
            best = max(passed, key=lambda x: x['sharpe_median'])
            c = best['config']
            P(f"    Best (Sharpe): l={c['lot']:.3f} m={c['mult']:.1f} bb={c['bb']:.2f} g={c['grid']:.2f} sl={c['sl']:.1f}")
            P(f"      CAGR: P5={best['cagr_p5']:.1f}% Med={best['cagr_median']:.1f}%")
            P(f"      DD: P95={best['dd_p95']:.1f}% | Sharpe: Med={best['sharpe_median']:.2f}")

            safe = [m for m in passed if m['dd_p95'] < 35]
            if safe:
                best_safe = max(safe, key=lambda x: x['cagr_median'])
                c = best_safe['config']
                P(f"    Safest high-CAGR: l={c['lot']:.3f} m={c['mult']:.1f} bb={c['bb']:.2f} g={c['grid']:.2f} sl={c['sl']:.1f}")
                P(f"      CAGR: P5={best_safe['cagr_p5']:.1f}% Med={best_safe['cagr_median']:.1f}%")
                P(f"      DD: P95={best_safe['dd_p95']:.1f}% | Sharpe: Med={best_safe['sharpe_median']:.2f}")

    # Save results
    results = {
        'overlap_pareto': ol_pareto,
        'ny_pareto': ny_pareto,
        'full_pareto': full_pareto,
        'overlap_mc': [{k:v for k,v in m.items() if k != 'config'} for m in ol_mc],
        'ny_mc': [{k:v for k,v in m.items() if k != 'config'} for m in ny_mc],
        'full_mc': [{k:v for k,v in m.items() if k != 'config'} for m in full_mc],
    }
    with open('/tmp/sweep_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    P(f"\n  Results saved to /tmp/sweep_results.json")
    P(f"{'='*90}")


if __name__ == '__main__':
    main()
