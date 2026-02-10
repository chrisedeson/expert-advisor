#!/usr/bin/env python3
"""
Comprehensive EA Optimization Suite

Performs ALL optimization tasks:
1. Monte Carlo stress testing
2. XAUUSD (Gold) and H4 timeframe testing
3. Session filters (London/NY overlap)
4. Walk-forward validation
5. Full parameter sweep with Pareto frontier
6. Final Monte Carlo validation of top configs

Run: source .venv/bin/activate && python -u scripts/comprehensive_optimization.py 2>/dev/null | tee /tmp/optimization_results.txt
"""

import sys
import os
import copy
import itertools
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

# Suppress ALL loguru output BEFORE importing anything that uses it
from loguru import logger
logger.remove()

import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.backtesting.protected_grid_engine import (
    ProtectedGridBacktester,
    BacktestResult,
)


def P(msg=""):
    """Print with immediate flush."""
    print(msg, flush=True)


# ============================================================================
# CONFIGURATION
# ============================================================================

PAIR_PARAMS = {
    'EURUSD': {'pip_size': 0.0001, 'pip_value_per_lot': 10.0, 'spread_pips': 0.7, 'slippage_pips': 0.2},
    'GBPUSD': {'pip_size': 0.0001, 'pip_value_per_lot': 10.0, 'spread_pips': 0.9, 'slippage_pips': 0.3},
    'USDJPY': {'pip_size': 0.01, 'pip_value_per_lot': 6.67, 'spread_pips': 0.8, 'slippage_pips': 0.3},
    'AUDUSD': {'pip_size': 0.0001, 'pip_value_per_lot': 10.0, 'spread_pips': 0.8, 'slippage_pips': 0.3},
    'XAUUSD': {'pip_size': 0.01, 'pip_value_per_lot': 1.0, 'spread_pips': 3.0, 'slippage_pips': 0.5},
}


def make_config(lot=0.0228, mult=1.5, use_trend=True, compound_equity=False,
                bb_mult=2.0, grid_spacing=0.75, sl_mult=1.5,
                crisis_on=True, cb_on=True, recovery_off=True, profit_prot_off=True):
    """Create a config with full control over all parameters."""
    return {
        'grid_strategy': {
            'base_lot_size': lot,
            'lot_multiplier': mult,
            'max_grid_levels': 5,
            'use_trend_filter': use_trend,
            'compound_on_equity': compound_equity,
            'bb_entry_mult': bb_mult,
            'grid_spacing_atr': grid_spacing,
            'sl_atr_mult': sl_mult,
        },
        'volatility_filter': {
            'atr_period': 14, 'avg_period': 50,
            'normal_threshold': 10.0, 'crisis_threshold': 20.0, 'cooldown_days': 0,
        },
        'circuit_breaker': {
            'daily_limit': 0.20 if cb_on else 1.0,
            'weekly_limit': 0.35 if cb_on else 1.0,
            'monthly_limit': 0.50 if cb_on else 1.0,
        },
        'crisis_detector': {
            'volatility_spike_threshold': 6.0 if crisis_on else 100.0,
            'rapid_drawdown_threshold': 0.50 if crisis_on else 1.0,
            'rapid_drawdown_days': 3,
            'consecutive_stops_threshold': 15 if crisis_on else 999,
        },
        'recovery_manager': {'drawdown_threshold': 1.0 if recovery_off else 0.40},
        'profit_protector': {'profit_threshold': 100.0 if profit_prot_off else 1.0},
    }


def load_data(pair: str, timeframe: str = 'H1') -> Optional[pd.DataFrame]:
    """Load and prepare historical data."""
    data_path = project_root / 'data' / 'processed' / f'{pair}_{timeframe}.parquet'
    if not data_path.exists():
        return None
    df = pd.read_parquet(data_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
    if 'atr' not in df.columns:
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))
        )
        df['atr'] = df['tr'].rolling(14).mean()
        df.drop('tr', axis=1, inplace=True)
    return df


def resample_to_h4(h1_data: pd.DataFrame) -> pd.DataFrame:
    """Resample H1 data to H4."""
    h4 = h1_data.resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum',
    }).dropna()
    h4['tr'] = np.maximum(
        h4['high'] - h4['low'],
        np.maximum(abs(h4['high'] - h4['close'].shift(1)), abs(h4['low'] - h4['close'].shift(1)))
    )
    h4['atr'] = h4['tr'].rolling(14).mean()
    h4.drop('tr', axis=1, inplace=True)
    h4.dropna(inplace=True)
    return h4


def filter_sessions(data: pd.DataFrame, sessions: str) -> pd.DataFrame:
    """Filter data to specific trading sessions."""
    df = data.copy()
    hours = df.index.hour
    masks = {
        'london_ny': (hours >= 7) & (hours <= 17),
        'london': (hours >= 7) & (hours <= 16),
        'ny': (hours >= 12) & (hours <= 21),
        'overlap': (hours >= 12) & (hours <= 16),
        'asian': (hours >= 0) & (hours <= 8),
    }
    return df[masks.get(sessions, slice(None))]


def run_bt(data, config, pair, capital=500.0):
    """Run a single backtest."""
    params = PAIR_PARAMS.get(pair, PAIR_PARAMS['EURUSD'])
    try:
        bt = ProtectedGridBacktester(
            initial_balance=capital, config=config,
            spread_pips=params['spread_pips'], slippage_pips=params['slippage_pips'],
            pip_size=params['pip_size'], pip_value_per_lot=params['pip_value_per_lot'],
        )
        return bt.run_backtest(data)
    except Exception:
        return None


# ============================================================================
# PHASE 1: MONTE CARLO
# ============================================================================

def monte_carlo(data, config, pair, capital=500.0, n_sims=50, label=""):
    """Monte Carlo stress test with randomized spread, slippage, start, and trade skips."""
    base = PAIR_PARAMS.get(pair, PAIR_PARAMS['EURUSD'])
    is_gold = pair == 'XAUUSD'

    P(f"\n{'='*80}")
    P(f"MONTE CARLO: {label or pair} ({n_sims} sims)")
    P(f"  lot={config['grid_strategy']['base_lot_size']}, mult={config['grid_strategy']['lot_multiplier']}, "
      f"bb={config['grid_strategy'].get('bb_entry_mult', 2.0)}, "
      f"grid={config['grid_strategy'].get('grid_spacing_atr', 0.75)}, "
      f"sl={config['grid_strategy'].get('sl_atr_mult', 1.5)}")
    P(f"{'='*80}")

    np.random.seed(42)
    cagrs, dds, sharpes, pfs = [], [], [], []

    for sim in range(n_sims):
        spread = np.random.uniform(2.0, 5.0) if is_gold else np.random.uniform(0.5, 2.0)
        slippage = np.random.uniform(0.0, 2.0) if is_gold else np.random.uniform(0.0, 1.0)
        start_off = np.random.randint(0, min(500, len(data) // 10))
        sim_data = data.iloc[start_off:].copy()

        sim_cfg = copy.deepcopy(config)
        sim_cfg['grid_strategy']['base_lot_size'] *= np.random.uniform(0.9, 1.1)

        skip_mask = np.random.random(len(sim_data)) < 0.05
        sim_data.loc[skip_mask, 'close'] = sim_data.loc[skip_mask, 'open']

        try:
            bt = ProtectedGridBacktester(
                initial_balance=capital, config=sim_cfg,
                spread_pips=spread, slippage_pips=slippage,
                pip_size=base['pip_size'], pip_value_per_lot=base['pip_value_per_lot'],
            )
            r = bt.run_backtest(sim_data)
            cagrs.append(r.cagr); dds.append(r.max_drawdown)
            sharpes.append(r.sharpe_ratio); pfs.append(r.profit_factor)
        except Exception:
            cagrs.append(-1.0); dds.append(1.0); sharpes.append(0.0); pfs.append(0.0)

        if (sim + 1) % 10 == 0:
            good = sum(1 for c in cagrs if c > 0)
            P(f"  [{sim+1}/{n_sims}] Profitable: {good}/{sim+1} ({good/(sim+1)*100:.0f}%)")

    c, d, s, p = np.array(cagrs), np.array(dds), np.array(sharpes), np.array(pfs)
    valid = c > -0.99
    nv = valid.sum()

    if nv == 0:
        P("  ALL SIMULATIONS FAILED")
        return {'passes': False, 'profitable_pct': 0, 'cagr_median': 0, 'cagr_p5': 0,
                'dd_p95': 100, 'dd_median': 100, 'sharpe_median': 0, 'sharpe_p5': 0, 'pf_median': 0}

    prof_pct = (c[valid] > 0).mean() * 100
    cm = np.median(c[valid]) * 100
    c5 = np.percentile(c[valid], 5) * 100
    c25 = np.percentile(c[valid], 25) * 100
    c75 = np.percentile(c[valid], 75) * 100
    c95 = np.percentile(c[valid], 95) * 100
    dm = np.median(d[valid]) * 100
    d95 = np.percentile(d[valid], 95) * 100
    dmax = np.max(d[valid]) * 100
    sm = np.median(s[valid])
    s5 = np.percentile(s[valid], 5)
    pm = np.median(p[valid])

    passes = prof_pct >= 90 and c5 > 0 and d95 < 50 and s5 > 0.3

    P(f"\n  RESULTS ({nv} valid):")
    P(f"  Profitable: {prof_pct:.0f}%  |  CAGR: P5={c5:.1f}% Med={cm:.1f}% P95={c95:.1f}%")
    P(f"  DD: Med={dm:.1f}% P95={d95:.1f}% Max={dmax:.1f}%  |  Sharpe: P5={s5:.2f} Med={sm:.2f}")
    P(f"  PF: Med={pm:.2f}  |  VERDICT: {'PASS' if passes else 'FAIL'}")
    P(f"{'='*80}")

    return {
        'label': label, 'passes': passes, 'profitable_pct': prof_pct,
        'cagr_median': cm, 'cagr_p5': c5, 'cagr_p25': c25, 'cagr_p75': c75, 'cagr_p95': c95,
        'dd_median': dm, 'dd_p95': d95, 'dd_max': dmax,
        'sharpe_median': sm, 'sharpe_p5': s5, 'pf_median': pm,
    }


# ============================================================================
# PHASE 2: ASSETS & TIMEFRAMES
# ============================================================================

def test_assets_timeframes():
    P(f"\n{'#'*90}")
    P(f"# PHASE 2: ASSET & TIMEFRAME TESTING")
    P(f"{'#'*90}")

    configs = {
        'Conservative': make_config(lot=0.0228, mult=1.5),
        'Moderate': make_config(lot=0.04, mult=2.0),
        'Aggressive': make_config(lot=0.06, mult=2.5),
    }

    P(f"\n  {'Pair':>8} {'TF':>4} {'Config':>14} {'CAGR':>8} {'DD':>8} {'Sharpe':>8} {'PF':>8} {'Trades':>8} {'Final$':>10}")
    P(f"  {'-'*82}")

    results = {}
    for pair in ['EURUSD', 'GBPUSD', 'XAUUSD']:
        for tf in ['H1', 'H4']:
            data = load_data(pair, tf)
            if data is None and tf == 'H4':
                h1 = load_data(pair, 'H1')
                if h1 is not None:
                    data = resample_to_h4(h1)
            if data is None:
                P(f"  {pair:>8} {tf:>4} -- NO DATA --")
                continue

            for cn, cfg in configs.items():
                c = copy.deepcopy(cfg)
                if pair == 'XAUUSD':
                    c['grid_strategy']['base_lot_size'] *= 10.0
                r = run_bt(data, c, pair)
                if r is None:
                    P(f"  {pair:>8} {tf:>4} {cn:>14} -- FAILED --")
                    continue
                results[f"{pair}_{tf}_{cn}"] = r
                mk = " ***" if r.max_drawdown < 0.40 and r.sharpe_ratio > 1.0 and r.cagr > 0.20 else ""
                P(f"  {pair:>8} {tf:>4} {cn:>14} {r.cagr*100:>7.1f}% {r.max_drawdown*100:>7.1f}% "
                  f"{r.sharpe_ratio:>8.2f} {r.profit_factor:>8.2f} {r.total_trades:>8} ${r.final_balance:>9.0f}{mk}")
    return results


# ============================================================================
# PHASE 3: SESSION FILTERS
# ============================================================================

def test_sessions():
    P(f"\n{'#'*90}")
    P(f"# PHASE 3: SESSION FILTER TESTING")
    P(f"{'#'*90}")

    data = load_data('EURUSD', 'H1')
    if data is None:
        P("  No data!"); return {}

    P(f"\n  {'Session':>12} {'Config':>14} {'Bars':>8} {'CAGR':>8} {'DD':>8} {'Sharpe':>8} {'PF':>8} {'Trades':>8}")
    P(f"  {'-'*78}")

    results = {}
    for session in ['all', 'london_ny', 'london', 'ny', 'overlap', 'asian']:
        filt = data if session == 'all' else filter_sessions(data, session)
        for cn, cfg in [('Conservative', make_config(lot=0.0228, mult=1.5)),
                        ('Aggressive', make_config(lot=0.06, mult=2.5))]:
            r = run_bt(filt, cfg, 'EURUSD')
            if r is None: continue
            results[f"{session}_{cn}"] = r
            mk = " ***" if r.sharpe_ratio > 1.0 and r.cagr > 0.20 else ""
            P(f"  {session:>12} {cn:>14} {len(filt):>8} {r.cagr*100:>7.1f}% {r.max_drawdown*100:>7.1f}% "
              f"{r.sharpe_ratio:>8.2f} {r.profit_factor:>8.2f} {r.total_trades:>8}{mk}")
    return results


# ============================================================================
# PHASE 4: WALK-FORWARD
# ============================================================================

def walk_forward():
    P(f"\n{'#'*90}")
    P(f"# PHASE 4: WALK-FORWARD VALIDATION")
    P(f"{'#'*90}")

    data = load_data('EURUSD', 'H1')
    if data is None:
        P("  No data!"); return {}

    n = len(data)
    results = {}

    for cn, cfg in [('Conservative', make_config(lot=0.0228, mult=1.5)),
                    ('Moderate', make_config(lot=0.04, mult=2.0)),
                    ('Aggressive', make_config(lot=0.06, mult=2.5))]:
        P(f"\n  CONFIG: {cn}")
        P(f"  {'Split':>12} {'Period':>8} {'CAGR':>8} {'DD':>8} {'Sharpe':>8} {'PF':>8} {'Trades':>8}")
        P(f"  {'-'*64}")

        for sn, pct in [('70/30', 0.70), ('60/40', 0.60), ('50/50', 0.50)]:
            si = int(n * pct)
            r_is = run_bt(data.iloc[:si], cfg, 'EURUSD')
            r_oos = run_bt(data.iloc[si:], cfg, 'EURUSD')
            if r_is and r_oos:
                deg = (r_is.cagr - r_oos.cagr) / r_is.cagr * 100 if r_is.cagr > 0 else 0
                verdict = "OVERFITTED" if deg > 50 else "MOD OVERFIT" if deg > 25 else "OOS BETTER" if deg < 0 else "ROBUST"
                P(f"  {sn:>12} {'IS':>8} {r_is.cagr*100:>7.1f}% {r_is.max_drawdown*100:>7.1f}% "
                  f"{r_is.sharpe_ratio:>8.2f} {r_is.profit_factor:>8.2f} {r_is.total_trades:>8}")
                P(f"  {'':>12} {'OOS':>8} {r_oos.cagr*100:>7.1f}% {r_oos.max_drawdown*100:>7.1f}% "
                  f"{r_oos.sharpe_ratio:>8.2f} {r_oos.profit_factor:>8.2f} {r_oos.total_trades:>8}")
                P(f"  {'':>12} {'':>8}  Degradation: {deg:.0f}% -> {verdict}")
                results[f"{cn}_{sn}"] = {'is_cagr': r_is.cagr, 'oos_cagr': r_oos.cagr,
                                         'degradation': deg, 'verdict': verdict}
    return results


# ============================================================================
# PHASE 5: PARAMETER SWEEP
# ============================================================================

def parameter_sweep():
    P(f"\n{'#'*90}")
    P(f"# PHASE 5: PARAMETER SWEEP")
    P(f"{'#'*90}")

    data = load_data('EURUSD', 'H1')
    if data is None:
        P("  No data!"); return [], []

    # Focused sweep: known-good ranges only
    lots = [0.02, 0.03, 0.04, 0.05, 0.06]
    mults = [1.5, 2.0, 2.5]
    bbs = [2.0, 2.25, 2.5]
    grids = [0.65, 0.75, 1.0]
    sls = [1.25, 1.5, 2.0]

    total = len(lots) * len(mults) * len(bbs) * len(grids) * len(sls)
    P(f"  Combinations: {total}")

    all_res = []
    count = 0
    viable = 0

    P(f"\n  {'#':>4} {'Lot':>5} {'M':>4} {'BB':>5} {'G':>5} {'SL':>4} "
      f"{'CAGR':>7} {'DD':>7} {'Shp':>6} {'PF':>6} {'Trd':>5} {'Scr':>6}")
    P(f"  {'-'*75}")

    for lot, mult, bb, grid, sl in itertools.product(lots, mults, bbs, grids, sls):
        count += 1
        if mult >= 3.0 and lot >= 0.06:
            continue

        cfg = make_config(lot=lot, mult=mult, bb_mult=bb, grid_spacing=grid, sl_mult=sl)
        r = run_bt(data, cfg, 'EURUSD')
        if r is None:
            continue

        dd_pen = max(0, r.max_drawdown - 0.40) * 5
        score = (r.cagr * 100 * 0.3 + r.sharpe_ratio * 20 * 0.3 +
                 r.profit_factor * 10 * 0.2 + (1 - r.max_drawdown) * 30 * 0.2 - dd_pen * 100)

        res = {'lot': lot, 'mult': mult, 'bb': bb, 'grid': grid, 'sl': sl,
               'cagr': r.cagr, 'dd': r.max_drawdown, 'sharpe': r.sharpe_ratio,
               'pf': r.profit_factor, 'trades': r.total_trades, 'final': r.final_balance, 'score': score}
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

    P(f"\n  TOP 20 (by score):")
    P(f"  {'#':>3} {'Lot':>5} {'M':>4} {'BB':>5} {'G':>5} {'SL':>4} "
      f"{'CAGR':>7} {'DD':>7} {'Shp':>6} {'PF':>6} {'Scr':>6}")
    P(f"  {'-'*60}")
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
        if not dom:
            pareto.append(a)
    pareto.sort(key=lambda x: x['cagr'])

    P(f"\n  PARETO FRONTIER ({len(pareto)} configs):")
    P(f"  {'#':>3} {'Lot':>5} {'M':>4} {'BB':>5} {'G':>5} {'SL':>4} "
      f"{'CAGR':>7} {'DD':>7} {'Shp':>6} {'PF':>6} {'Profile':>15}")
    P(f"  {'-'*75}")
    for i, r in enumerate(pareto):
        prof = "Conservative" if r['dd'] < 0.25 else "Moderate" if r['dd'] < 0.35 else "Aggressive" if r['dd'] < 0.45 else "High Risk"
        P(f"  {i+1:>3} {r['lot']:>5.3f} {r['mult']:>4.1f} {r['bb']:>5.2f} {r['grid']:>5.2f} {r['sl']:>4.1f} "
          f"{r['cagr']*100:>6.1f}% {r['dd']*100:>6.1f}% {r['sharpe']:>6.2f} {r['pf']:>6.2f} {prof:>15}")

    return all_res, pareto


# ============================================================================
# PHASE 6: VALIDATE PARETO WITH MC
# ============================================================================

def validate_pareto(pareto):
    P(f"\n{'#'*90}")
    P(f"# PHASE 6: MONTE CARLO VALIDATION OF PARETO CONFIGS")
    P(f"{'#'*90}")

    data = load_data('EURUSD', 'H1')
    if data is None:
        return []

    # Only validate configs within our target zone
    candidates = [c for c in pareto if c['dd'] < 0.45]
    if len(candidates) > 8:
        candidates.sort(key=lambda x: x['score'], reverse=True)
        candidates = candidates[:8]

    mc_results = []
    for i, cfg in enumerate(candidates):
        config = make_config(lot=cfg['lot'], mult=cfg['mult'], bb_mult=cfg['bb'],
                             grid_spacing=cfg['grid'], sl_mult=cfg['sl'])
        label = f"#{i+1}: l={cfg['lot']} m={cfg['mult']} bb={cfg['bb']} g={cfg['grid']} sl={cfg['sl']}"
        mc = monte_carlo(data, config, 'EURUSD', n_sims=50, label=label)
        mc['config'] = cfg
        mc_results.append(mc)

    P(f"\n{'='*90}")
    P(f"MC VALIDATION SUMMARY")
    P(f"{'='*90}")
    P(f"  {'#':>3} {'Lot':>5} {'M':>4} {'BB':>5} {'G':>5} {'SL':>4} "
      f"{'BT%':>6} {'MC Med':>7} {'MC P5':>7} {'DD95':>6} {'Pass':>5}")
    P(f"  {'-'*65}")
    for mc in mc_results:
        c = mc['config']
        P(f"  {mc_results.index(mc)+1:>3} {c['lot']:>5.3f} {c['mult']:>4.1f} {c['bb']:>5.2f} "
          f"{c['grid']:>5.2f} {c['sl']:>4.1f} {c['cagr']*100:>5.1f}% {mc['cagr_median']:>6.1f}% "
          f"{mc['cagr_p5']:>6.1f}% {mc['dd_p95']:>5.1f}% {'PASS' if mc['passes'] else 'FAIL':>5}")

    passed = sum(1 for m in mc_results if m['passes'])
    P(f"\n  {passed}/{len(mc_results)} PASSED Monte Carlo validation")
    return mc_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = datetime.now()
    P(f"{'='*90}")
    P(f"COMPREHENSIVE EA OPTIMIZATION SUITE")
    P(f"Started: {t0}")
    P(f"{'='*90}")

    eur = load_data('EURUSD', 'H1')

    # --- PHASE 1: MC on existing configs ---
    P(f"\n{'#'*90}")
    P(f"# PHASE 1: MONTE CARLO ON CURRENT BEST CONFIGS")
    P(f"{'#'*90}")

    if eur is not None:
        mc_con = monte_carlo(eur, make_config(lot=0.0228, mult=1.5), 'EURUSD', n_sims=50,
                             label="Conservative (lot=0.0228, mult=1.5)")
        mc_mod = monte_carlo(eur, make_config(lot=0.04, mult=2.0), 'EURUSD', n_sims=50,
                             label="Moderate (lot=0.04, mult=2.0)")
        mc_agg = monte_carlo(eur, make_config(lot=0.06, mult=2.5), 'EURUSD', n_sims=50,
                             label="Aggressive (lot=0.06, mult=2.5)")

    # --- PHASE 2 ---
    asset_res = test_assets_timeframes()

    # --- PHASE 3 ---
    session_res = test_sessions()

    # --- PHASE 4 ---
    wf_res = walk_forward()

    # --- PHASE 5 ---
    sweep_res, pareto = parameter_sweep()

    # --- PHASE 6 ---
    mc_val = validate_pareto(pareto)

    # --- FINAL SUMMARY ---
    elapsed = (datetime.now() - t0).total_seconds()
    P(f"\n\n{'='*90}")
    P(f"FINAL SUMMARY ({elapsed/60:.1f} min)")
    P(f"{'='*90}")
    P(f"  Combos tested: {len(sweep_res)}")
    P(f"  Pareto configs: {len(pareto)}")
    P(f"  MC validated: {sum(1 for m in mc_val if m['passes'])}")

    validated = [m for m in mc_val if m['passes']]
    if validated:
        best_s = max(validated, key=lambda x: x['sharpe_median'])
        c = best_s['config']
        P(f"\n  BEST RISK-ADJUSTED: lot={c['lot']}, mult={c['mult']}, bb={c['bb']}, grid={c['grid']}, sl={c['sl']}")
        P(f"    MC CAGR: Med={best_s['cagr_median']:.1f}% P5={best_s['cagr_p5']:.1f}%")
        P(f"    MC DD: Med={best_s['dd_median']:.1f}% P95={best_s['dd_p95']:.1f}%")
        P(f"    MC Sharpe: {best_s['sharpe_median']:.2f}")

        safe = [m for m in validated if m['dd_p95'] < 40]
        if safe:
            best_c = max(safe, key=lambda x: x['cagr_median'])
            c = best_c['config']
            P(f"\n  BEST CAGR (DD<40%): lot={c['lot']}, mult={c['mult']}, bb={c['bb']}, grid={c['grid']}, sl={c['sl']}")
            P(f"    MC CAGR: Med={best_c['cagr_median']:.1f}% P5={best_c['cagr_p5']:.1f}%")
            P(f"    MC DD: Med={best_c['dd_median']:.1f}% P95={best_c['dd_p95']:.1f}%")

        most_c = min(validated, key=lambda x: x['dd_p95'])
        c = most_c['config']
        P(f"\n  SAFEST: lot={c['lot']}, mult={c['mult']}, bb={c['bb']}, grid={c['grid']}, sl={c['sl']}")
        P(f"    MC CAGR: Med={most_c['cagr_median']:.1f}% P5={most_c['cagr_p5']:.1f}%")
        P(f"    MC DD: Med={most_c['dd_median']:.1f}% P95={most_c['dd_p95']:.1f}%")
    else:
        P("\n  WARNING: No configs passed MC validation!")

    P(f"\n  Phase 1 MC Results:")
    if eur is not None:
        P(f"    Conservative: {'PASS' if mc_con['passes'] else 'FAIL'} | CAGR P5={mc_con['cagr_p5']:.1f}% Med={mc_con['cagr_median']:.1f}% | DD P95={mc_con['dd_p95']:.1f}%")
        P(f"    Moderate:     {'PASS' if mc_mod['passes'] else 'FAIL'} | CAGR P5={mc_mod['cagr_p5']:.1f}% Med={mc_mod['cagr_median']:.1f}% | DD P95={mc_mod['dd_p95']:.1f}%")
        P(f"    Aggressive:   {'PASS' if mc_agg['passes'] else 'FAIL'} | CAGR P5={mc_agg['cagr_p5']:.1f}% Med={mc_agg['cagr_median']:.1f}% | DD P95={mc_agg['dd_p95']:.1f}%")

    P(f"\n{'='*90}")


if __name__ == '__main__':
    main()
