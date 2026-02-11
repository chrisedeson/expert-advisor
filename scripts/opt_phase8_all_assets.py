#!/usr/bin/env python3
"""Phase 8: Multi-Asset Backtest - Test ALL 20 instruments with overlap strategy.

Tests each instrument on H1 with:
1. Full data (baseline)
2. Overlap filter (12-16 UTC)
3. MC validation on winners

Finds which instruments work with our BB grid strategy.
"""
import sys, copy, itertools
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

# Instrument specifications
# pip_size: minimum price movement
# pip_value_per_lot: USD value of 1 pip for 1 standard lot
# spread_pips: typical spread in pips
# slippage_pips: typical slippage in pips
# lot_scale: lot size relative to EURUSD (to keep risk similar with $500 account)
INSTRUMENTS = {
    # Major Forex
    'EURUSD':  {'pip_size': 0.0001, 'pip_value': 10.0,  'spread': 0.7,  'slip': 0.2, 'lot_scale': 1.0,  'class': 'forex_major'},
    'GBPUSD':  {'pip_size': 0.0001, 'pip_value': 10.0,  'spread': 0.9,  'slip': 0.3, 'lot_scale': 1.0,  'class': 'forex_major'},
    'USDJPY':  {'pip_size': 0.01,   'pip_value': 6.67,  'spread': 0.8,  'slip': 0.3, 'lot_scale': 1.0,  'class': 'forex_major'},
    'AUDUSD':  {'pip_size': 0.0001, 'pip_value': 10.0,  'spread': 0.8,  'slip': 0.3, 'lot_scale': 1.0,  'class': 'forex_major'},
    'NZDUSD':  {'pip_size': 0.0001, 'pip_value': 10.0,  'spread': 1.0,  'slip': 0.3, 'lot_scale': 1.0,  'class': 'forex_major'},
    'USDCAD':  {'pip_size': 0.0001, 'pip_value': 7.50,  'spread': 1.0,  'slip': 0.3, 'lot_scale': 1.0,  'class': 'forex_major'},
    'USDCHF':  {'pip_size': 0.0001, 'pip_value': 10.80, 'spread': 1.0,  'slip': 0.3, 'lot_scale': 1.0,  'class': 'forex_major'},

    # Cross Forex
    'EURGBP':  {'pip_size': 0.0001, 'pip_value': 12.50, 'spread': 1.0,  'slip': 0.3, 'lot_scale': 1.0,  'class': 'forex_cross'},
    'EURJPY':  {'pip_size': 0.01,   'pip_value': 6.67,  'spread': 1.0,  'slip': 0.3, 'lot_scale': 1.0,  'class': 'forex_cross'},
    'EURAUD':  {'pip_size': 0.0001, 'pip_value': 6.50,  'spread': 1.5,  'slip': 0.5, 'lot_scale': 1.0,  'class': 'forex_cross'},
    'EURCHF':  {'pip_size': 0.0001, 'pip_value': 10.80, 'spread': 1.5,  'slip': 0.5, 'lot_scale': 1.0,  'class': 'forex_cross'},
    'GBPJPY':  {'pip_size': 0.01,   'pip_value': 6.67,  'spread': 1.5,  'slip': 0.5, 'lot_scale': 1.0,  'class': 'forex_cross'},

    # Crypto - much higher volatility, need smaller lots
    'BTCUSD':  {'pip_size': 0.01,   'pip_value': 1.0,   'spread': 50.0, 'slip': 10.0, 'lot_scale': 0.01, 'class': 'crypto'},
    'ETHUSD':  {'pip_size': 0.01,   'pip_value': 1.0,   'spread': 3.0,  'slip': 1.0,  'lot_scale': 0.1,  'class': 'crypto'},
    'XRPUSD':  {'pip_size': 0.0001, 'pip_value': 10.0,  'spread': 5.0,  'slip': 1.0,  'lot_scale': 1.0,  'class': 'crypto'},

    # Metals
    'XAUUSD':  {'pip_size': 0.01,   'pip_value': 1.0,   'spread': 3.0,  'slip': 0.5,  'lot_scale': 0.1,  'class': 'metals'},
    'XAGUSD':  {'pip_size': 0.001,  'pip_value': 5.0,   'spread': 3.0,  'slip': 0.5,  'lot_scale': 0.1,  'class': 'metals'},

    # Indices - CFDs
    'US30':    {'pip_size': 0.01,   'pip_value': 1.0,   'spread': 3.0,  'slip': 1.0,  'lot_scale': 0.01, 'class': 'indices'},
    'US500':   {'pip_size': 0.01,   'pip_value': 1.0,   'spread': 0.5,  'slip': 0.2,  'lot_scale': 0.1,  'class': 'indices'},
    'USTEC':   {'pip_size': 0.01,   'pip_value': 1.0,   'spread': 2.0,  'slip': 0.5,  'lot_scale': 0.01, 'class': 'indices'},
}

def make_config(lot, mult=2.0, bb=2.0, grid=0.75, sl=1.5):
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


def load_data(symbol):
    p = project_root / 'data' / 'processed' / f'{symbol}_H1.parquet'
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
    # Compute ATR if missing
    if 'atr' not in df.columns:
        df['tr'] = np.maximum(df['high']-df['low'],
                    np.maximum(abs(df['high']-df['close'].shift(1)),
                               abs(df['low']-df['close'].shift(1))))
        df['atr'] = df['tr'].rolling(14).mean()
        df.drop('tr', axis=1, inplace=True)
    return df


def run_bt(data, config, capital, spec):
    try:
        bt = ProtectedGridBacktester(
            initial_balance=capital,
            config=config,
            spread_pips=spec['spread'],
            slippage_pips=spec['slip'],
            pip_size=spec['pip_size'],
            pip_value_per_lot=spec['pip_value']
        )
        return bt.run_backtest(data)
    except Exception as e:
        return None


def monte_carlo(data, config, spec, n_sims=50, label=""):
    P(f"    MC: {label} ({n_sims} sims)")
    np.random.seed(42)
    cagrs, dds, sharpes, pfs = [], [], [], []
    for sim in range(n_sims):
        spread = spec['spread'] * np.random.uniform(0.7, 2.5)
        slip = spec['slip'] * np.random.uniform(0.5, 3.0)
        off = np.random.randint(0, min(500, len(data)//10))
        sd = data.iloc[off:].copy()
        sc = copy.deepcopy(config)
        sc['grid_strategy']['base_lot_size'] *= np.random.uniform(0.9, 1.1)
        # Corrupt 5% of candles
        mask = np.random.random(len(sd)) < 0.05
        sd.loc[mask, 'close'] = sd.loc[mask, 'open']
        try:
            bt = ProtectedGridBacktester(
                initial_balance=500.0, config=sc,
                spread_pips=spread, slippage_pips=slip,
                pip_size=spec['pip_size'], pip_value_per_lot=spec['pip_value']
            )
            r = bt.run_backtest(sd)
            cagrs.append(r.cagr); dds.append(r.max_drawdown)
            sharpes.append(r.sharpe_ratio); pfs.append(r.profit_factor)
        except:
            cagrs.append(-1.0); dds.append(1.0); sharpes.append(0.0); pfs.append(0.0)
        if (sim+1) % 10 == 0:
            good = sum(1 for c in cagrs if c > 0)
            P(f"      [{sim+1}/{n_sims}] Prof: {good}/{sim+1} ({good/(sim+1)*100:.0f}%)")

    c,d,s,p = [np.array(x) for x in [cagrs,dds,sharpes,pfs]]
    v = c > -0.99; nv = v.sum()
    if nv == 0:
        P("      ALL FAILED")
        return {'passes': False, 'profitable_pct': 0}

    pp = (c[v]>0).mean()*100
    cm = np.median(c[v])*100; c5 = np.percentile(c[v],5)*100
    dm = np.median(d[v])*100; d95 = np.percentile(d[v],95)*100
    sm = np.median(s[v]); s5 = np.percentile(s[v],5)
    pm = np.median(p[v])
    passes = pp >= 90 and c5 > 0 and d95 < 50 and s5 > 0.3

    P(f"      Prof:{pp:.0f}% CAGR:P5={c5:.1f}% Med={cm:.1f}% DD:P95={d95:.1f}% Shp:P5={s5:.2f} Med={sm:.2f} PF:{pm:.2f} -> {'PASS' if passes else 'FAIL'}")
    return {
        'passes': passes, 'profitable_pct': pp,
        'cagr_median': cm, 'cagr_p5': c5,
        'dd_median': dm, 'dd_p95': d95,
        'sharpe_median': sm, 'sharpe_p5': s5,
        'pf_median': pm,
    }


def test_instrument(symbol, spec, lots_to_test):
    """Test a single instrument on full data and overlap filter."""
    P(f"\n{'='*70}")
    P(f"  {symbol} ({spec['class'].upper()})")
    P(f"  pip_size={spec['pip_size']}, pip_value={spec['pip_value']}, spread={spec['spread']}, slip={spec['slip']}")
    P(f"{'='*70}")

    data = load_data(symbol)
    if data is None:
        P(f"  ERROR: No data for {symbol}")
        return None

    P(f"  Data: {len(data)} bars, {data.index[0].date()} to {data.index[-1].date()}")

    results = {'symbol': symbol, 'class': spec['class'], 'bars': len(data)}

    # Test multiple lot sizes on full data
    P(f"\n  --- FULL DATA ---")
    P(f"  {'Lot':>6} {'Mult':>5} {'CAGR':>7} {'DD':>7} {'Shp':>6} {'PF':>6} {'Trd':>5} {'Final':>8}")
    P(f"  {'-'*55}")

    full_results = []
    for lot in lots_to_test:
        adjusted_lot = lot * spec['lot_scale']
        if adjusted_lot < 0.001:
            adjusted_lot = 0.001  # minimum lot
        for mult in [1.5, 2.0, 2.5]:
            cfg = make_config(adjusted_lot, mult)
            r = run_bt(data, cfg, 500.0, spec)
            if r is None:
                continue
            full_results.append({
                'lot': adjusted_lot, 'mult': mult,
                'cagr': r.cagr, 'dd': r.max_drawdown,
                'sharpe': r.sharpe_ratio, 'pf': r.profit_factor,
                'trades': r.total_trades, 'final': r.final_balance,
            })
            if r.cagr > 0.05 and r.total_trades > 20:
                mk = " ***" if r.sharpe_ratio > 1.0 and r.max_drawdown < 0.30 else ""
                P(f"  {adjusted_lot:>6.3f} {mult:>5.1f} {r.cagr*100:>6.1f}% {r.max_drawdown*100:>6.1f}% "
                  f"{r.sharpe_ratio:>6.2f} {r.profit_factor:>6.2f} {r.total_trades:>5} ${r.final_balance:>7.0f}{mk}")

    if not full_results:
        P(f"  No viable results on full data")
        results['full_best'] = None
        results['overlap_best'] = None
        results['mc'] = None
        return results

    # Find best full-data result
    viable_full = [r for r in full_results if r['cagr'] > 0 and r['trades'] > 20]
    if viable_full:
        best_full = max(viable_full, key=lambda x: x['sharpe'])
        results['full_best'] = best_full
        P(f"\n  Best full: lot={best_full['lot']:.3f} m={best_full['mult']:.1f} "
          f"CAGR={best_full['cagr']*100:.1f}% DD={best_full['dd']*100:.1f}% Shp={best_full['sharpe']:.2f}")
    else:
        results['full_best'] = None
        P(f"  No profitable configs on full data")

    # Test overlap filter
    overlap = data[(data.index.hour >= 12) & (data.index.hour <= 16)]
    P(f"\n  --- OVERLAP (12-16 UTC) --- {len(overlap)} bars")

    if len(overlap) < 500:
        P(f"  Too few bars for overlap test")
        results['overlap_best'] = None
        results['mc'] = None
        return results

    P(f"  {'Lot':>6} {'Mult':>5} {'CAGR':>7} {'DD':>7} {'Shp':>6} {'PF':>6} {'Trd':>5} {'Final':>8}")
    P(f"  {'-'*55}")

    overlap_results = []
    for lot in lots_to_test:
        adjusted_lot = lot * spec['lot_scale']
        if adjusted_lot < 0.001:
            adjusted_lot = 0.001
        for mult in [1.5, 2.0, 2.5]:
            cfg = make_config(adjusted_lot, mult)
            r = run_bt(overlap, cfg, 500.0, spec)
            if r is None:
                continue
            overlap_results.append({
                'lot': adjusted_lot, 'mult': mult,
                'cagr': r.cagr, 'dd': r.max_drawdown,
                'sharpe': r.sharpe_ratio, 'pf': r.profit_factor,
                'trades': r.total_trades, 'final': r.final_balance,
            })
            if r.cagr > 0.05 and r.total_trades > 10:
                mk = " ***" if r.sharpe_ratio > 1.0 and r.max_drawdown < 0.30 else ""
                P(f"  {adjusted_lot:>6.3f} {mult:>5.1f} {r.cagr*100:>6.1f}% {r.max_drawdown*100:>6.1f}% "
                  f"{r.sharpe_ratio:>6.2f} {r.profit_factor:>6.2f} {r.total_trades:>5} ${r.final_balance:>7.0f}{mk}")

    viable_ol = [r for r in overlap_results if r['cagr'] > 0 and r['trades'] > 10]
    if viable_ol:
        best_ol = max(viable_ol, key=lambda x: x['sharpe'])
        results['overlap_best'] = best_ol
        P(f"\n  Best overlap: lot={best_ol['lot']:.3f} m={best_ol['mult']:.1f} "
          f"CAGR={best_ol['cagr']*100:.1f}% DD={best_ol['dd']*100:.1f}% Shp={best_ol['sharpe']:.2f}")

        # Compare full vs overlap
        if results['full_best']:
            sharpe_imp = (best_ol['sharpe'] - results['full_best']['sharpe']) / max(results['full_best']['sharpe'], 0.01) * 100
            P(f"  Overlap vs Full: Sharpe {sharpe_imp:+.0f}% improvement")
    else:
        results['overlap_best'] = None
        P(f"  No profitable configs on overlap data")

    # MC validate the best overlap config if it looks promising
    if results['overlap_best'] and results['overlap_best']['sharpe'] > 0.8 and results['overlap_best']['cagr'] > 0.10:
        best = results['overlap_best']
        cfg = make_config(best['lot'], best['mult'])
        mc = monte_carlo(overlap, cfg, spec, n_sims=50,
                        label=f"{symbol} l={best['lot']:.3f} m={best['mult']:.1f}")
        results['mc'] = mc
    else:
        results['mc'] = None
        if results['overlap_best']:
            P(f"  Skipping MC (Sharpe {results['overlap_best']['sharpe']:.2f} or CAGR {results['overlap_best']['cagr']*100:.1f}% too low)")
        else:
            P(f"  Skipping MC (no viable overlap config)")

    return results


def main():
    t0 = datetime.now()
    P(f"{'#'*90}")
    P(f"# PHASE 8: MULTI-ASSET BACKTEST - ALL 20 INSTRUMENTS")
    P(f"# Testing overlap strategy (12-16 UTC) across forex, crypto, metals, indices")
    P(f"# Started: {t0}")
    P(f"{'#'*90}")

    # Conservative lot sizes to test (will be scaled by lot_scale per instrument)
    lots = [0.02, 0.03, 0.04, 0.05, 0.06]

    all_results = {}
    winners = []
    losers = []

    # Test order: known winners first, then unknowns
    order = [
        # Known good
        'EURUSD',
        # Known marginal
        'GBPUSD',
        # Known bad (verify)
        'USDJPY', 'AUDUSD',
        # New forex
        'NZDUSD', 'USDCAD', 'USDCHF',
        # Crosses
        'EURGBP', 'EURJPY', 'EURAUD', 'EURCHF', 'GBPJPY',
        # Metals
        'XAUUSD', 'XAGUSD',
        # Crypto
        'BTCUSD', 'ETHUSD', 'XRPUSD',
        # Indices
        'US30', 'US500', 'USTEC',
    ]

    for symbol in order:
        spec = INSTRUMENTS[symbol]
        result = test_instrument(symbol, spec, lots)
        if result:
            all_results[symbol] = result
            # Classify
            if result.get('mc') and result['mc']['passes']:
                winners.append(symbol)
            elif result.get('overlap_best') and result['overlap_best']['sharpe'] > 0.5:
                winners.append(symbol)  # promising even without full MC pass

    elapsed = (datetime.now() - t0).total_seconds()

    # === FINAL SUMMARY ===
    P(f"\n\n{'#'*90}")
    P(f"# FINAL SUMMARY ({elapsed/60:.1f} min)")
    P(f"{'#'*90}")

    P(f"\n{'='*90}")
    P(f"  RESULTS BY INSTRUMENT")
    P(f"{'='*90}")
    P(f"  {'Symbol':<10} {'Class':<14} {'Full CAGR':>10} {'Full Shp':>9} {'OL CAGR':>9} {'OL Shp':>8} {'OL DD':>7} {'MC':>6} {'Verdict':>10}")
    P(f"  {'-'*85}")

    mc_passed = []
    promising = []
    failed = []

    for symbol in order:
        r = all_results.get(symbol)
        if not r:
            P(f"  {symbol:<10} {'?':<14} {'N/A':>10} {'N/A':>9} {'N/A':>9} {'N/A':>8} {'N/A':>7} {'N/A':>6} {'NO DATA':>10}")
            continue

        # Full data metrics
        if r['full_best']:
            f_cagr = f"{r['full_best']['cagr']*100:.1f}%"
            f_shp = f"{r['full_best']['sharpe']:.2f}"
        else:
            f_cagr = "N/A"
            f_shp = "N/A"

        # Overlap metrics
        if r['overlap_best']:
            o_cagr = f"{r['overlap_best']['cagr']*100:.1f}%"
            o_shp = f"{r['overlap_best']['sharpe']:.2f}"
            o_dd = f"{r['overlap_best']['dd']*100:.1f}%"
        else:
            o_cagr = "N/A"
            o_shp = "N/A"
            o_dd = "N/A"

        # MC result
        if r.get('mc'):
            mc_str = "PASS" if r['mc']['passes'] else "FAIL"
        else:
            mc_str = "N/A"

        # Verdict
        if r.get('mc') and r['mc']['passes']:
            verdict = "WINNER"
            mc_passed.append(symbol)
        elif r.get('overlap_best') and r['overlap_best']['sharpe'] > 0.8:
            verdict = "PROMISING"
            promising.append(symbol)
        elif r.get('full_best') and r['full_best']['cagr'] > 0:
            verdict = "MARGINAL"
        else:
            verdict = "FAILED"
            failed.append(symbol)

        P(f"  {symbol:<10} {r['class']:<14} {f_cagr:>10} {f_shp:>9} {o_cagr:>9} {o_shp:>8} {o_dd:>7} {mc_str:>6} {verdict:>10}")

    P(f"\n{'='*90}")
    P(f"  SUMMARY")
    P(f"{'='*90}")
    P(f"  MC PASSED ({len(mc_passed)}): {', '.join(mc_passed) if mc_passed else 'None'}")
    P(f"  PROMISING ({len(promising)}): {', '.join(promising) if promising else 'None'}")
    P(f"  FAILED    ({len(failed)}): {', '.join(failed) if failed else 'None'}")

    # Detailed MC results for winners
    if mc_passed:
        P(f"\n{'='*90}")
        P(f"  MC-VALIDATED WINNERS - DETAILS")
        P(f"{'='*90}")
        for symbol in mc_passed:
            r = all_results[symbol]
            mc = r['mc']
            ol = r['overlap_best']
            P(f"\n  {symbol}:")
            P(f"    Config: lot={ol['lot']:.3f}, mult={ol['mult']:.1f}")
            P(f"    Backtest: CAGR={ol['cagr']*100:.1f}%, DD={ol['dd']*100:.1f}%, Sharpe={ol['sharpe']:.2f}, PF={ol['pf']:.2f}")
            P(f"    MC (50 sims): Prof={mc['profitable_pct']:.0f}%, CAGR P5={mc['cagr_p5']:.1f}%, Med={mc['cagr_median']:.1f}%")
            P(f"    MC DD: P95={mc['dd_p95']:.1f}%, Sharpe P5={mc['sharpe_p5']:.2f}, Med={mc['sharpe_median']:.2f}")

    # Portfolio suggestions
    if len(mc_passed) >= 2:
        P(f"\n{'='*90}")
        P(f"  PORTFOLIO SUGGESTION")
        P(f"{'='*90}")
        P(f"  {len(mc_passed)} instruments passed MC validation.")
        P(f"  Next step: test correlations and build diversified portfolio.")
        P(f"  Primary: {mc_passed[0]} (benchmark)")
        if len(mc_passed) > 1:
            P(f"  Diversification candidates: {', '.join(mc_passed[1:])}")

    P(f"\n  Total time: {elapsed/60:.1f} minutes")
    P(f"{'#'*90}")


if __name__ == '__main__':
    main()
