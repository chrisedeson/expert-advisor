#!/usr/bin/env python3
"""Phase 7: Walk-Forward on Overlap-Filtered Data + H4 Overlap + Deep MC.

Three critical experiments:
1. Walk-forward on overlap data to verify session filter isn't overfitted
2. H4 + overlap filter combination (best PF + best filter)
3. Deep MC (100 sims) on top production configs
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

def load_data(pair, tf):
    p = project_root / 'data' / 'processed' / f'{pair}_{tf}.parquet'
    df = pd.read_parquet(p)
    if not isinstance(df.index, pd.DatetimeIndex):
        df['time'] = pd.to_datetime(df['time']); df.set_index('time', inplace=True)
    if 'atr' not in df.columns:
        df['tr'] = np.maximum(df['high']-df['low'], np.maximum(abs(df['high']-df['close'].shift(1)), abs(df['low']-df['close'].shift(1))))
        df['atr'] = df['tr'].rolling(14).mean(); df.drop('tr', axis=1, inplace=True)
    return df

def run_bt(data, config, capital=500.0, spread=0.7, slip=0.2):
    try:
        bt = ProtectedGridBacktester(initial_balance=capital, config=config,
            spread_pips=spread, slippage_pips=slip, pip_size=0.0001, pip_value_per_lot=10.0)
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
        if (sim+1) % 10 == 0:
            good = sum(1 for c in cagrs if c > 0)
            P(f"    [{sim+1}/{n_sims}] Prof: {good}/{sim+1} ({good/(sim+1)*100:.0f}%)")
    c,d,s,p = [np.array(x) for x in [cagrs,dds,sharpes,pfs]]
    v = c > -0.99; nv = v.sum()
    if nv == 0:
        P("    ALL FAILED")
        return {'passes':False,'profitable_pct':0,'cagr_median':0,'cagr_p5':0,'dd_p95':100,'sharpe_median':0}
    pp=(c[v]>0).mean()*100; cm=np.median(c[v])*100; c5=np.percentile(c[v],5)*100
    c25=np.percentile(c[v],25)*100; c75=np.percentile(c[v],75)*100; c95=np.percentile(c[v],95)*100
    dm=np.median(d[v])*100; d95=np.percentile(d[v],95)*100; dmax=np.max(d[v])*100
    sm=np.median(s[v]); s5=np.percentile(s[v],5); pm=np.median(p[v])
    passes = pp >= 90 and c5 > 0 and d95 < 50 and s5 > 0.3
    P(f"    Prof:{pp:.0f}% CAGR:P5={c5:.1f}% Med={cm:.1f}% P95={c95:.1f}%")
    P(f"    DD:Med={dm:.1f}% P95={d95:.1f}% Max={dmax:.1f}%")
    P(f"    Sharpe:P5={s5:.2f} Med={sm:.2f} | PF:Med={pm:.2f} -> {'PASS' if passes else 'FAIL'}")
    return {'passes':passes,'profitable_pct':pp,'cagr_median':cm,'cagr_p5':c5,'cagr_p25':c25,
            'cagr_p75':c75,'cagr_p95':c95,'dd_median':dm,'dd_p95':d95,'dd_max':dmax,
            'sharpe_median':sm,'sharpe_p5':s5,'pf_median':pm}


def walk_forward(data, config, splits, label=""):
    """Walk-forward: train on IS, test on OOS."""
    P(f"\n  Walk-Forward: {label}")
    for is_pct in splits:
        split_idx = int(len(data) * is_pct)
        is_data = data.iloc[:split_idx]
        oos_data = data.iloc[split_idx:]

        is_r = run_bt(is_data, config)
        oos_r = run_bt(oos_data, config)

        if is_r is None or oos_r is None:
            P(f"    {int(is_pct*100)}/{int((1-is_pct)*100)}: FAILED")
            continue

        deg = (1 - oos_r.cagr / is_r.cagr) * 100 if is_r.cagr > 0 else 999
        verdict = "GOOD FIT" if deg < 30 else "MOD FIT" if deg < 50 else "OVERFITTED"

        P(f"    {int(is_pct*100)}/{int((1-is_pct)*100)} split ({len(is_data)}/{len(oos_data)} bars):")
        P(f"      IS:  CAGR={is_r.cagr*100:5.1f}% DD={is_r.max_drawdown*100:5.1f}% Shp={is_r.sharpe_ratio:.2f} PF={is_r.profit_factor:.2f}")
        P(f"      OOS: CAGR={oos_r.cagr*100:5.1f}% DD={oos_r.max_drawdown*100:5.1f}% Shp={oos_r.sharpe_ratio:.2f} PF={oos_r.profit_factor:.2f}")
        P(f"      Degradation: {deg:.0f}% -> {verdict}")


def main():
    t0 = datetime.now()
    P(f"{'='*90}")
    P(f"PHASE 7: WALK-FORWARD ON OVERLAP + H4 OVERLAP + DEEP MC")
    P(f"Started: {t0}")
    P(f"{'='*90}")

    eur_h1 = load_data('EURUSD', 'H1')
    eur_h4 = load_data('EURUSD', 'H4')

    # Overlap filter
    overlap_h1 = eur_h1[(eur_h1.index.hour >= 12) & (eur_h1.index.hour <= 16)]
    overlap_h4 = eur_h4[(eur_h4.index.hour >= 12) & (eur_h4.index.hour <= 16)]

    P(f"  H1 full: {len(eur_h1)} bars, H1 overlap: {len(overlap_h1)} bars")
    P(f"  H4 full: {len(eur_h4)} bars, H4 overlap: {len(overlap_h4)} bars")

    # ============================================================
    # EXPERIMENT 1: Walk-Forward on Overlap H1 Data
    # ============================================================
    P(f"\n{'#'*80}")
    P(f"# EXPERIMENT 1: WALK-FORWARD ON OVERLAP-FILTERED H1 DATA")
    P(f"{'#'*80}")
    P(f"\nThis tests whether the overlap filter edge is real or overfitted.")

    # Test multiple configs at different risk levels
    configs_to_test = [
        ("Ultra-Safe",    make_config(0.020, 1.5, 2.0, 0.75, 1.5)),
        ("Conservative",  make_config(0.020, 2.0, 2.0, 0.75, 1.5)),
        ("Balanced",      make_config(0.030, 2.0, 2.0, 0.75, 1.5)),
        ("Growth",        make_config(0.040, 2.5, 2.0, 0.75, 1.5)),
        ("Aggressive",    make_config(0.050, 2.5, 2.0, 0.75, 1.5)),
        ("Max Growth",    make_config(0.060, 2.5, 2.0, 0.75, 1.5)),
    ]

    for name, cfg in configs_to_test:
        walk_forward(overlap_h1, cfg, [0.70, 0.50, 0.60], label=f"Overlap H1 - {name}")

    # Also test anchored walk-forward (expanding window)
    P(f"\n  --- Anchored Walk-Forward (3 periods) ---")
    P(f"  Split overlap data into 3 equal periods, test IS on period 1, OOS on period 2+3")
    n = len(overlap_h1)
    p1 = overlap_h1.iloc[:n//3]
    p2 = overlap_h1.iloc[n//3:2*n//3]
    p3 = overlap_h1.iloc[2*n//3:]
    p12 = overlap_h1.iloc[:2*n//3]

    for name, cfg in [("Conservative", make_config(0.020, 2.0, 2.0, 0.75, 1.5)),
                      ("Growth", make_config(0.040, 2.5, 2.0, 0.75, 1.5)),
                      ("Aggressive", make_config(0.050, 2.5, 2.0, 0.75, 1.5))]:
        P(f"\n  Anchored WF: {name}")
        r1 = run_bt(p1, cfg)
        r2 = run_bt(p2, cfg)
        r3 = run_bt(p3, cfg)
        r12 = run_bt(p12, cfg)
        r_full = run_bt(overlap_h1, cfg)

        if all(r is not None for r in [r1, r2, r3, r12, r_full]):
            P(f"    Period 1 (oldest):  CAGR={r1.cagr*100:5.1f}% DD={r1.max_drawdown*100:5.1f}% Shp={r1.sharpe_ratio:.2f} PF={r1.profit_factor:.2f} Trades={r1.total_trades}")
            P(f"    Period 2 (middle):  CAGR={r2.cagr*100:5.1f}% DD={r2.max_drawdown*100:5.1f}% Shp={r2.sharpe_ratio:.2f} PF={r2.profit_factor:.2f} Trades={r2.total_trades}")
            P(f"    Period 3 (newest):  CAGR={r3.cagr*100:5.1f}% DD={r3.max_drawdown*100:5.1f}% Shp={r3.sharpe_ratio:.2f} PF={r3.profit_factor:.2f} Trades={r3.total_trades}")
            P(f"    Period 1+2:         CAGR={r12.cagr*100:5.1f}% DD={r12.max_drawdown*100:5.1f}% Shp={r12.sharpe_ratio:.2f} PF={r12.profit_factor:.2f} Trades={r12.total_trades}")
            P(f"    Full:               CAGR={r_full.cagr*100:5.1f}% DD={r_full.max_drawdown*100:5.1f}% Shp={r_full.sharpe_ratio:.2f} PF={r_full.profit_factor:.2f} Trades={r_full.total_trades}")
            # Check consistency
            all_pf = [r.profit_factor for r in [r1, r2, r3] if r.total_trades > 10]
            all_pos = [r.cagr > 0 for r in [r1, r2, r3]]
            P(f"    Profitable in all periods: {all(all_pos)}")
            P(f"    PF range: {min(all_pf):.2f} - {max(all_pf):.2f}")

    # ============================================================
    # EXPERIMENT 2: H4 + Overlap Filter
    # ============================================================
    P(f"\n{'#'*80}")
    P(f"# EXPERIMENT 2: H4 TIMEFRAME + OVERLAP FILTER")
    P(f"{'#'*80}")
    P(f"\nH4 had PF=2.10 (best of any TF). Overlap filter was the game changer. Combining?")

    # First baseline: H4 full vs H4 overlap
    P(f"\n  --- H4 Full vs H4 Overlap Baseline ---")
    for name, lot, mult in [("Conservative", 0.020, 2.0), ("Balanced", 0.030, 2.0),
                             ("Growth", 0.040, 2.5), ("Aggressive", 0.050, 2.5)]:
        cfg = make_config(lot, mult, 2.0, 0.75, 1.5)
        r_full = run_bt(eur_h4, cfg)
        r_overlap = run_bt(overlap_h4, cfg)
        if r_full and r_overlap:
            P(f"\n  {name} (lot={lot}, mult={mult}):")
            P(f"    H4 Full:    CAGR={r_full.cagr*100:5.1f}% DD={r_full.max_drawdown*100:5.1f}% Shp={r_full.sharpe_ratio:.2f} PF={r_full.profit_factor:.2f} Trades={r_full.total_trades}")
            P(f"    H4 Overlap: CAGR={r_overlap.cagr*100:5.1f}% DD={r_overlap.max_drawdown*100:5.1f}% Shp={r_overlap.sharpe_ratio:.2f} PF={r_overlap.profit_factor:.2f} Trades={r_overlap.total_trades}")

    # Sweep H4 overlap
    P(f"\n  --- H4 Overlap Parameter Sweep ---")
    lots = [0.02, 0.03, 0.04, 0.05, 0.06]
    mults = [1.5, 2.0, 2.5]
    h4_results = []
    P(f"  {'Lot':>5} {'M':>4} {'CAGR':>7} {'DD':>7} {'Shp':>6} {'PF':>6} {'Trd':>5}")
    P(f"  {'-'*42}")
    for lot, mult in itertools.product(lots, mults):
        cfg = make_config(lot, mult, 2.0, 0.75, 1.5)
        r = run_bt(overlap_h4, cfg)
        if r and r.cagr > 0:
            mk = " ***" if r.max_drawdown < 0.30 and r.sharpe_ratio > 1.0 else ""
            P(f"  {lot:>5.3f} {mult:>4.1f} {r.cagr*100:>6.1f}% {r.max_drawdown*100:>6.1f}% {r.sharpe_ratio:>6.2f} {r.profit_factor:>6.2f} {r.total_trades:>5}{mk}")
            h4_results.append({'lot':lot,'mult':mult,'cagr':r.cagr,'dd':r.max_drawdown,
                              'sharpe':r.sharpe_ratio,'pf':r.profit_factor,'trades':r.total_trades})

    # MC validate best H4 overlap configs
    if h4_results:
        h4_sorted = sorted(h4_results, key=lambda x: x['sharpe'], reverse=True)
        h4_best = [r for r in h4_sorted if r['dd'] < 0.35 and r['sharpe'] > 0.8][:3]
        if h4_best:
            P(f"\n  MC Validation of top H4 Overlap configs:")
            for cfg_dict in h4_best:
                cfg = make_config(cfg_dict['lot'], cfg_dict['mult'], 2.0, 0.75, 1.5)
                label = f"H4 l={cfg_dict['lot']:.3f} m={cfg_dict['mult']:.1f}"
                monte_carlo(overlap_h4, cfg, 20, label)

    # ============================================================
    # EXPERIMENT 3: DEEP MC (100 sims) on Best Production Configs
    # ============================================================
    P(f"\n{'#'*80}")
    P(f"# EXPERIMENT 3: DEEP MC VALIDATION (100 sims)")
    P(f"{'#'*80}")
    P(f"\n100-sim MC for production-grade confidence on top configs.")

    # The best overlap configs from Phase 5+6
    production_configs = [
        ("Conservative (l=0.020 m=2.0)", make_config(0.020, 2.0, 2.0, 0.75, 1.5)),
        ("Balanced (l=0.030 m=2.0)",     make_config(0.030, 2.0, 2.0, 0.75, 1.5)),
        ("Growth (l=0.040 m=2.5)",       make_config(0.040, 2.5, 2.0, 0.75, 1.5)),
        ("Best Risk-Adj (l=0.050 m=2.5)",make_config(0.050, 2.5, 2.0, 0.75, 1.5)),
        ("Max Return (l=0.060 m=2.5)",   make_config(0.060, 2.5, 2.0, 0.75, 1.5)),
    ]

    for name, cfg in production_configs:
        monte_carlo(overlap_h1, cfg, 100, f"DEEP {name}")

    # ============================================================
    # EXPERIMENT 4: SL=1.2 exploration (NY data showed promise)
    # ============================================================
    P(f"\n{'#'*80}")
    P(f"# EXPERIMENT 4: SL=1.2 EXPLORATION ON OVERLAP DATA")
    P(f"{'#'*80}")
    P(f"\nNY session showed sl=1.2 has slightly higher CAGR at low DD. Testing on overlap.")

    for name, lot, mult in [("Conservative", 0.020, 2.0), ("Balanced", 0.030, 2.0),
                             ("Growth", 0.040, 2.5), ("Aggressive", 0.050, 2.5)]:
        cfg_12 = make_config(lot, mult, 2.0, 0.75, 1.2)
        cfg_15 = make_config(lot, mult, 2.0, 0.75, 1.5)
        cfg_20 = make_config(lot, mult, 2.0, 0.75, 2.0)
        r12 = run_bt(overlap_h1, cfg_12)
        r15 = run_bt(overlap_h1, cfg_15)
        r20 = run_bt(overlap_h1, cfg_20)
        P(f"\n  {name} (lot={lot}, mult={mult}):")
        for sl_name, r in [("SL=1.2", r12), ("SL=1.5", r15), ("SL=2.0", r20)]:
            if r:
                P(f"    {sl_name}: CAGR={r.cagr*100:5.1f}% DD={r.max_drawdown*100:5.1f}% Shp={r.sharpe_ratio:.2f} PF={r.profit_factor:.2f} Trades={r.total_trades}")

    # If sl=1.2 looks good, MC validate it
    cfg_12_growth = make_config(0.040, 2.5, 2.0, 0.75, 1.2)
    r_check = run_bt(overlap_h1, cfg_12_growth)
    if r_check and r_check.sharpe_ratio > 1.0:
        monte_carlo(overlap_h1, cfg_12_growth, 50, "SL=1.2 Growth (l=0.040 m=2.5)")

    # ============================================================
    # SUMMARY
    # ============================================================
    elapsed = (datetime.now() - t0).total_seconds()
    P(f"\n\n{'='*90}")
    P(f"PHASE 7 COMPLETE ({elapsed/60:.1f} min)")
    P(f"{'='*90}")


if __name__ == '__main__':
    main()
