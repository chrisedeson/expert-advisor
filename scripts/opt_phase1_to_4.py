#!/usr/bin/env python3
"""Phases 1-4: MC validation, asset tests, session filters, walk-forward."""
import sys, copy
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

def P(msg=""): print(msg, flush=True)

PAIR_PARAMS = {
    'EURUSD': {'pip_size': 0.0001, 'pip_value_per_lot': 10.0, 'spread_pips': 0.7, 'slippage_pips': 0.2},
    'GBPUSD': {'pip_size': 0.0001, 'pip_value_per_lot': 10.0, 'spread_pips': 0.9, 'slippage_pips': 0.3},
    'XAUUSD': {'pip_size': 0.01, 'pip_value_per_lot': 1.0, 'spread_pips': 3.0, 'slippage_pips': 0.5},
}

def make_config(lot=0.0228, mult=1.5, bb_mult=2.0, grid_spacing=0.75, sl_mult=1.5):
    return {
        'grid_strategy': {
            'base_lot_size': lot, 'lot_multiplier': mult, 'max_grid_levels': 5,
            'use_trend_filter': True, 'compound_on_equity': False,
            'bb_entry_mult': bb_mult, 'grid_spacing_atr': grid_spacing, 'sl_atr_mult': sl_mult,
        },
        'volatility_filter': {'atr_period': 14, 'avg_period': 50, 'normal_threshold': 10.0, 'crisis_threshold': 20.0, 'cooldown_days': 0},
        'circuit_breaker': {'daily_limit': 0.20, 'weekly_limit': 0.35, 'monthly_limit': 0.50},
        'crisis_detector': {'volatility_spike_threshold': 6.0, 'rapid_drawdown_threshold': 0.50, 'rapid_drawdown_days': 3, 'consecutive_stops_threshold': 15},
        'recovery_manager': {'drawdown_threshold': 1.0},
        'profit_protector': {'profit_threshold': 100.0},
    }

def load_data(pair, tf='H1'):
    p = project_root / 'data' / 'processed' / f'{pair}_{tf}.parquet'
    if not p.exists(): return None
    import pandas as pd
    df = pd.read_parquet(p)
    if not isinstance(df.index, pd.DatetimeIndex):
        df['time'] = pd.to_datetime(df['time']); df.set_index('time', inplace=True)
    if 'atr' not in df.columns:
        df['tr'] = np.maximum(df['high']-df['low'], np.maximum(abs(df['high']-df['close'].shift(1)), abs(df['low']-df['close'].shift(1))))
        df['atr'] = df['tr'].rolling(14).mean(); df.drop('tr', axis=1, inplace=True)
    return df

def resample_h4(h1):
    h4 = h1.resample('4h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    h4['tr'] = np.maximum(h4['high']-h4['low'], np.maximum(abs(h4['high']-h4['close'].shift(1)), abs(h4['low']-h4['close'].shift(1))))
    h4['atr'] = h4['tr'].rolling(14).mean(); h4.drop('tr', axis=1, inplace=True); h4.dropna(inplace=True)
    return h4

def run_bt(data, config, pair, capital=500.0):
    params = PAIR_PARAMS.get(pair, PAIR_PARAMS['EURUSD'])
    try:
        bt = ProtectedGridBacktester(initial_balance=capital, config=config,
            spread_pips=params['spread_pips'], slippage_pips=params['slippage_pips'],
            pip_size=params['pip_size'], pip_value_per_lot=params['pip_value_per_lot'])
        return bt.run_backtest(data)
    except: return None

def monte_carlo(data, config, pair, n_sims=20, label=""):
    base = PAIR_PARAMS.get(pair, PAIR_PARAMS['EURUSD'])
    is_gold = pair == 'XAUUSD'
    P(f"\n{'='*80}")
    P(f"MC: {label} ({n_sims} sims)")
    P(f"{'='*80}")
    np.random.seed(42)
    cagrs, dds, sharpes, pfs = [], [], [], []
    for sim in range(n_sims):
        spread = np.random.uniform(2.0, 5.0) if is_gold else np.random.uniform(0.5, 2.0)
        slip = np.random.uniform(0.0, 2.0) if is_gold else np.random.uniform(0.0, 1.0)
        off = np.random.randint(0, min(500, len(data)//10))
        sd = data.iloc[off:].copy()
        sc = copy.deepcopy(config)
        sc['grid_strategy']['base_lot_size'] *= np.random.uniform(0.9, 1.1)
        mask = np.random.random(len(sd)) < 0.05
        sd.loc[mask, 'close'] = sd.loc[mask, 'open']
        try:
            bt = ProtectedGridBacktester(initial_balance=500.0, config=sc,
                spread_pips=spread, slippage_pips=slip, pip_size=base['pip_size'], pip_value_per_lot=base['pip_value_per_lot'])
            r = bt.run_backtest(sd)
            cagrs.append(r.cagr); dds.append(r.max_drawdown); sharpes.append(r.sharpe_ratio); pfs.append(r.profit_factor)
        except:
            cagrs.append(-1.0); dds.append(1.0); sharpes.append(0.0); pfs.append(0.0)
        if (sim+1) % 5 == 0:
            good = sum(1 for c in cagrs if c > 0)
            P(f"  [{sim+1}/{n_sims}] Prof: {good}/{sim+1} ({good/(sim+1)*100:.0f}%)")
    c, d, s, p = [np.array(x) for x in [cagrs, dds, sharpes, pfs]]
    v = c > -0.99; nv = v.sum()
    if nv == 0:
        P("  ALL FAILED"); return {'passes': False, 'profitable_pct': 0, 'cagr_median': 0, 'cagr_p5': 0, 'dd_p95': 100, 'dd_median': 100, 'sharpe_median': 0, 'sharpe_p5': 0, 'pf_median': 0}
    pp = (c[v]>0).mean()*100
    cm=np.median(c[v])*100; c5=np.percentile(c[v],5)*100; c25=np.percentile(c[v],25)*100
    c75=np.percentile(c[v],75)*100; c95=np.percentile(c[v],95)*100
    dm=np.median(d[v])*100; d95=np.percentile(d[v],95)*100; dmax=np.max(d[v])*100
    sm=np.median(s[v]); s5=np.percentile(s[v],5); pm=np.median(p[v])
    passes = pp >= 90 and c5 > 0 and d95 < 50 and s5 > 0.3
    P(f"\n  Prof: {pp:.0f}% | CAGR: P5={c5:.1f}% Med={cm:.1f}% P95={c95:.1f}%")
    P(f"  DD: Med={dm:.1f}% P95={d95:.1f}% Max={dmax:.1f}% | Sharpe: P5={s5:.2f} Med={sm:.2f}")
    P(f"  PF: Med={pm:.2f} | VERDICT: {'PASS' if passes else 'FAIL'}")
    return {'passes':passes, 'profitable_pct':pp, 'cagr_median':cm, 'cagr_p5':c5, 'cagr_p25':c25, 'cagr_p75':c75, 'cagr_p95':c95,
            'dd_median':dm, 'dd_p95':d95, 'dd_max':dmax, 'sharpe_median':sm, 'sharpe_p5':s5, 'pf_median':pm}

def main():
    t0 = datetime.now()
    P(f"{'='*90}")
    P(f"PHASES 1-4: MC + Assets + Sessions + Walk-Forward")
    P(f"Started: {t0}")
    P(f"{'='*90}")

    eur = load_data('EURUSD', 'H1')

    # === PHASE 1: MC on 3 configs ===
    P(f"\n{'#'*80}")
    P(f"# PHASE 1: MONTE CARLO (20 sims each)")
    P(f"{'#'*80}")
    mc_con = monte_carlo(eur, make_config(lot=0.0228, mult=1.5), 'EURUSD', 20, "Conservative l=0.0228 m=1.5")
    mc_mod = monte_carlo(eur, make_config(lot=0.04, mult=2.0), 'EURUSD', 20, "Moderate l=0.04 m=2.0")
    mc_agg = monte_carlo(eur, make_config(lot=0.06, mult=2.5), 'EURUSD', 20, "Aggressive l=0.06 m=2.5")

    # === PHASE 2: Assets & Timeframes ===
    P(f"\n{'#'*80}")
    P(f"# PHASE 2: ASSETS & TIMEFRAMES")
    P(f"{'#'*80}")
    P(f"\n  {'Pair':>8} {'TF':>4} {'Config':>14} {'CAGR':>8} {'DD':>8} {'Sharpe':>8} {'PF':>8} {'Trades':>8}")
    P(f"  {'-'*70}")

    for pair in ['EURUSD', 'GBPUSD', 'XAUUSD']:
        for tf in ['H1', 'H4']:
            data = load_data(pair, tf)
            if data is None and tf == 'H4':
                h1 = load_data(pair, 'H1')
                if h1 is not None: data = resample_h4(h1)
            if data is None:
                P(f"  {pair:>8} {tf:>4} -- NO DATA --"); continue
            for cn, cfg in [('Conservative', make_config(lot=0.0228, mult=1.5)),
                            ('Aggressive', make_config(lot=0.06, mult=2.5))]:
                c = copy.deepcopy(cfg)
                if pair == 'XAUUSD': c['grid_strategy']['base_lot_size'] *= 10.0
                r = run_bt(data, c, pair)
                if r is None: P(f"  {pair:>8} {tf:>4} {cn:>14} -- FAILED --"); continue
                mk = " ***" if r.max_drawdown < 0.40 and r.sharpe_ratio > 1.0 and r.cagr > 0.20 else ""
                P(f"  {pair:>8} {tf:>4} {cn:>14} {r.cagr*100:>7.1f}% {r.max_drawdown*100:>7.1f}% "
                  f"{r.sharpe_ratio:>8.2f} {r.profit_factor:>8.2f} {r.total_trades:>8}{mk}")

    # === PHASE 3: Session Filters ===
    P(f"\n{'#'*80}")
    P(f"# PHASE 3: SESSION FILTERS")
    P(f"{'#'*80}")
    P(f"\n  {'Session':>12} {'Config':>14} {'CAGR':>8} {'DD':>8} {'Sharpe':>8} {'PF':>8} {'Trades':>8}")
    P(f"  {'-'*70}")

    for session in ['all', 'london_ny', 'london', 'ny', 'overlap', 'asian']:
        import pandas as pd
        filt = eur if session == 'all' else eur[{
            'london_ny': (eur.index.hour>=7)&(eur.index.hour<=17),
            'london': (eur.index.hour>=7)&(eur.index.hour<=16),
            'ny': (eur.index.hour>=12)&(eur.index.hour<=21),
            'overlap': (eur.index.hour>=12)&(eur.index.hour<=16),
            'asian': (eur.index.hour>=0)&(eur.index.hour<=8),
        }[session]]
        for cn, cfg in [('Conservative', make_config(lot=0.0228, mult=1.5)),
                        ('Aggressive', make_config(lot=0.06, mult=2.5))]:
            r = run_bt(filt, cfg, 'EURUSD')
            if r is None: continue
            mk = " ***" if r.sharpe_ratio > 1.0 and r.cagr > 0.20 else ""
            P(f"  {session:>12} {cn:>14} {r.cagr*100:>7.1f}% {r.max_drawdown*100:>7.1f}% "
              f"{r.sharpe_ratio:>8.2f} {r.profit_factor:>8.2f} {r.total_trades:>8}{mk}")

    # === PHASE 4: Walk-Forward ===
    P(f"\n{'#'*80}")
    P(f"# PHASE 4: WALK-FORWARD VALIDATION")
    P(f"{'#'*80}")
    n = len(eur)
    for cn, cfg in [('Conservative', make_config(lot=0.0228, mult=1.5)),
                    ('Moderate', make_config(lot=0.04, mult=2.0)),
                    ('Aggressive', make_config(lot=0.06, mult=2.5))]:
        P(f"\n  {cn}:")
        P(f"  {'Split':>8} {'Per':>5} {'CAGR':>8} {'DD':>8} {'Sharpe':>8} {'PF':>8}")
        P(f"  {'-'*50}")
        for sn, pct in [('70/30', 0.70), ('50/50', 0.50)]:
            si = int(n * pct)
            r_is = run_bt(eur.iloc[:si], cfg, 'EURUSD')
            r_oos = run_bt(eur.iloc[si:], cfg, 'EURUSD')
            if r_is and r_oos:
                deg = (r_is.cagr - r_oos.cagr) / r_is.cagr * 100 if r_is.cagr > 0 else 0
                vd = "OVERFITTED" if deg > 50 else "MOD FIT" if deg > 25 else "OOS BETTER" if deg < 0 else "ROBUST"
                P(f"  {sn:>8} {'IS':>5} {r_is.cagr*100:>7.1f}% {r_is.max_drawdown*100:>7.1f}% {r_is.sharpe_ratio:>8.2f} {r_is.profit_factor:>8.2f}")
                P(f"  {'':>8} {'OOS':>5} {r_oos.cagr*100:>7.1f}% {r_oos.max_drawdown*100:>7.1f}% {r_oos.sharpe_ratio:>8.2f} {r_oos.profit_factor:>8.2f}")
                P(f"  {'':>8} {'':>5}  Deg: {deg:.0f}% -> {vd}")

    elapsed = (datetime.now() - t0).total_seconds()
    P(f"\n{'='*80}")
    P(f"PHASES 1-4 COMPLETE ({elapsed/60:.1f} min)")
    P(f"  Conservative MC: {'PASS' if mc_con['passes'] else 'FAIL'} | CAGR P5={mc_con['cagr_p5']:.1f}% Med={mc_con['cagr_median']:.1f}% | DD P95={mc_con['dd_p95']:.1f}%")
    P(f"  Moderate MC:     {'PASS' if mc_mod['passes'] else 'FAIL'} | CAGR P5={mc_mod['cagr_p5']:.1f}% Med={mc_mod['cagr_median']:.1f}% | DD P95={mc_mod['dd_p95']:.1f}%")
    P(f"  Aggressive MC:   {'PASS' if mc_agg['passes'] else 'FAIL'} | CAGR P5={mc_agg['cagr_p5']:.1f}% Med={mc_agg['cagr_median']:.1f}% | DD P95={mc_agg['dd_p95']:.1f}%")
    P(f"{'='*80}")

if __name__ == '__main__':
    main()
