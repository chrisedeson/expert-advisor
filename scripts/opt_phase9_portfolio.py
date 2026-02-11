#!/usr/bin/env python3
"""Phase 9: 5-Instrument Portfolio Optimization.

Analyzes EURUSD, GBPUSD, EURJPY, XAGUSD, US500 together:
1. Individual backtests on overlap (12-16 UTC)
2. Profit Factor by time period (monthly, quarterly, yearly, 2yr, 3yr, 5yr)
3. Correlation analysis between instruments
4. Portfolio allocation optimization (conservative/balanced/growth/aggressive)
5. Combined portfolio MC validation
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

def P(msg="", end="\n"): print(msg, flush=True, end=end)

# ─── The 5 MC-validated winners ─────────────────────────────────────────
WINNERS = {
    'EURUSD': {'pip_size': 0.0001, 'pip_value': 10.0,  'spread': 0.7,  'slip': 0.2, 'lot_scale': 1.0},
    'GBPUSD': {'pip_size': 0.0001, 'pip_value': 10.0,  'spread': 0.9,  'slip': 0.3, 'lot_scale': 1.0},
    'EURJPY': {'pip_size': 0.01,   'pip_value': 6.67,  'spread': 1.0,  'slip': 0.3, 'lot_scale': 1.0},
    'XAGUSD': {'pip_size': 0.001,  'pip_value': 5.0,   'spread': 3.0,  'slip': 0.5, 'lot_scale': 0.1},
    'US500':  {'pip_size': 0.01,   'pip_value': 1.0,   'spread': 0.5,  'slip': 0.2, 'lot_scale': 0.1},
}

# ─── Risk profiles ──────────────────────────────────────────────────────
RISK_PROFILES = {
    'Conservative': {'base_lot': 0.02, 'mult': 1.5, 'description': 'Low risk, steady growth'},
    'Balanced':     {'base_lot': 0.03, 'mult': 2.0, 'description': 'Moderate risk/reward'},
    'Growth':       {'base_lot': 0.04, 'mult': 2.0, 'description': 'Higher returns, more volatility'},
    'Aggressive':   {'base_lot': 0.05, 'mult': 2.5, 'description': 'High risk, high reward'},
    'Max Return':   {'base_lot': 0.06, 'mult': 2.5, 'description': 'Maximum CAGR, significant DD'},
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


def load_overlap(symbol):
    p = project_root / 'data' / 'processed' / f'{symbol}_H1.parquet'
    df = pd.read_parquet(p)
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
    if 'atr' not in df.columns:
        df['tr'] = np.maximum(df['high']-df['low'],
                    np.maximum(abs(df['high']-df['close'].shift(1)),
                               abs(df['low']-df['close'].shift(1))))
        df['atr'] = df['tr'].rolling(14).mean()
        df.drop('tr', axis=1, inplace=True)
    # Apply overlap filter
    overlap = df[(df.index.hour >= 12) & (df.index.hour <= 16)]
    return overlap


def run_bt(data, config, capital, spec):
    try:
        bt = ProtectedGridBacktester(
            initial_balance=capital, config=config,
            spread_pips=spec['spread'], slippage_pips=spec['slip'],
            pip_size=spec['pip_size'], pip_value_per_lot=spec['pip_value']
        )
        return bt.run_backtest(data)
    except:
        return None


def compute_pf_by_period(trades, label=""):
    """Compute profit factor broken down by time periods."""
    if not trades:
        return {}

    # Build trade DataFrame
    records = []
    for t in trades:
        if t.exit_time and t.net_pnl is not None:
            records.append({
                'exit_time': t.exit_time,
                'net_pnl': t.net_pnl,
                'gross_pnl': t.gross_pnl if t.gross_pnl else t.net_pnl,
            })
    if not records:
        return {}

    tdf = pd.DataFrame(records)
    tdf['exit_time'] = pd.to_datetime(tdf['exit_time'])
    tdf.set_index('exit_time', inplace=True)
    tdf.sort_index(inplace=True)

    results = {}

    # Overall PF
    wins = tdf[tdf['net_pnl'] > 0]['net_pnl'].sum()
    losses = abs(tdf[tdf['net_pnl'] < 0]['net_pnl'].sum())
    results['overall'] = wins / losses if losses > 0 else float('inf')

    # Monthly PF
    monthly_pfs = []
    for period, group in tdf.resample('ME'):
        if len(group) == 0:
            continue
        w = group[group['net_pnl'] > 0]['net_pnl'].sum()
        l = abs(group[group['net_pnl'] < 0]['net_pnl'].sum())
        pf = w / l if l > 0 else (float('inf') if w > 0 else 0)
        monthly_pfs.append({'period': period, 'pf': pf, 'trades': len(group), 'net': group['net_pnl'].sum()})
    results['monthly'] = monthly_pfs

    # Quarterly PF
    quarterly_pfs = []
    for period, group in tdf.resample('QE'):
        if len(group) == 0:
            continue
        w = group[group['net_pnl'] > 0]['net_pnl'].sum()
        l = abs(group[group['net_pnl'] < 0]['net_pnl'].sum())
        pf = w / l if l > 0 else (float('inf') if w > 0 else 0)
        quarterly_pfs.append({'period': period, 'pf': pf, 'trades': len(group), 'net': group['net_pnl'].sum()})
    results['quarterly'] = quarterly_pfs

    # Yearly PF
    yearly_pfs = []
    for period, group in tdf.resample('YE'):
        if len(group) == 0:
            continue
        w = group[group['net_pnl'] > 0]['net_pnl'].sum()
        l = abs(group[group['net_pnl'] < 0]['net_pnl'].sum())
        pf = w / l if l > 0 else (float('inf') if w > 0 else 0)
        yearly_pfs.append({'period': period, 'pf': pf, 'trades': len(group), 'net': group['net_pnl'].sum()})
    results['yearly'] = yearly_pfs

    # Rolling period PFs (2yr, 3yr, 5yr windows)
    start = tdf.index[0]
    end = tdf.index[-1]
    total_days = (end - start).days

    for years, name in [(2, '2yr'), (3, '3yr'), (5, '5yr')]:
        window_days = years * 365
        if total_days < window_days:
            results[name] = [{'note': f'Data < {years} years'}]
            continue
        rolling = []
        # Slide in 6-month increments
        for offset_months in range(0, int((total_days - window_days) / 30) + 1, 6):
            ws = start + pd.Timedelta(days=offset_months * 30)
            we = ws + pd.Timedelta(days=window_days)
            window_trades = tdf[(tdf.index >= ws) & (tdf.index < we)]
            if len(window_trades) < 10:
                continue
            w = window_trades[window_trades['net_pnl'] > 0]['net_pnl'].sum()
            l = abs(window_trades[window_trades['net_pnl'] < 0]['net_pnl'].sum())
            pf = w / l if l > 0 else float('inf')
            rolling.append({'start': ws.date(), 'end': we.date(), 'pf': pf, 'trades': len(window_trades)})
        results[name] = rolling

    return results


def print_pf_breakdown(pf_data, symbol):
    """Print profit factor breakdown for an instrument."""
    P(f"\n  {'─'*60}")
    P(f"  {symbol} - Profit Factor Breakdown")
    P(f"  {'─'*60}")

    P(f"  Overall PF: {pf_data.get('overall', 0):.2f}")

    # Monthly
    monthly = pf_data.get('monthly', [])
    if monthly:
        pfs = [m['pf'] for m in monthly if m['pf'] != float('inf')]
        profitable_months = sum(1 for m in monthly if m['net'] > 0)
        P(f"\n  MONTHLY ({len(monthly)} months, {profitable_months} profitable = {profitable_months/len(monthly)*100:.0f}%)")
        P(f"  {'Month':>10} {'PF':>6} {'Trades':>6} {'Net P&L':>10}")
        P(f"  {'-'*36}")
        for m in monthly:
            pf_str = f"{m['pf']:.2f}" if m['pf'] != float('inf') else "INF"
            P(f"  {str(m['period'].date()):>10} {pf_str:>6} {m['trades']:>6} ${m['net']:>9.2f}")
        if pfs:
            P(f"\n  Monthly PF: Min={min(pfs):.2f}, Med={np.median(pfs):.2f}, Mean={np.mean(pfs):.2f}, Max={max(pfs):.2f}")

    # Quarterly
    quarterly = pf_data.get('quarterly', [])
    if quarterly:
        pfs = [q['pf'] for q in quarterly if q['pf'] != float('inf')]
        profitable_q = sum(1 for q in quarterly if q['net'] > 0)
        P(f"\n  QUARTERLY ({len(quarterly)} quarters, {profitable_q} profitable = {profitable_q/len(quarterly)*100:.0f}%)")
        P(f"  {'Quarter':>10} {'PF':>6} {'Trades':>6} {'Net P&L':>10}")
        P(f"  {'-'*36}")
        for q in quarterly:
            pf_str = f"{q['pf']:.2f}" if q['pf'] != float('inf') else "INF"
            P(f"  {str(q['period'].date()):>10} {pf_str:>6} {q['trades']:>6} ${q['net']:>9.2f}")
        if pfs:
            P(f"\n  Quarterly PF: Min={min(pfs):.2f}, Med={np.median(pfs):.2f}, Mean={np.mean(pfs):.2f}, Max={max(pfs):.2f}")

    # Yearly
    yearly = pf_data.get('yearly', [])
    if yearly:
        pfs = [y['pf'] for y in yearly if y['pf'] != float('inf')]
        profitable_y = sum(1 for y in yearly if y['net'] > 0)
        P(f"\n  YEARLY ({len(yearly)} years, {profitable_y} profitable = {profitable_y/len(yearly)*100:.0f}%)")
        P(f"  {'Year':>10} {'PF':>6} {'Trades':>6} {'Net P&L':>10}")
        P(f"  {'-'*36}")
        for y in yearly:
            pf_str = f"{y['pf']:.2f}" if y['pf'] != float('inf') else "INF"
            P(f"  {str(y['period'].year):>10} {pf_str:>6} {y['trades']:>6} ${y['net']:>9.2f}")
        if pfs:
            P(f"\n  Yearly PF: Min={min(pfs):.2f}, Med={np.median(pfs):.2f}, Mean={np.mean(pfs):.2f}, Max={max(pfs):.2f}")

    # Rolling multi-year
    for name, label in [('2yr', '2-YEAR ROLLING'), ('3yr', '3-YEAR ROLLING'), ('5yr', '5-YEAR ROLLING')]:
        data = pf_data.get(name, [])
        if data and 'note' not in data[0]:
            pfs = [d['pf'] for d in data if d['pf'] != float('inf')]
            P(f"\n  {label} ({len(data)} windows)")
            P(f"  {'Start':>12} {'End':>12} {'PF':>6} {'Trades':>6}")
            P(f"  {'-'*40}")
            for d in data:
                pf_str = f"{d['pf']:.2f}" if d['pf'] != float('inf') else "INF"
                P(f"  {str(d['start']):>12} {str(d['end']):>12} {pf_str:>6} {d['trades']:>6}")
            if pfs:
                P(f"\n  {name.upper()} PF: Min={min(pfs):.2f}, Med={np.median(pfs):.2f}, Max={max(pfs):.2f}")


def compute_correlations(equity_curves):
    """Compute return correlations between instruments."""
    # Align all equity curves to common daily returns
    daily_returns = {}
    for symbol, ec in equity_curves.items():
        daily = ec['equity'].resample('D').last().dropna()
        daily_returns[symbol] = daily.pct_change().dropna()

    # Build aligned DataFrame
    ret_df = pd.DataFrame(daily_returns)
    ret_df.dropna(inplace=True)

    return ret_df.corr()


def portfolio_backtest(results_dict, allocation, capital=500.0, profile_name=""):
    """Combine multiple instrument backtests into a portfolio.

    allocation: dict of {symbol: weight} where weights sum to 1.0
    Each instrument gets weight * capital allocated.
    """
    # Collect all equity curves, scale by allocation
    all_curves = []
    total_trades = 0
    total_wins = 0
    all_trades = []

    for symbol, weight in allocation.items():
        if symbol not in results_dict or results_dict[symbol] is None:
            continue
        r = results_dict[symbol]
        ec = r.equity_curve.copy()
        alloc_capital = capital * weight
        # Scale equity curve: ratio of actual to initial, then scale to allocated capital
        ec['scaled_equity'] = (ec['equity'] / r.initial_balance) * alloc_capital
        ec['symbol'] = symbol
        all_curves.append(ec[['scaled_equity']])
        total_trades += r.total_trades
        total_wins += r.winning_trades

        # Collect trades with scaled PnL
        for t in r.trades:
            if t.exit_time and t.net_pnl is not None:
                all_trades.append({
                    'exit_time': t.exit_time,
                    'net_pnl': t.net_pnl * weight,
                    'symbol': symbol,
                })

    if not all_curves:
        return None

    # Combine: resample all to daily, sum
    combined = pd.DataFrame()
    for i, curve in enumerate(all_curves):
        daily = curve.resample('D').last().dropna()
        daily.columns = [f'eq_{i}']
        if combined.empty:
            combined = daily
        else:
            combined = combined.join(daily, how='outer')

    combined.ffill(inplace=True)
    combined.bfill(inplace=True)
    combined['total_equity'] = combined.sum(axis=1)

    # Compute metrics
    eq = combined['total_equity']
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = abs(dd.min())

    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / capital) ** (1/years) - 1 if years > 0 else 0

    daily_returns = eq.pct_change().dropna()
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

    # PF from combined trades
    trade_df = pd.DataFrame(all_trades)
    if len(trade_df) > 0:
        wins_total = trade_df[trade_df['net_pnl'] > 0]['net_pnl'].sum()
        losses_total = abs(trade_df[trade_df['net_pnl'] < 0]['net_pnl'].sum())
        pf = wins_total / losses_total if losses_total > 0 else float('inf')
    else:
        pf = 0

    return {
        'cagr': cagr, 'max_dd': max_dd, 'sharpe': sharpe, 'pf': pf,
        'total_trades': total_trades, 'final_balance': eq.iloc[-1],
        'equity_curve': eq, 'years': years,
        'win_rate': total_wins / total_trades if total_trades > 0 else 0,
    }


def monte_carlo_portfolio(data_dict, configs_dict, specs, allocation, n_sims=50, capital=500.0, label=""):
    """MC validate a portfolio allocation."""
    P(f"\n    MC Portfolio: {label} ({n_sims} sims)")
    np.random.seed(42)
    cagrs, dds, sharpes, pfs = [], [], [], []

    for sim in range(n_sims):
        port_results = {}
        for symbol, weight in allocation.items():
            if weight <= 0:
                continue
            spec = specs[symbol]
            data = data_dict[symbol]
            spread = spec['spread'] * np.random.uniform(0.7, 2.5)
            slip = spec['slip'] * np.random.uniform(0.5, 3.0)
            off = np.random.randint(0, min(500, len(data)//10))
            sd = data.iloc[off:].copy()
            sc = copy.deepcopy(configs_dict[symbol])
            sc['grid_strategy']['base_lot_size'] *= np.random.uniform(0.9, 1.1)
            mask = np.random.random(len(sd)) < 0.05
            sd.loc[mask, 'close'] = sd.loc[mask, 'open']
            try:
                bt = ProtectedGridBacktester(
                    initial_balance=capital * weight, config=sc,
                    spread_pips=spread, slippage_pips=slip,
                    pip_size=spec['pip_size'], pip_value_per_lot=spec['pip_value']
                )
                r = bt.run_backtest(sd)
                port_results[symbol] = r
            except:
                pass

        if not port_results:
            cagrs.append(-1.0); dds.append(1.0); sharpes.append(0.0); pfs.append(0.0)
            continue

        # Combine equity curves
        curves = []
        total_pnl_pos = 0
        total_pnl_neg = 0
        for sym, r in port_results.items():
            ec = r.equity_curve.copy()
            curves.append(ec['equity'].resample('D').last().dropna())
            for t in r.trades:
                if t.net_pnl is not None:
                    if t.net_pnl > 0:
                        total_pnl_pos += t.net_pnl
                    else:
                        total_pnl_neg += abs(t.net_pnl)

        combined = pd.concat(curves, axis=1).ffill().bfill()
        total_eq = combined.sum(axis=1)
        peak = total_eq.cummax()
        dd_series = (total_eq - peak) / peak
        max_dd = abs(dd_series.min())
        years = (total_eq.index[-1] - total_eq.index[0]).days / 365.25
        cagr_val = (total_eq.iloc[-1] / capital) ** (1/max(years, 0.1)) - 1
        daily_ret = total_eq.pct_change().dropna()
        sharpe_val = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
        pf_val = total_pnl_pos / total_pnl_neg if total_pnl_neg > 0 else 0

        cagrs.append(cagr_val)
        dds.append(max_dd)
        sharpes.append(sharpe_val)
        pfs.append(pf_val)

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


def main():
    t0 = datetime.now()
    P(f"{'#'*90}")
    P(f"# PHASE 9: 5-INSTRUMENT PORTFOLIO OPTIMIZATION")
    P(f"# Instruments: EURUSD, GBPUSD, EURJPY, XAGUSD, US500")
    P(f"# Started: {t0}")
    P(f"{'#'*90}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Load data and run individual backtests per risk profile
    # ═══════════════════════════════════════════════════════════════════════
    P(f"\n{'='*90}")
    P(f"  STEP 1: INDIVIDUAL INSTRUMENT BACKTESTS")
    P(f"{'='*90}")

    data_dict = {}
    for symbol in WINNERS:
        data_dict[symbol] = load_overlap(symbol)
        P(f"  {symbol}: {len(data_dict[symbol])} overlap bars")

    # Run each profile on each instrument
    all_results = {}  # {profile: {symbol: BacktestResult}}
    all_pf_data = {}  # {profile: {symbol: pf_breakdown}}

    for profile_name, profile in RISK_PROFILES.items():
        P(f"\n  ── {profile_name.upper()} (lot={profile['base_lot']}, mult={profile['mult']}) ──")
        P(f"  {'Symbol':<10} {'CAGR':>7} {'DD':>7} {'Sharpe':>7} {'PF':>6} {'Trades':>6} {'Final':>8} {'WR':>5}")
        P(f"  {'-'*62}")

        all_results[profile_name] = {}
        all_pf_data[profile_name] = {}

        for symbol, spec in WINNERS.items():
            lot = profile['base_lot'] * spec['lot_scale']
            if lot < 0.001:
                lot = 0.001
            cfg = make_config(lot, profile['mult'])
            r = run_bt(data_dict[symbol], cfg, 500.0, spec)
            all_results[profile_name][symbol] = r

            if r:
                P(f"  {symbol:<10} {r.cagr*100:>6.1f}% {r.max_drawdown*100:>6.1f}% {r.sharpe_ratio:>7.2f} "
                  f"{r.profit_factor:>6.2f} {r.total_trades:>6} ${r.final_balance:>7.0f} {r.win_rate*100:>4.0f}%")

                # Compute PF breakdown
                pf_data = compute_pf_by_period(r.trades, f"{symbol}_{profile_name}")
                all_pf_data[profile_name][symbol] = pf_data
            else:
                P(f"  {symbol:<10} FAILED")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: PF Time-Period Breakdown (using Balanced profile as reference)
    # ═══════════════════════════════════════════════════════════════════════
    P(f"\n\n{'='*90}")
    P(f"  STEP 2: PROFIT FACTOR BY TIME PERIOD (Balanced profile)")
    P(f"{'='*90}")

    for symbol in WINNERS:
        pf_data = all_pf_data.get('Balanced', {}).get(symbol, {})
        if pf_data:
            print_pf_breakdown(pf_data, symbol)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: Correlation Analysis
    # ═══════════════════════════════════════════════════════════════════════
    P(f"\n\n{'='*90}")
    P(f"  STEP 3: CORRELATION ANALYSIS")
    P(f"{'='*90}")

    # Use Balanced profile equity curves
    eq_curves = {}
    for symbol in WINNERS:
        r = all_results['Balanced'].get(symbol)
        if r and not r.equity_curve.empty:
            eq_curves[symbol] = r.equity_curve

    if len(eq_curves) >= 2:
        corr = compute_correlations(eq_curves)
        P(f"\n  Daily Return Correlations:")
        P(f"  {'':>10}", end="")
        for s in corr.columns:
            P(f" {s:>8}", end="")
        P()
        for s1 in corr.index:
            P(f"  {s1:>10}", end="")
            for s2 in corr.columns:
                val = corr.loc[s1, s2]
                marker = " *" if abs(val) < 0.3 and s1 != s2 else ""
                P(f" {val:>7.3f}{marker}", end="")
            P()
        P(f"\n  * = Low correlation (<0.3) = good for diversification")

        # Average pairwise correlation
        pairs = []
        symbols = list(corr.columns)
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                pairs.append((symbols[i], symbols[j], corr.iloc[i, j]))
        pairs.sort(key=lambda x: x[2])
        P(f"\n  Pairwise correlations (sorted):")
        for s1, s2, c in pairs:
            quality = "EXCELLENT" if abs(c) < 0.2 else "GOOD" if abs(c) < 0.3 else "OK" if abs(c) < 0.5 else "HIGH"
            P(f"    {s1}-{s2}: {c:.3f} [{quality}]")

        avg_corr = np.mean([abs(c) for _, _, c in pairs])
        P(f"\n  Average pairwise |correlation|: {avg_corr:.3f}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: Portfolio Allocation Testing
    # ═══════════════════════════════════════════════════════════════════════
    P(f"\n\n{'='*90}")
    P(f"  STEP 4: PORTFOLIO ALLOCATION TESTING")
    P(f"{'='*90}")

    # Define allocation schemes
    allocations = {
        'Equal Weight':         {'EURUSD': 0.20, 'GBPUSD': 0.20, 'EURJPY': 0.20, 'XAGUSD': 0.20, 'US500': 0.20},
        'Forex Heavy':          {'EURUSD': 0.30, 'GBPUSD': 0.20, 'EURJPY': 0.30, 'XAGUSD': 0.10, 'US500': 0.10},
        'Sharpe Weighted':      {'EURUSD': 0.25, 'GBPUSD': 0.15, 'EURJPY': 0.30, 'XAGUSD': 0.10, 'US500': 0.20},
        'Forex Only':           {'EURUSD': 0.40, 'GBPUSD': 0.25, 'EURJPY': 0.35, 'XAGUSD': 0.00, 'US500': 0.00},
        'Max Diversification':  {'EURUSD': 0.20, 'GBPUSD': 0.15, 'EURJPY': 0.20, 'XAGUSD': 0.20, 'US500': 0.25},
        'Conservative Core':    {'EURUSD': 0.35, 'GBPUSD': 0.20, 'EURJPY': 0.25, 'XAGUSD': 0.05, 'US500': 0.15},
    }

    # Test each allocation scheme at each risk profile
    for profile_name, profile in RISK_PROFILES.items():
        P(f"\n  ── {profile_name.upper()} PORTFOLIO RESULTS ──")
        P(f"  {'Allocation':<22} {'CAGR':>7} {'DD':>7} {'Sharpe':>7} {'PF':>6} {'Trades':>6} {'Final':>8}")
        P(f"  {'-'*70}")

        for alloc_name, alloc in allocations.items():
            port = portfolio_backtest(all_results[profile_name], alloc, capital=500.0)
            if port:
                P(f"  {alloc_name:<22} {port['cagr']*100:>6.1f}% {port['max_dd']*100:>6.1f}% {port['sharpe']:>7.2f} "
                  f"{port['pf']:>6.2f} {port['total_trades']:>6} ${port['final_balance']:>7.0f}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: MC Validate Best Portfolios
    # ═══════════════════════════════════════════════════════════════════════
    P(f"\n\n{'='*90}")
    P(f"  STEP 5: MONTE CARLO PORTFOLIO VALIDATION (50 sims each)")
    P(f"{'='*90}")

    # MC validate the best allocations for each risk profile
    mc_combos = [
        ('Conservative', 'Conservative Core'),
        ('Balanced', 'Sharpe Weighted'),
        ('Growth', 'Max Diversification'),
        ('Aggressive', 'Sharpe Weighted'),
        ('Max Return', 'Equal Weight'),
    ]

    mc_results = {}
    for profile_name, alloc_name in mc_combos:
        profile = RISK_PROFILES[profile_name]
        alloc = allocations[alloc_name]

        # Build configs for each symbol
        configs = {}
        for symbol, spec in WINNERS.items():
            if alloc.get(symbol, 0) <= 0:
                continue
            lot = profile['base_lot'] * spec['lot_scale']
            if lot < 0.001:
                lot = 0.001
            configs[symbol] = make_config(lot, profile['mult'])

        label = f"{profile_name} + {alloc_name}"
        mc = monte_carlo_portfolio(
            data_dict, configs, WINNERS, alloc,
            n_sims=50, capital=500.0, label=label
        )
        mc_results[label] = mc

    # ═══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    elapsed = (datetime.now() - t0).total_seconds()
    P(f"\n\n{'#'*90}")
    P(f"# FINAL PORTFOLIO SUMMARY ({elapsed/60:.1f} min)")
    P(f"{'#'*90}")

    P(f"\n  INDIVIDUAL INSTRUMENT QUALITY (Balanced profile, overlap):")
    P(f"  {'Symbol':<10} {'CAGR':>7} {'DD':>7} {'Sharpe':>7} {'PF':>6} {'Yearly PF Range':>20}")
    P(f"  {'-'*60}")
    for symbol in WINNERS:
        r = all_results['Balanced'].get(symbol)
        pf_data = all_pf_data.get('Balanced', {}).get(symbol, {})
        if r:
            yearly = pf_data.get('yearly', [])
            yearly_pfs = [y['pf'] for y in yearly if y['pf'] != float('inf')]
            yr_range = f"{min(yearly_pfs):.1f}-{max(yearly_pfs):.1f}" if yearly_pfs else "N/A"
            P(f"  {symbol:<10} {r.cagr*100:>6.1f}% {r.max_drawdown*100:>6.1f}% {r.sharpe_ratio:>7.2f} "
              f"{r.profit_factor:>6.2f} {yr_range:>20}")

    P(f"\n  MC PORTFOLIO RESULTS:")
    P(f"  {'Combo':<40} {'Pass':>5} {'Prof%':>6} {'CAGR P5':>8} {'CAGR Med':>9} {'DD P95':>7} {'Shp P5':>7} {'PF Med':>7}")
    P(f"  {'-'*90}")
    for label, mc in mc_results.items():
        pass_str = "PASS" if mc['passes'] else "FAIL"
        P(f"  {label:<40} {pass_str:>5} {mc['profitable_pct']:>5.0f}% {mc.get('cagr_p5',0):>7.1f}% {mc.get('cagr_median',0):>8.1f}% "
          f"{mc.get('dd_p95',0):>6.1f}% {mc.get('sharpe_p5',0):>7.2f} {mc.get('pf_median',0):>7.2f}")

    P(f"\n  RECOMMENDED LIVE CONFIGURATIONS:")
    for label, mc in mc_results.items():
        if mc['passes']:
            P(f"  GO: {label}")
            P(f"      Expected: CAGR {mc['cagr_p5']:.0f}-{mc['cagr_median']:.0f}%, DD up to {mc['dd_p95']:.0f}%, Sharpe {mc['sharpe_p5']:.2f}+")

    P(f"\n  Total time: {elapsed/60:.1f} minutes")
    P(f"{'#'*90}")


if __name__ == '__main__':
    main()
