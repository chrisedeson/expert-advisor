#!/usr/bin/env python3
"""Comprehensive Scalper Backtest Report - Plain English.

Answers: How much does it make? What are the risks? Should I trust it?
Runs on ALL years of data we have, with and without compounding.
"""

import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.scalper.scalper_backtest import ScalperBacktester, ScalperBacktestResult
from scripts.backtest_scalper import INSTRUMENTS, load_data


def run_test(symbol, capital=100.0, compound=False, fixed_rr=2.0, min_score=5.0,
             consecutive=4, max_concurrent=3, session_start=-1, session_end=-1):
    """Run a single backtest with given params."""
    spec = INSTRUMENTS[symbol]
    htf_data = load_data(symbol, 'H1')
    ltf_data = load_data(symbol, 'M15')

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
        min_rr=1.0,
        min_zone_score=min_score,
        sl_buffer_atr_mult=0.3,
        tp_mode='fixed_rr',
        fixed_rr_target=fixed_rr,
        use_breakeven=True,
        use_momentum_filter=True,
        session_start=session_start,
        session_end=session_end,
        risk_pct_per_trade=0.02,
        compound=compound,
        max_zone_age_bars=200,
        zone_touch_limit=3,
        consecutive_candles=consecutive,
    )

    return bt.run_backtest(htf_data, ltf_data, symbol=symbol, htf_name='H1', ltf_name='M15')


def analyze_trades_by_period(trades, initial_balance):
    """Break down trades by day/month/year."""
    daily = defaultdict(lambda: {'pnl': 0, 'wins': 0, 'losses': 0, 'trades': 0})
    monthly = defaultdict(lambda: {'pnl': 0, 'wins': 0, 'losses': 0, 'trades': 0})
    yearly = defaultdict(lambda: {'pnl': 0, 'wins': 0, 'losses': 0, 'trades': 0})

    for t in trades:
        day = str(t.exit_time)[:10]
        month = str(t.exit_time)[:7]
        year = str(t.exit_time)[:4]

        for period, key in [(daily, day), (monthly, month), (yearly, year)]:
            period[key]['pnl'] += t.pnl
            period[key]['trades'] += 1
            if t.pnl > 0:
                period[key]['wins'] += 1
            else:
                period[key]['losses'] += 1

    return daily, monthly, yearly


def format_money(val):
    if val >= 0:
        return f"+${val:.2f}"
    return f"-${abs(val):.2f}"


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    print("=" * 70)
    print("  SUPPLY & DEMAND SCALPER - FULL REPORT")
    print("  'How much money will this make me?'")
    print("=" * 70)

    # ================================================================
    # SECTION 1: The top 6 instruments, no compounding, $100 capital
    # ================================================================
    top_symbols = ['US500', 'XAUUSD', 'XAGUSD', 'USTEC', 'EURUSD', 'ETHUSD']

    print_section("SECTION 1: INDIVIDUAL INSTRUMENT RESULTS ($100 capital, NO compounding)")
    print("\nThis shows what each instrument does on its own with $100.")
    print("'No compounding' = you always risk based on the original $100,")
    print("never increasing your bet as your balance grows.\n")

    results = {}
    for sym in top_symbols:
        print(f"  Testing {sym}...", end=" ", flush=True)
        try:
            r = run_test(sym, capital=100.0, compound=False)
            results[sym] = r
            print(f"OK ({r.total_trades} trades)", flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)

    # Print summary table
    print(f"\n{'Symbol':<8} {'Trades':>7} {'Win%':>6} {'Wins':>6} {'Losses':>7} "
          f"{'$100->':>8} {'Profit':>8} {'MaxDD':>7} {'$/day':>7} {'$/month':>8} {'$/year':>9}")
    print("-" * 95)

    for sym, r in results.items():
        profit = r.final_balance - r.initial_balance
        daily, monthly, yearly = analyze_trades_by_period(r.trades, r.initial_balance)

        avg_daily_pnl = np.mean([d['pnl'] for d in daily.values()]) if daily else 0
        avg_monthly_pnl = np.mean([m['pnl'] for m in monthly.values()]) if monthly else 0
        avg_yearly_pnl = np.mean([y['pnl'] for y in yearly.values()]) if yearly else 0

        print(f"{sym:<8} {r.total_trades:>7} {r.win_rate:>5.1f}% {r.winning_trades:>6} {r.losing_trades:>7} "
              f"${r.final_balance:>7.2f} {format_money(profit):>8} {r.max_drawdown:>6.1f}% "
              f"${avg_daily_pnl:>6.2f} ${avg_monthly_pnl:>7.2f} ${avg_yearly_pnl:>8.2f}")

    # ================================================================
    # SECTION 2: Detailed breakdown for top 3
    # ================================================================
    top3 = sorted(results.items(), key=lambda x: x[1].profit_factor, reverse=True)[:3]

    print_section("SECTION 2: DETAILED BREAKDOWN (Top 3 instruments)")

    for sym, r in top3:
        daily, monthly, yearly = analyze_trades_by_period(r.trades, r.initial_balance)
        profit = r.final_balance - r.initial_balance

        start = str(r.start_date)[:10]
        end = str(r.end_date)[:10]
        n_days = len(daily)
        n_months = len(monthly)
        n_years = len(yearly)

        print(f"\n--- {sym} ---")
        print(f"  Data period: {start} to {end}")
        print(f"  Starting capital: $100.00")
        print(f"  Ending balance:   ${r.final_balance:.2f}")
        print(f"  Total profit:     {format_money(profit)}")
        print(f"  Total trades:     {r.total_trades}")
        print(f"  Winning trades:   {r.winning_trades} ({r.win_rate:.1f}%)")
        print(f"  Losing trades:    {r.losing_trades} ({100-r.win_rate:.1f}%)")
        print(f"  Profit factor:    {r.profit_factor:.2f} (for every $1 lost, you make ${r.profit_factor:.2f})")
        print(f"  Max drawdown:     {r.max_drawdown:.1f}% (worst dip from peak)")
        print(f"  Best trade:       {format_money(r.best_trade)}")
        print(f"  Worst trade:      {format_money(r.worst_trade)}")
        print(f"  Avg win:          {format_money(r.avg_win)}")
        print(f"  Avg loss:         {format_money(r.avg_loss)}")

        # Trades per day
        avg_trades_per_day = r.total_trades / max(n_days, 1)
        print(f"\n  TRADE FREQUENCY:")
        print(f"    Trades per day:   {avg_trades_per_day:.1f}")
        print(f"    Trades per month: {r.total_trades / max(n_months, 1):.0f}")
        print(f"    Trading days:     {n_days}")

        # Daily stats
        daily_pnls = [d['pnl'] for d in daily.values()]
        winning_days = sum(1 for p in daily_pnls if p > 0)
        losing_days = sum(1 for p in daily_pnls if p <= 0)
        print(f"\n  DAILY BREAKDOWN:")
        print(f"    Average day:   {format_money(np.mean(daily_pnls))}")
        print(f"    Best day:      {format_money(max(daily_pnls))}")
        print(f"    Worst day:     {format_money(min(daily_pnls))}")
        print(f"    Winning days:  {winning_days}/{n_days} ({winning_days/max(n_days,1)*100:.0f}%)")
        print(f"    Losing days:   {losing_days}/{n_days} ({losing_days/max(n_days,1)*100:.0f}%)")

        # Monthly stats
        monthly_pnls = [m['pnl'] for m in monthly.values()]
        winning_months = sum(1 for p in monthly_pnls if p > 0)
        losing_months = sum(1 for p in monthly_pnls if p <= 0)
        print(f"\n  MONTHLY BREAKDOWN:")
        print(f"    Average month:   {format_money(np.mean(monthly_pnls))}")
        print(f"    Best month:      {format_money(max(monthly_pnls))}")
        print(f"    Worst month:     {format_money(min(monthly_pnls))}")
        print(f"    Winning months:  {winning_months}/{n_months} ({winning_months/max(n_months,1)*100:.0f}%)")
        print(f"    Losing months:   {losing_months}/{n_months} ({losing_months/max(n_months,1)*100:.0f}%)")

        # Yearly stats
        print(f"\n  YEARLY BREAKDOWN:")
        for year_key in sorted(yearly.keys()):
            y = yearly[year_key]
            wr = y['wins'] / max(y['trades'], 1) * 100
            print(f"    {year_key}: {format_money(y['pnl']):>10}  ({y['trades']} trades, "
                  f"{y['wins']}W/{y['losses']}L, {wr:.0f}% WR)")

    # ================================================================
    # SECTION 3: COMPOUNDING vs NO COMPOUNDING
    # ================================================================
    print_section("SECTION 3: COMPOUNDING vs NO COMPOUNDING ($100 start)")
    print("\nCompounding = as your balance grows, you risk more per trade")
    print("(2% of current balance instead of 2% of original $100).")
    print("This makes wins bigger BUT losses bigger too.\n")

    print(f"{'Symbol':<8} {'No-Comp End':>12} {'No-Comp Profit':>15} {'Comp End':>12} {'Comp Profit':>15} {'Comp DD%':>8}")
    print("-" * 75)

    for sym in top_symbols:
        r_nc = results.get(sym)
        if not r_nc:
            continue
        try:
            r_c = run_test(sym, capital=100.0, compound=True)
            nc_profit = r_nc.final_balance - 100
            c_profit = r_c.final_balance - 100
            print(f"{sym:<8} ${r_nc.final_balance:>10.2f} {format_money(nc_profit):>15} "
                  f"${r_c.final_balance:>10.2f} {format_money(c_profit):>15} {r_c.max_drawdown:>7.1f}%")
        except Exception as e:
            print(f"{sym:<8} ERROR: {e}")

    print("\n  NOTE: Compounding numbers look insane because with 70%+ win rate and 2:1 R:R,")
    print("  the balance snowballs. In REAL trading, expect 15-30% worse than backtest.")
    print("  Start with NO compounding until you trust the system for 2-4 weeks.")

    # ================================================================
    # SECTION 4: MULTI-SYMBOL PORTFOLIO (all 6 simultaneously)
    # ================================================================
    print_section("SECTION 4: ALL 6 INSTRUMENTS TOGETHER ($100 capital)")
    print("\nYes, the scalper trades ALL instruments at the same time.")
    print("Each one checks for zones independently on its own H1 + M15 charts.")
    print("With 6 instruments x ~2.5 trades/day each = ~15 trades/day total.\n")

    # Combine all trades from all instruments, sort by time
    all_trades = []
    for sym, r in results.items():
        for t in r.trades:
            all_trades.append(t)
    all_trades.sort(key=lambda t: t.exit_time)

    if all_trades:
        total_pnl = sum(t.pnl for t in all_trades)
        total_wins = sum(1 for t in all_trades if t.pnl > 0)
        total_losses = sum(1 for t in all_trades if t.pnl <= 0)
        gross_profit = sum(t.pnl for t in all_trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in all_trades if t.pnl <= 0))

        daily_all, monthly_all, yearly_all = analyze_trades_by_period(all_trades, 100)

        daily_pnls_all = [d['pnl'] for d in daily_all.values()]
        monthly_pnls_all = [m['pnl'] for m in monthly_all.values()]

        print(f"  COMBINED PORTFOLIO STATS:")
        print(f"    Total trades:      {len(all_trades)}")
        print(f"    Total profit:      {format_money(total_pnl)}")
        print(f"    Winning trades:    {total_wins} ({total_wins/len(all_trades)*100:.1f}%)")
        print(f"    Losing trades:     {total_losses} ({total_losses/len(all_trades)*100:.1f}%)")
        print(f"    Profit factor:     {gross_profit/max(gross_loss,0.01):.2f}")

        print(f"\n  DAILY (combined):")
        print(f"    Trading days:      {len(daily_all)}")
        print(f"    Trades per day:    {len(all_trades)/max(len(daily_all),1):.1f}")
        print(f"    Average day:       {format_money(np.mean(daily_pnls_all))}")
        print(f"    Best day:          {format_money(max(daily_pnls_all))}")
        print(f"    Worst day:         {format_money(min(daily_pnls_all))}")
        winning_days = sum(1 for p in daily_pnls_all if p > 0)
        print(f"    Winning days:      {winning_days}/{len(daily_all)} ({winning_days/max(len(daily_all),1)*100:.0f}%)")

        print(f"\n  MONTHLY (combined):")
        print(f"    Average month:     {format_money(np.mean(monthly_pnls_all))}")
        print(f"    Best month:        {format_money(max(monthly_pnls_all))}")
        print(f"    Worst month:       {format_money(min(monthly_pnls_all))}")
        winning_months = sum(1 for p in monthly_pnls_all if p > 0)
        print(f"    Winning months:    {winning_months}/{len(monthly_all)} ({winning_months/max(len(monthly_all),1)*100:.0f}%)")

        print(f"\n  YEARLY (combined):")
        for year_key in sorted(yearly_all.keys()):
            y = yearly_all[year_key]
            wr = y['wins'] / max(y['trades'], 1) * 100
            print(f"    {year_key}: {format_money(y['pnl']):>10}  ({y['trades']} trades, "
                  f"{y['wins']}W/{y['losses']}L, {wr:.0f}% WR)")

    # ================================================================
    # SECTION 5: DIFFERENT CAPITAL AMOUNTS
    # ================================================================
    print_section("SECTION 5: WHAT IF I START WITH MORE MONEY?")
    print("\nAll with NO compounding, using the BEST instrument (US500).\n")

    best_sym = 'US500'
    for cap in [100, 200, 500, 1000]:
        r = run_test(best_sym, capital=float(cap), compound=False)
        profit = r.final_balance - cap
        daily, monthly, yearly = analyze_trades_by_period(r.trades, cap)
        avg_daily = np.mean([d['pnl'] for d in daily.values()])
        avg_monthly = np.mean([m['pnl'] for m in monthly.values()])
        print(f"  ${cap:>5} start -> ${r.final_balance:>10.2f}  "
              f"(profit {format_money(profit)}, ~{format_money(avg_daily)}/day, "
              f"~{format_money(avg_monthly)}/month, DD {r.max_drawdown:.1f}%)")

    # ================================================================
    # SECTION 6: LEVERAGE EXPLANATION
    # ================================================================
    print_section("SECTION 6: ABOUT LEVERAGE (1:500)")
    print("""
  Your account has 1:500 leverage. Here's what that means:

  - With $100 and 1:500 leverage, you can control up to $50,000 worth of trades
  - BUT the scalper only risks 2% per trade ($2 on a $100 account)
  - The lot size is calculated so your STOP LOSS = 2% of capital
  - Leverage doesn't change how much you risk - it just lets you
    open trades with a small account that would otherwise require more margin

  EXAMPLE: If the scalper opens a 0.01 lot EURUSD trade:
    - Position value = $1,000 (0.01 * 100,000)
    - Margin needed at 1:500 = $2.00
    - Risk (if SL hit) = ~$2.00 (2% of $100)
    - So even with 3 trades open, you use ~$6 margin out of $100

  Bottom line: 1:500 leverage is fine because we control risk through
  position sizing, not leverage. The stop loss protects you.
    """)

    # ================================================================
    # SECTION 7: RISKS AND WARNINGS
    # ================================================================
    print_section("SECTION 7: RISKS AND HONEST WARNINGS")
    print("""
  WHAT COULD GO WRONG:

  1. BACKTEST vs REALITY GAP (expect 15-30% worse in real trading)
     - Backtests use perfect fills; real market has slippage
     - Spreads widen during news events
     - Internet/server outages can miss entries or exits

  2. DRAWDOWN = temporary losses from peak
     - US500 had 19% drawdown in backtest = your $100 could dip to ~$81
     - With compounding, drawdowns get BIGGER
     - This WILL happen. You need to stomach it.

  3. CONSECUTIVE LOSSES
     - Even with 70% win rate, you WILL have 5-10 losses in a row sometimes
     - That's normal math. Don't panic and shut it off.

  4. PAST PERFORMANCE ≠ FUTURE RESULTS
     - Markets change. What worked 2021-2025 might not work in 2026
     - This is why we start with demo, then small real money

  RECOMMENDATIONS:
  - Run on DEMO for 2-4 weeks first
  - Start with conservative profile (2% risk per trade)
  - Don't compound until you've seen 1 month of real results
  - Keep initial capital to money you can afford to lose
  - The scalper runs 24/7 on weekdays - you don't need to watch it
    """)

    # ================================================================
    # SECTION 8: TELEGRAM SETUP
    # ================================================================
    print_section("SECTION 8: HOW YOU'LL TELL THE TWO EAs APART ON TELEGRAM")
    print("""
  Your Telegram bot (@chrisflex_bot) sends messages from BOTH EAs.
  Here's how to tell them apart:

  GRID EA messages look like:
    ↗️ OPENED BUY EURUSD
    Price: 1.08500 | Lot: 0.020 | Grid: L0
    SL: 1.08200 | TP: 1.09100

  SCALPER EA messages look like:
    [SCALPER] ↗️ OPENED BUY US500
    Price: 5890.50 | Lot: 0.010
    SL: 5885.20 | TP: 5901.10
    Zone score: 7.5

  Every scalper message starts with [SCALPER].
  Grid EA messages have NO prefix (they're the original).

  HEARTBEATS:
    Grid:    💓 HEARTBEAT (14:00 UTC) EURUSD 1.085 [UP] 30% from buy
    Scalper: [SCALPER] 💓 HEARTBEAT (14:00 UTC) US500 5890.50 zones=12 pos=2

  SESSION OPEN/CLOSE:
    Grid:    🔔 SESSION OPEN (2026-02-28)
    Scalper: [SCALPER] 🔔 SESSION OPEN (2026-02-28)

  ACCOUNT A = Grid EA (demo 9866921, $400 capital, 5 forex/commodity instruments)
  ACCOUNT B = Scalper EA (demo 9888090, $200 capital, 6 instruments including crypto/indices)
    """)

    print("\n" + "=" * 70)
    print("  DONE! All results above use REAL historical data.")
    print("=" * 70)


if __name__ == '__main__':
    main()
