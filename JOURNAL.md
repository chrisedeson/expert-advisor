# Expert Advisor - Optimization Journal

> **FUTURE SESSIONS: Always update this journal with breakthroughs, lessons, and results.**
> **On context compression or new session, READ this first and WRITE to it before finishing.**

---

## Reality Check: Industry Benchmarks (Research Feb 2026)

Before you optimize, know what's REAL in the industry:

**Top Commercial EAs:**
- Waka Waka EA: 18% annual, 12% max DD (the gold standard for grid EAs)
- Forex Flex EA: 10-20% monthly, <5% DD
- FX Stabilizer: 1,955% total gain, 21.47% DD

**Professional Trader Returns:**
- Conservative: 20-50% annual, 5-15% DD
- Aggressive: 50-100% annual, 20-40% DD
- Hedge Funds: 20-80% annual, <20% DD

**Backtest-to-Live Gap (CRITICAL):**
- Expect 15-30% degradation on conservative setups
- Expect 30-35% degradation on aggressive setups
- Grid/martingale systems are ESPECIALLY sensitive to this gap
- Main killers: spread widening, slippage during volatility, missed fills

**Our EA vs the market:**
- Conservative 23% CAGR = Beats most hedge funds. REALISTIC.
- Aggressive 85% CAGR = Top 1% territory. Possible but brutal DD.
- Max 115% CAGR = Backtest fantasy. 81% DD won't survive live.

---

## Risk Profiles Discovered (EUR-only, $500 start, 5.6 years backtest)

| Profile | Lot | Mult | CAGR | Max DD | Sharpe | Final $ | Who it's for |
|---------|-----|------|------|--------|--------|---------|-------------|
| Conservative | 0.0258 | 1.5 | 23% | 19% | 1.01 | $1,650 | "I want steady growth, can't stomach big drops" |
| Growth | 0.04 | 2.0 | 60% | 45% | 1.06 | ~$8K | "I'm OK with some rough patches for bigger gains" |
| Aggressive | 0.05 | 2.0 | 63% | 50% | 1.09 | ~$13K | "I can handle seeing my account drop 50%" |
| High Risk | 0.06 | 2.5 | 85% | 57% | 1.09 | $36K | "I want max growth, I understand the swings" |
| Max Return | 0.10 | 2.5 | 115% | 81% | 1.08 | $112K | "Backtest only - won't survive live" |

**What this means with $100 starting capital (after live degradation):**
- Conservative (~18% live): $100 -> $118/yr, $227 in 5 years
- Growth (~45% live): $100 -> $145/yr, more volatile
- Aggressive (~50% live): $100 -> $160/yr, will see $40 lows
- Anything above: gambling territory

---

## Breakthroughs (What Worked)

### 1. Opposite Bollinger Band TP (Round 5)
Instead of fixed TP, exit at the opposite BB band. This lets winners ride with the market.

### 2. Two-Stage Trailing Stop (Round 10)
- Stage 1: Move SL to breakeven after 1.0*ATR profit
- Stage 2: Trail at 1.5*ATR after 2.0*ATR profit
- Result: PF jumped from 1.18 to 1.46

### 3. Crisis Detector HELPS at Aggressive Settings (Round 15)
Counter-intuitive: keeping the crisis detector ON at 6.0/0.50/15 gives 2.65x MORE money than turning it off. It saves you from the worst crashes.

### 4. Recovery Manager HURTS at High Lots (Multi-pair phase)
Recovery manager reduces lot size to 30% after drawdown and never recovers. At aggressive settings, disabling it is worth 3-4x more returns.

### 5. Portfolio Diversification (Multi-pair phase)
EUR 80% + GBP 20% gives Sharpe boost from 0.90 to 1.01 even though GBP alone is mediocre (Sharpe 0.43). Correlation between them is only 0.057.

### 6. BB(1.75) Middle Ground (Frequency experiments)
- BB(2.0): 621 trades, 85% CAGR, PF 1.38
- BB(1.75): 812 trades, 78% CAGR, PF 1.18 (30% more trades, modest CAGR drop)
- BB(1.5): 991 trades, 37% CAGR, PF 1.09 (too loose, dilutes edge)

---

## Lessons Learned (What to Avoid)

### Don't Do These:
1. **BB(35) or BB(50,1.5)**: Dilutes the edge, PF crashes
2. **Trail at 1.0*ATR**: Too tight for H1 retracements, gets stopped out
3. **Remove trend filter**: Strategy dies - only 10 trades, counter-trend losses trigger CB shutdown
4. **Grid spacing 0.5*ATR**: Account blowup (-$89). Too many grid levels too fast
5. **USDJPY or AUDUSD**: Strategy loses money on these pairs
6. **Equity compounding**: Slightly WORSE than cash-based compounding
7. **Lot multiplier 1.3**: Worse Sharpe than 1.5
8. **DD > 50% in live trading**: Psychologically unbearable, most traders quit
9. **Chasing 2,500% CAGR**: Doesn't exist. Even 115% CAGR has 81% DD.

### Config Gotchas:
- `load_config()` must return hardcoded defaults, NOT parse config.yaml
- Protection manager gets `cash_balance` (realized P&L only), NOT equity
- Profit protector threshold=1.0 to disable (otherwise fights compounding)
- Always run with venv: `source .venv/bin/activate && python ...`

---

## Current Best Configurations

### Conservative (4/5 GO criteria - PRODUCTION READY)
```
Portfolio: EUR 80% ($400) + GBP 20% ($100), lot=0.0258/0.0180
CAGR: 23.06%, Max DD: 19.09%, Sharpe: 1.01, PF: 1.43
Live estimate: ~18% CAGR, ~24% DD (still excellent)
```

### Aggressive (best risk-adjusted returns)
```
EUR only, lot=0.06, mult=2.5, crisis+CB ON, recovery mgr OFF
CAGR: 85.28%, Max DD: 56.58%, Sharpe: 1.09, PF: 1.38
Live estimate: ~60% CAGR, ~65% DD (painful but possible)
```

---

## Phase 8: Multi-Asset Backtest (20 Instruments, Feb 11 2026)

Tested ALL 20 available instruments with the overlap strategy (12-16 UTC) on H1 data.

### MC-Validated Winners (5 instruments!)

| Instrument | Class | OL CAGR | OL DD | OL Sharpe | MC Prof | MC CAGR P5 | MC Sharpe Med |
|------------|-------|---------|-------|-----------|---------|------------|---------------|
| **EURUSD** | Forex | 17.2% | 8.1% | **1.31** | 100% | 12.9% | 1.25 |
| **GBPUSD** | Forex | 14.6% | 10.6% | **1.28** | 100% | 8.7% | 1.03 |
| **EURJPY** | Cross | 17.7% | 12.8% | **1.41** | 100% | 14.5% | 1.35 |
| **XAGUSD** | Metal | 55.7% | 36.1% | 0.81 | 96% | 24.8% | 0.79 |
| **US500** | Index | 40.5% | 21.1% | **1.17** | 96% | 13.1% | 1.00 |

### Promising but Failed MC (4 instruments)
- **AUDUSD**: OL Sharpe 0.96, MC 88% profitable (close but not enough)
- **NZDUSD**: OL Sharpe 0.88, MC 92% but Sharpe P5 negative
- **EURAUD**: OL Sharpe 1.15 but MC only 64% profitable (too volatile)
- **XAUUSD**: OL Sharpe 1.25, CAGR 5.7% (good quality, too few returns at tested lots)

### Completely Failed (9 instruments)
- **Forex**: USDJPY, USDCAD, USDCHF, EURGBP, GBPJPY
- **Crypto**: BTCUSD, ETHUSD, XRPUSD (zero profitable configs!)
- **Indices**: US30, USTEC

### Key Takeaways
1. **EURJPY is the hidden gem** - highest Sharpe (1.41) of ALL instruments! Better than EURUSD!
2. **US500 (S&P 500) works!** - completely different asset class, great diversification
3. **Silver (XAGUSD) is aggressive but real** - 55.7% CAGR with PF 6.42!
4. **Crypto doesn't work AT ALL** with BB grid strategy - zero profitable configs
5. **Overlap filter transforms everything** - even USD pairs go from garbage to barely viable
6. **5-instrument portfolio possible** for real diversification across forex, metals, indices

### What This Means for Live Trading
With 5 validated instruments, we can build a diversified portfolio:
- **Core**: EURUSD + EURJPY + GBPUSD (forex, 100% MC pass rate)
- **Satellite**: US500 + XAGUSD (different asset classes, 96% MC pass)
- All trading 12-16 UTC only (London/NY overlap)
- Low correlation between asset classes = lower portfolio DD

---

## Open Questions / Next Steps
- ~~Monte Carlo validation~~ DONE - all 5 configs pass
- ~~H4 timeframe testing~~ DONE - doesn't work with overlap (too few trades)
- ~~XAUUSD testing~~ DONE - marginal, XAGUSD is better
- ~~Session filters~~ DONE - 12-16 UTC overlap is THE edge
- ~~Walk-forward validation~~ DONE - OOS outperforms IS
- Portfolio correlation testing (5-instrument portfolio)
- Build live trading engine
- Deploy and paper trade

---

*Last updated: 2026-02-11*
