# Expert Advisor - Optimization Journal

> **FUTURE SESSIONS: Always update this journal with breakthroughs, lessons, and results.**
> **On context compression or new session, READ this first and WRITE to it before finishing.**

---

## Risk Profiles Discovered (EUR-only, $500 start, 5.6 years backtest)

| Profile | Lot | Mult | CAGR | Max DD | Sharpe | Final $ | Who it's for |
|---------|-----|------|------|--------|--------|---------|-------------|
| Conservative | 0.0258 | 1.5 | 23% | 19% | 1.01 | $1,650 | "I want steady growth, can't stomach big drops" |
| Growth | 0.04 | 2.0 | 60% | 45% | 1.06 | ~$8K | "I'm OK with some rough patches for bigger gains" |
| Aggressive | 0.05 | 2.0 | 63% | 50% | 1.09 | ~$13K | "I can handle seeing my account drop 50%" |
| High Risk | 0.06 | 2.5 | 85% | 57% | 1.09 | $36K | "I want max growth, I understand the swings" |
| Max Return | 0.10 | 2.5 | 115% | 81% | 1.08 | $112K | "I'm a degen, let's ride" |

**In plain English:**
- Conservative: $500 becomes ~$1,650 in 5.6 years. Like a really good savings account.
- High Risk: $500 becomes ~$36,000 in 5.6 years. Your account might drop 57% at some point but recovers.
- Max Return: $500 becomes ~$112,000. Buckle up - your $500 might drop to $95 before bouncing back.

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

### Config Gotchas:
- `load_config()` must return hardcoded defaults, NOT parse config.yaml
- Protection manager gets `cash_balance` (realized P&L only), NOT equity
- Profit protector threshold=1.0 to disable (otherwise fights compounding)
- Always run with venv: `source .venv/bin/activate && python ...`

---

## Current Best Configurations

### Conservative (4/5 GO criteria - production ready)
```
EUR 80% ($400) + GBP 20% ($100), lot=0.0258/0.0180
CAGR: 23.06%, Max DD: 19.09%, Sharpe: 1.01, PF: 1.43
```

### Aggressive (best risk-adjusted returns)
```
EUR only, lot=0.06, mult=2.5, crisis+CB ON, recovery mgr OFF
CAGR: 85.28%, Max DD: 56.58%, Sharpe: 1.09, PF: 1.38
```

---

## Open Questions / Next Steps
- Can we boost trade frequency without diluting edge?
- BB(1.75) needs more testing across different lot sizes
- H4 timeframe? Different entry signals?
- The $500 -> $2,115 in 6 months target requires ~2,500% CAGR - H1 limits us to ~89 trades/year
- Deploy to EC2 for unlimited compute optimization runs

---

*Last updated: 2026-02-10*
