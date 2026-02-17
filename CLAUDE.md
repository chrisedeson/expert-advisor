# CLAUDE.md - Full Technical Context for Autonomous Sessions

> **This file is automatically loaded by Claude Code at startup.**
> **It contains everything you need to continue building this EA.**

## Mission
Multi-instrument grid trading EA across Forex, Metals, and Indices.
Optimization is COMPLETE. Now building the live trading engine.

### Current Status (Feb 2026)
- Optimization: DONE (Phases 1-9 complete)
- 5 MC-validated instruments: EURUSD, GBPUSD, EURJPY, XAGUSD, US500
- Session filter: 12-16 UTC (London/NY overlap) - THE key edge
- Portfolio MC: Conservative and Balanced pass at 100% profitable
- Live engine: DEPLOYED as systemd service on EC2 (paper trading)
- Next: Paper trade 1-3 months, then connect real broker for live

### Industry Benchmarks
- Waka Waka EA (best grid EA): 18% annual, 12% DD
- Professional traders: 20-50% annual, 5-15% DD
- Hedge funds: 20-80% annual, <20% DD
- **Our conservative portfolio: 15.6% CAGR, 3.7% DD, Sharpe 2.46**
- **Our balanced portfolio: 26.8% CAGR, 6.7% DD, Sharpe 2.46**
- Backtest-to-live degradation: expect 15-30% worse

## How to Run

```bash
# ALWAYS activate venv first
source .venv/bin/activate

# Backtest scripts (optimization complete - use for verification only)
python scripts/test_protected_system.py          # Single-pair conservative
python scripts/opt_phase8_all_assets.py          # Multi-asset screening
python scripts/opt_phase9_portfolio.py           # Portfolio optimization

# Redirect output for long runs:
python -u scripts/opt_phase9_portfolio.py > /tmp/results.txt 2>&1
```

## Architecture

### Core Engine: `src/backtesting/protected_grid_engine.py`
- Class: `ProtectedGridBacktester`
- Constructor: initial_balance, config dict, spread_pips, slippage_pips, pip_size, pip_value_per_lot
- `run_backtest(data)` returns BacktestResult with: cagr, max_drawdown, sharpe_ratio, profit_factor, win_rate, total_trades, final_balance, equity_curve, trades
- Entry: BB(50, 2.0) + SMA(50)/SMA(200) trend filter
- Grid: up to 5 levels, lot_multiplier per level, grid_spacing 0.75*ATR
- Exit: opposite BB band as TP, two-stage trail (BE at 1.0*ATR, trail at 1.5*ATR after 2.0*ATR)
- SL: 1.5 * ATR
- Compounding: equity_ratio = cash_balance / initial_balance

### Golden Parameters (DO NOT CHANGE)
```python
bb_entry_mult: 2.0          # BB standard deviation multiplier
grid_spacing_atr: 0.75      # Grid spacing as ATR multiple
sl_atr_mult: 1.5            # Stop loss ATR multiplier
use_trend_filter: True       # SMA50/200 trend filter - NEVER DISABLE
compound_on_equity: False    # Cash-based compounding
max_grid_levels: 5           # Max grid depth
```

### Protection System: `src/risk/`
- `protection_manager.py` - Orchestrates all. Gets CASH balance, NOT equity.
- `crisis_detector.py` - Spike/DD/consecutive loss (6.0/0.50/15)
- `circuit_breaker.py` - Daily/weekly/monthly limits (20%/35%/50%)
- `recovery_manager.py` - DISABLED (threshold=1.0)
- `volatility_filter.py` - Effectively disabled (10.0/20.0)
- `profit_protector.py` - DISABLED for compounding (threshold=100.0)

### Live Trading Engine: `src/live/` (DEPLOYED)
- `signal_engine.py` - Signal logic extracted from backtest engine (BB entry, trend filter, trailing stops)
- `live_engine.py` - Main loop with risk profiles, position management, state persistence
- `broker_interface.py` - Abstract broker API (swap in real broker later)
- `simulated_broker.py` - Paper trading broker (loads candle data from parquet files)
- `session_filter.py` - 12-16 UTC overlap filter (skips weekends)
- `state_manager.py` - JSON state persistence + JSONL trade log

### Live Engine Entry Points
- `scripts/run_live.py` - CLI with --profile, --capital, --status, --list-profiles
- `configs/live_config.yaml` - Live trading configuration
- `systemd/expert-advisor.service` - Auto-restart service (deployed on EC2)

### Live Engine Commands
```bash
# Check service status
sudo systemctl status expert-advisor

# View live logs
tail -f logs/service.log

# List available profiles
python scripts/run_live.py --list-profiles

# Run with different profile
python scripts/run_live.py --profile balanced --capital 1000

# Check engine status
python scripts/run_live.py --status
```

## MC-Validated Instruments (Phase 8)

### Winners - USE THESE
| Instrument | pip_size | pip_value | spread | slip | lot_scale | OL Sharpe | MC Prof |
|------------|----------|-----------|--------|------|-----------|-----------|---------|
| EURUSD | 0.0001 | 10.0 | 0.7 | 0.2 | 1.0 | 1.31 | 100% |
| GBPUSD | 0.0001 | 10.0 | 0.9 | 0.3 | 1.0 | 1.28 | 100% |
| EURJPY | 0.01 | 6.67 | 1.0 | 0.3 | 1.0 | 1.41 | 100% |
| XAGUSD | 0.001 | 5.0 | 3.0 | 0.5 | 0.1 | 0.81 | 96% |
| US500 | 0.01 | 1.0 | 0.5 | 0.2 | 0.1 | 1.17 | 96% |

### Failed - DON'T USE
- **Forex**: USDJPY, AUDUSD, NZDUSD, USDCAD, USDCHF, EURGBP, GBPJPY, EURAUD, EURCHF
- **Crypto**: BTCUSD, ETHUSD, XRPUSD (zero profitable configs)
- **Indices**: US30, USTEC
- **Metal**: XAUUSD (marginal, XAGUSD is better)

## Portfolio Results (Phase 9)

### Correlation Matrix (all pairs EXCELLENT, avg |corr| = 0.038)
```
              EURUSD  GBPUSD  EURJPY  XAGUSD   US500
EURUSD         1.000   0.143   0.046   0.006   0.010
GBPUSD         0.143   1.000   0.001   0.007   0.090
EURJPY         0.046   0.001   1.000  -0.011   0.064
XAGUSD         0.006   0.007  -0.011   1.000  -0.005
US500          0.010   0.090   0.064  -0.005   1.000
```

### Portfolio Backtest (overlap, $500 start, 5.6 years)
| Profile | Lot | Mult | CAGR | DD | Sharpe | PF | Trades | $500-> |
|---------|-----|------|------|----|--------|-----|--------|--------|
| **Conservative** | 0.02 | 1.5 | 15.6% | 3.7% | **2.46** | 2.30 | 768 | $968 |
| **Balanced** | 0.03 | 2.0 | 26.8% | 6.7% | **2.46** | 2.43 | 768 | $1,481 |
| Growth | 0.04 | 2.0 | 35.9% | 9.1% | 2.45 | 2.44 | 768 | $2,034 |
| Aggressive | 0.05 | 2.5 | 50.7% | 12.9% | 2.37 | 2.52 | 768 | $3,259 |
| Max Return | 0.06 | 2.5 | 61.0% | 15.5% | 2.32 | 2.50 | 768 | $4,417 |

### Portfolio MC Validation (50 sims)
| Profile | Pass | Prof% | CAGR P5 | CAGR Med | DD P95 | Sharpe P5 |
|---------|------|-------|---------|----------|--------|-----------|
| **Conservative** | **PASS** | 100% | 16.7% | 49.2% | 24.6% | 0.86 |
| **Balanced** | **PASS** | 100% | 36.4% | 88.3% | 45.7% | 0.78 |
| Growth | FAIL | 74% | -7.4% | 45.1% | 53.6% | -0.40 |
| Aggressive | FAIL | 44% | -11.9% | -2.2% | 66.1% | -0.44 |
| Max Return | FAIL | 36% | -15.0% | -7.4% | 73.2% | -0.53 |

### Recommended Allocations (Equal Weight works best for Sharpe)
```
Equal Weight:  EURUSD 20%, GBPUSD 20%, EURJPY 20%, XAGUSD 20%, US500 20%
Sharpe Weight: EURUSD 25%, GBPUSD 15%, EURJPY 30%, XAGUSD 10%, US500 20%
```

### Yearly Profit Factor (Balanced, per instrument)
| Year | EURUSD | GBPUSD | EURJPY | XAGUSD | US500 |
|------|--------|--------|--------|--------|-------|
| 2021 | 2.56 | 1.01 | 1.89 | 0.76 | 4.52 |
| 2022 | 1.62 | 0.94 | 1.12 | 1.39 | 1.57 |
| 2023 | 2.13 | 1.60 | 2.42 | 1.99 | 2.28 |
| 2024 | 2.90 | 2.82 | 1.83 | 5.36 | 1.31 |
| 2025 | 5.15 | 4.71 | 5.52 | 7.72 | 3.48 |

## Data Files

All in `data/processed/{PAIR}_{TIMEFRAME}.parquet`
- 20 instruments x 8 timeframes (M5, M15, M30, H1, H4, D, W, MN)
- H1 overlap data: ~5,900 bars per instrument
- Columns: time(index), open, high, low, close, volume, atr

## Optimization History (What's Been Tried - ALL COMPLETE)

### Things That WORK (Keep These)
1. BB(50, 2.0) entry with opposite BB as TP
2. SMA50/SMA200 trend filter (NEVER disable)
3. Two-stage trail: BE at 1.0*ATR, trail 1.5*ATR after 2.0*ATR
4. SL at 1.5*ATR (SL=1.2 fails MC, SL=2.0 lower Sharpe)
5. Crisis detector at 6.0/0.50/15 (helps at all settings)
6. Disabling recovery manager and profit protector
7. Grid spacing 0.75*ATR, lot mult 1.5-2.5
8. 12-16 UTC overlap filter (THE edge, +45% Sharpe improvement)
9. 5-instrument portfolio (near-zero correlations)
10. Walk-forward validated: OOS outperforms IS

### Things That FAILED (Don't Repeat)
1. BB(35,2.0), BB(50,1.5), BB(50,1.75) - dilute edge
2. Trail at 1.0*ATR - too tight for H1
3. Removing trend filter - strategy dies
4. Grid spacing 0.5*ATR - account blowup
5. Equity compounding - worse than cash
6. USDJPY, AUDUSD - lose money
7. All crypto - zero profitable configs
8. H4+overlap - too few trades (52)
9. Growth/Aggressive/Max profiles fail portfolio MC
10. Lot multiplier 1.3 - worse than 1.5

## Important Gotchas
1. **Always use venv**: `source .venv/bin/activate && python ...`
2. **load_config() is hardcoded**: Does NOT parse config.yaml
3. **Protection manager gets cash_balance**, not equity
4. **Recovery manager threshold=1.0 to disable**
5. **Profit protector threshold=100.0 to disable**
6. **loguru must be silenced**: `logger.remove()` before importing engine
7. **Use `python -u` + `flush=True`** for real-time output to files
8. **lot_scale for metals/indices**: XAGUSD and US500 use 0.1x lot scale
9. **$100 capital = 0.2x lot scale** (lot * 100/500)

## Git Info
- GitHub: `git@github.com:chrisedeson/expert-advisor.git` (SSH auth)
- Branch: master
- Always commit before risky changes

## JOURNAL.md
- User reads and edits JOURNAL.md directly - respect their format
- Update with new results in layman terms
- Don't overwrite user's edits
