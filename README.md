# Expert Advisor Trading System

Conservative Forex Expert Advisor with rigorous backtesting. Built to prove profitability before risking real capital.

## Overview

- **Market**: Forex (EURUSD, GBPUSD, USDJPY, AUDUSD)
- **Broker**: Exness MT5
- **Strategy**: Trend following with ADX filter
- **Target**: 10-20% annual return, max 15% drawdown
- **Capital**: Start with $100, scale after proof

## Quick Start

### Setup (WSL/Linux)

```bash
# Clone and install
git clone <repo-url>
cd expert-advisor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Export Data (Windows PowerShell)

MT5 Python library only works on Windows. Export data once, then develop in WSL.

```powershell
cd \\wsl$\Ubuntu\home\chris\expert-advisor
.venv-windows\Scripts\activate
pip install MetaTrader5 pandas pyarrow loguru
python scripts\export_mt5_data_windows.py
```

### Test Strategy (WSL)

```bash
# Test backtesting engine
python scripts/test_backtest_engine.py

# Test trend-following strategy
python scripts/test_trend_strategy.py
```
## Success Criteria (Before Live Trading)

**Must pass ALL:**
- CAGR > 12% (out-of-sample)
- Sharpe ratio > 1.0
- Max drawdown < 15%
- Profitable in ≥60% of 6-month periods
- Works on ≥3 of 4 pairs
- Out-of-sample Sharpe > 70% of in-sample
- PBO < 30%
- 4+ weeks successful paper trading

## Risk Management

- 1% risk per trade
- ATR-based stop-loss (2.5x ATR)
- 15% max drawdown circuit breaker
- Start $100, scale to $500+ after proof

## Disclaimer

Trading involves risk. Software provided for educational purposes only. Past performance does not guarantee future results. Only trade with capital you can afford to lose.
