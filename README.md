# Expert Advisor

Automated trading bot for Forex, Metals, and Indices. Uses Bollinger Band grid entries with trend filtering, only trades during the London/NY overlap (12-16 UTC).

Runs on IC Markets via cTrader API. Sends trade notifications to Telegram.

## What It Trades

| Instrument | Type | Why It Works |
|------------|------|-------------|
| EURUSD | Forex | Most liquid pair, consistent edge |
| GBPUSD | Forex | Strong trend-following behaviour |
| EURJPY | Cross | Best risk-adjusted returns of all 5 |
| XAGUSD | Silver | Different asset class, low correlation |
| US500 | S&P 500 | Stock index, diversifies the portfolio |

All 5 passed Monte Carlo stress testing (100 simulations with random spread/slippage).

## Risk Profiles

| Profile | Annual Return | Max Drawdown | Risk Level |
|---------|-------------|-------------|------------|
| **Conservative** | ~15% | ~4% | Low. Steady growth, small dips |
| **Balanced** | ~27% | ~7% | Medium. Better returns, still safe |
| Growth | ~36% | ~9% | Higher. Failed stress test as portfolio |
| Aggressive | ~51% | ~13% | High. Not recommended for live |

Conservative and Balanced both passed portfolio Monte Carlo at 100% profitable.

## Backtest Results (5.6 years, $500 start)

| Profile | CAGR | Max DD | Sharpe | Profit Factor | Trades | Final Balance |
|---------|------|--------|--------|---------------|--------|--------------|
| Conservative | 15.6% | 3.7% | 2.46 | 2.30 | 768 | $968 |
| Balanced | 26.8% | 6.7% | 2.46 | 2.43 | 768 | $1,481 |

Portfolio correlation between instruments averages 0.038 (nearly zero).

## How to Run

```bash
source .venv/bin/activate

# Demo trading (IC Markets cTrader)
python scripts/run_live.py --broker ctrader --profile conservative --capital 400

# List profiles
python scripts/run_live.py --list-profiles

# Check status
python scripts/run_live.py --status
```

## Setup

1. Create an IC Markets cTrader demo account
2. Register an API app at [open.ctrader.com](https://open.ctrader.com)
3. Run `python scripts/ctrader_auth.py` to authenticate
4. Add credentials to `.env`:
   ```
   CTRADER_CLIENT_ID=...
   CTRADER_CLIENT_SECRET=...
   CTRADER_ACCESS_TOKEN=...
   CTRADER_ACCOUNT_ID=...
   TELEGRAM_BOT_TOKEN=...
   TELEGRAM_CHAT_ID=...
   ```
5. Start trading: `python scripts/run_live.py --broker ctrader`

## Project Structure

```
src/backtesting/   - Backtest engine (Bollinger Band grid strategy)
src/risk/          - Crisis detector, circuit breaker, protection manager
src/live/          - Live trading engine, cTrader broker, signal engine
scripts/           - Entry points (run_live, ctrader_auth, backtests)
data/processed/    - H1 candle data (20 instruments, 8 timeframes)
```

## Strategy

- **Entry**: Price touches Bollinger Band (50-period, 2.0 std dev) + SMA 50/200 trend filter
- **Exit**: Opposite Bollinger Band as take-profit
- **Stop Loss**: 1.5x ATR
- **Trailing Stop**: Break-even at 1.0x ATR profit, trail at 1.5x ATR after 2.0x ATR
- **Grid**: Up to 5 levels with increasing lot size
- **Session**: 12-16 UTC only (London/New York overlap)

## Disclaimer

Trading involves risk. Past performance does not guarantee future results. Only trade with money you can afford to lose.
