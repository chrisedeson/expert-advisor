#!/usr/bin/env python3
"""
Export historical data from MT5 (Exness) to parquet files.
Run this on Windows where MT5 is installed.

Usage:
    python export_mt5_data.py

Requirements:
    pip install MetaTrader5 pandas pyarrow

Output:
    data/processed/{PAIR}_{TIMEFRAME}.parquet
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time

# ============================================================
# 20 MOST POPULAR INSTRUMENTS ACROSS ALL ASSET CLASSES
# ============================================================

INSTRUMENTS = {
    # ── FOREX MAJORS (7) ──
    "EURUSD":  {"pip_size": 0.0001, "pip_value": 10.0,  "spread": 0.7,  "slip": 0.2, "category": "Forex Major"},
    "GBPUSD":  {"pip_size": 0.0001, "pip_value": 10.0,  "spread": 0.9,  "slip": 0.3, "category": "Forex Major"},
    "USDJPY":  {"pip_size": 0.01,   "pip_value": 6.67,  "spread": 0.8,  "slip": 0.3, "category": "Forex Major"},
    "AUDUSD":  {"pip_size": 0.0001, "pip_value": 10.0,  "spread": 0.8,  "slip": 0.3, "category": "Forex Major"},
    "USDCAD":  {"pip_size": 0.0001, "pip_value": 7.40,  "spread": 1.2,  "slip": 0.3, "category": "Forex Major"},
    "NZDUSD":  {"pip_size": 0.0001, "pip_value": 10.0,  "spread": 1.2,  "slip": 0.3, "category": "Forex Major"},
    "USDCHF":  {"pip_size": 0.0001, "pip_value": 11.0,  "spread": 1.0,  "slip": 0.3, "category": "Forex Major"},

    # ── FOREX CROSSES (5) ──
    "EURGBP":  {"pip_size": 0.0001, "pip_value": 12.50, "spread": 1.0,  "slip": 0.3, "category": "Forex Cross"},
    "EURJPY":  {"pip_size": 0.01,   "pip_value": 6.67,  "spread": 1.2,  "slip": 0.3, "category": "Forex Cross"},
    "GBPJPY":  {"pip_size": 0.01,   "pip_value": 6.67,  "spread": 1.5,  "slip": 0.4, "category": "Forex Cross"},
    "EURCHF":  {"pip_size": 0.0001, "pip_value": 11.0,  "spread": 1.2,  "slip": 0.3, "category": "Forex Cross"},
    "EURAUD":  {"pip_size": 0.0001, "pip_value": 6.50,  "spread": 1.5,  "slip": 0.4, "category": "Forex Cross"},

    # ── METALS (2) ──
    "XAUUSD":  {"pip_size": 0.01,   "pip_value": 1.0,   "spread": 3.0,  "slip": 0.5, "category": "Metal"},
    "XAGUSD":  {"pip_size": 0.001,  "pip_value": 50.0,  "spread": 2.0,  "slip": 0.5, "category": "Metal"},

    # ── INDICES (3) ──
    "USTEC":   {"pip_size": 0.1,    "pip_value": 1.0,   "spread": 1.5,  "slip": 1.0, "category": "Index"},  # Nasdaq 100
    "US500":   {"pip_size": 0.1,    "pip_value": 1.0,   "spread": 0.5,  "slip": 0.5, "category": "Index"},  # S&P 500
    "US30":    {"pip_size": 1.0,    "pip_value": 1.0,   "spread": 2.0,  "slip": 1.0, "category": "Index"},  # Dow Jones 30

    # ── CRYPTO (3) ──
    "BTCUSD":  {"pip_size": 1.0,    "pip_value": 1.0,   "spread": 30.0, "slip": 5.0, "category": "Crypto"},
    "ETHUSD":  {"pip_size": 0.01,   "pip_value": 1.0,   "spread": 2.0,  "slip": 1.0, "category": "Crypto"},
    "XRPUSD":  {"pip_size": 0.0001, "pip_value": 1000,  "spread": 0.5,  "slip": 0.2, "category": "Crypto"},
}

# Common Exness symbol name variations
EXNESS_SUFFIXES = ["", "m", "c", ".a", ".b", "_i", "#", ".", "i"]

# Exness-specific alternative names for indices/crypto
SYMBOL_ALIASES = {
    "USTEC":  ["USTEC", "NAS100", "NASDAQ", "NASDAQm", "USTEC.a", "USTECm"],
    "US500":  ["US500", "SPX500", "SP500", "US500m", "US500.a", "SPXm"],
    "US30":   ["US30", "DJ30", "DJI30", "US30m", "US30.a"],
    "BTCUSD": ["BTCUSD", "BTCUSDm", "BTCUSDi", "BTC/USD", "#BTCUSD"],
    "ETHUSD": ["ETHUSD", "ETHUSDm", "ETHUSDi", "ETH/USD", "#ETHUSD"],
    "XRPUSD": ["XRPUSD", "XRPUSDm", "XRPUSDi", "XRP/USD", "#XRPUSD"],
    "XAGUSD": ["XAGUSD", "XAGUSDm", "XAGUSD.a"],
    "XAUUSD": ["XAUUSD", "XAUUSDm", "XAUUSD.a"],
}

# Timeframes - all useful intervals
# M1 is commented out by default (HUGE: ~5M bars per instrument for 10 years)
# Uncomment if you need it, but expect ~500MB+ per pair
TIMEFRAMES = {
    # "M1":  mt5.TIMEFRAME_M1,   # ~5,000,000 bars/10yr - uncomment if needed
    "M5":  mt5.TIMEFRAME_M5,     # ~1,000,000 bars/10yr
    "M15": mt5.TIMEFRAME_M15,    # ~350,000 bars/10yr
    "M30": mt5.TIMEFRAME_M30,    # ~175,000 bars/10yr
    "H1":  mt5.TIMEFRAME_H1,     # ~87,000 bars/10yr
    "H4":  mt5.TIMEFRAME_H4,     # ~22,000 bars/10yr
    "D":   mt5.TIMEFRAME_D1,     # ~2,600 bars/10yr
    "W":   mt5.TIMEFRAME_W1,     # ~520 bars/10yr
    "MN":  mt5.TIMEFRAME_MN1,    # ~120 bars/10yr
}

# 10 years of data
DATE_FROM = datetime(2015, 1, 1)
DATE_TO   = datetime.now()

OUTPUT_DIR = Path("data/processed")


# Max bars per request - chunk size tuned to stay well under MT5 limits
CHUNK_MONTHS = 6  # fetch 6 months at a time


def fetch_rates_chunked(symbol, tf_val, date_from, date_to):
    """Fetch rates in chunks to avoid MT5 'Invalid params' on large requests."""
    all_rates = []
    chunk_start = date_from

    while chunk_start < date_to:
        chunk_end = min(
            chunk_start + timedelta(days=CHUNK_MONTHS * 30),
            date_to
        )
        rates = mt5.copy_rates_range(symbol, tf_val, chunk_start, chunk_end)
        if rates is not None and len(rates) > 0:
            all_rates.append(rates)
        chunk_start = chunk_end
        time.sleep(0.05)  # small delay to not overwhelm the terminal

    if not all_rates:
        return None

    import numpy as np
    combined = np.concatenate(all_rates)
    # Remove duplicates (overlapping boundaries)
    _, unique_idx = np.unique(combined['time'], return_index=True)
    return combined[unique_idx]


def calculate_atr(df, period=14):
    """Calculate Average True Range."""
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def find_symbol(pair, available):
    """Find the correct MT5 symbol name, trying aliases and suffixes."""
    # Try aliases first (for indices/crypto with weird names)
    if pair in SYMBOL_ALIASES:
        for alias in SYMBOL_ALIASES[pair]:
            if alias in available:
                return alias

    # Try exact match + common suffixes
    for suffix in EXNESS_SUFFIXES:
        candidate = pair + suffix
        if candidate in available:
            return candidate

    return None


def main():
    if not mt5.initialize():
        print(f"MT5 initialize failed: {mt5.last_error()}")
        print("Make sure MT5 is running and logged into your Exness account!")
        return

    print(f"MT5 version: {mt5.version()}")
    print(f"Account: {mt5.account_info().login} ({mt5.account_info().server})")
    print(f"Date range: {DATE_FROM.strftime('%Y-%m-%d')} to {DATE_TO.strftime('%Y-%m-%d')} (~10 years)")
    print()

    available = [s.name for s in mt5.symbols_get()]
    print(f"Total symbols available on broker: {len(available)}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    skipped = []

    for pair, info in INSTRUMENTS.items():
        symbol = find_symbol(pair, available)

        if symbol is None:
            print(f"  [SKIP] {pair:10s} ({info['category']}) - not found in MT5")
            # Show similar symbols to help debug
            similar = [s for s in available if pair[:3] in s][:8]
            if similar:
                print(f"         Similar: {similar}")
            skipped.append(pair)
            continue

        mt5.symbol_select(symbol, True)
        print(f"  [{info['category']:12s}] {pair:10s} (MT5: {symbol})")

        for tf_name, tf_val in TIMEFRAMES.items():
            rates = fetch_rates_chunked(symbol, tf_val, DATE_FROM, DATE_TO)

            if rates is None or len(rates) == 0:
                print(f"    {tf_name}: NO DATA ({mt5.last_error()})")
                continue

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            df = df[['open', 'high', 'low', 'close', 'tick_volume']].copy()
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
            df['volume'] = df['volume'].astype(np.uint64)
            df['atr'] = calculate_atr(df, period=14)
            df.dropna(inplace=True)

            output_file = OUTPUT_DIR / f"{pair}_{tf_name}.parquet"
            df.to_parquet(output_file, engine='pyarrow')

            years = (df.index[-1] - df.index[0]).days / 365.25
            print(f"    {tf_name}: {len(df):>8,} bars  "
                  f"({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})  "
                  f"[{years:.1f} years]")

            results.append({
                'pair': pair, 'tf': tf_name, 'bars': len(df),
                'from': df.index[0], 'to': df.index[-1],
                'category': info['category'], 'symbol': symbol
            })

    # ── SUMMARY ──
    print("\n" + "=" * 80)
    print("EXPORT SUMMARY")
    print("=" * 80)

    categories = {}
    for r in results:
        cat = r['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    for cat, items in categories.items():
        print(f"\n  {cat}:")
        for r in items:
            print(f"    {r['pair']:10s} {r['tf']:3s}  {r['bars']:>8,} bars  "
                  f"({r['from'].strftime('%Y-%m-%d')} to {r['to'].strftime('%Y-%m-%d')})")

    print(f"\nTotal: {len(results)} files from {len(set(r['pair'] for r in results))} instruments")
    if skipped:
        print(f"Skipped: {skipped}")
    print(f"Output: {OUTPUT_DIR.resolve()}")

    # ── PIP CONFIG FOR BACKTESTER ──
    print("\n" + "=" * 80)
    print("PAIR_CONFIGS for test scripts (copy-paste into test_multi_pair.py):")
    print("=" * 80)
    print("PAIR_CONFIGS = {")
    exported_pairs = set(r['pair'] for r in results)
    for pair, info in INSTRUMENTS.items():
        if pair in exported_pairs:
            print(f'    "{pair}": {{"pip_size": {info["pip_size"]}, '
                  f'"pip_value_per_lot": {info["pip_value"]}, '
                  f'"spread_pips": {info["spread"]}, '
                  f'"slippage_pips": {info["slip"]}}},  # {info["category"]}')
    print("}")

    mt5.shutdown()
    print("\nDone! Transfer to EC2 with:")
    print("  scp data/processed/*.parquet expert-advisor:~/expert-advisor/data/processed/")


if __name__ == "__main__":
    main()
