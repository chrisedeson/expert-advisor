#!/usr/bin/env python3
"""
Export historical data from MT5 (RUN THIS FROM WINDOWS, NOT WSL)

This script should be run from Windows PowerShell/CMD because
MetaTrader5 Python library only works on Windows.

After exporting, the data can be used in WSL for backtesting.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 package not installed!")
    print()
    print("This script must be run from Windows with:")
    print("  pip install MetaTrader5")
    sys.exit(1)

import pandas as pd
from loguru import logger


def export_data(symbol, timeframe, start_date, end_date, output_dir):
    """Export historical data from MT5"""

    # Initialize MT5
    if not mt5.initialize():
        print(f"ERROR: MT5 initialization failed: {mt5.last_error()}")
        print()
        print("Make sure:")
        print("1. MetaTrader 5 terminal is installed")
        print("2. MT5 terminal is RUNNING (check system tray)")
        print("3. You are logged in to your Exness account in MT5")
        return False

    # Check if terminal is connected
    terminal_info = mt5.terminal_info()
    if terminal_info is None:
        print("ERROR: Cannot get terminal info. Is MT5 running?")
        mt5.shutdown()
        return False

    print(f"MT5 Terminal: {terminal_info.name} Build {terminal_info.build}")
    print(f"Connected: {terminal_info.connected}")

    # Login
    login = 297876007  # Your Exness account
    password = "Chrisflex@1219"
    server = "Exness-MT5Trial9"

    if not mt5.login(login, password=password, server=server):
        print(f"Login failed: {mt5.last_error()}")
        print()
        print("Check that:")
        print(f"  Login: {login}")
        print(f"  Server: {server}")
        print("  Password: (check .env file)")
        mt5.shutdown()
        return False

    print(f"Connected to MT5: {server}, Account: {login}")

    # Enable symbol in Market Watch (CRITICAL!)
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to enable symbol {symbol}: {mt5.last_error()}")
        print(f"Try enabling {symbol} manually in MT5 Market Watch")
        mt5.shutdown()
        return False

    # Wait for symbol to be available
    import time
    time.sleep(0.5)

    # Get data
    print(f"Fetching {symbol} {timeframe} from {start_date} to {end_date}...")

    # Convert timeframe string to MT5 constant
    timeframe_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D": mt5.TIMEFRAME_D1,
    }
    tf = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)

    # Fetch rates
    rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)

    if rates is None or len(rates) == 0:
        print(f"No data returned: {mt5.last_error()}")
        mt5.shutdown()
        return False

    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df = df[["open", "high", "low", "close", "tick_volume"]]
    df.rename(columns={"tick_volume": "volume"}, inplace=True)

    print(f"✓ Fetched {len(df)} candles")

    # Save to parquet
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{symbol}_{timeframe}.parquet"
    df.to_parquet(output_file, compression="snappy")

    print(f"✓ Saved to: {output_file}")
    print()
    print(f"Data range: {df.index[0]} to {df.index[-1]}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

    mt5.shutdown()
    return True


def main():
    """Export data for all major pairs"""

    # Exness uses "m" suffix for symbols
    # Added XAUUSDm (GOLD) - could be game changer!
    pairs = ["EURUSDm", "GBPUSDm", "USDJPYm", "AUDUSDm", "XAUUSDm"]

    # Focus on reliable timeframes with sufficient historical data
    # Note: Exness doesn't provide M1-M30 data back to 2019 (broker limitation)
    # H1, H4, D are more reliable for backtesting anyway (less overfitting, lower costs)
    timeframes = ["H1", "H4", "D"]

    start_date = datetime(2019, 1, 1)
    end_date = datetime(2025, 12, 31)

    output_dir = Path(__file__).parent.parent / "data" / "raw"

    print("=" * 60)
    print("MT5 Historical Data Export")
    print("=" * 60)
    print()
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Output directory: {output_dir}")
    print()

    success_count = 0
    total_count = len(pairs) * len(timeframes)

    for pair in pairs:
        for timeframe in timeframes:
            print(f"[{success_count + 1}/{total_count}] {pair} {timeframe}")
            if export_data(pair, timeframe, start_date, end_date, output_dir):
                success_count += 1
            print()

    print("=" * 60)
    print(f"Export complete: {success_count}/{total_count} successful")
    print("=" * 60)
    print()
    print("You can now use this data in WSL for backtesting!")
    print("The data is in: data/raw/")


if __name__ == "__main__":
    main()
