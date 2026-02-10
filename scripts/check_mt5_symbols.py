#!/usr/bin/env python3
"""
Check available symbols in MT5 (RUN THIS FROM WINDOWS)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 package not installed!")
    sys.exit(1)


def main():
    """Check available symbols"""

    # Initialize MT5
    if not mt5.initialize():
        print(f"MT5 initialization failed: {mt5.last_error()}")
        return

    # Login
    login = 297876007
    password = "Chrisflex@1219"
    server = "Exness-MT5Trial9"

    if not mt5.login(login, password=password, server=server):
        print(f"Login failed: {mt5.last_error()}")
        mt5.shutdown()
        return

    print("=" * 60)
    print("Available Symbols in MT5")
    print("=" * 60)
    print()

    # Get all symbols
    symbols = mt5.symbols_get()

    if symbols is None:
        print("No symbols found")
        mt5.shutdown()
        return

    print(f"Total symbols: {len(symbols)}")
    print()

    # Filter forex symbols
    forex_pairs = []
    for symbol in symbols:
        name = symbol.name
        # Look for major forex pairs
        if any(pair in name.upper() for pair in ["EUR", "GBP", "USD", "JPY", "AUD", "NZD", "CAD", "CHF"]):
            # Must be 6-9 characters (typical forex pair length with suffix)
            if 6 <= len(name) <= 12 and symbol.trade_mode != 0:  # tradeable
                forex_pairs.append({
                    "name": name,
                    "description": symbol.description,
                    "visible": symbol.visible,
                    "spread": symbol.spread,
                })

    print("FOREX PAIRS (tradeable):")
    print("-" * 60)
    for pair in sorted(forex_pairs, key=lambda x: x["name"]):
        status = "✓" if pair["visible"] else " "
        print(f"[{status}] {pair['name']:15s} - {pair['description']:30s} (spread: {pair['spread']} pips)")

    print()
    print("=" * 60)
    print()
    print("The symbols with ✓ are in your Market Watch")
    print()
    print("IMPORTANT: Note the EXACT symbol names above!")
    print("You may see names like:")
    print("  - EURUSDm")
    print("  - EURUSD.a")
    print("  - EURUSD")
    print("  - or other variations")

    mt5.shutdown()


if __name__ == "__main__":
    main()
