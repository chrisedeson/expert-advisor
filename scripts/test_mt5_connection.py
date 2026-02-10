#!/usr/bin/env python3
"""
Test MT5 connection to Exness
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.mt5_client import MT5Client
from src.utils.logger import logger


def main():
    """Test MT5 connection"""
    print("=" * 60)
    print("MT5 / Exness Connection Test")
    print("=" * 60)
    print()

    try:
        # Initialize client
        print("Initializing MT5 client...")
        client = MT5Client()
        print(f"✓ Client initialized: {client}")
        print()

        # Test connection
        print("Testing connection...")
        if not client.test_connection():
            print("✗ Connection test failed")
            client.disconnect()
            return 1
        print()

        # Get account info
        print("Account Information:")
        info = client.get_account_info()
        print(f"  Login:        {info['login']}")
        print(f"  Server:       {info['server']}")
        print(f"  Balance:      {info['balance']} {info['currency']}")
        print(f"  Equity:       {info['equity']} {info['currency']}")
        print(f"  Free Margin:  {info['margin_free']} {info['currency']}")
        print(f"  Leverage:     1:{info['leverage']}")
        print()

        # Get available symbols
        print("Fetching available symbols...")
        symbols = client.get_symbols()
        forex_pairs = [s for s in symbols if len(s) == 6 and s.isalpha()][:20]
        print(f"✓ Found {len(symbols)} total symbols")
        print(f"Forex pairs: {', '.join(forex_pairs[:10])}...")
        print()

        # Get current price for EURUSD
        print("Fetching current EURUSD price...")
        price = client.get_current_price("EURUSD")
        print(f"✓ EURUSD:")
        print(f"  Bid:    {price['bid']:.5f}")
        print(f"  Ask:    {price['ask']:.5f}")
        print(f"  Mid:    {price['mid']:.5f}")
        print(f"  Spread: {price['spread_pips']:.1f} pips")
        print(f"  Time:   {price['time']}")
        print()

        # Fetch sample historical data
        print("Fetching sample historical data (EURUSD, last 10 H1 candles)...")
        import MetaTrader5 as mt5
        df = client.get_bars("EURUSD", timeframe=mt5.TIMEFRAME_H1, count=10)
        print(f"✓ Fetched {len(df)} candles")
        print()
        print(df)
        print()

        # Disconnect
        client.disconnect()

        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print()
        print("You can now download historical data using:")
        print("  python scripts/download_data.py")
        print()

        return 0

    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        print()
        print("Please check that:")
        print("1. You have created .env file with your MT5 credentials")
        print("2. MT5 terminal is installed and running")
        print("3. Your Exness demo account is active")
        return 1

    except ConnectionError as e:
        print(f"✗ Connection error: {e}")
        print()
        print("Please check that:")
        print("1. MetaTrader 5 terminal is installed and running")
        print("2. Your login credentials are correct")
        print("3. The server name matches your account (Exness-MT5Trial9)")
        return 1

    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"✗ Test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
