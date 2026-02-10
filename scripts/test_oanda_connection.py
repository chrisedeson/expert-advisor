#!/usr/bin/env python3
"""
Test OANDA API connection
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.oanda_client import OandaClient
from src.utils.logger import logger


def main():
    """Test OANDA connection"""
    print("=" * 60)
    print("OANDA Connection Test")
    print("=" * 60)
    print()

    try:
        # Initialize client
        print("Initializing OANDA client...")
        client = OandaClient()
        print(f"✓ Client initialized: {client}")
        print()

        # Test connection
        print("Testing connection...")
        if not client.test_connection():
            print("✗ Connection test failed")
            return 1
        print()

        # Get available instruments
        print("Fetching available instruments...")
        instruments = client.get_instruments()
        major_pairs = [inst for inst in instruments if "USD" in inst]
        print(f"✓ Found {len(instruments)} instruments")
        print(f"Major pairs: {', '.join(major_pairs[:10])}...")
        print()

        # Get current price for EUR/USD
        print("Fetching current EUR/USD price...")
        price = client.get_current_price("EUR_USD")
        print(f"✓ EUR/USD:")
        print(f"  Bid:    {price['bid']:.5f}")
        print(f"  Ask:    {price['ask']:.5f}")
        print(f"  Mid:    {price['mid']:.5f}")
        print(f"  Spread: {price['spread_pips']:.1f} pips")
        print(f"  Time:   {price['time']}")
        print()

        # Fetch small sample of historical data
        print("Fetching sample historical data (EUR/USD, last 10 candles)...")
        df = client.get_candles("EUR_USD", granularity="H1", count=10)
        print(f"✓ Fetched {len(df)} candles")
        print()
        print(df.head())
        print()

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
        print("1. You have copied .env.example to .env")
        print("2. You have added your OANDA API credentials to .env")
        print("3. Your OANDA account is active")
        return 1

    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"✗ Test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
