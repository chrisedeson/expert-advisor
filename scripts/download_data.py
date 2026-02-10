#!/usr/bin/env python3
"""
Download historical forex data from OANDA
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_fetcher import DataFetcher
from src.data.data_storage import DataStorage
from src.utils.logger import logger
from src.utils.config import get_config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Download historical forex data from OANDA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download EUR/USD H1 data from 2023 to 2024
  python scripts/download_data.py --pair EUR_USD --start 2023-01-01 --end 2024-12-31

  # Download all configured pairs (from config.yaml)
  python scripts/download_data.py --all --granularity H4

  # Update existing data with latest candles
  python scripts/download_data.py --pair GBP_USD --update

  # Force refresh (re-download even if cached)
  python scripts/download_data.py --pair USD_JPY --force
        """
    )

    parser.add_argument(
        "--pair",
        type=str,
        help="Currency pair (e.g., EUR_USD, EURUSD)"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all pairs from config.yaml"
    )

    parser.add_argument(
        "--granularity",
        type=str,
        default="H1",
        choices=["M5", "M15", "M30", "H1", "H4", "D"],
        help="Timeframe (default: H1)"
    )

    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD). Defaults to config.yaml"
    )

    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD). Defaults to config.yaml"
    )

    parser.add_argument(
        "--update",
        action="store_true",
        help="Update existing data with latest candles"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force refresh (re-download even if cached)"
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify data quality after download"
    )

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    # Validate arguments
    if not args.pair and not args.all:
        print("Error: Must specify either --pair or --all")
        return 1

    if args.pair and args.all:
        print("Error: Cannot specify both --pair and --all")
        return 1

    # Parse dates
    start_date = datetime.strptime(args.start, "%Y-%m-%d") if args.start else None
    end_date = datetime.strptime(args.end, "%Y-%m-%d") if args.end else None

    # Initialize fetcher
    print("Initializing data fetcher...")
    fetcher = DataFetcher()
    print()

    try:
        if args.update:
            # Update mode
            if not args.pair:
                print("Error: --update requires --pair")
                return 1

            print(f"Updating {args.pair} {args.granularity}...")
            df = fetcher.update_data(args.pair, args.granularity)
            print(f"✓ Updated {len(df)} total candles")

        elif args.all:
            # Download all configured pairs
            print(f"Downloading all configured pairs ({args.granularity})...")
            results = fetcher.fetch_all_configured_pairs(
                granularity=args.granularity,
                force_refresh=args.force
            )

            print()
            print("=" * 60)
            print("Download Summary")
            print("=" * 60)
            for pair, df in results.items():
                if not df.empty:
                    print(f"✓ {pair}: {len(df)} candles")
                else:
                    print(f"✗ {pair}: Failed")

        else:
            # Download single pair
            print(f"Downloading {args.pair} {args.granularity}...")
            df = fetcher.fetch_pair(
                pair=args.pair,
                granularity=args.granularity,
                start_date=start_date,
                end_date=end_date,
                force_refresh=args.force
            )

            if df.empty:
                print("✗ No data downloaded")
                return 1

            print(f"✓ Downloaded {len(df)} candles")
            print()
            print("Data preview:")
            print(df.head())
            print()
            print(df.tail())

            # Verify data quality
            if args.verify:
                print()
                print("Verifying data quality...")
                quality = fetcher.verify_data_quality(df, args.pair)
                print(f"Status: {quality['status']}")
                print(f"Rows: {quality['rows']}")
                print(f"Date range: {quality['start_date']} to {quality['end_date']}")
                if quality['issues']:
                    print("Issues found:")
                    for issue in quality['issues']:
                        print(f"  - {issue}")
                else:
                    print("✓ No issues found")

        print()
        print("=" * 60)
        print("✓ Download complete!")
        print("=" * 60)

        # Show available data
        print()
        print("Available data:")
        storage = DataStorage()
        for info in storage.list_available_data():
            print(f"  {info['pair']:10s} {info['granularity']:4s}  {info['rows']:7d} candles  "
                  f"{info['start_date'].date()} to {info['end_date'].date()}  "
                  f"({info['size_mb']:.2f} MB)")

        return 0

    except Exception as e:
        logger.error(f"Download failed: {e}")
        print(f"✗ Download failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
