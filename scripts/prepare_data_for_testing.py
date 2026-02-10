#!/usr/bin/env python3
"""
Prepare Historical Data for Testing

Takes raw data files and prepares them for backtesting:
1. Loads data from raw directory
2. Ensures datetime index
3. Calculates ATR if missing
4. Validates data quality
5. Saves to processed directory

Run this before test_protected_system.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high = df['high']
    low = df['low']
    close = df['close']

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Average True Range
    atr = tr.rolling(window=period).mean()

    return atr


def process_file(raw_path: Path, processed_path: Path):
    """Process a single data file"""
    logger.info(f"Processing {raw_path.name}...")

    # Load data
    df = pd.read_parquet(raw_path)

    logger.info(f"  Loaded {len(df):,} rows")

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
        else:
            logger.error(f"  ❌ No 'time' column found")
            return False

    # Check required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required if col not in df.columns]
    if missing:
        logger.error(f"  ❌ Missing columns: {missing}")
        return False

    # Sort by time
    df.sort_index(inplace=True)

    # Calculate ATR if missing
    if 'atr' not in df.columns:
        logger.info(f"  Calculating ATR...")
        df['atr'] = calculate_atr(df)

    # Drop NaN rows (from ATR calculation)
    before = len(df)
    df.dropna(inplace=True)
    after = len(df)
    if before != after:
        logger.info(f"  Dropped {before - after} NaN rows")

    # Validate data
    if len(df) == 0:
        logger.error(f"  ❌ No data after processing")
        return False

    if df['atr'].isna().sum() > 0:
        logger.warning(f"  ⚠️  {df['atr'].isna().sum()} NaN ATR values remain")

    # Check date range
    start_date = df.index[0]
    end_date = df.index[-1]
    logger.info(f"  Date range: {start_date} to {end_date}")

    # Check for COVID period (March 2020)
    covid_period = df[(df.index >= '2020-03-01') & (df.index <= '2020-03-31')]
    if len(covid_period) > 0:
        logger.success(f"  ✅ Contains COVID period (March 2020): {len(covid_period)} bars")
    else:
        logger.warning(f"  ⚠️  Missing COVID period (March 2020)")

    # Save to processed directory
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_path)
    logger.success(f"  ✅ Saved to {processed_path}")
    logger.info(f"  Final rows: {len(df):,}, Columns: {list(df.columns)}")

    return True


def main():
    """Process all raw data files"""
    logger.info("=" * 80)
    logger.info("DATA PREPARATION FOR TESTING")
    logger.info("=" * 80)

    # Paths
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / 'data' / 'raw'
    processed_dir = project_root / 'data' / 'processed'

    if not raw_dir.exists():
        logger.error(f"Raw data directory not found: {raw_dir}")
        logger.error("Please run data export script first")
        return 1

    # Find all parquet files
    raw_files = list(raw_dir.glob('*.parquet'))

    if len(raw_files) == 0:
        logger.error("No parquet files found in raw directory")
        return 1

    logger.info(f"Found {len(raw_files)} raw data files")
    logger.info("")

    # Process each file
    success_count = 0
    for raw_path in raw_files:
        # Convert filename (EURUSDm_H1.parquet → EUR_USD_H1.parquet)
        filename = raw_path.stem.replace('m_', '_').replace('m', '_')
        processed_path = processed_dir / f'{filename}.parquet'

        try:
            if process_file(raw_path, processed_path):
                success_count += 1
            logger.info("")
        except Exception as e:
            logger.error(f"  ❌ Error processing {raw_path.name}: {e}")
            logger.info("")

    # Summary
    logger.info("=" * 80)
    logger.success(f"PROCESSING COMPLETE: {success_count}/{len(raw_files)} files successful")
    logger.info("=" * 80)

    if success_count > 0:
        logger.info("\n✅ Data is ready for testing!")
        logger.info("   Next step: python scripts/test_protected_system.py")
    else:
        logger.error("\n❌ No files processed successfully")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
