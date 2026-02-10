"""
Data fetcher for downloading and managing historical forex data
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional
import pandas as pd
from loguru import logger

from .mt5_client import MT5Client
from .data_storage import DataStorage
from ..utils.config import get_config
from ..utils.helpers import validate_pair


class DataFetcher:
    """
    High-level interface for fetching and managing historical data

    Features:
    - Automatic caching to avoid re-downloading
    - Multi-pair, multi-timeframe support
    - Data validation and quality checks
    - Incremental updates
    """

    def __init__(self, client: Optional[MT5Client] = None, storage: Optional[DataStorage] = None):
        """
        Initialize data fetcher

        Args:
            client: MT5 client (creates new one if not provided)
            storage: Data storage handler (creates new one if not provided)
        """
        self.client = client or MT5Client()
        self.storage = storage or DataStorage()
        self.config = get_config()

    def fetch_pair(
        self,
        pair: str,
        granularity: str = "H1",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch historical data for a currency pair

        Args:
            pair: Currency pair (e.g., 'EUR_USD', 'EURUSD')
            granularity: Timeframe (H1, H4, D, etc.)
            start_date: Start date (defaults to config)
            end_date: End date (defaults to config)
            force_refresh: Force re-download even if cached

        Returns:
            DataFrame with OHLCV data
        """
        # Normalize pair format
        pair = validate_pair(pair)

        # Use config defaults if not provided
        if start_date is None:
            start_date = pd.to_datetime(self.config.get("data.start_date", "2019-01-01"))
        if end_date is None:
            end_date = pd.to_datetime(self.config.get("data.end_date", "2025-12-31"))

        logger.info(f"Fetching {pair} {granularity} from {start_date.date()} to {end_date.date()}")

        # Check if data exists in cache
        if not force_refresh:
            cached_data = self.storage.load(pair, granularity, start_date, end_date)
            if cached_data is not None and not cached_data.empty:
                logger.info(f"Using cached data ({len(cached_data)} candles)")
                return cached_data

        # Fetch from MT5
        try:
            timeframe = MT5Client.timeframe_from_string(granularity)
            df = self.client.get_historical_data(
                symbol=pair,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )

            if df.empty:
                logger.warning(f"No data returned for {pair} {granularity}")
                return df

            # Save to cache
            self.storage.save(df, pair, granularity)
            logger.info(f"Saved {len(df)} candles to cache")

            return df

        except Exception as e:
            logger.error(f"Error fetching data for {pair}: {e}")
            raise

    def fetch_multiple_pairs(
        self,
        pairs: List[str],
        granularity: str = "H1",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        force_refresh: bool = False
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple currency pairs

        Args:
            pairs: List of currency pairs
            granularity: Timeframe
            start_date: Start date
            end_date: End date
            force_refresh: Force re-download

        Returns:
            Dictionary mapping pair to DataFrame
        """
        results = {}

        for pair in pairs:
            try:
                df = self.fetch_pair(
                    pair=pair,
                    granularity=granularity,
                    start_date=start_date,
                    end_date=end_date,
                    force_refresh=force_refresh
                )
                results[pair] = df
            except Exception as e:
                logger.error(f"Failed to fetch {pair}: {e}")
                results[pair] = pd.DataFrame()

        return results

    def fetch_all_configured_pairs(
        self,
        granularity: str = "H1",
        force_refresh: bool = False
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch data for all pairs configured in config.yaml

        Args:
            granularity: Timeframe
            force_refresh: Force re-download

        Returns:
            Dictionary mapping pair to DataFrame
        """
        pairs = self.config.currency_pairs
        logger.info(f"Fetching all configured pairs: {pairs}")

        return self.fetch_multiple_pairs(
            pairs=pairs,
            granularity=granularity,
            force_refresh=force_refresh
        )

    def update_data(
        self,
        pair: str,
        granularity: str = "H1"
    ) -> pd.DataFrame:
        """
        Update cached data with latest candles

        Fetches data from the last cached timestamp to now

        Args:
            pair: Currency pair
            granularity: Timeframe

        Returns:
            Updated DataFrame
        """
        pair = validate_pair(pair)

        # Load existing data
        existing_data = self.storage.load(pair, granularity)

        if existing_data is None or existing_data.empty:
            logger.info("No existing data, fetching from scratch")
            return self.fetch_pair(pair, granularity, force_refresh=True)

        # Get last timestamp
        last_timestamp = existing_data.index[-1]
        logger.info(f"Last cached data: {last_timestamp}")

        # Fetch new data from last timestamp to now
        timeframe = MT5Client.timeframe_from_string(granularity)
        new_data = self.client.get_historical_data(
            symbol=pair,
            timeframe=timeframe,
            start_date=last_timestamp,
            end_date=datetime.utcnow()
        )

        if new_data.empty:
            logger.info("No new data available")
            return existing_data

        # Combine and remove duplicates
        combined_data = pd.concat([existing_data, new_data])
        combined_data = combined_data[~combined_data.index.duplicated(keep="last")]
        combined_data.sort_index(inplace=True)

        # Save updated data
        self.storage.save(combined_data, pair, granularity)
        logger.info(f"Added {len(new_data)} new candles (total: {len(combined_data)})")

        return combined_data

    def get_available_pairs(self) -> List[str]:
        """Get list of available symbols from MT5"""
        return self.client.get_symbols()

    def verify_data_quality(self, df: pd.DataFrame, pair: str) -> dict:
        """
        Check data quality and return diagnostics

        Args:
            df: DataFrame to check
            pair: Currency pair name

        Returns:
            Dictionary with quality metrics
        """
        if df.empty:
            return {"status": "empty", "issues": ["No data"]}

        issues = []

        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            issues.append(f"Missing values: {missing[missing > 0].to_dict()}")

        # Check for gaps in time series
        time_diff = df.index.to_series().diff()
        expected_freq = pd.infer_freq(df.index)
        if expected_freq:
            gaps = time_diff[time_diff > pd.Timedelta(expected_freq) * 2]
            if len(gaps) > 0:
                issues.append(f"Found {len(gaps)} time gaps")

        # Check for outliers in price data
        for col in ["open", "high", "low", "close"]:
            z_scores = (df[col] - df[col].mean()) / df[col].std()
            outliers = (z_scores.abs() > 4).sum()
            if outliers > 0:
                issues.append(f"{col}: {outliers} outliers (Z-score > 4)")

        # Check OHLC validity
        invalid_ohlc = (
            (df["high"] < df["low"]) |
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["low"] > df["open"]) |
            (df["low"] > df["close"])
        ).sum()
        if invalid_ohlc > 0:
            issues.append(f"{invalid_ohlc} invalid OHLC candles")

        return {
            "status": "ok" if not issues else "issues_found",
            "rows": len(df),
            "start_date": df.index[0],
            "end_date": df.index[-1],
            "issues": issues
        }
