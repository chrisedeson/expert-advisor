"""
Data storage handler using Parquet format for fast I/O
"""

from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd
from loguru import logger

from ..utils.config import get_config


class DataStorage:
    """
    Handle storage and retrieval of historical market data

    Uses Parquet format for:
    - Fast read/write
    - Compression
    - Type preservation
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize data storage

        Args:
            base_dir: Base directory for data storage (defaults to config)
        """
        config = get_config()
        self.base_dir = base_dir or config.data_dir
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Data storage initialized: {self.base_dir}")

    def _get_file_path(self, pair: str, granularity: str, processed: bool = False) -> Path:
        """
        Get file path for a pair/granularity combination

        Args:
            pair: Currency pair
            granularity: Timeframe
            processed: Whether this is processed data

        Returns:
            Path to parquet file
        """
        dir_path = self.processed_dir if processed else self.raw_dir
        filename = f"{pair}_{granularity}.parquet"
        return dir_path / filename

    def save(
        self,
        df: pd.DataFrame,
        pair: str,
        granularity: str,
        processed: bool = False
    ) -> None:
        """
        Save DataFrame to Parquet file

        Args:
            df: DataFrame to save
            pair: Currency pair
            granularity: Timeframe
            processed: Whether this is processed data
        """
        if df.empty:
            logger.warning(f"Cannot save empty DataFrame for {pair} {granularity}")
            return

        file_path = self._get_file_path(pair, granularity, processed)

        try:
            df.to_parquet(
                file_path,
                engine="pyarrow",
                compression="snappy",
                index=True
            )
            logger.info(f"Saved {len(df)} rows to {file_path}")
        except Exception as e:
            logger.error(f"Error saving data to {file_path}: {e}")
            raise

    def load(
        self,
        pair: str,
        granularity: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        processed: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Load DataFrame from Parquet file

        Args:
            pair: Currency pair
            granularity: Timeframe
            start_date: Filter to this start date
            end_date: Filter to this end date
            processed: Whether to load processed data

        Returns:
            DataFrame or None if file doesn't exist
        """
        file_path = self._get_file_path(pair, granularity, processed)

        if not file_path.exists():
            logger.debug(f"File not found: {file_path}")
            return None

        try:
            df = pd.read_parquet(file_path, engine="pyarrow")

            # Filter by date range if provided
            if start_date is not None:
                df = df[df.index >= start_date]
            if end_date is not None:
                df = df[df.index <= end_date]

            logger.info(f"Loaded {len(df)} rows from {file_path}")
            return df

        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            return None

    def exists(self, pair: str, granularity: str, processed: bool = False) -> bool:
        """Check if data file exists"""
        file_path = self._get_file_path(pair, granularity, processed)
        return file_path.exists()

    def delete(self, pair: str, granularity: str, processed: bool = False) -> bool:
        """
        Delete data file

        Args:
            pair: Currency pair
            granularity: Timeframe
            processed: Whether to delete processed data

        Returns:
            True if deleted, False if file didn't exist
        """
        file_path = self._get_file_path(pair, granularity, processed)

        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted {file_path}")
            return True

        return False

    def list_available_data(self) -> list[dict]:
        """
        List all available data files

        Returns:
            List of dictionaries with file information
        """
        available = []

        for file_path in self.raw_dir.glob("*.parquet"):
            # Parse filename
            parts = file_path.stem.split("_")
            if len(parts) >= 2:
                pair = "_".join(parts[:-1])
                granularity = parts[-1]

                # Get file info
                stats = file_path.stat()
                size_mb = stats.st_size / (1024 * 1024)

                # Try to load and get date range
                try:
                    df = pd.read_parquet(file_path)
                    start_date = df.index.min()
                    end_date = df.index.max()
                    rows = len(df)
                except:
                    start_date = None
                    end_date = None
                    rows = 0

                available.append({
                    "pair": pair,
                    "granularity": granularity,
                    "file": str(file_path),
                    "size_mb": round(size_mb, 2),
                    "rows": rows,
                    "start_date": start_date,
                    "end_date": end_date,
                })

        return sorted(available, key=lambda x: (x["pair"], x["granularity"]))

    def get_data_info(self, pair: str, granularity: str) -> Optional[dict]:
        """
        Get information about a specific data file

        Args:
            pair: Currency pair
            granularity: Timeframe

        Returns:
            Dictionary with file information or None if not found
        """
        file_path = self._get_file_path(pair, granularity)

        if not file_path.exists():
            return None

        stats = file_path.stat()
        df = self.load(pair, granularity)

        if df is None or df.empty:
            return None

        return {
            "pair": pair,
            "granularity": granularity,
            "file": str(file_path),
            "size_mb": round(stats.st_size / (1024 * 1024), 2),
            "rows": len(df),
            "start_date": df.index.min(),
            "end_date": df.index.max(),
            "columns": list(df.columns),
        }
