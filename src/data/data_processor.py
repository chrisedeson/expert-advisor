"""
Data processing and transformation utilities
"""

from typing import Optional, List
import pandas as pd
import numpy as np
from loguru import logger

from ..utils.config import get_config


class DataProcessor:
    """
    Clean, transform, and prepare data for backtesting

    Features:
    - Handle missing data
    - Remove outliers
    - Resample timeframes
    - Add derived features
    - Apply realistic transaction costs
    """

    def __init__(self):
        """Initialize data processor"""
        self.config = get_config()

    def clean_data(
        self,
        df: pd.DataFrame,
        fill_method: str = "ffill",
        max_fill_limit: int = 5,
        remove_outliers: bool = True,
        z_score_threshold: float = 4.0
    ) -> pd.DataFrame:
        """
        Clean data by handling missing values and outliers

        Args:
            df: Input DataFrame
            fill_method: Method to fill missing values ('ffill', 'bfill', 'interpolate')
            max_fill_limit: Maximum number of consecutive fills
            remove_outliers: Whether to remove outliers
            z_score_threshold: Z-score threshold for outlier detection

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        initial_rows = len(df)

        # Check for missing values
        missing_before = df.isnull().sum().sum()
        if missing_before > 0:
            logger.warning(f"Found {missing_before} missing values")

            if fill_method == "ffill":
                df.fillna(method="ffill", limit=max_fill_limit, inplace=True)
            elif fill_method == "bfill":
                df.fillna(method="bfill", limit=max_fill_limit, inplace=True)
            elif fill_method == "interpolate":
                df.interpolate(method="linear", limit=max_fill_limit, inplace=True)

            # Drop any remaining NaN
            df.dropna(inplace=True)

            missing_after = df.isnull().sum().sum()
            logger.info(f"Filled/removed missing values: {missing_before} -> {missing_after}")

        # Remove outliers using z-score
        if remove_outliers:
            for col in ["open", "high", "low", "close"]:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > z_score_threshold
                outlier_count = outliers.sum()

                if outlier_count > 0:
                    logger.warning(f"Removing {outlier_count} outliers from {col}")
                    df = df[~outliers]

        final_rows = len(df)
        if final_rows < initial_rows:
            logger.info(f"Cleaned data: {initial_rows} -> {final_rows} rows")

        return df

    def add_spread(
        self,
        df: pd.DataFrame,
        pair: str,
        spread_pips: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Add bid/ask spread to mid prices

        Args:
            df: DataFrame with mid prices
            pair: Currency pair
            spread_pips: Spread in pips (uses config if not provided)

        Returns:
            DataFrame with bid/ask columns
        """
        df = df.copy()

        # Get spread from config if not provided
        if spread_pips is None:
            costs = self.config.get("transaction_costs", {})
            spread_pips = costs.get(pair, {}).get("spread", 1.0)

        # Convert pips to price
        # For JPY pairs, 1 pip = 0.01; for others, 1 pip = 0.0001
        pip_value = 0.01 if "JPY" in pair else 0.0001
        spread_price = spread_pips * pip_value

        # Calculate bid/ask from mid
        df["bid"] = df["close"] - (spread_price / 2)
        df["ask"] = df["close"] + (spread_price / 2)
        df["spread_pips"] = spread_pips

        logger.info(f"Added spread: {spread_pips} pips for {pair}")
        return df

    def resample_timeframe(
        self,
        df: pd.DataFrame,
        target_timeframe: str
    ) -> pd.DataFrame:
        """
        Resample data to different timeframe

        Args:
            df: Input DataFrame
            target_timeframe: Target timeframe ('5min', '15min', '1H', '4H', 'D')

        Returns:
            Resampled DataFrame
        """
        df = df.copy()

        # Map common timeframe names
        timeframe_map = {
            "M5": "5min",
            "M15": "15min",
            "M30": "30min",
            "H1": "1H",
            "H4": "4H",
            "D": "D",
        }
        target_timeframe = timeframe_map.get(target_timeframe, target_timeframe)

        # Resample OHLCV data
        resampled = df.resample(target_timeframe).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        })

        # Remove any rows with NaN (incomplete periods)
        resampled.dropna(inplace=True)

        logger.info(f"Resampled from {len(df)} to {len(resampled)} candles")
        return resampled

    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return columns"""
        df = df.copy()
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        return df

    def add_volatility(
        self,
        df: pd.DataFrame,
        window: int = 20,
        method: str = "std"
    ) -> pd.DataFrame:
        """
        Add volatility measure

        Args:
            df: Input DataFrame
            window: Rolling window size
            method: 'std' for standard deviation, 'atr' for ATR, 'parkinson' for Parkinson

        Returns:
            DataFrame with volatility column
        """
        df = df.copy()

        if method == "std":
            df["volatility"] = df["returns"].rolling(window).std()
        elif method == "atr":
            # True Range
            df["tr1"] = df["high"] - df["low"]
            df["tr2"] = abs(df["high"] - df["close"].shift())
            df["tr3"] = abs(df["low"] - df["close"].shift())
            df["tr"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
            df["volatility"] = df["tr"].rolling(window).mean()
            df.drop(["tr1", "tr2", "tr3", "tr"], axis=1, inplace=True)
        elif method == "parkinson":
            # Parkinson volatility (uses high-low range)
            df["volatility"] = np.sqrt(
                (1 / (4 * np.log(2))) *
                np.log(df["high"] / df["low"]).rolling(window).apply(lambda x: (x ** 2).sum())
            )

        return df

    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets (chronological)

        Args:
            df: Input DataFrame
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio

        Returns:
            Tuple of (train, validation, test) DataFrames
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Ratios must sum to 1.0"

        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]

        logger.info(f"Split data: Train={len(train)}, Val={len(val)}, Test={len(test)}")
        return train, val, test

    def align_multiple_pairs(
        self,
        data_dict: dict[str, pd.DataFrame],
        method: str = "inner"
    ) -> dict[str, pd.DataFrame]:
        """
        Align multiple currency pairs to common timestamps

        Args:
            data_dict: Dictionary mapping pair to DataFrame
            method: 'inner' (intersection) or 'outer' (union) join

        Returns:
            Dictionary with aligned DataFrames
        """
        if not data_dict:
            return {}

        # Find common timestamps
        all_timestamps = [df.index for df in data_dict.values()]

        if method == "inner":
            # Intersection of all timestamps
            common_timestamps = all_timestamps[0]
            for timestamps in all_timestamps[1:]:
                common_timestamps = common_timestamps.intersection(timestamps)
        else:
            # Union of all timestamps
            common_timestamps = all_timestamps[0]
            for timestamps in all_timestamps[1:]:
                common_timestamps = common_timestamps.union(timestamps)

        common_timestamps = common_timestamps.sort_values()

        # Align all DataFrames
        aligned = {}
        for pair, df in data_dict.items():
            aligned[pair] = df.reindex(common_timestamps, method="ffill")

        logger.info(f"Aligned {len(data_dict)} pairs to {len(common_timestamps)} common timestamps")
        return aligned

    def prepare_for_backtest(
        self,
        df: pd.DataFrame,
        pair: str,
        add_features: bool = True
    ) -> pd.DataFrame:
        """
        Complete preprocessing pipeline for backtesting

        Args:
            df: Raw OHLCV data
            pair: Currency pair
            add_features: Whether to add derived features

        Returns:
            Processed DataFrame ready for backtesting
        """
        logger.info(f"Preparing {pair} data for backtesting")

        # Clean data
        df = self.clean_data(df)

        # Add spread
        df = self.add_spread(df, pair)

        if add_features:
            # Add returns
            df = self.add_returns(df)

            # Add volatility (ATR)
            df = self.add_volatility(df, method="atr")

        # Remove any remaining NaN from indicators
        df.dropna(inplace=True)

        logger.info(f"Prepared {len(df)} candles for backtesting")
        return df
