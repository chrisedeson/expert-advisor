"""
Base Strategy Class

Abstract base class for all trading strategies.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Tuple, Optional
from loguru import logger


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    All strategies must implement:
    - generate_signals(): Create trading signals
    - get_parameters(): Return current parameters
    - get_parameter_space(): Return optimization bounds
    """

    def __init__(self, name: str, **kwargs):
        """
        Initialize strategy.

        Args:
            name: Strategy name
            **kwargs: Strategy-specific parameters
        """
        self.name = name
        self.parameters = kwargs

        logger.info(f"Initialized {self.name} strategy with params: {kwargs}")

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from data.

        Args:
            data: OHLCV DataFrame with DatetimeIndex

        Returns:
            Series with signals {-1, 0, 1} for short/flat/long
        """
        pass

    @abstractmethod
    def get_parameter_space(self) -> Dict:
        """
        Get parameter space for optimization.

        Returns:
            Dictionary with parameter names and bounds:
            {
                'param_name': (min_value, max_value, type),
                ...
            }
        """
        pass

    def get_parameters(self) -> Dict:
        """
        Get current strategy parameters.

        Returns:
            Dictionary of parameter names and values
        """
        return self.parameters.copy()

    def set_parameters(self, **kwargs) -> None:
        """
        Update strategy parameters.

        Args:
            **kwargs: Parameters to update
        """
        self.parameters.update(kwargs)
        logger.debug(f"Updated {self.name} parameters: {kwargs}")

    def validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data.

        Args:
            data: OHLCV DataFrame

        Raises:
            ValueError: If data is invalid
        """
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")

        if len(data) == 0:
            raise ValueError("Data is empty")

    def __repr__(self) -> str:
        params_str = ', '.join(f"{k}={v}" for k, v in self.parameters.items())
        return f"{self.name}({params_str})"
