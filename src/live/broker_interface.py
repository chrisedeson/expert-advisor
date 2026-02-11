"""Abstract broker interface for order execution."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime
import pandas as pd


@dataclass
class OrderResult:
    """Result of an order execution."""
    success: bool
    order_id: Optional[str] = None
    fill_price: Optional[float] = None
    error: Optional[str] = None


@dataclass
class AccountInfo:
    """Broker account information."""
    balance: float
    equity: float
    margin_used: float
    margin_free: float
    currency: str = "USD"


class BrokerInterface(ABC):
    """Abstract broker interface. Implement for each broker."""

    @abstractmethod
    def connect(self) -> bool:
        """Connect to broker. Returns True if successful."""
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect from broker."""
        pass

    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """Get current account balance and equity."""
        pass

    @abstractmethod
    def place_market_order(
        self,
        symbol: str,
        direction: str,  # 'BUY' or 'SELL'
        lot_size: float,
        stop_loss: float,
        take_profit: float,
        comment: str = "",
    ) -> OrderResult:
        """Place a market order with SL and TP."""
        pass

    @abstractmethod
    def close_position(self, order_id: str) -> OrderResult:
        """Close an open position by order ID."""
        pass

    @abstractmethod
    def modify_position(
        self,
        order_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> OrderResult:
        """Modify SL/TP of an open position."""
        pass

    @abstractmethod
    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get all open positions, optionally filtered by symbol."""
        pass

    @abstractmethod
    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        count: int = 250,
    ) -> Optional[pd.DataFrame]:
        """Get recent OHLCV candles for a symbol.

        Returns DataFrame with columns: open, high, low, close, volume
        Index should be DatetimeIndex in UTC.
        """
        pass

    @abstractmethod
    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """Get current bid/ask price. Returns {'bid': float, 'ask': float}."""
        pass
