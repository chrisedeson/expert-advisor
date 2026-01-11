"""
MetaTrader 5 (MT5) Client for Exness
Wrapper for MT5 Python API with retry logic and data fetching
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import MetaTrader5 as mt5
from loguru import logger

from ..utils.config import get_config


class MT5Client:
    """
    MT5 client for fetching market data and executing trades

    Features:
    - Automatic connection management
    - Retry logic with exponential backoff
    - Data fetching with chunking for large requests
    - Order execution
    - Position management
    """

    def __init__(
        self,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None
    ):
        """
        Initialize MT5 client

        Args:
            login: MT5 account login (from config if not provided)
            password: MT5 account password (from config if not provided)
            server: MT5 server (from config if not provided)
        """
        config = get_config()

        self.login = login or int(config.get_env("MT5_LOGIN"))
        self.password = password or config.get_env("MT5_PASSWORD")
        self.server = server or config.get_env("MT5_SERVER")

        self.connected = False
        self._connect()

    def _connect(self):
        """Establish connection to MT5 terminal"""
        # Initialize MT5
        if not mt5.initialize():
            error = mt5.last_error()
            raise ConnectionError(f"MT5 initialization failed: {error}")

        logger.info("MT5 initialized successfully")

        # Login to account
        if not mt5.login(self.login, password=self.password, server=self.server):
            error = mt5.last_error()
            mt5.shutdown()
            raise ConnectionError(f"MT5 login failed: {error}")

        self.connected = True
        logger.info(f"Connected to MT5: {self.server}, Account: {self.login}")

    def _ensure_connected(self):
        """Ensure MT5 connection is active"""
        if not self.connected or not mt5.terminal_info():
            logger.warning("MT5 connection lost, reconnecting...")
            self._connect()

    def disconnect(self):
        """Disconnect from MT5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5")

    def get_account_info(self) -> Dict:
        """Get account information"""
        self._ensure_connected()

        account_info = mt5.account_info()
        if account_info is None:
            error = mt5.last_error()
            raise RuntimeError(f"Failed to get account info: {error}")

        return {
            "login": account_info.login,
            "server": account_info.server,
            "balance": account_info.balance,
            "equity": account_info.equity,
            "margin": account_info.margin,
            "margin_free": account_info.margin_free,
            "margin_level": account_info.margin_level,
            "profit": account_info.profit,
            "currency": account_info.currency,
            "leverage": account_info.leverage,
        }

    def get_symbols(self) -> List[str]:
        """Get list of available symbols"""
        self._ensure_connected()

        symbols = mt5.symbols_get()
        if symbols is None:
            error = mt5.last_error()
            raise RuntimeError(f"Failed to get symbols: {error}")

        return [s.name for s in symbols]

    def get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol information"""
        self._ensure_connected()

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            error = mt5.last_error()
            raise RuntimeError(f"Failed to get symbol info for {symbol}: {error}")

        return {
            "name": symbol_info.name,
            "bid": symbol_info.bid,
            "ask": symbol_info.ask,
            "spread": symbol_info.spread,
            "digits": symbol_info.digits,
            "point": symbol_info.point,
            "trade_contract_size": symbol_info.trade_contract_size,
            "volume_min": symbol_info.volume_min,
            "volume_max": symbol_info.volume_max,
            "volume_step": symbol_info.volume_step,
        }

    def get_current_price(self, symbol: str) -> Dict[str, float]:
        """
        Get current bid/ask price

        Args:
            symbol: Currency pair (e.g., 'EURUSD')

        Returns:
            Dictionary with 'bid', 'ask', 'mid', 'spread_pips', and 'time'
        """
        self._ensure_connected()

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            error = mt5.last_error()
            raise RuntimeError(f"Failed to get tick for {symbol}: {error}")

        # Calculate spread in pips
        point = mt5.symbol_info(symbol).point
        spread_pips = (tick.ask - tick.bid) / point

        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "mid": (tick.bid + tick.ask) / 2,
            "spread_pips": spread_pips,
            "time": datetime.fromtimestamp(tick.time),
        }

    def get_bars(
        self,
        symbol: str,
        timeframe: int = mt5.TIMEFRAME_H1,
        count: int = 1000,
        start_pos: int = 0
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars

        Args:
            symbol: Currency pair
            timeframe: MT5 timeframe constant (e.g., mt5.TIMEFRAME_H1)
            count: Number of bars (max 99999)
            start_pos: Start position (0 = most recent)

        Returns:
            DataFrame with OHLCV data
        """
        self._ensure_connected()

        # Make sure symbol is available
        if not mt5.symbol_select(symbol, True):
            error = mt5.last_error()
            raise RuntimeError(f"Failed to select symbol {symbol}: {error}")

        # Fetch rates
        rates = mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)

        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            logger.warning(f"No data returned for {symbol}: {error}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)

        # Rename columns to standard format
        df.rename(columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "tick_volume": "volume",
            "real_volume": "real_volume"
        }, inplace=True)

        # Keep only OHLCV
        df = df[["open", "high", "low", "close", "volume"]]

        logger.info(f"Fetched {len(df)} bars for {symbol}")
        return df

    def get_bars_range(
        self,
        symbol: str,
        timeframe: int,
        date_from: datetime,
        date_to: datetime
    ) -> pd.DataFrame:
        """
        Fetch bars within date range

        Args:
            symbol: Currency pair
            timeframe: MT5 timeframe constant
            date_from: Start date (UTC)
            date_to: End date (UTC)

        Returns:
            DataFrame with OHLCV data
        """
        self._ensure_connected()

        # Make sure symbol is available
        if not mt5.symbol_select(symbol, True):
            error = mt5.last_error()
            raise RuntimeError(f"Failed to select symbol {symbol}: {error}")

        # Fetch rates
        rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)

        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            logger.warning(f"No data returned for {symbol}: {error}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)

        # Keep only OHLCV
        df = df[["open", "high", "low", "close", "tick_volume"]]
        df.rename(columns={"tick_volume": "volume"}, inplace=True)

        logger.info(f"Fetched {len(df)} bars for {symbol} from {date_from} to {date_to}")
        return df

    def get_historical_data(
        self,
        symbol: str,
        timeframe: int,
        start_date: datetime,
        end_date: datetime,
        chunk_size: int = 50000
    ) -> pd.DataFrame:
        """
        Fetch large amounts of historical data by chunking

        MT5 can handle up to ~99999 bars per request, but we chunk for reliability

        Args:
            symbol: Currency pair
            timeframe: MT5 timeframe constant
            start_date: Start date (UTC)
            end_date: End date (UTC)
            chunk_size: Bars per chunk (default: 50000)

        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching historical data: {symbol} from {start_date} to {end_date}")

        all_data = []
        current_start = start_date

        # Determine chunk duration based on timeframe
        timeframe_minutes = {
            mt5.TIMEFRAME_M1: 1,
            mt5.TIMEFRAME_M5: 5,
            mt5.TIMEFRAME_M15: 15,
            mt5.TIMEFRAME_M30: 30,
            mt5.TIMEFRAME_H1: 60,
            mt5.TIMEFRAME_H4: 240,
            mt5.TIMEFRAME_D1: 1440,
        }
        minutes = timeframe_minutes.get(timeframe, 60)
        chunk_duration = timedelta(minutes=minutes * chunk_size)

        while current_start < end_date:
            current_end = min(current_start + chunk_duration, end_date)

            try:
                df_chunk = self.get_bars_range(
                    symbol=symbol,
                    timeframe=timeframe,
                    date_from=current_start,
                    date_to=current_end
                )

                if not df_chunk.empty:
                    all_data.append(df_chunk)
                    logger.debug(f"Fetched chunk: {current_start} to {current_end} ({len(df_chunk)} bars)")

                current_start = current_end
                time.sleep(0.1)  # Small delay to avoid rate limits

            except Exception as e:
                logger.error(f"Error fetching chunk {current_start} to {current_end}: {e}")
                raise

        if not all_data:
            logger.error(f"No data fetched for {symbol}")
            return pd.DataFrame()

        # Combine all chunks
        df = pd.concat(all_data)
        df = df[~df.index.duplicated(keep="first")]
        df.sort_index(inplace=True)

        logger.info(f"Total bars fetched: {len(df)}")
        return df

    def test_connection(self) -> bool:
        """Test MT5 connection"""
        try:
            self._ensure_connected()
            info = self.get_account_info()
            logger.info(f"Connection successful. Account: {info['login']}")
            logger.info(f"Balance: {info['balance']} {info['currency']}")
            logger.info(f"Equity: {info['equity']} {info['currency']}")
            logger.info(f"Leverage: 1:{info['leverage']}")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    @staticmethod
    def timeframe_from_string(timeframe_str: str) -> int:
        """
        Convert string timeframe to MT5 constant

        Args:
            timeframe_str: Timeframe string (e.g., 'M5', 'H1', 'D')

        Returns:
            MT5 timeframe constant
        """
        mapping = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D": mt5.TIMEFRAME_D1,
            "D1": mt5.TIMEFRAME_D1,
        }
        return mapping.get(timeframe_str.upper(), mt5.TIMEFRAME_H1)

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()

    def __repr__(self) -> str:
        return f"MT5Client(server={self.server}, account={self.login}, connected={self.connected})"
