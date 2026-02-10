"""
OANDA API Client
Wrapper for OANDA v20 REST API with retry logic and rate limiting
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
from v20 import V20Error
from v20.context import Context
from loguru import logger

from ..utils.config import get_config


class OandaClient:
    """
    OANDA API client for fetching market data and executing trades

    Features:
    - Automatic retry logic with exponential backoff
    - Rate limiting protection
    - Connection pooling
    - Error handling
    """

    def __init__(self, api_key: Optional[str] = None, account_id: Optional[str] = None, environment: Optional[str] = None):
        """
        Initialize OANDA client

        Args:
            api_key: OANDA API key (from config if not provided)
            account_id: OANDA account ID (from config if not provided)
            environment: 'practice' or 'live' (from config if not provided)
        """
        config = get_config()

        self.api_key = api_key or config.oanda_api_key
        self.account_id = account_id or config.oanda_account_id
        self.environment = environment or config.oanda_environment

        # Initialize V20 context
        if self.environment == "practice":
            self.hostname = "api-fxpractice.oanda.com"
        else:
            self.hostname = "api-fxtrade.oanda.com"

        self.ctx = Context(
            hostname=self.hostname,
            token=self.api_key,
            poll_timeout=10
        )

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 10 requests per second max

        logger.info(f"OANDA client initialized: {self.environment} environment")

    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _retry_request(self, func, max_retries: int = 3, backoff_factor: float = 2.0):
        """
        Retry API request with exponential backoff

        Args:
            func: Function to retry
            max_retries: Maximum number of retries
            backoff_factor: Backoff multiplier

        Returns:
            Function result

        Raises:
            V20Error: If all retries fail
        """
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                return func()
            except V20Error as e:
                if attempt == max_retries - 1:
                    logger.error(f"Request failed after {max_retries} attempts: {e}")
                    raise

                wait_time = backoff_factor ** attempt
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

    def get_account_summary(self) -> Dict:
        """Get account summary"""
        def request():
            response = self.ctx.account.summary(self.account_id)
            return response.body

        return self._retry_request(request)

    def get_instruments(self) -> List[str]:
        """Get list of available instruments"""
        def request():
            response = self.ctx.account.instruments(self.account_id)
            return [inst.name for inst in response.body["instruments"]]

        return self._retry_request(request)

    def get_candles(
        self,
        instrument: str,
        granularity: str = "H1",
        count: Optional[int] = None,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
        price: str = "M"  # M = mid, B = bid, A = ask, BA = bid+ask
    ) -> pd.DataFrame:
        """
        Fetch historical candle data

        Args:
            instrument: Currency pair (e.g., 'EUR_USD')
            granularity: Timeframe (S5, S10, S15, S30, M1, M2, M4, M5, M10, M15, M30, H1, H2, H3, H4, H6, H8, H12, D, W, M)
            count: Number of candles (max 5000)
            from_time: Start time (UTC)
            to_time: End time (UTC)
            price: Price type ('M' for mid, 'B' for bid, 'A' for ask)

        Returns:
            DataFrame with OHLCV data
        """
        def request():
            params = {
                "granularity": granularity,
                "price": price
            }

            if count is not None:
                params["count"] = min(count, 5000)  # OANDA max
            if from_time is not None:
                params["fromTime"] = from_time.isoformat() + "Z"
            if to_time is not None:
                params["toTime"] = to_time.isoformat() + "Z"

            response = self.ctx.instrument.candles(instrument, **params)
            return response.body

        body = self._retry_request(request)
        candles = body["candles"]

        if not candles:
            logger.warning(f"No candles returned for {instrument} {granularity}")
            return pd.DataFrame()

        # Parse candles into DataFrame
        data = []
        for candle in candles:
            if not candle.complete:
                continue  # Skip incomplete candles

            row = {
                "time": pd.to_datetime(candle.time),
                "open": float(candle.mid.o if price == "M" else candle.bid.o),
                "high": float(candle.mid.h if price == "M" else candle.bid.h),
                "low": float(candle.mid.l if price == "M" else candle.bid.l),
                "close": float(candle.mid.c if price == "M" else candle.bid.c),
                "volume": int(candle.volume),
            }
            data.append(row)

        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index("time", inplace=True)
            df.sort_index(inplace=True)

        logger.info(f"Fetched {len(df)} candles for {instrument} ({granularity})")
        return df

    def get_historical_data(
        self,
        instrument: str,
        granularity: str,
        start_date: datetime,
        end_date: datetime,
        price: str = "M"
    ) -> pd.DataFrame:
        """
        Fetch large amounts of historical data by chunking requests

        OANDA limits to 5000 candles per request, so we chunk the date range

        Args:
            instrument: Currency pair (e.g., 'EUR_USD')
            granularity: Timeframe
            start_date: Start date (UTC)
            end_date: End date (UTC)
            price: Price type

        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching historical data: {instrument} {granularity} from {start_date} to {end_date}")

        # Determine chunk size based on granularity
        chunk_sizes = {
            "S5": timedelta(hours=7),    # 5000 * 5s = 6.94 hours
            "S10": timedelta(hours=14),
            "S15": timedelta(hours=21),
            "S30": timedelta(days=1, hours=17),
            "M1": timedelta(days=3, hours=11),
            "M5": timedelta(days=17),
            "M15": timedelta(days=52),
            "M30": timedelta(days=104),
            "H1": timedelta(days=208),
            "H4": timedelta(days=833),   # ~2.3 years
            "D": timedelta(days=5000),   # ~13.7 years
        }

        chunk_size = chunk_sizes.get(granularity, timedelta(days=100))

        all_data = []
        current_start = start_date

        while current_start < end_date:
            current_end = min(current_start + chunk_size, end_date)

            try:
                df_chunk = self.get_candles(
                    instrument=instrument,
                    granularity=granularity,
                    from_time=current_start,
                    to_time=current_end,
                    price=price
                )

                if not df_chunk.empty:
                    all_data.append(df_chunk)
                    logger.debug(f"Fetched chunk: {current_start} to {current_end} ({len(df_chunk)} candles)")
                else:
                    logger.warning(f"Empty chunk: {current_start} to {current_end}")

                current_start = current_end

                # Small delay between chunks
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error fetching chunk {current_start} to {current_end}: {e}")
                raise

        if not all_data:
            logger.error(f"No data fetched for {instrument}")
            return pd.DataFrame()

        # Combine all chunks
        df = pd.concat(all_data)
        df = df[~df.index.duplicated(keep="first")]  # Remove any duplicates
        df.sort_index(inplace=True)

        logger.info(f"Total candles fetched: {len(df)}")
        return df

    def get_current_price(self, instrument: str) -> Dict[str, float]:
        """
        Get current bid/ask price

        Args:
            instrument: Currency pair

        Returns:
            Dictionary with 'bid', 'ask', and 'mid' prices
        """
        def request():
            response = self.ctx.pricing.get(self.account_id, instruments=instrument)
            price = response.body["prices"][0]
            return {
                "bid": float(price.bids[0].price),
                "ask": float(price.asks[0].price),
                "mid": (float(price.bids[0].price) + float(price.asks[0].price)) / 2,
                "spread_pips": (float(price.asks[0].price) - float(price.bids[0].price)) * 10000,
                "time": pd.to_datetime(price.time)
            }

        return self._retry_request(request)

    def test_connection(self) -> bool:
        """Test OANDA API connection"""
        try:
            summary = self.get_account_summary()
            logger.info(f"Connection successful. Account ID: {summary['account'].id}")
            logger.info(f"Balance: {summary['account'].balance} {summary['account'].currency}")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def __repr__(self) -> str:
        return f"OandaClient(environment={self.environment}, account={self.account_id[:8]}...)"
