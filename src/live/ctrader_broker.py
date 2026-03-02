"""cTrader Open API broker implementation.

Connects to IC Markets cTrader via the Spotware Open API.
Implements BrokerInterface for the live trading engine.

Uses Twisted reactor internally but exposes synchronous methods
to match the BrokerInterface contract.
"""
import os
import time
import logging
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict
from pathlib import Path

import numpy as np
import pandas as pd

from ctrader_open_api import Client, Protobuf, TcpProtocol, Auth, EndPoints
from ctrader_open_api.messages.OpenApiMessages_pb2 import (
    ProtoOAApplicationAuthReq,
    ProtoOAAccountAuthReq,
    ProtoOASymbolsListReq,
    ProtoOASymbolByIdReq,
    ProtoOANewOrderReq,
    ProtoOAClosePositionReq,
    ProtoOAAmendPositionSLTPReq,
    ProtoOAReconcileReq,
    ProtoOATraderReq,
    ProtoOAGetTrendbarsReq,
    ProtoOASubscribeSpotsReq,
    ProtoOAUnsubscribeSpotsReq,
    ProtoOAGetAccountListByAccessTokenReq,
)
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import (
    ProtoOAOrderType,
    ProtoOATradeSide,
    ProtoOATrendbarPeriod,
)
from twisted.internet import reactor, defer, threads

from .broker_interface import BrokerInterface, OrderResult, AccountInfo

logger = logging.getLogger("ctrader_broker")

# Trendbar period mapping
TIMEFRAME_MAP = {
    "M1": 1, "M5": 5, "M15": 7, "M30": 8,
    "H1": 9, "H4": 10, "D1": 12, "D": 12,
}


def _load_env():
    """Load .env file into os.environ."""
    for env_path in [Path(".env"), Path(__file__).parent.parent.parent / ".env"]:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())
            break


class CTraderBroker(BrokerInterface):
    """IC Markets cTrader broker via Open API."""

    # Default symbols for the Grid EA
    DEFAULT_SYMBOLS = ["EURUSD", "GBPUSD", "EURJPY", "XAGUSD", "US500"]

    def __init__(
        self,
        client_id: str = None,
        client_secret: str = None,
        access_token: str = None,
        account_id: int = None,
        is_live: bool = False,
        refresh_token: str = None,
        account_env: str = None,
        watch_symbols: List[str] = None,
    ):
        _load_env()

        self.client_id = client_id or os.environ.get("CTRADER_CLIENT_ID", "")
        self.client_secret = client_secret or os.environ.get("CTRADER_CLIENT_SECRET", "")
        self.access_token = access_token or os.environ.get("CTRADER_ACCESS_TOKEN", "")
        # Support custom env var for account ID (e.g. CTRADER_SCALPER_ACCOUNT_ID)
        account_env_var = account_env or "CTRADER_ACCOUNT_ID"
        self.account_id = int(account_id or os.environ.get(account_env_var, "") or os.environ.get("CTRADER_ACCOUNT_ID", "0"))
        self.refresh_token = refresh_token or os.environ.get("CTRADER_REFRESH_TOKEN", "")
        self.is_live = is_live
        self.watch_symbols = watch_symbols or self.DEFAULT_SYMBOLS

        if not all([self.client_id, self.client_secret, self.access_token, self.account_id]):
            raise ValueError(
                "Missing cTrader credentials. Run: python scripts/ctrader_auth.py"
            )

        # Connection
        host = EndPoints.PROTOBUF_LIVE_HOST if is_live else EndPoints.PROTOBUF_DEMO_HOST
        self.client = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)

        # State
        self._connected = threading.Event()
        self._authenticated = threading.Event()
        self._symbol_map: Dict[str, int] = {}  # name -> symbolId
        self._symbol_id_to_name: Dict[int, str] = {}  # symbolId -> name
        self._symbol_details: Dict[int, Dict] = {}  # symbolId -> details
        self._spot_prices: Dict[int, Dict] = {}  # symbolId -> {bid, ask}
        self._positions: Dict[int, Dict] = {}  # positionId -> position data
        self._pending_responses: Dict[str, threading.Event] = {}
        self._pending_results: Dict[str, object] = {}
        self._reactor_thread = None
        self._balance = 0.0
        self._equity = 0.0
        self._last_token_refresh = 0  # epoch timestamp of last refresh attempt
        self._auth_failure_callback = None  # optional callback for auth failures

        # Callbacks
        self.client.setConnectedCallback(self._on_connected)
        self.client.setDisconnectedCallback(self._on_disconnected)
        self.client.setMessageReceivedCallback(self._on_message)

    def connect(self) -> bool:
        """Connect and authenticate with cTrader."""
        logger.info(f"Connecting to cTrader ({'LIVE' if self.is_live else 'DEMO'})...")

        # Start reactor in background thread
        if self._reactor_thread is None or not self._reactor_thread.is_alive():
            self._reactor_thread = threading.Thread(target=self._run_reactor, daemon=True)
            self._reactor_thread.start()

        # Start the client service
        reactor.callFromThread(self.client.startService)

        # Wait for connection
        if not self._connected.wait(timeout=15):
            logger.error("Connection timeout")
            return False

        # Wait for authentication
        if not self._authenticated.wait(timeout=15):
            logger.error("Authentication timeout")
            return False

        logger.info("Connected and authenticated")

        # Load symbols
        self._load_symbols()
        time.sleep(1)  # Wait for symbol response

        # Subscribe to price feeds for our instruments
        for name in self.watch_symbols:
            sid = self._symbol_map.get(name)
            if sid:
                self._subscribe_spots(sid)

        # Get initial account info and positions
        self._request_trader_info()
        self._request_positions()
        time.sleep(2)  # Wait for responses

        return True

    def disconnect(self):
        """Disconnect from cTrader."""
        logger.info("Disconnecting from cTrader...")
        try:
            reactor.callFromThread(self.client.stopService)
        except Exception:
            pass

    def get_account_info(self) -> AccountInfo:
        """Get account balance and equity."""
        self._request_trader_info()
        time.sleep(0.5)
        return AccountInfo(
            balance=self._balance,
            equity=self._equity or self._balance,
            margin_used=0,
            margin_free=self._equity or self._balance,
        )

    def place_market_order(
        self,
        symbol: str,
        direction: str,
        lot_size: float,
        stop_loss: float,
        take_profit: float,
        comment: str = "",
    ) -> OrderResult:
        """Place a market order via cTrader."""
        symbol_id = self._symbol_map.get(symbol)
        if not symbol_id:
            return OrderResult(success=False, error=f"Unknown symbol: {symbol}")

        details = self._symbol_details.get(symbol_id, {})
        digits = details.get("digits", 5)

        # Convert lot size to volume (1.0 lot = lot_size_units volume)
        step_volume = details.get("step_volume", 1000)  # minimum increment
        min_volume = details.get("min_volume", 1000)  # minimum order size
        lot_size_units = details.get("lot_size", 100000)
        volume = int(lot_size * lot_size_units)
        # Round to step volume (but don't force up to step_volume if smaller)
        if step_volume > 0 and volume >= step_volume:
            volume = (volume // step_volume) * step_volume
        # Check minimum volume - reject if too small rather than silently inflating
        if volume < min_volume:
            min_lot = min_volume / lot_size_units if lot_size_units > 0 else 0
            return OrderResult(
                success=False,
                error=f"Volume {volume} below minimum {min_volume} "
                      f"(need at least {min_lot:.4f} lot for {symbol})"
            )

        # Round prices to symbol's allowed digits
        sl_rounded = round(stop_loss, digits)
        tp_rounded = round(take_profit, digits)

        # cTrader: market orders can't have absolute SL/TP inline
        # Place order first, then modify to add SL/TP
        req = ProtoOANewOrderReq()
        req.ctidTraderAccountId = self.account_id
        req.symbolId = symbol_id
        req.orderType = ProtoOAOrderType.Value("MARKET")
        req.tradeSide = ProtoOATradeSide.Value("BUY" if direction == "BUY" else "SELL")
        req.volume = volume
        if comment:
            req.comment = comment

        logger.info(
            f"Placing {direction} {symbol} vol={volume} "
            f"SL={sl_rounded} TP={tp_rounded}"
        )

        response = self._send_and_wait(req, timeout=10)
        if response is None:
            return OrderResult(success=False, error="Order timeout")

        result = Protobuf.extract(response)
        position_id = None
        fill_price = None

        if hasattr(result, "position"):
            pos = result.position
            position_id = pos.positionId
            # Try price field, also check tradeData for rate
            if hasattr(pos, "price") and pos.price > 0:
                fill_price = round(pos.price / 100000.0, digits)
            if hasattr(pos, "tradeData"):
                td = pos.tradeData
                logger.info(f"Fill tradeData: symbolId={td.symbolId}, volume={td.volume}, "
                            f"tradeSide={td.tradeSide}")
        elif hasattr(result, "order"):
            order = result.order
            position_id = order.positionId if hasattr(order, "positionId") else None
            if hasattr(order, "executionPrice") and order.executionPrice > 0:
                fill_price = round(order.executionPrice / 100000.0, digits)
            logger.info(f"Fill via order: positionId={position_id}, "
                        f"execPrice={getattr(order, 'executionPrice', 'N/A')}")
        elif hasattr(result, "errorCode"):
            return OrderResult(success=False, error=f"Error {result.errorCode}: {getattr(result, 'description', '')}")

        if position_id is None:
            return OrderResult(success=False, error="No position ID in fill response")

        # If fill price still unknown, use current spot as estimate
        if not fill_price:
            spot = self._spot_prices.get(symbol_id, {})
            if direction == "BUY" and "ask" in spot:
                fill_price = spot["ask"]
            elif "bid" in spot:
                fill_price = spot["bid"]
            if fill_price:
                logger.info(f"Using spot price as fill estimate: {fill_price}")

        # Now add SL/TP via modify
        logger.info(f"Setting SL/TP on position {position_id}: SL={sl_rounded} TP={tp_rounded}")
        modify_result = self.modify_position(
            str(position_id),
            stop_loss=sl_rounded,
            take_profit=tp_rounded,
        )
        if not modify_result.success:
            logger.warning(f"SL/TP modify failed: {modify_result.error} (position {position_id} still open)")

        return OrderResult(
            success=True,
            order_id=str(position_id),
            fill_price=fill_price,
        )

    def close_position(self, order_id: str) -> OrderResult:
        """Close a position by ID."""
        position_id = int(order_id)

        # Get position volume for full close
        pos_data = self._positions.get(position_id, {})
        volume = pos_data.get("volume", 0)
        if volume == 0:
            # Try to reconcile first
            self._request_positions()
            time.sleep(1)
            pos_data = self._positions.get(position_id, {})
            volume = pos_data.get("volume", 0)

        req = ProtoOAClosePositionReq()
        req.ctidTraderAccountId = self.account_id
        req.positionId = position_id
        req.volume = volume

        logger.info(f"Closing position {position_id}, volume={volume}")

        response = self._send_and_wait(req, timeout=10)
        if response is None:
            return OrderResult(success=False, error="Close timeout")

        result = Protobuf.extract(response)
        msg_type = type(result).__name__

        # Check for error response
        if msg_type == "ProtoOAErrorRes" or (hasattr(result, "errorCode") and str(result.errorCode)):
            error_code = str(getattr(result, "errorCode", ""))
            description = str(getattr(result, "description", ""))
            error_msg = f"{error_code} {description}".strip()
            # Include POSITION_NOT_FOUND in error for upstream handling
            if "NOT_FOUND" in error_code or "NOT_FOUND" in description:
                return OrderResult(success=False, error=f"POSITION_NOT_FOUND")
            if error_msg:
                return OrderResult(success=False, error=f"Error: {error_msg}")

        # Get fill price from spot (result.position.price is unreliable for closes)
        fill_price = None
        symbol_id = pos_data.get("symbolId", 0)
        spot = self._spot_prices.get(symbol_id, {})
        # For a close: SELL close = buy back (ask), BUY close = sell (bid)
        direction = pos_data.get("direction", "")
        if direction == "BUY" and "bid" in spot:
            fill_price = spot["bid"]
        elif direction == "SELL" and "ask" in spot:
            fill_price = spot["ask"]
        elif "bid" in spot:
            fill_price = spot["bid"]

        if fill_price:
            logger.info(f"Close fill price (from spot): {fill_price}")

        return OrderResult(success=True, order_id=order_id, fill_price=fill_price)

    def modify_position(
        self,
        order_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> OrderResult:
        """Modify SL/TP on an open position."""
        position_id = int(order_id)

        # Look up symbol digits for price rounding
        pos_data = self._positions.get(position_id, {})
        symbol_id = pos_data.get("symbolId", 0)
        details = self._symbol_details.get(symbol_id, {})
        digits = details.get("digits", 5)

        req = ProtoOAAmendPositionSLTPReq()
        req.ctidTraderAccountId = self.account_id
        req.positionId = position_id
        if stop_loss is not None:
            req.stopLoss = round(stop_loss, digits)
        if take_profit is not None:
            req.takeProfit = round(take_profit, digits)

        response = self._send_and_wait(req, timeout=5)
        if response is None:
            return OrderResult(success=False, error="Modify timeout")

        return OrderResult(success=True, order_id=order_id)

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get open positions."""
        self._request_positions()
        time.sleep(1)

        positions = list(self._positions.values())
        if symbol:
            symbol_id = self._symbol_map.get(symbol)
            positions = [p for p in positions if p.get("symbolId") == symbol_id]
        return positions

    def get_symbol_name(self, symbol_id: int) -> Optional[str]:
        """Get symbol name from symbolId."""
        return self._symbol_id_to_name.get(symbol_id)

    def get_symbol_lot_size(self, symbol_id: int) -> int:
        """Get lot_size (units per lot) for a symbol. E.g. 100000 for forex, 100 for indices."""
        details = self._symbol_details.get(symbol_id, {})
        return details.get("lot_size", 100000)

    def get_candles(
        self,
        symbol: str,
        timeframe: str = "H1",
        count: int = 250,
    ) -> Optional[pd.DataFrame]:
        """Get historical candles from cTrader."""
        # Wait for connection if disconnected (auto-reconnect in progress)
        if not self._connected.is_set():
            logger.warning(f"Connection lost, waiting for reconnect...")
            if not self._connected.wait(timeout=30):
                logger.error("Reconnect timeout - no connection")
                return None
            # Wait for auth after reconnect
            if not self._authenticated.wait(timeout=15):
                logger.error("Re-authentication timeout")
                return None
            logger.info("Reconnected successfully")

        symbol_id = self._symbol_map.get(symbol)
        if not symbol_id:
            logger.warning(f"Unknown symbol for candles: {symbol}")
            return None

        details = self._symbol_details.get(symbol_id, {})
        digits = details.get("digits", 5)

        period = TIMEFRAME_MAP.get(timeframe, 9)  # default H1

        # Calculate time range
        now_ms = int(time.time() * 1000)
        # Hours per bar for each timeframe
        hours_map = {"M1": 1/60, "M5": 5/60, "M15": 0.25, "M30": 0.5, "H1": 1, "H4": 4, "D1": 24, "D": 24}
        hours_per_bar = hours_map.get(timeframe, 1)
        from_ms = now_ms - int(count * hours_per_bar * 3600 * 1000 * 1.5)  # 1.5x buffer for weekends

        req = ProtoOAGetTrendbarsReq()
        req.ctidTraderAccountId = self.account_id
        req.symbolId = symbol_id
        req.period = period
        req.fromTimestamp = from_ms
        req.toTimestamp = now_ms

        response = self._send_and_wait(req, timeout=15)
        if response is None:
            logger.warning(f"Candle request timeout for {symbol}")
            return None

        result = Protobuf.extract(response)
        if hasattr(result, "errorCode"):
            logger.error(f"Candle request error for {symbol}: {result.errorCode} - {getattr(result, 'description', '')}")
            return None
        if not hasattr(result, "trendbar") or len(result.trendbar) == 0:
            logger.warning(f"No candle data for {symbol}")
            return None

        # Parse trendbars (delta-encoded)
        # cTrader trendbar prices are ALWAYS in relative format divided by 100000,
        # then rounded to symbol digits. See: https://help.ctrader.com/open-api/symbol-data/
        rows = []
        for bar in result.trendbar:
            raw_low = bar.low
            raw_open = raw_low + (bar.deltaOpen if bar.deltaOpen else 0)
            raw_high = raw_low + (bar.deltaHigh if bar.deltaHigh else 0)
            raw_close = raw_low + (bar.deltaClose if bar.deltaClose else 0)
            low = round(raw_low / 100000.0, digits)
            open_ = round(raw_open / 100000.0, digits)
            high = round(raw_high / 100000.0, digits)
            close = round(raw_close / 100000.0, digits)
            volume = bar.volume if hasattr(bar, "volume") else 0
            ts = datetime.fromtimestamp(
                bar.utcTimestampInMinutes * 60, tz=timezone.utc
            )
            rows.append({
                "time": ts,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            })

        df = pd.DataFrame(rows)
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)

        # Calculate ATR(14) to match backtest engine expectations
        tr = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1)),
            ),
        )
        df["atr"] = tr.rolling(14).mean()

        return df.tail(count)

    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """Get current bid/ask from spot subscription."""
        symbol_id = self._symbol_map.get(symbol)
        if not symbol_id or symbol_id not in self._spot_prices:
            # Fall back to last candle close
            candles = self.get_candles(symbol, "M1", count=1)
            if candles is not None and len(candles) > 0:
                mid = candles.iloc[-1]["close"]
                return {"bid": mid, "ask": mid, "mid": mid}
            return None

        prices = self._spot_prices[symbol_id]
        return {
            "bid": prices.get("bid", 0),
            "ask": prices.get("ask", 0),
            "mid": (prices.get("bid", 0) + prices.get("ask", 0)) / 2,
        }

    # --- Internal methods ---

    def _run_reactor(self):
        """Run Twisted reactor in background thread."""
        reactor.run(installSignalHandlers=False)

    def _on_connected(self, client):
        """Handle connection established."""
        logger.info("TCP connected, authenticating app...")
        self._connected.set()

        # Step 1: App auth
        req = ProtoOAApplicationAuthReq()
        req.clientId = self.client_id
        req.clientSecret = self.client_secret
        d = client.send(req)
        d.addCallback(self._on_app_auth)
        d.addErrback(self._on_error)

    def _on_app_auth(self, msg):
        """Handle app auth response, proceed to account auth."""
        logger.info("App authenticated, authenticating account...")
        req = ProtoOAAccountAuthReq()
        req.ctidTraderAccountId = self.account_id
        req.accessToken = self.access_token
        d = self.client.send(req)
        d.addCallback(self._on_account_auth)
        d.addErrback(self._on_error)

    def _on_account_auth(self, msg):
        """Handle account auth response."""
        logger.info(f"Account {self.account_id} authenticated")
        self._authenticated.set()

    def _on_disconnected(self, client, reason):
        """Handle disconnection - attempt automatic reconnect."""
        logger.warning(f"Disconnected: {reason}")
        self._connected.clear()
        self._authenticated.clear()
        # Auto-reconnect after a brief delay
        def _reconnect():
            time.sleep(5)
            logger.info("Attempting auto-reconnect...")
            try:
                reactor.callFromThread(self.client.startService)
            except Exception as e:
                logger.error(f"Reconnect failed: {e}")
        threading.Thread(target=_reconnect, daemon=True).start()

    def _on_message(self, client, msg):
        """Handle all incoming messages."""
        try:
            result = Protobuf.extract(msg)
            msg_type = type(result).__name__

            # Spot price updates
            # cTrader prices are always relative / 100000, rounded to digits
            if msg_type == "ProtoOASpotEvent":
                sid = result.symbolId
                details = self._symbol_details.get(sid, {})
                digits = details.get("digits", 5)
                if sid not in self._spot_prices:
                    self._spot_prices[sid] = {}
                if result.HasField("bid"):
                    self._spot_prices[sid]["bid"] = round(result.bid / 100000.0, digits)
                if result.HasField("ask"):
                    self._spot_prices[sid]["ask"] = round(result.ask / 100000.0, digits)

            # Execution events (order fills, closes)
            elif msg_type == "ProtoOAExecutionEvent":
                if hasattr(result, "position"):
                    pos = result.position
                    self._positions[pos.positionId] = {
                        "positionId": pos.positionId,
                        "symbolId": pos.tradeData.symbolId if hasattr(pos, "tradeData") else 0,
                        "volume": pos.tradeData.volume if hasattr(pos, "tradeData") else 0,
                        "direction": "BUY" if (hasattr(pos, "tradeData") and pos.tradeData.tradeSide == 1) else "SELL",
                    }

            # Trader info response
            elif msg_type == "ProtoOATraderRes":
                if hasattr(result, "trader"):
                    self._balance = result.trader.balance / 100  # cents to dollars

            # Symbol list response
            elif msg_type == "ProtoOASymbolsListRes":
                for sym in result.symbol:
                    self._symbol_map[sym.symbolName] = sym.symbolId
                    self._symbol_id_to_name[sym.symbolId] = sym.symbolName
                logger.info(f"Loaded {len(result.symbol)} symbols")
                # Request details for our symbols
                our_ids = [
                    self._symbol_map[n]
                    for n in self.watch_symbols
                    if n in self._symbol_map
                ]
                if our_ids:
                    self._request_symbol_details(our_ids)

            # Symbol details response
            elif msg_type == "ProtoOASymbolByIdRes":
                for sym in result.symbol:
                    self._symbol_details[sym.symbolId] = {
                        "symbolId": sym.symbolId,
                        "digits": sym.digits,
                        "lot_size": sym.lotSize if hasattr(sym, "lotSize") else 100000,
                        "step_volume": sym.stepVolume if hasattr(sym, "stepVolume") else 1000,
                        "min_volume": sym.minVolume if hasattr(sym, "minVolume") else 1000,
                        "pip_position": sym.pipPosition if hasattr(sym, "pipPosition") else 4,
                    }

            # Reconcile (positions) response
            elif msg_type == "ProtoOAReconcileRes":
                self._positions.clear()
                for pos in result.position:
                    self._positions[pos.positionId] = {
                        "positionId": pos.positionId,
                        "symbolId": pos.tradeData.symbolId,
                        "volume": pos.tradeData.volume,
                        "direction": "BUY" if pos.tradeData.tradeSide == 1 else "SELL",
                        "price": pos.price,
                    }

            # Error response
            elif msg_type == "ProtoOAErrorRes":
                error_code = str(getattr(result, "errorCode", ""))
                description = str(getattr(result, "description", ""))
                logger.error(f"cTrader error: {error_code} - {description}")
                # Check if token expired or account not authorized
                if ("INVALID_ACCESS_TOKEN" in error_code or
                    ("INVALID_REQUEST" in error_code and "not authorized" in description.lower())):
                    self._refresh_access_token()

        except Exception as e:
            logger.debug(f"Message parse: {e}")

    def _on_error(self, failure):
        """Handle Twisted deferred errors."""
        logger.error(f"Error: {failure}")

    def _send_and_wait(self, req, timeout=10):
        """Send a request and wait for the response synchronously."""
        msg_id = str(id(req))
        event = threading.Event()
        self._pending_responses[msg_id] = event
        self._pending_results[msg_id] = None

        def _send():
            d = self.client.send(req, clientMsgId=msg_id, responseTimeoutInSeconds=timeout)
            d.addCallback(lambda msg: self._resolve_pending(msg_id, msg))
            d.addErrback(lambda f: self._resolve_pending(msg_id, None))

        reactor.callFromThread(_send)
        event.wait(timeout=timeout + 2)

        result = self._pending_results.pop(msg_id, None)
        self._pending_responses.pop(msg_id, None)
        return result

    def _resolve_pending(self, msg_id, msg):
        """Resolve a pending synchronous request."""
        self._pending_results[msg_id] = msg
        event = self._pending_responses.get(msg_id)
        if event:
            event.set()

    def _load_symbols(self):
        """Request symbol list from cTrader."""
        def _send():
            req = ProtoOASymbolsListReq()
            req.ctidTraderAccountId = self.account_id
            d = self.client.send(req)
            d.addErrback(self._on_error)

        reactor.callFromThread(_send)

    def _request_symbol_details(self, symbol_ids):
        """Request detailed symbol info."""
        def _send():
            req = ProtoOASymbolByIdReq()
            req.ctidTraderAccountId = self.account_id
            for sid in symbol_ids:
                req.symbolId.append(sid)
            d = self.client.send(req)
            d.addErrback(self._on_error)

        reactor.callFromThread(_send)

    def _subscribe_spots(self, symbol_id):
        """Subscribe to spot price updates."""
        def _send():
            req = ProtoOASubscribeSpotsReq()
            req.ctidTraderAccountId = self.account_id
            req.symbolId.append(symbol_id)
            d = self.client.send(req)
            d.addErrback(self._on_error)

        reactor.callFromThread(_send)

    def _request_trader_info(self):
        """Request trader/account info."""
        def _send():
            req = ProtoOATraderReq()
            req.ctidTraderAccountId = self.account_id
            d = self.client.send(req)
            d.addErrback(self._on_error)

        reactor.callFromThread(_send)

    def _request_positions(self):
        """Request current open positions."""
        def _send():
            req = ProtoOAReconcileReq()
            req.ctidTraderAccountId = self.account_id
            d = self.client.send(req)
            d.addErrback(self._on_error)

        reactor.callFromThread(_send)

    def _refresh_access_token(self):
        """Refresh the access token using the refresh token."""
        # Cooldown: don't retry more than once every 60 seconds
        now = time.time()
        if now - self._last_token_refresh < 60:
            return
        self._last_token_refresh = now

        if not self.refresh_token:
            logger.error("No refresh token available - manual re-auth needed")
            if self._auth_failure_callback:
                self._auth_failure_callback("No refresh token available. Run: python scripts/ctrader_auth.py")
            return

        logger.info("Refreshing access token...")
        auth = Auth(self.client_id, self.client_secret, "http://localhost:5000/callback")
        try:
            result = auth.refreshToken(self.refresh_token)
            if "access_token" in result:
                self.access_token = result["access_token"]
                if "refresh_token" in result:
                    self.refresh_token = result["refresh_token"]

                # Update .env file
                env_path = Path(__file__).parent.parent.parent / ".env"
                if env_path.exists():
                    content = env_path.read_text()
                    lines = content.splitlines()
                    for i, line in enumerate(lines):
                        if line.startswith("CTRADER_ACCESS_TOKEN="):
                            lines[i] = f"CTRADER_ACCESS_TOKEN={self.access_token}"
                        if line.startswith("CTRADER_REFRESH_TOKEN=") and "refresh_token" in result:
                            lines[i] = f"CTRADER_REFRESH_TOKEN={self.refresh_token}"
                    env_path.write_text("\n".join(lines) + "\n")

                # Re-authenticate account with new token
                self._authenticated.clear()
                req = ProtoOAAccountAuthReq()
                req.ctidTraderAccountId = self.account_id
                req.accessToken = self.access_token

                def _send_reauth():
                    d = self.client.send(req)
                    d.addCallback(self._on_account_auth)
                    d.addErrback(self._on_error)

                reactor.callFromThread(_send_reauth)
                logger.info("Token refreshed, re-authenticating account...")
            else:
                logger.error(f"Token refresh failed: {result}")
                if self._auth_failure_callback:
                    self._auth_failure_callback(f"Token refresh failed: {result}")
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            if self._auth_failure_callback:
                self._auth_failure_callback(f"Token refresh error: {e}")
