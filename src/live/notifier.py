"""Telegram notification manager for the live trading engine.

Sends trade events (opens, closes, errors, daily summaries) to Telegram.
Uses the Bot API via requests (sync) - no async needed.
All calls are fail-silent to never crash the trading engine.
"""
import os
import logging
import threading
from datetime import datetime, timezone
from typing import Optional, Dict
from pathlib import Path

import requests

logger = logging.getLogger("notifier")


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


class TelegramNotifier:
    """Send trade notifications to Telegram."""

    def __init__(
        self,
        bot_token: str = None,
        chat_id: str = None,
    ):
        _load_env()
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.bot_token and self.chat_id)
        self._api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

        if not self.enabled:
            logger.warning("Telegram notifications disabled (missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID)")

    def send(self, message: str) -> bool:
        """Send a message to Telegram. Non-blocking, fail-silent."""
        if not self.enabled:
            return False
        threading.Thread(target=self._send_sync, args=(message,), daemon=True).start()
        return True

    def _send_sync(self, message: str):
        """Actually send the message (runs in background thread)."""
        try:
            resp = requests.post(
                self._api_url,
                json={
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                },
                timeout=10,
            )
            if not resp.ok:
                logger.debug(f"Telegram send failed: {resp.status_code} {resp.text[:100]}")
        except Exception as e:
            logger.debug(f"Telegram send error: {e}")

    def send_startup(self, profile: str, capital: float, broker: str, instruments: list):
        """Notify engine startup."""
        msg = (
            f"<b>EA Started</b>\n"
            f"Profile: {profile}\n"
            f"Capital: ${capital:.0f}\n"
            f"Broker: {broker}\n"
            f"Instruments: {', '.join(instruments)}\n"
            f"Session: 12-16 UTC"
        )
        self.send(msg)

    def send_trade_opened(
        self,
        symbol: str,
        direction: str,
        price: float,
        lot_size: float,
        stop_loss: float,
        take_profit: float,
        grid_level: int = 0,
    ):
        """Notify trade opened."""
        arrow = "\u2197\ufe0f" if direction == "BUY" else "\u2198\ufe0f"
        msg = (
            f"{arrow} <b>OPENED {direction} {symbol}</b>\n"
            f"Price: {price:.5f} | Lot: {lot_size:.3f} | Grid: L{grid_level}\n"
            f"SL: {stop_loss:.5f} | TP: {take_profit:.5f}"
        )
        self.send(msg)

    def send_trade_closed(
        self,
        symbol: str,
        direction: str,
        exit_price: float,
        reason: str,
        pips: float,
        pnl: float,
    ):
        """Notify trade closed."""
        icon = "\u2705" if pnl >= 0 else "\u274c"
        sign = "+" if pnl >= 0 else ""
        msg = (
            f"{icon} <b>CLOSED {direction} {symbol}</b>\n"
            f"Exit: {exit_price:.5f} | {sign}{pips:.1f} pips | {sign}${pnl:.2f}\n"
            f"Reason: {reason}"
        )
        self.send(msg)

    def send_error(self, error_msg: str):
        """Notify error."""
        msg = f"\u26a0\ufe0f <b>ERROR</b>\n{error_msg}"
        self.send(msg)

    def send_daily_summary(self, status: Dict):
        """Send end-of-session summary."""
        msg = (
            f"\U0001f4ca <b>SESSION SUMMARY</b>\n"
            f"Balance: ${status.get('balance', 0):.2f}\n"
            f"Equity: ${status.get('equity', 0):.2f}\n"
            f"Open positions: {status.get('open_positions', 0)}\n"
            f"Signals today: {status.get('signals', 0)}\n"
            f"Trades opened: {status.get('trades_opened', 0)}\n"
            f"Trades closed: {status.get('trades_closed', 0)}"
        )
        self.send(msg)
