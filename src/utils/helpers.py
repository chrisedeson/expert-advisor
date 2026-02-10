"""
Helper utilities for Expert Advisor
"""

from datetime import datetime, timezone
from typing import Optional
import pandas as pd


def format_currency(amount: float, currency: str = "USD") -> str:
    """Format amount as currency"""
    return f"${amount:,.2f}" if currency == "USD" else f"{amount:,.2f} {currency}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage"""
    return f"{value * 100:.{decimals}f}%"


def format_pips(pips: float, pair: str = "EUR_USD") -> str:
    """Format pips value"""
    # JPY pairs have different pip convention
    if "JPY" in pair:
        return f"{pips / 100:.2f} pips"
    return f"{pips:.1f} pips"


def utc_now() -> datetime:
    """Get current UTC time"""
    return datetime.now(timezone.utc)


def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime"""
    return pd.to_datetime(date_str)


def validate_pair(pair: str) -> str:
    """
    Validate and normalize currency pair format

    Args:
        pair: Currency pair (e.g., "EURUSD", "EUR/USD", "EUR-USD", "EUR_USD")

    Returns:
        Normalized pair format for MT5 (e.g., "EURUSD")
    """
    # Remove common separators
    pair = pair.replace("/", "").replace("-", "").replace("_", "").upper()

    if len(pair) != 6:
        raise ValueError(f"Invalid currency pair format: {pair}")

    # MT5 format is simple: EURUSD (no separator)
    return pair


def calculate_pip_value(pair: str, position_size: int, account_currency: str = "USD") -> float:
    """
    Calculate pip value for a position

    Args:
        pair: Currency pair (e.g., "EUR_USD")
        position_size: Position size in units
        account_currency: Account currency (default: USD)

    Returns:
        Pip value in account currency
    """
    # For pairs with USD as quote currency (XXX_USD)
    if pair.endswith("_USD"):
        # 1 pip = 0.0001 for most pairs
        return position_size * 0.0001

    # For JPY pairs (XXX_JPY)
    if pair.endswith("_JPY"):
        # 1 pip = 0.01 for JPY pairs
        # Need to convert to USD
        return position_size * 0.01 / 110  # Approximate, should use real rate

    # For pairs with USD as base currency (USD_XXX)
    # Need to convert to USD
    return position_size * 0.0001  # Simplified


def calculate_position_size(
    capital: float,
    risk_per_trade: float,
    stop_loss_pips: float,
    pair: str,
    account_currency: str = "USD"
) -> int:
    """
    Calculate position size based on risk parameters

    Args:
        capital: Account capital
        risk_per_trade: Risk per trade as decimal (e.g., 0.01 for 1%)
        stop_loss_pips: Stop loss distance in pips
        pair: Currency pair
        account_currency: Account currency

    Returns:
        Position size in units
    """
    # Calculate risk amount
    risk_amount = capital * risk_per_trade

    # Calculate pip value (approximate)
    # For accurate calculation, need real-time exchange rates
    pip_value_per_unit = 0.0001 if not pair.endswith("_JPY") else 0.01

    # Position size = risk amount / (stop loss pips * pip value per unit)
    position_size = risk_amount / (stop_loss_pips * pip_value_per_unit)

    # Round to integer units
    return int(position_size)


def is_trading_session(dt: datetime, session: str = "london_ny_overlap") -> bool:
    """
    Check if given time is within trading session

    Args:
        dt: Datetime to check (should be UTC)
        session: Trading session name

    Returns:
        True if within session, False otherwise
    """
    # Convert to UTC if not already
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    hour = dt.hour

    if session == "london_ny_overlap":
        # London/NY overlap: 12:00-16:00 UTC
        return 12 <= hour < 16
    elif session == "london":
        # London session: 08:00-16:00 UTC
        return 8 <= hour < 16
    elif session == "ny":
        # NY session: 13:00-22:00 UTC
        return 13 <= hour < 22
    elif session == "asian":
        # Asian session: 00:00-08:00 UTC
        return 0 <= hour < 8

    return True  # Default: always trade


def format_timedelta(td: pd.Timedelta) -> str:
    """Format timedelta as human-readable string"""
    days = td.days
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")

    return " ".join(parts) if parts else "0m"
