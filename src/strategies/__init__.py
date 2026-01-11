"""Strategies module for Expert Advisor."""

from .base_strategy import BaseStrategy
from .trend_following import TrendFollowingStrategy
from . import indicators

__all__ = [
    'BaseStrategy',
    'TrendFollowingStrategy',
    'indicators',
]
