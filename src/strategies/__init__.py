"""Strategies module for Expert Advisor."""

from .base_strategy import BaseStrategy
from .trend_following import TrendFollowingStrategy
from .enhanced_trend_following import EnhancedTrendFollowingStrategy
from . import indicators

__all__ = [
    'BaseStrategy',
    'TrendFollowingStrategy',
    'EnhancedTrendFollowingStrategy',
    'indicators',
]
