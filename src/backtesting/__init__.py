"""Backtesting module for Expert Advisor."""

from .engine import BacktestEngine
from .transaction_costs import TransactionCostModel
from .risk_manager import RiskManager
from .portfolio import Portfolio, Position

try:
    from .protected_grid_engine import (
        ProtectedGridBacktester,
        BacktestResult,
        ProtectedTrade,
        ProtectionEvent,
        print_backtest_summary,
    )

    __all__ = [
        'BacktestEngine',
        'TransactionCostModel',
        'RiskManager',
        'Portfolio',
        'Position',
        'ProtectedGridBacktester',
        'BacktestResult',
        'ProtectedTrade',
        'ProtectionEvent',
        'print_backtest_summary',
    ]
except ImportError as e:
    # Protected grid engine requires risk module, may not be available yet
    __all__ = [
        'BacktestEngine',
        'TransactionCostModel',
        'RiskManager',
        'Portfolio',
        'Position',
    ]
