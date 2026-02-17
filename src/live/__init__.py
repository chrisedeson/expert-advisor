"""Live trading engine for the Expert Advisor.

Modules:
- signal_engine: Signal generation (extracted from backtest engine)
- live_engine: Main trading loop
- broker_interface: Abstract broker API
- simulated_broker: Paper trading broker
- session_filter: 12-16 UTC session filter
- state_manager: State persistence
"""
from .signal_engine import SignalEngine, Signal, Position
from .live_engine import LiveEngine, INSTRUMENTS, RISK_PROFILES
from .broker_interface import BrokerInterface
from .simulated_broker import SimulatedBroker
from .session_filter import SessionFilter
from .state_manager import StateManager

__all__ = [
    'SignalEngine', 'Signal', 'Position',
    'LiveEngine', 'INSTRUMENTS', 'RISK_PROFILES',
    'BrokerInterface', 'SimulatedBroker',
    'SessionFilter', 'StateManager',
]
