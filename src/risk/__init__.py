"""Risk management module for Smart Grid EA."""

try:
    from .volatility_filter import VolatilityFilter, MarketCondition
    from .circuit_breaker import CircuitBreaker, CircuitBreakerState
    from .recovery_manager import RecoveryManager, RecoveryPhase
    from .profit_protector import ProfitProtector
    from .crisis_detector import CrisisDetector, CrisisLevel

    __all__ = [
        'VolatilityFilter',
        'MarketCondition',
        'CircuitBreaker',
        'CircuitBreakerState',
        'RecoveryManager',
        'RecoveryPhase',
        'ProfitProtector',
        'CrisisDetector',
        'CrisisLevel',
    ]
except ImportError as e:
    # Allow module to be imported even if dependencies not yet available
    print(f"Warning: Some risk management components not available: {e}")
    __all__ = []
