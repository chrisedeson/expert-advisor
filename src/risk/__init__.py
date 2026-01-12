"""Risk management module for Smart Grid EA."""

try:
    from .volatility_filter import VolatilityFilter, MarketCondition
    from .circuit_breaker import CircuitBreaker, CircuitBreakerState

    __all__ = [
        'VolatilityFilter',
        'MarketCondition',
        'CircuitBreaker',
        'CircuitBreakerState',
    ]
except ImportError as e:
    # Allow module to be imported even if dependencies not yet available
    print(f"Warning: Some risk management components not available: {e}")
    __all__ = []
