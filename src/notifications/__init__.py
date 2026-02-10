"""Notifications module for Smart Grid EA."""

try:
    from .alert_manager import AlertManager, AlertLevel, AlertType

    __all__ = [
        'AlertManager',
        'AlertLevel',
        'AlertType',
    ]
except ImportError as e:
    print(f"Warning: Alert system not available: {e}")
    __all__ = []
