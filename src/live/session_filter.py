"""Session filter: only allow trading during London/NY overlap (12-16 UTC)."""
from datetime import datetime, timezone


class SessionFilter:
    """Filters trading to specific UTC hours."""

    def __init__(self, start_hour: int = 12, end_hour: int = 16):
        self.start_hour = start_hour
        self.end_hour = end_hour

    def is_active(self, dt: datetime = None) -> bool:
        """Check if current time is within the trading session."""
        if dt is None:
            dt = datetime.now(timezone.utc)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        # Weekday check: Mon=0, Sun=6
        if dt.weekday() >= 5:
            return False
        return self.start_hour <= dt.hour <= self.end_hour

    def next_session_start(self, dt: datetime = None) -> datetime:
        """Return the next session start time."""
        if dt is None:
            dt = datetime.now(timezone.utc)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        # If we're before today's session, return today's start
        today_start = dt.replace(hour=self.start_hour, minute=0, second=0, microsecond=0)
        if dt < today_start and dt.weekday() < 5:
            return today_start
        # Otherwise, next weekday
        from datetime import timedelta
        next_day = dt + timedelta(days=1)
        next_day = next_day.replace(hour=self.start_hour, minute=0, second=0, microsecond=0)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        return next_day
