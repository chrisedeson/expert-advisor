"""State persistence for live trading. Saves/loads engine state to JSON.

Ensures we can restart the engine without losing position tracking.
"""
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional


class StateManager:
    """Persists live trading state to disk."""

    def __init__(self, state_file: str = "state/live_state.json"):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def save(self, state: Dict):
        """Save full engine state to JSON."""
        # Convert datetimes to strings
        serializable = self._make_serializable(state)
        serializable['saved_at'] = datetime.now(timezone.utc).isoformat()

        with open(self.state_file, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)

    def load(self) -> Optional[Dict]:
        """Load engine state from JSON. Returns None if no state file."""
        if not self.state_file.exists():
            return None
        with open(self.state_file, 'r') as f:
            return json.load(f)

    def _make_serializable(self, obj):
        """Recursively convert non-serializable objects."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        return obj

    def save_trade_log(self, trade: Dict, log_file: str = "state/trade_log.jsonl"):
        """Append a trade to the trade log (JSONL format)."""
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a') as f:
            f.write(json.dumps(trade, default=str) + '\n')
