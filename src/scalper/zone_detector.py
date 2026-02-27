"""Supply & Demand Zone Detector.

Detects institutional supply/demand zones based on impulsive moves,
fair value gaps, break of structure, and EMA trend alignment.

Rules from "Trade with Pat" strategy:
- DEMAND: 4+ consecutive green candles = impulsive move up
  Zone = last red candle before the green push (open to low)
- SUPPLY: 4+ consecutive red candles = impulsive move down
  Zone = last green candle before the red push (open to high)
- Zone valid until price CLOSES through it (wicks OK)
- Strength scored 0-10 based on FVG, BOS, EMA, confluence
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from datetime import datetime


@dataclass
class SupplyDemandZone:
    """A supply or demand zone."""
    zone_type: str              # 'supply' or 'demand'
    top: float                  # Upper boundary
    bottom: float               # Lower boundary
    creation_idx: int           # Bar index when zone was created
    creation_time: datetime     # Timestamp
    strength: float = 0.0       # Score 0-10
    touches: int = 0            # Times price tested this zone
    is_valid: bool = True       # Invalid if price closes through
    invalidation_idx: int = -1  # Bar where zone was invalidated
    has_fvg: bool = False
    has_bos: bool = False
    ema_aligned: bool = False
    consecutive_count: int = 4  # How many consecutive candles formed this
    clean_push: bool = True     # No opposite candles in the push
    push_size: float = 0.0      # Size of the impulsive move

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def height(self) -> float:
        return self.top - self.bottom


class ZoneDetector:
    """Detects supply and demand zones from OHLC data.

    Multi-timeframe capable: detect zones on HTF, check entries on LTF.
    """

    def __init__(
        self,
        min_consecutive: int = 4,
        ema_period: int = 200,
        min_zone_score: float = 5.0,
        max_zone_age_bars: int = 500,
        zone_touch_limit: int = 3,
        momentum_lookback: int = 5,
    ):
        self.min_consecutive = min_consecutive
        self.ema_period = ema_period
        self.min_zone_score = min_zone_score
        self.max_zone_age_bars = max_zone_age_bars
        self.zone_touch_limit = zone_touch_limit
        self.momentum_lookback = momentum_lookback

    def detect_zones(self, df: pd.DataFrame) -> List[SupplyDemandZone]:
        """Find all supply/demand zones in OHLC data.

        Args:
            df: DataFrame with columns [open, high, low, close] (lowercase).
                Must have enough bars for EMA calculation.

        Returns:
            List of SupplyDemandZone objects, sorted by creation index.
        """
        if len(df) < self.ema_period + 10:
            return []

        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values

        # Compute EMA
        ema = self._ema(closes, self.ema_period)

        # Detect FVGs for the whole series
        fvg_bull, fvg_bear = self._detect_fvgs(highs, lows, closes, opens)

        # Detect swing highs/lows for BOS
        swing_highs, swing_lows = self._detect_swings(highs, lows, period=10)

        zones = []
        n = len(df)
        i = self.ema_period  # Start after EMA warmup

        while i < n:
            # Check for consecutive green candles (demand zone setup)
            green_count = self._count_consecutive(closes, opens, i, bullish=True)
            if green_count >= self.min_consecutive:
                zone = self._build_demand_zone(
                    df, i, green_count, ema, fvg_bull, swing_highs, swing_lows
                )
                if zone is not None:
                    zones.append(zone)
                i += green_count
                continue

            # Check for consecutive red candles (supply zone setup)
            red_count = self._count_consecutive(closes, opens, i, bullish=False)
            if red_count >= self.min_consecutive:
                zone = self._build_supply_zone(
                    df, i, red_count, ema, fvg_bear, swing_highs, swing_lows
                )
                if zone is not None:
                    zones.append(zone)
                i += red_count
                continue

            i += 1

        return zones

    def update_zones(
        self,
        zones: List[SupplyDemandZone],
        df: pd.DataFrame,
        current_idx: int,
    ) -> List[SupplyDemandZone]:
        """Update zone validity and touches based on current price action.

        Invalidates zones where price CLOSED through them.
        Counts touches where price entered zone but didn't close through.

        Args:
            zones: Existing zones to update.
            df: Full OHLC data.
            current_idx: Current bar index to check up to.

        Returns:
            Filtered list of still-valid zones.
        """
        if current_idx >= len(df):
            return [z for z in zones if z.is_valid]

        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values

        active = []
        for zone in zones:
            if not zone.is_valid:
                continue

            # Check age
            age = current_idx - zone.creation_idx
            if age > self.max_zone_age_bars:
                zone.is_valid = False
                continue

            # Check if price closed through zone
            # For demand: invalidated if close < zone.bottom
            # For supply: invalidated if close > zone.top
            bar_close = closes[current_idx]
            bar_high = highs[current_idx]
            bar_low = lows[current_idx]

            if zone.zone_type == 'demand':
                if bar_close < zone.bottom:
                    zone.is_valid = False
                    zone.invalidation_idx = current_idx
                    continue
                # Touch: price entered zone (low touched or went into zone)
                if bar_low <= zone.top and bar_low >= zone.bottom:
                    zone.touches += 1
            else:  # supply
                if bar_close > zone.top:
                    zone.is_valid = False
                    zone.invalidation_idx = current_idx
                    continue
                if bar_high >= zone.bottom and bar_high <= zone.top:
                    zone.touches += 1

            # Touch limit
            if self.zone_touch_limit > 0 and zone.touches > self.zone_touch_limit:
                zone.is_valid = False
                continue

            active.append(zone)

        return active

    def get_active_zones(
        self,
        zones: List[SupplyDemandZone],
        current_idx: int,
        min_score: float = None,
    ) -> List[SupplyDemandZone]:
        """Get zones that are valid and meet minimum score at current_idx."""
        if min_score is None:
            min_score = self.min_zone_score

        return [
            z for z in zones
            if z.is_valid
            and z.strength >= min_score
            and (current_idx - z.creation_idx) <= self.max_zone_age_bars
        ]

    def check_momentum(
        self,
        df: pd.DataFrame,
        zone: SupplyDemandZone,
        current_idx: int,
    ) -> str:
        """Check if price is approaching the zone with slow or fast momentum.

        Returns 'slow' (good - small candles), 'fast' (bad - huge candle), or 'neutral'.
        """
        if current_idx < self.momentum_lookback:
            return 'neutral'

        closes = df['close'].values
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values

        # Get ATR-like measure for context
        recent_ranges = []
        for j in range(max(0, current_idx - 20), current_idx):
            recent_ranges.append(highs[j] - lows[j])
        avg_range = np.mean(recent_ranges) if recent_ranges else 0

        if avg_range <= 0:
            return 'neutral'

        # Check the last few candles approaching the zone
        approach_ranges = []
        for j in range(current_idx - self.momentum_lookback, current_idx + 1):
            if j >= 0 and j < len(df):
                approach_ranges.append(highs[j] - lows[j])

        if not approach_ranges:
            return 'neutral'

        # Current candle size relative to average
        current_range = highs[current_idx] - lows[current_idx]
        ratio = current_range / avg_range

        # Also check body size (close-open) vs range
        body = abs(closes[current_idx] - opens[current_idx])
        body_ratio = body / current_range if current_range > 0 else 0

        # Fast: big candle slamming into zone (ratio > 2x average, big body)
        if ratio > 2.0 and body_ratio > 0.6:
            return 'fast'

        # Slow: small candles, wicks, indecision
        avg_approach = np.mean(approach_ranges)
        if avg_approach < avg_range * 0.8:
            return 'slow'

        return 'neutral'

    def _count_consecutive(
        self,
        closes: np.ndarray,
        opens: np.ndarray,
        start_idx: int,
        bullish: bool,
    ) -> int:
        """Count consecutive bullish or bearish candles starting at start_idx."""
        count = 0
        i = start_idx
        n = len(closes)
        while i < n:
            if bullish:
                if closes[i] > opens[i]:
                    count += 1
                else:
                    break
            else:
                if closes[i] < opens[i]:
                    count += 1
                else:
                    break
            i += 1
        return count

    def _build_demand_zone(
        self,
        df: pd.DataFrame,
        push_start: int,
        green_count: int,
        ema: np.ndarray,
        fvg_bull: set,
        swing_highs: dict,
        swing_lows: dict,
    ) -> Optional[SupplyDemandZone]:
        """Build a demand zone from a bullish impulsive move.

        Zone = last red candle before the green push.
        Zone top = that candle's open (or high)
        Zone bottom = that candle's low
        """
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values

        # Find the last red (bearish) candle before the push
        zone_candle_idx = push_start - 1
        while zone_candle_idx >= 0 and closes[zone_candle_idx] >= opens[zone_candle_idx]:
            zone_candle_idx -= 1

        if zone_candle_idx < 0:
            return None

        # Zone boundaries: the base candle
        zone_top = max(opens[zone_candle_idx], closes[zone_candle_idx])
        zone_bottom = lows[zone_candle_idx]

        if zone_top <= zone_bottom:
            return None

        push_end = push_start + green_count - 1
        push_size = highs[push_end] - lows[push_start]

        # Check for doji/tiny candles in the zone candle
        zone_body = abs(closes[zone_candle_idx] - opens[zone_candle_idx])
        zone_range = highs[zone_candle_idx] - lows[zone_candle_idx]
        if zone_range <= 0:
            return None

        # Check for clean push (no red candles in the green run)
        clean = True
        for j in range(push_start, push_start + green_count):
            if j < len(closes) and closes[j] <= opens[j]:
                clean = False
                break

        # EMA alignment: price should be above EMA for demand
        ema_aligned = closes[push_start] > ema[push_start] if push_start < len(ema) else False

        # Check for FVG in the push
        has_fvg = False
        for j in range(push_start, push_start + green_count):
            if j in fvg_bull:
                has_fvg = True
                break

        # Check for BOS: did the push break a previous swing high?
        has_bos = False
        for sh_idx, sh_level in swing_highs.items():
            if sh_idx < push_start and sh_idx > push_start - 100:
                if highs[push_end] > sh_level:
                    has_bos = True
                    break

        # Score the zone
        score = self._score_zone(
            green_count, has_fvg, has_bos, ema_aligned, clean, push_size, zone_range
        )

        ts = df.index[zone_candle_idx] if hasattr(df.index, '__getitem__') else datetime.now()

        return SupplyDemandZone(
            zone_type='demand',
            top=zone_top,
            bottom=zone_bottom,
            creation_idx=zone_candle_idx,
            creation_time=ts,
            strength=score,
            has_fvg=has_fvg,
            has_bos=has_bos,
            ema_aligned=ema_aligned,
            consecutive_count=green_count,
            clean_push=clean,
            push_size=push_size,
        )

    def _build_supply_zone(
        self,
        df: pd.DataFrame,
        push_start: int,
        red_count: int,
        ema: np.ndarray,
        fvg_bear: set,
        swing_highs: dict,
        swing_lows: dict,
    ) -> Optional[SupplyDemandZone]:
        """Build a supply zone from a bearish impulsive move.

        Zone = last green candle before the red push.
        Zone bottom = that candle's open (or low)
        Zone top = that candle's high
        """
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values

        # Find the last green (bullish) candle before the push
        zone_candle_idx = push_start - 1
        while zone_candle_idx >= 0 and closes[zone_candle_idx] <= opens[zone_candle_idx]:
            zone_candle_idx -= 1

        if zone_candle_idx < 0:
            return None

        zone_top = highs[zone_candle_idx]
        zone_bottom = min(opens[zone_candle_idx], closes[zone_candle_idx])

        if zone_top <= zone_bottom:
            return None

        push_end = push_start + red_count - 1
        push_size = highs[push_start] - lows[push_end]

        zone_range = highs[zone_candle_idx] - lows[zone_candle_idx]
        if zone_range <= 0:
            return None

        # Clean push
        clean = True
        for j in range(push_start, push_start + red_count):
            if j < len(closes) and closes[j] >= opens[j]:
                clean = False
                break

        # EMA: price below EMA for supply
        ema_aligned = closes[push_start] < ema[push_start] if push_start < len(ema) else False

        # FVG in push
        has_fvg = False
        for j in range(push_start, push_start + red_count):
            if j in fvg_bear:
                has_fvg = True
                break

        # BOS: push broke a previous swing low
        has_bos = False
        for sl_idx, sl_level in swing_lows.items():
            if sl_idx < push_start and sl_idx > push_start - 100:
                if lows[push_end] < sl_level:
                    has_bos = True
                    break

        score = self._score_zone(
            red_count, has_fvg, has_bos, ema_aligned, clean, push_size, zone_range
        )

        ts = df.index[zone_candle_idx] if hasattr(df.index, '__getitem__') else datetime.now()

        return SupplyDemandZone(
            zone_type='supply',
            top=zone_top,
            bottom=zone_bottom,
            creation_idx=zone_candle_idx,
            creation_time=ts,
            strength=score,
            has_fvg=has_fvg,
            has_bos=has_bos,
            ema_aligned=ema_aligned,
            consecutive_count=red_count,
            clean_push=clean,
            push_size=push_size,
        )

    def _score_zone(
        self,
        consecutive_count: int,
        has_fvg: bool,
        has_bos: bool,
        ema_aligned: bool,
        clean_push: bool,
        push_size: float,
        zone_range: float,
    ) -> float:
        """Score zone strength 0-10."""
        score = 0.0

        # Base: consecutive candle count (4=2pts, 5=3pts, 6+=4pts)
        if consecutive_count >= 6:
            score += 4.0
        elif consecutive_count >= 5:
            score += 3.0
        elif consecutive_count >= 4:
            score += 2.0
        else:
            score += 1.0

        # FVG presence (+2)
        if has_fvg:
            score += 2.0

        # Break of structure (+2)
        if has_bos:
            score += 2.0

        # EMA trend alignment (+1.5)
        if ema_aligned:
            score += 1.5

        # Clean push (no opposite candles) (+1)
        if clean_push:
            score += 1.0

        # Push-to-zone ratio: bigger push relative to zone = stronger
        if zone_range > 0:
            ratio = push_size / zone_range
            if ratio > 5.0:
                score += 1.0
            elif ratio > 3.0:
                score += 0.5

        return min(score, 10.0)

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Compute EMA."""
        ema = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return ema
        ema[period - 1] = np.mean(data[:period])
        multiplier = 2.0 / (period + 1)
        for i in range(period, len(data)):
            ema[i] = data[i] * multiplier + ema[i - 1] * (1 - multiplier)
        return ema

    def _detect_fvgs(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        opens: np.ndarray,
    ) -> Tuple[set, set]:
        """Detect Fair Value Gaps (3-candle pattern with gap).

        Bullish FVG: candle1.high < candle3.low (gap up)
        Bearish FVG: candle1.low > candle3.high (gap down)

        Returns sets of indices (middle candle index) for bull and bear FVGs.
        """
        fvg_bull = set()
        fvg_bear = set()
        n = len(highs)

        for i in range(2, n):
            # Bullish FVG: candle[i-2].high < candle[i].low AND candle[i-1] is bullish
            if highs[i - 2] < lows[i] and closes[i - 1] > opens[i - 1]:
                fvg_bull.add(i - 1)

            # Bearish FVG: candle[i-2].low > candle[i].high AND candle[i-1] is bearish
            if lows[i - 2] > highs[i] and closes[i - 1] < opens[i - 1]:
                fvg_bear.add(i - 1)

        return fvg_bull, fvg_bear

    def _detect_swings(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        period: int = 10,
    ) -> Tuple[dict, dict]:
        """Detect swing highs and lows using rolling max/min for speed.

        Swing high: bar whose high is highest in [i-period, i+period]
        Swing low: bar whose low is lowest in [i-period, i+period]

        Returns dict mapping index -> level.
        """
        swing_highs = {}
        swing_lows = {}
        n = len(highs)
        if n < 2 * period + 1:
            return swing_highs, swing_lows

        # Use pandas rolling for speed
        win = 2 * period + 1
        h_series = pd.Series(highs)
        l_series = pd.Series(lows)
        roll_max = h_series.rolling(win, center=True).max().values
        roll_min = l_series.rolling(win, center=True).min().values

        for i in range(period, n - period):
            if highs[i] == roll_max[i]:
                swing_highs[i] = highs[i]
            if lows[i] == roll_min[i]:
                swing_lows[i] = lows[i]

        return swing_highs, swing_lows

    def find_tp_targets(
        self,
        df: pd.DataFrame,
        zone: SupplyDemandZone,
        current_idx: int,
        num_targets: int = 3,
    ) -> List[float]:
        """Find take-profit targets from the zone.

        For demand zones: look for resistance levels above
        For supply zones: look for support levels below

        Uses: recent swing points and FVG midpoints.
        """
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        opens = df['open'].values

        targets = []
        lookback = min(100, current_idx)

        if zone.zone_type == 'demand':
            # Find resistance levels above zone.top
            current_price = closes[current_idx]

            # Recent swing highs
            swing_highs, _ = self._detect_swings(
                highs[current_idx - lookback:current_idx + 1],
                lows[current_idx - lookback:current_idx + 1],
                period=5,
            )
            for rel_idx, level in swing_highs.items():
                abs_level = level
                if abs_level > current_price:
                    targets.append(abs_level)

            # FVG midpoints above
            fvg_bull, fvg_bear = self._detect_fvgs(
                highs[current_idx - lookback:current_idx + 1],
                lows[current_idx - lookback:current_idx + 1],
                closes[current_idx - lookback:current_idx + 1],
                opens[current_idx - lookback:current_idx + 1],
            )
            for fvg_idx in fvg_bear:  # Bearish FVGs above = targets
                abs_idx = current_idx - lookback + fvg_idx
                if abs_idx < len(highs):
                    mid = (highs[abs_idx] + lows[abs_idx]) / 2
                    if mid > current_price:
                        targets.append(mid)

        else:  # supply
            current_price = closes[current_idx]

            # Recent swing lows below
            _, swing_lows = self._detect_swings(
                highs[current_idx - lookback:current_idx + 1],
                lows[current_idx - lookback:current_idx + 1],
                period=5,
            )
            for rel_idx, level in swing_lows.items():
                abs_level = level
                if abs_level < current_price:
                    targets.append(abs_level)

            # FVG midpoints below
            fvg_bull, _ = self._detect_fvgs(
                highs[current_idx - lookback:current_idx + 1],
                lows[current_idx - lookback:current_idx + 1],
                closes[current_idx - lookback:current_idx + 1],
                opens[current_idx - lookback:current_idx + 1],
            )
            for fvg_idx in fvg_bull:
                abs_idx = current_idx - lookback + fvg_idx
                if abs_idx < len(lows):
                    mid = (highs[abs_idx] + lows[abs_idx]) / 2
                    if mid < current_price:
                        targets.append(mid)

        # Sort: closest first for demand (ascending), closest for supply (descending)
        if zone.zone_type == 'demand':
            targets.sort()
        else:
            targets.sort(reverse=True)

        # Deduplicate close targets (within 0.1% of each other)
        unique_targets = []
        for t in targets:
            if not unique_targets or abs(t - unique_targets[-1]) / max(abs(t), 1e-8) > 0.001:
                unique_targets.append(t)

        return unique_targets[:num_targets]
