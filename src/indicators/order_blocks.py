"""
Order Block Module (Simplified)

Identifies key support/resistance zones based on consolidation-then-breakout patterns.
Focuses on basic gap detection and consolidation zones per strategy requirements.
"""
import backtrader as bt
import numpy as np
from typing import List, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field


@dataclass
class OrderBlock:
    """Container for Order Block zone data"""
    ob_type: str           # 'bullish' or 'bearish'
    midpoint: float
    upper: float
    lower: float
    created_bar: int
    tested: bool = False
    test_bar: Optional[int] = None

    def contains_price(self, price: float) -> bool:
        """Check if price is within the OB zone"""
        return self.lower <= price <= self.upper

    def is_above(self, price: float) -> bool:
        """Check if price is above the OB zone"""
        return price > self.upper

    def is_below(self, price: float) -> bool:
        """Check if price is below the OB zone"""
        return price < self.lower


class OrderBlockAnalyzer(bt.Indicator):
    """
    Order Block analyzer for identifying S/R zones.

    Simplified approach:
    - Detects consolidation periods (tight range < ATR threshold)
    - Validates breakouts from consolidation
    - Tracks zone freshness (tested vs untested)

    Lines:
        ob_bullish_mid: Midpoint of nearest bullish OB (or nan)
        ob_bearish_mid: Midpoint of nearest bearish OB (or nan)
        ob_bullish_upper: Upper bound of bullish OB
        ob_bullish_lower: Lower bound of bullish OB
        ob_bearish_upper: Upper bound of bearish OB
        ob_bearish_lower: Lower bound of bearish OB
        at_bullish_ob: 1 if price at bullish OB zone, 0 otherwise
        at_bearish_ob: 1 if price at bearish OB zone, 0 otherwise
    """

    lines = (
        'ob_bullish_mid',
        'ob_bullish_upper',
        'ob_bullish_lower',
        'ob_bearish_mid',
        'ob_bearish_upper',
        'ob_bearish_lower',
        'at_bullish_ob',
        'at_bearish_ob',
        'consolidation',
    )

    params = (
        ('consolidation_bars', 5),     # Minimum bars for consolidation
        ('range_threshold', 0.02),     # Range < 2% of ATR
        ('breakout_threshold', 0.03),  # > 3% move to confirm breakout
        ('buffer', 0.005),             # 0.5% buffer around OB
        ('max_age_bars', 50),          # Remove untested OB after N bars
        ('atr_period', 14),
    )

    def __init__(self):
        # ATR for range comparison
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)

        # Store active Order Blocks
        self._bullish_obs: List[OrderBlock] = []
        self._bearish_obs: List[OrderBlock] = []

        # Consolidation tracking
        self._consolidation_start = None
        self._consolidation_high = None
        self._consolidation_low = None
        self._in_consolidation = False

    def next(self):
        # Initialize lines
        self._reset_lines()

        if len(self) < self.p.consolidation_bars + 5:
            return

        current_price = self.data.close[0]

        # Update consolidation detection
        self._update_consolidation()

        # Check for breakouts from consolidation
        if self._in_consolidation:
            self._check_breakout()

        # Update OB zone testing
        self._update_ob_testing(current_price)

        # Clean old untested OBs
        self._cleanup_old_obs()

        # Find nearest OBs and set output lines
        self._set_nearest_obs(current_price)

        # Check if price is at OB zone
        self._check_at_ob_zone(current_price)

    def _reset_lines(self):
        """Reset all output lines to nan/0"""
        self.l.ob_bullish_mid[0] = float('nan')
        self.l.ob_bullish_upper[0] = float('nan')
        self.l.ob_bullish_lower[0] = float('nan')
        self.l.ob_bearish_mid[0] = float('nan')
        self.l.ob_bearish_upper[0] = float('nan')
        self.l.ob_bearish_lower[0] = float('nan')
        self.l.at_bullish_ob[0] = 0
        self.l.at_bearish_ob[0] = 0
        self.l.consolidation[0] = 0

    def _update_consolidation(self):
        """Detect consolidation periods"""
        bars = self.p.consolidation_bars
        current_atr = self.atr[0]

        if current_atr == 0:
            return

        # Calculate range of last N bars
        highs = [self.data.high[-i] for i in range(bars)]
        lows = [self.data.low[-i] for i in range(bars)]
        range_high = max(highs)
        range_low = min(lows)
        range_pct = (range_high - range_low) / self.data.close[0]

        # Check if in consolidation (range < threshold * ATR)
        atr_pct = current_atr / self.data.close[0]
        is_tight_range = range_pct < self.p.range_threshold

        if is_tight_range and not self._in_consolidation:
            # Start new consolidation
            self._in_consolidation = True
            self._consolidation_start = len(self) - bars
            self._consolidation_high = range_high
            self._consolidation_low = range_low
            self.l.consolidation[0] = 1

        elif is_tight_range and self._in_consolidation:
            # Update consolidation range
            self._consolidation_high = max(self._consolidation_high, range_high)
            self._consolidation_low = min(self._consolidation_low, range_low)
            self.l.consolidation[0] = 1

        elif not is_tight_range and self._in_consolidation:
            # Consolidation ended, check for breakout
            self.l.consolidation[0] = 0

    def _check_breakout(self):
        """Check for breakout from consolidation zone"""
        if not self._in_consolidation:
            return

        current_close = self.data.close[0]
        cons_range = self._consolidation_high - self._consolidation_low
        cons_mid = (self._consolidation_high + self._consolidation_low) / 2

        # Bullish breakout: Close above consolidation high by threshold
        breakout_up = current_close > self._consolidation_high * (1 + self.p.breakout_threshold)

        # Bearish breakout: Close below consolidation low by threshold
        breakout_down = current_close < self._consolidation_low * (1 - self.p.breakout_threshold)

        if breakout_up:
            # Create bullish Order Block (the consolidation zone becomes support)
            buffer = cons_mid * self.p.buffer
            ob = OrderBlock(
                ob_type='bullish',
                midpoint=cons_mid,
                upper=self._consolidation_high + buffer,
                lower=self._consolidation_low - buffer,
                created_bar=len(self)
            )
            self._bullish_obs.append(ob)
            self._in_consolidation = False

        elif breakout_down:
            # Create bearish Order Block (the consolidation zone becomes resistance)
            buffer = cons_mid * self.p.buffer
            ob = OrderBlock(
                ob_type='bearish',
                midpoint=cons_mid,
                upper=self._consolidation_high + buffer,
                lower=self._consolidation_low - buffer,
                created_bar=len(self)
            )
            self._bearish_obs.append(ob)
            self._in_consolidation = False

    def _update_ob_testing(self, current_price: float):
        """Mark OBs as tested when price enters the zone"""
        # Check bullish OBs
        for ob in self._bullish_obs:
            if not ob.tested and ob.contains_price(current_price):
                ob.tested = True
                ob.test_bar = len(self)

        # Check bearish OBs
        for ob in self._bearish_obs:
            if not ob.tested and ob.contains_price(current_price):
                ob.tested = True
                ob.test_bar = len(self)

    def _cleanup_old_obs(self):
        """Remove old untested OBs beyond max age"""
        current_bar = len(self)
        max_age = self.p.max_age_bars

        # Clean bullish OBs
        self._bullish_obs = [
            ob for ob in self._bullish_obs
            if ob.tested or (current_bar - ob.created_bar) <= max_age
        ]

        # Clean bearish OBs
        self._bearish_obs = [
            ob for ob in self._bearish_obs
            if ob.tested or (current_bar - ob.created_bar) <= max_age
        ]

    def _set_nearest_obs(self, current_price: float):
        """Find and set nearest untested OBs"""
        # Find nearest bullish OB (below current price, untested)
        bullish_below = [
            ob for ob in self._bullish_obs
            if not ob.tested and ob.midpoint < current_price
        ]
        if bullish_below:
            nearest_bullish = max(bullish_below, key=lambda x: x.midpoint)
            self.l.ob_bullish_mid[0] = nearest_bullish.midpoint
            self.l.ob_bullish_upper[0] = nearest_bullish.upper
            self.l.ob_bullish_lower[0] = nearest_bullish.lower

        # Find nearest bearish OB (above current price, untested)
        bearish_above = [
            ob for ob in self._bearish_obs
            if not ob.tested and ob.midpoint > current_price
        ]
        if bearish_above:
            nearest_bearish = min(bearish_above, key=lambda x: x.midpoint)
            self.l.ob_bearish_mid[0] = nearest_bearish.midpoint
            self.l.ob_bearish_upper[0] = nearest_bearish.upper
            self.l.ob_bearish_lower[0] = nearest_bearish.lower

    def _check_at_ob_zone(self, current_price: float):
        """Check if current price is at an OB zone"""
        # Check bullish OBs
        for ob in self._bullish_obs:
            if not ob.tested and ob.contains_price(current_price):
                self.l.at_bullish_ob[0] = 1
                break

        # Check bearish OBs
        for ob in self._bearish_obs:
            if not ob.tested and ob.contains_price(current_price):
                self.l.at_bearish_ob[0] = 1
                break

    # Helper methods for strategy access
    def get_active_bullish_obs(self) -> List[OrderBlock]:
        """Get list of untested bullish OBs"""
        return [ob for ob in self._bullish_obs if not ob.tested]

    def get_active_bearish_obs(self) -> List[OrderBlock]:
        """Get list of untested bearish OBs"""
        return [ob for ob in self._bearish_obs if not ob.tested]

    def get_nearest_bullish_ob(self, price: float) -> Optional[OrderBlock]:
        """Get nearest bullish OB below current price"""
        bullish_below = [
            ob for ob in self._bullish_obs
            if not ob.tested and ob.midpoint < price
        ]
        if bullish_below:
            return max(bullish_below, key=lambda x: x.midpoint)
        return None

    def get_nearest_bearish_ob(self, price: float) -> Optional[OrderBlock]:
        """Get nearest bearish OB above current price"""
        bearish_above = [
            ob for ob in self._bearish_obs
            if not ob.tested and ob.midpoint > price
        ]
        if bearish_above:
            return min(bearish_above, key=lambda x: x.midpoint)
        return None

    def is_price_above_bullish_ob(self, price: float, buffer_pct: float = 0.005) -> bool:
        """
        Check if price is above a bullish OB midpoint with buffer.

        Used for HTF filter validation.
        """
        nearest = self.get_nearest_bullish_ob(price)
        if nearest:
            threshold = nearest.midpoint * (1 + buffer_pct)
            return price > threshold
        return False

    def is_price_below_bearish_ob(self, price: float, buffer_pct: float = 0.005) -> bool:
        """
        Check if price is below a bearish OB midpoint with buffer.

        Used for HTF filter validation.
        """
        nearest = self.get_nearest_bearish_ob(price)
        if nearest:
            threshold = nearest.midpoint * (1 - buffer_pct)
            return price < threshold
        return False
