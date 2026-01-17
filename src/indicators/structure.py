"""
Price Structure Module

Detects market structure (HH/HL for bullish, LH/LL for bearish)
using swing point analysis for multi-timeframe trading strategy.
"""
import backtrader as bt
import numpy as np
from typing import Tuple, List, Optional
from collections import deque
from dataclasses import dataclass
from enum import IntEnum


class StructureType(IntEnum):
    """Enumeration for structure point types"""
    NONE = 0
    HH = 1   # Higher High
    HL = 2   # Higher Low
    LH = -1  # Lower High
    LL = -2  # Lower Low


@dataclass
class SwingPoint:
    """Container for swing point data"""
    bar_index: int
    price: float
    point_type: str  # 'high' or 'low'
    structure: StructureType = StructureType.NONE


class StructureAnalyzer(bt.Indicator):
    """
    Market structure analyzer detecting:
    - Swing highs and lows
    - HH/HL/LH/LL classification
    - Trend state (bullish/bearish/neutral)
    - Structure breaks

    Lines:
        swing_high: Price of detected swing high (nan if not)
        swing_low: Price of detected swing low (nan if not)
        structure: Current structure type (HH=1, HL=2, LH=-1, LL=-2)
        trend: Current trend state (1=bullish, -1=bearish, 0=neutral)
        hh_count: Consecutive HH count
        hl_count: Consecutive HL count
        lh_count: Consecutive LH count
        ll_count: Consecutive LL count
        structure_break: 1 if bullish break, -1 if bearish break, 0 none
    """

    lines = (
        'swing_high',
        'swing_low',
        'structure',
        'trend',
        'hh_count',
        'hl_count',
        'lh_count',
        'll_count',
        'structure_break',
        'last_swing_high',
        'last_swing_low',
    )

    params = (
        ('lookback', 5),           # Bars to look back for swing detection
        ('threshold', 0.005),      # Minimum % move for valid structure
        ('consecutive_req', 3),    # Consecutive HH/HL or LH/LL required
    )

    def __init__(self):
        # Swing point tracking
        self._swing_highs: deque = deque(maxlen=20)
        self._swing_lows: deque = deque(maxlen=20)

        # Structure tracking
        self._hh_count = 0
        self._hl_count = 0
        self._lh_count = 0
        self._ll_count = 0

        # Last confirmed swing levels
        self._last_swing_high = None
        self._last_swing_low = None

        # Trend state
        self._trend = 0  # 1=bullish, -1=bearish, 0=neutral

    def next(self):
        # Initialize lines
        self.l.swing_high[0] = float('nan')
        self.l.swing_low[0] = float('nan')
        self.l.structure[0] = 0
        self.l.structure_break[0] = 0

        # Need enough bars
        if len(self) < self.p.lookback * 2 + 1:
            self._set_output_lines()
            return

        # Detect swing points (with delay for confirmation)
        self._detect_swing_high()
        self._detect_swing_low()

        # Check for structure breaks
        self._check_structure_break()

        # Set output lines
        self._set_output_lines()

    def _detect_swing_high(self):
        """Detect swing high at lookback position"""
        lookback = self.p.lookback
        center_idx = -lookback - 1

        # Check if center bar is higher than all surrounding bars
        center_high = self.data.high[center_idx]
        is_swing_high = True

        for i in range(-lookback * 2, 0):
            if i == center_idx:
                continue
            if self.data.high[i] >= center_high:
                is_swing_high = False
                break

        if is_swing_high:
            bar_index = len(self) + center_idx
            swing_point = SwingPoint(
                bar_index=bar_index,
                price=center_high,
                point_type='high'
            )

            # Classify structure
            if self._swing_highs:
                prev_high = self._swing_highs[-1].price
                pct_change = (center_high - prev_high) / prev_high

                if pct_change >= self.p.threshold:
                    swing_point.structure = StructureType.HH
                    self._hh_count += 1
                    self._lh_count = 0  # Reset LH
                elif pct_change <= -self.p.threshold:
                    swing_point.structure = StructureType.LH
                    self._lh_count += 1
                    self._hh_count = 0  # Reset HH

            self._swing_highs.append(swing_point)
            self._last_swing_high = center_high
            self.l.swing_high[center_idx] = center_high
            self.l.structure[center_idx] = swing_point.structure

    def _detect_swing_low(self):
        """Detect swing low at lookback position"""
        lookback = self.p.lookback
        center_idx = -lookback - 1

        # Check if center bar is lower than all surrounding bars
        center_low = self.data.low[center_idx]
        is_swing_low = True

        for i in range(-lookback * 2, 0):
            if i == center_idx:
                continue
            if self.data.low[i] <= center_low:
                is_swing_low = False
                break

        if is_swing_low:
            bar_index = len(self) + center_idx
            swing_point = SwingPoint(
                bar_index=bar_index,
                price=center_low,
                point_type='low'
            )

            # Classify structure
            if self._swing_lows:
                prev_low = self._swing_lows[-1].price
                pct_change = (center_low - prev_low) / prev_low

                if pct_change >= self.p.threshold:
                    swing_point.structure = StructureType.HL
                    self._hl_count += 1
                    self._ll_count = 0  # Reset LL
                elif pct_change <= -self.p.threshold:
                    swing_point.structure = StructureType.LL
                    self._ll_count += 1
                    self._hl_count = 0  # Reset HL

            self._swing_lows.append(swing_point)
            self._last_swing_low = center_low
            self.l.swing_low[center_idx] = center_low

            # Update structure line (prioritize low structure if on same bar)
            if self.l.structure[center_idx] == 0:
                self.l.structure[center_idx] = swing_point.structure

    def _check_structure_break(self):
        """Check for structure invalidation (break)"""
        current_close = self.data.close[0]

        # Bearish structure break: Close below last swing low
        if self._last_swing_low is not None:
            if current_close < self._last_swing_low:
                self.l.structure_break[0] = -1
                # Reset bullish counts
                self._hh_count = 0
                self._hl_count = 0

        # Bullish structure break: Close above last swing high
        if self._last_swing_high is not None:
            if current_close > self._last_swing_high:
                self.l.structure_break[0] = 1
                # Reset bearish counts
                self._lh_count = 0
                self._ll_count = 0

    def _set_output_lines(self):
        """Set output lines with current state"""
        self.l.hh_count[0] = self._hh_count
        self.l.hl_count[0] = self._hl_count
        self.l.lh_count[0] = self._lh_count
        self.l.ll_count[0] = self._ll_count

        # Update last swing levels
        self.l.last_swing_high[0] = self._last_swing_high if self._last_swing_high else float('nan')
        self.l.last_swing_low[0] = self._last_swing_low if self._last_swing_low else float('nan')

        # Determine trend
        self._update_trend()
        self.l.trend[0] = self._trend

    def _update_trend(self):
        """Update trend state based on structure counts"""
        req = self.p.consecutive_req

        # Bullish: Both HH and HL counts meet requirement
        if self._hh_count >= req and self._hl_count >= req:
            self._trend = 1
        # Bearish: Both LH and LL counts meet requirement
        elif self._lh_count >= req and self._ll_count >= req:
            self._trend = -1
        # Mixed or neutral
        else:
            # Keep previous trend if counts are building
            if self._hh_count >= 1 and self._hl_count >= 1:
                if self._trend != -1:
                    self._trend = 0  # Becoming bullish
            elif self._lh_count >= 1 and self._ll_count >= 1:
                if self._trend != 1:
                    self._trend = 0  # Becoming bearish

    # Helper methods for strategy access
    def is_bullish(self) -> bool:
        """Check if current structure is bullish"""
        return self._trend == 1

    def is_bearish(self) -> bool:
        """Check if current structure is bearish"""
        return self._trend == -1

    def get_swing_points(self) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """Get recent swing highs and lows"""
        return list(self._swing_highs), list(self._swing_lows)

    def get_consecutive_counts(self) -> dict:
        """Get all consecutive structure counts"""
        return {
            'HH': self._hh_count,
            'HL': self._hl_count,
            'LH': self._lh_count,
            'LL': self._ll_count,
        }

    def get_last_swing_high(self) -> Optional[float]:
        """Get price of last swing high"""
        return self._last_swing_high

    def get_last_swing_low(self) -> Optional[float]:
        """Get price of last swing low"""
        return self._last_swing_low

    def get_last_impulse_leg(self, direction: str = 'long') -> Tuple[float, float]:
        """
        Get the last impulse leg for pullback calculation.

        Args:
            direction: 'long' for HL->HH, 'short' for LH->LL

        Returns:
            Tuple of (start_price, end_price)
        """
        if direction == 'long':
            if len(self._swing_lows) > 0 and len(self._swing_highs) > 0:
                last_hl = self._swing_lows[-1].price
                last_hh = self._swing_highs[-1].price
                return (last_hl, last_hh)
        else:  # short
            if len(self._swing_highs) > 0 and len(self._swing_lows) > 0:
                last_lh = self._swing_highs[-1].price
                last_ll = self._swing_lows[-1].price
                return (last_lh, last_ll)

        return (0, 0)

    def calculate_retracement(self, current_price: float, direction: str = 'long') -> float:
        """
        Calculate retracement percentage of current price from last impulse leg.

        Args:
            current_price: Current price
            direction: 'long' or 'short'

        Returns:
            Retracement percentage (0.0 to 1.0+)
        """
        start, end = self.get_last_impulse_leg(direction)
        if start == 0 or end == 0:
            return 0.0

        leg_size = abs(end - start)
        if leg_size == 0:
            return 0.0

        if direction == 'long':
            # For long, retracement = (HH - current) / (HH - HL)
            return (end - current_price) / leg_size
        else:
            # For short, retracement = (current - LL) / (LH - LL)
            return (current_price - end) / leg_size
