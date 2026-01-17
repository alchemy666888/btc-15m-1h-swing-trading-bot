"""
MACD Analyzer Module

Extended MACD indicator with histogram analysis, divergence detection,
and momentum state tracking for the swing trading strategy.
"""
import backtrader as bt
import numpy as np
from typing import Tuple, List, Optional
from collections import deque


class MACDAnalyzer(bt.Indicator):
    """
    Extended MACD indicator with:
    - Histogram sequence tracking
    - Magnitude change detection (narrowing/expanding)
    - Bullish/Bearish divergence detection
    - Reversal signal pattern recognition
    """

    lines = (
        'macd',
        'signal',
        'histogram',
        'hist_sequence',      # Count of consecutive same-direction bars
        'hist_magnitude',     # Bar-to-bar magnitude change
        'divergence',         # 1=bullish, -1=bearish, 0=none
        'reversal_signal',    # 1=bullish reversal, -1=bearish reversal, 0=none
    )

    params = (
        ('fast_period', 12),
        ('slow_period', 26),
        ('signal_period', 9),
        ('histogram_seq_min', 4),
        ('histogram_seq_max', 6),
        ('magnitude_threshold', 0.20),  # 20% change
        ('divergence_threshold', 0.10), # 10%
        ('divergence_lookback', 10),    # bars to look back for divergence
    )

    def __init__(self):
        # Standard MACD calculation
        ema_fast = bt.indicators.EMA(self.data.close, period=self.p.fast_period)
        ema_slow = bt.indicators.EMA(self.data.close, period=self.p.slow_period)

        self.l.macd = ema_fast - ema_slow
        self.l.signal = bt.indicators.EMA(self.l.macd, period=self.p.signal_period)
        self.l.histogram = self.l.macd - self.l.signal

        # State tracking (initialized in prenext)
        self._hist_values = deque(maxlen=20)
        self._price_highs = deque(maxlen=20)
        self._price_lows = deque(maxlen=20)
        self._hist_highs = deque(maxlen=20)
        self._hist_lows = deque(maxlen=20)

        # Sequence tracking
        self._sequence_count = 0
        self._sequence_direction = 0  # 1 for positive, -1 for negative
        self._prev_magnitudes = deque(maxlen=10)

    def next(self):
        hist = self.l.histogram[0]
        prev_hist = self.l.histogram[-1] if len(self) > 1 else 0

        # Track histogram values
        self._hist_values.append(hist)

        # Update sequence tracking
        self._update_sequence(hist, prev_hist)

        # Calculate magnitude change
        magnitude_state = self._calculate_magnitude_change(hist, prev_hist)
        self.l.hist_magnitude[0] = magnitude_state

        # Set sequence count
        self.l.hist_sequence[0] = self._sequence_count * self._sequence_direction

        # Detect divergence
        divergence = self._detect_divergence()
        self.l.divergence[0] = divergence

        # Detect reversal signal
        reversal = self._detect_reversal_signal()
        self.l.reversal_signal[0] = reversal

        # Track price and histogram extremes for divergence
        self._update_extremes()

    def _update_sequence(self, hist: float, prev_hist: float):
        """Update histogram sequence count and direction"""
        current_dir = 1 if hist > 0 else -1 if hist < 0 else 0

        if current_dir == 0:
            # Reset on zero histogram
            self._sequence_count = 0
            self._sequence_direction = 0
        elif current_dir == self._sequence_direction:
            # Same direction, increment count
            self._sequence_count += 1
        else:
            # Direction change, reset
            self._sequence_count = 1
            self._sequence_direction = current_dir

    def _calculate_magnitude_change(self, hist: float, prev_hist: float) -> float:
        """
        Calculate magnitude change state.

        Returns:
            1.0 = expanding (magnitude increasing)
            -1.0 = narrowing (magnitude decreasing)
            0.0 = neutral
        """
        if abs(prev_hist) < 1e-10:  # Avoid division by zero
            return 0.0

        magnitude_change = (abs(hist) - abs(prev_hist)) / abs(prev_hist)
        self._prev_magnitudes.append(magnitude_change)

        if magnitude_change > self.p.magnitude_threshold:
            return 1.0  # Expanding
        elif magnitude_change < -self.p.magnitude_threshold:
            return -1.0  # Narrowing
        else:
            return 0.0  # Neutral

    def _update_extremes(self):
        """Track price and histogram highs/lows for divergence detection"""
        if len(self) < 3:
            return

        # Check for local high (price)
        if (self.data.high[-1] > self.data.high[-2] and
            self.data.high[-1] > self.data.high[0]):
            self._price_highs.append((len(self) - 1, self.data.high[-1]))
            self._hist_highs.append((len(self) - 1, self.l.histogram[-1]))

        # Check for local low (price)
        if (self.data.low[-1] < self.data.low[-2] and
            self.data.low[-1] < self.data.low[0]):
            self._price_lows.append((len(self) - 1, self.data.low[-1]))
            self._hist_lows.append((len(self) - 1, self.l.histogram[-1]))

    def _detect_divergence(self) -> float:
        """
        Detect bullish or bearish divergence.

        Bullish divergence: Price makes lower low, MACD makes higher low
        Bearish divergence: Price makes higher high, MACD makes lower high

        Returns:
            1.0 = bullish divergence
            -1.0 = bearish divergence
            0.0 = no divergence
        """
        lookback = self.p.divergence_lookback

        # Need at least 2 points for comparison
        if len(self._price_lows) >= 2 and len(self._hist_lows) >= 2:
            # Check bullish divergence (lower low in price, higher low in MACD)
            price_low1, price_val1 = self._price_lows[-2]
            price_low2, price_val2 = self._price_lows[-1]

            # Check if within lookback period
            if len(self) - price_low1 <= lookback:
                hist_val1 = self._hist_lows[-2][1]
                hist_val2 = self._hist_lows[-1][1]

                # Price lower low, MACD higher low
                if price_val2 < price_val1 and hist_val2 > hist_val1:
                    price_change = (price_val1 - price_val2) / price_val1
                    if price_change >= self.p.divergence_threshold * 0.5:
                        return 1.0  # Bullish divergence

        if len(self._price_highs) >= 2 and len(self._hist_highs) >= 2:
            # Check bearish divergence (higher high in price, lower high in MACD)
            price_high1, price_val1 = self._price_highs[-2]
            price_high2, price_val2 = self._price_highs[-1]

            # Check if within lookback period
            if len(self) - price_high1 <= lookback:
                hist_val1 = self._hist_highs[-2][1]
                hist_val2 = self._hist_highs[-1][1]

                # Price higher high, MACD lower high
                if price_val2 > price_val1 and hist_val2 < hist_val1:
                    price_change = (price_val2 - price_val1) / price_val1
                    if price_change >= self.p.divergence_threshold * 0.5:
                        return -1.0  # Bearish divergence

        return 0.0

    def _detect_reversal_signal(self) -> float:
        """
        Detect histogram reversal pattern.

        Bullish reversal: 4-6 negative bars narrowing, then transition to positive
        Bearish reversal: 4-6 positive bars narrowing, then transition to negative

        Returns:
            1.0 = bullish reversal signal
            -1.0 = bearish reversal signal
            0.0 = no signal
        """
        if len(self._hist_values) < self.p.histogram_seq_min + 2:
            return 0.0

        hist_list = list(self._hist_values)
        current = hist_list[-1]
        prev = hist_list[-2]

        # Check for transition from negative to positive (bullish)
        if current > 0 and prev <= 0:
            # Look back for narrowing sequence
            neg_count = 0
            narrowing_count = 0

            for i in range(len(hist_list) - 2, max(0, len(hist_list) - self.p.histogram_seq_max - 2), -1):
                if hist_list[i] < 0:
                    neg_count += 1
                    if i > 0 and abs(hist_list[i]) < abs(hist_list[i-1]):
                        narrowing_count += 1
                else:
                    break

            if (self.p.histogram_seq_min <= neg_count <= self.p.histogram_seq_max and
                narrowing_count >= 2):
                return 1.0  # Bullish reversal

        # Check for transition from positive to negative (bearish)
        if current < 0 and prev >= 0:
            # Look back for narrowing sequence
            pos_count = 0
            narrowing_count = 0

            for i in range(len(hist_list) - 2, max(0, len(hist_list) - self.p.histogram_seq_max - 2), -1):
                if hist_list[i] > 0:
                    pos_count += 1
                    if i > 0 and abs(hist_list[i]) < abs(hist_list[i-1]):
                        narrowing_count += 1
                else:
                    break

            if (self.p.histogram_seq_min <= pos_count <= self.p.histogram_seq_max and
                narrowing_count >= 2):
                return -1.0  # Bearish reversal

        return 0.0

    # Helper methods for strategy access
    def get_histogram_sequence(self) -> Tuple[int, int]:
        """
        Get count and direction of consecutive histogram bars.

        Returns:
            Tuple of (count, direction) where direction is 1 (pos) or -1 (neg)
        """
        return self._sequence_count, self._sequence_direction

    def is_macd_above_zero(self, min_bars: int = 3) -> bool:
        """Check if MACD line has been above zero for min_bars"""
        if len(self) < min_bars:
            return False

        for i in range(min_bars):
            if self.l.macd[-i] <= 0:
                return False
        return True

    def is_macd_below_zero(self, min_bars: int = 3) -> bool:
        """Check if MACD line has been below zero for min_bars"""
        if len(self) < min_bars:
            return False

        for i in range(min_bars):
            if self.l.macd[-i] >= 0:
                return False
        return True

    def is_histogram_expanding(self, min_bars: int = 2) -> bool:
        """Check if histogram magnitude has been expanding"""
        if len(self._prev_magnitudes) < min_bars:
            return False

        return all(m > 0 for m in list(self._prev_magnitudes)[-min_bars:])

    def is_histogram_narrowing(self, min_bars: int = 2) -> bool:
        """Check if histogram magnitude has been narrowing"""
        if len(self._prev_magnitudes) < min_bars:
            return False

        return all(m < 0 for m in list(self._prev_magnitudes)[-min_bars:])
