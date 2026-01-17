"""
LTF Trigger Module (15min Timeframe)

Generates precise entry signals when HTF filter is valid.
Implements pullback detection, structure confirmation, and MACD entry triggers.
"""
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
import yaml


class EntrySignal(Enum):
    """Entry signal types"""
    NONE = 0
    LONG = 1
    SHORT = -1


@dataclass
class LTFConditions:
    """Container for LTF entry conditions"""
    # Structure conditions
    structure_valid: bool = False
    hh_count: int = 0
    hl_count: int = 0
    lh_count: int = 0
    ll_count: int = 0

    # MACD conditions
    macd_position_valid: bool = False  # Above/below zero line
    macd_above_zero_bars: int = 0
    macd_below_zero_bars: int = 0
    histogram_expanding: bool = False
    histogram_positive_count: int = 0
    histogram_negative_count: int = 0

    # Pullback conditions
    pullback_valid: bool = False
    retracement_pct: float = 0.0
    at_support: bool = False
    at_resistance: bool = False

    # Entry confirmation
    macd_cross_up: bool = False
    macd_cross_down: bool = False
    volume_confirmed: bool = False

    def is_long_entry_valid(self, config: dict) -> bool:
        """Check if all long entry conditions are met"""
        req = config['structure']['ltf_consecutive_required']
        min_pullback = config['entry']['pullback_min']
        max_pullback = config['entry']['pullback_max']
        min_bars_above_zero = config['entry']['macd_above_zero_bars']
        min_histogram_bars = config['macd']['histogram_sequence_min']

        structure_ok = self.hh_count >= req and self.hl_count >= req
        macd_position_ok = self.macd_above_zero_bars >= min_bars_above_zero
        histogram_ok = self.histogram_positive_count >= min_histogram_bars and self.histogram_expanding
        pullback_ok = min_pullback <= self.retracement_pct <= max_pullback
        support_ok = self.at_support
        entry_ok = self.macd_cross_up

        return (structure_ok and macd_position_ok and histogram_ok and
                pullback_ok and support_ok and entry_ok)

    def is_short_entry_valid(self, config: dict) -> bool:
        """Check if all short entry conditions are met"""
        req = config['structure']['ltf_consecutive_required']
        min_pullback = config['entry']['pullback_min']
        max_pullback = config['entry']['pullback_max']
        min_bars_below_zero = config['entry']['macd_above_zero_bars']
        min_histogram_bars = config['macd']['histogram_sequence_min']

        structure_ok = self.lh_count >= req and self.ll_count >= req
        macd_position_ok = self.macd_below_zero_bars >= min_bars_below_zero
        histogram_ok = self.histogram_negative_count >= min_histogram_bars and self.histogram_expanding
        pullback_ok = min_pullback <= self.retracement_pct <= max_pullback
        resistance_ok = self.at_resistance
        entry_ok = self.macd_cross_down

        return (structure_ok and macd_position_ok and histogram_ok and
                pullback_ok and resistance_ok and entry_ok)


class LTFTrigger:
    """
    Lower Timeframe (15min) entry trigger for swing trading strategy.

    Entry conditions (Long when HTF bullish):
    - LTF Structure: 3+ consecutive HH and HL
    - MACD above zero for 3+ bars
    - Histogram: 4+ positive bars with expanding magnitude
    - Pullback: 20-50% retracement of last impulse leg
    - At support: Previous HL, trendline, or 15min OB
    - Confirmation: MACD crosses up with new positive histogram bar

    Entry conditions (Short when HTF bearish):
    - Mirror conditions for short side
    """

    def __init__(self, config_path: str = "config/parameters.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self._conditions = LTFConditions()
        self._last_signal = EntrySignal.NONE

        # MACD position tracking
        self._macd_above_zero_count = 0
        self._macd_below_zero_count = 0
        self._prev_macd = 0

        # Histogram tracking
        self._histogram_pos_count = 0
        self._histogram_neg_count = 0

    def check_entry(
        self,
        htf_state: str,  # 'bullish' or 'bearish'
        structure_analyzer,
        macd_analyzer,
        order_block_analyzer,
        current_price: float,
        volume: float,
        volume_ma: float
    ) -> EntrySignal:
        """
        Check for entry signal on 15min bar close.

        Args:
            htf_state: Current HTF filter state ('bullish' or 'bearish')
            structure_analyzer: LTF StructureAnalyzer indicator
            macd_analyzer: LTF MACDAnalyzer indicator
            order_block_analyzer: LTF OrderBlockAnalyzer indicator
            current_price: Current 15min close price
            volume: Current bar volume
            volume_ma: Volume moving average

        Returns:
            EntrySignal (NONE, LONG, or SHORT)
        """
        # Only check if HTF is valid
        if htf_state not in ('bullish', 'bearish'):
            self._last_signal = EntrySignal.NONE
            return EntrySignal.NONE

        # Update conditions
        self._update_conditions(
            htf_state,
            structure_analyzer,
            macd_analyzer,
            order_block_analyzer,
            current_price,
            volume,
            volume_ma
        )

        # Check for entry signal
        if htf_state == 'bullish' and self._conditions.is_long_entry_valid(self.config):
            self._last_signal = EntrySignal.LONG
            return EntrySignal.LONG

        elif htf_state == 'bearish' and self._conditions.is_short_entry_valid(self.config):
            self._last_signal = EntrySignal.SHORT
            return EntrySignal.SHORT

        self._last_signal = EntrySignal.NONE
        return EntrySignal.NONE

    def _update_conditions(
        self,
        htf_state: str,
        structure_analyzer,
        macd_analyzer,
        order_block_analyzer,
        current_price: float,
        volume: float,
        volume_ma: float
    ):
        """Update all LTF condition checks"""
        # Structure conditions
        counts = structure_analyzer.get_consecutive_counts()
        self._conditions.hh_count = counts['HH']
        self._conditions.hl_count = counts['HL']
        self._conditions.lh_count = counts['LH']
        self._conditions.ll_count = counts['LL']

        req = self.config['structure']['ltf_consecutive_required']
        if htf_state == 'bullish':
            self._conditions.structure_valid = (
                counts['HH'] >= req and counts['HL'] >= req
            )
        else:
            self._conditions.structure_valid = (
                counts['LH'] >= req and counts['LL'] >= req
            )

        # MACD position tracking
        current_macd = macd_analyzer.l.macd[0]
        current_histogram = macd_analyzer.l.histogram[0]

        if current_macd > 0:
            self._macd_above_zero_count += 1
            self._macd_below_zero_count = 0
        else:
            self._macd_below_zero_count += 1
            self._macd_above_zero_count = 0

        self._conditions.macd_above_zero_bars = self._macd_above_zero_count
        self._conditions.macd_below_zero_bars = self._macd_below_zero_count

        min_bars = self.config['entry']['macd_above_zero_bars']
        if htf_state == 'bullish':
            self._conditions.macd_position_valid = self._macd_above_zero_count >= min_bars
        else:
            self._conditions.macd_position_valid = self._macd_below_zero_count >= min_bars

        # Histogram tracking
        if current_histogram > 0:
            self._histogram_pos_count += 1
            self._histogram_neg_count = 0
        else:
            self._histogram_neg_count += 1
            self._histogram_pos_count = 0

        self._conditions.histogram_positive_count = self._histogram_pos_count
        self._conditions.histogram_negative_count = self._histogram_neg_count
        self._conditions.histogram_expanding = macd_analyzer.is_histogram_expanding(2)

        # MACD cross detection
        prev_macd = self._prev_macd
        prev_signal = macd_analyzer.l.signal[-1] if len(macd_analyzer) > 1 else 0
        current_signal = macd_analyzer.l.signal[0]

        self._conditions.macd_cross_up = (
            current_macd > current_signal and
            prev_macd <= prev_signal and
            current_histogram > 0
        )

        self._conditions.macd_cross_down = (
            current_macd < current_signal and
            prev_macd >= prev_signal and
            current_histogram < 0
        )

        self._prev_macd = current_macd

        # Pullback detection
        if htf_state == 'bullish':
            retracement = structure_analyzer.calculate_retracement(current_price, 'long')
            self._conditions.retracement_pct = retracement
        else:
            retracement = structure_analyzer.calculate_retracement(current_price, 'short')
            self._conditions.retracement_pct = retracement

        min_pullback = self.config['entry']['pullback_min']
        max_pullback = self.config['entry']['pullback_max']
        self._conditions.pullback_valid = min_pullback <= retracement <= max_pullback

        # Support/Resistance check
        self._conditions.at_support = self._check_at_support(
            htf_state, structure_analyzer, order_block_analyzer, current_price
        )
        self._conditions.at_resistance = self._check_at_resistance(
            htf_state, structure_analyzer, order_block_analyzer, current_price
        )

        # Volume confirmation
        vol_threshold = self.config['entry']['volume_confirmation']
        self._conditions.volume_confirmed = volume > volume_ma * vol_threshold

    def _check_at_support(
        self,
        htf_state: str,
        structure_analyzer,
        order_block_analyzer,
        current_price: float
    ) -> bool:
        """Check if price is at a support level for long entry"""
        if htf_state != 'bullish':
            return False

        # Check previous HL (within 0.5% buffer)
        last_hl = structure_analyzer.get_last_swing_low()
        if last_hl is not None:
            buffer = current_price * 0.005
            if abs(current_price - last_hl) <= buffer:
                return True

        # Check bullish OB zone
        if order_block_analyzer.l.at_bullish_ob[0] == 1:
            return True

        return False

    def _check_at_resistance(
        self,
        htf_state: str,
        structure_analyzer,
        order_block_analyzer,
        current_price: float
    ) -> bool:
        """Check if price is at a resistance level for short entry"""
        if htf_state != 'bearish':
            return False

        # Check previous LH (within 0.5% buffer)
        last_lh = structure_analyzer.get_last_swing_high()
        if last_lh is not None:
            buffer = current_price * 0.005
            if abs(current_price - last_lh) <= buffer:
                return True

        # Check bearish OB zone
        if order_block_analyzer.l.at_bearish_ob[0] == 1:
            return True

        return False

    def calculate_stop_loss(
        self,
        direction: str,  # 'long' or 'short'
        structure_analyzer,
        order_block_analyzer,
        current_price: float
    ) -> float:
        """
        Calculate stop loss price based on strategy rules.

        Long SL: min(low of last 2 bars, recent HL - buffer, daily OB low)
        Short SL: max(high of last 2 bars, recent LH + buffer, daily OB high)

        Args:
            direction: 'long' or 'short'
            structure_analyzer: LTF StructureAnalyzer
            order_block_analyzer: LTF OrderBlockAnalyzer (or daily)
            current_price: Current entry price

        Returns:
            Stop loss price
        """
        buffer_pct = self.config['risk']['sl_buffer']
        buffer = current_price * buffer_pct

        if direction == 'long':
            # Get recent HL
            last_hl = structure_analyzer.get_last_swing_low()
            sl_from_hl = (last_hl - buffer) if last_hl else current_price * 0.98

            # Get bullish OB low if nearby
            nearest_ob = order_block_analyzer.get_nearest_bullish_ob(current_price)
            sl_from_ob = nearest_ob.lower if nearest_ob else sl_from_hl

            # Use the higher (tighter) stop loss
            return max(sl_from_hl, sl_from_ob)

        else:  # short
            # Get recent LH
            last_lh = structure_analyzer.get_last_swing_high()
            sl_from_lh = (last_lh + buffer) if last_lh else current_price * 1.02

            # Get bearish OB high if nearby
            nearest_ob = order_block_analyzer.get_nearest_bearish_ob(current_price)
            sl_from_ob = nearest_ob.upper if nearest_ob else sl_from_lh

            # Use the lower (tighter) stop loss
            return min(sl_from_lh, sl_from_ob)

    def calculate_take_profit(
        self,
        direction: str,  # 'long' or 'short'
        htf_structure_analyzer,
        current_price: float
    ) -> float:
        """
        Calculate take profit price based on previous HTF swing.

        Long TP: Previous 1H swing high
        Short TP: Previous 1H swing low

        Args:
            direction: 'long' or 'short'
            htf_structure_analyzer: HTF StructureAnalyzer
            current_price: Current entry price

        Returns:
            Take profit price
        """
        if direction == 'long':
            last_swing_high = htf_structure_analyzer.get_last_swing_high()
            if last_swing_high and last_swing_high > current_price:
                return last_swing_high
            # Fallback: 2% profit target
            return current_price * 1.02

        else:  # short
            last_swing_low = htf_structure_analyzer.get_last_swing_low()
            if last_swing_low and last_swing_low < current_price:
                return last_swing_low
            # Fallback: 2% profit target
            return current_price * 0.98

    def get_conditions(self) -> LTFConditions:
        """Get current condition state for debugging/logging"""
        return self._conditions

    def reset(self):
        """Reset trigger state (called after entry or on new session)"""
        self._conditions = LTFConditions()
        self._macd_above_zero_count = 0
        self._macd_below_zero_count = 0
        self._histogram_pos_count = 0
        self._histogram_neg_count = 0
        self._prev_macd = 0
        self._last_signal = EntrySignal.NONE
