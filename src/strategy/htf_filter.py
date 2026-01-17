"""
HTF Filter Module (1H Timeframe)

Validates trend conditions before enabling LTF entries.
Combines structure analysis, Order Block positioning, and MACD divergence/reversal.
"""
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import yaml


class HTFState(Enum):
    """HTF filter state enumeration"""
    NEUTRAL = 0
    BULLISH_VALID = 1
    BEARISH_VALID = -1
    COOLDOWN = 2


@dataclass
class HTFConditions:
    """Container for HTF validation conditions"""
    # Structure conditions
    structure_valid: bool = False
    hh_count: int = 0
    hl_count: int = 0
    lh_count: int = 0
    ll_count: int = 0

    # Order Block conditions
    ob_valid: bool = False
    price_vs_ob: str = ""  # 'above_bullish', 'below_bearish', 'invalid'

    # MACD conditions
    macd_valid: bool = False
    histogram_sequence: int = 0
    histogram_direction: int = 0
    magnitude_narrowing: bool = False
    reversal_signal: bool = False
    divergence: int = 0  # 1=bullish, -1=bearish, 0=none

    # Price behavior during histogram sequence
    price_drawdown_pct: float = 0.0

    def is_bullish_valid(self, config: dict) -> bool:
        """Check if all bullish conditions are met"""
        req = config['structure']['htf_consecutive_required']
        max_drawdown = config['macd']['price_drawdown_max']

        structure_ok = self.hh_count >= req and self.hl_count >= req
        ob_ok = self.price_vs_ob == 'above_bullish'
        macd_ok = (
            self.reversal_signal or
            self.divergence == 1 or
            (self.magnitude_narrowing and self.histogram_direction == -1)
        )
        drawdown_ok = self.price_drawdown_pct <= max_drawdown

        return structure_ok and ob_ok and macd_ok and drawdown_ok

    def is_bearish_valid(self, config: dict) -> bool:
        """Check if all bearish conditions are met"""
        req = config['structure']['htf_consecutive_required']
        max_drawdown = config['macd']['price_drawdown_max']

        structure_ok = self.lh_count >= req and self.ll_count >= req
        ob_ok = self.price_vs_ob == 'below_bearish'
        macd_ok = (
            self.reversal_signal or
            self.divergence == -1 or
            (self.magnitude_narrowing and self.histogram_direction == 1)
        )
        drawdown_ok = self.price_drawdown_pct <= max_drawdown

        return structure_ok and ob_ok and macd_ok and drawdown_ok


class HTFFilter:
    """
    Higher Timeframe (1H) filter for swing trading strategy.

    Validates trend conditions:
    - Structure: 3+ consecutive HH/HL (bullish) or LH/LL (bearish)
    - Order Block: Price above bullish OB (long) or below bearish OB (short)
    - MACD: Divergence/reversal pattern with controlled price movement

    State management:
    - Re-evaluates on each 1H bar close
    - Implements cooldown after structure invalidation
    """

    def __init__(self, config_path: str = "config/parameters.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self._state = HTFState.NEUTRAL
        self._conditions = HTFConditions()
        self._cooldown_bars = 0
        self._invalidation_cooldown = self.config['filters']['invalidation_cooldown']

        # Price tracking for drawdown calculation
        self._sequence_start_price = None
        self._sequence_high = None
        self._sequence_low = None

    @property
    def state(self) -> HTFState:
        """Current HTF filter state"""
        return self._state

    @property
    def is_bullish(self) -> bool:
        """Check if HTF is bullish valid"""
        return self._state == HTFState.BULLISH_VALID

    @property
    def is_bearish(self) -> bool:
        """Check if HTF is bearish valid"""
        return self._state == HTFState.BEARISH_VALID

    @property
    def is_neutral(self) -> bool:
        """Check if HTF is neutral"""
        return self._state in (HTFState.NEUTRAL, HTFState.COOLDOWN)

    def update(
        self,
        structure_analyzer,
        order_block_analyzer,
        macd_analyzer,
        current_price: float,
        structure_break: bool = False
    ) -> HTFState:
        """
        Update HTF filter state based on current indicator readings.

        Args:
            structure_analyzer: StructureAnalyzer indicator instance
            order_block_analyzer: OrderBlockAnalyzer indicator instance
            macd_analyzer: MACDAnalyzer indicator instance
            current_price: Current 1H close price
            structure_break: Whether structure was broken this bar

        Returns:
            Updated HTFState
        """
        # Handle cooldown
        if self._state == HTFState.COOLDOWN:
            self._cooldown_bars -= 1
            if self._cooldown_bars <= 0:
                self._state = HTFState.NEUTRAL
            return self._state

        # Check for structure break (invalidation)
        if structure_break:
            self._enter_cooldown()
            return self._state

        # Update conditions
        self._update_conditions(
            structure_analyzer,
            order_block_analyzer,
            macd_analyzer,
            current_price
        )

        # Evaluate state
        if self._conditions.is_bullish_valid(self.config):
            self._state = HTFState.BULLISH_VALID
        elif self._conditions.is_bearish_valid(self.config):
            self._state = HTFState.BEARISH_VALID
        else:
            self._state = HTFState.NEUTRAL

        return self._state

    def _enter_cooldown(self):
        """Enter cooldown state after invalidation"""
        self._state = HTFState.COOLDOWN
        self._cooldown_bars = self._invalidation_cooldown
        self._reset_conditions()

    def _reset_conditions(self):
        """Reset all condition tracking"""
        self._conditions = HTFConditions()
        self._sequence_start_price = None
        self._sequence_high = None
        self._sequence_low = None

    def _update_conditions(
        self,
        structure_analyzer,
        order_block_analyzer,
        macd_analyzer,
        current_price: float
    ):
        """Update all condition checks"""
        # Structure conditions
        counts = structure_analyzer.get_consecutive_counts()
        self._conditions.hh_count = counts['HH']
        self._conditions.hl_count = counts['HL']
        self._conditions.lh_count = counts['LH']
        self._conditions.ll_count = counts['LL']

        req = self.config['structure']['htf_consecutive_required']
        self._conditions.structure_valid = (
            (counts['HH'] >= req and counts['HL'] >= req) or
            (counts['LH'] >= req and counts['LL'] >= req)
        )

        # Order Block conditions
        buffer = self.config['order_blocks']['buffer']

        if order_block_analyzer.is_price_above_bullish_ob(current_price, buffer):
            self._conditions.price_vs_ob = 'above_bullish'
            self._conditions.ob_valid = True
        elif order_block_analyzer.is_price_below_bearish_ob(current_price, buffer):
            self._conditions.price_vs_ob = 'below_bearish'
            self._conditions.ob_valid = True
        else:
            self._conditions.price_vs_ob = 'invalid'
            self._conditions.ob_valid = False

        # MACD conditions
        seq_count, seq_dir = macd_analyzer.get_histogram_sequence()
        self._conditions.histogram_sequence = seq_count
        self._conditions.histogram_direction = seq_dir

        self._conditions.magnitude_narrowing = macd_analyzer.is_histogram_narrowing(2)
        self._conditions.reversal_signal = macd_analyzer.l.reversal_signal[0] != 0
        self._conditions.divergence = int(macd_analyzer.l.divergence[0])

        # Track price movement during histogram sequence
        self._update_price_tracking(current_price, seq_count, seq_dir)

        # Check overall MACD validity
        min_seq = self.config['macd']['histogram_sequence_min']
        max_seq = self.config['macd']['histogram_sequence_max']

        self._conditions.macd_valid = (
            self._conditions.reversal_signal or
            self._conditions.divergence != 0 or
            (min_seq <= seq_count <= max_seq and self._conditions.magnitude_narrowing)
        )

    def _update_price_tracking(
        self,
        current_price: float,
        seq_count: int,
        seq_dir: int
    ):
        """Track price movement during histogram sequence for drawdown check"""
        if seq_count == 1:
            # New sequence started
            self._sequence_start_price = current_price
            self._sequence_high = current_price
            self._sequence_low = current_price
        elif seq_count > 1 and self._sequence_start_price is not None:
            # Update high/low
            self._sequence_high = max(self._sequence_high, current_price)
            self._sequence_low = min(self._sequence_low, current_price)

            # Calculate drawdown based on direction
            if seq_dir == -1:  # Negative histogram (potential bullish setup)
                # Looking for limited price drop during negative histogram
                drawdown = (self._sequence_start_price - self._sequence_low) / self._sequence_start_price
            else:  # Positive histogram (potential bearish setup)
                # Looking for limited price rise during positive histogram
                drawdown = (self._sequence_high - self._sequence_start_price) / self._sequence_start_price

            self._conditions.price_drawdown_pct = abs(drawdown)
        else:
            self._conditions.price_drawdown_pct = 0.0

    def get_conditions(self) -> HTFConditions:
        """Get current condition state for debugging/logging"""
        return self._conditions

    def get_state_string(self) -> str:
        """Get human-readable state string"""
        state_map = {
            HTFState.NEUTRAL: "NEUTRAL",
            HTFState.BULLISH_VALID: "BULLISH_VALID",
            HTFState.BEARISH_VALID: "BEARISH_VALID",
            HTFState.COOLDOWN: f"COOLDOWN ({self._cooldown_bars} bars)"
        }
        return state_map.get(self._state, "UNKNOWN")
