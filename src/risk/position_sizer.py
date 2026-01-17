"""
Position Sizer Module

Risk-based position sizing for the swing trading strategy.
Implements fixed percentage risk with leverage support.
"""
import yaml
from dataclasses import dataclass
from typing import Optional


@dataclass
class PositionSize:
    """Container for position sizing results"""
    size: float           # Number of contracts/units
    notional_value: float # Total position value
    margin_required: float # Margin used (notional / leverage)
    risk_amount: float    # Dollar risk per trade
    risk_distance: float  # Distance to stop loss
    leverage_used: float  # Actual leverage applied


class PositionSizer:
    """
    Risk-based position sizer for futures trading.

    Calculates position size based on:
    - Account balance
    - Risk percentage per trade
    - Stop loss distance
    - Leverage limits

    Formula:
    Position Size = (Account Balance × Risk%) / |Entry Price - Stop Loss|
    Notional Position = Position Size × Entry Price
    With Leverage: Max Position = Account Balance × Leverage / Entry Price
    """

    def __init__(self, config_path: str = "config/parameters.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.risk_per_trade = self.config['risk']['risk_per_trade']
        self.max_leverage = self.config['general']['leverage']

    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
        leverage: Optional[float] = None
    ) -> PositionSize:
        """
        Calculate position size based on risk parameters.

        Args:
            account_balance: Current account equity
            entry_price: Planned entry price
            stop_loss: Stop loss price
            leverage: Leverage to use (default from config)

        Returns:
            PositionSize object with sizing details
        """
        leverage = leverage or self.max_leverage

        # Calculate risk distance (as percentage)
        risk_distance = abs(entry_price - stop_loss)
        risk_distance_pct = risk_distance / entry_price

        # Risk amount in dollars
        risk_amount = account_balance * self.risk_per_trade

        # Position size based on risk
        # Size (in base currency) = Risk Amount / Risk Distance
        if risk_distance > 0:
            position_size = risk_amount / risk_distance
        else:
            position_size = 0

        # Calculate notional value
        notional_value = position_size * entry_price

        # Check leverage constraint
        max_notional = account_balance * leverage
        if notional_value > max_notional:
            # Reduce position to fit leverage limit
            notional_value = max_notional
            position_size = notional_value / entry_price

        # Actual margin required
        margin_required = notional_value / leverage

        # Actual leverage used
        leverage_used = notional_value / account_balance if account_balance > 0 else 0

        return PositionSize(
            size=position_size,
            notional_value=notional_value,
            margin_required=margin_required,
            risk_amount=risk_amount,
            risk_distance=risk_distance,
            leverage_used=leverage_used
        )

    def calculate_pnl(
        self,
        entry_price: float,
        exit_price: float,
        position_size: float,
        direction: str  # 'long' or 'short'
    ) -> float:
        """
        Calculate profit/loss for a trade.

        Args:
            entry_price: Entry price
            exit_price: Exit price
            position_size: Position size
            direction: 'long' or 'short'

        Returns:
            PnL in dollars
        """
        if direction == 'long':
            pnl = (exit_price - entry_price) * position_size
        else:  # short
            pnl = (entry_price - exit_price) * position_size

        return pnl

    def calculate_r_multiple(
        self,
        entry_price: float,
        exit_price: float,
        stop_loss: float,
        direction: str  # 'long' or 'short'
    ) -> float:
        """
        Calculate R-multiple (risk-adjusted return).

        R = 1 means you made 1R (your original risk)
        R = 2 means you made 2R (twice your original risk)
        R = -1 means you lost 1R (your stop loss was hit)

        Args:
            entry_price: Entry price
            exit_price: Exit price
            stop_loss: Stop loss price
            direction: 'long' or 'short'

        Returns:
            R-multiple
        """
        risk_distance = abs(entry_price - stop_loss)
        if risk_distance == 0:
            return 0

        if direction == 'long':
            profit_distance = exit_price - entry_price
        else:  # short
            profit_distance = entry_price - exit_price

        return profit_distance / risk_distance

    def check_margin_call(
        self,
        account_balance: float,
        position_value: float,
        unrealized_pnl: float,
        margin_call_level: float = 0.10  # 10% of position value
    ) -> bool:
        """
        Check if position should be force-closed due to margin.

        Args:
            account_balance: Current account equity
            position_value: Current position notional value
            unrealized_pnl: Current unrealized P&L
            margin_call_level: Equity threshold (default 10%)

        Returns:
            True if margin call triggered
        """
        current_equity = account_balance + unrealized_pnl
        min_equity = position_value * margin_call_level

        return current_equity < min_equity


class BacktraderSizer:
    """
    Backtrader-compatible sizer wrapper.

    Use with: cerebro.addsizer(BacktraderSizer.get_sizer_class())
    """

    @staticmethod
    def get_sizer_class():
        """Return a Backtrader Sizer class"""
        import backtrader as bt

        class RiskBasedSizer(bt.Sizer):
            params = (
                ('risk_per_trade', 0.02),
                ('leverage', 10),
            )

            def _getsizing(self, comminfo, cash, data, isbuy):
                # Get current price
                price = data.close[0]

                # Get strategy's stop loss distance (if available)
                strategy = self.strategy
                if hasattr(strategy, 'stop_loss') and strategy.stop_loss:
                    sl_distance = abs(price - strategy.stop_loss)
                else:
                    # Default to 2% stop
                    sl_distance = price * 0.02

                if sl_distance == 0:
                    return 0

                # Calculate risk amount
                risk_amount = cash * self.p.risk_per_trade

                # Position size
                size = risk_amount / sl_distance

                # Apply leverage constraint
                max_size = (cash * self.p.leverage) / price
                size = min(size, max_size)

                return int(size)

        return RiskBasedSizer
