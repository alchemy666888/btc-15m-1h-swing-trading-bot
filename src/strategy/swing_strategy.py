"""
Multi-timeframe Swing Trading Strategy for Backtrader

Main strategy class that orchestrates:
- HTF (1H) trend filtering
- LTF (15min) entry triggers
- Risk-based position sizing
- Trade management (SL/TP)
"""
import backtrader as bt
import yaml
from datetime import datetime
from typing import Optional, List
from dataclasses import dataclass, field

from src.indicators.macd_analyzer import MACDAnalyzer
from src.indicators.structure import StructureAnalyzer
from src.indicators.order_blocks import OrderBlockAnalyzer
from src.strategy.htf_filter import HTFFilter, HTFState
from src.strategy.ltf_trigger import LTFTrigger, EntrySignal
from src.risk.position_sizer import PositionSizer


@dataclass
class TradeRecord:
    """Container for trade log entries"""
    entry_datetime: datetime
    exit_datetime: Optional[datetime] = None
    direction: str = ""  # 'long' or 'short'
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    size: float = 0.0
    notional_value: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    r_multiple: float = 0.0
    exit_reason: str = ""  # 'stop_loss', 'take_profit', 'invalidation', 'manual'
    htf_state: str = ""
    macd_at_entry: float = 0.0


class SwingStrategy(bt.Strategy):
    """
    Multi-timeframe swing trading strategy.

    Data Feeds Expected:
    - data0: 15min (primary LTF)
    - data1: 1H (HTF)
    - data2: Daily (for Order Blocks)

    Entry Logic:
    1. HTF (1H) validates trend direction
    2. LTF (15min) generates entry trigger
    3. Risk-based position sizing

    Exit Logic:
    1. Stop loss hit
    2. Take profit hit
    3. Structure invalidation
    4. Opposing HTF signal
    """

    params = (
        ('config_path', 'config/parameters.yaml'),
        ('printlog', True),
    )

    def __init__(self):
        # Load configuration
        with open(self.p.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize position sizer
        self.position_sizer = PositionSizer(self.p.config_path)

        # Initialize HTF filter and LTF trigger
        self.htf_filter = HTFFilter(self.p.config_path)
        self.ltf_trigger = LTFTrigger(self.p.config_path)

        # Data references
        self.data_ltf = self.datas[0]  # 15min
        self.data_htf = self.datas[1] if len(self.datas) > 1 else self.datas[0]  # 1H
        self.data_daily = self.datas[2] if len(self.datas) > 2 else self.data_htf  # Daily

        # LTF Indicators (15min)
        self.ltf_macd = MACDAnalyzer(
            self.data_ltf,
            fast_period=self.config['macd']['fast_period'],
            slow_period=self.config['macd']['slow_period'],
            signal_period=self.config['macd']['signal_period'],
            histogram_seq_min=self.config['macd']['histogram_sequence_min'],
            histogram_seq_max=self.config['macd']['histogram_sequence_max'],
            magnitude_threshold=self.config['macd']['magnitude_change_threshold'],
        )

        self.ltf_structure = StructureAnalyzer(
            self.data_ltf,
            lookback=self.config['structure']['ltf_lookback'],
            threshold=self.config['structure']['ltf_threshold'],
            consecutive_req=self.config['structure']['ltf_consecutive_required'],
        )

        self.ltf_ob = OrderBlockAnalyzer(
            self.data_ltf,
            consolidation_bars=self.config['order_blocks']['consolidation_bars'],
            range_threshold=self.config['order_blocks']['range_threshold'],
            breakout_threshold=self.config['order_blocks']['breakout_threshold'],
            buffer=self.config['order_blocks']['buffer'],
            max_age_bars=self.config['order_blocks']['max_age_bars'],
        )

        # HTF Indicators (1H)
        self.htf_macd = MACDAnalyzer(
            self.data_htf,
            fast_period=self.config['macd']['fast_period'],
            slow_period=self.config['macd']['slow_period'],
            signal_period=self.config['macd']['signal_period'],
        )

        self.htf_structure = StructureAnalyzer(
            self.data_htf,
            lookback=self.config['structure']['htf_lookback'],
            threshold=self.config['structure']['htf_threshold'],
            consecutive_req=self.config['structure']['htf_consecutive_required'],
        )

        # Daily OB Analyzer
        self.daily_ob = OrderBlockAnalyzer(
            self.data_daily,
            consolidation_bars=self.config['order_blocks']['consolidation_bars'],
            range_threshold=self.config['order_blocks']['range_threshold'],
            breakout_threshold=self.config['order_blocks']['breakout_threshold'],
        )

        # Volume MA for confirmation
        self.volume_ma = bt.indicators.SMA(
            self.data_ltf.volume,
            period=self.config['filters']['volume_ma_period']
        )

        # ATR for volatility filter
        self.atr = bt.indicators.ATR(
            self.data_ltf,
            period=self.config['filters']['atr_period']
        )
        self.atr_ma = bt.indicators.SMA(
            self.atr,
            period=self.config['filters']['atr_period'] * 2
        )

        # Track HTF bar count for detecting new 1H bar
        self._htf_bar_count = 0

        # Order tracking
        self.order = None
        self.stop_order = None
        self.tp_order = None

        # Position tracking
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.trade_direction = None
        self.position_size = None
        self.current_trade: Optional[TradeRecord] = None

        # Trade log
        self.trade_log: List[TradeRecord] = []

    def log(self, txt, dt=None):
        """Logging function"""
        if self.p.printlog:
            dt = dt or self.data_ltf.datetime.datetime(0)
            print(f'{dt.isoformat()} {txt}')

    def notify_order(self, order):
        """Handle order notifications"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED @ {order.executed.price:.2f}')
            else:
                self.log(f'SELL EXECUTED @ {order.executed.price:.2f}')

            # Track entry
            if self.order == order:
                self.entry_price = order.executed.price
                self.order = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected: {order.status}')
            self.order = None

    def notify_trade(self, trade):
        """Handle trade notifications"""
        if not trade.isclosed:
            return

        # Record trade results
        if self.current_trade:
            self.current_trade.exit_datetime = self.data_ltf.datetime.datetime(0)
            self.current_trade.exit_price = trade.price
            self.current_trade.pnl = trade.pnl
            self.current_trade.pnl_pct = (trade.pnl / self.entry_price) * 100 if self.entry_price else 0

            # Calculate R-multiple
            if self.stop_loss and self.entry_price:
                self.current_trade.r_multiple = self.position_sizer.calculate_r_multiple(
                    self.entry_price,
                    trade.price,
                    self.stop_loss,
                    self.current_trade.direction
                )

            self.trade_log.append(self.current_trade)

            self.log(f'TRADE CLOSED - PnL: ${trade.pnl:.2f} ({self.current_trade.pnl_pct:.2f}%) R: {self.current_trade.r_multiple:.2f}')

        # Reset tracking
        self.current_trade = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.trade_direction = None

    def next(self):
        """Main strategy logic executed on each 15min bar"""
        # Check if new HTF bar closed
        htf_bar_closed = len(self.data_htf) > self._htf_bar_count
        if htf_bar_closed:
            self._htf_bar_count = len(self.data_htf)
            self._update_htf_filter()

        # Skip if pending order
        if self.order:
            return

        # Get current state
        current_price = self.data_ltf.close[0]
        current_volume = self.data_ltf.volume[0]
        volume_ma = self.volume_ma[0]

        # Check volatility filter
        if self._is_high_volatility():
            return

        # Manage existing position
        if self.position:
            self._manage_position(current_price)
        else:
            # Look for new entry
            self._check_entry(current_price, current_volume, volume_ma)

    def _update_htf_filter(self):
        """Update HTF filter on 1H bar close"""
        current_price = self.data_htf.close[0]
        structure_break = self.htf_structure.l.structure_break[0] != 0

        self.htf_filter.update(
            structure_analyzer=self.htf_structure,
            order_block_analyzer=self.daily_ob,
            macd_analyzer=self.htf_macd,
            current_price=current_price,
            structure_break=structure_break
        )

    def _is_high_volatility(self) -> bool:
        """Check if current volatility is too high"""
        if len(self.atr) < 2 or len(self.atr_ma) < 1:
            return False

        multiplier = self.config['filters']['high_volatility_multiplier']
        return self.atr[0] > self.atr_ma[0] * multiplier

    def _check_entry(self, current_price: float, volume: float, volume_ma: float):
        """Check for entry signals"""
        # Get HTF state
        htf_state = None
        if self.htf_filter.is_bullish:
            htf_state = 'bullish'
        elif self.htf_filter.is_bearish:
            htf_state = 'bearish'

        if htf_state is None:
            return

        # Check LTF trigger
        signal = self.ltf_trigger.check_entry(
            htf_state=htf_state,
            structure_analyzer=self.ltf_structure,
            macd_analyzer=self.ltf_macd,
            order_block_analyzer=self.ltf_ob,
            current_price=current_price,
            volume=volume,
            volume_ma=volume_ma
        )

        if signal == EntrySignal.LONG:
            self._enter_long(current_price)
        elif signal == EntrySignal.SHORT:
            self._enter_short(current_price)

    def _enter_long(self, current_price: float):
        """Execute long entry"""
        # Calculate stop loss and take profit
        stop_loss = self.ltf_trigger.calculate_stop_loss(
            'long', self.ltf_structure, self.ltf_ob, current_price
        )
        take_profit = self.ltf_trigger.calculate_take_profit(
            'long', self.htf_structure, current_price
        )

        # Calculate position size
        account_value = self.broker.getvalue()
        pos_size = self.position_sizer.calculate_position_size(
            account_value, current_price, stop_loss
        )

        if pos_size.size <= 0:
            return

        # Store trade info
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trade_direction = 'long'
        self.position_size = pos_size.size

        # Create trade record
        self.current_trade = TradeRecord(
            entry_datetime=self.data_ltf.datetime.datetime(0),
            direction='long',
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            size=pos_size.size,
            notional_value=pos_size.notional_value,
            htf_state=self.htf_filter.get_state_string(),
            macd_at_entry=self.ltf_macd.l.macd[0]
        )

        self.log(f'LONG SIGNAL - Entry: {current_price:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}, Size: {pos_size.size:.4f}')

        # Execute entry
        self.order = self.buy(size=pos_size.size)

    def _enter_short(self, current_price: float):
        """Execute short entry"""
        # Calculate stop loss and take profit
        stop_loss = self.ltf_trigger.calculate_stop_loss(
            'short', self.ltf_structure, self.ltf_ob, current_price
        )
        take_profit = self.ltf_trigger.calculate_take_profit(
            'short', self.htf_structure, current_price
        )

        # Calculate position size
        account_value = self.broker.getvalue()
        pos_size = self.position_sizer.calculate_position_size(
            account_value, current_price, stop_loss
        )

        if pos_size.size <= 0:
            return

        # Store trade info
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trade_direction = 'short'
        self.position_size = pos_size.size

        # Create trade record
        self.current_trade = TradeRecord(
            entry_datetime=self.data_ltf.datetime.datetime(0),
            direction='short',
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            size=pos_size.size,
            notional_value=pos_size.notional_value,
            htf_state=self.htf_filter.get_state_string(),
            macd_at_entry=self.ltf_macd.l.macd[0]
        )

        self.log(f'SHORT SIGNAL - Entry: {current_price:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}, Size: {pos_size.size:.4f}')

        # Execute entry
        self.order = self.sell(size=pos_size.size)

    def _manage_position(self, current_price: float):
        """Manage open position - check SL/TP and invalidation"""
        if self.trade_direction == 'long':
            # Check stop loss
            if current_price <= self.stop_loss:
                self.log(f'LONG STOP LOSS HIT @ {current_price:.2f}')
                if self.current_trade:
                    self.current_trade.exit_reason = 'stop_loss'
                self.order = self.close()
                return

            # Check take profit
            if current_price >= self.take_profit:
                self.log(f'LONG TAKE PROFIT HIT @ {current_price:.2f}')
                if self.current_trade:
                    self.current_trade.exit_reason = 'take_profit'
                self.order = self.close()
                return

            # Check structure invalidation
            if self.ltf_structure.l.structure_break[0] == -1:
                self.log(f'LONG STRUCTURE INVALIDATED @ {current_price:.2f}')
                if self.current_trade:
                    self.current_trade.exit_reason = 'invalidation'
                self.order = self.close()
                return

            # Check opposing HTF signal
            if self.htf_filter.is_bearish:
                self.log(f'HTF TURNED BEARISH - Closing Long @ {current_price:.2f}')
                if self.current_trade:
                    self.current_trade.exit_reason = 'htf_reversal'
                self.order = self.close()
                return

        elif self.trade_direction == 'short':
            # Check stop loss
            if current_price >= self.stop_loss:
                self.log(f'SHORT STOP LOSS HIT @ {current_price:.2f}')
                if self.current_trade:
                    self.current_trade.exit_reason = 'stop_loss'
                self.order = self.close()
                return

            # Check take profit
            if current_price <= self.take_profit:
                self.log(f'SHORT TAKE PROFIT HIT @ {current_price:.2f}')
                if self.current_trade:
                    self.current_trade.exit_reason = 'take_profit'
                self.order = self.close()
                return

            # Check structure invalidation
            if self.ltf_structure.l.structure_break[0] == 1:
                self.log(f'SHORT STRUCTURE INVALIDATED @ {current_price:.2f}')
                if self.current_trade:
                    self.current_trade.exit_reason = 'invalidation'
                self.order = self.close()
                return

            # Check opposing HTF signal
            if self.htf_filter.is_bullish:
                self.log(f'HTF TURNED BULLISH - Closing Short @ {current_price:.2f}')
                if self.current_trade:
                    self.current_trade.exit_reason = 'htf_reversal'
                self.order = self.close()
                return

    def stop(self):
        """Called when backtest ends"""
        final_value = self.broker.getvalue()
        initial_value = self.config['general']['initial_capital']
        total_return = ((final_value - initial_value) / initial_value) * 100

        self.log(f'Final Portfolio Value: ${final_value:.2f}')
        self.log(f'Total Return: {total_return:.2f}%')
        self.log(f'Total Trades: {len(self.trade_log)}')
