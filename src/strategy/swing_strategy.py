"""
Multi-timeframe Swing Trading Strategy for Backtrader
Implements both bullish (long) and bearish (short) setups
"""
import backtrader as bt
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class SwingTradingStrategy(bt.Strategy):
    """
    Multi-timeframe swing trading strategy for crypto perpetual futures.

    Entry Conditions (Long):
    - Higher TF: Bullish structure (HH/HL), MACD cross up or bullish divergence
    - Lower TF: Bullish BOS, price at support zone or bullish FVG
    - Confirmation: Volume > average, RSI not overbought

    Entry Conditions (Short):
    - Higher TF: Bearish structure (LH/LL), MACD cross down or bearish divergence
    - Lower TF: Bearish BOS, price at resistance zone or bearish FVG
    - Confirmation: Volume > average, RSI not oversold
    """

    params = (
        # Risk Management
        ('risk_per_trade', config.RISK_PER_TRADE),
        ('leverage', config.LEVERAGE),
        ('risk_reward', config.RISK_REWARD_TARGET),
        ('atr_sl_mult', config.ATR_SL_MULTIPLIER),

        # Indicator Parameters
        ('ema_fast', config.EMA_FAST),
        ('ema_slow', config.EMA_SLOW),
        ('ema_trend', config.EMA_TREND),
        ('macd_fast', config.MACD_FAST),
        ('macd_slow', config.MACD_SLOW),
        ('macd_signal', config.MACD_SIGNAL),
        ('rsi_period', config.RSI_PERIOD),
        ('rsi_overbought', config.RSI_OVERBOUGHT),
        ('rsi_oversold', config.RSI_OVERSOLD),
        ('atr_period', config.ATR_PERIOD),

        # Swing Detection
        ('swing_lookback', config.SWING_LOOKBACK),
        ('break_threshold', config.BREAK_THRESHOLD),
        ('volume_mult', config.VOLUME_MULTIPLIER),

        # Trading Fees
        ('commission', config.TAKER_FEE),
        ('slippage', config.SLIPPAGE),
    )

    def __init__(self):
        # Data references
        self.data_close = self.datas[0].close
        self.data_high = self.datas[0].high
        self.data_low = self.datas[0].low
        self.data_volume = self.datas[0].volume

        # Higher timeframe reference (if available)
        self.htf_data = self.datas[1] if len(self.datas) > 1 else self.datas[0]

        # Technical Indicators - Lower Timeframe
        self.ema_fast = bt.indicators.EMA(self.data_close, period=self.p.ema_fast)
        self.ema_slow = bt.indicators.EMA(self.data_close, period=self.p.ema_slow)
        self.ema_trend = bt.indicators.EMA(self.data_close, period=self.p.ema_trend)

        self.macd = bt.indicators.MACD(
            self.data_close,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal
        )

        self.rsi = bt.indicators.RSI(self.data_close, period=self.p.rsi_period)
        self.atr = bt.indicators.ATR(self.datas[0], period=self.p.atr_period)

        self.volume_sma = bt.indicators.SMA(self.data_volume, period=20)

        # Higher Timeframe Indicators
        self.htf_ema_fast = bt.indicators.EMA(self.htf_data.close, period=self.p.ema_fast)
        self.htf_ema_slow = bt.indicators.EMA(self.htf_data.close, period=self.p.ema_slow)
        self.htf_macd = bt.indicators.MACD(
            self.htf_data.close,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal
        )
        self.htf_rsi = bt.indicators.RSI(self.htf_data.close, period=self.p.rsi_period)

        # Swing point tracking
        self.swing_highs = []
        self.swing_lows = []
        self.last_swing_high = None
        self.last_swing_low = None

        # Trade tracking
        self.order = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.trade_direction = None  # 'long' or 'short'

        # Trade log for reporting
        self.trade_log = []

    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.date(0)
        # print(f'{dt.isoformat()} {txt}')

    def notify_order(self, order):
        """Handle order notifications"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}')
            else:
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}')

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        """Handle trade notifications"""
        if not trade.isclosed:
            return

        trade_info = {
            'entry_date': bt.num2date(trade.dtopen),
            'exit_date': bt.num2date(trade.dtclose),
            'direction': self.trade_direction,
            'entry_price': trade.price,
            'exit_price': self.data_close[0],
            'pnl': trade.pnl,
            'pnl_pct': (trade.pnl / trade.price) * 100,
            'size': abs(trade.size),
        }
        self.trade_log.append(trade_info)

        self.log(f'TRADE CLOSED - PnL: {trade.pnl:.2f} ({trade_info["pnl_pct"]:.2f}%)')

    def update_swing_points(self):
        """Update swing highs and lows"""
        lookback = self.p.swing_lookback

        if len(self.data_close) < lookback * 2 + 1:
            return

        # Check for swing high
        current_idx = len(self.data_close) - lookback - 1
        is_swing_high = True
        is_swing_low = True

        center_high = self.data_high[-lookback - 1]
        center_low = self.data_low[-lookback - 1]

        for i in range(-lookback * 2, 0):
            if i == -lookback - 1:
                continue
            if self.data_high[i] >= center_high:
                is_swing_high = False
            if self.data_low[i] <= center_low:
                is_swing_low = False

        if is_swing_high:
            self.swing_highs.append({
                'price': center_high,
                'bar': len(self.data_close) - lookback - 1
            })
            self.last_swing_high = center_high

        if is_swing_low:
            self.swing_lows.append({
                'price': center_low,
                'bar': len(self.data_close) - lookback - 1
            })
            self.last_swing_low = center_low

    def check_bullish_setup(self):
        """Check for bullish (long) entry conditions"""
        if len(self.data_close) < 50:
            return False

        # Higher TF conditions
        htf_bullish_trend = self.htf_ema_fast[0] > self.htf_ema_slow[0]
        htf_macd_bullish = self.htf_macd.macd[0] > self.htf_macd.signal[0]

        # Lower TF conditions
        ltf_ema_bullish = self.ema_fast[0] > self.ema_slow[0]
        ltf_macd_cross_up = (self.macd.macd[0] > self.macd.signal[0] and
                             self.macd.macd[-1] <= self.macd.signal[-1])

        # RSI condition (not overbought)
        rsi_ok = self.rsi[0] < self.p.rsi_overbought

        # Volume confirmation
        volume_ok = self.data_volume[0] > self.volume_sma[0] * self.p.volume_mult * 0.8

        # Support zone check (price near recent swing low)
        at_support = False
        if self.last_swing_low is not None:
            distance_to_support = (self.data_close[0] - self.last_swing_low) / self.data_close[0]
            at_support = distance_to_support < 0.02  # Within 2% of support

        # Bullish BOS (Break of Structure)
        bullish_bos = False
        if self.last_swing_high is not None and len(self.swing_highs) > 1:
            if self.data_close[0] > self.last_swing_high and self.data_close[-1] <= self.last_swing_high:
                bullish_bos = True

        # Combined conditions
        htf_ok = htf_bullish_trend or htf_macd_bullish
        ltf_ok = ltf_ema_bullish or ltf_macd_cross_up
        entry_trigger = at_support or bullish_bos

        return htf_ok and ltf_ok and rsi_ok and (volume_ok or entry_trigger)

    def check_bearish_setup(self):
        """Check for bearish (short) entry conditions"""
        if len(self.data_close) < 50:
            return False

        # Higher TF conditions
        htf_bearish_trend = self.htf_ema_fast[0] < self.htf_ema_slow[0]
        htf_macd_bearish = self.htf_macd.macd[0] < self.htf_macd.signal[0]

        # Lower TF conditions
        ltf_ema_bearish = self.ema_fast[0] < self.ema_slow[0]
        ltf_macd_cross_down = (self.macd.macd[0] < self.macd.signal[0] and
                               self.macd.macd[-1] >= self.macd.signal[-1])

        # RSI condition (not oversold)
        rsi_ok = self.rsi[0] > self.p.rsi_oversold

        # Volume confirmation
        volume_ok = self.data_volume[0] > self.volume_sma[0] * self.p.volume_mult * 0.8

        # Resistance zone check (price near recent swing high)
        at_resistance = False
        if self.last_swing_high is not None:
            distance_to_resistance = (self.last_swing_high - self.data_close[0]) / self.data_close[0]
            at_resistance = distance_to_resistance < 0.02  # Within 2% of resistance

        # Bearish BOS (Break of Structure)
        bearish_bos = False
        if self.last_swing_low is not None and len(self.swing_lows) > 1:
            if self.data_close[0] < self.last_swing_low and self.data_close[-1] >= self.last_swing_low:
                bearish_bos = True

        # Combined conditions
        htf_ok = htf_bearish_trend or htf_macd_bearish
        ltf_ok = ltf_ema_bearish or ltf_macd_cross_down
        entry_trigger = at_resistance or bearish_bos

        return htf_ok and ltf_ok and rsi_ok and (volume_ok or entry_trigger)

    def calculate_position_size(self, stop_distance):
        """Calculate position size based on risk management"""
        account_value = self.broker.getvalue()
        risk_amount = account_value * self.p.risk_per_trade

        # Position size = Risk Amount / Stop Distance
        if stop_distance > 0:
            position_size = risk_amount / stop_distance
            # Apply leverage
            max_position = (account_value * self.p.leverage) / self.data_close[0]
            position_size = min(position_size, max_position)
            return position_size
        return 0

    def next(self):
        """Main strategy logic executed on each bar"""
        # Update swing points
        self.update_swing_points()

        # Skip if we have a pending order
        if self.order:
            return

        # Check if we have a position
        if not self.position:
            # Look for entry signals
            if self.check_bullish_setup():
                self.enter_long()
            elif self.check_bearish_setup():
                self.enter_short()
        else:
            # Manage existing position
            self.manage_position()

    def enter_long(self):
        """Enter a long position"""
        atr = self.atr[0]
        stop_distance = atr * self.p.atr_sl_mult

        # Calculate stop loss and take profit
        self.entry_price = self.data_close[0]
        self.stop_loss = self.entry_price - stop_distance
        self.take_profit = self.entry_price + (stop_distance * self.p.risk_reward)

        # Calculate position size
        size = self.calculate_position_size(stop_distance)

        if size > 0:
            self.log(f'LONG ENTRY Signal - Price: {self.entry_price:.2f}, SL: {self.stop_loss:.2f}, TP: {self.take_profit:.2f}')
            self.order = self.buy(size=size)
            self.trade_direction = 'long'

    def enter_short(self):
        """Enter a short position"""
        atr = self.atr[0]
        stop_distance = atr * self.p.atr_sl_mult

        # Calculate stop loss and take profit
        self.entry_price = self.data_close[0]
        self.stop_loss = self.entry_price + stop_distance
        self.take_profit = self.entry_price - (stop_distance * self.p.risk_reward)

        # Calculate position size
        size = self.calculate_position_size(stop_distance)

        if size > 0:
            self.log(f'SHORT ENTRY Signal - Price: {self.entry_price:.2f}, SL: {self.stop_loss:.2f}, TP: {self.take_profit:.2f}')
            self.order = self.sell(size=size)
            self.trade_direction = 'short'

    def manage_position(self):
        """Manage existing position - check SL/TP"""
        current_price = self.data_close[0]

        if self.trade_direction == 'long':
            # Check stop loss
            if current_price <= self.stop_loss:
                self.log(f'LONG STOP LOSS HIT - Price: {current_price:.2f}')
                self.order = self.close()
            # Check take profit
            elif current_price >= self.take_profit:
                self.log(f'LONG TAKE PROFIT HIT - Price: {current_price:.2f}')
                self.order = self.close()

        elif self.trade_direction == 'short':
            # Check stop loss
            if current_price >= self.stop_loss:
                self.log(f'SHORT STOP LOSS HIT - Price: {current_price:.2f}')
                self.order = self.close()
            # Check take profit
            elif current_price <= self.take_profit:
                self.log(f'SHORT TAKE PROFIT HIT - Price: {current_price:.2f}')
                self.order = self.close()

    def stop(self):
        """Called when backtest ends"""
        self.log(f'Final Portfolio Value: {self.broker.getvalue():.2f}')
