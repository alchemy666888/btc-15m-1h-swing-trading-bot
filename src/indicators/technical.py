"""
Technical indicators module for swing trading strategy
"""
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from typing import Tuple, List, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class TechnicalIndicators:
    """Calculate technical indicators for the trading strategy"""

    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to the dataframe"""
        df = df.copy()

        # Trend indicators
        df = TechnicalIndicators.add_ema(df, config.EMA_FAST, 'ema_fast')
        df = TechnicalIndicators.add_ema(df, config.EMA_SLOW, 'ema_slow')
        df = TechnicalIndicators.add_ema(df, config.EMA_TREND, 'ema_trend')

        # MACD
        df = TechnicalIndicators.add_macd(df)

        # RSI
        df = TechnicalIndicators.add_rsi(df, config.RSI_PERIOD)

        # ATR
        df = TechnicalIndicators.add_atr(df, config.ATR_PERIOD)

        # Volume moving average
        df = TechnicalIndicators.add_volume_ma(df, 20)

        # Swing highs/lows
        df = TechnicalIndicators.add_swing_points(df, config.SWING_LOOKBACK)

        # Support/Resistance zones
        df = TechnicalIndicators.add_support_resistance(df)

        # Fair Value Gaps
        df = TechnicalIndicators.add_fvg(df)

        # Structure (HH, HL, LH, LL)
        df = TechnicalIndicators.add_market_structure(df)

        return df

    @staticmethod
    def add_ema(df: pd.DataFrame, period: int, col_name: str) -> pd.DataFrame:
        """Add Exponential Moving Average"""
        df[col_name] = df['close'].ewm(span=period, adjust=False).mean()
        return df

    @staticmethod
    def add_macd(
        df: pd.DataFrame,
        fast: int = config.MACD_FAST,
        slow: int = config.MACD_SLOW,
        signal: int = config.MACD_SIGNAL
    ) -> pd.DataFrame:
        """Add MACD indicator"""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()

        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # MACD cross signals
        df['macd_cross_up'] = (df['macd'] > df['macd_signal']) & \
                              (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        df['macd_cross_down'] = (df['macd'] < df['macd_signal']) & \
                                 (df['macd'].shift(1) >= df['macd_signal'].shift(1))

        return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = config.RSI_PERIOD) -> pd.DataFrame:
        """Add RSI indicator"""
        delta = df['close'].diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        rs = avg_gain / avg_loss.replace(0, np.inf)
        df['rsi'] = 100 - (100 / (1 + rs))

        # Overbought/Oversold
        df['rsi_overbought'] = df['rsi'] > config.RSI_OVERBOUGHT
        df['rsi_oversold'] = df['rsi'] < config.RSI_OVERSOLD

        return df

    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = config.ATR_PERIOD) -> pd.DataFrame:
        """Add Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=period, min_periods=1).mean()

        return df

    @staticmethod
    def add_volume_ma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add volume moving average"""
        df['volume_ma'] = df['volume'].rolling(window=period, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['high_volume'] = df['volume_ratio'] > config.VOLUME_MULTIPLIER
        return df

    @staticmethod
    def add_swing_points(
        df: pd.DataFrame,
        lookback: int = config.SWING_LOOKBACK
    ) -> pd.DataFrame:
        """
        Detect swing highs and lows using local extrema.
        """
        df = df.copy()

        # Initialize columns
        df['swing_high'] = np.nan
        df['swing_low'] = np.nan
        df['is_swing_high'] = False
        df['is_swing_low'] = False

        # Find local maxima (swing highs)
        high_idx = argrelextrema(df['high'].values, np.greater_equal, order=lookback)[0]

        # Find local minima (swing lows)
        low_idx = argrelextrema(df['low'].values, np.less_equal, order=lookback)[0]

        # Mark swing points
        df.loc[df.index[high_idx], 'swing_high'] = df.loc[df.index[high_idx], 'high']
        df.loc[df.index[high_idx], 'is_swing_high'] = True

        df.loc[df.index[low_idx], 'swing_low'] = df.loc[df.index[low_idx], 'low']
        df.loc[df.index[low_idx], 'is_swing_low'] = True

        # Forward fill swing levels for reference
        df['last_swing_high'] = df['swing_high'].ffill()
        df['last_swing_low'] = df['swing_low'].ffill()

        return df

    @staticmethod
    def add_market_structure(df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify market structure: HH, HL, LH, LL
        """
        df = df.copy()

        # Initialize structure columns
        df['structure'] = ''
        df['trend'] = ''

        # Get swing points
        swing_high_mask = df['is_swing_high']
        swing_low_mask = df['is_swing_low']

        # Track previous swing highs and lows
        prev_swing_high = None
        prev_swing_low = None
        prev_prev_swing_high = None
        prev_prev_swing_low = None

        for i in range(len(df)):
            if swing_high_mask.iloc[i]:
                current_high = df['high'].iloc[i]
                if prev_swing_high is not None:
                    if current_high > prev_swing_high:
                        df.loc[df.index[i], 'structure'] = 'HH'
                    else:
                        df.loc[df.index[i], 'structure'] = 'LH'
                prev_prev_swing_high = prev_swing_high
                prev_swing_high = current_high

            if swing_low_mask.iloc[i]:
                current_low = df['low'].iloc[i]
                if prev_swing_low is not None:
                    if current_low > prev_swing_low:
                        df.loc[df.index[i], 'structure'] = 'HL'
                    else:
                        df.loc[df.index[i], 'structure'] = 'LL'
                prev_prev_swing_low = prev_swing_low
                prev_swing_low = current_low

        # Determine trend based on structure
        df['trend'] = df['structure'].map({
            'HH': 'bullish',
            'HL': 'bullish',
            'LH': 'bearish',
            'LL': 'bearish'
        }).ffill().fillna('neutral')

        return df

    @staticmethod
    def add_support_resistance(
        df: pd.DataFrame,
        lookback: int = config.SR_LOOKBACK,
        threshold: float = config.SR_ZONE_THRESHOLD
    ) -> pd.DataFrame:
        """
        Identify key support and resistance zones based on swing points.
        """
        df = df.copy()

        # Initialize S/R columns
        df['nearest_support'] = np.nan
        df['nearest_resistance'] = np.nan
        df['at_support'] = False
        df['at_resistance'] = False

        for i in range(lookback, len(df)):
            current_price = df['close'].iloc[i]
            lookback_df = df.iloc[max(0, i - lookback):i]

            # Find support levels (recent swing lows below current price)
            swing_lows = lookback_df[lookback_df['is_swing_low']]['low'].values
            supports_below = swing_lows[swing_lows < current_price]

            if len(supports_below) > 0:
                nearest_support = supports_below.max()
                df.loc[df.index[i], 'nearest_support'] = nearest_support

                # Check if price is at support zone
                support_zone = current_price * threshold
                if abs(current_price - nearest_support) <= support_zone:
                    df.loc[df.index[i], 'at_support'] = True

            # Find resistance levels (recent swing highs above current price)
            swing_highs = lookback_df[lookback_df['is_swing_high']]['high'].values
            resistances_above = swing_highs[swing_highs > current_price]

            if len(resistances_above) > 0:
                nearest_resistance = resistances_above.min()
                df.loc[df.index[i], 'nearest_resistance'] = nearest_resistance

                # Check if price is at resistance zone
                resistance_zone = current_price * threshold
                if abs(current_price - nearest_resistance) <= resistance_zone:
                    df.loc[df.index[i], 'at_resistance'] = True

        return df

    @staticmethod
    def add_fvg(
        df: pd.DataFrame,
        min_size: float = config.FVG_MIN_SIZE
    ) -> pd.DataFrame:
        """
        Detect Fair Value Gaps (FVG) - imbalances in price action.

        Bullish FVG: Gap between high of candle N-1 and low of candle N+1
        Bearish FVG: Gap between low of candle N-1 and high of candle N+1
        """
        df = df.copy()

        df['bullish_fvg'] = False
        df['bearish_fvg'] = False
        df['fvg_top'] = np.nan
        df['fvg_bottom'] = np.nan

        for i in range(2, len(df)):
            candle_prev = df.iloc[i - 2]  # N-1
            candle_next = df.iloc[i]       # N+1

            current_price = df['close'].iloc[i]

            # Bullish FVG: Low of N+1 > High of N-1
            if candle_next['low'] > candle_prev['high']:
                gap_size = (candle_next['low'] - candle_prev['high']) / current_price
                if gap_size >= min_size:
                    df.loc[df.index[i], 'bullish_fvg'] = True
                    df.loc[df.index[i], 'fvg_top'] = candle_next['low']
                    df.loc[df.index[i], 'fvg_bottom'] = candle_prev['high']

            # Bearish FVG: High of N+1 < Low of N-1
            if candle_next['high'] < candle_prev['low']:
                gap_size = (candle_prev['low'] - candle_next['high']) / current_price
                if gap_size >= min_size:
                    df.loc[df.index[i], 'bearish_fvg'] = True
                    df.loc[df.index[i], 'fvg_top'] = candle_prev['low']
                    df.loc[df.index[i], 'fvg_bottom'] = candle_next['high']

        return df

    @staticmethod
    def detect_divergence(
        df: pd.DataFrame,
        lookback: int = 14
    ) -> pd.DataFrame:
        """
        Detect bullish and bearish divergences between price and MACD.
        """
        df = df.copy()

        df['bullish_divergence'] = False
        df['bearish_divergence'] = False

        for i in range(lookback * 2, len(df)):
            lookback_df = df.iloc[i - lookback:i + 1]

            # Get swing lows in lookback period
            swing_low_mask = lookback_df['is_swing_low']
            swing_low_idx = lookback_df[swing_low_mask].index

            if len(swing_low_idx) >= 2:
                # Check for bullish divergence
                # Price makes lower low, MACD makes higher low
                idx1, idx2 = swing_low_idx[-2], swing_low_idx[-1]
                price1, price2 = df.loc[idx1, 'low'], df.loc[idx2, 'low']
                macd1, macd2 = df.loc[idx1, 'macd_hist'], df.loc[idx2, 'macd_hist']

                if price2 < price1 and macd2 > macd1:
                    df.loc[df.index[i], 'bullish_divergence'] = True

            # Get swing highs in lookback period
            swing_high_mask = lookback_df['is_swing_high']
            swing_high_idx = lookback_df[swing_high_mask].index

            if len(swing_high_idx) >= 2:
                # Check for bearish divergence
                # Price makes higher high, MACD makes lower high
                idx1, idx2 = swing_high_idx[-2], swing_high_idx[-1]
                price1, price2 = df.loc[idx1, 'high'], df.loc[idx2, 'high']
                macd1, macd2 = df.loc[idx1, 'macd_hist'], df.loc[idx2, 'macd_hist']

                if price2 > price1 and macd2 < macd1:
                    df.loc[df.index[i], 'bearish_divergence'] = True

        return df

    @staticmethod
    def detect_bos(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Break of Structure (BOS) - when price breaks beyond recent swing.
        """
        df = df.copy()

        df['bullish_bos'] = False
        df['bearish_bos'] = False

        for i in range(1, len(df)):
            current_close = df['close'].iloc[i]
            prev_close = df['close'].iloc[i - 1]

            last_swing_high = df['last_swing_high'].iloc[i - 1]
            last_swing_low = df['last_swing_low'].iloc[i - 1]

            if pd.notna(last_swing_high) and pd.notna(last_swing_low):
                # Bullish BOS: Close breaks above last swing high
                break_threshold_high = last_swing_high * (1 + config.BREAK_THRESHOLD)
                if current_close > last_swing_high and prev_close <= last_swing_high:
                    df.loc[df.index[i], 'bullish_bos'] = True

                # Bearish BOS: Close breaks below last swing low
                break_threshold_low = last_swing_low * (1 - config.BREAK_THRESHOLD)
                if current_close < last_swing_low and prev_close >= last_swing_low:
                    df.loc[df.index[i], 'bearish_bos'] = True

        return df


def main():
    """Test indicators"""
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=500, freq='1h')

    # Simulate BTC-like price movement
    price = 95000
    prices = [price]
    for _ in range(499):
        price = price * (1 + np.random.normal(0, 0.01))
        prices.append(price)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': [p * (1 + np.random.normal(0, 0.002)) for p in prices],
        'volume': [np.random.uniform(1000, 5000) for _ in prices]
    })

    # Add indicators
    df = TechnicalIndicators.add_all_indicators(df)
    df = TechnicalIndicators.detect_divergence(df)
    df = TechnicalIndicators.detect_bos(df)

    print("Indicators added successfully!")
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nSample data:")
    print(df[['timestamp', 'close', 'macd', 'rsi', 'atr', 'trend']].tail(10))


if __name__ == "__main__":
    main()
