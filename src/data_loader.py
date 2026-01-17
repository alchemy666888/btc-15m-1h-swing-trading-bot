"""
Data Loader Module

Handles loading OHLCV data from CSV files and preparing it for Backtrader.
Supports multi-timeframe data with resampling capabilities.
"""
import os
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
from typing import Optional, Dict, Tuple
import backtrader as bt


def load_config(config_path: str = "config/parameters.yaml") -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class SyntheticDataGenerator:
    """Generates realistic synthetic BTC price data for testing"""

    @staticmethod
    def generate(
        start_date: str = "2025-01-01",
        end_date: str = "2025-12-31",
        timeframe: str = "15m",
        initial_price: float = 95000.0
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data with realistic price movements.

        Args:
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            timeframe: Candle timeframe ('15m' or '1h')
            initial_price: Starting price

        Returns:
            DataFrame with timestamp, open, high, low, close, volume
        """
        # Parse dates
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        # Calculate number of candles
        if timeframe == "15m":
            freq = "15min"
            candle_volatility = 0.003  # 0.3% per candle
        elif timeframe == "1h":
            freq = "1h"
            candle_volatility = 0.006  # 0.6% per candle
        else:
            freq = "1h"
            candle_volatility = 0.006

        timestamps = pd.date_range(start=start, end=end, freq=freq)
        n_candles = len(timestamps)

        print(f"Generating {n_candles} synthetic {timeframe} candles...")

        # Generate price series with mean reversion and trends
        np.random.seed(42)  # For reproducibility
        prices = [initial_price]
        mean_price = initial_price

        for i in range(1, n_candles):
            # Random component
            random_return = np.random.normal(0, candle_volatility)

            # Mean reversion
            deviation = (prices[-1] - mean_price) / mean_price
            mean_reversion = -0.05 * deviation

            # Momentum (trending behavior)
            if i > 10:
                recent_returns = [(prices[j] - prices[j-1]) / prices[j-1] for j in range(max(1, i-10), i)]
                momentum = 0.1 * np.mean(recent_returns) if recent_returns else 0
            else:
                momentum = 0

            total_return = random_return + mean_reversion + momentum
            new_price = prices[-1] * (1 + total_return)
            mean_price = 0.999 * mean_price + 0.001 * new_price
            prices.append(new_price)

        prices = np.array(prices)

        # Generate OHLC
        opens = prices.copy()
        closes = prices.copy()
        highs = []
        lows = []

        for i in range(n_candles):
            intra_vol = abs(np.random.normal(0, candle_volatility * 0.5))
            high = max(opens[i], closes[i]) * (1 + intra_vol)
            low = min(opens[i], closes[i]) * (1 - intra_vol)
            highs.append(high)
            lows.append(low)

        # Generate volume
        base_volume = 5000
        volumes = []
        for i in range(n_candles):
            price_change = abs(closes[i] - opens[i]) / opens[i] if opens[i] > 0 else 0
            vol_multiplier = 1 + price_change * 50
            volume = base_volume * vol_multiplier * np.random.uniform(0.5, 2.0)
            volumes.append(volume)

        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })

        return df


class DataLoader:
    """
    Data loader for BTC swing trading backtester.

    Loads CSV data from raw folder and prepares multi-timeframe feeds.
    Generates synthetic data if no CSV files are found.
    """

    def __init__(self, config_path: str = "config/parameters.yaml"):
        self.config = load_config(config_path)
        self.raw_data_path = self.config['paths']['raw_data']
        self.processed_data_path = self.config['paths']['processed_data']

        # Create directories if they don't exist
        os.makedirs(self.raw_data_path, exist_ok=True)
        os.makedirs(self.processed_data_path, exist_ok=True)

    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load OHLCV data from CSV file.

        Args:
            filepath: Path to CSV file

        Returns:
            DataFrame with timestamp, open, high, low, close, volume
        """
        df = pd.read_csv(filepath, parse_dates=['timestamp'])

        # Validate required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Validate OHLC relationships
        invalid_rows = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        if invalid_rows.any():
            print(f"Warning: {invalid_rows.sum()} rows have invalid OHLC relationships")

        return df

    def load_primary_data(self) -> pd.DataFrame:
        """
        Load primary (15min) timeframe data.
        Generates synthetic data if no CSV file is found.

        Returns:
            DataFrame with 15min OHLCV data
        """
        # Look for 15m data file
        files = os.listdir(self.raw_data_path) if os.path.exists(self.raw_data_path) else []
        ltf_file = None

        for f in files:
            if '15m' in f.lower() and f.endswith('.csv'):
                ltf_file = os.path.join(self.raw_data_path, f)
                break

        if ltf_file is None:
            print(f"No 15min data file found in {self.raw_data_path}")
            print("Generating synthetic 15min data...")
            start_date = self.config['general']['start_date']
            end_date = self.config['general']['end_date']
            df = SyntheticDataGenerator.generate(start_date, end_date, "15m")

            # Save for future use
            filename = f"BTC_USDT_15m_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv"
            filepath = os.path.join(self.raw_data_path, filename)
            df.to_csv(filepath, index=False)
            print(f"Saved synthetic data to: {filepath}")
            return df

        print(f"Loading primary data from: {ltf_file}")
        df = self.load_csv(ltf_file)
        print(f"Loaded {len(df)} candles ({df['timestamp'].min()} to {df['timestamp'].max()})")

        return df

    def load_htf_data(self) -> pd.DataFrame:
        """
        Load higher timeframe (1H) data.
        Generates synthetic data if no CSV file is found.

        Returns:
            DataFrame with 1H OHLCV data
        """
        # Look for 1h data file
        files = os.listdir(self.raw_data_path) if os.path.exists(self.raw_data_path) else []
        htf_file = None

        for f in files:
            if '1h' in f.lower() and f.endswith('.csv'):
                htf_file = os.path.join(self.raw_data_path, f)
                break

        if htf_file is None:
            print(f"No 1H data file found in {self.raw_data_path}")
            print("Generating synthetic 1H data...")
            start_date = self.config['general']['start_date']
            end_date = self.config['general']['end_date']
            df = SyntheticDataGenerator.generate(start_date, end_date, "1h")

            # Save for future use
            filename = f"BTC_USDT_1h_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv"
            filepath = os.path.join(self.raw_data_path, filename)
            df.to_csv(filepath, index=False)
            print(f"Saved synthetic data to: {filepath}")
            return df

        print(f"Loading HTF data from: {htf_file}")
        df = self.load_csv(htf_file)
        print(f"Loaded {len(df)} candles ({df['timestamp'].min()} to {df['timestamp'].max()})")

        return df

    def resample_to_daily(self, df_15m: pd.DataFrame) -> pd.DataFrame:
        """
        Resample 15min data to daily timeframe.

        Args:
            df_15m: 15min OHLCV DataFrame

        Returns:
            Daily OHLCV DataFrame
        """
        df = df_15m.copy()
        df = df.set_index('timestamp')

        daily = df.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        daily = daily.reset_index()
        print(f"Resampled to {len(daily)} daily candles")

        return daily

    def prepare_backtrader_data(
        self,
        df: pd.DataFrame,
        name: str = "data"
    ) -> bt.feeds.PandasData:
        """
        Convert DataFrame to Backtrader data feed.

        Args:
            df: OHLCV DataFrame
            name: Name for the data feed

        Returns:
            Backtrader PandasData feed
        """
        df = df.copy()
        df = df.set_index('timestamp')

        data = bt.feeds.PandasData(
            dataname=df,
            name=name,
            datetime=None,  # Use index
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1
        )

        return data

    def load_all_timeframes(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all required timeframes: 15min, 1H, Daily.

        Returns:
            Tuple of (df_15m, df_1h, df_daily)
        """
        # Load primary 15min data
        df_15m = self.load_primary_data()

        # Load 1H data
        df_1h = self.load_htf_data()

        # Resample to daily
        df_daily = self.resample_to_daily(df_15m)

        # Save processed daily data
        daily_path = os.path.join(self.processed_data_path, 'BTC_USDT_1d.csv')
        df_daily.to_csv(daily_path, index=False)
        print(f"Daily data saved to: {daily_path}")

        return df_15m, df_1h, df_daily

    def get_backtrader_feeds(self) -> Dict[str, bt.feeds.PandasData]:
        """
        Get all timeframe data as Backtrader feeds.

        Returns:
            Dictionary with 'ltf', 'htf', 'daily' feeds
        """
        df_15m, df_1h, df_daily = self.load_all_timeframes()

        feeds = {
            'ltf': self.prepare_backtrader_data(df_15m, name='15min'),
            'htf': self.prepare_backtrader_data(df_1h, name='1h'),
            'daily': self.prepare_backtrader_data(df_daily, name='daily')
        }

        return feeds


def main():
    """Test data loading"""
    loader = DataLoader()

    # Load all timeframes
    df_15m, df_1h, df_daily = loader.load_all_timeframes()

    print(f"\n15min data: {len(df_15m)} candles")
    print(df_15m.head())

    print(f"\n1H data: {len(df_1h)} candles")
    print(df_1h.head())

    print(f"\nDaily data: {len(df_daily)} candles")
    print(df_daily.head())


if __name__ == "__main__":
    main()
