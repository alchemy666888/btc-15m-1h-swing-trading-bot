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


class DataLoader:
    """
    Data loader for BTC swing trading backtester.

    Loads CSV data from raw folder and prepares multi-timeframe feeds.
    """

    def __init__(self, config_path: str = "config/parameters.yaml"):
        self.config = load_config(config_path)
        self.raw_data_path = self.config['paths']['raw_data']
        self.processed_data_path = self.config['paths']['processed_data']
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

        Returns:
            DataFrame with 15min OHLCV data
        """
        # Look for 15m data file
        files = os.listdir(self.raw_data_path)
        ltf_file = None

        for f in files:
            if '15m' in f.lower() and f.endswith('.csv'):
                ltf_file = os.path.join(self.raw_data_path, f)
                break

        if ltf_file is None:
            raise FileNotFoundError(f"No 15min data file found in {self.raw_data_path}")

        print(f"Loading primary data from: {ltf_file}")
        df = self.load_csv(ltf_file)
        print(f"Loaded {len(df)} candles ({df['timestamp'].min()} to {df['timestamp'].max()})")

        return df

    def load_htf_data(self) -> pd.DataFrame:
        """
        Load higher timeframe (1H) data.

        Returns:
            DataFrame with 1H OHLCV data
        """
        # Look for 1h data file
        files = os.listdir(self.raw_data_path)
        htf_file = None

        for f in files:
            if '1h' in f.lower() and f.endswith('.csv'):
                htf_file = os.path.join(self.raw_data_path, f)
                break

        if htf_file is None:
            raise FileNotFoundError(f"No 1H data file found in {self.raw_data_path}")

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
