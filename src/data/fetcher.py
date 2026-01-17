"""
Data fetching module for Binance historical OHLCV data via ccxt
Includes synthetic data generation for testing when API is unavailable
"""
import os
import time
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class SyntheticDataGenerator:
    """Generate realistic synthetic BTC price data for testing"""

    @staticmethod
    def generate_ohlcv(
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1h",
        initial_price: float = 95000,
        volatility: float = 0.02,
        trend: float = 0.0001,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data with realistic BTC-like characteristics.

        Args:
            start_date: Start date for data generation
            end_date: End date for data generation
            timeframe: Candlestick timeframe ('1h', '15m', etc.)
            initial_price: Starting price
            volatility: Daily volatility (e.g., 0.02 = 2%)
            trend: Daily drift/trend
            seed: Random seed for reproducibility

        Returns:
            DataFrame with OHLCV data
        """
        np.random.seed(seed)

        # Determine frequency
        freq_map = {
            '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '1h', '4h': '4h', '1d': '1D'
        }
        freq = freq_map.get(timeframe, '1h')

        # Generate timestamps
        timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
        n_candles = len(timestamps)

        if n_candles == 0:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # Adjust volatility based on timeframe
        tf_multiplier = {
            '1m': 0.1, '5m': 0.22, '15m': 0.39, '30m': 0.55,
            '1h': 0.78, '4h': 1.56, '1d': 3.0
        }.get(timeframe, 1.0)

        candle_volatility = volatility * tf_multiplier / np.sqrt(24)  # Adjusted for timeframe
        candle_trend = trend * tf_multiplier / 24

        # Generate price series using geometric brownian motion with mean reversion
        prices = [initial_price]
        mean_price = initial_price

        for i in range(1, n_candles):
            # Random walk with slight mean reversion
            random_return = np.random.normal(candle_trend, candle_volatility)

            # Add mean reversion (price tends to return to moving average)
            mean_reversion = -0.001 * (prices[-1] - mean_price) / mean_price

            # Add momentum (trending behavior)
            if i > 10:
                recent_returns = [(prices[j] - prices[j-1]) / prices[j-1] for j in range(max(1, i-10), i)]
                momentum = 0.1 * np.mean(recent_returns) if recent_returns else 0
            else:
                momentum = 0

            # Combine factors
            total_return = random_return + mean_reversion + momentum

            new_price = prices[-1] * (1 + total_return)

            # Update mean price slowly
            mean_price = 0.999 * mean_price + 0.001 * new_price

            prices.append(new_price)

        prices = np.array(prices)

        # Generate OHLC from close prices
        opens = prices.copy()
        closes = prices.copy()

        # Add some intra-candle variation
        highs = []
        lows = []

        for i in range(n_candles):
            intra_vol = abs(np.random.normal(0, candle_volatility * 0.5))
            high = max(opens[i], closes[i]) * (1 + intra_vol)
            low = min(opens[i], closes[i]) * (1 - intra_vol)
            highs.append(high)
            lows.append(low)

        # Generate volume with some correlation to price movement
        base_volume = 5000
        volumes = []
        for i in range(n_candles):
            price_change = abs(closes[i] - opens[i]) / opens[i] if opens[i] > 0 else 0
            vol_multiplier = 1 + price_change * 50  # Higher volume on bigger moves
            volume = base_volume * vol_multiplier * np.random.uniform(0.5, 2.0)
            volumes.append(volume)

        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })

        return df


class BinanceDataFetcher:
    """Fetches historical OHLCV data from Binance Futures API"""

    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
            }
        })
        self.data_dir = config.DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)

    def fetch_ohlcv(
        self,
        symbol: str = config.SYMBOL,
        timeframe: str = "1h",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Binance with pagination and caching.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Candlestick timeframe (e.g., '1h', '15m')
            start_date: Start date for data fetch
            end_date: End date for data fetch
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with columns [timestamp, open, high, low, close, volume]
        """
        start_date = start_date or config.START_DATE
        end_date = end_date or config.END_DATE

        # Check cache first
        cache_file = self._get_cache_filename(symbol, timeframe, start_date, end_date)
        if use_cache and os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}")
            return pd.read_csv(cache_file, parse_dates=['timestamp'])

        print(f"Fetching {symbol} {timeframe} data from {start_date} to {end_date}")

        all_data = []
        since = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)

        # Determine limit based on timeframe
        limit = 1000

        while since < end_ms:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=limit
                )

                if not ohlcv:
                    break

                all_data.extend(ohlcv)

                # Update since to last timestamp + 1
                since = ohlcv[-1][0] + 1

                # Progress indicator
                current_date = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
                print(f"  Fetched up to {current_date.strftime('%Y-%m-%d %H:%M')}")

                # Rate limiting
                time.sleep(self.exchange.rateLimit / 1000)

                # Break if we've passed end date
                if ohlcv[-1][0] >= end_ms:
                    break

            except ccxt.RateLimitExceeded:
                print("Rate limit exceeded, waiting 60 seconds...")
                time.sleep(60)
            except Exception as e:
                print(f"Error fetching data: {e}")
                error_count = getattr(self, '_error_count', 0) + 1
                self._error_count = error_count
                if error_count > 5:
                    print("\nAPI unavailable. Falling back to synthetic data generation...")
                    return self._generate_synthetic_fallback(timeframe, start_date, end_date)
                time.sleep(5)
                continue

        # If no data fetched, use synthetic
        if not all_data:
            print("\nNo data fetched. Using synthetic data generation...")
            return self._generate_synthetic_fallback(timeframe, start_date, end_date)

        # Convert to DataFrame
        df = pd.DataFrame(
            all_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Filter to exact date range
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]

        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Cache the data
        if use_cache:
            df.to_csv(cache_file, index=False)
            print(f"Data cached to {cache_file}")

        print(f"Fetched {len(df)} candles")
        return df

    def _get_cache_filename(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> str:
        """Generate cache filename based on parameters"""
        symbol_clean = symbol.replace('/', '_')
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        return os.path.join(
            self.data_dir,
            f"{symbol_clean}_{timeframe}_{start_str}_{end_str}.csv"
        )

    def fetch_multi_timeframe(
        self,
        symbol: str = config.SYMBOL,
        higher_tf: str = config.HIGHER_TIMEFRAME,
        lower_tf: str = config.LOWER_TIMEFRAME,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch data for both higher and lower timeframes.

        Returns:
            Tuple of (higher_tf_df, lower_tf_df)
        """
        print(f"\n{'='*60}")
        print(f"Fetching multi-timeframe data for {symbol}")
        print(f"Higher TF: {higher_tf}, Lower TF: {lower_tf}")
        print(f"{'='*60}\n")

        higher_df = self.fetch_ohlcv(
            symbol, higher_tf, start_date, end_date, use_cache
        )

        lower_df = self.fetch_ohlcv(
            symbol, lower_tf, start_date, end_date, use_cache
        )

        return higher_df, lower_df

    def align_timeframes(
        self,
        higher_df: pd.DataFrame,
        lower_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Align lower timeframe data with higher timeframe.
        Adds higher TF indicators to lower TF data.

        Args:
            higher_df: Higher timeframe DataFrame
            lower_df: Lower timeframe DataFrame

        Returns:
            Lower TF DataFrame with higher TF data merged
        """
        # Create a copy of lower_df
        aligned_df = lower_df.copy()

        # For each lower TF row, find corresponding higher TF row
        # by flooring to higher TF period
        higher_df = higher_df.copy()
        higher_df = higher_df.set_index('timestamp')

        # Rename higher TF columns
        higher_df = higher_df.add_prefix('htf_')

        # Merge using asof join (forward fill from higher TF)
        aligned_df = aligned_df.set_index('timestamp')
        aligned_df = pd.merge_asof(
            aligned_df.reset_index(),
            higher_df.reset_index(),
            on='timestamp',
            direction='backward'
        )

        return aligned_df

    def _generate_synthetic_fallback(
        self,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Generate synthetic data when API is unavailable"""
        print(f"Generating synthetic {timeframe} data...")

        # Use different seeds for different timeframes for variety
        seed = hash(timeframe) % 10000

        df = SyntheticDataGenerator.generate_ohlcv(
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            initial_price=95000,  # Starting BTC price for 2025
            volatility=0.025,     # ~2.5% daily volatility
            trend=0.0002,         # Slight upward trend
            seed=seed
        )

        # Cache the synthetic data
        cache_file = self._get_cache_filename(config.SYMBOL, timeframe, start_date, end_date)
        df.to_csv(cache_file, index=False)
        print(f"Synthetic data cached to {cache_file}")
        print(f"Generated {len(df)} candles")

        return df

    def generate_synthetic_data(
        self,
        symbol: str = config.SYMBOL,
        higher_tf: str = config.HIGHER_TIMEFRAME,
        lower_tf: str = config.LOWER_TIMEFRAME,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate synthetic multi-timeframe data (for testing without API).

        Returns:
            Tuple of (higher_tf_df, lower_tf_df)
        """
        start_date = start_date or config.START_DATE
        end_date = end_date or config.END_DATE

        print(f"\n{'='*60}")
        print(f"Generating synthetic multi-timeframe data")
        print(f"Higher TF: {higher_tf}, Lower TF: {lower_tf}")
        print(f"{'='*60}\n")

        higher_df = self._generate_synthetic_fallback(higher_tf, start_date, end_date)
        lower_df = self._generate_synthetic_fallback(lower_tf, start_date, end_date)

        return higher_df, lower_df


def main():
    """Test data fetching"""
    fetcher = BinanceDataFetcher()

    # Fetch multi-timeframe data
    higher_df, lower_df = fetcher.fetch_multi_timeframe()

    print(f"\nHigher TF ({config.HIGHER_TIMEFRAME}) data shape: {higher_df.shape}")
    print(higher_df.head())

    print(f"\nLower TF ({config.LOWER_TIMEFRAME}) data shape: {lower_df.shape}")
    print(lower_df.head())

    # Test alignment
    aligned = fetcher.align_timeframes(higher_df, lower_df)
    print(f"\nAligned data shape: {aligned.shape}")
    print(aligned.head())


if __name__ == "__main__":
    main()
