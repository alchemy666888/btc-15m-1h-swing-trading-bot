"""
Configuration file for BTC Swing Trading Bot Backtester
"""
from datetime import datetime

# =============================================================================
# GENERAL SETTINGS
# =============================================================================
SYMBOL = "BTC/USDT"
EXCHANGE = "binance"

# Backtest Period
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 12, 31)

# Timeframes
HIGHER_TIMEFRAME = "1h"
LOWER_TIMEFRAME = "15m"

# =============================================================================
# ACCOUNT SETTINGS
# =============================================================================
INITIAL_CAPITAL = 10000  # USD
LEVERAGE = 10
RISK_PER_TRADE = 0.02  # 2% risk per trade

# =============================================================================
# TRADING FEES (Binance Futures)
# =============================================================================
MAKER_FEE = 0.0002  # 0.02%
TAKER_FEE = 0.0004  # 0.04%
SLIPPAGE = 0.0005   # 0.05%

# =============================================================================
# STRATEGY PARAMETERS
# =============================================================================

# Swing Detection
SWING_LOOKBACK = 5  # Bars to look back for swing high/low detection
MIN_SWING_PERCENT = 0.005  # Minimum 0.5% move to qualify as swing

# MACD Parameters
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# RSI Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# EMA Parameters
EMA_FAST = 9
EMA_SLOW = 21
EMA_TREND = 50

# ATR Parameters
ATR_PERIOD = 14
ATR_SL_MULTIPLIER = 1.5  # Stop loss = ATR * multiplier

# Structure Break Confirmation
BREAK_THRESHOLD = 0.005  # 0.5% break below/above swing level
VOLUME_MULTIPLIER = 1.5  # Volume > 1.5x average for confirmation

# Support/Resistance Zones
SR_LOOKBACK = 20  # Bars to look back for S/R zones
SR_ZONE_THRESHOLD = 0.002  # 0.2% zone width

# Fair Value Gap (FVG)
FVG_MIN_SIZE = 0.001  # Minimum 0.1% gap size

# =============================================================================
# RISK MANAGEMENT
# =============================================================================
MAX_POSITION_SIZE = 0.5  # Max 50% of capital in single trade
RISK_REWARD_TARGET = 2.0  # Target R:R ratio for take profit

# =============================================================================
# MONTE CARLO SETTINGS
# =============================================================================
MONTE_CARLO_RUNS = 1000
MONTE_CARLO_CONFIDENCE = 0.95  # 95% confidence interval

# =============================================================================
# DATA PATHS
# =============================================================================
DATA_DIR = "data"
REPORTS_DIR = "reports"
