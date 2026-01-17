# Development Plan: Quantitative Backtesting of Bearish Swing Trading Strategy for Crypto Perpetual Futures

## 1. Project Overview
This development plan outlines the steps to implement, backtest, and report on a quantitative bearish swing trading strategy for cryptocurrency perpetual futures, based on the provided multi-timeframe price action and momentum rules. The strategy focuses on detecting topping patterns (distribution → markdown) in assets like BTCUSDT, using elements such as structure breaks, order blocks (OB), fair value gaps (FVG), and MACD divergence.

- **Objective**: Translate the manual swing trading rules into a rule-based Python system, fetch historical data from Binance API, perform backtesting, and generate a comprehensive report to evaluate performance.
- **Asset Focus**: Use BTCUSDT perpetual futures (high liquidity, similar volatility to XMRUSDT).
- **Timeframes**: Higher (1H/4H), Lower (15min/5min).
- **Direction**: Bearish shorts only (as per the original setup).
- **Assumptions**: Leverage 5-10x typical for futures; account for funding rates if possible, but keep simple initially.
- **Tools/Language**: Pure Python with libraries for data handling, technical indicators, and backtesting. No external paid services; rely on free Binance API.

## 2. Requirements and Setup
- **Python Version**: Use Python 3.10+ for compatibility with libraries.
- **Key Libraries**:
  - `ccxt`: For fetching historical OHLCV data from Binance API (supports free public endpoints).
  - `pandas`: For data manipulation and DataFrame operations.
  - `numpy`: For numerical computations (e.g., averages, thresholds).
  - `ta-lib` or `pandas_ta`: For technical indicators like MACD, RSI, ATR, EMA.
  - `backtrader` or `vectorbt`: As the backtesting engine (backtrader is flexible for custom strategies; vectorbt is faster for vectorized operations).
  - `matplotlib` or `plotly`: For visualizing backtest results in the report.
  - `requests`: If needed for direct API calls (but ccxt handles most).
- **Environment Setup**:
  - Create a virtual environment (e.g., via `venv`).
  - Install libraries via pip: `pip install ccxt pandas numpy pandas_ta backtrader matplotlib`.
  - No API key needed for public historical data fetches from Binance.
- **Data Storage**: Save fetched data as CSV files for reproducibility; load into DataFrames for processing.

## 3. Data Fetching from Binance API
- **Source**: Use Binance public API via ccxt to fetch historical candlestick data for BTCUSDT perpetual futures.
- **Steps**:
  - Initialize ccxt exchange: Create a Binance instance with futures enabled.
  - Fetch OHLCV: Use `fetch_ohlcv` method for multiple timeframes (e.g., '1h', '4h', '15m', '5m').
  - Parameters: Symbol='BTCUSDT', timeframe='1h' (etc.), since= start timestamp (e.g., last 1-2 years), limit=1000 (paginate for more data).
  - Handle Pagination: Loop to fetch in chunks until desired period (e.g., 2020-present for robust testing).
  - Data Cleaning: Convert to pandas DataFrame with columns [timestamp, open, high, low, close, volume]; handle timezone (UTC); remove duplicates or gaps.
  - Multi-Timeframe Sync: Resample/align lower TF data to higher TF for alignment checks.
  - Volume Consideration: Use trading volume; no need for tick data.
- **Edge Cases**: Handle rate limits (sleep between calls); fallback to CSV if API fails.

## 4. Strategy Implementation
Implement the strategy logic as a class/module inheriting from the backtesting framework's strategy base. Define rules based on the provided outline without writing actual code—focus on logical flow.

- **Multi-Timeframe Alignment**:
  - On higher TF: Detect extended uptrend (series of HH/HL using zigzag or peak detection); check bearish MACD divergence (histogram decreasing while price HH) or MACD cross down.
  - On lower TF: Confirm BOS (close below recent swing low after LH); use pivot point detection for HH/LH/LL/HL.
- **Structure Break Confirmation**:
  - Identify CHOCH: Track swing highs/lows (e.g., using scipy.signal.find_peaks for local extrema).
  - Quantify Break: Close below swing low by X% (e.g., 0.5-1% adjustable parameter).
  - Volume Filter: Compare candle volume to rolling 20-period average; require >1.5-2x.
- **Order Block + FVG Confluence**:
  - Detect Bullish OB: Last demand zone (e.g., base of consolidation before high, using range between wick low and body high).
  - Bearish OB: After BOS, define proximal (body) / distal (wick) edges from retest candles.
  - FVG Detection: Identify gaps where high of candle N+1 < low of candle N-1 (inefficient moves).
  - Entry: On pullback to OB/FVG (price enters zone); confirm with rejection patterns (pin bar: wick > body*2; engulfing: current close opposite prior).
  - Add MACD: Histogram negative and expanding (delta >0).
- **Risk Management**:
  - Position Sizing: Fixed % risk per trade (e.g., 1-2% of account).
  - SL: Above LH or OB high + buffer (ATR-based).
  - Invalidation: If price closes above broken low with high volume, exit.
  - Partial Profits: Scale out at 1:1, 1:2 R:R.
- **Targets**:
  - Calculate Measured Move: Project prior leg distance down from break point.
  - Trail SL: Using recent swing lows or ATR multiples after breakeven.
- **Additional Filters**:
  - Market Correlation: Fetch BTC/ETH data; skip if in strong uptrend (e.g., above 200 EMA).
  - Overbought: RSI >80 on higher TF; distance from EMAs > threshold.
  - Time Filter: Avoid high-liquidity hours; use timestamp to check (e.g., Asia/EU overlap: 00:00-08:00 UTC).
- **Parameters**: Make tunable (e.g., volume multiplier, % break threshold, ATR periods) for optimization.

## 5. Backtesting Setup
- **Framework Choice**: Use backtrader for event-driven simulation (handles multi-TF easily) or vectorbt for speed on large datasets.
- **Backtest Logic**:
  - Load Data: Multi-TF DataFrames.
  - Simulate Trades: Loop through bars; check conditions sequentially (filter → confirmation → entry → management).
  - Account for Fees: Binance futures fees (0.02-0.04% maker/taker) + slippage (0.05-0.1%).
  - Position Handling: Short entries; calculate PNL with leverage.
  - Time Period: Test on 1-3 years of data; split into in-sample (optimize) / out-of-sample (validate).
  - Monte Carlo: Run multiple iterations with randomized slippage/noise for robustness.
- **Optimization**: Use grid search or genetic algo on parameters (e.g., via backtrader's optstrategy).

## 6. Generating Backtest Report
- **Report Structure**: Output as HTML/PDF via pandas/matplotlib or Jupyter notebook export.
- **Key Metrics**:
  - Performance: Total return, CAGR, max drawdown, Sharpe/Sortino ratio, win rate, avg win/loss, profit factor.
  - Trade Stats: Number of trades, avg duration, longest win/loss streak.
  - Equity Curve: Plot cumulative PNL over time.
  - Drawdown Plot: Underwater chart.
  - Trade Breakdown: Table of entries/exits with reasons (e.g., "Entered on OB pullback, exited at target").
  - Per-Trade Analysis: Histogram of returns; edge by timeframe/market condition.
  - Risk Metrics: Avg risk per trade, exposure time, VaR.
  - Comparison: Benchmark vs BTC buy-hold.
- **Visuals**: 
  - Candlestick charts with entries/exits marked.
  - Indicator overlays (MACD, EMAs, OBs/FVGs as rectangles).
  - Heatmaps for parameter sensitivity.
- **Error Handling**: Log failed conditions; report false positives (e.g., filtered signals).
- **Thoroughness**: Include forward-testing plan (paper trading via API); stress test on volatile periods (e.g., 2022 bear market).

## 7. Development Timeline and Best Practices
- **Phases**:
  1. Week 1: Setup env, fetch/test data.
  2. Week 2: Implement core strategy logic.
  3. Week 3: Build backtester, run initial tests.
  4. Week 4: Optimize, generate report, refine.
- **Best Practices**: 
  - Modular Code: Separate data, strategy, backtest modules.
  - Version Control: Use Git.
  - Debugging: Unit tests for indicators (e.g., manual vs auto swing detection).
  - Scalability: Design to add long setups or other assets later.
  - Limitations: Note curve-fitting risks; real trading adds execution delays.
- **Next Steps**: After implementation, forward-test on live data; integrate alerts if expanding to algo trading.

This plan provides a complete roadmap for building a robust quantitative version of the strategy. If adjustments needed (e.g., add long side, different asset), provide details.
