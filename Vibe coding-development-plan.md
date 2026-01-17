# Quantitative BTC Swing Trading Strategy Backtester

---

# Project Overview

**Objective:** Build a multi-timeframe BTC swing trading backtester using Backtrader that implements HTF (1H) trend filtering with LTF (15min) entry triggers, MACD confirmation, and Order Block support/resistance detection.

**Tech Stack:**
- Python 3.10+
- Backtrader (backtesting engine)
- Pandas (data manipulation)
- Plotly (interactive HTML charts)
- Jinja2 (HTML report templating)

---

# Phase 1: Project Setup & Data Infrastructure

## 1.1 Directory Structure

```
btc-swing-backtester/
├── data/
│   ├── raw/                    # Raw OHLCV data files
│   └── processed/              # Resampled/cleaned data
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data fetching & preprocessing
│   ├── indicators/
│   │   ├── __init__.py
│   │   ├── macd_analyzer.py    # MACD with divergence detection
│   │   ├── structure.py        # HH/HL/LH/LL detection
│   │   └── order_blocks.py     # OB zone identification
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── htf_filter.py       # 1H trend validation logic
│   │   ├── ltf_trigger.py      # 15min entry trigger logic
│   │   └── swing_strategy.py   # Main Backtrader Strategy class
│   ├── risk/
│   │   ├── __init__.py
│   │   └── position_sizer.py   # Risk-based position sizing
│   └── reporting/
│       ├── __init__.py
│       ├── metrics.py          # Performance calculations
│       ├── charts.py           # Plotly chart generators
│       └── html_report.py      # Report assembly
├── templates/
│   └── report_template.html    # Jinja2 HTML template
├── config/
│   └── parameters.yaml         # Tunable parameters
├── tests/
│   └── ...                     # Unit tests
├── main.py                     # Entry point
├── requirements.txt
└── README.md
```

## 1.2 Data Requirements

**Primary Data:**
- Symbol: BTCUSDT perpetual
- Timeframes needed: 15min (base), 1H (resampled), Daily (for OB)
- Fields: timestamp, open, high, low, close, volume
- Minimum history: 365 days for robust testing

**Data Loading Tasks:**
1. Fetch 15min OHLCV data (primary timeframe)
2. Resample to 1H within Backtrader using `bt.TimeFrame.Minutes, compression=60`
3. Resample to Daily for Order Block detection
4. Handle timezone normalization (UTC)
5. Validate data integrity (no gaps, proper OHLC relationships)

**Backtrader Multi-Timeframe Setup:**
- Load 15min as `data0` (primary)
- Add 1H as `data1` via resampling
- Add Daily as `data2` via resampling
- Use `bt.ind.ResampledData` or cerebro's resampledata method

---

# Phase 2: Indicator Development

## 2.1 MACD Analyzer Module

**Purpose:** Extended MACD indicator with histogram analysis, divergence detection, and momentum state tracking.

**Components to Build:**

| Component | Description | Default Parameters |
|-----------|-------------|-------------------|
| Standard MACD | EMA fast/slow/signal | 12, 26, 9 |
| Histogram Tracker | Track consecutive pos/neg bars | - |
| Magnitude Change | Calculate bar-to-bar % change | 20% threshold |
| Divergence Detector | Price vs MACD peak comparison | 10% divergence threshold, 5-10 bar lookback |
| State Machine | Track narrowing → expanding → reversal | 4-6 bars for signal |

**Key Functions:**
1. `get_histogram_sequence()` → Returns count and direction of consecutive histogram bars
2. `detect_magnitude_change()` → Returns "narrowing", "expanding", or "neutral"
3. `detect_divergence()` → Returns "bullish_div", "bearish_div", or "none"
4. `get_reversal_signal()` → Boolean for histogram transition pattern

**Implementation Notes:**
- Must work on both 1H and 15min data feeds
- Cache calculations to avoid redundant processing
- Expose as Backtrader indicator for strategy access

---

## 2.2 Price Structure Module

**Purpose:** Detect market structure (HH/HL for bullish, LH/LL for bearish) using swing point analysis.

**Components to Build:**

| Component | Description | Default Parameters |
|-----------|-------------|-------------------|
| Swing Point Detector | Identify local highs/lows | 5-bar lookback window |
| Structure Classifier | Classify as HH/HL/LH/LL | 0.5% minimum move (HTF), 0.3% (LTF) |
| Trend State | Track consecutive structure points | 3 consecutive required |
| Structure Break | Detect invalidation | Close below/above key swing |

**Key Functions:**
1. `find_swing_points(lookback=5)` → Returns list of swing highs/lows with timestamps
2. `classify_structure()` → Returns "bullish" (HH+HL), "bearish" (LH+LL), or "neutral"
3. `get_consecutive_count()` → Number of consecutive valid structure points
4. `check_structure_break()` → Boolean if structure invalidated

**Implementation Notes:**
- Use rolling window comparison for swing detection
- Percentage thresholds should be parameterized
- Return both current state and historical context

---

## 2.3 Order Block Module (Simplified)

**Purpose:** Identify key support/resistance zones based on consolidation-then-breakout patterns.

**Simplified Approach (per requirements):**
- Focus on basic gap detection and consolidation zones
- Skip complex multi-day OB hierarchies

**Components to Build:**

| Component | Description | Default Parameters |
|-----------|-------------|-------------------|
| Consolidation Detector | Find tight range periods | Range < 2% of ATR, 5-bar minimum |
| Breakout Validator | Confirm directional breakout | >3% move from consolidation |
| Zone Calculator | Compute OB midpoint and bounds | Midpoint ± 0.5% buffer |
| Zone Freshness | Track if OB has been tested | Boolean flag |

**Key Functions:**
1. `find_consolidation_zones(atr_threshold=0.02, min_bars=5)` → List of zone objects
2. `validate_breakout(zone, direction, threshold=0.03)` → Boolean
3. `get_active_order_blocks()` → Returns untested OB zones
4. `check_price_vs_ob(current_price, ob_zone)` → "above", "below", "inside"

**Implementation Notes:**
- Run on Daily timeframe data
- Store OB zones in a list, mark as "tested" when price revisits
- Use simple high/low of consolidation range, not complex imbalance logic

---

# Phase 3: Strategy Logic Implementation

## 3.1 HTF Filter (1H Timeframe)

**Purpose:** Validate trend conditions before enabling LTF entries.

**Bullish HTF Valid Conditions (ALL must be true):**

```
□ Structure Check
  ├── At least 3 consecutive Higher Highs (each > previous by 0.5%)
  └── At least 3 consecutive Higher Lows (each > previous by 0.5%)

□ Order Block Check
  └── Current 1H close > Bullish Daily OB midpoint + 0.5% buffer

□ MACD Divergence/Reversal Check
  ├── Count 4-6 consecutive negative histogram bars
  ├── During this sequence: price drawdown < 1%
  ├── Detect histogram magnitude decreasing by ≥20% per bar (2-3 bars)
  └── Transition to ≥2 positive bars with increasing magnitude
```

**Bearish HTF Valid Conditions (ALL must be true):**

```
□ Structure Check
  ├── At least 3 consecutive Lower Highs (each < previous by 0.5%)
  └── At least 3 consecutive Lower Lows (each < previous by 0.5%)

□ Order Block Check
  └── Current 1H close < Bearish Daily OB midpoint - 0.5% buffer

□ MACD Divergence/Reversal Check
  ├── Count 4-6 consecutive positive histogram bars
  ├── During this sequence: price upside < 1%
  ├── Detect histogram magnitude decreasing by ≥20% per bar (2-3 bars)
  └── Transition to ≥2 negative bars with increasing magnitude
```

**State Management:**
- Re-evaluate HTF conditions on every 1H bar close
- Store current HTF state: "bullish_valid", "bearish_valid", or "neutral"
- Implement 10-bar cooldown after structure invalidation

---

## 3.2 LTF Trigger (15min Timeframe)

**Purpose:** Generate precise entry signals when HTF filter is valid.

**Long Entry Conditions (when HTF = "bullish_valid"):**

```
□ LTF Structure
  ├── At least 3 consecutive HH (>0.3% each)
  └── At least 3 consecutive HL (>0.3% each)

□ LTF MACD State
  ├── MACD line > 0 for ≥3 bars
  └── Histogram: ≥4 consecutive positive bars, magnitude expanding ≥20%

□ Pullback Detection
  ├── Price retraces 20-50% of last impulse leg (HL to HH)
  └── Touches support (previous HL midpoint ±0.5%, or trendline, or 15min OB)

□ Entry Confirmation
  ├── MACD line dips during pullback but stays > 0
  ├── MACD crosses up with new positive histogram bar
  └── TRIGGER: Enter LONG at bar close
```

**Short Entry Conditions (when HTF = "bearish_valid"):**

```
□ LTF Structure
  ├── At least 3 consecutive LH (<-0.3% each)
  └── At least 3 consecutive LL (<-0.3% each)

□ LTF MACD State
  ├── MACD line < 0 for ≥3 bars
  └── Histogram: ≥4 consecutive negative bars, magnitude expanding ≥20%

□ Pullback Detection
  ├── Price retraces 20-50% of last impulse leg (LH to LL)
  └── Touches resistance (previous LH midpoint ±0.5%, or trendline, or 15min OB)

□ Entry Confirmation
  ├── MACD line bounces during pullback but stays < 0
  ├── MACD crosses down with new negative histogram bar
  └── TRIGGER: Enter SHORT at bar close
```

---

## 3.3 Trade Management

**Stop Loss Calculation:**

| Direction | Stop Loss Logic |
|-----------|-----------------|
| Long | `min(low of last 2 bars, recent HL - 0.5%, daily OB low if within 2%)` |
| Short | `max(high of last 2 bars, recent LH + 0.5%, daily OB high if within 2%)` |

**Take Profit Calculation (Single Target):**

| Direction | Take Profit Logic |
|-----------|-------------------|
| Long | Previous 1H swing high (highest high in last 20 HTF bars) |
| Short | Previous 1H swing low (lowest low in last 20 HTF bars) |

**Position Sizing:**

```
Position Size = (Account Balance × Risk%) / |Entry Price - Stop Loss|
             = ($10,000 × 2%) / |Entry - SL|
             = $200 / Risk Distance

With 10x Leverage:
  Notional Position = Position Size × Leverage
  Actual Margin Used = Position Size
```

**Exit Rules:**
1. Stop Loss hit → Exit 100%
2. Take Profit hit → Exit 100%
3. Structure invalidation → Exit at next bar close
4. Opposing HTF signal → Exit and potentially reverse

---

# Phase 4: Backtrader Strategy Implementation

## 4.1 Strategy Class Structure

**Main Strategy Class Responsibilities:**

```
SwingStrategy(bt.Strategy)
├── params (all tunable parameters with defaults)
├── __init__()
│   ├── Initialize indicators for each timeframe
│   ├── Setup MACD analyzers (1H and 15min)
│   ├── Setup structure detectors (1H and 15min)
│   └── Setup Order Block tracker (Daily)
├── prenext() / nextstart() / next()
│   ├── Only operate on 15min bar closes
│   ├── Check if 1H bar just closed (for HTF update)
│   ├── Update HTF filter state
│   ├── Check LTF entry conditions
│   └── Manage open positions
├── notify_order()
│   └── Track order fills, log entries/exits
└── notify_trade()
    └── Record trade results for reporting
```

**Parameter Defaults:**

```yaml
# Structure Detection
htf_structure_lookback: 5          # bars for swing detection
htf_structure_threshold: 0.005     # 0.5% minimum move
htf_consecutive_required: 3        # consecutive HH/HL or LH/LL
ltf_structure_lookback: 5
ltf_structure_threshold: 0.003     # 0.3%
ltf_consecutive_required: 3

# MACD Settings
macd_fast: 12
macd_slow: 26
macd_signal: 9
histogram_sequence_min: 4          # minimum consecutive bars
histogram_sequence_max: 6          # maximum for divergence check
magnitude_change_threshold: 0.20   # 20%
divergence_threshold: 0.10         # 10% price vs MACD diff
price_drawdown_max: 0.01           # 1% during histogram sequence

# Order Blocks
ob_consolidation_bars: 5
ob_range_threshold: 0.02           # < 2% of ATR
ob_breakout_threshold: 0.03        # > 3% move
ob_buffer: 0.005                   # 0.5%

# Entry Triggers
pullback_min: 0.20                 # 20% retracement
pullback_max: 0.50                 # 50% retracement
macd_above_zero_bars: 3            # bars MACD must be > 0

# Risk Management
risk_per_trade: 0.02               # 2%
leverage: 10
sl_buffer: 0.005                   # 0.5%
htf_swing_lookback: 20             # bars for TP target

# Filters
invalidation_cooldown: 10          # bars after structure break
atr_period: 14
high_volatility_multiplier: 2.0    # skip if ATR > 2x average
volume_confirmation: 1.5           # entry volume > 1.5x average
```

---

## 4.2 Backtrader Cerebro Configuration

**Setup Requirements:**

```
Cerebro Configuration:
├── Initial Cash: $10,000
├── Commission: 0.04% per side (taker fee)
├── Slippage: 0.05% (fixed percentage)
├── Position Sizing: Custom sizer based on risk calculation
├── Data Feeds:
│   ├── data0: 15min BTCUSDT (primary)
│   ├── data1: Resampled to 1H
│   └── data2: Resampled to Daily
└── Analyzers:
    ├── TradeAnalyzer (built-in)
    ├── SharpeRatio
    ├── DrawDown
    ├── Returns
    └── Custom metrics collector
```

**Leverage Handling:**
- Backtrader doesn't natively support leverage
- Implement via custom commission scheme or position sizing
- Track notional exposure vs actual margin
- Add margin call check (if equity < 10% of position value, force close)

---

# Phase 5: Reporting Module

## 5.1 Metrics to Calculate

**Performance Metrics:**

| Category | Metric | Target |
|----------|--------|--------|
| Returns | Total Return % | - |
| Returns | Annualized Return % | - |
| Returns | Max Drawdown % | <20% |
| Returns | Profit Factor | >1.5 |
| Risk | Sharpe Ratio | >1.0 |
| Risk | Sortino Ratio | >1.5 |
| Risk | Calmar Ratio | - |
| Trades | Total Trades | - |
| Trades | Win Rate % | >50% |
| Trades | Avg Win / Avg Loss | >1.5 |
| Trades | Max Consecutive Wins | - |
| Trades | Max Consecutive Losses | - |
| Trades | Avg Trade Duration | - |
| Direction | Long Win Rate % | - |
| Direction | Short Win Rate % | - |
| Direction | Long Total PnL | - |
| Direction | Short Total PnL | - |

**Trade Log Fields:**
- Entry datetime, price, direction
- Exit datetime, price, reason
- Stop loss, take profit levels
- Position size, notional value
- PnL ($), PnL (%), R-multiple
- HTF state at entry
- MACD values at entry

---

## 5.2 Interactive Charts (Plotly)

**Charts to Generate:**

| Chart | Type | Description |
|-------|------|-------------|
| Equity Curve | Line | Account value over time with drawdown shading |
| Drawdown Chart | Area | Underwater equity curve |
| Trade Distribution | Histogram | PnL distribution with win/loss coloring |
| Monthly Returns | Heatmap | Calendar heatmap of monthly returns |
| Price + Signals | Candlestick | BTCUSDT with entry/exit markers |
| Win Rate by Month | Bar | Monthly win rate comparison |
| R-Multiple Distribution | Histogram | Risk-adjusted returns |
| Direction Comparison | Grouped Bar | Long vs Short performance |

**Interactivity Features:**
- Zoom and pan on all time-series charts
- Hover tooltips with trade details
- Click on equity curve to highlight corresponding trade
- Toggle visibility of long/short trades
- Date range selector

---

## 5.3 HTML Report Structure

**Report Sections:**

```html
<!DOCTYPE html>
<html>
<head>
    <title>BTC Swing Strategy Backtest Report</title>
    <!-- Plotly.js CDN -->
    <!-- Custom CSS -->
</head>
<body>
    <header>
        <h1>BTC Swing Trading Strategy - Backtest Report</h1>
        <p>Generated: {{ timestamp }}</p>
        <p>Period: {{ start_date }} to {{ end_date }}</p>
    </header>

    <section id="summary">
        <h2>Executive Summary</h2>
        <!-- Key metrics cards: Return, Sharpe, Win Rate, Max DD -->
        <!-- Target comparison table -->
    </section>

    <section id="equity">
        <h2>Equity Performance</h2>
        <!-- Equity curve chart -->
        <!-- Drawdown chart -->
    </section>

    <section id="trades">
        <h2>Trade Analysis</h2>
        <!-- Trade distribution histogram -->
        <!-- Direction comparison -->
        <!-- Monthly performance heatmap -->
    </section>

    <section id="signals">
        <h2>Signal Analysis</h2>
        <!-- Price chart with signals (last 30 days sample) -->
        <!-- HTF/LTF state breakdown -->
    </section>

    <section id="parameters">
        <h2>Strategy Parameters</h2>
        <!-- Parameter table -->
    </section>

    <section id="trade-log">
        <h2>Trade Log</h2>
        <!-- Sortable/filterable trade table -->
    </section>

    <footer>
        <p>Strategy: Quantitative BTC Swing | Framework: Backtrader</p>
    </footer>
</body>
</html>
```

---

# Phase 6: Development Sequence

## Step-by-Step Implementation Order

**Week 1: Foundation**
```
Day 1-2: Project Setup
├── Create directory structure
├── Setup requirements.txt (backtrader, pandas, plotly, jinja2, pyyaml)
├── Create config/parameters.yaml with all defaults
└── Implement data_loader.py (fetch/load CSV data)

Day 3-4: Basic Indicators
├── Implement structure.py (swing detection, HH/HL/LH/LL)
├── Unit test swing detection with known data
└── Implement basic MACD wrapper for Backtrader

Day 5: MACD Extensions
├── Add histogram sequence tracking
├── Add magnitude change detection
└── Add divergence detection logic
```

**Week 2: Strategy Core**
```
Day 1-2: Order Blocks
├── Implement consolidation detection
├── Implement breakout validation
├── Implement zone tracking (tested/untested)
└── Test on daily data

Day 3-4: HTF Filter
├── Implement htf_filter.py
├── Combine structure + OB + MACD conditions
├── Implement state management (bullish_valid/bearish_valid/neutral)
└── Test with 1H data feed

Day 5: LTF Trigger
├── Implement ltf_trigger.py
├── Pullback detection logic
├── Entry confirmation logic
└── Integration with HTF state
```

**Week 3: Strategy Assembly**
```
Day 1-2: Main Strategy Class
├── Create SwingStrategy(bt.Strategy)
├── Wire up all indicators
├── Implement next() logic flow
├── Add position sizing (risk-based)
└── Add stop loss / take profit logic

Day 3-4: Trade Management
├── Implement notify_order() logging
├── Implement notify_trade() recording
├── Add structure invalidation exits
├── Add high volatility filter
└── Add volume confirmation filter

Day 5: Cerebro Integration
├── Setup multi-timeframe data feeds
├── Configure commission/slippage
├── Add built-in analyzers
└── End-to-end test run
```

**Week 4: Reporting & Polish**
```
Day 1-2: Metrics Module
├── Implement all metric calculations
├── Create trade log export
└── Validate against Backtrader analyzers

Day 3-4: Charts & Visualization
├── Implement equity curve (Plotly)
├── Implement trade distribution
├── Implement monthly heatmap
├── Implement signal overlay chart

Day 5: HTML Report Assembly
├── Create Jinja2 template
├── Wire up all components
├── Style with CSS
└── Final testing and documentation
```

---

# Phase 7: Testing & Validation Checklist

## Unit Tests

```
□ structure.py
  ├── Test swing high detection with synthetic data
  ├── Test swing low detection
  ├── Test HH/HL sequence identification
  ├── Test LH/LL sequence identification
  └── Test structure break detection

□ macd_analyzer.py
  ├── Test histogram counting
  ├── Test magnitude change detection
  ├── Test divergence detection
  └── Test reversal signal pattern

□ order_blocks.py
  ├── Test consolidation detection
  ├── Test breakout validation
  └── Test zone freshness tracking

□ position_sizer.py
  ├── Test risk calculation accuracy
  ├── Test leverage application
  └── Test margin constraints
```

## Integration Tests

```
□ HTF Filter
  ├── Verify all conditions checked correctly
  ├── Verify state transitions
  └── Verify cooldown after invalidation

□ LTF Trigger
  ├── Verify only fires when HTF valid
  ├── Verify pullback detection accuracy
  └── Verify MACD confirmation timing

□ Full Strategy
  ├── Run on 30-day sample data
  ├── Verify entry/exit prices match expectations
  ├── Verify position sizes match risk rules
  └── Verify no duplicate entries
```

## Validation Checks

```
□ Sanity Checks
  ├── No trades during invalidation cooldown
  ├── Stop loss always set on entry
  ├── Take profit always set on entry
  ├── Max 1 position per direction
  └── Commission/slippage applied correctly

□ Edge Cases
  ├── Data gaps (missing bars)
  ├── Extreme volatility periods
  ├── Low liquidity hours filtering
  ├── Position closed at backtest end
  └── Rapid signal changes (no overtrading)
```

---

# Appendix A: Key Implementation Notes

## Multi-Timeframe Synchronization

Backtrader processes bars chronologically. When using multiple timeframes:
- 15min bars close every 15 minutes
- 1H bars only "close" every 4th 15min bar
- Daily bars only "close" once per day

**Solution:** In `next()`, check if higher timeframe bar just completed:
```
Check: len(self.data1) > len(self.data1_prev)  # 1H bar closed
```

## MACD Histogram State Tracking

Track histogram state across bars using class variables:
- `self.hist_sequence_count`
- `self.hist_sequence_direction` (+1 or -1)
- `self.hist_magnitudes` (list of last N values)

Reset counters on direction change.

## Order Block Persistence

OB zones persist until tested. Store as list of dictionaries:
```
{
    'type': 'bullish' or 'bearish',
    'midpoint': float,
    'upper': float,
    'lower': float,
    'created_bar': int,
    'tested': bool
}
```

Mark as tested when price enters zone. Remove after 50 bars if untested (optional cleanup).

## Pullback Retracement Calculation

```
Impulse leg (long): from last HL to last HH
Retracement % = (HH - Current Price) / (HH - HL)
Valid if: 0.20 ≤ Retracement % ≤ 0.50
```

---

# Appendix B: File Dependency Map

```
main.py
├── imports config/parameters.yaml
├── imports src/data_loader.py
│   └── uses pandas
├── imports src/strategy/swing_strategy.py
│   ├── imports src/indicators/macd_analyzer.py
│   ├── imports src/indicators/structure.py
│   ├── imports src/indicators/order_blocks.py
│   ├── imports src/strategy/htf_filter.py
│   ├── imports src/strategy/ltf_trigger.py
│   └── imports src/risk/position_sizer.py
└── imports src/reporting/html_report.py
    ├── imports src/reporting/metrics.py
    ├── imports src/reporting/charts.py
    └── uses templates/report_template.html
```

---

# Appendix C: Prompt Patterns for Vibe Coding

When working with AI assistants, use these prompt patterns:

**For Indicator Development:**
> "Implement a Backtrader indicator class that detects [specific pattern]. It should take parameters [X, Y, Z] and expose lines for [output1, output2]. Include docstrings and type hints."

**For Strategy Logic:**
> "In the SwingStrategy.next() method, add logic to check [condition set]. Only proceed if [prerequisites]. Use the helper method [helper_name] to calculate [value]."

**For Bug Fixes:**
> "The [component] is producing [incorrect behavior]. Expected: [correct behavior]. The relevant code is in [file]. Debug and fix while maintaining existing interface."

**For Report Generation:**
> "Create a Plotly [chart type] showing [data series]. Include hover tooltips with [fields]. Add [interactivity feature]. Return the figure object for embedding in HTML."

---

*End of Development Plan*
