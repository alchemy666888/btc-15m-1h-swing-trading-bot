#!/usr/bin/env python3
"""
BTC Swing Trading Backtester - Main Entry Point

Multi-timeframe swing trading strategy backtester for crypto perpetual futures.
Supports both bullish (long) and bearish (short) setups.

Usage:
    python main.py [--synthetic] [--no-monte-carlo] [--report-only]

Options:
    --synthetic         Use synthetic data (when Binance API unavailable)
    --no-monte-carlo    Skip Monte Carlo simulation
    --report-only       Only generate report from cached results
"""
import argparse
import sys
import os
import json
import pickle
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.data.fetcher import BinanceDataFetcher
from src.indicators.technical import TechnicalIndicators
from src.backtest.engine import BacktestEngine
from src.backtest.monte_carlo import MonteCarloSimulator, MonteCarloResult
from src.report.html_report import HTMLReportGenerator


def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║       BTC SWING TRADING STRATEGY BACKTESTER                  ║
    ║       Multi-Timeframe Analysis (1H + 15min)                  ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_config():
    """Print current configuration"""
    print(f"\n{'='*60}")
    print("CONFIGURATION")
    print(f"{'='*60}")
    print(f"Symbol:             {config.SYMBOL}")
    print(f"Backtest Period:    {config.START_DATE.strftime('%Y-%m-%d')} to {config.END_DATE.strftime('%Y-%m-%d')}")
    print(f"Higher Timeframe:   {config.HIGHER_TIMEFRAME}")
    print(f"Lower Timeframe:    {config.LOWER_TIMEFRAME}")
    print(f"Initial Capital:    ${config.INITIAL_CAPITAL:,.2f}")
    print(f"Leverage:           {config.LEVERAGE}x")
    print(f"Risk per Trade:     {config.RISK_PER_TRADE*100:.1f}%")
    print(f"Risk:Reward Target: 1:{config.RISK_REWARD_TARGET}")
    print(f"{'='*60}\n")


def fetch_data(use_synthetic: bool = False):
    """Fetch or load price data"""
    print("\n[1/5] FETCHING DATA")
    print("-" * 40)

    fetcher = BinanceDataFetcher()

    if use_synthetic:
        print("Using synthetic data generation...")
        higher_df, lower_df = fetcher.generate_synthetic_data(
            symbol=config.SYMBOL,
            higher_tf=config.HIGHER_TIMEFRAME,
            lower_tf=config.LOWER_TIMEFRAME,
            start_date=config.START_DATE,
            end_date=config.END_DATE
        )
    else:
        higher_df, lower_df = fetcher.fetch_multi_timeframe(
            symbol=config.SYMBOL,
            higher_tf=config.HIGHER_TIMEFRAME,
            lower_tf=config.LOWER_TIMEFRAME,
            start_date=config.START_DATE,
            end_date=config.END_DATE,
            use_cache=True
        )

    print(f"\nHigher TF data: {len(higher_df)} candles")
    print(f"Lower TF data:  {len(lower_df)} candles")

    return higher_df, lower_df


def add_indicators(higher_df, lower_df):
    """Add technical indicators to dataframes"""
    print("\n[2/5] CALCULATING INDICATORS")
    print("-" * 40)

    indicators = TechnicalIndicators()

    print("Adding indicators to higher timeframe...")
    higher_df = indicators.add_all_indicators(higher_df)
    higher_df = indicators.detect_divergence(higher_df)
    higher_df = indicators.detect_bos(higher_df)

    print("Adding indicators to lower timeframe...")
    lower_df = indicators.add_all_indicators(lower_df)
    lower_df = indicators.detect_divergence(lower_df)
    lower_df = indicators.detect_bos(lower_df)

    print(f"Higher TF columns: {len(higher_df.columns)}")
    print(f"Lower TF columns:  {len(lower_df.columns)}")

    return higher_df, lower_df


def run_backtest(higher_df, lower_df):
    """Run the backtest"""
    print("\n[3/5] RUNNING BACKTEST")
    print("-" * 40)

    engine = BacktestEngine(
        initial_capital=config.INITIAL_CAPITAL,
        leverage=config.LEVERAGE,
        commission=config.TAKER_FEE,
        slippage=config.SLIPPAGE
    )

    results = engine.run(lower_df, higher_df)
    engine.print_summary()

    # Get benchmark
    benchmark = engine.get_buy_and_hold_benchmark(lower_df)
    print(f"\nBuy & Hold Benchmark:")
    print(f"  Start Price:  ${benchmark['start_price']:,.2f}")
    print(f"  End Price:    ${benchmark['end_price']:,.2f}")
    print(f"  Return:       {benchmark['return_pct']:.2f}%")
    print(f"  Max Drawdown: {benchmark['max_drawdown']:.2f}%")

    return results, benchmark, engine


def run_monte_carlo(results):
    """Run Monte Carlo simulation"""
    print("\n[4/5] RUNNING MONTE CARLO SIMULATION")
    print("-" * 40)

    simulator = MonteCarloSimulator(
        n_simulations=config.MONTE_CARLO_RUNS,
        confidence_level=config.MONTE_CARLO_CONFIDENCE,
        initial_capital=config.INITIAL_CAPITAL
    )

    # Extract trade returns from results
    trade_log = results.get('trade_log', [])

    if trade_log:
        trade_returns = [t['pnl_pct'] for t in trade_log]
        print(f"Running {config.MONTE_CARLO_RUNS} simulations with {len(trade_returns)} trades...")

        mc_result = simulator.run_trade_resampling(trade_returns)
        simulator.print_summary(mc_result)
    else:
        print("No trades to simulate. Skipping Monte Carlo.")
        # Use time returns instead
        time_returns = results.get('time_returns', {})
        if time_returns:
            daily_returns = [ret * 100 for ret in time_returns.values()]
            mc_result = simulator.run_returns_shuffle(daily_returns)
            simulator.print_summary(mc_result)
        else:
            mc_result = None

    return mc_result


def generate_report(results, mc_result, lower_df, benchmark):
    """Generate HTML report"""
    print("\n[5/5] GENERATING REPORT")
    print("-" * 40)

    generator = HTMLReportGenerator(output_dir=config.REPORTS_DIR)

    # Add date range to results
    results['start_date'] = config.START_DATE.strftime('%Y-%m-%d')
    results['end_date'] = config.END_DATE.strftime('%Y-%m-%d')

    report_path = generator.generate_report(
        backtest_results=results,
        monte_carlo_result=mc_result,
        price_data=lower_df,
        benchmark=benchmark,
        report_name="btc_swing_trading"
    )

    return report_path


def save_results(results, mc_result, benchmark, filepath="data/backtest_results.pkl"):
    """Save backtest results for later analysis"""
    data = {
        'results': results,
        'mc_result': mc_result,
        'benchmark': benchmark,
        'timestamp': datetime.now().isoformat()
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Results saved to {filepath}")


def load_results(filepath="data/backtest_results.pkl"):
    """Load saved backtest results"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['results'], data['mc_result'], data['benchmark']


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="BTC Swing Trading Backtester")
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data (when Binance API unavailable)')
    parser.add_argument('--no-monte-carlo', action='store_true',
                        help='Skip Monte Carlo simulation')
    parser.add_argument('--report-only', action='store_true',
                        help='Only generate report from cached results')

    args = parser.parse_args()

    print_banner()
    print_config()

    try:
        if args.report_only:
            # Load cached results and generate report
            print("Loading cached results...")
            results, mc_result, benchmark = load_results()
            fetcher = BinanceDataFetcher()
            _, lower_df = fetcher.generate_synthetic_data()
            report_path = generate_report(results, mc_result, lower_df, benchmark)
        else:
            # Full pipeline
            # Step 1: Fetch data
            higher_df, lower_df = fetch_data(use_synthetic=args.synthetic)

            # Step 2: Add indicators
            higher_df, lower_df = add_indicators(higher_df, lower_df)

            # Step 3: Run backtest
            results, benchmark, engine = run_backtest(higher_df, lower_df)

            # Step 4: Monte Carlo simulation
            if not args.no_monte_carlo:
                mc_result = run_monte_carlo(results)
            else:
                mc_result = None
                print("\n[4/5] SKIPPING MONTE CARLO SIMULATION")

            # Step 5: Generate report
            report_path = generate_report(results, mc_result, lower_df, benchmark)

            # Save results
            save_results(results, mc_result, benchmark)

        print(f"\n{'='*60}")
        print("BACKTEST COMPLETE")
        print(f"{'='*60}")
        print(f"\nReport generated: {report_path}")
        print("\nTo view the report, open it in a web browser:")
        print(f"  file://{os.path.abspath(report_path)}")

    except KeyboardInterrupt:
        print("\n\nBacktest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during backtest: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
