#!/usr/bin/env python3
"""
BTC Swing Trading Backtester - Main Entry Point

Multi-timeframe swing trading strategy backtester using:
- HTF (1H) trend filtering
- LTF (15min) entry triggers
- Risk-based position sizing
- Monte Carlo robustness testing

Usage:
    python main.py [--no-monte-carlo] [--quiet]

Options:
    --no-monte-carlo    Skip Monte Carlo simulation
    --quiet             Suppress trade-by-trade logging
"""
import argparse
import sys
import os
import pickle
from datetime import datetime
import yaml

import backtrader as bt
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import DataLoader, load_config
from src.strategy.swing_strategy import SwingStrategy
from src.backtest.monte_carlo import MonteCarloSimulator
from src.report.html_report import HTMLReportGenerator


def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║       BTC SWING TRADING STRATEGY BACKTESTER                  ║
    ║       Multi-Timeframe Analysis (1H + 15min)                  ║
    ║       HTF Filter + LTF Trigger + MACD Confirmation           ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_config(config: dict):
    """Print current configuration"""
    print(f"\n{'='*60}")
    print("CONFIGURATION")
    print(f"{'='*60}")
    print(f"Symbol:             {config['general']['symbol']}")
    print(f"Backtest Period:    {config['general']['start_date']} to {config['general']['end_date']}")
    print(f"Higher Timeframe:   {config['timeframes']['higher']}")
    print(f"Lower Timeframe:    {config['timeframes']['primary']}")
    print(f"Initial Capital:    ${config['general']['initial_capital']:,.2f}")
    print(f"Leverage:           {config['general']['leverage']}x")
    print(f"Risk per Trade:     {config['risk']['risk_per_trade']*100:.1f}%")
    print(f"Commission:         {config['fees']['commission']*100:.3f}%")
    print(f"{'='*60}\n")


def load_data():
    """Load price data from CSV files"""
    print("\n[1/5] LOADING DATA")
    print("-" * 40)

    loader = DataLoader()
    df_15m, df_1h, df_daily = loader.load_all_timeframes()

    print(f"\n15min data: {len(df_15m)} candles")
    print(f"1H data:    {len(df_1h)} candles")
    print(f"Daily data: {len(df_daily)} candles")

    return df_15m, df_1h, df_daily


def run_backtest(df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_daily: pd.DataFrame, config: dict, quiet: bool = False):
    """Run the backtest using Backtrader"""
    print("\n[2/5] RUNNING BACKTEST")
    print("-" * 40)

    # Create Cerebro engine
    cerebro = bt.Cerebro()

    # Add strategy
    cerebro.addstrategy(SwingStrategy, printlog=not quiet)

    # Prepare data feeds
    df_15m = df_15m.copy().set_index('timestamp')
    df_1h = df_1h.copy().set_index('timestamp')
    df_daily = df_daily.copy().set_index('timestamp')

    # Add data feeds (LTF first as primary)
    data_15m = bt.feeds.PandasData(dataname=df_15m, name='15min')
    data_1h = bt.feeds.PandasData(dataname=df_1h, name='1h')
    data_daily = bt.feeds.PandasData(dataname=df_daily, name='daily')

    cerebro.adddata(data_15m)
    cerebro.adddata(data_1h)
    cerebro.adddata(data_daily)

    # Broker settings
    cerebro.broker.setcash(config['general']['initial_capital'])
    cerebro.broker.setcommission(commission=config['fees']['commission'])
    cerebro.broker.set_slippage_perc(config['fees']['slippage'])

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.01)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')

    print(f"\nStarting backtest...")
    print(f"Initial Capital: ${config['general']['initial_capital']:,.2f}")

    # Run backtest
    results = cerebro.run()
    strategy = results[0]

    # Extract results
    final_value = cerebro.broker.getvalue()
    initial_value = config['general']['initial_capital']

    backtest_results = extract_results(strategy, initial_value, final_value, config)

    print_summary(backtest_results)

    return backtest_results, strategy


def extract_results(strategy, initial_value: float, final_value: float, config: dict) -> dict:
    """Extract and compile backtest results"""
    # Get analyzer results
    sharpe = strategy.analyzers.sharpe.get_analysis()
    drawdown = strategy.analyzers.drawdown.get_analysis()
    trades = strategy.analyzers.trades.get_analysis()
    time_returns = strategy.analyzers.time_return.get_analysis()

    # Calculate metrics
    total_return = (final_value - initial_value) / initial_value * 100
    total_trades = trades.get('total', {}).get('total', 0)
    won_trades = trades.get('won', {}).get('total', 0)
    lost_trades = trades.get('lost', {}).get('total', 0)
    win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0

    gross_profit = trades.get('won', {}).get('pnl', {}).get('total', 0)
    gross_loss = abs(trades.get('lost', {}).get('pnl', {}).get('total', 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

    avg_win = trades.get('won', {}).get('pnl', {}).get('average', 0)
    avg_loss = abs(trades.get('lost', {}).get('pnl', {}).get('average', 0))

    # Build equity curve
    equity_curve = []
    running_value = initial_value
    for date, ret in sorted(time_returns.items()):
        running_value = running_value * (1 + ret)
        equity_curve.append({'date': date, 'value': running_value, 'return': ret})

    # Convert trade log
    trade_log = []
    for trade in strategy.trade_log:
        trade_log.append({
            'entry_datetime': trade.entry_datetime.isoformat() if trade.entry_datetime else '',
            'exit_datetime': trade.exit_datetime.isoformat() if trade.exit_datetime else '',
            'direction': trade.direction,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'stop_loss': trade.stop_loss,
            'take_profit': trade.take_profit,
            'size': trade.size,
            'pnl': trade.pnl,
            'pnl_pct': trade.pnl_pct,
            'r_multiple': trade.r_multiple,
            'exit_reason': trade.exit_reason,
        })

    results = {
        'initial_capital': initial_value,
        'final_value': final_value,
        'total_return_pct': total_return,
        'total_return_usd': final_value - initial_value,
        'sharpe_ratio': sharpe.get('sharperatio', 0) or 0,
        'max_drawdown_pct': drawdown.get('max', {}).get('drawdown', 0),
        'max_drawdown_usd': drawdown.get('max', {}).get('moneydown', 0),
        'total_trades': total_trades,
        'won_trades': won_trades,
        'lost_trades': lost_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_trade': trades.get('pnl', {}).get('net', {}).get('average', 0),
        'longest_winning_streak': trades.get('streak', {}).get('won', {}).get('longest', 0),
        'longest_losing_streak': trades.get('streak', {}).get('lost', {}).get('longest', 0),
        'equity_curve': equity_curve,
        'trade_log': trade_log,
        'time_returns': time_returns,
        'start_date': config['general']['start_date'],
        'end_date': config['general']['end_date'],
    }

    # Calculate Sortino Ratio
    if equity_curve:
        daily_returns = [e['return'] for e in equity_curve]
        negative_returns = [r for r in daily_returns if r < 0]
        if negative_returns:
            downside_std = np.std(negative_returns)
            avg_return = np.mean(daily_returns)
            results['sortino_ratio'] = (avg_return / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        else:
            results['sortino_ratio'] = 0
    else:
        results['sortino_ratio'] = 0

    # Calculate SQN
    sqn = strategy.analyzers.sqn.get_analysis()
    results['sqn'] = sqn.get('sqn', 0) or 0

    # Calculate CAGR
    if equity_curve:
        days = len(equity_curve)
        years = days / 365
        if years > 0:
            results['cagr'] = ((final_value / initial_value) ** (1 / years) - 1) * 100
        else:
            results['cagr'] = 0
    else:
        results['cagr'] = 0

    return results


def print_summary(results: dict):
    """Print backtest summary"""
    print(f"\n{'='*60}")
    print("BACKTEST RESULTS SUMMARY")
    print(f"{'='*60}")

    print(f"\n--- Portfolio Performance ---")
    print(f"Initial Capital:     ${results['initial_capital']:>12,.2f}")
    print(f"Final Value:         ${results['final_value']:>12,.2f}")
    print(f"Total Return:        {results['total_return_pct']:>12.2f}%")
    print(f"Total Return (USD):  ${results['total_return_usd']:>12,.2f}")

    print(f"\n--- Risk Metrics ---")
    print(f"Sharpe Ratio:        {results['sharpe_ratio']:>12.2f}")
    print(f"Sortino Ratio:       {results['sortino_ratio']:>12.2f}")
    print(f"Max Drawdown:        {results['max_drawdown_pct']:>12.2f}%")
    print(f"SQN:                 {results['sqn']:>12.2f}")

    print(f"\n--- Trade Statistics ---")
    print(f"Total Trades:        {results['total_trades']:>12}")
    print(f"Won Trades:          {results['won_trades']:>12}")
    print(f"Lost Trades:         {results['lost_trades']:>12}")
    print(f"Win Rate:            {results['win_rate']:>12.2f}%")
    print(f"Profit Factor:       {results['profit_factor']:>12.2f}")
    print(f"Avg Win:             ${results['avg_win']:>12,.2f}")
    print(f"Avg Loss:            ${results['avg_loss']:>12,.2f}")

    print(f"\n{'='*60}\n")


def get_benchmark(df: pd.DataFrame, initial_capital: float) -> dict:
    """Calculate buy and hold benchmark"""
    start_price = df['close'].iloc[0]
    end_price = df['close'].iloc[-1]
    buy_hold_return = (end_price - start_price) / start_price * 100
    buy_hold_final = initial_capital * (1 + buy_hold_return / 100)

    # Calculate max drawdown
    df = df.copy()
    df['cummax'] = df['close'].cummax()
    df['drawdown'] = (df['close'] - df['cummax']) / df['cummax'] * 100
    max_drawdown = df['drawdown'].min()

    return {
        'start_price': start_price,
        'end_price': end_price,
        'return_pct': buy_hold_return,
        'final_value': buy_hold_final,
        'max_drawdown': max_drawdown
    }


def run_monte_carlo(results: dict, config: dict):
    """Run Monte Carlo simulation"""
    print("\n[3/5] RUNNING MONTE CARLO SIMULATION")
    print("-" * 40)

    simulator = MonteCarloSimulator(
        n_simulations=1000,
        confidence_level=0.95,
        initial_capital=config['general']['initial_capital']
    )

    trade_log = results.get('trade_log', [])

    if trade_log:
        trade_returns = [t['pnl_pct'] for t in trade_log]
        print(f"Running 1000 simulations with {len(trade_returns)} trades...")

        mc_result = simulator.run_trade_resampling(trade_returns)
        simulator.print_summary(mc_result)
        return mc_result
    else:
        print("No trades to simulate.")
        return None


def generate_report(results: dict, mc_result, df_15m: pd.DataFrame, benchmark: dict, config: dict):
    """Generate HTML report"""
    print("\n[4/5] GENERATING REPORT")
    print("-" * 40)

    generator = HTMLReportGenerator(output_dir=config['paths']['reports'])

    report_path = generator.generate_report(
        backtest_results=results,
        monte_carlo_result=mc_result,
        price_data=df_15m,
        benchmark=benchmark,
        report_name="btc_swing_trading"
    )

    return report_path


def save_results(results: dict, mc_result, benchmark: dict, filepath: str = "data/processed/backtest_results.pkl"):
    """Save backtest results"""
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


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="BTC Swing Trading Backtester")
    parser.add_argument('--no-monte-carlo', action='store_true',
                        help='Skip Monte Carlo simulation')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress trade-by-trade logging')

    args = parser.parse_args()

    # Load configuration
    config = load_config()

    print_banner()
    print_config(config)

    try:
        # Step 1: Load data
        df_15m, df_1h, df_daily = load_data()

        # Step 2: Run backtest
        results, strategy = run_backtest(df_15m, df_1h, df_daily, config, quiet=args.quiet)

        # Step 3: Monte Carlo (optional)
        if not args.no_monte_carlo and results['total_trades'] > 0:
            mc_result = run_monte_carlo(results, config)
        else:
            mc_result = None
            if args.no_monte_carlo:
                print("\n[3/5] SKIPPING MONTE CARLO SIMULATION")
            else:
                print("\n[3/5] NO TRADES - SKIPPING MONTE CARLO")

        # Step 4: Calculate benchmark
        print("\n[4/5] CALCULATING BENCHMARK")
        print("-" * 40)
        benchmark = get_benchmark(df_15m, config['general']['initial_capital'])
        print(f"Buy & Hold Return: {benchmark['return_pct']:.2f}%")
        print(f"Buy & Hold Max DD: {benchmark['max_drawdown']:.2f}%")

        # Step 5: Generate report
        print("\n[5/5] GENERATING REPORT")
        print("-" * 40)
        report_path = generate_report(results, mc_result, df_15m, benchmark, config)

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
