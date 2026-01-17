"""
Backtesting Engine using Backtrader
"""
import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.strategy.swing_strategy import SwingTradingStrategy


class PandasData(bt.feeds.PandasData):
    """Custom Pandas data feed for backtrader"""
    params = (
        ('datetime', 'timestamp'),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', None),
    )


class BacktestEngine:
    """
    Backtesting engine for running strategy backtests.
    """

    def __init__(
        self,
        initial_capital: float = config.INITIAL_CAPITAL,
        leverage: int = config.LEVERAGE,
        commission: float = config.TAKER_FEE,
        slippage: float = config.SLIPPAGE
    ):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.commission = commission
        self.slippage = slippage
        self.results = None
        self.cerebro = None
        self.strategy_instance = None

    def setup_cerebro(
        self,
        lower_tf_data: pd.DataFrame,
        higher_tf_data: pd.DataFrame
    ) -> bt.Cerebro:
        """
        Setup backtrader Cerebro engine with data and strategy.

        Args:
            lower_tf_data: Lower timeframe DataFrame
            higher_tf_data: Higher timeframe DataFrame

        Returns:
            Configured Cerebro instance
        """
        self.cerebro = bt.Cerebro()

        # Add strategy
        self.cerebro.addstrategy(SwingTradingStrategy)

        # Prepare data feeds
        lower_tf_data = lower_tf_data.copy()
        higher_tf_data = higher_tf_data.copy()

        # Ensure timestamp is the index
        if 'timestamp' in lower_tf_data.columns:
            lower_tf_data['timestamp'] = pd.to_datetime(lower_tf_data['timestamp'])
            lower_tf_data = lower_tf_data.set_index('timestamp')

        if 'timestamp' in higher_tf_data.columns:
            higher_tf_data['timestamp'] = pd.to_datetime(higher_tf_data['timestamp'])
            higher_tf_data = higher_tf_data.set_index('timestamp')

        # Create data feeds
        data_lower = bt.feeds.PandasData(
            dataname=lower_tf_data,
            name='lower_tf'
        )

        data_higher = bt.feeds.PandasData(
            dataname=higher_tf_data,
            name='higher_tf'
        )

        # Add data to cerebro (lower TF first as primary)
        self.cerebro.adddata(data_lower)
        self.cerebro.adddata(data_higher)

        # Set broker parameters
        self.cerebro.broker.setcash(self.initial_capital)
        self.cerebro.broker.setcommission(commission=self.commission)

        # Add slippage
        self.cerebro.broker.set_slippage_perc(self.slippage)

        # Add analyzers
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.01)
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
        self.cerebro.addanalyzer(bt.analyzers.VWR, _name='vwr')
        self.cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')

        return self.cerebro

    def run(
        self,
        lower_tf_data: pd.DataFrame,
        higher_tf_data: pd.DataFrame
    ) -> Dict:
        """
        Run the backtest and return results.

        Args:
            lower_tf_data: Lower timeframe DataFrame
            higher_tf_data: Higher timeframe DataFrame

        Returns:
            Dictionary with backtest results
        """
        print(f"\n{'='*60}")
        print("Running Backtest")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Leverage: {self.leverage}x")
        print(f"Commission: {self.commission*100:.3f}%")
        print(f"{'='*60}\n")

        # Setup cerebro
        self.setup_cerebro(lower_tf_data, higher_tf_data)

        # Run backtest
        results = self.cerebro.run()
        self.strategy_instance = results[0]

        # Extract results
        self.results = self._extract_results(results[0])

        return self.results

    def _extract_results(self, strategy) -> Dict:
        """Extract and compile backtest results"""
        final_value = self.cerebro.broker.getvalue()
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100

        # Get analyzer results
        sharpe = strategy.analyzers.sharpe.get_analysis()
        drawdown = strategy.analyzers.drawdown.get_analysis()
        trades = strategy.analyzers.trades.get_analysis()
        returns = strategy.analyzers.returns.get_analysis()
        sqn = strategy.analyzers.sqn.get_analysis()
        time_returns = strategy.analyzers.time_return.get_analysis()

        # Calculate additional metrics
        trade_log = strategy.trade_log if hasattr(strategy, 'trade_log') else []

        # Win rate calculation
        total_trades = trades.get('total', {}).get('total', 0)
        won_trades = trades.get('won', {}).get('total', 0)
        lost_trades = trades.get('lost', {}).get('total', 0)
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0

        # Profit factor
        gross_profit = trades.get('won', {}).get('pnl', {}).get('total', 0)
        gross_loss = abs(trades.get('lost', {}).get('pnl', {}).get('total', 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        # Average trade metrics
        avg_win = trades.get('won', {}).get('pnl', {}).get('average', 0)
        avg_loss = abs(trades.get('lost', {}).get('pnl', {}).get('average', 0))

        # Convert time returns to equity curve
        equity_curve = []
        running_value = self.initial_capital
        for date, ret in sorted(time_returns.items()):
            running_value = running_value * (1 + ret)
            equity_curve.append({
                'date': date,
                'value': running_value,
                'return': ret
            })

        results = {
            # Portfolio metrics
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return_pct': total_return,
            'total_return_usd': final_value - self.initial_capital,

            # Risk metrics
            'sharpe_ratio': sharpe.get('sharperatio', 0) or 0,
            'max_drawdown_pct': drawdown.get('max', {}).get('drawdown', 0),
            'max_drawdown_usd': drawdown.get('max', {}).get('moneydown', 0),
            'sqn': sqn.get('sqn', 0) or 0,

            # Trade statistics
            'total_trades': total_trades,
            'won_trades': won_trades,
            'lost_trades': lost_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': trades.get('pnl', {}).get('net', {}).get('average', 0),

            # Trade details
            'longest_winning_streak': trades.get('streak', {}).get('won', {}).get('longest', 0),
            'longest_losing_streak': trades.get('streak', {}).get('lost', {}).get('longest', 0),

            # Equity curve data
            'equity_curve': equity_curve,
            'trade_log': trade_log,

            # Time returns for further analysis
            'time_returns': time_returns,
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

        # Calculate CAGR
        if equity_curve:
            days = len(equity_curve)
            years = days / 365
            if years > 0:
                results['cagr'] = ((final_value / self.initial_capital) ** (1 / years) - 1) * 100
            else:
                results['cagr'] = 0
        else:
            results['cagr'] = 0

        return results

    def get_buy_and_hold_benchmark(
        self,
        data: pd.DataFrame
    ) -> Dict:
        """Calculate buy and hold benchmark for comparison"""
        start_price = data['close'].iloc[0]
        end_price = data['close'].iloc[-1]

        buy_hold_return = (end_price - start_price) / start_price * 100
        buy_hold_final = self.initial_capital * (1 + buy_hold_return / 100)

        # Calculate drawdown
        data = data.copy()
        data['cummax'] = data['close'].cummax()
        data['drawdown'] = (data['close'] - data['cummax']) / data['cummax'] * 100
        max_drawdown = data['drawdown'].min()

        return {
            'start_price': start_price,
            'end_price': end_price,
            'return_pct': buy_hold_return,
            'final_value': buy_hold_final,
            'max_drawdown': max_drawdown
        }

    def print_summary(self):
        """Print a summary of backtest results"""
        if not self.results:
            print("No results available. Run backtest first.")
            return

        r = self.results

        print(f"\n{'='*60}")
        print("BACKTEST RESULTS SUMMARY")
        print(f"{'='*60}")

        print(f"\n--- Portfolio Performance ---")
        print(f"Initial Capital:     ${r['initial_capital']:>12,.2f}")
        print(f"Final Value:         ${r['final_value']:>12,.2f}")
        print(f"Total Return:        {r['total_return_pct']:>12.2f}%")
        print(f"Total Return (USD):  ${r['total_return_usd']:>12,.2f}")
        print(f"CAGR:                {r['cagr']:>12.2f}%")

        print(f"\n--- Risk Metrics ---")
        print(f"Sharpe Ratio:        {r['sharpe_ratio']:>12.2f}")
        print(f"Sortino Ratio:       {r['sortino_ratio']:>12.2f}")
        print(f"Max Drawdown:        {r['max_drawdown_pct']:>12.2f}%")
        print(f"Max Drawdown (USD):  ${r['max_drawdown_usd']:>12,.2f}")
        print(f"SQN:                 {r['sqn']:>12.2f}")

        print(f"\n--- Trade Statistics ---")
        print(f"Total Trades:        {r['total_trades']:>12}")
        print(f"Won Trades:          {r['won_trades']:>12}")
        print(f"Lost Trades:         {r['lost_trades']:>12}")
        print(f"Win Rate:            {r['win_rate']:>12.2f}%")
        print(f"Profit Factor:       {r['profit_factor']:>12.2f}")
        print(f"Avg Win:             ${r['avg_win']:>12,.2f}")
        print(f"Avg Loss:            ${r['avg_loss']:>12,.2f}")
        print(f"Avg Trade:           ${r['avg_trade']:>12,.2f}")
        print(f"Longest Win Streak:  {r['longest_winning_streak']:>12}")
        print(f"Longest Loss Streak: {r['longest_losing_streak']:>12}")

        print(f"\n{'='*60}\n")


def main():
    """Test backtest engine with sample data"""
    # Create sample data
    np.random.seed(42)
    dates_15m = pd.date_range('2025-01-01', periods=1000, freq='15min')
    dates_1h = pd.date_range('2025-01-01', periods=250, freq='1h')

    def generate_price_data(dates):
        price = 95000
        data = []
        for _ in range(len(dates)):
            open_price = price
            change = np.random.normal(0, 0.005)
            close_price = price * (1 + change)
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.002)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.002)))
            volume = np.random.uniform(1000, 5000)
            data.append([open_price, high_price, low_price, close_price, volume])
            price = close_price
        return data

    lower_data = pd.DataFrame(
        generate_price_data(dates_15m),
        columns=['open', 'high', 'low', 'close', 'volume']
    )
    lower_data['timestamp'] = dates_15m

    higher_data = pd.DataFrame(
        generate_price_data(dates_1h),
        columns=['open', 'high', 'low', 'close', 'volume']
    )
    higher_data['timestamp'] = dates_1h

    # Run backtest
    engine = BacktestEngine()
    results = engine.run(lower_data, higher_data)
    engine.print_summary()

    # Get benchmark
    benchmark = engine.get_buy_and_hold_benchmark(lower_data)
    print(f"Buy & Hold Return: {benchmark['return_pct']:.2f}%")


if __name__ == "__main__":
    main()
