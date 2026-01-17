"""
Monte Carlo Simulation for Robustness Testing
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


@dataclass
class MonteCarloResult:
    """Container for Monte Carlo simulation results"""
    mean_return: float
    median_return: float
    std_return: float
    min_return: float
    max_return: float
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    probability_profit: float
    probability_loss: float
    mean_max_drawdown: float
    worst_max_drawdown: float
    best_max_drawdown: float
    var_95: float  # Value at Risk at 95%
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    equity_curves: List[np.ndarray]
    final_values: np.ndarray
    max_drawdowns: np.ndarray


class MonteCarloSimulator:
    """
    Monte Carlo simulation for strategy robustness testing.

    Uses trade resampling and random walk methods to generate
    possible equity curves and assess strategy risk.
    """

    def __init__(
        self,
        n_simulations: int = config.MONTE_CARLO_RUNS,
        confidence_level: float = config.MONTE_CARLO_CONFIDENCE,
        initial_capital: float = config.INITIAL_CAPITAL
    ):
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
        self.initial_capital = initial_capital

    def run_trade_resampling(
        self,
        trade_returns: List[float],
        n_trades_per_sim: Optional[int] = None
    ) -> MonteCarloResult:
        """
        Monte Carlo simulation using trade resampling with replacement.

        This method randomly samples from historical trade returns to
        generate possible equity curves.

        Args:
            trade_returns: List of individual trade returns (as percentages)
            n_trades_per_sim: Number of trades per simulation (default: len(trade_returns))

        Returns:
            MonteCarloResult with simulation statistics
        """
        if len(trade_returns) == 0:
            return self._empty_result()

        trade_returns = np.array(trade_returns)
        n_trades = n_trades_per_sim or len(trade_returns)

        equity_curves = []
        final_values = []
        max_drawdowns = []

        for _ in range(self.n_simulations):
            # Resample trades with replacement
            sampled_returns = np.random.choice(trade_returns, size=n_trades, replace=True)

            # Calculate equity curve
            equity = [self.initial_capital]
            for ret in sampled_returns:
                new_value = equity[-1] * (1 + ret / 100)
                equity.append(new_value)

            equity = np.array(equity)
            equity_curves.append(equity)
            final_values.append(equity[-1])

            # Calculate max drawdown
            running_max = np.maximum.accumulate(equity)
            drawdowns = (equity - running_max) / running_max * 100
            max_drawdowns.append(drawdowns.min())

        return self._compile_results(equity_curves, np.array(final_values), np.array(max_drawdowns))

    def run_returns_shuffle(
        self,
        daily_returns: List[float]
    ) -> MonteCarloResult:
        """
        Monte Carlo simulation by shuffling daily returns.

        This preserves the distribution of returns but randomizes the sequence.

        Args:
            daily_returns: List of daily returns (as percentages)

        Returns:
            MonteCarloResult with simulation statistics
        """
        if len(daily_returns) == 0:
            return self._empty_result()

        daily_returns = np.array(daily_returns)

        equity_curves = []
        final_values = []
        max_drawdowns = []

        for _ in range(self.n_simulations):
            # Shuffle returns
            shuffled_returns = np.random.permutation(daily_returns)

            # Calculate equity curve
            equity = [self.initial_capital]
            for ret in shuffled_returns:
                new_value = equity[-1] * (1 + ret / 100)
                equity.append(new_value)

            equity = np.array(equity)
            equity_curves.append(equity)
            final_values.append(equity[-1])

            # Calculate max drawdown
            running_max = np.maximum.accumulate(equity)
            drawdowns = (equity - running_max) / running_max * 100
            max_drawdowns.append(drawdowns.min())

        return self._compile_results(equity_curves, np.array(final_values), np.array(max_drawdowns))

    def run_with_noise(
        self,
        equity_curve: List[float],
        noise_std: float = 0.5
    ) -> MonteCarloResult:
        """
        Monte Carlo simulation adding random noise to equity curve.

        This simulates execution uncertainty, slippage variation, etc.

        Args:
            equity_curve: Original equity curve values
            noise_std: Standard deviation of noise to add (as percentage)

        Returns:
            MonteCarloResult with simulation statistics
        """
        if len(equity_curve) == 0:
            return self._empty_result()

        equity_curve = np.array(equity_curve)

        # Calculate returns from equity curve
        returns = np.diff(equity_curve) / equity_curve[:-1] * 100

        equity_curves = []
        final_values = []
        max_drawdowns = []

        for _ in range(self.n_simulations):
            # Add random noise to returns
            noise = np.random.normal(0, noise_std, len(returns))
            noisy_returns = returns + noise

            # Calculate equity curve
            equity = [self.initial_capital]
            for ret in noisy_returns:
                new_value = equity[-1] * (1 + ret / 100)
                equity.append(new_value)

            equity = np.array(equity)
            equity_curves.append(equity)
            final_values.append(equity[-1])

            # Calculate max drawdown
            running_max = np.maximum.accumulate(equity)
            drawdowns = (equity - running_max) / running_max * 100
            max_drawdowns.append(drawdowns.min())

        return self._compile_results(equity_curves, np.array(final_values), np.array(max_drawdowns))

    def _compile_results(
        self,
        equity_curves: List[np.ndarray],
        final_values: np.ndarray,
        max_drawdowns: np.ndarray
    ) -> MonteCarloResult:
        """Compile simulation results into MonteCarloResult"""
        # Calculate returns
        returns = (final_values - self.initial_capital) / self.initial_capital * 100

        # Value at Risk
        var_95 = np.percentile(returns, (1 - self.confidence_level) * 100)

        # Conditional VaR (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95

        return MonteCarloResult(
            mean_return=np.mean(returns),
            median_return=np.median(returns),
            std_return=np.std(returns),
            min_return=np.min(returns),
            max_return=np.max(returns),
            percentile_5=np.percentile(returns, 5),
            percentile_25=np.percentile(returns, 25),
            percentile_75=np.percentile(returns, 75),
            percentile_95=np.percentile(returns, 95),
            probability_profit=np.mean(returns > 0) * 100,
            probability_loss=np.mean(returns < 0) * 100,
            mean_max_drawdown=np.mean(max_drawdowns),
            worst_max_drawdown=np.min(max_drawdowns),
            best_max_drawdown=np.max(max_drawdowns),
            var_95=var_95,
            cvar_95=cvar_95,
            equity_curves=equity_curves,
            final_values=final_values,
            max_drawdowns=max_drawdowns
        )

    def _empty_result(self) -> MonteCarloResult:
        """Return empty result when no data available"""
        return MonteCarloResult(
            mean_return=0,
            median_return=0,
            std_return=0,
            min_return=0,
            max_return=0,
            percentile_5=0,
            percentile_25=0,
            percentile_75=0,
            percentile_95=0,
            probability_profit=0,
            probability_loss=0,
            mean_max_drawdown=0,
            worst_max_drawdown=0,
            best_max_drawdown=0,
            var_95=0,
            cvar_95=0,
            equity_curves=[],
            final_values=np.array([]),
            max_drawdowns=np.array([])
        )

    def print_summary(self, result: MonteCarloResult):
        """Print Monte Carlo simulation summary"""
        print(f"\n{'='*60}")
        print("MONTE CARLO SIMULATION RESULTS")
        print(f"Simulations: {self.n_simulations}")
        print(f"Confidence Level: {self.confidence_level*100:.0f}%")
        print(f"{'='*60}")

        print(f"\n--- Return Distribution ---")
        print(f"Mean Return:       {result.mean_return:>10.2f}%")
        print(f"Median Return:     {result.median_return:>10.2f}%")
        print(f"Std Dev:           {result.std_return:>10.2f}%")
        print(f"Min Return:        {result.min_return:>10.2f}%")
        print(f"Max Return:        {result.max_return:>10.2f}%")

        print(f"\n--- Percentiles ---")
        print(f"5th Percentile:    {result.percentile_5:>10.2f}%")
        print(f"25th Percentile:   {result.percentile_25:>10.2f}%")
        print(f"75th Percentile:   {result.percentile_75:>10.2f}%")
        print(f"95th Percentile:   {result.percentile_95:>10.2f}%")

        print(f"\n--- Probabilities ---")
        print(f"Probability Profit:{result.probability_profit:>10.2f}%")
        print(f"Probability Loss:  {result.probability_loss:>10.2f}%")

        print(f"\n--- Risk Metrics ---")
        print(f"VaR (95%):         {result.var_95:>10.2f}%")
        print(f"CVaR (95%):        {result.cvar_95:>10.2f}%")
        print(f"Mean Max Drawdown: {result.mean_max_drawdown:>10.2f}%")
        print(f"Worst Max Drawdown:{result.worst_max_drawdown:>10.2f}%")

        print(f"\n{'='*60}\n")


def main():
    """Test Monte Carlo simulation"""
    # Sample trade returns (percentages)
    np.random.seed(42)
    trade_returns = np.random.normal(0.5, 2, 100).tolist()  # Mean 0.5%, std 2%

    simulator = MonteCarloSimulator(n_simulations=1000)

    print("Testing Trade Resampling Method:")
    result = simulator.run_trade_resampling(trade_returns)
    simulator.print_summary(result)

    print("Testing Returns Shuffle Method:")
    daily_returns = np.random.normal(0.1, 1, 252).tolist()  # One year of daily returns
    result = simulator.run_returns_shuffle(daily_returns)
    simulator.print_summary(result)


if __name__ == "__main__":
    main()
